
"""End-to-end telemetry tests for AegisOps incident orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import AuditLedger
from aegisops.domain import (
    ActionType,
    IncidentSeverity,
    IncidentState,
    ProposedAction,
    RiskLevel,
)
from aegisops.observability import InMemoryTelemetrySink
from aegisops.orchestrator import (
    IncidentOrchestrator,
    OrchestrationStatus,
)
from aegisops.otel import create_otel_runtime
from aegisops.policy import PolicyEngine
from aegisops.tool_runtime import ToolRuntime


class RecordingExporter(SpanExporter):
    """Capture exported spans without external services."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(
        self,
        spans: Sequence[ReadableSpan],
    ) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def build_system(
    *,
    blocked_tools: set[str] | None = None,
):
    """Build a complete control plane sharing one tracer."""

    ledger = AuditLedger()
    policy = PolicyEngine(blocked_tools=blocked_tools or set())
    approvals = ApprovalManager(audit_ledger=ledger)
    gate = ActionGate(
        policy_engine=policy,
        audit_ledger=ledger,
    )

    exporter = RecordingExporter()
    otel = create_otel_runtime(
        service_name="aegisops-test",
        deployment_environment="test",
        exporter=exporter,
    )
    sink = InMemoryTelemetrySink()

    runtime = ToolRuntime(
        action_gate=gate,
        approval_manager=approvals,
        audit_ledger=ledger,
        otel_runtime=otel,
        telemetry_sink=sink,
    )

    orchestrator = IncidentOrchestrator(
        policy_engine=policy,
        approval_manager=approvals,
        tool_runtime=runtime,
        audit_ledger=ledger,
        otel_runtime=otel,
        telemetry_sink=sink,
    )

    return orchestrator, runtime, ledger, otel, exporter, sink


def make_incident(environment: str = "staging") -> IncidentState:
    return IncidentState(
        title="Payment API degradation",
        summary="Latency increased following deployment.",
        service="payment-api",
        environment=environment,
        severity=IncidentSeverity.SEV2,
    )


def make_action(
    *,
    tool_name: str = "prometheus.query",
    action_type: ActionType = ActionType.READ,
    risk: RiskLevel = RiskLevel.LOW,
) -> ProposedAction:
    return ProposedAction(
        agent="investigator-agent",
        action_type=action_type,
        description="Execute controlled action",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
            "api_token": "secret-do-not-export",
        },
        risk=risk,
    )


def get_spans(exporter: RecordingExporter, name: str):
    return [span for span in exporter.spans if span.name == name]


@pytest.mark.asyncio
async def test_successful_workflow_has_correlated_tool_child() -> None:
    orchestrator, runtime, ledger, otel, exporter, sink = build_system()

    async def query(arguments):
        return {"service": arguments["service"], "healthy": True}

    runtime.register(name="prometheus.query", handler=query)
    incident = make_incident()
    action = make_action()

    try:
        result = await orchestrator.submit_action(
            incident=incident,
            action=action,
        )

        assert result.status is OrchestrationStatus.COMPLETED
        assert otel.force_flush()

        workflows = get_spans(
            exporter, "invoke_workflow incident-response"
        )
        tools = get_spans(
            exporter, "execute_tool prometheus.query"
        )

        assert len(workflows) == 1
        assert len(tools) == 1

        parent, child = workflows[0], tools[0]

        assert parent.context is not None
        assert child.context is not None
        assert child.parent is not None

        assert parent.context.trace_id == child.context.trace_id
        assert child.parent.span_id == parent.context.span_id

        assert parent.attributes["gen_ai.operation.name"] == "invoke_workflow"
        assert parent.attributes["gen_ai.workflow.name"] == "incident-response"
        assert parent.attributes["aegisops.orchestration.status"] == "completed"

        assert child.attributes["gen_ai.operation.name"] == "execute_tool"
        assert child.attributes["gen_ai.tool.name"] == "prometheus.query"
        assert child.attributes["aegisops.incident.id"] == str(incident.id)
        assert child.attributes["aegisops.action.id"] == str(action.id)

        assert ledger.verify_integrity()
        assert len(sink.spans) == 2
    finally:
        otel.shutdown()


@pytest.mark.asyncio
async def test_production_deployment_pauses_without_tool_span() -> None:
    orchestrator, runtime, ledger, otel, exporter, _ = build_system()
    executions = []

    async def deploy(arguments):
        executions.append(arguments["service"])
        return "deployed"

    runtime.register(name="kubernetes.deploy", handler=deploy)

    incident = make_incident("production")
    action = make_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
    )

    try:
        result = await orchestrator.submit_action(
            incident=incident, action=action
        )

        assert result.status is OrchestrationStatus.APPROVAL_REQUIRED
        assert executions == []
        assert otel.force_flush()

        workflows = get_spans(
            exporter, "invoke_workflow incident-response"
        )
        assert len(workflows) == 1
        assert workflows[0].attributes[
            "aegisops.orchestration.status"
        ] == "approval_required"

        assert get_spans(
            exporter, "execute_tool kubernetes.deploy"
        ) == []
        assert ledger.verify_integrity()
    finally:
        otel.shutdown()


@pytest.mark.asyncio
async def test_approval_resumes_under_new_workflow_span() -> None:
    orchestrator, runtime, ledger, otel, exporter, _ = build_system()
    executions = []

    async def deploy(arguments):
        executions.append(arguments["service"])
        return {"deployment_id": "deploy-001"}

    runtime.register(name="kubernetes.deploy", handler=deploy)
    incident = make_incident("production")
    action = make_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
    )

    try:
        paused = await orchestrator.submit_action(
            incident=incident, action=action
        )
        assert paused.status is OrchestrationStatus.APPROVAL_REQUIRED

        resumed = await orchestrator.approve_and_resume(
            incident=incident,
            action_id=action.id,
            approver="platform-lead",
            reason="Reviewed",
        )

        assert resumed.status is OrchestrationStatus.COMPLETED
        assert executions == ["payment-api"]
        assert otel.force_flush()

        workflows = get_spans(
            exporter, "invoke_workflow incident-response"
        )
        tools = get_spans(
            exporter, "execute_tool kubernetes.deploy"
        )

        assert len(workflows) == 2
        assert len(tools) == 1

        resume_span = next(
            span for span in workflows
            if span.attributes["aegisops.workflow.phase"]
            == "approve_and_resume"
        )

        assert tools[0].parent is not None
        assert resume_span.context is not None
        assert tools[0].parent.span_id == resume_span.context.span_id

        assert ledger.verify_integrity()
    finally:
        otel.shutdown()


@pytest.mark.asyncio
async def test_rejection_does_not_execute_tool() -> None:
    orchestrator, runtime, ledger, otel, exporter, _ = build_system()
    executions = []

    async def deploy(arguments):
        executions.append(arguments["service"])
        return "deployed"

    runtime.register(name="kubernetes.deploy", handler=deploy)
    incident = make_incident("production")
    action = make_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
    )

    try:
        await orchestrator.submit_action(
            incident=incident, action=action
        )

        result = orchestrator.reject_action(
            incident=incident,
            action_id=action.id,
            approver="security-lead",
            reason="Unsafe change",
        )

        assert result.status is OrchestrationStatus.REJECTED
        assert executions == []
        assert otel.force_flush()

        assert get_spans(
            exporter, "execute_tool kubernetes.deploy"
        ) == []

        workflows = get_spans(
            exporter, "invoke_workflow incident-response"
        )
        assert len(workflows) == 2
        assert any(
            span.attributes["aegisops.orchestration.status"]
            == "rejected"
            for span in workflows
        )
        assert ledger.verify_integrity()
    finally:
        otel.shutdown()


@pytest.mark.asyncio
async def test_tool_failure_marks_workflow_and_tool_error() -> None:
    orchestrator, runtime, ledger, otel, exporter, _ = build_system()

    async def failing_query(arguments):
        raise ConnectionError("dependency unavailable")

    runtime.register(name="prometheus.query", handler=failing_query)
    incident = make_incident()
    action = make_action()

    try:
        result = await orchestrator.submit_action(
            incident=incident, action=action
        )

        assert result.status is OrchestrationStatus.FAILED
        assert otel.force_flush()

        workflow = get_spans(
            exporter, "invoke_workflow incident-response"
        )[0]
        tool = get_spans(
            exporter, "execute_tool prometheus.query"
        )[0]

        assert workflow.status.status_code is StatusCode.ERROR
        assert tool.status.status_code is StatusCode.ERROR
        assert workflow.attributes["error.type"] == "tool_execution_error"
        assert tool.attributes["aegisops.tool.outcome"] == "error"
        assert ledger.verify_integrity()
    finally:
        otel.shutdown()


@pytest.mark.asyncio
async def test_telemetry_does_not_expose_secret_arguments() -> None:
    orchestrator, runtime, ledger, otel, exporter, _ = build_system()

    async def query(arguments):
        return {"healthy": True}

    runtime.register(name="prometheus.query", handler=query)
    incident = make_incident()
    action = make_action()

    try:
        result = await orchestrator.submit_action(
            incident=incident, action=action
        )
        assert result.status is OrchestrationStatus.COMPLETED
        assert otel.force_flush()

        for span in exporter.spans:
            assert "secret-do-not-export" not in str(span.attributes)
            assert "gen_ai.tool.call.arguments" not in (
                span.attributes or {}
            )

        for event in ledger.events:
            assert "secret-do-not-export" not in str(event.payload)

        assert ledger.verify_integrity()
    finally:
        otel.shutdown()
