"""Integration tests for observable AegisOps tool execution."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    SpanKind as OTelSpanKind,
    StatusCode,
)

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import (
    AuditEventType,
    AuditLedger,
)
from aegisops.domain import (
    ActionType,
    IncidentSeverity,
    IncidentState,
    ProposedAction,
    RiskLevel,
)
from aegisops.observability import (
    InMemoryTelemetrySink,
    SpanKind,
)
from aegisops.otel import (
    OpenTelemetryObservation,
    OpenTelemetryRuntime,
    create_otel_runtime,
)
from aegisops.policy import PolicyEngine
from aegisops.resilience import CircuitBreaker
from aegisops.tool_runtime import (
    ToolExecutionBlockedError,
    ToolExecutionError,
    ToolRuntime,
)


class RecordingExporter(SpanExporter):
    """Capture exported OpenTelemetry spans deterministically."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(
        self,
        spans: Sequence[ReadableSpan],
    ) -> SpanExportResult:
        """Capture exported spans."""

        self.spans.extend(
            spans
        )

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No external resources require shutdown."""

    def force_flush(
        self,
        timeout_millis: int = 30_000,
    ) -> bool:
        """Exporter does not buffer independently."""

        del timeout_millis

        return True


def build_incident(
    *,
    environment: str = "staging",
) -> IncidentState:
    """Create a representative incident."""

    return IncidentState(
        title="Payment API degradation",
        summary="Operational investigation in progress.",
        service="payment-api",
        environment=environment,
        severity=IncidentSeverity.SEV2,
    )


def build_action(
    *,
    tool_name: str = "prometheus.query",
    action_type: ActionType = ActionType.READ,
    risk: RiskLevel = RiskLevel.LOW,
    agent: str = "investigator-agent",
) -> ProposedAction:
    """Create a representative proposed tool action."""

    return ProposedAction(
        agent=agent,
        action_type=action_type,
        description="Execute controlled operational tool.",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
            "api_token": "super-secret-value",
        },
        risk=risk,
    )


def build_observable_runtime(
    *,
    blocked_tools: set[str] | None = None,
) -> tuple[
    ToolRuntime,
    AuditLedger,
    RecordingExporter,
    OpenTelemetryRuntime,
    InMemoryTelemetrySink,
]:
    """Build a complete observable ToolRuntime."""

    ledger = AuditLedger()

    policy = PolicyEngine(
        blocked_tools=blocked_tools or set(),
    )

    approvals = ApprovalManager(
        audit_ledger=ledger,
    )

    gate = ActionGate(
        policy_engine=policy,
        audit_ledger=ledger,
    )

    exporter = RecordingExporter()

    otel_runtime = create_otel_runtime(
        service_name="aegisops-test",
        service_version="0.1.0-test",
        deployment_environment="test",
        exporter=exporter,
    )

    telemetry_sink = InMemoryTelemetrySink()

    runtime = ToolRuntime(
        action_gate=gate,
        approval_manager=approvals,
        audit_ledger=ledger,
        otel_runtime=otel_runtime,
        telemetry_sink=telemetry_sink,
    )

    return (
        runtime,
        ledger,
        exporter,
        otel_runtime,
        telemetry_sink,
    )


def span_by_name(
    exporter: RecordingExporter,
    name: str,
) -> ReadableSpan:
    """Return exactly one exported span with the requested name."""

    matching = [
        span
        for span in exporter.spans
        if span.name == name
    ]

    assert len(matching) == 1

    return matching[0]


@pytest.mark.asyncio
async def test_successful_tool_emits_genai_execution_span() -> None:
    """Successful tool execution should emit standard GenAI telemetry."""

    (
        runtime,
        ledger,
        exporter,
        otel_runtime,
        telemetry_sink,
    ) = build_observable_runtime()

    async def query(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        return {
            "service": arguments["service"],
            "p95_ms": 220,
        }

    runtime.register(
        name="prometheus.query",
        handler=query,
        idempotent=True,
        timeout_seconds=5.0,
    )

    incident = build_incident()

    action = build_action()

    try:
        result = await runtime.execute(
            incident=incident,
            action=action,
        )

        assert result == {
            "service": "payment-api",
            "p95_ms": 220,
        }

        assert otel_runtime.force_flush()

        span = span_by_name(
            exporter,
            "execute_tool prometheus.query",
        )

        assert span.kind is OTelSpanKind.INTERNAL

        assert span.attributes is not None

        assert (
            span.attributes["gen_ai.operation.name"]
            == "execute_tool"
        )

        assert (
            span.attributes["gen_ai.tool.name"]
            == "prometheus.query"
        )

        assert (
            span.attributes["gen_ai.agent.name"]
            == "investigator-agent"
        )

        assert (
            span.attributes["aegisops.incident.id"]
            == str(incident.id)
        )

        assert (
            span.attributes["aegisops.action.id"]
            == str(action.id)
        )

        assert (
            span.attributes["aegisops.tool.idempotent"]
            is True
        )

        assert (
            span.attributes[
                "aegisops.tool.circuit_state.before"
            ]
            == "closed"
        )

        assert (
            span.attributes[
                "aegisops.tool.circuit_state.after"
            ]
            == "closed"
        )

        assert (
            span.attributes["aegisops.tool.outcome"]
            == "success"
        )

        assert (
            span.status.status_code
            is StatusCode.OK
        )

        assert len(
            telemetry_sink.spans
        ) == 1

        assert ledger.verify_integrity() is True

    finally:
        otel_runtime.shutdown()


@pytest.mark.asyncio
async def test_tool_span_inherits_existing_agent_trace() -> None:
    """Tool execution must become a child of the active agent span."""

    (
        runtime,
        _,
        exporter,
        otel_runtime,
        telemetry_sink,
    ) = build_observable_runtime()

    async def query(
        arguments: dict[str, object],
    ) -> str:
        return str(
            arguments["service"]
        )

    runtime.register(
        name="prometheus.query",
        handler=query,
        idempotent=True,
    )

    incident = build_incident()

    action = build_action()

    try:
        with OpenTelemetryObservation(
            runtime=otel_runtime,
            sink=telemetry_sink,
            name="invoke_agent investigator-agent",
            kind=SpanKind.AGENT,
            incident_id=str(incident.id),
            agent_name="investigator-agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "investigator-agent",
            },
        ):
            result = await runtime.execute(
                incident=incident,
                action=action,
            )

        assert result == "payment-api"

        assert otel_runtime.force_flush()

        agent_span = span_by_name(
            exporter,
            "invoke_agent investigator-agent",
        )

        tool_span = span_by_name(
            exporter,
            "execute_tool prometheus.query",
        )

        assert agent_span.context is not None
        assert tool_span.context is not None
        assert tool_span.parent is not None

        assert (
            agent_span.context.trace_id
            == tool_span.context.trace_id
        )

        assert (
            tool_span.parent.span_id
            == agent_span.context.span_id
        )

    finally:
        otel_runtime.shutdown()


@pytest.mark.asyncio
async def test_tool_timeout_emits_error_telemetry() -> None:
    """Timeouts must produce ERROR spans with bounded classification."""

    (
        runtime,
        ledger,
        exporter,
        otel_runtime,
        _,
    ) = build_observable_runtime()

    async def slow_tool(
        arguments: dict[str, object],
    ) -> str:
        await asyncio.sleep(
            0.05
        )

        return str(
            arguments["service"]
        )

    runtime.register(
        name="telemetry.slow_query",
        handler=slow_tool,
        idempotent=False,
        timeout_seconds=0.001,
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="telemetry.slow_query",
    )

    try:
        with pytest.raises(
            ToolExecutionError,
            match="timed out",
        ):
            await runtime.execute(
                incident=incident,
                action=action,
            )

        assert otel_runtime.force_flush()

        span = span_by_name(
            exporter,
            "execute_tool telemetry.slow_query",
        )

        assert (
            span.status.status_code
            is StatusCode.ERROR
        )

        assert span.attributes is not None

        assert (
            span.attributes["aegisops.tool.outcome"]
            == "error"
        )

        assert (
            span.attributes["error.type"]
            == "timeout"
        )

        exception_events = [
            event
            for event in span.events
            if event.name == "exception"
        ]

        assert len(
            exception_events
        ) == 1

        assert ledger.verify_integrity() is True

    finally:
        otel_runtime.shutdown()


@pytest.mark.asyncio
async def test_blocked_action_never_creates_execution_span() -> None:
    """Policy-denied operations must never appear as executed tools."""

    (
        runtime,
        ledger,
        exporter,
        otel_runtime,
        telemetry_sink,
    ) = build_observable_runtime(
        blocked_tools={
            "shell.delete_cluster",
        }
    )

    executions = 0

    async def dangerous_tool(
        arguments: dict[str, object],
    ) -> str:
        nonlocal executions

        executions += 1

        return str(
            arguments["service"]
        )

    runtime.register(
        name="shell.delete_cluster",
        handler=dangerous_tool,
        idempotent=False,
    )

    incident = build_incident()

    action = build_action(
        tool_name="shell.delete_cluster",
        action_type=ActionType.EXECUTE,
        risk=RiskLevel.CRITICAL,
    )

    try:
        with pytest.raises(
            ToolExecutionBlockedError,
            match="denied by policy",
        ):
            await runtime.execute(
                incident=incident,
                action=action,
            )

        assert executions == 0

        assert otel_runtime.force_flush()

        assert exporter.spans == []

        assert telemetry_sink.spans == ()

        event_types = [
            event.event_type
            for event in ledger.events
        ]

        assert AuditEventType.ACTION_PROPOSED in event_types
        assert AuditEventType.POLICY_DECISION in event_types

        assert AuditEventType.TOOL_STARTED not in event_types

        assert ledger.verify_integrity() is True

    finally:
        otel_runtime.shutdown()


@pytest.mark.asyncio
async def test_sensitive_arguments_are_not_exported() -> None:
    """Raw tool arguments must not leak into telemetry or audit records."""

    (
        runtime,
        ledger,
        exporter,
        otel_runtime,
        _,
    ) = build_observable_runtime()

    async def query(
        arguments: dict[str, object],
    ) -> str:
        return str(
            arguments["service"]
        )

    runtime.register(
        name="prometheus.query",
        handler=query,
        idempotent=True,
    )

    incident = build_incident()

    action = build_action()

    try:
        await runtime.execute(
            incident=incident,
            action=action,
        )

        assert otel_runtime.force_flush()

        span = span_by_name(
            exporter,
            "execute_tool prometheus.query",
        )

        assert span.attributes is not None

        serialized_attributes = str(
            dict(
                span.attributes
            )
        )

        assert (
            "super-secret-value"
            not in serialized_attributes
        )

        assert (
            "gen_ai.tool.call.arguments"
            not in span.attributes
        )

        started_events = [
            event
            for event in ledger.events
            if event.event_type
            is AuditEventType.TOOL_STARTED
        ]

        assert len(
            started_events
        ) == 1

        payload = started_events[
            0
        ].payload

        assert "arguments" not in payload

        assert payload[
            "argument_keys"
        ] == [
            "api_token",
            "service",
        ]

        assert (
            "super-secret-value"
            not in str(
                payload
            )
        )

    finally:
        otel_runtime.shutdown()


@pytest.mark.asyncio
async def test_failure_records_post_failure_circuit_state() -> None:
    """Telemetry should expose circuit transition after dependency failure."""

    (
        runtime,
        _,
        exporter,
        otel_runtime,
        _,
    ) = build_observable_runtime()

    async def unavailable_tool(
        arguments: dict[str, object],
    ) -> None:
        raise ConnectionError(
            f"{arguments['service']} unavailable"
        )

    runtime.register(
        name="prometheus.query",
        handler=unavailable_tool,
        idempotent=False,
        circuit_breaker=CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=60.0,
        ),
    )

    incident = build_incident()

    action = build_action()

    try:
        with pytest.raises(
            ToolExecutionError,
        ):
            await runtime.execute(
                incident=incident,
                action=action,
            )

        assert otel_runtime.force_flush()

        span = span_by_name(
            exporter,
            "execute_tool prometheus.query",
        )

        assert span.attributes is not None

        assert (
            span.attributes[
                "aegisops.tool.circuit_state.before"
            ]
            == "closed"
        )

        assert (
            span.attributes[
                "aegisops.tool.circuit_state.after"
            ]
            == "open"
        )

        assert (
            span.attributes["aegisops.tool.outcome"]
            == "error"
        )

        assert (
            span.attributes["error.type"]
            == "ConnectionError"
        )

    finally:
        otel_runtime.shutdown()