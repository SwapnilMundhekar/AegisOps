"""End-to-end tests for the AegisOps incident orchestrator."""

from __future__ import annotations

import pytest

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import (
    ActionType,
    ApprovalStatus,
    IncidentSeverity,
    IncidentState,
    ProposedAction,
    RiskLevel,
)
from aegisops.orchestrator import (
    IncidentOrchestrator,
    OrchestrationStatus,
)
from aegisops.policy import PolicyEngine
from aegisops.tool_runtime import ToolRuntime


def build_incident(
    *,
    environment: str = "staging",
) -> IncidentState:
    """Create a representative incident."""

    return IncidentState(
        title="Payment API degradation",
        summary="Latency increased after the latest deployment.",
        service="payment-api",
        environment=environment,
        severity=IncidentSeverity.SEV2,
    )


def build_action(
    *,
    tool_name: str = "prometheus.query",
    action_type: ActionType = ActionType.READ,
    risk: RiskLevel = RiskLevel.LOW,
    requires_approval: bool = False,
) -> ProposedAction:
    """Create a representative agent-proposed action."""

    return ProposedAction(
        agent="investigator-agent",
        action_type=action_type,
        description="Perform controlled operational action.",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
        },
        risk=risk,
        requires_approval=requires_approval,
    )


def build_control_plane(
    *,
    blocked_tools: set[str] | None = None,
) -> tuple[
    IncidentOrchestrator,
    ToolRuntime,
    ApprovalManager,
    AuditLedger,
]:
    """Build all control-plane components with shared state."""

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

    runtime = ToolRuntime(
        action_gate=gate,
        approval_manager=approvals,
        audit_ledger=ledger,
    )

    orchestrator = IncidentOrchestrator(
        policy_engine=policy,
        approval_manager=approvals,
        tool_runtime=runtime,
        audit_ledger=ledger,
    )

    return (
        orchestrator,
        runtime,
        approvals,
        ledger,
    )


@pytest.mark.asyncio
async def test_safe_action_executes_end_to_end() -> None:
    """A safe read-only action should execute without human approval."""

    (
        orchestrator,
        runtime,
        _,
        ledger,
    ) = build_control_plane()

    runtime.register(
        name="prometheus.query",
        handler=lambda arguments: {
            "service": arguments["service"],
            "healthy": True,
        },
    )

    incident = build_incident()

    action = build_action()

    result = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert result.status is OrchestrationStatus.COMPLETED

    assert result.result == {
        "service": "payment-api",
        "healthy": True,
    }

    assert len(incident.proposed_actions) == 1

    event_types = [
        event.event_type
        for event in ledger.events
    ]

    assert AuditEventType.ACTION_PROPOSED in event_types
    assert AuditEventType.POLICY_DECISION in event_types
    assert AuditEventType.TOOL_STARTED in event_types
    assert AuditEventType.TOOL_COMPLETED in event_types

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_production_deploy_pauses_for_approval() -> None:
    """Policy must pause a production deploy before tool execution."""

    (
        orchestrator,
        runtime,
        _,
        ledger,
    ) = build_control_plane()

    executions: list[str] = []

    def deploy_tool(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        executions.append("executed")

        return {
            "deployed": True,
            "service": arguments["service"],
        }

    runtime.register(
        name="kubernetes.deploy",
        handler=deploy_tool,
    )

    incident = build_incident(
        environment="production",
    )

    action = build_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
        requires_approval=False,
    )

    result = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        result.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    assert executions == []

    assert len(incident.proposed_actions) == 1

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_approved_action_resumes_and_executes() -> None:
    """A paused action should execute after explicit human approval."""

    (
        orchestrator,
        runtime,
        approvals,
        ledger,
    ) = build_control_plane()

    executions: list[str] = []

    def deploy_tool(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        executions.append(
            str(arguments["service"])
        )

        return {
            "deployment_id": "deploy-001",
            "status": "completed",
        }

    runtime.register(
        name="kubernetes.deploy",
        handler=deploy_tool,
    )

    incident = build_incident(
        environment="production",
    )

    action = build_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
    )

    paused = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    resumed = await orchestrator.approve_and_resume(
        incident=incident,
        action_id=action.id,
        approver="platform-lead",
        reason="Change reviewed and approved.",
    )

    assert resumed.status is OrchestrationStatus.COMPLETED

    assert resumed.result == {
        "deployment_id": "deploy-001",
        "status": "completed",
    }

    assert executions == [
        "payment-api",
    ]

    assert approvals.is_approved(
        incident=incident,
        action_id=action.id,
    )

    assert (
        incident.approvals[0].status
        is ApprovalStatus.APPROVED
    )

    assert len(incident.proposed_actions) == 1

    event_types = [
        event.event_type
        for event in ledger.events
    ]

    assert AuditEventType.APPROVAL_RECORDED in event_types
    assert AuditEventType.TOOL_COMPLETED in event_types

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_rejected_action_never_executes() -> None:
    """A rejected production action must remain non-executable."""

    (
        orchestrator,
        runtime,
        approvals,
        ledger,
    ) = build_control_plane()

    executions: list[str] = []

    def deploy_tool(
        arguments: dict[str, object],
    ) -> str:
        executions.append(
            str(arguments["service"])
        )

        return "deployed"

    runtime.register(
        name="kubernetes.deploy",
        handler=deploy_tool,
    )

    incident = build_incident(
        environment="production",
    )

    action = build_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
    )

    paused = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    rejected = orchestrator.reject_action(
        incident=incident,
        action_id=action.id,
        approver="security-lead",
        reason="Deployment window is closed.",
    )

    assert rejected.status is OrchestrationStatus.REJECTED

    assert executions == []

    assert approvals.is_rejected(
        incident=incident,
        action_id=action.id,
    )

    second_attempt = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        second_attempt.status
        is OrchestrationStatus.REJECTED
    )

    assert executions == []

    assert len(incident.proposed_actions) == 1

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_blocked_tool_never_reaches_handler() -> None:
    """An explicitly denied tool must fail closed."""

    (
        orchestrator,
        runtime,
        _,
        ledger,
    ) = build_control_plane(
        blocked_tools={
            "shell.delete_cluster",
        }
    )

    executions: list[str] = []

    def dangerous_tool(
        arguments: dict[str, object],
    ) -> str:
        executions.append(
            str(arguments["service"])
        )

        return "deleted"

    runtime.register(
        name="shell.delete_cluster",
        handler=dangerous_tool,
    )

    incident = build_incident()

    action = build_action(
        tool_name="shell.delete_cluster",
        action_type=ActionType.EXECUTE,
        risk=RiskLevel.CRITICAL,
    )

    result = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert result.status is OrchestrationStatus.BLOCKED

    assert executions == []

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_tool_failure_is_audited() -> None:
    """Tool failure should return FAILED and preserve forensic evidence."""

    (
        orchestrator,
        runtime,
        _,
        ledger,
    ) = build_control_plane()

    def failing_tool(
        arguments: dict[str, object],
    ) -> None:
        raise RuntimeError(
            f"Telemetry unavailable for "
            f"{arguments['service']}"
        )

    runtime.register(
        name="prometheus.query",
        handler=failing_tool,
    )

    incident = build_incident()

    action = build_action()

    result = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert result.status is OrchestrationStatus.FAILED

    event_types = [
        event.event_type
        for event in ledger.events
    ]

    assert AuditEventType.TOOL_FAILED in event_types
    assert AuditEventType.SECURITY_EVENT in event_types

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_action_registration_is_idempotent() -> None:
    """Submitting the same action twice must not duplicate incident state."""

    (
        orchestrator,
        runtime,
        _,
        _,
    ) = build_control_plane()

    runtime.register(
        name="prometheus.query",
        handler=lambda arguments: {
            "service": arguments["service"],
        },
    )

    incident = build_incident()

    action = build_action()

    first = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    second = await orchestrator.submit_action(
        incident=incident,
        action=action,
    )

    assert first.status is OrchestrationStatus.COMPLETED
    assert second.status is OrchestrationStatus.COMPLETED

    assert len(incident.proposed_actions) == 1

    assert incident.proposed_actions[0].id == action.id