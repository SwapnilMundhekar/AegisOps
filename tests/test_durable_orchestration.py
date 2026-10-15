"""Integration tests for crash-resumable AegisOps orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import (
    AuditEventType,
    AuditLedger,
)
from aegisops.checkpoint import (
    CheckpointConflictError,
    SQLiteCheckpointStore,
)
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


def build_incident() -> IncidentState:
    """Create a production incident requiring controlled remediation."""

    return IncidentState(
        title="Payment API degradation",
        summary="Latency increased after deployment.",
        service="payment-api",
        environment="production",
        severity=IncidentSeverity.SEV2,
    )


def build_deploy_action() -> ProposedAction:
    """Create a production deployment action."""

    return ProposedAction(
        agent="deployment-agent",
        action_type=ActionType.DEPLOY,
        description="Deploy payment API remediation.",
        tool_name="kubernetes.deploy",
        tool_arguments={
            "service": "payment-api",
            "version": "1.8.3",
        },
        risk=RiskLevel.HIGH,
    )


def build_process(
    *,
    database: Path,
    executions: list[str],
) -> tuple[
    IncidentOrchestrator,
    ToolRuntime,
    AuditLedger,
]:
    """Build one independent AegisOps process sharing durable storage."""

    ledger = AuditLedger()

    policy = PolicyEngine()

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

    async def deploy_tool(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        service = str(
            arguments["service"]
        )

        executions.append(
            service
        )

        return {
            "service": service,
            "deployment_id": "deploy-001",
            "status": "completed",
        }

    runtime.register(
        name="kubernetes.deploy",
        handler=deploy_tool,
        idempotent=False,
    )

    checkpoint_store = SQLiteCheckpointStore(
        database
    )

    orchestrator = IncidentOrchestrator(
        policy_engine=policy,
        approval_manager=approvals,
        tool_runtime=runtime,
        audit_ledger=ledger,
        checkpoint_store=checkpoint_store,
    )

    return (
        orchestrator,
        runtime,
        ledger,
    )


@pytest.mark.asyncio
async def test_restart_restores_pending_action_and_resumes_after_approval(
    tmp_path: Path,
) -> None:
    """A process restart must preserve the pending approval boundary."""

    database = tmp_path / "aegisops.db"

    executions: list[str] = []

    (
        first_process,
        _,
        first_ledger,
    ) = build_process(
        database=database,
        executions=executions,
    )

    incident = build_incident()

    action = build_deploy_action()

    paused = await first_process.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    assert executions == []

    assert (
        first_process.checkpoint_version(
            incident.id
        )
        == 1
    )

    assert first_ledger.verify_integrity() is True

    del first_process

    (
        restarted_process,
        _,
        restarted_ledger,
    ) = build_process(
        database=database,
        executions=executions,
    )

    restored = restarted_process.restore_incident(
        incident.id
    )

    assert restored.id == incident.id

    assert len(
        restored.proposed_actions
    ) == 1

    assert (
        restored.proposed_actions[0].id
        == action.id
    )

    assert restored.approvals == []

    assert (
        restarted_process.checkpoint_version(
            restored.id
        )
        == 1
    )

    resumed = await restarted_process.approve_and_resume(
        incident=restored,
        action_id=action.id,
        approver="platform-lead",
        reason="Change reviewed after process restart.",
    )

    assert (
        resumed.status
        is OrchestrationStatus.COMPLETED
    )

    assert executions == [
        "payment-api",
    ]

    assert (
        restarted_process.checkpoint_version(
            restored.id
        )
        == 2
    )

    assert len(
        restored.approvals
    ) == 1

    assert (
        restored.approvals[0].status
        is ApprovalStatus.APPROVED
    )

    assert restarted_ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_two_restored_workers_cannot_resume_same_checkpoint(
    tmp_path: Path,
) -> None:
    """Optimistic concurrency must prevent double resume."""

    database = tmp_path / "aegisops.db"

    executions: list[str] = []

    (
        creator,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    incident = build_incident()

    action = build_deploy_action()

    paused = await creator.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    (
        worker_a,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    (
        worker_b,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    state_a = worker_a.restore_incident(
        incident.id
    )

    state_b = worker_b.restore_incident(
        incident.id
    )

    assert (
        worker_a.checkpoint_version(
            incident.id
        )
        == 1
    )

    assert (
        worker_b.checkpoint_version(
            incident.id
        )
        == 1
    )

    completed = await worker_a.approve_and_resume(
        incident=state_a,
        action_id=action.id,
        approver="platform-lead-a",
        reason="Approved by worker A.",
    )

    assert (
        completed.status
        is OrchestrationStatus.COMPLETED
    )

    assert executions == [
        "payment-api",
    ]

    with pytest.raises(
        CheckpointConflictError,
        match="Checkpoint conflict",
    ):
        await worker_b.approve_and_resume(
            incident=state_b,
            action_id=action.id,
            approver="platform-lead-b",
            reason="Concurrent stale approval.",
        )

    assert executions == [
        "payment-api",
    ]

    durable_store = SQLiteCheckpointStore(
        database
    )

    latest = durable_store.load_latest(
        incident.id
    )

    assert latest.version == 2

    assert len(
        latest.state.approvals
    ) == 1

    assert (
        latest.state.approvals[0].status
        is ApprovalStatus.APPROVED
    )


@pytest.mark.asyncio
async def test_unrestored_process_cannot_overwrite_existing_state(
    tmp_path: Path,
) -> None:
    """A fresh process must restore durable state before modifying it."""

    database = tmp_path / "aegisops.db"

    executions: list[str] = []

    (
        first_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    incident = build_incident()

    stale_copy = incident.model_copy(
        deep=True
    )

    first_action = build_deploy_action()

    paused = await first_process.submit_action(
        incident=incident,
        action=first_action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    (
        fresh_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    stale_action = ProposedAction(
        agent="deployment-agent",
        action_type=ActionType.DEPLOY,
        description="Stale process deployment attempt.",
        tool_name="kubernetes.deploy",
        tool_arguments={
            "service": "payment-api",
            "version": "1.8.4",
        },
        risk=RiskLevel.HIGH,
    )

    with pytest.raises(
        CheckpointConflictError,
        match="Restore it with restore_incident",
    ):
        await fresh_process.submit_action(
            incident=stale_copy,
            action=stale_action,
        )

    assert executions == []

    durable_store = SQLiteCheckpointStore(
        database
    )

    latest = durable_store.load_latest(
        incident.id
    )

    assert latest.version == 1

    assert len(
        latest.state.proposed_actions
    ) == 1

    assert (
        latest.state.proposed_actions[0].id
        == first_action.id
    )


@pytest.mark.asyncio
async def test_rejection_survives_restart_and_prevents_execution(
    tmp_path: Path,
) -> None:
    """A durable rejection must remain authoritative after restart."""

    database = tmp_path / "aegisops.db"

    executions: list[str] = []

    (
        first_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    incident = build_incident()

    action = build_deploy_action()

    paused = await first_process.submit_action(
        incident=incident,
        action=action,
    )

    assert (
        paused.status
        is OrchestrationStatus.APPROVAL_REQUIRED
    )

    (
        reviewer_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    restored = reviewer_process.restore_incident(
        incident.id
    )

    rejected = reviewer_process.reject_action(
        incident=restored,
        action_id=action.id,
        approver="security-lead",
        reason="Change window is closed.",
    )

    assert (
        rejected.status
        is OrchestrationStatus.REJECTED
    )

    assert executions == []

    assert (
        reviewer_process.checkpoint_version(
            incident.id
        )
        == 2
    )

    (
        restarted_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    restored_again = restarted_process.restore_incident(
        incident.id
    )

    assert len(
        restored_again.approvals
    ) == 1

    assert (
        restored_again.approvals[0].status
        is ApprovalStatus.REJECTED
    )

    result = await restarted_process.submit_action(
        incident=restored_again,
        action=restored_again.proposed_actions[0],
    )

    assert (
        result.status
        is OrchestrationStatus.REJECTED
    )

    assert executions == []


@pytest.mark.asyncio
async def test_restore_is_audited_and_tracks_exact_version(
    tmp_path: Path,
) -> None:
    """Restoration should expose the exact durable checkpoint version."""

    database = tmp_path / "aegisops.db"

    executions: list[str] = []

    (
        first_process,
        _,
        _,
    ) = build_process(
        database=database,
        executions=executions,
    )

    incident = build_incident()

    action = build_deploy_action()

    await first_process.submit_action(
        incident=incident,
        action=action,
    )

    (
        restarted_process,
        _,
        restarted_ledger,
    ) = build_process(
        database=database,
        executions=executions,
    )

    restored = restarted_process.restore_incident(
        incident.id
    )

    assert (
        restarted_process.checkpoint_version(
            restored.id
        )
        == 1
    )

    restore_events = [
        event
        for event in restarted_ledger.events
        if (
            event.event_type
            is AuditEventType.INCIDENT_STATE_CHANGED
            and event.payload.get("event")
            == "checkpoint_restored"
        )
    ]

    assert len(
        restore_events
    ) == 1

    assert (
        restore_events[0].payload[
            "checkpoint_version"
        ]
        == 1
    )

    assert restarted_ledger.verify_integrity() is True