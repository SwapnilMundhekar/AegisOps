"""Integration tests for the core AegisOps control-plane components."""

from __future__ import annotations

import pytest

from aegisops.action_gate import ActionGate, GateStatus
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import (
    ActionType,
    Evidence,
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    ProposedAction,
    RiskLevel,
)
from aegisops.policy import PolicyEffect, PolicyEngine


def build_incident(
    *,
    environment: str = "staging",
) -> IncidentState:
    """Create a representative incident for control-plane tests."""

    return IncidentState(
        title="Payment API latency spike",
        summary="P95 latency increased after deployment.",
        service="payment-api",
        environment=environment,
        severity=IncidentSeverity.SEV2,
    )


def build_action(
    *,
    action_type: ActionType = ActionType.READ,
    risk: RiskLevel = RiskLevel.LOW,
    requires_approval: bool = False,
    tool_name: str = "prometheus.query",
) -> ProposedAction:
    """Create a representative agent-proposed action."""

    return ProposedAction(
        agent="investigator-agent",
        action_type=action_type,
        description="Inspect service telemetry.",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
        },
        risk=risk,
        requires_approval=requires_approval,
    )


def test_valid_incident_transition() -> None:
    """A valid lifecycle transition should succeed."""

    incident = build_incident()

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    assert incident.status is IncidentStatus.TRIAGING


def test_invalid_incident_transition_is_blocked() -> None:
    """Agents must not bypass the deterministic state machine."""

    incident = build_incident()

    with pytest.raises(
        ValueError,
        match="Invalid incident transition",
    ):
        incident.transition_to(
            IncidentStatus.EXECUTING
        )


def test_evidence_is_attached_to_incident() -> None:
    """Investigation evidence should be persisted in incident state."""

    incident = build_incident()

    evidence = Evidence(
        source="prometheus",
        summary="Database connection saturation detected.",
        confidence=0.94,
        reference="metric://db/connection_pool",
    )

    incident.add_evidence(evidence)

    assert len(incident.evidence) == 1
    assert incident.evidence[0].id == evidence.id


def test_low_risk_read_action_is_allowed() -> None:
    """Low-risk read-only operations should execute automatically."""

    policy = PolicyEngine()
    action = build_action()

    decision = policy.evaluate(
        action,
        environment="staging",
    )

    assert decision.effect is PolicyEffect.ALLOW


def test_production_deployment_requires_approval() -> None:
    """Production deployment must never execute autonomously."""

    policy = PolicyEngine()

    action = build_action(
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
        requires_approval=True,
        tool_name="kubernetes.deploy",
    )

    decision = policy.evaluate(
        action,
        environment="production",
    )

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL


def test_blocked_tool_is_denied() -> None:
    """Explicitly blocked tools must fail closed."""

    policy = PolicyEngine(
        blocked_tools={
            "shell.delete_cluster",
        }
    )

    action = build_action(
        action_type=ActionType.EXECUTE,
        risk=RiskLevel.CRITICAL,
        tool_name="shell.delete_cluster",
    )

    decision = policy.evaluate(
        action,
        environment="staging",
    )

    assert decision.effect is PolicyEffect.DENY


def test_action_gate_allows_safe_action() -> None:
    """Safe actions should pass through the execution gate."""

    ledger = AuditLedger()

    gate = ActionGate(
        policy_engine=PolicyEngine(),
        audit_ledger=ledger,
    )

    action = build_action()

    result = gate.evaluate(
        action,
        environment="staging",
    )

    assert result.status is GateStatus.EXECUTE


def test_action_gate_pauses_sensitive_action() -> None:
    """Sensitive operations should stop at the approval boundary."""

    ledger = AuditLedger()

    gate = ActionGate(
        policy_engine=PolicyEngine(),
        audit_ledger=ledger,
    )

    action = build_action(
        action_type=ActionType.DEPLOY,
        risk=RiskLevel.HIGH,
        requires_approval=True,
        tool_name="kubernetes.deploy",
    )

    result = gate.evaluate(
        action,
        environment="production",
    )

    assert result.status is GateStatus.PAUSE_FOR_APPROVAL


def test_action_gate_blocks_denied_tool() -> None:
    """A denied action must never reach execution."""

    ledger = AuditLedger()

    policy = PolicyEngine(
        blocked_tools={
            "shell.delete_cluster",
        }
    )

    gate = ActionGate(
        policy_engine=policy,
        audit_ledger=ledger,
    )

    action = build_action(
        action_type=ActionType.EXECUTE,
        risk=RiskLevel.CRITICAL,
        tool_name="shell.delete_cluster",
    )

    result = gate.evaluate(
        action,
        environment="staging",
    )

    assert result.status is GateStatus.BLOCK


def test_action_gate_creates_audit_events() -> None:
    """Every gated action should leave an auditable decision trail."""

    ledger = AuditLedger()

    gate = ActionGate(
        policy_engine=PolicyEngine(),
        audit_ledger=ledger,
    )

    action = build_action()

    gate.evaluate(
        action,
        environment="staging",
    )

    assert len(ledger.events) == 2

    assert (
        ledger.events[0].event_type
        is AuditEventType.ACTION_PROPOSED
    )

    assert (
        ledger.events[1].event_type
        is AuditEventType.POLICY_DECISION
    )


def test_audit_chain_integrity() -> None:
    """A valid hash-chained audit ledger should verify successfully."""

    ledger = AuditLedger()

    ledger.append(
        event_type=AuditEventType.INCIDENT_CREATED,
        actor="system",
        subject_id="INC-001",
        payload={
            "service": "payment-api",
        },
    )

    ledger.append(
        event_type=AuditEventType.SECURITY_EVENT,
        actor="security-agent",
        subject_id="INC-001",
        payload={
            "finding": "prompt injection blocked",
        },
    )

    assert ledger.verify_integrity() is True

    assert (
        ledger.events[1].previous_hash
        == ledger.events[0].event_hash
    )


def test_audit_sequence_is_monotonic() -> None:
    """Audit events must receive deterministic sequence numbers."""

    ledger = AuditLedger()

    first = ledger.append(
        event_type=AuditEventType.INCIDENT_CREATED,
        actor="system",
        subject_id="INC-001",
    )

    second = ledger.append(
        event_type=AuditEventType.INCIDENT_STATE_CHANGED,
        actor="triage-agent",
        subject_id="INC-001",
    )

    assert first.sequence == 1
    assert second.sequence == 2