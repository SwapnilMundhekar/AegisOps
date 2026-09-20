"""Central execution gate for agent-proposed actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import ProposedAction
from aegisops.policy import (
    PolicyDecision,
    PolicyEffect,
    PolicyEngine,
)


class GateStatus(StrEnum):
    """Execution outcome assigned to a proposed agent action."""

    EXECUTE = "execute"
    PAUSE_FOR_APPROVAL = "pause_for_approval"
    BLOCK = "block"


class GateResult(BaseModel):
    """Immutable decision returned by the AegisOps action gate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: GateStatus
    action_id: str
    policy_decision: PolicyDecision


class ActionGate:
    """Enforce policy before an agent action reaches a tool."""

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        audit_ledger: AuditLedger,
    ) -> None:
        self._policy_engine = policy_engine
        self._audit_ledger = audit_ledger

    def evaluate(
        self,
        action: ProposedAction,
        *,
        environment: str,
    ) -> GateResult:
        """Evaluate, classify, and audit a proposed agent action."""

        self._record_proposal(
            action=action,
            environment=environment,
        )

        policy_decision = self._policy_engine.evaluate(
            action,
            environment=environment,
        )

        status = self._map_policy_effect(
            policy_decision.effect
        )

        self._record_policy_decision(
            action=action,
            environment=environment,
            decision=policy_decision,
            status=status,
        )

        return GateResult(
            status=status,
            action_id=str(action.id),
            policy_decision=policy_decision,
        )

    def _record_proposal(
        self,
        *,
        action: ProposedAction,
        environment: str,
    ) -> None:
        """Record an agent action before policy evaluation."""

        self._audit_ledger.append(
            event_type=AuditEventType.ACTION_PROPOSED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "environment": environment,
                "action_type": action.action_type.value,
                "description": action.description,
                "tool_name": action.tool_name,
                "risk": action.risk.value,
                "requires_approval": action.requires_approval,
            },
        )

    def _record_policy_decision(
        self,
        *,
        action: ProposedAction,
        environment: str,
        decision: PolicyDecision,
        status: GateStatus,
    ) -> None:
        """Record the deterministic authorization result."""

        self._audit_ledger.append(
            event_type=AuditEventType.POLICY_DECISION,
            actor="aegisops-policy-engine",
            subject_id=str(action.id),
            payload={
                "environment": environment,
                "effect": decision.effect.value,
                "gate_status": status.value,
                "reasons": [
                    reason.value
                    for reason in decision.reasons
                ],
                "policy_version": decision.policy_version,
            },
        )

    @staticmethod
    def _map_policy_effect(
        effect: PolicyEffect,
    ) -> GateStatus:
        """Translate a policy outcome into execution behaviour."""

        mapping = {
            PolicyEffect.ALLOW: GateStatus.EXECUTE,
            PolicyEffect.REQUIRE_APPROVAL: (
                GateStatus.PAUSE_FOR_APPROVAL
            ),
            PolicyEffect.DENY: GateStatus.BLOCK,
        }

        return mapping[effect]