"""Policy-authoritative human approval management for AegisOps."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import (
    ApprovalDecision,
    ApprovalStatus,
    IncidentState,
    ProposedAction,
)
from aegisops.policy import (
    PolicyDecision,
    PolicyEffect,
)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ApprovalError(RuntimeError):
    """Raised when an approval request cannot be processed safely."""


class ApprovalManager:
    """Manage policy-authorized human approval decisions."""

    def __init__(
        self,
        *,
        audit_ledger: AuditLedger,
    ) -> None:
        self._audit_ledger = audit_ledger

    def approve(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        policy_decision: PolicyDecision,
        approver: str,
        reason: str | None = None,
    ) -> ApprovalDecision:
        """Approve an action that policy explicitly requires review for."""

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        self._validate_policy_decision(
            action=action,
            policy_decision=policy_decision,
        )

        decision = self._get_or_create_pending_decision(
            incident=incident,
            action_id=action_id,
        )

        decision.status = ApprovalStatus.APPROVED
        decision.approver = approver
        decision.reason = reason
        decision.decided_at = utc_now()

        incident.updated_at = utc_now()

        self._record_decision(
            incident=incident,
            action=action,
            decision=decision,
            policy_decision=policy_decision,
        )

        return decision

    def reject(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        policy_decision: PolicyDecision,
        approver: str,
        reason: str,
    ) -> ApprovalDecision:
        """Reject an action that policy explicitly requires review for."""

        if not reason.strip():
            raise ApprovalError(
                "A rejection must include a reason."
            )

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        self._validate_policy_decision(
            action=action,
            policy_decision=policy_decision,
        )

        decision = self._get_or_create_pending_decision(
            incident=incident,
            action_id=action_id,
        )

        decision.status = ApprovalStatus.REJECTED
        decision.approver = approver
        decision.reason = reason
        decision.decided_at = utc_now()

        incident.updated_at = utc_now()

        self._record_decision(
            incident=incident,
            action=action,
            decision=decision,
            policy_decision=policy_decision,
        )

        return decision

    @staticmethod
    def is_approved(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> bool:
        """Return True only after an explicit human approval."""

        return any(
            decision.action_id == action_id
            and decision.status is ApprovalStatus.APPROVED
            for decision in incident.approvals
        )

    @staticmethod
    def is_rejected(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> bool:
        """Return True after an explicit human rejection."""

        return any(
            decision.action_id == action_id
            and decision.status is ApprovalStatus.REJECTED
            for decision in incident.approvals
        )

    @staticmethod
    def _get_action(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> ProposedAction:
        """Resolve an action from authoritative incident state."""

        for action in incident.proposed_actions:
            if action.id == action_id:
                return action

        raise ApprovalError(
            f"Action {action_id} does not exist in "
            f"incident {incident.id}."
        )

    def _get_or_create_pending_decision(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> ApprovalDecision:
        """Get an existing pending decision or create one."""

        existing = self._find_existing_decision(
            incident=incident,
            action_id=action_id,
        )

        if existing is not None:
            self._validate_pending(existing)
            return existing

        decision = ApprovalDecision(
            action_id=action_id,
        )

        incident.approvals.append(decision)

        return decision

    @staticmethod
    def _find_existing_decision(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> ApprovalDecision | None:
        """Return the existing approval decision when present."""

        for decision in incident.approvals:
            if decision.action_id == action_id:
                return decision

        return None

    @staticmethod
    def _validate_pending(
        decision: ApprovalDecision,
    ) -> None:
        """Prevent replay or double-processing of a decision."""

        if decision.status is not ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Action {decision.action_id} already has "
                f"a final decision: {decision.status.value}."
            )

    @staticmethod
    def _validate_policy_decision(
        *,
        action: ProposedAction,
        policy_decision: PolicyDecision,
    ) -> None:
        """Verify that policy actually requires approval for this action."""

        if policy_decision.action_id != str(action.id):
            raise ApprovalError(
                "Policy decision does not belong to the "
                f"requested action {action.id}."
            )

        if (
            policy_decision.effect
            is not PolicyEffect.REQUIRE_APPROVAL
        ):
            raise ApprovalError(
                f"Policy does not require approval for action "
                f"{action.id}; effect is "
                f"{policy_decision.effect.value}."
            )

    def _record_decision(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        decision: ApprovalDecision,
        policy_decision: PolicyDecision,
    ) -> None:
        """Record the decision in the tamper-evident audit ledger."""

        self._audit_ledger.append(
            event_type=AuditEventType.APPROVAL_RECORDED,
            actor=decision.approver or "unknown",
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "action_type": action.action_type.value,
                "tool_name": action.tool_name,
                "decision": decision.status.value,
                "reason": decision.reason,
                "policy_version": (
                    policy_decision.policy_version
                ),
                "policy_reasons": [
                    reason.value
                    for reason
                    in policy_decision.reasons
                ],
                "decided_at": (
                    decision.decided_at.isoformat()
                    if decision.decided_at
                    else None
                ),
            },
        )