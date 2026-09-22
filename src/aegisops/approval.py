"""Human approval management for sensitive AegisOps actions."""

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


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ApprovalError(RuntimeError):
    """Raised when an approval request cannot be processed safely."""


class ApprovalManager:
    """Manage approval and rejection decisions for sensitive actions."""

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
        approver: str,
        reason: str | None = None,
    ) -> ApprovalDecision:
        """Approve a pending action."""

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        self._validate_requires_approval(action)

        decision = self._find_existing_decision(
            incident=incident,
            action_id=action_id,
        )

        if decision is not None:
            self._validate_pending(decision)
        else:
            decision = ApprovalDecision(
                action_id=action_id,
            )
            incident.approvals.append(decision)

        decision.status = ApprovalStatus.APPROVED
        decision.approver = approver
        decision.reason = reason
        decision.decided_at = utc_now()

        incident.updated_at = utc_now()

        self._record_decision(
            incident=incident,
            action=action,
            decision=decision,
        )

        return decision

    def reject(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str,
    ) -> ApprovalDecision:
        """Reject a pending action."""

        if not reason.strip():
            raise ApprovalError(
                "A rejection must include a reason."
            )

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        self._validate_requires_approval(action)

        decision = self._find_existing_decision(
            incident=incident,
            action_id=action_id,
        )

        if decision is not None:
            self._validate_pending(decision)
        else:
            decision = ApprovalDecision(
                action_id=action_id,
            )
            incident.approvals.append(decision)

        decision.status = ApprovalStatus.REJECTED
        decision.approver = approver
        decision.reason = reason
        decision.decided_at = utc_now()

        incident.updated_at = utc_now()

        self._record_decision(
            incident=incident,
            action=action,
            decision=decision,
        )

        return decision

    @staticmethod
    def is_approved(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> bool:
        """Return True only when the action has an explicit approval."""

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
        """Return True when the action has been explicitly rejected."""

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
        """Find an action in the authoritative incident state."""

        for action in incident.proposed_actions:
            if action.id == action_id:
                return action

        raise ApprovalError(
            f"Action {action_id} does not exist in incident {incident.id}."
        )

    @staticmethod
    def _find_existing_decision(
        *,
        incident: IncidentState,
        action_id: UUID,
    ) -> ApprovalDecision | None:
        """Return an existing decision for the action if one exists."""

        for decision in incident.approvals:
            if decision.action_id == action_id:
                return decision

        return None

    @staticmethod
    def _validate_requires_approval(
        action: ProposedAction,
    ) -> None:
        """Prevent approval records for actions that do not require them."""

        if not action.requires_approval:
            raise ApprovalError(
                f"Action {action.id} does not require human approval."
            )

    @staticmethod
    def _validate_pending(
        decision: ApprovalDecision,
    ) -> None:
        """Prevent replay or double-processing of an approval."""

        if decision.status is not ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Action {decision.action_id} already has a final "
                f"decision: {decision.status.value}."
            )

    def _record_decision(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        decision: ApprovalDecision,
    ) -> None:
        """Write the human decision to the tamper-evident ledger."""

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
                "decided_at": (
                    decision.decided_at.isoformat()
                    if decision.decided_at
                    else None
                ),
            },
        )