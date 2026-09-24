"""Incident-level orchestration for the AegisOps control plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from aegisops.approval import ApprovalManager
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import IncidentState, ProposedAction
from aegisops.policy import (
    PolicyDecision,
    PolicyEffect,
    PolicyEngine,
)
from aegisops.tool_runtime import (
    ToolApprovalRequiredError,
    ToolExecutionBlockedError,
    ToolExecutionError,
    ToolRuntime,
)


class OrchestrationStatus(StrEnum):
    """High-level result of an orchestrated agent action."""

    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Outcome returned by the incident orchestrator."""

    action_id: UUID
    status: OrchestrationStatus
    policy_decision: PolicyDecision
    result: object | None = None
    error: str | None = None


class IncidentOrchestrator:
    """Coordinate policy, approvals, execution, and audit state."""

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        approval_manager: ApprovalManager,
        tool_runtime: ToolRuntime,
        audit_ledger: AuditLedger,
    ) -> None:
        self._policy_engine = policy_engine
        self._approval_manager = approval_manager
        self._tool_runtime = tool_runtime
        self._audit_ledger = audit_ledger

    async def submit_action(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> OrchestrationResult:
        """Submit an agent-proposed action to the control plane."""

        self._register_action(
            incident=incident,
            action=action,
        )

        policy_decision = self._policy_engine.evaluate(
            action,
            environment=incident.environment,
        )

        if policy_decision.effect is PolicyEffect.DENY:
            return await self._attempt_execution(
                incident=incident,
                action=action,
                policy_decision=policy_decision,
            )

        if (
            policy_decision.effect
            is PolicyEffect.REQUIRE_APPROVAL
        ):
            if self._approval_manager.is_rejected(
                incident=incident,
                action_id=action.id,
            ):
                return OrchestrationResult(
                    action_id=action.id,
                    status=OrchestrationStatus.REJECTED,
                    policy_decision=policy_decision,
                    error="Action was rejected by a human reviewer.",
                )

            if not self._approval_manager.is_approved(
                incident=incident,
                action_id=action.id,
            ):
                return await self._pause_for_approval(
                    incident=incident,
                    action=action,
                    policy_decision=policy_decision,
                )

        return await self._attempt_execution(
            incident=incident,
            action=action,
            policy_decision=policy_decision,
        )

    async def approve_and_resume(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str | None = None,
    ) -> OrchestrationResult:
        """Approve a pending action and resume execution."""

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        policy_decision = self._policy_engine.evaluate(
            action,
            environment=incident.environment,
        )

        self._approval_manager.approve(
            incident=incident,
            action_id=action.id,
            policy_decision=policy_decision,
            approver=approver,
            reason=reason,
        )

        return await self._attempt_execution(
            incident=incident,
            action=action,
            policy_decision=policy_decision,
        )

    def reject_action(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str,
    ) -> OrchestrationResult:
        """Reject a pending action without executing its tool."""

        action = self._get_action(
            incident=incident,
            action_id=action_id,
        )

        policy_decision = self._policy_engine.evaluate(
            action,
            environment=incident.environment,
        )

        self._approval_manager.reject(
            incident=incident,
            action_id=action.id,
            policy_decision=policy_decision,
            approver=approver,
            reason=reason,
        )

        return OrchestrationResult(
            action_id=action.id,
            status=OrchestrationStatus.REJECTED,
            policy_decision=policy_decision,
            error="Action rejected by human reviewer.",
        )

    async def _pause_for_approval(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        policy_decision: PolicyDecision,
    ) -> OrchestrationResult:
        """Pass through the runtime so the pause is audited."""

        try:
            await self._tool_runtime.execute(
                incident=incident,
                action=action,
            )

        except ToolApprovalRequiredError:
            return OrchestrationResult(
                action_id=action.id,
                status=OrchestrationStatus.APPROVAL_REQUIRED,
                policy_decision=policy_decision,
            )

        except ToolExecutionBlockedError as exc:
            return OrchestrationResult(
                action_id=action.id,
                status=OrchestrationStatus.BLOCKED,
                policy_decision=policy_decision,
                error=str(exc),
            )

        raise RuntimeError(
            "Approval-required action unexpectedly executed."
        )

    async def _attempt_execution(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        policy_decision: PolicyDecision,
    ) -> OrchestrationResult:
        """Execute an authorized action through the controlled runtime."""

        try:
            result = await self._tool_runtime.execute(
                incident=incident,
                action=action,
            )

        except ToolExecutionBlockedError as exc:
            return OrchestrationResult(
                action_id=action.id,
                status=OrchestrationStatus.BLOCKED,
                policy_decision=policy_decision,
                error=str(exc),
            )

        except ToolApprovalRequiredError as exc:
            return OrchestrationResult(
                action_id=action.id,
                status=OrchestrationStatus.APPROVAL_REQUIRED,
                policy_decision=policy_decision,
                error=str(exc),
            )

        except ToolExecutionError as exc:
            self._record_orchestration_failure(
                incident=incident,
                action=action,
                error=exc,
            )

            return OrchestrationResult(
                action_id=action.id,
                status=OrchestrationStatus.FAILED,
                policy_decision=policy_decision,
                error=str(exc),
            )

        return OrchestrationResult(
            action_id=action.id,
            status=OrchestrationStatus.COMPLETED,
            policy_decision=policy_decision,
            result=result,
        )

    @staticmethod
    def _register_action(
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> None:
        """Register an action exactly once in authoritative state."""

        if any(
            existing.id == action.id
            for existing in incident.proposed_actions
        ):
            return

        incident.add_action(action)

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

        raise ValueError(
            f"Action {action_id} does not exist in "
            f"incident {incident.id}."
        )

    def _record_orchestration_failure(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        error: Exception,
    ) -> None:
        """Record orchestration-level execution failure."""

        self._audit_ledger.append(
            event_type=AuditEventType.SECURITY_EVENT,
            actor="aegisops-orchestrator",
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "event": "orchestration_failure",
                "tool_name": action.tool_name,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )