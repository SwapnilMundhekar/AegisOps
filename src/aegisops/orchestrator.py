"""Incident-level workflow orchestration for the AegisOps control plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from aegisops.approval import ApprovalManager
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import (
    IncidentState,
    ProposedAction,
)
from aegisops.observability import (
    SpanKind,
    SpanRecord,
    SpanStatus,
    TelemetrySink,
)
from aegisops.otel import (
    OpenTelemetryObservation,
    OpenTelemetryRuntime,
)
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


class _NoOpTelemetrySink:
    """Discard local spans when only OpenTelemetry export is required."""

    def emit(
        self,
        span: SpanRecord,
    ) -> None:
        """Discard a completed local span."""

        del span


class IncidentOrchestrator:
    """Coordinate policy, approval, execution, audit, and tracing."""

    WORKFLOW_NAME = "incident-response"

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        approval_manager: ApprovalManager,
        tool_runtime: ToolRuntime,
        audit_ledger: AuditLedger,
        otel_runtime: OpenTelemetryRuntime | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._approval_manager = approval_manager
        self._tool_runtime = tool_runtime
        self._audit_ledger = audit_ledger

        self._otel_runtime = otel_runtime

        if (
            telemetry_sink is not None
            and otel_runtime is None
        ):
            raise ValueError(
                "telemetry_sink requires an OpenTelemetry runtime."
            )

        self._telemetry_sink: TelemetrySink | None = None

        if otel_runtime is not None:
            self._telemetry_sink = (
                telemetry_sink
                if telemetry_sink is not None
                else _NoOpTelemetrySink()
            )

    async def submit_action(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> OrchestrationResult:
        """Submit an agent-proposed action to the control plane."""

        if not self._telemetry_enabled():
            return await self._submit_action(
                incident=incident,
                action=action,
            )

        with self._workflow_observation(
            incident=incident,
            action_id=action.id,
            phase="submit",
        ) as observation:
            result = await self._submit_action(
                incident=incident,
                action=action,
            )

            self._annotate_result(
                observation=observation,
                result=result,
            )

            return result

    async def approve_and_resume(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str | None = None,
    ) -> OrchestrationResult:
        """Approve a pending action and resume execution."""

        if not self._telemetry_enabled():
            return await self._approve_and_resume(
                incident=incident,
                action_id=action_id,
                approver=approver,
                reason=reason,
            )

        with self._workflow_observation(
            incident=incident,
            action_id=action_id,
            phase="approve_and_resume",
        ) as observation:
            result = await self._approve_and_resume(
                incident=incident,
                action_id=action_id,
                approver=approver,
                reason=reason,
            )

            self._annotate_result(
                observation=observation,
                result=result,
            )

            return result

    def reject_action(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str,
    ) -> OrchestrationResult:
        """Reject a pending action without executing its tool."""

        if not self._telemetry_enabled():
            return self._reject_action(
                incident=incident,
                action_id=action_id,
                approver=approver,
                reason=reason,
            )

        with self._workflow_observation(
            incident=incident,
            action_id=action_id,
            phase="reject",
        ) as observation:
            result = self._reject_action(
                incident=incident,
                action_id=action_id,
                approver=approver,
                reason=reason,
            )

            self._annotate_result(
                observation=observation,
                result=result,
            )

            return result

    async def _submit_action(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> OrchestrationResult:
        """Execute the submit path without creating another workflow span."""

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
                    error=(
                        "Action was rejected by a human reviewer."
                    ),
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

    async def _approve_and_resume(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str | None,
    ) -> OrchestrationResult:
        """Apply approval and resume without creating another span."""

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

    def _reject_action(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str,
    ) -> OrchestrationResult:
        """Apply rejection without creating another workflow span."""

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

        incident.add_action(
            action
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

    def _telemetry_enabled(
        self,
    ) -> bool:
        """Return True when workflow telemetry is configured."""

        return (
            self._otel_runtime is not None
            and self._telemetry_sink is not None
        )

    def _workflow_observation(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        phase: str,
    ) -> OpenTelemetryObservation:
        """Create the root workflow observation for one control action."""

        assert self._otel_runtime is not None
        assert self._telemetry_sink is not None

        return OpenTelemetryObservation(
            runtime=self._otel_runtime,
            sink=self._telemetry_sink,
            name=(
                f"invoke_workflow "
                f"{self.WORKFLOW_NAME}"
            ),
            kind=SpanKind.INTERNAL,
            incident_id=str(incident.id),
            action_id=str(action_id),
            attributes={
                "gen_ai.operation.name": (
                    "invoke_workflow"
                ),
                "gen_ai.workflow.name": (
                    self.WORKFLOW_NAME
                ),
                "aegisops.workflow.phase": phase,
                "aegisops.environment": (
                    incident.environment
                ),
                "aegisops.service": (
                    incident.service
                ),
                "aegisops.incident.severity": (
                    incident.severity.value
                ),
            },
        )

    @staticmethod
    def _annotate_result(
        *,
        observation: OpenTelemetryObservation,
        result: OrchestrationResult,
    ) -> None:
        """Attach low-cardinality workflow outcome metadata."""

        observation.set_attribute(
            "aegisops.orchestration.status",
            result.status.value,
        )

        observation.set_attribute(
            "aegisops.policy.effect",
            result.policy_decision.effect.value,
        )

        observation.set_attribute(
            "aegisops.policy.version",
            result.policy_decision.policy_version,
        )

        observation.set_attribute(
            "aegisops.policy.reasons",
            [
                reason.value
                for reason
                in result.policy_decision.reasons
            ],
        )

        if result.status is OrchestrationStatus.FAILED:
            observation.set_attribute(
                "error.type",
                "tool_execution_error",
            )

            observation.set_status(
                SpanStatus.ERROR
            )