"""Durable incident workflow orchestration for the AegisOps control plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from aegisops.approval import ApprovalManager
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.checkpoint import (
    CheckpointConflictError,
    SQLiteCheckpointStore,
)
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
    """Coordinate policy, approvals, execution, persistence, and tracing."""

    WORKFLOW_NAME = "incident-response"

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        approval_manager: ApprovalManager,
        tool_runtime: ToolRuntime,
        audit_ledger: AuditLedger,
        checkpoint_store: SQLiteCheckpointStore | None = None,
        otel_runtime: OpenTelemetryRuntime | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._approval_manager = approval_manager
        self._tool_runtime = tool_runtime
        self._audit_ledger = audit_ledger

        self._checkpoint_store = checkpoint_store

        self._checkpoint_versions: dict[
            UUID,
            int,
        ] = {}

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

    def restore_incident(
        self,
        incident_id: UUID | str,
    ) -> IncidentState:
        """Restore authoritative incident state from durable storage.

        The restored checkpoint version is retained locally so all
        subsequent writes use optimistic concurrency control.
        """

        if self._checkpoint_store is None:
            raise RuntimeError(
                "No checkpoint store is configured."
            )

        snapshot = self._checkpoint_store.load_latest(
            incident_id
        )

        self._checkpoint_versions[
            snapshot.incident_id
        ] = snapshot.version

        incident = snapshot.state.model_copy(
            deep=True
        )

        self._audit_ledger.append(
            event_type=AuditEventType.INCIDENT_STATE_CHANGED,
            actor="aegisops-checkpoint-store",
            subject_id=str(
                incident.id
            ),
            payload={
                "event": "checkpoint_restored",
                "checkpoint_version": snapshot.version,
                "incident_status": incident.status.value,
            },
        )

        return incident

    def checkpoint_version(
        self,
        incident_id: UUID | str,
    ) -> int | None:
        """Return the orchestrator's tracked durable version."""

        normalized_id = (
            incident_id
            if isinstance(
                incident_id,
                UUID,
            )
            else UUID(
                str(
                    incident_id
                )
            )
        )

        return self._checkpoint_versions.get(
            normalized_id
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
                incident=incident,
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
                incident=incident,
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
                incident=incident,
            )

            return result

    async def _submit_action(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> OrchestrationResult:
        """Execute the submit path without another workflow span."""

        self._register_action(
            incident=incident,
            action=action,
        )

        self._persist_checkpoint(
            incident=incident,
            reason="action_registered",
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
        """Persist approval before allowing its side effect."""

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

        self._persist_checkpoint(
            incident=incident,
            reason="approval_recorded",
        )

        return await self._attempt_execution(
            incident=incident,
            action=action,
            policy_decision=policy_decision,
            checkpoint_before_execution=False,
        )

    def _reject_action(
        self,
        *,
        incident: IncidentState,
        action_id: UUID,
        approver: str,
        reason: str,
    ) -> OrchestrationResult:
        """Persist rejection before returning control."""

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

        self._persist_checkpoint(
            incident=incident,
            reason="rejection_recorded",
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
        """Pass through runtime so policy pause remains audited."""

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
        checkpoint_before_execution: bool = True,
    ) -> OrchestrationResult:
        """Execute an authorized action through the controlled runtime."""

        if checkpoint_before_execution:
            self._persist_checkpoint(
                incident=incident,
                reason="before_tool_execution",
            )

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

    def _persist_checkpoint(
        self,
        *,
        incident: IncidentState,
        reason: str,
    ) -> int | None:
        """Persist authoritative state with compare-and-swap semantics.

        A process must explicitly restore an existing incident before
        updating it. This prevents a fresh or stale process from
        silently overwriting durable state owned by another worker.
        """

        if self._checkpoint_store is None:
            return None

        expected_version = self._checkpoint_versions.get(
            incident.id
        )

        if expected_version is None:
            current_version = (
                self._checkpoint_store.current_version(
                    incident.id
                )
            )

            if current_version != 0:
                raise CheckpointConflictError(
                    "Incident already has durable state. "
                    "Restore it with restore_incident() before "
                    f"continuing: incident={incident.id}, "
                    f"stored_version={current_version}."
                )

            expected_version = 0

        snapshot = self._checkpoint_store.save(
            incident,
            expected_version=expected_version,
        )

        self._checkpoint_versions[
            incident.id
        ] = snapshot.version

        self._audit_ledger.append(
            event_type=AuditEventType.INCIDENT_STATE_CHANGED,
            actor="aegisops-checkpoint-store",
            subject_id=str(
                incident.id
            ),
            payload={
                "event": "checkpoint_saved",
                "checkpoint_version": snapshot.version,
                "previous_version": expected_version,
                "reason": reason,
                "incident_status": incident.status.value,
            },
        )

        return snapshot.version

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

        attributes: dict[str, object] = {
            "gen_ai.operation.name": "invoke_workflow",
            "gen_ai.workflow.name": self.WORKFLOW_NAME,
            "aegisops.workflow.phase": phase,
            "aegisops.environment": incident.environment,
            "aegisops.service": incident.service,
            "aegisops.incident.severity": (
                incident.severity.value
            ),
            "aegisops.checkpoint.enabled": (
                self._checkpoint_store is not None
            ),
        }

        checkpoint_version = (
            self._checkpoint_versions.get(
                incident.id
            )
        )

        if checkpoint_version is not None:
            attributes[
                "aegisops.checkpoint.version"
            ] = checkpoint_version

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
            attributes=attributes,
        )

    def _annotate_result(
        self,
        *,
        observation: OpenTelemetryObservation,
        result: OrchestrationResult,
        incident: IncidentState,
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

        checkpoint_version = (
            self._checkpoint_versions.get(
                incident.id
            )
        )

        if checkpoint_version is not None:
            observation.set_attribute(
                "aegisops.checkpoint.version",
                checkpoint_version,
            )

        if result.status is OrchestrationStatus.FAILED:
            observation.set_attribute(
                "error.type",
                "tool_execution_error",
            )

            observation.set_status(
                SpanStatus.ERROR
            )