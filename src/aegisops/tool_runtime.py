"""Policy-controlled, durable, resilient, and observable tool runtime."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from aegisops.action_gate import (
    ActionGate,
    GateStatus,
)
from aegisops.approval import ApprovalManager
from aegisops.audit import (
    AuditEventType,
    AuditLedger,
)
from aegisops.domain import (
    IncidentState,
    ProposedAction,
)
from aegisops.execution_journal import (
    ExecutionAlreadyCompletedError,
    ExecutionJournalError,
    ExecutionLeaseHeldError,
    ExecutionLeaseLostError,
    ExecutionOutcomeUnknownError,
    ExecutionRecord,
    ExecutionStatus,
    SQLiteExecutionJournal,
)
from aegisops.observability import (
    SpanKind,
    SpanRecord,
    TelemetrySink,
)
from aegisops.otel import (
    OpenTelemetryObservation,
    OpenTelemetryRuntime,
)
from aegisops.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    ResilientExecutor,
    RetryPolicy,
)


ToolHandler = Callable[
    [dict[str, object]],
    object | Awaitable[object],
]


class ToolRuntimeError(RuntimeError):
    """Base error raised by the controlled tool runtime."""


class ToolNotRegisteredError(ToolRuntimeError):
    """Raised when an action references an unknown tool."""


class ToolExecutionBlockedError(ToolRuntimeError):
    """Raised when policy prevents tool execution."""


class ToolApprovalRequiredError(ToolRuntimeError):
    """Raised when required human approval has not been granted."""


class ToolExecutionError(ToolRuntimeError):
    """Raised when an underlying tool invocation fails."""


class ToolExecutionInProgressError(ToolExecutionError):
    """Raised when another worker owns the execution lease."""


class ToolExecutionAlreadyCompletedError(ToolExecutionError):
    """Raised when durable state says the side effect already succeeded."""


class ToolExecutionUncertainError(ToolExecutionError):
    """Raised when repeating an external side effect may be unsafe."""


class _NoOpTelemetrySink:
    """Discard local spans when only OpenTelemetry export is required."""

    def emit(
        self,
        span: SpanRecord,
    ) -> None:
        """Discard a completed AegisOps span record."""

        del span


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Execution metadata for a registered tool."""

    name: str
    handler: ToolHandler

    idempotent: bool
    timeout_seconds: float | None

    executor: ResilientExecutor

    idempotency_argument: str | None = None


class ToolRuntime:
    """Execute tools behind policy, durability, and reliability controls."""

    def __init__(
        self,
        *,
        action_gate: ActionGate,
        approval_manager: ApprovalManager,
        audit_ledger: AuditLedger,
        execution_journal: SQLiteExecutionJournal | None = None,
        worker_id: str | None = None,
        execution_lease_seconds: float = 30.0,
        otel_runtime: OpenTelemetryRuntime | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        if execution_lease_seconds <= 0:
            raise ValueError(
                "execution_lease_seconds must be greater than zero."
            )

        self._action_gate = action_gate
        self._approval_manager = approval_manager
        self._audit_ledger = audit_ledger

        self._execution_journal = execution_journal

        self._worker_id = (
            worker_id.strip()
            if worker_id is not None
            else f"aegisops-worker-{uuid4()}"
        )

        if not self._worker_id:
            raise ValueError(
                "worker_id must not be empty."
            )

        self._execution_lease_seconds = (
            execution_lease_seconds
        )

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

        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        handler: ToolHandler,
        idempotent: bool = False,
        timeout_seconds: float | None = 30.0,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        idempotency_argument: str | None = None,
    ) -> None:
        """Register a tool and its execution reliability contract.

        idempotency_argument can be used for downstream APIs that
        accept a client-generated idempotency token. A stable token
        derived from incident/action/tool identity is then injected
        into the invocation arguments.

        Because that token is what makes replay safe, the tool must
        also be declared idempotent.
        """

        normalized_name = self._normalize_name(
            name
        )

        if not normalized_name:
            raise ValueError(
                "Tool name must not be empty."
            )

        if normalized_name in self._tools:
            raise ValueError(
                f"Tool already registered: {normalized_name}"
            )

        if (
            timeout_seconds is not None
            and timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be greater than zero "
                "or None."
            )

        normalized_idempotency_argument: str | None = None

        if idempotency_argument is not None:
            normalized_idempotency_argument = (
                idempotency_argument.strip()
            )

            if not normalized_idempotency_argument:
                raise ValueError(
                    "idempotency_argument must not be empty."
                )

            if not idempotent:
                raise ValueError(
                    "A tool using a downstream idempotency key "
                    "must be registered with idempotent=True."
                )

            if self._execution_journal is None:
                raise ValueError(
                    "idempotency_argument requires an "
                    "execution journal."
                )

        executor = ResilientExecutor(
            retry_policy=(
                retry_policy
                or RetryPolicy()
            ),
            circuit_breaker=(
                circuit_breaker
                or CircuitBreaker()
            ),
        )

        self._tools[
            normalized_name
        ] = RegisteredTool(
            name=normalized_name,
            handler=handler,
            idempotent=idempotent,
            timeout_seconds=timeout_seconds,
            executor=executor,
            idempotency_argument=(
                normalized_idempotency_argument
            ),
        )

    def unregister(
        self,
        name: str,
    ) -> None:
        """Remove a registered tool."""

        normalized_name = self._normalize_name(
            name
        )

        if normalized_name not in self._tools:
            raise ToolNotRegisteredError(
                f"Tool is not registered: {normalized_name}"
            )

        del self._tools[
            normalized_name
        ]

    def registered_tools(
        self,
    ) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""

        return tuple(
            sorted(
                self._tools
            )
        )

    def circuit_state(
        self,
        name: str,
    ) -> str:
        """Return the current circuit state for a registered tool."""

        tool = self._get_tool(
            name
        )

        return (
            tool.executor
            .circuit_breaker
            .state
            .value
        )

    async def execute(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> object:
        """Authorize, journal, lease, and execute a tool action."""

        tool = self._get_tool(
            action.tool_name
        )

        gate_result = self._action_gate.evaluate(
            action,
            environment=incident.environment,
        )

        if gate_result.status is GateStatus.BLOCK:
            raise ToolExecutionBlockedError(
                f"Action {action.id} was denied by policy."
            )

        if (
            gate_result.status
            is GateStatus.PAUSE_FOR_APPROVAL
            and not self._approval_manager.is_approved(
                incident=incident,
                action_id=action.id,
            )
        ):
            raise ToolApprovalRequiredError(
                f"Action {action.id} requires human approval."
            )

        execution_record = self._claim_execution(
            incident=incident,
            action=action,
            tool=tool,
        )

        idempotency_key = (
            execution_record.idempotency_key
            if execution_record is not None
            else None
        )

        invocation_arguments = (
            self._build_invocation_arguments(
                action=action,
                tool=tool,
                idempotency_key=idempotency_key,
            )
        )

        self._record_tool_started(
            incident=incident,
            action=action,
            tool=tool,
            execution_record=execution_record,
        )

        if (
            self._otel_runtime is not None
            and self._telemetry_sink is not None
        ):
            return await self._execute_observed(
                incident=incident,
                action=action,
                tool=tool,
                arguments=invocation_arguments,
                execution_record=execution_record,
            )

        return await self._execute_and_finalize(
            incident=incident,
            action=action,
            tool=tool,
            arguments=invocation_arguments,
            execution_record=execution_record,
        )

    async def _execute_observed(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        arguments: dict[str, object],
        execution_record: ExecutionRecord | None,
    ) -> object:
        """Execute a tool inside a correlated GenAI tool span."""

        assert self._otel_runtime is not None
        assert self._telemetry_sink is not None

        attributes: dict[str, object] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "gen_ai.agent.name": action.agent,
            "aegisops.action.type": (
                action.action_type.value
            ),
            "aegisops.risk.level": action.risk.value,
            "aegisops.environment": incident.environment,
            "aegisops.tool.idempotent": tool.idempotent,
            "aegisops.tool.circuit_state.before": (
                self.circuit_state(
                    tool.name
                )
            ),
            "aegisops.execution.journaled": (
                execution_record is not None
            ),
        }

        if tool.timeout_seconds is not None:
            attributes[
                "aegisops.tool.timeout_seconds"
            ] = tool.timeout_seconds

        if execution_record is not None:
            attributes[
                "aegisops.execution.attempt"
            ] = execution_record.attempt_count

            attributes[
                "aegisops.execution.status.before"
            ] = execution_record.status.value

        with OpenTelemetryObservation(
            runtime=self._otel_runtime,
            sink=self._telemetry_sink,
            name=f"execute_tool {tool.name}",
            kind=SpanKind.TOOL,
            incident_id=str(
                incident.id
            ),
            action_id=str(
                action.id
            ),
            agent_name=action.agent,
            attributes=attributes,
        ) as observation:
            try:
                result = await self._execute_and_finalize(
                    incident=incident,
                    action=action,
                    tool=tool,
                    arguments=arguments,
                    execution_record=execution_record,
                )

            except ToolExecutionError as exc:
                observation.set_attribute(
                    "aegisops.tool.outcome",
                    "error",
                )

                observation.set_attribute(
                    "aegisops.tool.circuit_state.after",
                    self.circuit_state(
                        tool.name
                    ),
                )

                observation.set_attribute(
                    "error.type",
                    self._error_type(
                        exc
                    ),
                )

                journal_status = (
                    self._journal_status(
                        execution_record
                    )
                )

                if journal_status is not None:
                    observation.set_attribute(
                        "aegisops.execution.status.after",
                        journal_status,
                    )

                raise

            observation.set_attribute(
                "aegisops.tool.outcome",
                "success",
            )

            observation.set_attribute(
                "aegisops.tool.circuit_state.after",
                self.circuit_state(
                    tool.name
                ),
            )

            journal_status = self._journal_status(
                execution_record
            )

            if journal_status is not None:
                observation.set_attribute(
                    "aegisops.execution.status.after",
                    journal_status,
                )

            return result

    async def _execute_and_finalize(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        arguments: dict[str, object],
        execution_record: ExecutionRecord | None,
    ) -> object:
        """Execute a leased action and durably finalize its outcome."""

        try:
            result = await self._invoke_authorized_tool(
                tool=tool,
                arguments=arguments,
            )

        except ToolExecutionError as exc:
            journal_status = self._finalize_journal_failure(
                execution_record=execution_record,
                tool=tool,
                error=exc,
            )

            self._record_tool_failed(
                incident=incident,
                action=action,
                tool=tool,
                error=exc,
                journal_status=journal_status,
            )

            raise

        try:
            journal_status = (
                self._finalize_journal_success(
                    execution_record=execution_record,
                    result=result,
                )
            )

        except ExecutionJournalError as exc:
            uncertain = ToolExecutionUncertainError(
                "Tool returned successfully but AegisOps could "
                "not durably finalize execution ownership; "
                f"manual reconciliation is required: {tool.name}"
            )

            self._record_tool_failed(
                incident=incident,
                action=action,
                tool=tool,
                error=uncertain,
                journal_status=self._journal_status(
                    execution_record
                ),
            )

            raise uncertain from exc

        self._record_tool_completed(
            incident=incident,
            action=action,
            tool=tool,
            result=result,
            journal_status=journal_status,
        )

        return result

    async def _invoke_authorized_tool(
        self,
        *,
        tool: RegisteredTool,
        arguments: dict[str, object],
    ) -> object:
        """Run a leased external operation through resilience controls."""

        try:
            return await tool.executor.execute(
                lambda: self._invoke_with_timeout(
                    tool=tool,
                    arguments=arguments,
                ),
                idempotent=tool.idempotent,
            )

        except Exception as exc:
            if isinstance(
                exc,
                CircuitOpenError,
            ):
                raise ToolExecutionError(
                    f"Tool circuit is open: {tool.name}"
                ) from exc

            if isinstance(
                exc,
                TimeoutError,
            ):
                raise ToolExecutionError(
                    f"Tool execution timed out: {tool.name}"
                ) from exc

            raise ToolExecutionError(
                f"Tool execution failed: {tool.name}"
            ) from exc

    def _claim_execution(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
    ) -> ExecutionRecord | None:
        """Prepare and exclusively claim a durable external operation."""

        if self._execution_journal is None:
            return None

        prepared = self._execution_journal.prepare(
            incident_id=incident.id,
            action_id=action.id,
            tool_name=tool.name,
            idempotent=tool.idempotent,
        )

        try:
            return self._execution_journal.claim(
                idempotency_key=prepared.idempotency_key,
                worker_id=self._worker_id,
                lease_seconds=self._execution_lease_seconds,
            )

        except ExecutionLeaseHeldError as exc:
            raise ToolExecutionInProgressError(
                "Tool execution is currently owned by another "
                f"worker: {tool.name}"
            ) from exc

        except ExecutionAlreadyCompletedError as exc:
            raise ToolExecutionAlreadyCompletedError(
                "Tool execution was already completed and will "
                f"not be repeated: {tool.name}"
            ) from exc

        except ExecutionOutcomeUnknownError as exc:
            raise ToolExecutionUncertainError(
                "Previous execution outcome is uncertain; "
                f"automatic replay is blocked: {tool.name}"
            ) from exc

    def _finalize_journal_success(
        self,
        *,
        execution_record: ExecutionRecord | None,
        result: object,
    ) -> str | None:
        """Durably mark the leased operation successful."""

        if (
            self._execution_journal is None
            or execution_record is None
        ):
            return None

        lease_token = execution_record.lease_token

        if lease_token is None:
            raise ExecutionLeaseLostError(
                "Claimed execution has no lease token."
            )

        completed = self._execution_journal.record_success(
            idempotency_key=execution_record.idempotency_key,
            lease_token=lease_token,
            result_digest=self._result_digest(
                result
            ),
        )

        return completed.status.value

    def _finalize_journal_failure(
        self,
        *,
        execution_record: ExecutionRecord | None,
        tool: RegisteredTool,
        error: ToolExecutionError,
    ) -> str | None:
        """Durably classify a failed external attempt."""

        if (
            self._execution_journal is None
            or execution_record is None
        ):
            return None

        lease_token = execution_record.lease_token

        if lease_token is None:
            raise ToolExecutionUncertainError(
                "Execution lease disappeared while recording "
                f"failure: {tool.name}"
            )

        retry_safe = self._failure_retry_safe(
            tool=tool,
            error=error,
        )

        try:
            failed = self._execution_journal.record_failure(
                idempotency_key=execution_record.idempotency_key,
                lease_token=lease_token,
                error_type=self._error_type(
                    error
                ),
                retry_safe=retry_safe,
            )

        except ExecutionJournalError as exc:
            raise ToolExecutionUncertainError(
                "AegisOps could not durably classify the failed "
                f"external operation: {tool.name}"
            ) from exc

        return failed.status.value

    def _build_invocation_arguments(
        self,
        *,
        action: ProposedAction,
        tool: RegisteredTool,
        idempotency_key: str | None,
    ) -> dict[str, object]:
        """Build isolated tool arguments and inject stable dedupe key."""

        arguments = dict(
            action.tool_arguments
        )

        argument_name = tool.idempotency_argument

        if argument_name is None:
            return arguments

        if idempotency_key is None:
            raise ToolExecutionError(
                "Downstream idempotency injection requires "
                "a durable execution journal."
            )

        existing = arguments.get(
            argument_name
        )

        if (
            existing is not None
            and existing != idempotency_key
        ):
            raise ToolExecutionError(
                "Tool arguments contain a conflicting "
                f"idempotency key: {argument_name}"
            )

        arguments[
            argument_name
        ] = idempotency_key

        return arguments

    async def _invoke_with_timeout(
        self,
        *,
        tool: RegisteredTool,
        arguments: dict[str, object],
    ) -> object:
        """Invoke a tool with its configured execution timeout."""

        invocation = self._invoke_handler(
            tool=tool,
            arguments=arguments,
        )

        if tool.timeout_seconds is None:
            return await invocation

        return await asyncio.wait_for(
            invocation,
            timeout=tool.timeout_seconds,
        )

    @staticmethod
    async def _invoke_handler(
        *,
        tool: RegisteredTool,
        arguments: dict[str, object],
    ) -> object:
        """Execute async handlers directly and sync handlers off-loop."""

        if inspect.iscoroutinefunction(
            tool.handler
        ):
            return await tool.handler(
                arguments
            )

        result = await asyncio.to_thread(
            tool.handler,
            arguments,
        )

        if inspect.isawaitable(
            result
        ):
            return await result

        return result

    def _record_tool_started(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        execution_record: ExecutionRecord | None,
    ) -> None:
        """Record the start of a leased tool invocation."""

        payload: dict[str, object] = {
            "incident_id": str(
                incident.id
            ),
            "tool_name": tool.name,
            "argument_keys": sorted(
                action.tool_arguments
            ),
            "idempotent": tool.idempotent,
            "timeout_seconds": tool.timeout_seconds,
            "circuit_state": self.circuit_state(
                tool.name
            ),
            "journaled": (
                execution_record is not None
            ),
        }

        if execution_record is not None:
            payload[
                "execution_attempt"
            ] = execution_record.attempt_count

            payload[
                "execution_status"
            ] = execution_record.status.value

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_STARTED,
            actor=action.agent,
            subject_id=str(
                action.id
            ),
            payload=payload,
        )

    def _record_tool_completed(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        result: object,
        journal_status: str | None,
    ) -> None:
        """Record successful durable tool execution."""

        payload: dict[str, object] = {
            "incident_id": str(
                incident.id
            ),
            "tool_name": tool.name,
            "result_type": type(
                result
            ).__name__,
            "circuit_state": self.circuit_state(
                tool.name
            ),
        }

        if journal_status is not None:
            payload[
                "execution_status"
            ] = journal_status

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_COMPLETED,
            actor=action.agent,
            subject_id=str(
                action.id
            ),
            payload=payload,
        )

    def _record_tool_failed(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        error: Exception,
        journal_status: str | None,
    ) -> None:
        """Record tool failure without leaking raw arguments."""

        payload: dict[str, object] = {
            "incident_id": str(
                incident.id
            ),
            "tool_name": tool.name,
            "error_type": type(
                error
            ).__name__,
            "error": str(
                error
            ),
            "idempotent": tool.idempotent,
            "circuit_state": self.circuit_state(
                tool.name
            ),
        }

        if journal_status is not None:
            payload[
                "execution_status"
            ] = journal_status

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_FAILED,
            actor=action.agent,
            subject_id=str(
                action.id
            ),
            payload=payload,
        )

    def _journal_status(
        self,
        execution_record: ExecutionRecord | None,
    ) -> str | None:
        """Return the current durable execution status."""

        if (
            self._execution_journal is None
            or execution_record is None
        ):
            return None

        try:
            latest = self._execution_journal.load(
                execution_record.idempotency_key
            )

        except ExecutionJournalError:
            return None

        return latest.status.value

    def _get_tool(
        self,
        name: str,
    ) -> RegisteredTool:
        """Resolve a registered tool by normalized identity."""

        normalized_name = self._normalize_name(
            name
        )

        try:
            return self._tools[
                normalized_name
            ]

        except KeyError as exc:
            raise ToolNotRegisteredError(
                f"Tool is not registered: {normalized_name}"
            ) from exc

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """Normalize tool identity for registry and policy lookups."""

        return name.strip().casefold()

    @staticmethod
    def _failure_retry_safe(
        *,
        tool: RegisteredTool,
        error: ToolExecutionError,
    ) -> bool:
        """Determine whether another durable attempt is safe.

        Idempotent operations may be repeated.

        For non-idempotent tools, a circuit-open failure is safe because
        the circuit rejected the call before invoking the tool handler.
        Other failures are conservatively treated as ambiguous.
        """

        if tool.idempotent:
            return True

        cause = error.__cause__

        return isinstance(
            cause,
            CircuitOpenError,
        )

    @staticmethod
    def _error_type(
        error: ToolExecutionError,
    ) -> str:
        """Return a bounded root failure classification."""

        cause = error.__cause__

        if cause is None:
            return type(
                error
            ).__name__

        if isinstance(
            cause,
            TimeoutError,
        ):
            return "timeout"

        if isinstance(
            cause,
            CircuitOpenError,
        ):
            return "circuit_open"

        return type(
            cause
        ).__name__

    @staticmethod
    def _result_digest(
        result: object,
    ) -> str:
        """Create a stable result fingerprint without storing payload."""

        canonical = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()