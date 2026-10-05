"""Policy-controlled and resilient tool execution runtime for AegisOps."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Execution metadata for a registered tool."""

    name: str
    handler: ToolHandler

    idempotent: bool
    timeout_seconds: float | None

    executor: ResilientExecutor


class ToolRuntime:
    """Execute registered tools behind AegisOps control boundaries."""

    def __init__(
        self,
        *,
        action_gate: ActionGate,
        approval_manager: ApprovalManager,
        audit_ledger: AuditLedger,
    ) -> None:
        self._action_gate = action_gate
        self._approval_manager = approval_manager
        self._audit_ledger = audit_ledger

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
    ) -> None:
        """Register a tool and its execution reliability contract.

        Tools default to non-idempotent so retries are fail-safe rather
        than opt-out. Each registered tool receives an independent
        circuit breaker.
        """

        normalized_name = self._normalize_name(name)

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

        self._tools[normalized_name] = RegisteredTool(
            name=normalized_name,
            handler=handler,
            idempotent=idempotent,
            timeout_seconds=timeout_seconds,
            executor=executor,
        )

    def unregister(
        self,
        name: str,
    ) -> None:
        """Remove a registered tool."""

        normalized_name = self._normalize_name(name)

        if normalized_name not in self._tools:
            raise ToolNotRegisteredError(
                f"Tool is not registered: {normalized_name}"
            )

        del self._tools[normalized_name]

    def registered_tools(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""

        return tuple(
            sorted(self._tools)
        )

    def circuit_state(
        self,
        name: str,
    ) -> str:
        """Return the current circuit state for a registered tool."""

        tool = self._get_tool(name)

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
        """Authorize and execute a registered tool action."""

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

        self._record_tool_started(
            incident=incident,
            action=action,
            tool=tool,
        )

        try:
            result = await tool.executor.execute(
                lambda: self._invoke_with_timeout(
                    tool=tool,
                    arguments=action.tool_arguments,
                ),
                idempotent=tool.idempotent,
            )

        except Exception as exc:
            self._record_tool_failed(
                incident=incident,
                action=action,
                tool=tool,
                error=exc,
            )

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

        self._record_tool_completed(
            incident=incident,
            action=action,
            tool=tool,
            result=result,
        )

        return result

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

        if inspect.isawaitable(result):
            return await result

        return result

    def _record_tool_started(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
    ) -> None:
        """Record the start of an authorized tool invocation."""

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_STARTED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "tool_name": tool.name,
                "arguments": action.tool_arguments,
                "idempotent": tool.idempotent,
                "timeout_seconds": tool.timeout_seconds,
                "circuit_state": self.circuit_state(
                    tool.name
                ),
            },
        )

    def _record_tool_completed(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        result: object,
    ) -> None:
        """Record successful tool execution."""

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_COMPLETED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "tool_name": tool.name,
                "result_type": type(result).__name__,
                "circuit_state": self.circuit_state(
                    tool.name
                ),
            },
        )

    def _record_tool_failed(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
        tool: RegisteredTool,
        error: Exception,
    ) -> None:
        """Record execution failure and current circuit state."""

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_FAILED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "tool_name": tool.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "idempotent": tool.idempotent,
                "circuit_state": self.circuit_state(
                    tool.name
                ),
            },
        )

    def _get_tool(
        self,
        name: str,
    ) -> RegisteredTool:
        """Resolve a registered tool by normalized identity."""

        normalized_name = self._normalize_name(name)

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