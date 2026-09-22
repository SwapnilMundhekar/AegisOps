"""Controlled tool execution runtime for AegisOps."""

from __future__ import annotations

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
    """Raised when human approval has not yet been granted."""


class ToolExecutionError(ToolRuntimeError):
    """Raised when an underlying tool fails."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Tool metadata stored in the runtime registry."""

    name: str
    handler: ToolHandler


class ToolRuntime:
    """Execute registered tools behind AegisOps policy controls."""

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
    ) -> None:
        """Register a tool with a unique normalized name."""

        normalized_name = self._normalize_name(name)

        if not normalized_name:
            raise ValueError(
                "Tool name must not be empty."
            )

        if normalized_name in self._tools:
            raise ValueError(
                f"Tool already registered: {normalized_name}"
            )

        self._tools[normalized_name] = RegisteredTool(
            name=normalized_name,
            handler=handler,
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

    async def execute(
        self,
        *,
        incident: IncidentState,
        action: ProposedAction,
    ) -> object:
        """Evaluate policy and execute an authorized tool action."""

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

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_STARTED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "tool_name": tool.name,
                "arguments": action.tool_arguments,
            },
        )

        try:
            result = tool.handler(
                action.tool_arguments
            )

            if inspect.isawaitable(result):
                result = await result

        except Exception as exc:
            self._audit_ledger.append(
                event_type=AuditEventType.TOOL_FAILED,
                actor=action.agent,
                subject_id=str(action.id),
                payload={
                    "incident_id": str(incident.id),
                    "tool_name": tool.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

            raise ToolExecutionError(
                f"Tool execution failed: {tool.name}"
            ) from exc

        self._audit_ledger.append(
            event_type=AuditEventType.TOOL_COMPLETED,
            actor=action.agent,
            subject_id=str(action.id),
            payload={
                "incident_id": str(incident.id),
                "tool_name": tool.name,
                "result_type": type(result).__name__,
            },
        )

        return result

    def _get_tool(
        self,
        name: str,
    ) -> RegisteredTool:
        """Resolve a registered tool by normalized name."""

        normalized_name = self._normalize_name(name)

        try:
            return self._tools[normalized_name]
        except KeyError as exc:
            raise ToolNotRegisteredError(
                f"Tool is not registered: {normalized_name}"
            ) from exc

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """Normalize tool identity for safe registry lookups."""

        return name.strip().casefold()