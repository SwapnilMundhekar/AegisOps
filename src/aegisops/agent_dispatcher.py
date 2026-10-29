
"""Bounded, contract-validated specialist agent execution for AegisOps."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from aegisops.agent_contracts import (
    AgentContractViolation,
    AgentResult,
    AgentRunStatus,
    AgentTask,
    SupervisorRoute,
)
from aegisops.supervisor import Supervisor


class AgentDispatchError(RuntimeError):
    """Raised when an agent cannot be safely scheduled for execution."""


class AgentHandler(Protocol):
    """An asynchronous implementation of a specialist agent."""

    async def __call__(self, task: AgentTask) -> AgentResult:
        """Execute a task and return its structured result."""
        ...


class AgentDispatcher:
    """Execute registered specialist agents under explicit contracts.

    Responsibilities:
    - Validate tasks against application-owned agent specifications.
    - Dispatch tasks only to registered asynchronous handlers.
    - Limit concurrently executing handlers.
    - Enforce an overall deadline, including queue waiting time.
    - Validate identity, correlation, and proposed-action permissions.
    - Return structured failures for execution errors and timeouts.

    This component does not execute tools, approve actions, or perform
    automatic agent handoffs. Such operations must use their existing
    trusted control-plane boundaries.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        max_concurrent_tasks: int = 4,
    ) -> None:
        if not isinstance(max_concurrent_tasks, int):
            raise TypeError("max_concurrent_tasks must be an integer.")

        if isinstance(max_concurrent_tasks, bool):
            raise TypeError("max_concurrent_tasks must not be a boolean.")

        if max_concurrent_tasks < 1:
            raise ValueError(
                "max_concurrent_tasks must be at least 1."
            )

        if max_concurrent_tasks > 128:
            raise ValueError(
                "max_concurrent_tasks must not exceed 128."
            )

        self._supervisor = supervisor
        self._handlers: dict[str, AgentHandler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._max_concurrent_tasks = max_concurrent_tasks

    @property
    def max_concurrent_tasks(self) -> int:
        return self._max_concurrent_tasks

    @property
    def registered_handlers(self) -> tuple[str, ...]:
        """Return handler names in deterministic order."""
        return tuple(sorted(self._handlers))

    def register_handler(
        self,
        agent_name: str,
        handler: AgentHandler,
        *,
        replace: bool = False,
    ) -> None:
        """Register an asynchronous handler for a known agent.

        An agent must already exist in the Supervisor registry.
        A synchronous handler is not accepted.
        """
        self._supervisor.get_agent(agent_name)

        is_async_function = inspect.iscoroutinefunction(handler)

        is_async_callable = inspect.iscoroutinefunction(
            getattr(handler, "__call__", None)
        )

        if not (is_async_function or is_async_callable):
            raise TypeError(
                f"Handler for '{agent_name}' must be asynchronous."
            )

        if agent_name in self._handlers and not replace:
            raise AgentDispatchError(
                f"Handler for '{agent_name}' is already registered."
            )

        self._handlers[agent_name] = handler

    def unregister_handler(self, agent_name: str) -> None:
        """Remove a handler without altering the Supervisor registry."""
        try:
            del self._handlers[agent_name]
        except KeyError as exc:
            raise AgentDispatchError(
                f"No handler is registered for '{agent_name}'."
            ) from exc

    def _get_handler(self, agent_name: str) -> AgentHandler:
        """Return a registered handler or reject execution."""
        try:
            return self._handlers[agent_name]
        except KeyError as exc:
            raise AgentDispatchError(
                f"No handler is registered for '{agent_name}'."
            ) from exc

    @staticmethod
    def _failed_result(
        task: AgentTask,
        *,
        code: str,
        summary: str,
    ) -> AgentResult:
        """Create a correctly correlated, non-actionable failure."""
        return AgentResult(
            task_id=task.task_id,
            agent_name=task.target_agent,
            status=AgentRunStatus.FAILED,
            summary=summary,
            findings=(),
            proposed_action_ids=(),
            error_code=code,
        )

    def _validate_assignment(self, task: AgentTask) -> None:
        """Recheck the latest agent specification and handler registry."""
        agent = self._supervisor.get_agent(task.target_agent)
        agent.validate_task(task)
        self._get_handler(task.target_agent)

    async def _invoke(self, task: AgentTask) -> AgentResult:
        """Invoke one agent and reject invalid or unauthorized results."""
        agent = self._supervisor.get_agent(task.target_agent)
        agent.validate_task(task)

        handler = self._get_handler(task.target_agent)

        try:
            result = await handler(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._failed_result(
                task,
                code="AGENT_EXECUTION_ERROR",
                summary=(
                    "The specialist agent failed during execution. "
                    "The result contains no authorized actions."
                ),
            )

        if not isinstance(result, AgentResult):
            return self._failed_result(
                task,
                code="INVALID_AGENT_RESULT",
                summary=(
                    "The specialist agent returned an invalid result type."
                ),
            )

        try:
            # Recheck current permissions in case registration changed
            # while the handler was running.
            current_agent = self._supervisor.get_agent(
                task.target_agent
            )
            current_agent.validate_task(task)
            current_agent.validate_result(task, result)

            # A handoff is only a request. Never execute it here.
            if result.status is AgentRunStatus.HANDOFF:
                if result.handoff_to == task.target_agent:
                    raise AgentContractViolation(
                        "An agent cannot hand off to itself."
                    )

                if result.handoff_to is None:
                    raise AgentContractViolation(
                        "A handoff must specify another agent."
                    )

                self._supervisor.get_agent(result.handoff_to)

        except (
            AgentContractViolation,
            KeyError,
            ValueError,
        ):
            return self._failed_result(
                task,
                code="AGENT_CONTRACT_ERROR",
                summary=(
                    "The specialist agent result violated its "
                    "assigned contract or current permissions."
                ),
            )

        return result

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute one validated specialist task.

        The timeout includes time spent waiting for a concurrency slot.

        A timeout requests cancellation of the running coroutine.
        It does not prove external side effects were rolled back.
        """
        self._validate_assignment(task)

        try:
            async with asyncio.timeout(task.deadline_seconds):
                async with self._semaphore:
                    # Recheck permissions and registration after waiting.
                    self._validate_assignment(task)
                    return await self._invoke(task)

        except TimeoutError:
            return self._failed_result(
                task,
                code="AGENT_TIMEOUT",
                summary=(
                    "The specialist agent exceeded its task deadline. "
                    "Any uncertain external side effects require "
                    "separate reconciliation."
                ),
            )

    async def execute_route(
        self,
        route: SupervisorRoute,
        *,
        incident_id: str,
        context: Mapping[str, Any] | None = None,
        correlation_id: UUID | None = None,
    ) -> AgentResult:
        """Validate a supervisor route and execute its assigned agent."""
        task = self._supervisor.dispatch(
            route,
            incident_id=incident_id,
            context=dict(context or {}),
            correlation_id=correlation_id,
        )

        return await self.execute(task)
