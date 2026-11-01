"""Bounded, observable, contract-validated specialist agent execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID

from aegisops.agent_contracts import (
    AgentContractViolation,
    AgentResult,
    AgentRunStatus,
    AgentTask,
    SupervisorRoute,
)
from aegisops.observability import Observation, SpanKind, SpanStatus, TelemetrySink
from aegisops.otel import OpenTelemetryObservation, OpenTelemetryRuntime
from aegisops.supervisor import Supervisor


class AgentDispatchError(RuntimeError):
    """Raised when a specialist cannot be safely scheduled."""


class AgentHandler(Protocol):
    """An asynchronous specialist agent implementation."""

    async def __call__(self, task: AgentTask) -> AgentResult:
        """Run one specialist task."""
        ...


AgentObservation = Observation | OpenTelemetryObservation


class AgentDispatcher:
    """Run registered async agents with bounded concurrency and optional tracing.

    Agent execution does not authorize external tool use, remediation,
    action approval, or automatic handoffs. Those must pass through the
    established trusted control-plane boundaries.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        max_concurrent_tasks: int = 4,
        telemetry_sink: TelemetrySink | None = None,
        otel_runtime: OpenTelemetryRuntime | None = None,
    ) -> None:
        if isinstance(max_concurrent_tasks, bool) or not isinstance(
            max_concurrent_tasks, int
        ):
            raise TypeError("max_concurrent_tasks must be an integer, not a boolean.")
        if not 1 <= max_concurrent_tasks <= 128:
            raise ValueError("max_concurrent_tasks must be between 1 and 128.")
        if otel_runtime is not None and telemetry_sink is None:
            raise ValueError("otel_runtime requires telemetry_sink.")

        self._supervisor = supervisor
        self._handlers: dict[str, AgentHandler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._max_concurrent_tasks = max_concurrent_tasks
        self._telemetry_sink = telemetry_sink
        self._otel_runtime = otel_runtime

    @property
    def max_concurrent_tasks(self) -> int:
        return self._max_concurrent_tasks

    @property
    def registered_handlers(self) -> tuple[str, ...]:
        """Return registered handler names in deterministic order."""
        return tuple(sorted(self._handlers))

    def register_handler(
        self,
        agent_name: str,
        handler: AgentHandler,
        *,
        replace: bool = False,
    ) -> None:
        """Bind an asynchronous handler to an existing supervisor agent."""
        self._supervisor.get_agent(agent_name)
        is_async_function = inspect.iscoroutinefunction(handler)
        is_async_callable = inspect.iscoroutinefunction(
            getattr(handler, "__call__", None)
        )
        if not (is_async_function or is_async_callable):
            raise TypeError(f"Handler for '{agent_name}' must be asynchronous.")
        if agent_name in self._handlers and not replace:
            raise AgentDispatchError(
                f"Handler for '{agent_name}' is already registered."
            )
        self._handlers[agent_name] = handler

    def unregister_handler(self, agent_name: str) -> None:
        """Unbind a handler without changing the supervisor registry."""
        try:
            del self._handlers[agent_name]
        except KeyError as exc:
            raise AgentDispatchError(
                f"No handler is registered for '{agent_name}'."
            ) from exc

    def _get_handler(self, agent_name: str) -> AgentHandler:
        try:
            return self._handlers[agent_name]
        except KeyError as exc:
            raise AgentDispatchError(
                f"No handler is registered for '{agent_name}'."
            ) from exc

    def _observation(
        self,
        task: AgentTask,
        *,
        name: str,
        attributes: dict[str, object] | None = None,
    ) -> AbstractContextManager[AgentObservation | None]:
        """Create a span without including prompt, context or result content."""
        if self._telemetry_sink is None:
            return nullcontext(None)

        safe_attributes: dict[str, object] = {
            "aegisops.agent.task_id": str(task.task_id),
            "aegisops.agent.priority": task.priority.value,
            "aegisops.agent.deadline_seconds": task.deadline_seconds,
            "aegisops.agent.max_steps": task.max_steps,
            "aegisops.agent.action_proposals_requested": bool(
                task.requested_action_types
            ),
        }
        if task.correlation_id is not None:
            safe_attributes["aegisops.agent.correlation_id"] = str(
                task.correlation_id
            )
        if attributes is not None:
            safe_attributes.update(attributes)

        kwargs: dict[str, Any] = {
            "name": name,
            "kind": SpanKind.AGENT,
            "sink": self._telemetry_sink,
            "incident_id": task.incident_id,
            "agent_name": task.target_agent,
            "attributes": safe_attributes,
        }
        if self._otel_runtime is not None:
            return OpenTelemetryObservation(runtime=self._otel_runtime, **kwargs)
        return Observation(**kwargs)

    @staticmethod
    def _record_result(
        observation: AgentObservation | None,
        result: AgentResult,
    ) -> None:
        if observation is None:
            return
        observation.set_attribute("aegisops.agent.result_status", result.status.value)
        if result.error_code is not None:
            observation.set_attribute("aegisops.agent.error_code", result.error_code)
        observation.set_attribute(
            "aegisops.agent.proposed_action_count", len(result.proposed_action_ids)
        )
        observation.set_status(
            SpanStatus.ERROR if result.status is AgentRunStatus.FAILED else SpanStatus.OK
        )

    @staticmethod
    def _failed_result(
        task: AgentTask,
        *,
        code: str,
        summary: str,
    ) -> AgentResult:
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
        agent = self._supervisor.get_agent(task.target_agent)
        agent.validate_task(task)
        self._get_handler(task.target_agent)

    async def _invoke(self, task: AgentTask) -> AgentResult:
        """Invoke a handler and verify its result against current permissions."""
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
                summary="The specialist agent returned an invalid result type.",
            )

        try:
            current_agent = self._supervisor.get_agent(task.target_agent)
            current_agent.validate_task(task)
            current_agent.validate_result(task, result)
            if result.status is AgentRunStatus.HANDOFF:
                if result.handoff_to == task.target_agent:
                    raise AgentContractViolation("An agent cannot hand off to itself.")
                if result.handoff_to is None:
                    raise AgentContractViolation("A handoff must specify another agent.")
                self._supervisor.get_agent(result.handoff_to)
        except (AgentContractViolation, KeyError, ValueError):
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
        """Execute one agent with an inclusive queue-and-execution timeout.

        Cancellation is cooperative. Timeout never guarantees that an
        external side effect was rolled back; use operation reconciliation.
        """
        with self._observation(task, name="aegisops.agent.dispatch") as dispatch_span:
            queued_at = perf_counter()
            acquired = False
            try:
                self._validate_assignment(task)
                async with asyncio.timeout(task.deadline_seconds):
                    async with self._semaphore:
                        acquired = True
                        if dispatch_span is not None:
                            dispatch_span.set_attribute(
                                "aegisops.agent.queue_wait_ms",
                                round((perf_counter() - queued_at) * 1000, 3),
                            )
                        # Permission and handler registration can change in the queue.
                        self._validate_assignment(task)
                        with self._observation(
                            task, name="aegisops.agent.invoke"
                        ) as execution_span:
                            started = perf_counter()
                            try:
                                result = await self._invoke(task)
                                self._record_result(execution_span, result)
                            finally:
                                if execution_span is not None:
                                    execution_span.set_attribute(
                                        "aegisops.agent.execution_ms",
                                        round((perf_counter() - started) * 1000, 3),
                                    )
                        self._record_result(dispatch_span, result)
                        return result
            except TimeoutError:
                if dispatch_span is not None:
                    dispatch_span.set_attribute(
                        "aegisops.agent.timed_out_in_queue", not acquired
                    )
                    if not acquired:
                        dispatch_span.set_attribute(
                            "aegisops.agent.queue_wait_ms",
                            round((perf_counter() - queued_at) * 1000, 3),
                        )
                failure = self._failed_result(
                    task,
                    code="AGENT_TIMEOUT",
                    summary=(
                        "The specialist agent exceeded its task deadline. "
                        "Any uncertain external side effects require "
                        "separate reconciliation."
                    ),
                )
                self._record_result(dispatch_span, failure)
                return failure
            except asyncio.CancelledError:
                if dispatch_span is not None:
                    dispatch_span.set_attribute(
                        "aegisops.agent.result_status", "cancelled"
                    )
                    dispatch_span.set_status(SpanStatus.ERROR)
                raise
            except Exception:
                if dispatch_span is not None:
                    dispatch_span.set_attribute(
                        "aegisops.agent.result_status", "rejected"
                    )
                    dispatch_span.set_status(SpanStatus.ERROR)
                raise

    async def execute_route(
        self,
        route: SupervisorRoute,
        *,
        incident_id: str,
        context: Mapping[str, Any] | None = None,
        correlation_id: UUID | None = None,
    ) -> AgentResult:
        """Build a validated task from a supervisor route, then execute it."""
        task = self._supervisor.dispatch(
            route,
            incident_id=incident_id,
            context=dict(context or {}),
            correlation_id=correlation_id,
        )
        return await self.execute(task)
