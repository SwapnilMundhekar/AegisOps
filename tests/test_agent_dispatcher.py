
"""Contract and concurrency tests for the asynchronous agent dispatcher."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from aegisops.agent_contracts import (
    AgentCapability,
    AgentContractViolation,
    AgentResult,
    AgentRole,
    AgentRunStatus,
    AgentSpec,
    AgentTask,
)
from aegisops.agent_dispatcher import AgentDispatchError, AgentDispatcher
from aegisops.domain import ActionType
from aegisops.supervisor import Supervisor


def make_agent(
    name: str = "investigator",
    *,
    capabilities: frozenset[AgentCapability] | None = None,
    permitted_actions: frozenset[ActionType] | None = None,
    timeout_seconds: float = 5.0,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        display_name=name.title(),
        role=AgentRole.INVESTIGATOR,
        description="Specialist agent for dispatcher tests.",
        capabilities=capabilities
        if capabilities is not None
        else frozenset({AgentCapability.TELEMETRY_READ}),
        permitted_action_types=permitted_actions or frozenset(),
        max_steps=10,
        timeout_seconds=timeout_seconds,
    )


def make_task(
    name: str = "investigator",
    *,
    deadline_seconds: float = 2.0,
    required_capabilities: frozenset[AgentCapability] | None = None,
    requested_action_types: frozenset[ActionType] | None = None,
) -> AgentTask:
    return AgentTask(
        incident_id="INC-1001",
        target_agent=name,
        objective="Investigate service health.",
        required_capabilities=required_capabilities
        if required_capabilities is not None
        else frozenset({AgentCapability.TELEMETRY_READ}),
        requested_action_types=requested_action_types or frozenset(),
        deadline_seconds=deadline_seconds,
        max_steps=2,
    )


def complete(
    task: AgentTask,
    *,
    agent_name: str | None = None,
    proposed_action_ids: tuple[str, ...] = (),
) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        agent_name=agent_name or task.target_agent,
        status=AgentRunStatus.COMPLETED,
        summary="Investigation completed.",
        proposed_action_ids=proposed_action_ids,
    )


def make_dispatcher(
    *,
    max_concurrent_tasks: int = 4,
) -> AgentDispatcher:
    supervisor = Supervisor([make_agent()])
    return AgentDispatcher(
        supervisor,
        max_concurrent_tasks=max_concurrent_tasks,
    )


@pytest.mark.parametrize("limit", [0, -1, 129])
def test_invalid_concurrency_limit_is_rejected(limit: int) -> None:
    with pytest.raises(ValueError):
        AgentDispatcher(
            Supervisor([make_agent()]),
            max_concurrent_tasks=limit,
        )


@pytest.mark.parametrize("limit", [True, False, 1.5, "4"])
def test_non_integer_concurrency_limit_is_rejected(
    limit: object,
) -> None:
    with pytest.raises(TypeError):
        AgentDispatcher(
            Supervisor([make_agent()]),
            max_concurrent_tasks=limit,  # type: ignore[arg-type]
        )


def test_dispatcher_reports_configured_limit() -> None:
    dispatcher = make_dispatcher(max_concurrent_tasks=2)

    assert dispatcher.max_concurrent_tasks == 2
    assert dispatcher.registered_handlers == ()


def test_rejects_handler_for_unknown_agent() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task)

    with pytest.raises(KeyError, match="not registered"):
        dispatcher.register_handler("unknown_agent", handler)


def test_rejects_synchronous_handler() -> None:
    dispatcher = make_dispatcher()

    def handler(task: AgentTask) -> AgentResult:
        return complete(task)

    with pytest.raises(TypeError, match="asynchronous"):
        dispatcher.register_handler("investigator", handler)


def test_rejects_duplicate_handler_registration() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task)

    dispatcher.register_handler("investigator", handler)

    with pytest.raises(AgentDispatchError, match="already registered"):
        dispatcher.register_handler("investigator", handler)


@pytest.mark.asyncio
async def test_replacement_handler_is_used() -> None:
    dispatcher = make_dispatcher()

    async def old_handler(task: AgentTask) -> AgentResult:
        return complete(task).model_copy(update={"summary": "Old"})

    async def new_handler(task: AgentTask) -> AgentResult:
        return complete(task).model_copy(update={"summary": "New"})

    dispatcher.register_handler("investigator", old_handler)
    dispatcher.register_handler(
        "investigator",
        new_handler,
        replace=True,
    )

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.COMPLETED
    assert result.summary == "New"


def test_unregister_handler() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task)

    dispatcher.register_handler("investigator", handler)
    assert dispatcher.registered_handlers == ("investigator",)

    dispatcher.unregister_handler("investigator")
    assert dispatcher.registered_handlers == ()

    with pytest.raises(AgentDispatchError, match="No handler"):
        dispatcher.unregister_handler("investigator")


@pytest.mark.asyncio
async def test_executes_registered_async_handler() -> None:
    dispatcher = make_dispatcher()
    observed: list[AgentTask] = []

    async def handler(task: AgentTask) -> AgentResult:
        observed.append(task)
        return complete(task)

    dispatcher.register_handler("investigator", handler)
    task = make_task()

    result = await dispatcher.execute(task)

    assert result.status is AgentRunStatus.COMPLETED
    assert result.task_id == task.task_id
    assert result.agent_name == "investigator"
    assert observed == [task]


@pytest.mark.asyncio
async def test_execute_route_preserves_task_identity_and_context() -> None:
    supervisor = Supervisor([make_agent()])
    dispatcher = AgentDispatcher(supervisor)
    received: list[AgentTask] = []

    async def handler(task: AgentTask) -> AgentResult:
        received.append(task)
        return complete(task)

    dispatcher.register_handler("investigator", handler)

    route = supervisor.route(
        incident_id="INC-2002",
        objective="Inspect API telemetry.",
        required_capabilities={AgentCapability.TELEMETRY_READ},
        deadline_seconds=2.0,
        max_steps=2,
    )

    correlation_id = uuid4()

    result = await dispatcher.execute_route(
        route,
        incident_id="INC-2002",
        context={"service": "payments-api"},
        correlation_id=correlation_id,
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.task_id == route.task_id
    assert received[0].incident_id == "INC-2002"
    assert received[0].context == {"service": "payments-api"}
    assert received[0].correlation_id == correlation_id


@pytest.mark.asyncio
async def test_unregistered_handler_cannot_execute() -> None:
    dispatcher = make_dispatcher()

    with pytest.raises(AgentDispatchError, match="No handler"):
        await dispatcher.execute(make_task())


@pytest.mark.asyncio
async def test_task_missing_agent_capability_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task)

    dispatcher.register_handler("investigator", handler)

    unauthorized_task = make_task(
        required_capabilities=frozenset(
            {AgentCapability.LOG_ANALYSIS}
        )
    )

    with pytest.raises(AgentContractViolation):
        await dispatcher.execute(unauthorized_task)


@pytest.mark.asyncio
async def test_handler_exception_becomes_structured_failure() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        raise RuntimeError("Sensitive internal exception detail")

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_EXECUTION_ERROR"
    assert result.proposed_action_ids == ()
    assert "Sensitive internal exception detail" not in result.summary


@pytest.mark.asyncio
async def test_invalid_result_type_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return "not an AgentResult"  # type: ignore[return-value]

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "INVALID_AGENT_RESULT"


@pytest.mark.asyncio
async def test_wrong_task_id_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=uuid4(),
            agent_name=task.target_agent,
            status=AgentRunStatus.COMPLETED,
            summary="Wrong task correlation.",
        )

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_CONTRACT_ERROR"


@pytest.mark.asyncio
async def test_wrong_agent_identity_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task, agent_name="different_agent")

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_CONTRACT_ERROR"


@pytest.mark.asyncio
async def test_unauthorized_action_proposal_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task, proposed_action_ids=("ACTION-123",))

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_CONTRACT_ERROR"
    assert result.proposed_action_ids == ()


@pytest.mark.asyncio
async def test_authorized_action_proposal_is_returned_not_executed() -> None:
    capabilities = frozenset(
        {
            AgentCapability.TELEMETRY_READ,
            AgentCapability.PROPOSE_ACTION,
        }
    )
    supervisor = Supervisor(
        [
            make_agent(
                capabilities=capabilities,
                permitted_actions=frozenset({ActionType.READ}),
            )
        ]
    )
    dispatcher = AgentDispatcher(supervisor)

    async def handler(task: AgentTask) -> AgentResult:
        return complete(task, proposed_action_ids=("ACTION-READ-1",))

    dispatcher.register_handler("investigator", handler)

    task = make_task(
        required_capabilities=capabilities,
        requested_action_types=frozenset({ActionType.READ}),
    )

    result = await dispatcher.execute(task)

    assert result.status is AgentRunStatus.COMPLETED
    assert result.proposed_action_ids == ("ACTION-READ-1",)


@pytest.mark.asyncio
async def test_unknown_handoff_target_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_name=task.target_agent,
            status=AgentRunStatus.HANDOFF,
            summary="Handoff requested.",
            handoff_to="missing_agent",
        )

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_CONTRACT_ERROR"


@pytest.mark.asyncio
async def test_self_handoff_is_rejected() -> None:
    dispatcher = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_name=task.target_agent,
            status=AgentRunStatus.HANDOFF,
            summary="Attempted self-handoff.",
            handoff_to=task.target_agent,
        )

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_CONTRACT_ERROR"


@pytest.mark.asyncio
async def test_valid_handoff_does_not_execute_target_agent() -> None:
    supervisor = Supervisor(
        [
            make_agent(),
            make_agent("diagnostician"),
        ]
    )
    dispatcher = AgentDispatcher(supervisor)
    target_called = False

    async def investigator(task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_name=task.target_agent,
            status=AgentRunStatus.HANDOFF,
            summary="A diagnostic review is recommended.",
            handoff_to="diagnostician",
        )

    async def diagnostician(task: AgentTask) -> AgentResult:
        nonlocal target_called
        target_called = True
        return complete(task)

    dispatcher.register_handler("investigator", investigator)
    dispatcher.register_handler("diagnostician", diagnostician)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.HANDOFF
    assert result.handoff_to == "diagnostician"
    assert target_called is False


@pytest.mark.asyncio
async def test_handler_timeout_produces_structured_failure() -> None:
    dispatcher = make_dispatcher()
    cancelled = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return complete(task)

    dispatcher.register_handler("investigator", handler)

    result = await dispatcher.execute(
        make_task(deadline_seconds=0.02)
    )

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_TIMEOUT"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_caller_cancellation_propagates() -> None:
    dispatcher = make_dispatcher()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return complete(task)

    dispatcher.register_handler("investigator", handler)
    running = asyncio.create_task(dispatcher.execute(make_task()))

    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        running.cancel()

        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_concurrency_limit_prevents_second_handler_start() -> None:
    dispatcher = make_dispatcher(max_concurrent_tasks=1)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    active = 0
    max_active = 0

    async def handler(task: AgentTask) -> AgentResult:
        nonlocal calls, active, max_active
        calls += 1
        active += 1
        max_active = max(max_active, active)
        started.set()

        try:
            await release.wait()
            return complete(task)
        finally:
            active -= 1

    dispatcher.register_handler("investigator", handler)

    first = asyncio.create_task(dispatcher.execute(make_task()))
    second = None

    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(dispatcher.execute(make_task()))

        # Allow the second task to attempt semaphore acquisition.
        await asyncio.sleep(0.01)
        assert calls == 1

        release.set()
        first_result, second_result = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=2,
        )

        assert first_result.status is AgentRunStatus.COMPLETED
        assert second_result.status is AgentRunStatus.COMPLETED
        assert calls == 2
        assert max_active == 1
    finally:
        release.set()
        for running in (first, second):
            if running is not None and not running.done():
                running.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_queue_wait_is_included_in_timeout() -> None:
    dispatcher = make_dispatcher(max_concurrent_tasks=1)
    first_started = asyncio.Event()
    release = asyncio.Event()
    invoked: list[str] = []

    async def handler(task: AgentTask) -> AgentResult:
        invoked.append(task.objective)
        first_started.set()
        await release.wait()
        return complete(task)

    dispatcher.register_handler("investigator", handler)

    first_task = make_task()
    second_task = make_task(deadline_seconds=0.02).model_copy(
        update={"objective": "Queued operation"}
    )
    first = asyncio.create_task(dispatcher.execute(first_task))

    try:
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second_result = await dispatcher.execute(second_task)

        assert second_result.status is AgentRunStatus.FAILED
        assert second_result.error_code == "AGENT_TIMEOUT"
        assert invoked == ["Investigate service health."]
    finally:
        release.set()
        await asyncio.wait_for(first, timeout=2)


@pytest.mark.asyncio
async def test_permission_revocation_during_execution_is_detected() -> None:
    supervisor = Supervisor([make_agent()])
    dispatcher = AgentDispatcher(supervisor)
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        started.set()
        await release.wait()
        return complete(task)

    dispatcher.register_handler("investigator", handler)
    running = asyncio.create_task(dispatcher.execute(make_task()))

    try:
        await asyncio.wait_for(started.wait(), timeout=1)

        supervisor.register(
            make_agent(capabilities=frozenset()),
            replace=True,
        )
        release.set()

        result = await asyncio.wait_for(running, timeout=2)
        assert result.status is AgentRunStatus.FAILED
        assert result.error_code == "AGENT_CONTRACT_ERROR"
    finally:
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
