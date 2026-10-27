
from __future__ import annotations

import pytest

from aegisops.agent_contracts import (
    AgentCapability,
    AgentPriority,
    AgentRole,
    AgentSpec,
)
from aegisops.supervisor import (
    NoEligibleAgentError,
    Supervisor,
)


def make_agent(
    name: str,
    capabilities: set[AgentCapability],
    *,
    max_steps: int = 10,
    timeout_seconds: float = 60,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        display_name=name.replace("_", " ").title(),
        role=AgentRole.INVESTIGATOR,
        description=f"Test specialist agent named {name}.",
        capabilities=frozenset(capabilities),
        permitted_action_types=frozenset(),
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )


def test_registered_agents_are_returned_in_name_order() -> None:
    supervisor = Supervisor(
        [
            make_agent("zeta_agent", set()),
            make_agent("alpha_agent", set()),
            make_agent("middle_agent", set()),
        ]
    )

    assert [agent.name for agent in supervisor.agents] == [
        "alpha_agent",
        "middle_agent",
        "zeta_agent",
    ]


def test_duplicate_registration_is_rejected() -> None:
    agent = make_agent("investigator", set())
    supervisor = Supervisor([agent])

    with pytest.raises(ValueError, match="already registered"):
        supervisor.register(agent)


def test_registration_can_replace_an_existing_agent() -> None:
    original = make_agent(
        "investigator",
        {AgentCapability.TELEMETRY_READ},
    )
    replacement = make_agent(
        "investigator",
        {AgentCapability.LOG_ANALYSIS},
    )
    supervisor = Supervisor([original])

    supervisor.register(replacement, replace=True)

    assert supervisor.get_agent(
        "investigator"
    ).capabilities == frozenset(
        {AgentCapability.LOG_ANALYSIS}
    )


def test_route_selects_agent_with_required_capability() -> None:
    telemetry_agent = make_agent(
        "telemetry_agent",
        {AgentCapability.TELEMETRY_READ},
    )
    log_agent = make_agent(
        "log_agent",
        {AgentCapability.LOG_ANALYSIS},
    )
    supervisor = Supervisor([log_agent, telemetry_agent])

    route = supervisor.route(
        incident_id="INC-001",
        objective="Inspect current telemetry",
        required_capabilities={
            AgentCapability.TELEMETRY_READ
        },
    )

    assert route.agent_name == "telemetry_agent"
    assert (
        AgentCapability.TELEMETRY_READ
        in route.required_capabilities
    )


def test_route_prefers_agent_with_fewer_extra_capabilities() -> None:
    focused_agent = make_agent(
        "focused_agent",
        {AgentCapability.TELEMETRY_READ},
    )
    broad_agent = make_agent(
        "broad_agent",
        {
            AgentCapability.TELEMETRY_READ,
            AgentCapability.LOG_ANALYSIS,
            AgentCapability.INCIDENT_TRIAGE,
        },
    )
    supervisor = Supervisor([broad_agent, focused_agent])

    route = supervisor.route(
        incident_id="INC-002",
        objective="Read telemetry",
        required_capabilities={
            AgentCapability.TELEMETRY_READ
        },
    )

    assert route.agent_name == "focused_agent"


def test_eligible_preferred_agent_takes_priority() -> None:
    focused_agent = make_agent(
        "focused_agent",
        {AgentCapability.TELEMETRY_READ},
    )
    broad_agent = make_agent(
        "broad_agent",
        {
            AgentCapability.TELEMETRY_READ,
            AgentCapability.LOG_ANALYSIS,
        },
    )
    supervisor = Supervisor([focused_agent, broad_agent])

    route = supervisor.route(
        incident_id="INC-003",
        objective="Inspect telemetry",
        required_capabilities={
            AgentCapability.TELEMETRY_READ
        },
        preferred_agent="broad_agent",
    )

    assert route.agent_name == "broad_agent"
    assert "preferred agent" in route.rationale.lower()


def test_route_fails_when_no_agent_has_required_capability() -> None:
    supervisor = Supervisor(
        [
            make_agent(
                "log_agent",
                {AgentCapability.LOG_ANALYSIS},
            )
        ]
    )

    with pytest.raises(
        NoEligibleAgentError,
        match="No registered agent satisfies",
    ):
        supervisor.route(
            incident_id="INC-004",
            objective="Inspect telemetry",
            required_capabilities={
                AgentCapability.TELEMETRY_READ
            },
        )


def test_route_fails_when_no_agents_are_registered() -> None:
    supervisor = Supervisor()

    with pytest.raises(
        NoEligibleAgentError,
        match="no agents are registered",
    ):
        supervisor.route(
            incident_id="INC-005",
            objective="Investigate an incident",
        )


def test_route_rejects_task_exceeding_agent_step_limit() -> None:
    supervisor = Supervisor(
        [
            make_agent(
                "limited_agent",
                {AgentCapability.INCIDENT_TRIAGE},
                max_steps=3,
            )
        ]
    )

    with pytest.raises(
        NoEligibleAgentError,
        match="No registered agent satisfies",
    ):
        supervisor.route(
            incident_id="INC-006",
            objective="Investigate the incident",
            required_capabilities={
                AgentCapability.INCIDENT_TRIAGE
            },
            max_steps=5,
        )


def test_route_rejects_task_exceeding_agent_timeout() -> None:
    supervisor = Supervisor(
        [
            make_agent(
                "short_timeout_agent",
                {AgentCapability.INCIDENT_TRIAGE},
                timeout_seconds=15,
            )
        ]
    )

    with pytest.raises(
        NoEligibleAgentError,
        match="No registered agent satisfies",
    ):
        supervisor.route(
            incident_id="INC-007",
            objective="Investigate the incident",
            required_capabilities={
                AgentCapability.INCIDENT_TRIAGE
            },
            deadline_seconds=30,
        )


def test_dispatch_creates_a_validated_agent_task() -> None:
    supervisor = Supervisor(
        [
            make_agent(
                "triage_agent",
                {AgentCapability.INCIDENT_TRIAGE},
            )
        ]
    )

    route = supervisor.route(
        incident_id="INC-008",
        objective="Classify the incident",
        required_capabilities={
            AgentCapability.INCIDENT_TRIAGE
        },
        priority=AgentPriority.HIGH,
    )

    task = supervisor.dispatch(
        route,
        incident_id="INC-008",
        context={"service": "api"},
    )

    assert task.task_id == route.task_id
    assert task.target_agent == "triage_agent"
    assert task.incident_id == "INC-008"
    assert task.objective == "Classify the incident"
    assert task.context == {"service": "api"}
    assert task.priority == AgentPriority.HIGH


def test_unregister_removes_and_returns_agent() -> None:
    agent = make_agent("temporary_agent", set())
    supervisor = Supervisor([agent])

    removed = supervisor.unregister("temporary_agent")

    assert removed == agent
    assert supervisor.agents == ()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        supervisor.get_agent("temporary_agent")


def test_dispatch_rejects_unregistered_route_target() -> None:
    supervisor = Supervisor(
        [
            make_agent(
                "triage_agent",
                {AgentCapability.INCIDENT_TRIAGE},
            )
        ]
    )

    route = supervisor.route(
        incident_id="INC-009",
        objective="Classify the incident",
        required_capabilities={
            AgentCapability.INCIDENT_TRIAGE
        },
    )

    supervisor.unregister("triage_agent")

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        supervisor.dispatch(
            route,
            incident_id="INC-009",
        )
