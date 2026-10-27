
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID, uuid4

from aegisops.agent_contracts import (
    AgentCapability,
    AgentContractViolation,
    AgentPriority,
    AgentSpec,
    AgentTask,
    SupervisorRoute,
)
from aegisops.domain import ActionType


class NoEligibleAgentError(RuntimeError):
    """Raised when no registered agent can safely accept a task."""


class Supervisor:
    """Select and validate specialist agents using explicit contracts."""

    def __init__(self, agents: Iterable[AgentSpec] = ()) -> None:
        self._agents: dict[str, AgentSpec] = {}

        for agent in agents:
            self.register(agent)

    @property
    def agents(self) -> tuple[AgentSpec, ...]:
        """Return registered agents in deterministic name order."""
        return tuple(
            self._agents[name]
            for name in sorted(self._agents)
        )

    def register(
        self,
        agent: AgentSpec,
        *,
        replace: bool = False,
    ) -> None:
        """Register an agent, rejecting accidental duplicate names."""
        existing = self._agents.get(agent.name)

        if existing is not None and not replace:
            raise ValueError(
                f"Agent '{agent.name}' is already registered."
            )

        self._agents[agent.name] = agent

    def unregister(self, agent_name: str) -> AgentSpec:
        """Remove and return a registered agent."""
        try:
            return self._agents.pop(agent_name)
        except KeyError as exc:
            raise KeyError(
                f"Agent '{agent_name}' is not registered."
            ) from exc

    def get_agent(self, agent_name: str) -> AgentSpec:
        """Return one registered agent by its unique name."""
        try:
            return self._agents[agent_name]
        except KeyError as exc:
            raise KeyError(
                f"Agent '{agent_name}' is not registered."
            ) from exc

    def route(
        self,
        *,
        incident_id: str,
        objective: str,
        required_capabilities: Iterable[AgentCapability] = (),
        requested_action_types: Iterable[ActionType] = (),
        priority: AgentPriority = AgentPriority.NORMAL,
        deadline_seconds: float = 60,
        max_steps: int = 10,
        context: Mapping[str, Any] | None = None,
        correlation_id: UUID | None = None,
        preferred_agent: str | None = None,
    ) -> SupervisorRoute:
        """Choose the best eligible agent and return a typed route.

        Eligibility is determined by the agent contract. Among eligible
        agents, selection prefers the requested agent when specified,
        then the agent with the fewest unnecessary capabilities and
        permitted action types, followed by alphabetical agent name.
        """
        if not self._agents:
            raise NoEligibleAgentError(
                "Cannot route the task because no agents are registered."
            )

        task_id = uuid4()
        required = frozenset(required_capabilities)
        requested_actions = frozenset(requested_action_types)
        task_context = dict(context or {})

        eligible: list[AgentSpec] = []
        rejection_reasons: dict[str, str] = {}

        for agent in self.agents:
            try:
                candidate_task = AgentTask(
                    task_id=task_id,
                    incident_id=incident_id,
                    target_agent=agent.name,
                    objective=objective,
                    context=task_context,
                    required_capabilities=required,
                    requested_action_types=requested_actions,
                    priority=priority,
                    deadline_seconds=deadline_seconds,
                    max_steps=max_steps,
                    correlation_id=correlation_id,
                )
                agent.validate_task(candidate_task)
            except (AgentContractViolation, ValueError) as exc:
                rejection_reasons[agent.name] = str(exc)
                continue

            eligible.append(agent)

        if not eligible:
            details = "; ".join(
                f"{name}: {reason}"
                for name, reason in sorted(rejection_reasons.items())
            )
            raise NoEligibleAgentError(
                "No registered agent satisfies the task contract. "
                f"Rejections: {details or 'no eligible candidates'}"
            )

        def ranking(agent: AgentSpec) -> tuple[int, int, int, str]:
            preferred_rank = (
                0 if agent.name == preferred_agent else 1
            )
            extra_capabilities = len(
                agent.capabilities - required
            )
            extra_action_types = len(
                agent.permitted_action_types - requested_actions
            )

            return (
                preferred_rank,
                extra_capabilities,
                extra_action_types,
                agent.name,
            )

        selected = min(eligible, key=ranking)

        rationale_parts = [
            f"Agent '{selected.name}' satisfies the task contract."
        ]

        if required:
            capability_names = ", ".join(
                sorted(capability.value for capability in required)
            )
            rationale_parts.append(
                f"Required capabilities: {capability_names}."
            )

        if requested_actions:
            action_names = ", ".join(
                sorted(action.value for action in requested_actions)
            )
            rationale_parts.append(
                f"Requested action types: {action_names}."
            )

        if selected.name == preferred_agent:
            rationale_parts.append(
                "The preferred agent was eligible and selected."
            )

        return SupervisorRoute(
            task_id=task_id,
            agent_name=selected.name,
            objective=objective,
            rationale=" ".join(rationale_parts),
            required_capabilities=required,
            requested_action_types=requested_actions,
            priority=priority,
            deadline_seconds=deadline_seconds,
            max_steps=max_steps,
        )

    def dispatch(
        self,
        route: SupervisorRoute,
        *,
        incident_id: str,
        context: Mapping[str, Any] | None = None,
        correlation_id: UUID | None = None,
    ) -> AgentTask:
        """Convert a route into a task and revalidate its agent contract.

        This method creates a task for downstream execution. It does not
        execute tools, perform remediation, or bypass action approval.
        """
        agent = self.get_agent(route.agent_name)

        task = route.to_task(
            incident_id=incident_id,
            context=dict(context or {}),
            correlation_id=correlation_id,
        )

        try:
            agent.validate_task(task)
        except (AgentContractViolation, ValueError) as exc:
            raise NoEligibleAgentError(
                f"Route for agent '{agent.name}' failed validation: {exc}"
            ) from exc

        return task
