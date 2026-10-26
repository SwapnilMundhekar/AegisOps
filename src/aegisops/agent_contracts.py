
"""Typed contracts for governed AegisOps agent orchestration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegisops.domain import ActionType


_AGENT_NAME_PATTERN = r"^[a-z][a-z0-9._-]{0,63}$"
_FINDING_CODE_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"
_ERROR_CODE_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"


class AgentContractViolation(ValueError):
    """Raised when an agent task or result violates its declared contract."""


class AgentRole(StrEnum):
    """Functional role assigned to an agent."""

    SUPERVISOR = "supervisor"
    INVESTIGATOR = "investigator"
    DIAGNOSTICIAN = "diagnostician"
    PLANNER = "planner"
    REMEDIATOR = "remediator"
    VERIFIER = "verifier"
    COMMUNICATOR = "communicator"


class AgentCapability(StrEnum):
    """Capabilities that can be granted to an agent explicitly."""

    INCIDENT_TRIAGE = "incident_triage"
    TELEMETRY_READ = "telemetry_read"
    LOG_ANALYSIS = "log_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    REMEDIATION_PLANNING = "remediation_planning"
    PROPOSE_ACTION = "propose_action"
    VERIFY_REMEDIATION = "verify_remediation"
    INCIDENT_COMMUNICATION = "incident_communication"


class AgentRunStatus(StrEnum):
    """Terminal or handoff state reported by an agent."""

    COMPLETED = "completed"
    NEEDS_HUMAN = "needs_human"
    HANDOFF = "handoff"
    FAILED = "failed"


class AgentPriority(StrEnum):
    """Scheduling priority for agent work."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentContract(BaseModel):
    """Shared validation rules for all orchestration contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class EvidenceItem(AgentContract):
    """A traceable observation supporting an agent finding."""

    source: str = Field(min_length=1, max_length=128)
    observation: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    observed_at: str | None = Field(default=None, max_length=64)
    reference: str | None = Field(default=None, max_length=2048)
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvidenceItem":
        if self.reference is not None and not self.reference.strip():
            raise ValueError("reference must not be blank when supplied.")

        if any(not key.strip() for key in self.attributes):
            raise ValueError("Evidence attribute keys must not be blank.")

        return self


class AgentSpec(AgentContract):
    """Application-owned definition of an agent's permissions and limits."""

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=_AGENT_NAME_PATTERN,
    )
    display_name: str = Field(min_length=1, max_length=128)
    role: AgentRole
    description: str = Field(min_length=1, max_length=2000)

    capabilities: frozenset[AgentCapability] = Field(
        default_factory=frozenset
    )

    permitted_action_types: frozenset[ActionType] = Field(
        default_factory=frozenset
    )

    max_steps: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    @model_validator(mode="after")
    def validate_permissions(self) -> "AgentSpec":
        can_propose = (
            AgentCapability.PROPOSE_ACTION in self.capabilities
        )

        if can_propose and not self.permitted_action_types:
            raise ValueError(
                "Agents with PROPOSE_ACTION capability must declare "
                "at least one permitted action type."
            )

        if self.permitted_action_types and not can_propose:
            raise ValueError(
                "permitted_action_types requires PROPOSE_ACTION capability."
            )

        return self

    def validate_task(self, task: "AgentTask") -> None:
        """Reject tasks exceeding this agent's declared permissions."""

        if task.target_agent != self.name:
            raise AgentContractViolation(
                f"Task targets {task.target_agent!r}, not {self.name!r}."
            )

        missing_capabilities = (
            task.required_capabilities - self.capabilities
        )

        if missing_capabilities:
            names = ", ".join(
                sorted(capability.value for capability in missing_capabilities)
            )
            raise AgentContractViolation(
                f"Agent {self.name!r} lacks required capabilities: {names}."
            )

        unauthorized_actions = (
            task.requested_action_types - self.permitted_action_types
        )

        if unauthorized_actions:
            names = ", ".join(
                sorted(action.value for action in unauthorized_actions)
            )
            raise AgentContractViolation(
                f"Agent {self.name!r} cannot propose these action types: "
                f"{names}."
            )

        if task.max_steps > self.max_steps:
            raise AgentContractViolation(
                f"Task allows {task.max_steps} steps, but agent "
                f"{self.name!r} is limited to {self.max_steps}."
            )

        if task.deadline_seconds > self.timeout_seconds:
            raise AgentContractViolation(
                f"Task timeout {task.deadline_seconds}s exceeds "
                f"agent {self.name!r} timeout {self.timeout_seconds}s."
            )

    def validate_result(
        self,
        task: "AgentTask",
        result: "AgentResult",
    ) -> None:
        """Validate agent identity, task correlation and action proposals."""

        if result.task_id != task.task_id:
            raise AgentContractViolation(
                "Agent result task_id does not match the assigned task."
            )

        if result.agent_name != self.name:
            raise AgentContractViolation(
                f"Result claims agent {result.agent_name!r}; "
                f"expected {self.name!r}."
            )

        if result.proposed_action_ids:
            if AgentCapability.PROPOSE_ACTION not in self.capabilities:
                raise AgentContractViolation(
                    f"Agent {self.name!r} is not permitted to propose actions."
                )

            if not task.requested_action_types:
                raise AgentContractViolation(
                    "Agent returned proposed actions for a task that did "
                    "not authorize action proposals."
                )


class AgentTask(AgentContract):
    """Validated unit of work dispatched by the supervisor."""

    task_id: UUID = Field(default_factory=uuid4)
    incident_id: str = Field(min_length=1, max_length=128)

    target_agent: str = Field(
        min_length=1,
        max_length=64,
        pattern=_AGENT_NAME_PATTERN,
    )

    objective: str = Field(min_length=1, max_length=4000)

    context: dict[str, Any] = Field(default_factory=dict)

    required_capabilities: frozenset[AgentCapability] = Field(
        default_factory=frozenset
    )

    requested_action_types: frozenset[ActionType] = Field(
        default_factory=frozenset
    )

    priority: AgentPriority = AgentPriority.NORMAL
    deadline_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    max_steps: int = Field(default=10, ge=1, le=100)

    correlation_id: UUID | None = None

    @model_validator(mode="after")
    def validate_task_content(self) -> "AgentTask":
        if any(not key.strip() for key in self.context):
            raise ValueError("Context keys must not be blank.")

        if self.requested_action_types and (
            AgentCapability.PROPOSE_ACTION not in self.required_capabilities
        ):
            raise ValueError(
                "Tasks requesting action proposals must explicitly require "
                "the PROPOSE_ACTION capability."
            )

        return self


class AgentFinding(AgentContract):
    """A structured finding returned by a specialist agent."""

    code: str = Field(
        min_length=2,
        max_length=64,
        pattern=_FINDING_CODE_PATTERN,
    )

    summary: str = Field(min_length=1, max_length=2000)

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)

    recommended_next_step: str | None = Field(
        default=None,
        max_length=2000,
    )


class AgentResult(AgentContract):
    """Structured result returned from one agent invocation."""

    task_id: UUID
    agent_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=_AGENT_NAME_PATTERN,
    )

    status: AgentRunStatus

    summary: str = Field(min_length=1, max_length=4000)

    findings: tuple[AgentFinding, ...] = Field(default_factory=tuple)

    proposed_action_ids: tuple[str, ...] = Field(default_factory=tuple)

    handoff_to: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=_AGENT_NAME_PATTERN,
    )

    error_code: str | None = Field(
        default=None,
        min_length=2,
        max_length=64,
        pattern=_ERROR_CODE_PATTERN,
    )

    @model_validator(mode="after")
    def validate_result_state(self) -> "AgentResult":
        if self.status is AgentRunStatus.HANDOFF and not self.handoff_to:
            raise ValueError(
                "HANDOFF results must specify handoff_to."
            )

        if self.status is not AgentRunStatus.HANDOFF and self.handoff_to:
            raise ValueError(
                "handoff_to is only valid for HANDOFF results."
            )

        if self.status is AgentRunStatus.FAILED and not self.error_code:
            raise ValueError(
                "FAILED results must specify error_code."
            )

        if self.status is not AgentRunStatus.FAILED and self.error_code:
            raise ValueError(
                "error_code is only valid for FAILED results."
            )

        if (
            self.status is AgentRunStatus.FAILED
            and self.proposed_action_ids
        ):
            raise ValueError(
                "FAILED results must not contain proposed action IDs."
            )

        if any(not action_id.strip() for action_id in self.proposed_action_ids):
            raise ValueError("Proposed action IDs must not be blank.")

        if len(set(self.proposed_action_ids)) != len(
            self.proposed_action_ids
        ):
            raise ValueError("Proposed action IDs must be unique.")

        return self


class SupervisorRoute(AgentContract):
    """A supervisor's typed routing decision before task dispatch."""

    task_id: UUID = Field(default_factory=uuid4)

    agent_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=_AGENT_NAME_PATTERN,
    )

    objective: str = Field(min_length=1, max_length=4000)

    rationale: str = Field(min_length=1, max_length=2000)

    required_capabilities: frozenset[AgentCapability] = Field(
        default_factory=frozenset
    )

    requested_action_types: frozenset[ActionType] = Field(
        default_factory=frozenset
    )

    priority: AgentPriority = AgentPriority.NORMAL
    deadline_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    max_steps: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_route(self) -> "SupervisorRoute":
        if self.requested_action_types and (
            AgentCapability.PROPOSE_ACTION not in self.required_capabilities
        ):
            raise ValueError(
                "Routes requesting action proposals must explicitly require "
                "the PROPOSE_ACTION capability."
            )

        return self

    def to_task(
        self,
        *,
        incident_id: str,
        context: dict[str, Any] | None = None,
        correlation_id: UUID | None = None,
    ) -> AgentTask:
        """Convert a route into a validated, correlated agent task."""

        return AgentTask(
            task_id=self.task_id,
            incident_id=incident_id,
            target_agent=self.agent_name,
            objective=self.objective,
            context={} if context is None else context,
            required_capabilities=self.required_capabilities,
            requested_action_types=self.requested_action_types,
            priority=self.priority,
            deadline_seconds=self.deadline_seconds,
            max_steps=self.max_steps,
            correlation_id=correlation_id,
        )
