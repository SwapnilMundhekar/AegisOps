"""Core domain models for the AegisOps agent control plane."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class IncidentSeverity(StrEnum):
    """Operational severity classification."""

    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    SEV4 = "sev4"


class IncidentStatus(StrEnum):
    """Lifecycle states for an autonomous incident workflow."""

    DETECTED = "detected"
    TRIAGING = "triaging"
    INVESTIGATING = "investigating"
    REMEDIATION_PLANNED = "remediation_planned"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    FAILED = "failed"


class RiskLevel(StrEnum):
    """Risk classification applied to proposed agent actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(StrEnum):
    """Categories of actions an agent may request."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"


class ApprovalStatus(StrEnum):
    """Human approval state for a sensitive action."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Evidence(BaseModel):
    """Evidence collected by an agent during investigation."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    reference: str | None = None
    collected_at: datetime = Field(default_factory=utc_now)


class ProposedAction(BaseModel):
    """A tool or infrastructure action proposed by an agent."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    agent: str
    action_type: ActionType
    description: str
    tool_name: str
    tool_arguments: dict[str, object] = Field(default_factory=dict)
    risk: RiskLevel
    requires_approval: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalDecision(BaseModel):
    """Human decision associated with a proposed action."""

    model_config = ConfigDict(extra="forbid")

    action_id: UUID
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str | None = None
    reason: str | None = None
    decided_at: datetime | None = None


class IncidentState(BaseModel):
    """Authoritative state carried through the AegisOps incident workflow."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    allowed_transitions: ClassVar[
        dict[IncidentStatus, frozenset[IncidentStatus]]
    ] = {
        IncidentStatus.DETECTED: frozenset(
            {
                IncidentStatus.TRIAGING,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.TRIAGING: frozenset(
            {
                IncidentStatus.INVESTIGATING,
                IncidentStatus.RESOLVED,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.INVESTIGATING: frozenset(
            {
                IncidentStatus.REMEDIATION_PLANNED,
                IncidentStatus.RESOLVED,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.REMEDIATION_PLANNED: frozenset(
            {
                IncidentStatus.AWAITING_APPROVAL,
                IncidentStatus.EXECUTING,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.AWAITING_APPROVAL: frozenset(
            {
                IncidentStatus.EXECUTING,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.EXECUTING: frozenset(
            {
                IncidentStatus.VERIFYING,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.VERIFYING: frozenset(
            {
                IncidentStatus.RESOLVED,
                IncidentStatus.EXECUTING,
                IncidentStatus.FAILED,
            }
        ),
        IncidentStatus.RESOLVED: frozenset(),
        IncidentStatus.FAILED: frozenset(),
    }

    id: UUID = Field(default_factory=uuid4)
    title: str
    summary: str
    service: str
    environment: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.DETECTED

    hypotheses: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    approvals: list[ApprovalDecision] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def transition_to(self, new_status: IncidentStatus) -> None:
        """Move the incident to a valid next lifecycle state."""
        valid_targets = self.allowed_transitions[self.status]

        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid incident transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status
        self.updated_at = utc_now()

    def add_evidence(self, evidence: Evidence) -> None:
        """Attach investigation evidence and update modification time."""
        self.evidence.append(evidence)
        self.updated_at = utc_now()

    def add_action(self, action: ProposedAction) -> None:
        """Register a proposed agent action."""
        self.proposed_actions.append(action)
        self.updated_at = utc_now()

    def requires_human_approval(self) -> bool:
        """Return True when at least one unresolved action needs approval."""
        pending_action_ids = {
            approval.action_id
            for approval in self.approvals
            if approval.status is ApprovalStatus.PENDING
        }

        return any(
            action.requires_approval
            and (
                action.id in pending_action_ids
                or not any(
                    approval.action_id == action.id
                    for approval in self.approvals
                )
            )
            for action in self.proposed_actions
        )