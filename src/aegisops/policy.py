"""Deterministic policy enforcement for AegisOps agent actions."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from aegisops.domain import ActionType, ProposedAction, RiskLevel


class PolicyEffect(StrEnum):
    """Outcome produced by the policy engine."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyReason(StrEnum):
    """Machine-readable reasons for policy decisions."""

    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"
    HIGH_RISK_ACTION = "high_risk_action"
    PRODUCTION_MUTATION = "production_mutation"
    SENSITIVE_ACTION_TYPE = "sensitive_action_type"
    BLOCKED_TOOL = "blocked_tool"
    POLICY_ALLOWED = "policy_allowed"


class PolicyDecision(BaseModel):
    """Immutable result returned by the policy engine."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    effect: PolicyEffect
    reasons: tuple[PolicyReason, ...]
    action_id: str
    policy_version: str


class PolicyEngine:
    """Deterministic authorization layer for proposed agent actions."""

    POLICY_VERSION = "2026-09-01"

    _MUTATING_ACTIONS = frozenset(
        {
            ActionType.WRITE,
            ActionType.EXECUTE,
            ActionType.DEPLOY,
            ActionType.ROLLBACK,
        }
    )

    _SENSITIVE_ACTIONS = frozenset(
        {
            ActionType.DEPLOY,
            ActionType.ROLLBACK,
        }
    )

    _HIGH_RISK_LEVELS = frozenset(
        {
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }
    )

    def __init__(
        self,
        *,
        production_environments: Iterable[str] | None = None,
        blocked_tools: Iterable[str] | None = None,
    ) -> None:
        """Create the policy engine with explicit trust boundaries."""

        self._production_environments = {
            environment.casefold()
            for environment in (
                production_environments
                or {
                    "prod",
                    "production",
                }
            )
        }

        self._blocked_tools = {
            tool.casefold()
            for tool in (blocked_tools or set())
        }

    def evaluate(
        self,
        action: ProposedAction,
        *,
        environment: str,
    ) -> PolicyDecision:
        """Evaluate whether an agent action may execute."""

        reasons: list[PolicyReason] = []

        if action.tool_name.casefold() in self._blocked_tools:
            return self._decision(
                action=action,
                effect=PolicyEffect.DENY,
                reasons=(PolicyReason.BLOCKED_TOOL,),
            )

        if action.requires_approval:
            reasons.append(
                PolicyReason.EXPLICIT_APPROVAL_REQUIRED
            )

        if action.risk in self._HIGH_RISK_LEVELS:
            reasons.append(
                PolicyReason.HIGH_RISK_ACTION
            )

        if action.action_type in self._SENSITIVE_ACTIONS:
            reasons.append(
                PolicyReason.SENSITIVE_ACTION_TYPE
            )

        if (
            environment.casefold() in self._production_environments
            and action.action_type in self._MUTATING_ACTIONS
        ):
            reasons.append(
                PolicyReason.PRODUCTION_MUTATION
            )

        if reasons:
            return self._decision(
                action=action,
                effect=PolicyEffect.REQUIRE_APPROVAL,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        return self._decision(
            action=action,
            effect=PolicyEffect.ALLOW,
            reasons=(PolicyReason.POLICY_ALLOWED,),
        )

    def _decision(
        self,
        *,
        action: ProposedAction,
        effect: PolicyEffect,
        reasons: tuple[PolicyReason, ...],
    ) -> PolicyDecision:
        """Build an immutable policy decision."""

        return PolicyDecision(
            effect=effect,
            reasons=reasons,
            action_id=str(action.id),
            policy_version=self.POLICY_VERSION,
        )