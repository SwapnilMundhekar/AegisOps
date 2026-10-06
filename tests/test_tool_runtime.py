"""Integration tests for the resilient AegisOps tool runtime."""

from __future__ import annotations

import asyncio

import pytest

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import AuditEventType, AuditLedger
from aegisops.domain import (
    ActionType,
    IncidentSeverity,
    IncidentState,
    ProposedAction,
    RiskLevel,
)
from aegisops.policy import PolicyEngine
from aegisops.resilience import (
    CircuitBreaker,
    RetryPolicy,
)
from aegisops.tool_runtime import (
    ToolExecutionError,
    ToolNotRegisteredError,
    ToolRuntime,
)


def build_incident() -> IncidentState:
    """Create a representative staging incident."""

    return IncidentState(
        title="Payment API degradation",
        summary="Operational investigation in progress.",
        service="payment-api",
        environment="staging",
        severity=IncidentSeverity.SEV2,
    )


def build_action(
    *,
    tool_name: str,
    action_type: ActionType = ActionType.READ,
    risk: RiskLevel = RiskLevel.LOW,
) -> ProposedAction:
    """Create a representative proposed tool action."""

    return ProposedAction(
        agent="investigator-agent",
        action_type=action_type,
        description="Execute controlled operational tool.",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
        },
        risk=risk,
    )


def build_runtime() -> tuple[
    ToolRuntime,
    AuditLedger,
]:
    """Create a runtime with shared policy and audit components."""

    ledger = AuditLedger()

    policy = PolicyEngine()

    approvals = ApprovalManager(
        audit_ledger=ledger,
    )

    gate = ActionGate(
        policy_engine=policy,
        audit_ledger=ledger,
    )

    runtime = ToolRuntime(
        action_gate=gate,
        approval_manager=approvals,
        audit_ledger=ledger,
    )

    return runtime, ledger


@pytest.mark.asyncio
async def test_idempotent_tool_retries_transient_failure() -> None:
    """Read-only idempotent tools should recover from transient errors."""

    runtime, ledger = build_runtime()

    attempts = 0

    async def prometheus_query(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        nonlocal attempts

        attempts += 1

        if attempts < 3:
            raise ConnectionError(
                "temporary Prometheus outage"
            )

        return {
            "service": arguments["service"],
            "p95_ms": 210,
        }

    runtime.register(
        name="prometheus.query",
        handler=prometheus_query,
        idempotent=True,
        timeout_seconds=1.0,
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="prometheus.query",
    )

    result = await runtime.execute(
        incident=incident,
        action=action,
    )

    assert attempts == 3

    assert result == {
        "service": "payment-api",
        "p95_ms": 210,
    }

    assert runtime.circuit_state(
        "prometheus.query"
    ) == "closed"

    event_types = [
        event.event_type
        for event in ledger.events
    ]

    assert AuditEventType.TOOL_STARTED in event_types
    assert AuditEventType.TOOL_COMPLETED in event_types

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_non_idempotent_tool_is_not_retried() -> None:
    """Mutation tools must not repeat ambiguous side effects."""

    runtime, ledger = build_runtime()

    attempts = 0

    async def deployment_tool(
        arguments: dict[str, object],
    ) -> None:
        nonlocal attempts

        attempts += 1

        raise ConnectionError(
            "connection lost after request submission"
        )

    runtime.register(
        name="staging.deploy",
        handler=deployment_tool,
        idempotent=False,
        timeout_seconds=1.0,
        retry_policy=RetryPolicy(
            max_attempts=5,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="staging.deploy",
        action_type=ActionType.WRITE,
        risk=RiskLevel.LOW,
    )

    with pytest.raises(
        ToolExecutionError,
        match="Tool execution failed",
    ):
        await runtime.execute(
            incident=incident,
            action=action,
        )

    assert attempts == 1

    event_types = [
        event.event_type
        for event in ledger.events
    ]

    assert AuditEventType.TOOL_FAILED in event_types

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_tool_timeout_is_enforced() -> None:
    """A tool exceeding its execution budget must fail safely."""

    runtime, ledger = build_runtime()

    async def slow_tool(
        arguments: dict[str, object],
    ) -> str:
        await asyncio.sleep(
            0.05
        )

        return str(
            arguments["service"]
        )

    runtime.register(
        name="telemetry.slow_query",
        handler=slow_tool,
        idempotent=False,
        timeout_seconds=0.001,
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="telemetry.slow_query",
    )

    with pytest.raises(
        ToolExecutionError,
        match="timed out",
    ):
        await runtime.execute(
            incident=incident,
            action=action,
        )

    assert ledger.verify_integrity() is True

    failed_events = [
        event
        for event in ledger.events
        if event.event_type
        is AuditEventType.TOOL_FAILED
    ]

    assert len(failed_events) == 1


@pytest.mark.asyncio
async def test_repeated_failure_opens_tool_circuit() -> None:
    """Persistent dependency failure should open that tool's circuit."""

    runtime, ledger = build_runtime()

    attempts = 0

    async def unavailable_service(
        arguments: dict[str, object],
    ) -> None:
        nonlocal attempts

        attempts += 1

        raise ConnectionError(
            f"{arguments['service']} unavailable"
        )

    runtime.register(
        name="prometheus.query",
        handler=unavailable_service,
        idempotent=False,
        timeout_seconds=1.0,
        circuit_breaker=CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=60.0,
        ),
    )

    incident = build_incident()

    first_action = build_action(
        tool_name="prometheus.query",
    )

    with pytest.raises(
        ToolExecutionError,
    ):
        await runtime.execute(
            incident=incident,
            action=first_action,
        )

    assert attempts == 1

    assert runtime.circuit_state(
        "prometheus.query"
    ) == "open"

    second_action = build_action(
        tool_name="prometheus.query",
    )

    with pytest.raises(
        ToolExecutionError,
        match="circuit is open",
    ):
        await runtime.execute(
            incident=incident,
            action=second_action,
        )

    assert attempts == 1

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_circuit_breakers_are_isolated_per_tool() -> None:
    """Failure in one dependency must not disable unrelated tools."""

    runtime, ledger = build_runtime()

    async def failing_prometheus(
        arguments: dict[str, object],
    ) -> None:
        raise ConnectionError(
            f"{arguments['service']} telemetry unavailable"
        )

    async def github_reader(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        return {
            "service": arguments["service"],
            "commit": "abc123",
        }

    runtime.register(
        name="prometheus.query",
        handler=failing_prometheus,
        idempotent=False,
        circuit_breaker=CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=60.0,
        ),
    )

    runtime.register(
        name="github.read",
        handler=github_reader,
        idempotent=True,
    )

    incident = build_incident()

    prometheus_action = build_action(
        tool_name="prometheus.query",
    )

    with pytest.raises(
        ToolExecutionError,
    ):
        await runtime.execute(
            incident=incident,
            action=prometheus_action,
        )

    assert runtime.circuit_state(
        "prometheus.query"
    ) == "open"

    assert runtime.circuit_state(
        "github.read"
    ) == "closed"

    github_action = build_action(
        tool_name="github.read",
    )

    result = await runtime.execute(
        incident=incident,
        action=github_action,
    )

    assert result == {
        "service": "payment-api",
        "commit": "abc123",
    }

    assert runtime.circuit_state(
        "github.read"
    ) == "closed"

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_sync_handler_executes_successfully() -> None:
    """Synchronous integrations should work behind the async runtime."""

    runtime, ledger = build_runtime()

    def sync_handler(
        arguments: dict[str, object],
    ) -> dict[str, object]:
        return {
            "service": arguments["service"],
            "status": "healthy",
        }

    runtime.register(
        name="legacy.health_check",
        handler=sync_handler,
        idempotent=True,
    )

    incident = build_incident()

    action = build_action(
        tool_name="legacy.health_check",
    )

    result = await runtime.execute(
        incident=incident,
        action=action,
    )

    assert result == {
        "service": "payment-api",
        "status": "healthy",
    }

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_unknown_tool_fails_explicitly() -> None:
    """Unregistered model-generated tool names must never be executed."""

    runtime, ledger = build_runtime()

    incident = build_incident()

    action = build_action(
        tool_name="invented.root_access",
    )

    with pytest.raises(
        ToolNotRegisteredError,
        match="Tool is not registered",
    ):
        await runtime.execute(
            incident=incident,
            action=action,
        )

    assert len(ledger.events) == 0