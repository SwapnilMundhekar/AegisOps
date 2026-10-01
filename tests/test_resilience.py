"""Tests for AegisOps retry and circuit-breaker resilience controls."""

from __future__ import annotations

import pytest

from aegisops.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ResilientExecutor,
    RetryPolicy,
)


def test_retry_policy_uses_exponential_backoff() -> None:
    """Retry delay should grow exponentially until capped."""

    policy = RetryPolicy(
        max_attempts=5,
        base_delay_seconds=1.0,
        max_delay_seconds=4.0,
        jitter_ratio=0.0,
    )

    assert policy.delay_for(
        1,
        random_value=0.5,
    ) == 1.0

    assert policy.delay_for(
        2,
        random_value=0.5,
    ) == 2.0

    assert policy.delay_for(
        3,
        random_value=0.5,
    ) == 4.0

    assert policy.delay_for(
        4,
        random_value=0.5,
    ) == 4.0


def test_retry_policy_applies_bounded_jitter() -> None:
    """Jitter should remain inside the configured percentage."""

    policy = RetryPolicy(
        base_delay_seconds=10.0,
        max_delay_seconds=10.0,
        jitter_ratio=0.20,
    )

    minimum = policy.delay_for(
        1,
        random_value=0.0,
    )

    midpoint = policy.delay_for(
        1,
        random_value=0.5,
    )

    maximum = policy.delay_for(
        1,
        random_value=1.0,
    )

    assert minimum == pytest.approx(8.0)
    assert midpoint == pytest.approx(10.0)
    assert maximum == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_idempotent_operation_retries_transient_failure() -> None:
    """Idempotent operations should retry transient failures."""

    attempts = 0
    delays: list[float] = []

    async def fake_sleep(
        delay: float,
    ) -> None:
        delays.append(delay)

    async def operation() -> str:
        nonlocal attempts

        attempts += 1

        if attempts < 3:
            raise ConnectionError(
                "temporary service failure"
            )

        return "success"

    executor = ResilientExecutor(
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
        sleep=fake_sleep,
        random_source=lambda: 0.5,
    )

    result = await executor.execute(
        operation,
        idempotent=True,
    )

    assert result == "success"
    assert attempts == 3

    assert delays == [
        1.0,
        2.0,
    ]

    assert (
        executor.circuit_breaker.state
        is CircuitState.CLOSED
    )

    assert executor.circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_retry_stops_after_max_attempts() -> None:
    """Retry loops must remain bounded."""

    attempts = 0
    delays: list[float] = []

    async def fake_sleep(
        delay: float,
    ) -> None:
        delays.append(delay)

    async def operation() -> None:
        nonlocal attempts

        attempts += 1

        raise TimeoutError(
            "dependency timeout"
        )

    executor = ResilientExecutor(
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
        sleep=fake_sleep,
        random_source=lambda: 0.5,
    )

    with pytest.raises(
        TimeoutError,
        match="dependency timeout",
    ):
        await executor.execute(
            operation,
            idempotent=True,
        )

    assert attempts == 3

    assert delays == [
        0.5,
        1.0,
    ]


@pytest.mark.asyncio
async def test_non_idempotent_operation_is_not_retried() -> None:
    """Unsafe side effects must not be repeated automatically."""

    attempts = 0
    delays: list[float] = []

    async def fake_sleep(
        delay: float,
    ) -> None:
        delays.append(delay)

    async def deploy_operation() -> None:
        nonlocal attempts

        attempts += 1

        raise ConnectionError(
            "connection lost after deployment request"
        )

    executor = ResilientExecutor(
        retry_policy=RetryPolicy(
            max_attempts=5,
            base_delay_seconds=1.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        ),
        sleep=fake_sleep,
    )

    with pytest.raises(
        ConnectionError,
        match="connection lost",
    ):
        await executor.execute(
            deploy_operation,
            idempotent=False,
        )

    assert attempts == 1
    assert delays == []


@pytest.mark.asyncio
async def test_non_retryable_exception_fails_immediately() -> None:
    """Permanent application errors should fail fast."""

    attempts = 0

    async def operation() -> None:
        nonlocal attempts

        attempts += 1

        raise ValueError(
            "invalid request"
        )

    executor = ResilientExecutor(
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
        )
    )

    with pytest.raises(
        ValueError,
        match="invalid request",
    ):
        await executor.execute(
            operation,
            idempotent=True,
        )

    assert attempts == 1


def test_circuit_opens_after_failure_threshold() -> None:
    """Repeated dependency failures should open the circuit."""

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        monotonic_clock=lambda: 100.0,
    )

    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED

    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.failure_count == 3


def test_open_circuit_fails_fast() -> None:
    """Calls must be rejected while the circuit is open."""

    now = 100.0

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        monotonic_clock=lambda: now,
    )

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN

    with pytest.raises(
        CircuitOpenError,
        match="Circuit is open",
    ):
        breaker.before_call()


def test_open_circuit_moves_to_half_open_after_timeout() -> None:
    """The breaker should allow a probe after the recovery interval."""

    clock = {
        "now": 100.0,
    }

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        monotonic_clock=lambda: clock["now"],
    )

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN

    clock["now"] = 131.0

    breaker.before_call()

    assert breaker.state is CircuitState.HALF_OPEN


def test_half_open_success_closes_circuit() -> None:
    """A successful recovery probe should close the circuit."""

    clock = {
        "now": 100.0,
    }

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=10.0,
        monotonic_clock=lambda: clock["now"],
    )

    breaker.record_failure()

    clock["now"] = 111.0

    breaker.before_call()

    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_half_open_failure_reopens_circuit() -> None:
    """A failed recovery probe must reopen the circuit."""

    clock = {
        "now": 100.0,
    }

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=10.0,
        monotonic_clock=lambda: clock["now"],
    )

    breaker.record_failure()

    clock["now"] = 111.0

    breaker.before_call()

    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_success_resets_previous_failures() -> None:
    """Successful execution should reset consecutive failure count."""

    attempts = 0

    async def fake_sleep(
        _: float,
    ) -> None:
        return None

    async def operation() -> str:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            raise ConnectionError(
                "temporary failure"
            )

        return "healthy"

    breaker = CircuitBreaker(
        failure_threshold=5,
    )

    executor = ResilientExecutor(
        retry_policy=RetryPolicy(
            max_attempts=2,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_ratio=0.0,
        ),
        circuit_breaker=breaker,
        sleep=fake_sleep,
    )

    result = await executor.execute(
        operation,
        idempotent=True,
    )

    assert result == "healthy"

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0