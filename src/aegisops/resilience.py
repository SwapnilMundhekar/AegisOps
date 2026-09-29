"""Reliability primitives for AegisOps tool and service execution."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar


T = TypeVar("T")


class CircuitState(StrEnum):
    """Lifecycle states for a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when execution is blocked by an open circuit."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configuration for bounded exponential-backoff retries."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    jitter_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if self.base_delay_seconds < 0:
            raise ValueError(
                "base_delay_seconds must not be negative."
            )

        if self.max_delay_seconds < 0:
            raise ValueError(
                "max_delay_seconds must not be negative."
            )

        if (
            self.max_delay_seconds
            < self.base_delay_seconds
        ):
            raise ValueError(
                "max_delay_seconds must be greater than "
                "or equal to base_delay_seconds."
            )

        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError(
                "jitter_ratio must be between 0 and 1."
            )

    def delay_for(
        self,
        failed_attempt: int,
        *,
        random_value: float,
    ) -> float:
        """Return bounded exponential delay with symmetric jitter."""

        if failed_attempt < 1:
            raise ValueError(
                "failed_attempt must be at least 1."
            )

        if not 0 <= random_value <= 1:
            raise ValueError(
                "random_value must be between 0 and 1."
            )

        exponential_delay = (
            self.base_delay_seconds
            * (2 ** (failed_attempt - 1))
        )

        bounded_delay = min(
            exponential_delay,
            self.max_delay_seconds,
        )

        jitter_multiplier = (
            1
            + (
                (random_value * 2) - 1
            )
            * self.jitter_ratio
        )

        return max(
            0.0,
            bounded_delay * jitter_multiplier,
        )


class CircuitBreaker:
    """Fail fast when a dependency repeatedly fails."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                "failure_threshold must be at least 1."
            )

        if recovery_timeout_seconds < 0:
            raise ValueError(
                "recovery_timeout_seconds must not be negative."
            )

        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = (
            recovery_timeout_seconds
        )
        self._clock = monotonic_clock

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return consecutive failures observed by the circuit."""
        return self._failure_count

    def before_call(self) -> None:
        """Verify that a dependency call is currently permitted."""

        if self._state is not CircuitState.OPEN:
            return

        if self._opened_at is None:
            raise CircuitOpenError(
                "Circuit is open."
            )

        elapsed = self._clock() - self._opened_at

        if elapsed >= self._recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            return

        remaining = (
            self._recovery_timeout_seconds
            - elapsed
        )

        raise CircuitOpenError(
            "Circuit is open; dependency probe is blocked "
            f"for another {remaining:.3f} seconds."
        )

    def record_success(self) -> None:
        """Close the circuit after a successful dependency call."""

        self._failure_count = 0
        self._opened_at = None
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record failure and open the circuit when required."""

        if self._state is CircuitState.HALF_OPEN:
            self._open()
            return

        self._failure_count += 1

        if self._failure_count >= self._failure_threshold:
            self._open()

    def _open(self) -> None:
        """Move the breaker into the open state."""

        self._state = CircuitState.OPEN
        self._opened_at = self._clock()


class ResilientExecutor:
    """Execute asynchronous operations with retry and circuit controls."""

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        sleep: Callable[
            [float],
            Awaitable[None],
        ] = asyncio.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._retry_policy = (
            retry_policy
            or RetryPolicy()
        )

        self._circuit_breaker = (
            circuit_breaker
            or CircuitBreaker()
        )

        self._sleep = sleep
        self._random_source = random_source

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Expose circuit state for monitoring and health reporting."""
        return self._circuit_breaker

    async def execute(
        self,
        operation: Callable[
            [],
            Awaitable[T],
        ],
        *,
        idempotent: bool,
        retry_on: tuple[type[BaseException], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ) -> T:
        """Execute an operation using bounded safe retries.

        Non-idempotent operations are never automatically retried.
        """

        attempt = 1

        while True:
            self._circuit_breaker.before_call()

            try:
                result = await operation()

            except BaseException as exc:
                self._circuit_breaker.record_failure()

                should_retry = (
                    idempotent
                    and isinstance(
                        exc,
                        retry_on,
                    )
                    and attempt
                    < self._retry_policy.max_attempts
                )

                if not should_retry:
                    raise

                delay = self._retry_policy.delay_for(
                    attempt,
                    random_value=self._random_source(),
                )

                await self._sleep(
                    delay
                )

                attempt += 1

                continue

            self._circuit_breaker.record_success()

            return result