"""Vendor-neutral distributed observability primitives for AegisOps."""

from __future__ import annotations

import contextvars
import secrets
import time
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Protocol


class SpanKind(StrEnum):
    """AegisOps operation categories."""

    INTERNAL = "internal"
    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    POLICY = "policy"
    APPROVAL = "approval"
    CHECKPOINT = "checkpoint"


class SpanStatus(StrEnum):
    """Final execution state of an observed span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TraceContext:
    """W3C-compatible distributed tracing identity."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    sampled: bool = True

    def __post_init__(self) -> None:
        self._validate_hex(
            value=self.trace_id,
            expected_length=32,
            field_name="trace_id",
        )

        self._validate_hex(
            value=self.span_id,
            expected_length=16,
            field_name="span_id",
        )

        if self.parent_span_id is not None:
            self._validate_hex(
                value=self.parent_span_id,
                expected_length=16,
                field_name="parent_span_id",
            )

    @classmethod
    def new_root(
        cls,
        *,
        sampled: bool = True,
    ) -> TraceContext:
        """Create a new root trace context."""

        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=None,
            sampled=sampled,
        )

    def child(self) -> TraceContext:
        """Create a child context in the same distributed trace."""

        return TraceContext(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            parent_span_id=self.span_id,
            sampled=self.sampled,
        )

    def to_traceparent(self) -> str:
        """Serialize context using W3C Trace Context format."""

        flags = "01" if self.sampled else "00"

        return (
            f"00-{self.trace_id}-"
            f"{self.span_id}-{flags}"
        )

    @classmethod
    def from_traceparent(
        cls,
        traceparent: str,
    ) -> TraceContext:
        """Parse a W3C traceparent header."""

        parts = traceparent.strip().split("-")

        if len(parts) != 4:
            raise ValueError(
                "traceparent must contain four fields."
            )

        version, trace_id, span_id, flags = parts

        if version != "00":
            raise ValueError(
                f"Unsupported traceparent version: {version}"
            )

        if flags not in {
            "00",
            "01",
        }:
            raise ValueError(
                f"Unsupported trace flags: {flags}"
            )

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            sampled=flags == "01",
        )

    @staticmethod
    def _validate_hex(
        *,
        value: str,
        expected_length: int,
        field_name: str,
    ) -> None:
        """Validate W3C trace identifier shape."""

        if len(value) != expected_length:
            raise ValueError(
                f"{field_name} must contain "
                f"{expected_length} hexadecimal characters."
            )

        try:
            parsed = int(
                value,
                16,
            )

        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be hexadecimal."
            ) from exc

        if parsed == 0:
            raise ValueError(
                f"{field_name} must not be all zeros."
            )


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """AegisOps business and distributed tracing correlation."""

    trace: TraceContext

    incident_id: str | None = None
    action_id: str | None = None
    agent_name: str | None = None


@dataclass(slots=True)
class SpanRecord:
    """Completed structured telemetry record."""

    name: str
    kind: SpanKind

    context: CorrelationContext

    started_ns: int
    ended_ns: int

    status: SpanStatus

    attributes: dict[str, object] = field(
        default_factory=dict
    )

    error_type: str | None = None
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        """Return span duration in milliseconds."""

        return (
            self.ended_ns
            - self.started_ns
        ) / 1_000_000


class TelemetrySink(Protocol):
    """Backend contract for completed telemetry spans."""

    def emit(
        self,
        span: SpanRecord,
    ) -> None:
        """Persist or export a completed span."""


class InMemoryTelemetrySink:
    """Deterministic telemetry collector for tests and local development."""

    def __init__(self) -> None:
        self._spans: list[SpanRecord] = []

    @property
    def spans(self) -> tuple[SpanRecord, ...]:
        """Return immutable view of completed spans."""

        return tuple(
            self._spans
        )

    def emit(
        self,
        span: SpanRecord,
    ) -> None:
        """Record a completed span."""

        self._spans.append(
            span
        )

    def clear(self) -> None:
        """Remove all recorded spans."""

        self._spans.clear()


_current_context: contextvars.ContextVar[
    CorrelationContext | None
] = contextvars.ContextVar(
    "aegisops_correlation_context",
    default=None,
)


def current_context() -> CorrelationContext | None:
    """Return the correlation context active in this execution context."""

    return _current_context.get()


class Observation:
    """Context manager representing one observable operation."""

    def __init__(
        self,
        *,
        name: str,
        kind: SpanKind,
        sink: TelemetrySink,
        incident_id: str | None = None,
        action_id: str | None = None,
        agent_name: str | None = None,
        attributes: dict[str, object] | None = None,
        trace_context: TraceContext | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError(
                "Span name must not be empty."
            )

        parent = current_context()

        if trace_context is not None:
            trace = trace_context

        elif parent is not None:
            trace = parent.trace.child()

        else:
            trace = TraceContext.new_root()

        self._context = CorrelationContext(
            trace=trace,
            incident_id=(
                incident_id
                if incident_id is not None
                else (
                    parent.incident_id
                    if parent is not None
                    else None
                )
            ),
            action_id=(
                action_id
                if action_id is not None
                else (
                    parent.action_id
                    if parent is not None
                    else None
                )
            ),
            agent_name=(
                agent_name
                if agent_name is not None
                else (
                    parent.agent_name
                    if parent is not None
                    else None
                )
            ),
        )

        self._name = name.strip()
        self._kind = kind
        self._sink = sink

        self._attributes: dict[str, object] = dict(
            attributes
            or {}
        )

        self._status = SpanStatus.UNSET

        self._error_type: str | None = None
        self._error_message: str | None = None

        self._started_ns = 0
        self._token: contextvars.Token[
            CorrelationContext | None
        ] | None = None

    @property
    def context(self) -> CorrelationContext:
        """Return this observation's immutable correlation context."""

        return self._context

    def set_attribute(
        self,
        key: str,
        value: object,
    ) -> None:
        """Attach structured metadata to the current span."""

        if not key.strip():
            raise ValueError(
                "Attribute key must not be empty."
            )

        self._attributes[
            key
        ] = value

    def set_status(
        self,
        status: SpanStatus,
    ) -> None:
        """Set the operation status."""

        self._status = status

    def __enter__(
        self,
    ) -> Observation:
        self._started_ns = time.perf_counter_ns()

        self._token = _current_context.set(
            self._context
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ended_ns = time.perf_counter_ns()

        if exc_value is not None:
            self._status = SpanStatus.ERROR
            self._error_type = type(
                exc_value
            ).__name__
            self._error_message = str(
                exc_value
            )

        elif self._status is SpanStatus.UNSET:
            self._status = SpanStatus.OK

        span = SpanRecord(
            name=self._name,
            kind=self._kind,
            context=self._context,
            started_ns=self._started_ns,
            ended_ns=ended_ns,
            status=self._status,
            attributes=dict(
                self._attributes
            ),
            error_type=self._error_type,
            error_message=self._error_message,
        )

        self._sink.emit(
            span
        )

        if self._token is not None:
            _current_context.reset(
                self._token
            )

        return False


def inject_trace_headers(
    context: CorrelationContext,
) -> dict[str, str]:
    """Create outbound W3C trace headers for another service."""

    return {
        "traceparent": (
            context.trace.to_traceparent()
        ),
    }


def extract_trace_headers(
    headers: dict[str, str],
) -> TraceContext | None:
    """Extract W3C tracing identity from inbound headers."""

    traceparent = next(
        (
            value
            for key, value in headers.items()
            if key.casefold() == "traceparent"
        ),
        None,
    )

    if traceparent is None:
        return None

    return TraceContext.from_traceparent(
        traceparent
    )