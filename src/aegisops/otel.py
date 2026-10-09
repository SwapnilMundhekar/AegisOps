"""OpenTelemetry adapter for AegisOps distributed observability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import TracebackType
from typing import Mapping

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext as OTelSpanContext,
    SpanKind as OTelSpanKind,
    Status,
    StatusCode,
    TraceFlags,
    TraceState,
    Tracer,
    set_span_in_context,
)

from aegisops.observability import (
    CorrelationContext,
    Observation,
    SpanKind,
    SpanStatus,
    TelemetrySink,
    TraceContext,
    current_context,
)


OTelAttributeValue = str | bool | int | float


@dataclass(slots=True)
class OpenTelemetryRuntime:
    """Configured OpenTelemetry runtime owned by AegisOps."""

    provider: TracerProvider
    tracer: Tracer
    processor: BatchSpanProcessor

    def force_flush(
        self,
        *,
        timeout_millis: int = 30_000,
    ) -> bool:
        """Flush queued spans to the configured exporter."""

        return self.provider.force_flush(
            timeout_millis=timeout_millis,
        )

    def shutdown(self) -> None:
        """Flush and shut down the tracing provider."""

        self.provider.shutdown()


def create_otel_runtime(
    *,
    service_name: str = "aegisops",
    service_version: str = "0.1.0",
    deployment_environment: str = "development",
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    exporter: SpanExporter | None = None,
) -> OpenTelemetryRuntime:
    """Create an isolated OpenTelemetry tracing runtime.

    When no exporter is supplied, OTLP over HTTP is used. If endpoint
    is omitted, the OTLP exporter may use its standard environment
    configuration.
    """

    if not service_name.strip():
        raise ValueError(
            "service_name must not be empty."
        )

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment.name": (
                deployment_environment
            ),
        }
    )

    provider = TracerProvider(
        resource=resource,
    )

    if exporter is None:
        exporter_arguments: dict[str, object] = {}

        if endpoint is not None:
            exporter_arguments[
                "endpoint"
            ] = endpoint

        if headers is not None:
            exporter_arguments[
                "headers"
            ] = dict(headers)

        exporter = OTLPSpanExporter(
            **exporter_arguments,
        )

    processor = BatchSpanProcessor(
        exporter
    )

    provider.add_span_processor(
        processor
    )

    tracer = provider.get_tracer(
        "aegisops",
        service_version,
    )

    return OpenTelemetryRuntime(
        provider=provider,
        tracer=tracer,
        processor=processor,
    )


class OpenTelemetryObservation:
    """Mirror an AegisOps Observation into OpenTelemetry.

    The OpenTelemetry-generated trace and span identifiers become the
    authoritative identifiers for the corresponding AegisOps
    Observation, keeping both telemetry representations correlated.
    """

    def __init__(
        self,
        *,
        runtime: OpenTelemetryRuntime,
        sink: TelemetrySink,
        name: str,
        kind: SpanKind,
        incident_id: str | None = None,
        action_id: str | None = None,
        agent_name: str | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError(
                "Observation name must not be empty."
            )

        self._runtime = runtime
        self._sink = sink

        self._name = name.strip()
        self._kind = kind

        self._incident_id = incident_id
        self._action_id = action_id
        self._agent_name = agent_name

        self._attributes: dict[str, object] = dict(
            attributes
            or {}
        )

        self._otel_span: Span | None = None
        self._observation: Observation | None = None

        self._explicit_status: SpanStatus | None = None

    @property
    def context(self) -> CorrelationContext:
        """Return correlation context after the observation starts."""

        if self._observation is None:
            raise RuntimeError(
                "Observation has not started."
            )

        return self._observation.context

    def __enter__(
        self,
    ) -> OpenTelemetryObservation:
        parent = current_context()

        otel_parent = self._build_parent_context(
            parent
        )

        otel_attributes = self._build_otel_attributes(
            parent
        )

        self._otel_span = self._runtime.tracer.start_span(
            self._name,
            context=otel_parent,
            kind=self._map_span_kind(
                self._kind
            ),
            attributes=otel_attributes,
        )

        span_context = (
            self._otel_span.get_span_context()
        )

        trace_context = TraceContext(
            trace_id=f"{span_context.trace_id:032x}",
            span_id=f"{span_context.span_id:016x}",
            parent_span_id=(
                parent.trace.span_id
                if parent is not None
                else None
            ),
            sampled=bool(
                span_context.trace_flags
                & TraceFlags.SAMPLED
            ),
        )

        self._observation = Observation(
            name=self._name,
            kind=self._kind,
            sink=self._sink,
            incident_id=self._incident_id,
            action_id=self._action_id,
            agent_name=self._agent_name,
            attributes=self._attributes,
            trace_context=trace_context,
        )

        self._observation.__enter__()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if (
            self._otel_span is None
            or self._observation is None
        ):
            raise RuntimeError(
                "Observation was not started correctly."
            )

        if exc_value is not None:
            self._otel_span.record_exception(
                exc_value
            )

            self._otel_span.set_status(
                Status(
                    StatusCode.ERROR,
                    str(exc_value),
                )
            )

        elif (
            self._explicit_status
            is SpanStatus.ERROR
        ):
            self._otel_span.set_status(
                Status(
                    StatusCode.ERROR
                )
            )

        else:
            self._otel_span.set_status(
                Status(
                    StatusCode.OK
                )
            )

        try:
            suppressed = self._observation.__exit__(
                exc_type,
                exc_value,
                traceback,
            )

        finally:
            self._otel_span.end()

        return suppressed

    def set_attribute(
        self,
        key: str,
        value: object,
    ) -> None:
        """Attach an attribute to both telemetry representations."""

        if not key.strip():
            raise ValueError(
                "Attribute key must not be empty."
            )

        self._attributes[
            key
        ] = value

        if self._observation is not None:
            self._observation.set_attribute(
                key,
                value,
            )

        if self._otel_span is not None:
            converted = self._convert_attribute(
                value
            )

            if converted is not None:
                self._otel_span.set_attribute(
                    key,
                    converted,
                )

    def set_status(
        self,
        status: SpanStatus,
    ) -> None:
        """Set matching AegisOps and OpenTelemetry status."""

        self._explicit_status = status

        if self._observation is not None:
            self._observation.set_status(
                status
            )

        if self._otel_span is None:
            return

        if status is SpanStatus.ERROR:
            self._otel_span.set_status(
                Status(
                    StatusCode.ERROR
                )
            )

        elif status is SpanStatus.OK:
            self._otel_span.set_status(
                Status(
                    StatusCode.OK
                )
            )

    def _build_otel_attributes(
        self,
        parent: CorrelationContext | None,
    ) -> dict[str, OTelAttributeValue]:
        """Build normalized OpenTelemetry span attributes."""

        incident_id = (
            self._incident_id
            if self._incident_id is not None
            else (
                parent.incident_id
                if parent is not None
                else None
            )
        )

        action_id = (
            self._action_id
            if self._action_id is not None
            else (
                parent.action_id
                if parent is not None
                else None
            )
        )

        agent_name = (
            self._agent_name
            if self._agent_name is not None
            else (
                parent.agent_name
                if parent is not None
                else None
            )
        )

        attributes: dict[
            str,
            OTelAttributeValue,
        ] = {
            "aegisops.span.kind": self._kind.value,
        }

        if incident_id is not None:
            attributes[
                "aegisops.incident.id"
            ] = incident_id

        if action_id is not None:
            attributes[
                "aegisops.action.id"
            ] = action_id

        if agent_name is not None:
            attributes[
                "aegisops.agent.name"
            ] = agent_name

        for key, value in self._attributes.items():
            converted = self._convert_attribute(
                value
            )

            if converted is not None:
                attributes[
                    key
                ] = converted

        return attributes

    @staticmethod
    def _build_parent_context(
        parent: CorrelationContext | None,
    ) -> Context | None:
        """Translate AegisOps parent context into OpenTelemetry."""

        if parent is None:
            return None

        trace_flags = TraceFlags(
            TraceFlags.SAMPLED
            if parent.trace.sampled
            else TraceFlags.DEFAULT
        )

        span_context = OTelSpanContext(
            trace_id=int(
                parent.trace.trace_id,
                16,
            ),
            span_id=int(
                parent.trace.span_id,
                16,
            ),
            is_remote=False,
            trace_flags=trace_flags,
            trace_state=TraceState(),
        )

        return set_span_in_context(
            NonRecordingSpan(
                span_context
            )
        )

    @staticmethod
    def _map_span_kind(
        kind: SpanKind,
    ) -> OTelSpanKind:
        """Map AegisOps operation types to OpenTelemetry kinds."""

        if kind is SpanKind.TOOL:
            return OTelSpanKind.CLIENT

        return OTelSpanKind.INTERNAL

    @staticmethod
    def _convert_attribute(
        value: object,
    ) -> OTelAttributeValue | None:
        """Convert arbitrary structured values into safe OTel values."""

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        ):
            return value

        try:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )

        except (TypeError, ValueError):
            return str(
                value
            )