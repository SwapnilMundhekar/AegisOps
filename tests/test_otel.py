"""Tests for the AegisOps OpenTelemetry bridge."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode

from aegisops.observability import (
    InMemoryTelemetrySink,
    SpanKind,
    SpanStatus,
)
from aegisops.otel import (
    OpenTelemetryObservation,
    create_otel_runtime,
)


class RecordingExporter(SpanExporter):
    """Deterministic exporter used by integration tests."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []
        self.shutdown_called = False

    def export(
        self,
        spans: Sequence[ReadableSpan],
    ) -> SpanExportResult:
        """Capture exported spans in memory."""

        self.spans.extend(
            spans
        )

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Record exporter shutdown."""

        self.shutdown_called = True

    def force_flush(
        self,
        timeout_millis: int = 30_000,
    ) -> bool:
        """No buffering exists inside this exporter."""

        del timeout_millis

        return True


def build_runtime() -> tuple[
    RecordingExporter,
    object,
]:
    """Create an isolated OpenTelemetry runtime."""

    exporter = RecordingExporter()

    runtime = create_otel_runtime(
        service_name="aegisops-test",
        service_version="0.1.0-test",
        deployment_environment="test",
        exporter=exporter,
    )

    return exporter, runtime


def span_by_name(
    exporter: RecordingExporter,
    name: str,
) -> ReadableSpan:
    """Resolve one exported span by name."""

    matching = [
        span
        for span in exporter.spans
        if span.name == name
    ]

    assert len(matching) == 1

    return matching[0]


def test_aegisops_and_otel_share_trace_identity() -> None:
    """Both telemetry systems must describe the same trace/span."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="incident.orchestrate",
            kind=SpanKind.AGENT,
            incident_id="INC-001",
            agent_name="supervisor-agent",
        ) as observation:
            aegisops_context = observation.context

        assert runtime.force_flush()

        assert len(sink.spans) == 1
        assert len(exporter.spans) == 1

        otel_span = exporter.spans[0]

        assert otel_span.context is not None

        otel_trace_id = (
            f"{otel_span.context.trace_id:032x}"
        )

        otel_span_id = (
            f"{otel_span.context.span_id:016x}"
        )

        assert (
            aegisops_context.trace.trace_id
            == otel_trace_id
        )

        assert (
            aegisops_context.trace.span_id
            == otel_span_id
        )

        assert (
            sink.spans[0].context.trace.trace_id
            == otel_trace_id
        )

    finally:
        runtime.shutdown()


def test_nested_observations_preserve_parent_child_relationship() -> None:
    """Child spans must remain in the same trace with correct parent."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="incident.orchestrate",
            kind=SpanKind.AGENT,
            incident_id="INC-002",
            agent_name="supervisor-agent",
        ) as parent:
            parent_context = parent.context

            with OpenTelemetryObservation(
                runtime=runtime,
                sink=sink,
                name="tool.prometheus.query",
                kind=SpanKind.TOOL,
                action_id="ACT-001",
            ) as child:
                child_context = child.context

        assert runtime.force_flush()

        parent_span = span_by_name(
            exporter,
            "incident.orchestrate",
        )

        child_span = span_by_name(
            exporter,
            "tool.prometheus.query",
        )

        assert parent_span.context is not None
        assert child_span.context is not None
        assert child_span.parent is not None

        assert (
            parent_span.context.trace_id
            == child_span.context.trace_id
        )

        assert (
            child_span.parent.span_id
            == parent_span.context.span_id
        )

        assert (
            child_context.trace.trace_id
            == parent_context.trace.trace_id
        )

        assert (
            child_context.trace.parent_span_id
            == parent_context.trace.span_id
        )

        assert child_context.incident_id == "INC-002"
        assert child_context.action_id == "ACT-001"

        assert (
            child_context.agent_name
            == "supervisor-agent"
        )

    finally:
        runtime.shutdown()


def test_business_correlation_becomes_otel_attributes() -> None:
    """Incident, action, and agent identity should be queryable."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="policy.evaluate",
            kind=SpanKind.POLICY,
            incident_id="INC-003",
            action_id="ACT-900",
            agent_name="deployment-agent",
            attributes={
                "aegisops.policy.version": "2026-09-01",
                "aegisops.risk.level": "high",
            },
        ):
            pass

        assert runtime.force_flush()

        span = span_by_name(
            exporter,
            "policy.evaluate",
        )

        assert span.attributes is not None

        assert (
            span.attributes["aegisops.incident.id"]
            == "INC-003"
        )

        assert (
            span.attributes["aegisops.action.id"]
            == "ACT-900"
        )

        assert (
            span.attributes["aegisops.agent.name"]
            == "deployment-agent"
        )

        assert (
            span.attributes["aegisops.span.kind"]
            == "policy"
        )

        assert (
            span.attributes["aegisops.policy.version"]
            == "2026-09-01"
        )

        assert (
            span.attributes["aegisops.risk.level"]
            == "high"
        )

    finally:
        runtime.shutdown()


def test_structured_attributes_are_serialized_deterministically() -> None:
    """Complex application attributes should remain exportable."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="tool.execute",
            kind=SpanKind.TOOL,
        ) as observation:
            observation.set_attribute(
                "aegisops.tool.arguments",
                {
                    "service": "payment-api",
                    "replicas": 3,
                },
            )

        assert runtime.force_flush()

        span = span_by_name(
            exporter,
            "tool.execute",
        )

        assert span.attributes is not None

        assert (
            span.attributes["aegisops.tool.arguments"]
            == '{"replicas":3,"service":"payment-api"}'
        )

    finally:
        runtime.shutdown()


def test_exception_marks_both_telemetry_layers_as_error() -> None:
    """An escaped exception must produce ERROR telemetry."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with pytest.raises(
            RuntimeError,
            match="dependency unavailable",
        ):
            with OpenTelemetryObservation(
                runtime=runtime,
                sink=sink,
                name="tool.github.read",
                kind=SpanKind.TOOL,
            ):
                raise RuntimeError(
                    "dependency unavailable"
                )

        assert runtime.force_flush()

        assert len(sink.spans) == 1

        assert (
            sink.spans[0].status
            is SpanStatus.ERROR
        )

        assert (
            sink.spans[0].error_type
            == "RuntimeError"
        )

        span = span_by_name(
            exporter,
            "tool.github.read",
        )

        assert (
            span.status.status_code
            is StatusCode.ERROR
        )

        assert (
            span.status.description
            == "dependency unavailable"
        )

        exception_events = [
            event
            for event in span.events
            if event.name == "exception"
        ]

        assert len(exception_events) == 1

    finally:
        runtime.shutdown()


def test_explicit_error_status_propagates_to_otel() -> None:
    """Application-detected failures should set OTel ERROR status."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="policy.block",
            kind=SpanKind.POLICY,
        ) as observation:
            observation.set_status(
                SpanStatus.ERROR
            )

        assert runtime.force_flush()

        span = span_by_name(
            exporter,
            "policy.block",
        )

        assert (
            span.status.status_code
            is StatusCode.ERROR
        )

        assert (
            sink.spans[0].status
            is SpanStatus.ERROR
        )

    finally:
        runtime.shutdown()


def test_runtime_resource_metadata_is_exported() -> None:
    """Service identity must travel with exported telemetry."""

    exporter, runtime = build_runtime()

    sink = InMemoryTelemetrySink()

    try:
        with OpenTelemetryObservation(
            runtime=runtime,
            sink=sink,
            name="health.check",
            kind=SpanKind.INTERNAL,
        ):
            pass

        assert runtime.force_flush()

        span = exporter.spans[0]

        assert (
            span.resource.attributes["service.name"]
            == "aegisops-test"
        )

        assert (
            span.resource.attributes["service.version"]
            == "0.1.0-test"
        )

        assert (
            span.resource.attributes[
                "deployment.environment.name"
            ]
            == "test"
        )

    finally:
        runtime.shutdown()


def test_local_runtime_does_not_replace_global_provider() -> None:
    """Creating AegisOps telemetry must not mutate process-global tracing."""

    global_provider_before = (
        trace.get_tracer_provider()
    )

    exporter, runtime = build_runtime()

    try:
        global_provider_after = (
            trace.get_tracer_provider()
        )

        assert (
            global_provider_after
            is global_provider_before
        )

    finally:
        runtime.shutdown()


def test_runtime_shutdown_reaches_exporter() -> None:
    """Runtime shutdown should cleanly close its exporter pipeline."""

    exporter, runtime = build_runtime()

    runtime.shutdown()

    assert exporter.shutdown_called is True