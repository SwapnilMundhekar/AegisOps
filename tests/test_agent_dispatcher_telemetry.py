"""Regression tests for AegisOps dispatcher tracing and data minimization."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import uuid4

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
import pytest

from aegisops.agent_contracts import (
    AgentCapability,
    AgentResult,
    AgentRole,
    AgentRunStatus,
    AgentSpec,
    AgentTask,
)
from aegisops.agent_dispatcher import AgentDispatcher
from aegisops.observability import (
    InMemoryTelemetrySink,
    SpanKind,
    SpanRecord,
    SpanStatus,
    current_context,
)
from aegisops.otel import create_otel_runtime
from aegisops.supervisor import Supervisor


class RecordingExporter(SpanExporter):
    """Capture OpenTelemetry spans without network access."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True


def make_task(*, deadline_seconds: float = 1.0) -> AgentTask:
    return AgentTask(
        incident_id="INC-OBS-001",
        target_agent="investigator",
        objective="PRIVATE_OBJECTIVE_CANARY",
        context={"authorization": "PRIVATE_TOKEN_CANARY"},
        required_capabilities=frozenset({AgentCapability.TELEMETRY_READ}),
        max_steps=2,
        deadline_seconds=deadline_seconds,
        correlation_id=uuid4(),
    )


def make_supervisor() -> Supervisor:
    return Supervisor(
        [
            AgentSpec(
                name="investigator",
                display_name="Investigator",
                description="Investigates incidents.",
                role=AgentRole.INVESTIGATOR,
                capabilities=frozenset({AgentCapability.TELEMETRY_READ}),
                max_steps=10,
                timeout_seconds=5,
            )
        ]
    )


def make_dispatcher(
    *,
    max_concurrent_tasks: int = 2,
) -> tuple[AgentDispatcher, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    dispatcher = AgentDispatcher(
        make_supervisor(),
        max_concurrent_tasks=max_concurrent_tasks,
        telemetry_sink=sink,
    )
    return dispatcher, sink


def completed(task: AgentTask) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        agent_name=task.target_agent,
        status=AgentRunStatus.COMPLETED,
        summary="PRIVATE_RESULT_CANARY",
    )


def find_span(sink: InMemoryTelemetrySink, name: str) -> SpanRecord:
    matching = [span for span in sink.spans if span.name == name]
    assert len(matching) == 1, f"Expected one span named {name!r}"
    return matching[0]


def register_success(dispatcher: AgentDispatcher) -> None:
    async def handler(task: AgentTask) -> AgentResult:
        return completed(task)

    dispatcher.register_handler("investigator", handler)


@pytest.mark.asyncio
async def test_success_emits_correlated_parent_and_child_spans() -> None:
    dispatcher, sink = make_dispatcher()
    register_success(dispatcher)
    task = make_task()

    result = await dispatcher.execute(task)

    assert result.status is AgentRunStatus.COMPLETED
    assert len(sink.spans) == 2
    parent = find_span(sink, "aegisops.agent.dispatch")
    child = find_span(sink, "aegisops.agent.invoke")
    assert parent.kind is SpanKind.AGENT
    assert child.kind is SpanKind.AGENT
    assert parent.status is SpanStatus.OK
    assert child.status is SpanStatus.OK
    assert parent.context.trace.trace_id == child.context.trace.trace_id
    assert child.context.trace.parent_span_id == parent.context.trace.span_id
    assert parent.context.incident_id == task.incident_id
    assert child.context.agent_name == task.target_agent
    assert parent.attributes["aegisops.agent.task_id"] == str(task.task_id)
    assert parent.attributes["aegisops.agent.correlation_id"] == str(
        task.correlation_id
    )
    assert child.attributes["aegisops.agent.result_status"] == "completed"
    assert child.attributes["aegisops.agent.proposed_action_count"] == 0
    assert current_context() is None


@pytest.mark.asyncio
async def test_execution_and_queue_durations_are_nonnegative() -> None:
    dispatcher, sink = make_dispatcher()
    register_success(dispatcher)

    await dispatcher.execute(make_task())

    parent = find_span(sink, "aegisops.agent.dispatch")
    child = find_span(sink, "aegisops.agent.invoke")
    assert parent.attributes["aegisops.agent.queue_wait_ms"] >= 0
    assert child.attributes["aegisops.agent.execution_ms"] >= 0
    assert parent.duration_ms >= 0
    assert child.duration_ms >= 0


@pytest.mark.asyncio
async def test_telemetry_omits_task_and_result_body_content() -> None:
    dispatcher, sink = make_dispatcher()
    register_success(dispatcher)

    await dispatcher.execute(make_task())

    for span in sink.spans:
        serialized_attributes = repr(span.attributes)
        assert "PRIVATE_OBJECTIVE_CANARY" not in serialized_attributes
        assert "PRIVATE_TOKEN_CANARY" not in serialized_attributes
        assert "PRIVATE_RESULT_CANARY" not in serialized_attributes
        assert "objective" not in span.attributes
        assert "context" not in span.attributes
        assert "summary" not in span.attributes


@pytest.mark.asyncio
async def test_no_sink_preserves_execution_without_trace_context() -> None:
    dispatcher = AgentDispatcher(make_supervisor())
    register_success(dispatcher)

    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.COMPLETED
    assert current_context() is None


@pytest.mark.asyncio
async def test_handler_error_records_safe_structured_failure() -> None:
    dispatcher, sink = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        del task
        raise RuntimeError("PRIVATE_EXCEPTION_CANARY")

    dispatcher.register_handler("investigator", handler)
    result = await dispatcher.execute(make_task())

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_EXECUTION_ERROR"
    assert len(sink.spans) == 2
    for name in ("aegisops.agent.dispatch", "aegisops.agent.invoke"):
        span = find_span(sink, name)
        assert span.status is SpanStatus.ERROR
        assert span.attributes["aegisops.agent.result_status"] == "failed"
        assert span.attributes["aegisops.agent.error_code"] == result.error_code
        assert "PRIVATE_EXCEPTION_CANARY" not in repr(span.attributes)
        assert span.error_message is None


@pytest.mark.asyncio
async def test_invalid_agent_result_is_reflected_in_traces() -> None:
    dispatcher, sink = make_dispatcher()

    async def handler(task: AgentTask) -> AgentResult:
        return completed(task).model_copy(update={"agent_name": "impostor"})

    dispatcher.register_handler("investigator", handler)
    result = await dispatcher.execute(make_task())

    assert result.error_code == "AGENT_CONTRACT_ERROR"
    assert find_span(sink, "aegisops.agent.dispatch").status is SpanStatus.ERROR
    assert find_span(sink, "aegisops.agent.invoke").status is SpanStatus.ERROR


@pytest.mark.asyncio
async def test_execution_timeout_records_dispatch_failure() -> None:
    dispatcher, sink = make_dispatcher()
    cancelled = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return completed(task)

    dispatcher.register_handler("investigator", handler)
    result = await dispatcher.execute(make_task(deadline_seconds=0.02))

    assert result.status is AgentRunStatus.FAILED
    assert result.error_code == "AGENT_TIMEOUT"
    assert cancelled.is_set()
    assert len(sink.spans) == 2
    parent = find_span(sink, "aegisops.agent.dispatch")
    assert parent.status is SpanStatus.ERROR
    assert parent.attributes["aegisops.agent.error_code"] == "AGENT_TIMEOUT"
    assert parent.attributes["aegisops.agent.timed_out_in_queue"] is False


@pytest.mark.asyncio
async def test_queue_timeout_does_not_emit_invoke_span() -> None:
    dispatcher, sink = make_dispatcher(max_concurrent_tasks=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        started.set()
        await release.wait()
        return completed(task)

    dispatcher.register_handler("investigator", handler)
    running = asyncio.create_task(dispatcher.execute(make_task()))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        queued = make_task(deadline_seconds=0.02)
        result = await dispatcher.execute(queued)
        assert result.error_code == "AGENT_TIMEOUT"
        queued_spans = [
            span
            for span in sink.spans
            if span.attributes.get("aegisops.agent.task_id") == str(queued.task_id)
        ]
        assert len(queued_spans) == 1
        assert queued_spans[0].name == "aegisops.agent.dispatch"
        assert queued_spans[0].status is SpanStatus.ERROR
        assert queued_spans[0].attributes["aegisops.agent.timed_out_in_queue"] is True
        assert queued_spans[0].attributes["aegisops.agent.queue_wait_ms"] >= 0
    finally:
        release.set()
        await asyncio.wait_for(running, timeout=2)


@pytest.mark.asyncio
async def test_caller_cancellation_marks_dispatch_span_error() -> None:
    dispatcher, sink = make_dispatcher()
    started = asyncio.Event()

    async def handler(task: AgentTask) -> AgentResult:
        started.set()
        await asyncio.sleep(10)
        return completed(task)

    dispatcher.register_handler("investigator", handler)
    running = asyncio.create_task(dispatcher.execute(make_task()))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    parent = find_span(sink, "aegisops.agent.dispatch")
    assert parent.status is SpanStatus.ERROR
    assert parent.attributes["aegisops.agent.result_status"] == "cancelled"
    assert current_context() is None


@pytest.mark.asyncio
async def test_rejected_assignment_produces_single_error_span() -> None:
    dispatcher, sink = make_dispatcher()
    task = make_task()
    # No handler is registered; this is rejected before the invoke span.
    with pytest.raises(RuntimeError, match="No handler"):
        await dispatcher.execute(task)

    assert len(sink.spans) == 1
    parent = sink.spans[0]
    assert parent.name == "aegisops.agent.dispatch"
    assert parent.status is SpanStatus.ERROR
    assert parent.attributes["aegisops.agent.result_status"] == "rejected"


def test_otel_runtime_requires_an_explicit_sink() -> None:
    exporter = RecordingExporter()
    runtime = create_otel_runtime(exporter=exporter)
    try:
        with pytest.raises(ValueError, match="requires telemetry_sink"):
            AgentDispatcher(make_supervisor(), otel_runtime=runtime)
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
async def test_otel_export_matches_internal_span_relationships() -> None:
    exporter = RecordingExporter()
    runtime = create_otel_runtime(exporter=exporter)
    sink = InMemoryTelemetrySink()
    try:
        dispatcher = AgentDispatcher(
            make_supervisor(), telemetry_sink=sink, otel_runtime=runtime
        )
        register_success(dispatcher)
        task = make_task()
        result = await dispatcher.execute(task)
        assert result.status is AgentRunStatus.COMPLETED
        assert runtime.force_flush()

        assert len(exporter.spans) == 2
        exported = {span.name: span for span in exporter.spans}
        parent = exported["aegisops.agent.dispatch"]
        child = exported["aegisops.agent.invoke"]
        assert parent.context is not None
        assert child.context is not None
        assert child.parent is not None
        assert parent.context.trace_id == child.context.trace_id
        assert child.parent.span_id == parent.context.span_id
        assert parent.status.status_code is StatusCode.OK
        assert child.status.status_code is StatusCode.OK
        assert child.attributes is not None
        assert child.attributes["aegisops.agent.result_status"] == "completed"

        internal_parent = find_span(sink, "aegisops.agent.dispatch")
        assert internal_parent.context.trace.trace_id == (
            f"{parent.context.trace_id:032x}"
        )
        assert internal_parent.context.trace.span_id == (
            f"{parent.context.span_id:016x}"
        )
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
async def test_otel_failure_status_and_attribute_privacy() -> None:
    exporter = RecordingExporter()
    runtime = create_otel_runtime(exporter=exporter)
    sink = InMemoryTelemetrySink()
    try:
        dispatcher = AgentDispatcher(
            make_supervisor(), telemetry_sink=sink, otel_runtime=runtime
        )

        async def handler(task: AgentTask) -> AgentResult:
            del task
            raise RuntimeError("PRIVATE_OTEL_EXCEPTION_CANARY")

        dispatcher.register_handler("investigator", handler)
        result = await dispatcher.execute(make_task())
        assert result.error_code == "AGENT_EXECUTION_ERROR"
        assert runtime.force_flush()
        assert len(exporter.spans) == 2
        for span in exporter.spans:
            assert span.status.status_code is StatusCode.ERROR
            attributes = span.attributes or {}
            assert attributes["aegisops.agent.error_code"] == result.error_code
            for secret in (
                "PRIVATE_OBJECTIVE_CANARY",
                "PRIVATE_TOKEN_CANARY",
                "PRIVATE_OTEL_EXCEPTION_CANARY",
            ):
                assert secret not in repr(attributes)
    finally:
        runtime.shutdown()
