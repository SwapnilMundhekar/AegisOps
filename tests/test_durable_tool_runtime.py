"""Cross-worker durability tests for the AegisOps ToolRuntime."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import AuditLedger
from aegisops.domain import (
    ActionType,
    IncidentSeverity,
    IncidentState,
    ProposedAction,
    RiskLevel,
)
from aegisops.execution_journal import (
    ExecutionStatus,
    SQLiteExecutionJournal,
)
from aegisops.policy import PolicyEngine
from aegisops.resilience import RetryPolicy
from aegisops.tool_runtime import (
    ToolExecutionAlreadyCompletedError,
    ToolExecutionError,
    ToolExecutionInProgressError,
    ToolExecutionUncertainError,
    ToolRuntime,
)


def build_incident() -> IncidentState:
    """Create a staging incident that does not require human approval."""

    return IncidentState(
        title="Payment API degradation",
        summary="Controlled remediation investigation.",
        service="payment-api",
        environment="staging",
        severity=IncidentSeverity.SEV2,
    )


def build_action(
    *,
    tool_name: str,
    action_type: ActionType = ActionType.READ,
) -> ProposedAction:
    """Create a stable action identity used across worker restarts."""

    return ProposedAction(
        agent="operations-agent",
        action_type=action_type,
        description="Execute durable controlled operation.",
        tool_name=tool_name,
        tool_arguments={
            "service": "payment-api",
            "version": "1.8.3",
        },
        risk=RiskLevel.LOW,
    )


def build_runtime(
    *,
    journal: SQLiteExecutionJournal,
    worker_id: str,
) -> tuple[
    ToolRuntime,
    AuditLedger,
]:
    """Create one independent runtime process sharing a journal."""

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
        execution_journal=journal,
        worker_id=worker_id,
        execution_lease_seconds=60.0,
    )

    return runtime, ledger


def expire_lease(
    *,
    database: Path,
    idempotency_key: str,
) -> None:
    """Simulate a worker crash by expiring its durable lease."""

    expired_at = (
        datetime.now(timezone.utc)
        - timedelta(minutes=5)
    )

    connection = sqlite3.connect(
        database
    )

    try:
        connection.execute(
            """
            UPDATE execution_journal
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            (
                expired_at.isoformat(),
                idempotency_key,
            ),
        )

        connection.commit()

    finally:
        connection.close()


@pytest.mark.asyncio
async def test_stable_downstream_idempotency_key_is_injected() -> None:
    """The downstream service must receive a stable operation identity."""

    database = Path(
        "tests-runtime-idempotency.db"
    )

    try:
        if database.exists():
            database.unlink()

        journal = SQLiteExecutionJournal(
            database
        )

        runtime, ledger = build_runtime(
            journal=journal,
            worker_id="worker-a",
        )

        received_keys: list[str] = []

        async def payment_tool(
            arguments: dict[str, object],
        ) -> dict[str, object]:
            received_keys.append(
                str(
                    arguments[
                        "idempotency_key"
                    ]
                )
            )

            return {
                "transaction_id": "txn-001",
            }

        runtime.register(
            name="payments.charge",
            handler=payment_tool,
            idempotent=True,
            idempotency_argument="idempotency_key",
        )

        incident = build_incident()

        action = build_action(
            tool_name="payments.charge",
            action_type=ActionType.WRITE,
        )

        expected_key = (
            SQLiteExecutionJournal.build_idempotency_key(
                incident_id=incident.id,
                action_id=action.id,
                tool_name="payments.charge",
            )
        )

        result = await runtime.execute(
            incident=incident,
            action=action,
        )

        assert result == {
            "transaction_id": "txn-001",
        }

        assert received_keys == [
            expected_key,
        ]

        assert (
            "idempotency_key"
            not in action.tool_arguments
        )

        record = journal.load(
            expected_key
        )

        assert (
            record.status
            is ExecutionStatus.SUCCEEDED
        )

        assert record.attempt_count == 1
        assert record.result_digest is not None

        assert ledger.verify_integrity() is True

    finally:
        if database.exists():
            database.unlink()


@pytest.mark.asyncio
async def test_live_execution_lease_blocks_second_worker(
    tmp_path: Path,
) -> None:
    """Two workers must never execute the same leased operation."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    runtime_a, _ = build_runtime(
        journal=journal,
        worker_id="worker-a",
    )

    runtime_b, _ = build_runtime(
        journal=journal,
        worker_id="worker-b",
    )

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    worker_b_executions = 0

    async def worker_a_handler(
        arguments: dict[str, object],
    ) -> str:
        handler_started.set()

        await release_handler.wait()

        return str(
            arguments["service"]
        )

    async def worker_b_handler(
        arguments: dict[str, object],
    ) -> str:
        nonlocal worker_b_executions

        worker_b_executions += 1

        return str(
            arguments["service"]
        )

    runtime_a.register(
        name="service.reconcile",
        handler=worker_a_handler,
        idempotent=True,
    )

    runtime_b.register(
        name="service.reconcile",
        handler=worker_b_handler,
        idempotent=True,
    )

    incident = build_incident()

    action = build_action(
        tool_name="service.reconcile",
    )

    worker_a_task = asyncio.create_task(
        runtime_a.execute(
            incident=incident,
            action=action,
        )
    )

    await handler_started.wait()

    with pytest.raises(
        ToolExecutionInProgressError,
        match="owned by another worker",
    ):
        await runtime_b.execute(
            incident=incident,
            action=action,
        )

    assert worker_b_executions == 0

    release_handler.set()

    result = await worker_a_task

    assert result == "payment-api"

    key = (
        SQLiteExecutionJournal.build_idempotency_key(
            incident_id=incident.id,
            action_id=action.id,
            tool_name=action.tool_name,
        )
    )

    record = journal.load(
        key
    )

    assert (
        record.status
        is ExecutionStatus.SUCCEEDED
    )

    assert record.attempt_count == 1


@pytest.mark.asyncio
async def test_completed_operation_is_not_executed_twice(
    tmp_path: Path,
) -> None:
    """A second process must deduplicate a completed side effect."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    runtime_a, _ = build_runtime(
        journal=journal,
        worker_id="worker-a",
    )

    runtime_b, _ = build_runtime(
        journal=journal,
        worker_id="worker-b",
    )

    executions: list[str] = []

    async def handler(
        arguments: dict[str, object],
    ) -> str:
        executions.append(
            str(
                arguments["service"]
            )
        )

        return "completed"

    runtime_a.register(
        name="inventory.upsert",
        handler=handler,
        idempotent=True,
    )

    runtime_b.register(
        name="inventory.upsert",
        handler=handler,
        idempotent=True,
    )

    incident = build_incident()

    action = build_action(
        tool_name="inventory.upsert",
        action_type=ActionType.WRITE,
    )

    first_result = await runtime_a.execute(
        incident=incident,
        action=action,
    )

    assert first_result == "completed"

    with pytest.raises(
        ToolExecutionAlreadyCompletedError,
        match="already completed",
    ):
        await runtime_b.execute(
            incident=incident,
            action=action,
        )

    assert executions == [
        "payment-api",
    ]


@pytest.mark.asyncio
async def test_expired_idempotent_operation_is_reclaimed(
    tmp_path: Path,
) -> None:
    """A dead worker's idempotent operation may be safely reclaimed."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    incident = build_incident()

    action = build_action(
        tool_name="prometheus.query",
    )

    prepared = journal.prepare(
        incident_id=incident.id,
        action_id=action.id,
        tool_name=action.tool_name,
        idempotent=True,
    )

    first_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="dead-worker",
        lease_seconds=60.0,
    )

    assert (
        first_claim.status
        is ExecutionStatus.IN_PROGRESS
    )

    expire_lease(
        database=database,
        idempotency_key=prepared.idempotency_key,
    )

    runtime, _ = build_runtime(
        journal=journal,
        worker_id="replacement-worker",
    )

    executions = 0

    async def query(
        arguments: dict[str, object],
    ) -> str:
        nonlocal executions

        executions += 1

        return str(
            arguments["service"]
        )

    runtime.register(
        name="prometheus.query",
        handler=query,
        idempotent=True,
    )

    result = await runtime.execute(
        incident=incident,
        action=action,
    )

    assert result == "payment-api"
    assert executions == 1

    latest = journal.load(
        prepared.idempotency_key
    )

    assert (
        latest.status
        is ExecutionStatus.SUCCEEDED
    )

    assert latest.attempt_count == 2


@pytest.mark.asyncio
async def test_expired_non_idempotent_operation_blocks_replay(
    tmp_path: Path,
) -> None:
    """Ambiguous mutation after worker death must become UNCERTAIN."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    incident = build_incident()

    action = build_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.WRITE,
    )

    prepared = journal.prepare(
        incident_id=incident.id,
        action_id=action.id,
        tool_name=action.tool_name,
        idempotent=False,
    )

    journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="dead-worker",
        lease_seconds=60.0,
    )

    expire_lease(
        database=database,
        idempotency_key=prepared.idempotency_key,
    )

    runtime, _ = build_runtime(
        journal=journal,
        worker_id="replacement-worker",
    )

    executions = 0

    async def deploy(
        arguments: dict[str, object],
    ) -> str:
        nonlocal executions

        executions += 1

        return str(
            arguments["service"]
        )

    runtime.register(
        name="kubernetes.deploy",
        handler=deploy,
        idempotent=False,
    )

    with pytest.raises(
        ToolExecutionUncertainError,
        match="automatic replay is blocked",
    ):
        await runtime.execute(
            incident=incident,
            action=action,
        )

    assert executions == 0

    latest = journal.load(
        prepared.idempotency_key
    )

    assert (
        latest.status
        is ExecutionStatus.UNCERTAIN
    )

    assert (
        latest.error_type
        == "expired_non_idempotent_lease"
    )

    assert latest.attempt_count == 1


@pytest.mark.asyncio
async def test_retry_safe_failure_reuses_same_downstream_key(
    tmp_path: Path,
) -> None:
    """A later worker must reuse the original downstream dedupe token."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    runtime_a, _ = build_runtime(
        journal=journal,
        worker_id="worker-a",
    )

    runtime_b, _ = build_runtime(
        journal=journal,
        worker_id="worker-b",
    )

    received_keys: list[str] = []

    async def first_handler(
        arguments: dict[str, object],
    ) -> None:
        received_keys.append(
            str(
                arguments[
                    "idempotency_key"
                ]
            )
        )

        raise ConnectionError(
            "confirmed transient failure"
        )

    async def second_handler(
        arguments: dict[str, object],
    ) -> str:
        received_keys.append(
            str(
                arguments[
                    "idempotency_key"
                ]
            )
        )

        return "success"

    runtime_a.register(
        name="payments.upsert",
        handler=first_handler,
        idempotent=True,
        idempotency_argument="idempotency_key",
        retry_policy=RetryPolicy(
            max_attempts=1,
        ),
    )

    runtime_b.register(
        name="payments.upsert",
        handler=second_handler,
        idempotent=True,
        idempotency_argument="idempotency_key",
        retry_policy=RetryPolicy(
            max_attempts=1,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="payments.upsert",
        action_type=ActionType.WRITE,
    )

    expected_key = (
        SQLiteExecutionJournal.build_idempotency_key(
            incident_id=incident.id,
            action_id=action.id,
            tool_name=action.tool_name,
        )
    )

    with pytest.raises(
        ToolExecutionError,
    ):
        await runtime_a.execute(
            incident=incident,
            action=action,
        )

    after_failure = journal.load(
        expected_key
    )

    assert (
        after_failure.status
        is ExecutionStatus.FAILED
    )

    result = await runtime_b.execute(
        incident=incident,
        action=action,
    )

    assert result == "success"

    assert received_keys == [
        expected_key,
        expected_key,
    ]

    completed = journal.load(
        expected_key
    )

    assert (
        completed.status
        is ExecutionStatus.SUCCEEDED
    )

    assert completed.attempt_count == 2


@pytest.mark.asyncio
async def test_ambiguous_mutation_failure_remains_blocked(
    tmp_path: Path,
) -> None:
    """A failed non-idempotent mutation must not be silently replayed."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    runtime_a, _ = build_runtime(
        journal=journal,
        worker_id="worker-a",
    )

    runtime_b, _ = build_runtime(
        journal=journal,
        worker_id="worker-b",
    )

    executions = 0

    async def first_handler(
        arguments: dict[str, object],
    ) -> None:
        nonlocal executions

        executions += 1

        raise ConnectionError(
            "connection lost after request submission"
        )

    async def second_handler(
        arguments: dict[str, object],
    ) -> str:
        nonlocal executions

        executions += 1

        return str(
            arguments["service"]
        )

    runtime_a.register(
        name="kubernetes.deploy",
        handler=first_handler,
        idempotent=False,
        retry_policy=RetryPolicy(
            max_attempts=1,
        ),
    )

    runtime_b.register(
        name="kubernetes.deploy",
        handler=second_handler,
        idempotent=False,
        retry_policy=RetryPolicy(
            max_attempts=1,
        ),
    )

    incident = build_incident()

    action = build_action(
        tool_name="kubernetes.deploy",
        action_type=ActionType.WRITE,
    )

    key = (
        SQLiteExecutionJournal.build_idempotency_key(
            incident_id=incident.id,
            action_id=action.id,
            tool_name=action.tool_name,
        )
    )

    with pytest.raises(
        ToolExecutionError,
    ):
        await runtime_a.execute(
            incident=incident,
            action=action,
        )

    uncertain = journal.load(
        key
    )

    assert (
        uncertain.status
        is ExecutionStatus.UNCERTAIN
    )

    assert executions == 1

    with pytest.raises(
        ToolExecutionUncertainError,
        match="automatic replay is blocked",
    ):
        await runtime_b.execute(
            incident=incident,
            action=action,
        )

    assert executions == 1