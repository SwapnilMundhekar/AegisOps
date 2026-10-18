"""Tests for durable AegisOps execution-journal semantics."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from aegisops.execution_journal import (
    ExecutionAlreadyCompletedError,
    ExecutionLeaseHeldError,
    ExecutionLeaseLostError,
    ExecutionOutcomeUnknownError,
    ExecutionStatus,
    SQLiteExecutionJournal,
)


def expire_lease(
    *,
    database: Path,
    idempotency_key: str,
) -> None:
    """Force a journal lease into the past for crash-recovery tests."""

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


def test_idempotency_key_is_stable_for_same_operation() -> None:
    """The same durable action identity must always produce the same key."""

    incident_id = uuid4()
    action_id = uuid4()

    first = SQLiteExecutionJournal.build_idempotency_key(
        incident_id=incident_id,
        action_id=action_id,
        tool_name="Kubernetes.Deploy",
    )

    second = SQLiteExecutionJournal.build_idempotency_key(
        incident_id=incident_id,
        action_id=action_id,
        tool_name="kubernetes.deploy",
    )

    assert first == second

    different_action = (
        SQLiteExecutionJournal.build_idempotency_key(
            incident_id=incident_id,
            action_id=uuid4(),
            tool_name="kubernetes.deploy",
        )
    )

    assert different_action != first


def test_prepare_is_idempotent(
    tmp_path: Path,
) -> None:
    """Repeated preparation must return one durable operation identity."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    incident_id = uuid4()
    action_id = uuid4()

    first = journal.prepare(
        incident_id=incident_id,
        action_id=action_id,
        tool_name="prometheus.query",
        idempotent=True,
    )

    second = journal.prepare(
        incident_id=incident_id,
        action_id=action_id,
        tool_name="prometheus.query",
        idempotent=True,
    )

    assert (
        first.idempotency_key
        == second.idempotency_key
    )

    assert first.status is ExecutionStatus.PREPARED
    assert second.status is ExecutionStatus.PREPARED

    assert first.attempt_count == 0
    assert second.attempt_count == 0


def test_live_lease_prevents_second_worker_claim(
    tmp_path: Path,
) -> None:
    """Only one worker may own an active execution lease."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="prometheus.query",
        idempotent=True,
    )

    claimed = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
        lease_seconds=60.0,
    )

    assert claimed.status is ExecutionStatus.IN_PROGRESS
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_token is not None
    assert claimed.attempt_count == 1

    with pytest.raises(
        ExecutionLeaseHeldError,
        match="currently leased",
    ):
        journal.claim(
            idempotency_key=prepared.idempotency_key,
            worker_id="worker-b",
            lease_seconds=60.0,
        )

    latest = journal.load(
        prepared.idempotency_key
    )

    assert latest.lease_owner == "worker-a"
    assert latest.attempt_count == 1


def test_successful_operation_cannot_be_claimed_again(
    tmp_path: Path,
) -> None:
    """Completed operations must deduplicate later execution attempts."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="github.read",
        idempotent=True,
    )

    claimed = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
    )

    assert claimed.lease_token is not None

    completed = journal.record_success(
        idempotency_key=prepared.idempotency_key,
        lease_token=claimed.lease_token,
        result_digest="sha256:result-001",
    )

    assert completed.status is ExecutionStatus.SUCCEEDED

    assert (
        completed.result_digest
        == "sha256:result-001"
    )

    assert completed.lease_owner is None
    assert completed.lease_token is None

    with pytest.raises(
        ExecutionAlreadyCompletedError,
        match="already succeeded",
    ):
        journal.claim(
            idempotency_key=prepared.idempotency_key,
            worker_id="worker-b",
        )


def test_stale_lease_token_cannot_finalize_execution(
    tmp_path: Path,
) -> None:
    """Only the current lease owner may finalize an external operation."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="prometheus.query",
        idempotent=True,
    )

    claimed = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
    )

    assert claimed.lease_token is not None

    with pytest.raises(
        ExecutionLeaseLostError,
        match="token does not match",
    ):
        journal.record_success(
            idempotency_key=prepared.idempotency_key,
            lease_token="stale-or-forged-token",
        )

    latest = journal.load(
        prepared.idempotency_key
    )

    assert latest.status is ExecutionStatus.IN_PROGRESS

    assert (
        latest.lease_token
        == claimed.lease_token
    )


def test_expired_idempotent_operation_can_be_reclaimed(
    tmp_path: Path,
) -> None:
    """A dead worker's idempotent operation may safely be attempted again."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="prometheus.query",
        idempotent=True,
    )

    first_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
        lease_seconds=60.0,
    )

    assert first_claim.lease_token is not None

    expire_lease(
        database=database,
        idempotency_key=prepared.idempotency_key,
    )

    second_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-b",
        lease_seconds=60.0,
    )

    assert second_claim.status is ExecutionStatus.IN_PROGRESS

    assert second_claim.lease_owner == "worker-b"

    assert (
        second_claim.lease_token
        != first_claim.lease_token
    )

    assert second_claim.attempt_count == 2


def test_expired_non_idempotent_operation_becomes_uncertain(
    tmp_path: Path,
) -> None:
    """A crashed mutation must not be replayed when outcome is ambiguous."""

    database = tmp_path / "journal.db"

    journal = SQLiteExecutionJournal(
        database
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="kubernetes.deploy",
        idempotent=False,
    )

    first_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
        lease_seconds=60.0,
    )

    assert first_claim.status is ExecutionStatus.IN_PROGRESS

    expire_lease(
        database=database,
        idempotency_key=prepared.idempotency_key,
    )

    with pytest.raises(
        ExecutionOutcomeUnknownError,
        match="automatic retry is unsafe",
    ):
        journal.claim(
            idempotency_key=prepared.idempotency_key,
            worker_id="worker-b",
        )

    latest = journal.load(
        prepared.idempotency_key
    )

    assert latest.status is ExecutionStatus.UNCERTAIN

    assert (
        latest.error_type
        == "expired_non_idempotent_lease"
    )

    assert latest.lease_owner is None
    assert latest.lease_token is None

    with pytest.raises(
        ExecutionOutcomeUnknownError,
        match="requires reconciliation",
    ):
        journal.claim(
            idempotency_key=prepared.idempotency_key,
            worker_id="worker-c",
        )


def test_explicit_ambiguous_failure_becomes_uncertain(
    tmp_path: Path,
) -> None:
    """An ambiguous external failure must fail closed."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="payments.charge",
        idempotent=False,
    )

    claimed = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
    )

    assert claimed.lease_token is not None

    uncertain = journal.record_failure(
        idempotency_key=prepared.idempotency_key,
        lease_token=claimed.lease_token,
        error_type="connection_lost_after_submit",
        retry_safe=False,
    )

    assert uncertain.status is ExecutionStatus.UNCERTAIN

    assert (
        uncertain.error_type
        == "connection_lost_after_submit"
    )

    with pytest.raises(
        ExecutionOutcomeUnknownError,
    ):
        journal.claim(
            idempotency_key=prepared.idempotency_key,
            worker_id="worker-b",
        )


def test_retry_safe_failure_can_be_claimed_again(
    tmp_path: Path,
) -> None:
    """A confirmed no-side-effect failure may be safely retried."""

    journal = SQLiteExecutionJournal(
        tmp_path / "journal.db"
    )

    prepared = journal.prepare(
        incident_id=uuid4(),
        action_id=uuid4(),
        tool_name="prometheus.query",
        idempotent=True,
    )

    first_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-a",
    )

    assert first_claim.lease_token is not None

    failed = journal.record_failure(
        idempotency_key=prepared.idempotency_key,
        lease_token=first_claim.lease_token,
        error_type="connection_refused",
        retry_safe=True,
    )

    assert failed.status is ExecutionStatus.FAILED

    second_claim = journal.claim(
        idempotency_key=prepared.idempotency_key,
        worker_id="worker-b",
    )

    assert second_claim.status is ExecutionStatus.IN_PROGRESS
    assert second_claim.lease_owner == "worker-b"
    assert second_claim.attempt_count == 2

    assert (
        second_claim.lease_token
        != first_claim.lease_token
    )