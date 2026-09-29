"""Tests for durable AegisOps incident checkpoint persistence."""

from __future__ import annotations

import sqlite3

import pytest

from aegisops.checkpoint import (
    CheckpointConflictError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    SQLiteCheckpointStore,
)
from aegisops.domain import (
    Evidence,
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
)


def build_incident() -> IncidentState:
    """Create a representative incident for persistence tests."""

    return IncidentState(
        title="Payment API latency spike",
        summary="Elevated latency detected after deployment.",
        service="payment-api",
        environment="production",
        severity=IncidentSeverity.SEV2,
    )


def test_checkpoint_round_trip(
    tmp_path,
) -> None:
    """Persisted incident state should round-trip without data loss."""

    database = tmp_path / "aegisops.db"

    store = SQLiteCheckpointStore(
        database,
    )

    incident = build_incident()

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    incident.add_evidence(
        Evidence(
            source="prometheus",
            summary="P95 latency exceeded threshold.",
            confidence=0.97,
            reference="metric://payment-api/p95",
        )
    )

    saved = store.save(
        incident
    )

    restored = store.load_latest(
        incident.id
    )

    assert saved.version == 1
    assert restored.version == 1

    assert restored.incident_id == incident.id

    assert (
        restored.state.status
        is IncidentStatus.TRIAGING
    )

    assert restored.state.service == "payment-api"

    assert len(restored.state.evidence) == 1

    assert (
        restored.state.evidence[0].summary
        == "P95 latency exceeded threshold."
    )

    assert restored.state_hash == saved.state_hash


def test_multiple_versions_are_preserved(
    tmp_path,
) -> None:
    """Every checkpoint save should create immutable history."""

    store = SQLiteCheckpointStore(
        tmp_path / "aegisops.db"
    )

    incident = build_incident()

    first = store.save(
        incident
    )

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    second = store.save(
        incident,
        expected_version=first.version,
    )

    incident.transition_to(
        IncidentStatus.INVESTIGATING
    )

    third = store.save(
        incident,
        expected_version=second.version,
    )

    assert first.version == 1
    assert second.version == 2
    assert third.version == 3

    assert store.list_versions(
        incident.id
    ) == (
        1,
        2,
        3,
    )

    version_one = store.load_version(
        incident.id,
        1,
    )

    version_three = store.load_version(
        incident.id,
        3,
    )

    assert (
        version_one.state.status
        is IncidentStatus.DETECTED
    )

    assert (
        version_three.state.status
        is IncidentStatus.INVESTIGATING
    )


def test_checkpoint_survives_store_restart(
    tmp_path,
) -> None:
    """A new store instance should resume previously persisted state."""

    database = tmp_path / "aegisops.db"

    first_process = SQLiteCheckpointStore(
        database
    )

    incident = build_incident()

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    saved = first_process.save(
        incident
    )

    del first_process

    restarted_process = SQLiteCheckpointStore(
        database
    )

    restored = restarted_process.load_latest(
        incident.id
    )

    assert restored.version == saved.version
    assert restored.incident_id == incident.id

    assert (
        restored.state.status
        is IncidentStatus.TRIAGING
    )

    assert (
        restarted_process.current_version(
            incident.id
        )
        == 1
    )


def test_stale_writer_is_rejected(
    tmp_path,
) -> None:
    """Optimistic concurrency must prevent lost updates."""

    store = SQLiteCheckpointStore(
        tmp_path / "aegisops.db"
    )

    incident = build_incident()

    first = store.save(
        incident
    )

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    second = store.save(
        incident,
        expected_version=first.version,
    )

    assert second.version == 2

    stale_copy = first.state.model_copy(
        deep=True
    )

    stale_copy.summary = (
        "A stale worker attempted to overwrite newer state."
    )

    with pytest.raises(
        CheckpointConflictError,
        match="Checkpoint conflict",
    ):
        store.save(
            stale_copy,
            expected_version=first.version,
        )

    latest = store.load_latest(
        incident.id
    )

    assert latest.version == 2

    assert (
        latest.state.status
        is IncidentStatus.TRIAGING
    )


def test_checkpoint_tampering_is_detected(
    tmp_path,
) -> None:
    """Modified persisted state must fail hash verification."""

    database = tmp_path / "aegisops.db"

    store = SQLiteCheckpointStore(
        database
    )

    incident = build_incident()

    saved = store.save(
        incident
    )

    connection = sqlite3.connect(
        database
    )

    try:
        connection.execute(
            """
            UPDATE incident_checkpoints
            SET state_json = ?
            WHERE incident_id = ?
              AND version = ?
            """,
            (
                '{"tampered":true}',
                str(incident.id),
                saved.version,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    with pytest.raises(
        CheckpointIntegrityError,
        match="integrity verification failed",
    ):
        store.load_latest(
            incident.id
        )


def test_missing_checkpoint_raises_explicit_error(
    tmp_path,
) -> None:
    """Unknown incidents should fail explicitly rather than silently."""

    store = SQLiteCheckpointStore(
        tmp_path / "aegisops.db"
    )

    incident = build_incident()

    with pytest.raises(
        CheckpointNotFoundError,
        match="No checkpoint exists",
    ):
        store.load_latest(
            incident.id
        )


def test_snapshot_is_independent_from_live_state(
    tmp_path,
) -> None:
    """Saved snapshots must not mutate with the in-memory incident."""

    store = SQLiteCheckpointStore(
        tmp_path / "aegisops.db"
    )

    incident = build_incident()

    snapshot = store.save(
        incident
    )

    original_summary = snapshot.state.summary

    incident.summary = "Changed after checkpoint."

    incident.transition_to(
        IncidentStatus.TRIAGING
    )

    assert snapshot.state.summary == original_summary

    assert (
        snapshot.state.status
        is IncidentStatus.DETECTED
    )

    restored = store.load_latest(
        incident.id
    )

    assert restored.state.summary == original_summary

    assert (
        restored.state.status
        is IncidentStatus.DETECTED
    )