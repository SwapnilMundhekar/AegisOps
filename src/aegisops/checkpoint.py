"""Durable, versioned checkpoint persistence for AegisOps."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from aegisops.domain import IncidentState


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class CheckpointError(RuntimeError):
    """Base error raised by the checkpoint store."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when an incident checkpoint does not exist."""


class CheckpointConflictError(CheckpointError):
    """Raised when optimistic concurrency validation fails."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when persisted state fails integrity verification."""


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Immutable representation of a persisted incident checkpoint."""

    incident_id: UUID
    version: int
    state: IncidentState
    state_hash: str
    created_at: datetime


class SQLiteCheckpointStore:
    """Persist versioned incident state in a durable SQLite database."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self._database_path = Path(database_path)

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the configured checkpoint database path."""
        return self._database_path

    def save(
        self,
        incident: IncidentState,
        *,
        expected_version: int | None = None,
    ) -> CheckpointSnapshot:
        """Persist a new immutable version of incident state.

        When expected_version is provided, the write succeeds only if
        the current stored version exactly matches it.
        """

        state_json = self._serialize_state(incident)
        state_hash = self._calculate_hash(state_json)
        created_at = utc_now()

        connection = self._connect()

        try:
            connection.execute("BEGIN IMMEDIATE")

            current_version = self._get_current_version(
                connection=connection,
                incident_id=incident.id,
            )

            if (
                expected_version is not None
                and current_version != expected_version
            ):
                connection.execute("ROLLBACK")

                raise CheckpointConflictError(
                    f"Checkpoint conflict for incident {incident.id}: "
                    f"expected version {expected_version}, "
                    f"found {current_version}."
                )

            new_version = current_version + 1

            connection.execute(
                """
                INSERT INTO incident_checkpoints (
                    incident_id,
                    version,
                    state_json,
                    state_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(incident.id),
                    new_version,
                    state_json,
                    state_hash,
                    created_at.isoformat(),
                ),
            )

            connection.execute("COMMIT")

        except CheckpointConflictError:
            raise

        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass

            raise

        finally:
            connection.close()

        return CheckpointSnapshot(
            incident_id=incident.id,
            version=new_version,
            state=incident.model_copy(deep=True),
            state_hash=state_hash,
            created_at=created_at,
        )

    def load_latest(
        self,
        incident_id: UUID | str,
    ) -> CheckpointSnapshot:
        """Load and verify the latest checkpoint for an incident."""

        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    incident_id,
                    version,
                    state_json,
                    state_hash,
                    created_at
                FROM incident_checkpoints
                WHERE incident_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (str(incident_id),),
            ).fetchone()

        finally:
            connection.close()

        if row is None:
            raise CheckpointNotFoundError(
                f"No checkpoint exists for incident {incident_id}."
            )

        return self._row_to_snapshot(row)

    def load_version(
        self,
        incident_id: UUID | str,
        version: int,
    ) -> CheckpointSnapshot:
        """Load and verify a specific incident version."""

        if version < 1:
            raise ValueError(
                "Checkpoint version must be at least 1."
            )

        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    incident_id,
                    version,
                    state_json,
                    state_hash,
                    created_at
                FROM incident_checkpoints
                WHERE incident_id = ?
                  AND version = ?
                LIMIT 1
                """,
                (
                    str(incident_id),
                    version,
                ),
            ).fetchone()

        finally:
            connection.close()

        if row is None:
            raise CheckpointNotFoundError(
                f"Checkpoint version {version} does not exist "
                f"for incident {incident_id}."
            )

        return self._row_to_snapshot(row)

    def list_versions(
        self,
        incident_id: UUID | str,
    ) -> tuple[int, ...]:
        """Return all persisted versions for an incident."""

        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT version
                FROM incident_checkpoints
                WHERE incident_id = ?
                ORDER BY version ASC
                """,
                (str(incident_id),),
            ).fetchall()

        finally:
            connection.close()

        return tuple(
            int(row["version"])
            for row in rows
        )

    def current_version(
        self,
        incident_id: UUID | str,
    ) -> int:
        """Return the current version or zero when none exists."""

        connection = self._connect()

        try:
            return self._get_current_version(
                connection=connection,
                incident_id=incident_id,
            )

        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """Create checkpoint storage and indexes when required."""

        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_checkpoints (
                    incident_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (incident_id, version)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_incident_checkpoints_latest
                ON incident_checkpoints (
                    incident_id,
                    version DESC
                )
                """
            )

        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for durable workflows."""

        connection = sqlite3.connect(
            self._database_path,
            timeout=5.0,
            isolation_level=None,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        connection.execute(
            "PRAGMA foreign_keys=ON"
        )

        connection.execute(
            "PRAGMA synchronous=FULL"
        )

        return connection

    @staticmethod
    def _get_current_version(
        *,
        connection: sqlite3.Connection,
        incident_id: UUID | str,
    ) -> int:
        """Return the latest stored version inside a transaction."""

        row = connection.execute(
            """
            SELECT COALESCE(MAX(version), 0) AS version
            FROM incident_checkpoints
            WHERE incident_id = ?
            """,
            (str(incident_id),),
        ).fetchone()

        if row is None:
            return 0

        return int(row["version"])

    @staticmethod
    def _serialize_state(
        incident: IncidentState,
    ) -> str:
        """Serialize incident state into deterministic canonical JSON."""

        return json.dumps(
            incident.model_dump(
                mode="json",
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _calculate_hash(
        state_json: str,
    ) -> str:
        """Calculate the SHA-256 integrity hash for serialized state."""

        return hashlib.sha256(
            state_json.encode("utf-8")
        ).hexdigest()

    def _row_to_snapshot(
        self,
        row: sqlite3.Row,
    ) -> CheckpointSnapshot:
        """Validate integrity and deserialize a database row."""

        state_json = str(
            row["state_json"]
        )

        stored_hash = str(
            row["state_hash"]
        )

        calculated_hash = self._calculate_hash(
            state_json
        )

        if calculated_hash != stored_hash:
            raise CheckpointIntegrityError(
                "Checkpoint integrity verification failed for "
                f"incident {row['incident_id']} "
                f"version {row['version']}."
            )

        try:
            incident = IncidentState.model_validate_json(
                state_json
            )

        except Exception as exc:
            raise CheckpointIntegrityError(
                "Checkpoint payload could not be reconstructed "
                f"for incident {row['incident_id']} "
                f"version {row['version']}."
            ) from exc

        return CheckpointSnapshot(
            incident_id=UUID(
                str(row["incident_id"])
            ),
            version=int(
                row["version"]
            ),
            state=incident,
            state_hash=stored_hash,
            created_at=datetime.fromisoformat(
                str(row["created_at"])
            ),
        )