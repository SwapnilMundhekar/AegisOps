"""Durable execution journal and idempotency controls for AegisOps."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ExecutionJournalError(RuntimeError):
    """Base error for durable execution-journal operations."""


class ExecutionNotFoundError(ExecutionJournalError):
    """Raised when an execution record does not exist."""


class ExecutionLeaseHeldError(ExecutionJournalError):
    """Raised when another worker currently owns a live lease."""


class ExecutionLeaseLostError(ExecutionJournalError):
    """Raised when a worker no longer owns the execution lease."""


class ExecutionAlreadyCompletedError(ExecutionJournalError):
    """Raised when an already-completed operation is claimed again."""


class ExecutionOutcomeUnknownError(ExecutionJournalError):
    """Raised when repeating an external side effect may be unsafe."""


class ExecutionIdentityConflictError(ExecutionJournalError):
    """Raised when an idempotency key is reused for another operation."""


class ExecutionStatus(StrEnum):
    """Durable lifecycle of one external side-effect operation."""

    PREPARED = "prepared"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Immutable view of one journaled external operation."""

    idempotency_key: str

    incident_id: UUID
    action_id: UUID

    tool_name: str
    idempotent: bool

    status: ExecutionStatus

    attempt_count: int

    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None

    result_digest: str | None
    error_type: str | None

    created_at: datetime
    updated_at: datetime


class SQLiteExecutionJournal:
    """SQLite-backed durable journal for external side effects."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self._database_path = Path(
            database_path
        )

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the configured journal database path."""
        return self._database_path

    @staticmethod
    def build_idempotency_key(
        *,
        incident_id: UUID | str,
        action_id: UUID | str,
        tool_name: str,
    ) -> str:
        """Build a stable operation key from durable business identity.

        The same action restored after a process restart receives the
        same key rather than generating a new retry identity.
        """

        normalized_tool = tool_name.strip().casefold()

        if not normalized_tool:
            raise ValueError(
                "tool_name must not be empty."
            )

        canonical = (
            f"{incident_id}:"
            f"{action_id}:"
            f"{normalized_tool}"
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def prepare(
        self,
        *,
        incident_id: UUID,
        action_id: UUID,
        tool_name: str,
        idempotent: bool,
    ) -> ExecutionRecord:
        """Create a durable execution intent exactly once.

        Repeating prepare() for the same operation is safe and returns
        the existing record. Reusing the key for a different operation
        fails closed.
        """

        normalized_tool = tool_name.strip().casefold()

        if not normalized_tool:
            raise ValueError(
                "tool_name must not be empty."
            )

        key = self.build_idempotency_key(
            incident_id=incident_id,
            action_id=action_id,
            tool_name=normalized_tool,
        )

        now = utc_now()

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            existing = self._load_by_key(
                connection=connection,
                idempotency_key=key,
            )

            if existing is not None:
                self._validate_identity(
                    existing=existing,
                    incident_id=incident_id,
                    action_id=action_id,
                    tool_name=normalized_tool,
                    idempotent=idempotent,
                )

                connection.execute(
                    "COMMIT"
                )

                return existing

            connection.execute(
                """
                INSERT INTO execution_journal (
                    idempotency_key,
                    incident_id,
                    action_id,
                    tool_name,
                    idempotent,
                    status,
                    attempt_count,
                    lease_owner,
                    lease_token,
                    lease_expires_at,
                    result_digest,
                    error_type,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (
                    key,
                    str(incident_id),
                    str(action_id),
                    normalized_tool,
                    int(idempotent),
                    ExecutionStatus.PREPARED.value,
                    0,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            self._rollback_safely(
                connection
            )
            raise

        finally:
            connection.close()

        return self.load(
            key
        )

    def claim(
        self,
        *,
        idempotency_key: str,
        worker_id: str,
        lease_seconds: float = 30.0,
    ) -> ExecutionRecord:
        """Atomically claim an operation for external execution.

        An expired idempotent operation may be reclaimed.

        An expired non-idempotent IN_PROGRESS operation is moved to
        UNCERTAIN instead of being executed again automatically.
        """

        if not worker_id.strip():
            raise ValueError(
                "worker_id must not be empty."
            )

        if lease_seconds <= 0:
            raise ValueError(
                "lease_seconds must be greater than zero."
            )

        now = utc_now()

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            record = self._require_record(
                connection=connection,
                idempotency_key=idempotency_key,
            )

            if record.status is ExecutionStatus.SUCCEEDED:
                raise ExecutionAlreadyCompletedError(
                    "Execution already succeeded: "
                    f"{idempotency_key}"
                )

            if record.status is ExecutionStatus.UNCERTAIN:
                raise ExecutionOutcomeUnknownError(
                    "Execution outcome is uncertain and requires "
                    f"reconciliation: {idempotency_key}"
                )

            if record.status is ExecutionStatus.IN_PROGRESS:
                if (
                    record.lease_expires_at is not None
                    and record.lease_expires_at > now
                ):
                    raise ExecutionLeaseHeldError(
                        "Execution is currently leased by "
                        f"{record.lease_owner}."
                    )

                if not record.idempotent:
                    self._mark_uncertain_in_transaction(
                        connection=connection,
                        record=record,
                        now=now,
                        error_type="expired_non_idempotent_lease",
                    )

                    connection.execute(
                        "COMMIT"
                    )

                    raise ExecutionOutcomeUnknownError(
                        "Non-idempotent operation had an expired "
                        "in-progress lease; automatic retry is unsafe: "
                        f"{idempotency_key}"
                    )

            lease_token = str(
                uuid4()
            )

            lease_expires_at = (
                now
                + timedelta(
                    seconds=lease_seconds
                )
            )

            connection.execute(
                """
                UPDATE execution_journal
                SET
                    status = ?,
                    attempt_count = attempt_count + 1,
                    lease_owner = ?,
                    lease_token = ?,
                    lease_expires_at = ?,
                    error_type = NULL,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    ExecutionStatus.IN_PROGRESS.value,
                    worker_id.strip(),
                    lease_token,
                    lease_expires_at.isoformat(),
                    now.isoformat(),
                    idempotency_key,
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except (
            ExecutionAlreadyCompletedError,
            ExecutionOutcomeUnknownError,
            ExecutionLeaseHeldError,
        ):
            self._rollback_safely(
                connection
            )
            raise

        except Exception:
            self._rollback_safely(
                connection
            )
            raise

        finally:
            connection.close()

        return self.load(
            idempotency_key
        )

    def heartbeat(
        self,
        *,
        idempotency_key: str,
        lease_token: str,
        lease_seconds: float = 30.0,
    ) -> ExecutionRecord:
        """Extend a live execution lease owned by this worker."""

        if lease_seconds <= 0:
            raise ValueError(
                "lease_seconds must be greater than zero."
            )

        now = utc_now()

        new_expiry = (
            now
            + timedelta(
                seconds=lease_seconds
            )
        )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            record = self._require_record(
                connection=connection,
                idempotency_key=idempotency_key,
            )

            self._validate_active_lease(
                record=record,
                lease_token=lease_token,
            )

            connection.execute(
                """
                UPDATE execution_journal
                SET
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    new_expiry.isoformat(),
                    now.isoformat(),
                    idempotency_key,
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            self._rollback_safely(
                connection
            )
            raise

        finally:
            connection.close()

        return self.load(
            idempotency_key
        )

    def record_success(
        self,
        *,
        idempotency_key: str,
        lease_token: str,
        result_digest: str | None = None,
    ) -> ExecutionRecord:
        """Durably record successful external execution."""

        now = utc_now()

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            record = self._require_record(
                connection=connection,
                idempotency_key=idempotency_key,
            )

            self._validate_active_lease(
                record=record,
                lease_token=lease_token,
            )

            connection.execute(
                """
                UPDATE execution_journal
                SET
                    status = ?,
                    lease_owner = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    result_digest = ?,
                    error_type = NULL,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    ExecutionStatus.SUCCEEDED.value,
                    result_digest,
                    now.isoformat(),
                    idempotency_key,
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            self._rollback_safely(
                connection
            )
            raise

        finally:
            connection.close()

        return self.load(
            idempotency_key
        )

    def record_failure(
        self,
        *,
        idempotency_key: str,
        lease_token: str,
        error_type: str,
        retry_safe: bool,
    ) -> ExecutionRecord:
        """Record an execution failure.

        retry_safe=True means the caller knows no ambiguous external
        side effect occurred, so another worker may claim it again.

        retry_safe=False records UNCERTAIN, deliberately preventing an
        automatic repeat of the side effect.
        """

        if not error_type.strip():
            raise ValueError(
                "error_type must not be empty."
            )

        now = utc_now()

        target_status = (
            ExecutionStatus.FAILED
            if retry_safe
            else ExecutionStatus.UNCERTAIN
        )

        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            record = self._require_record(
                connection=connection,
                idempotency_key=idempotency_key,
            )

            self._validate_active_lease(
                record=record,
                lease_token=lease_token,
            )

            connection.execute(
                """
                UPDATE execution_journal
                SET
                    status = ?,
                    lease_owner = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    error_type = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    target_status.value,
                    error_type.strip(),
                    now.isoformat(),
                    idempotency_key,
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            self._rollback_safely(
                connection
            )
            raise

        finally:
            connection.close()

        return self.load(
            idempotency_key
        )

    def load(
        self,
        idempotency_key: str,
    ) -> ExecutionRecord:
        """Load one execution record."""

        connection = self._connect()

        try:
            record = self._load_by_key(
                connection=connection,
                idempotency_key=idempotency_key,
            )

        finally:
            connection.close()

        if record is None:
            raise ExecutionNotFoundError(
                "Execution record does not exist: "
                f"{idempotency_key}"
            )

        return record

    def _initialize_database(
        self,
    ) -> None:
        """Create durable execution-journal storage."""

        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_journal (
                    idempotency_key TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    idempotent INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    result_digest TEXT,
                    error_type TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_execution_action_tool
                ON execution_journal (
                    incident_id,
                    action_id,
                    tool_name
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_execution_status
                ON execution_journal (
                    status,
                    updated_at
                )
                """
            )

        finally:
            connection.close()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """Open SQLite with durable write settings."""

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
    def _validate_identity(
        *,
        existing: ExecutionRecord,
        incident_id: UUID,
        action_id: UUID,
        tool_name: str,
        idempotent: bool,
    ) -> None:
        """Prevent an idempotency key from changing operation meaning."""

        if (
            existing.incident_id != incident_id
            or existing.action_id != action_id
            or existing.tool_name != tool_name
            or existing.idempotent is not idempotent
        ):
            raise ExecutionIdentityConflictError(
                "Idempotency key identity mismatch."
            )

    @staticmethod
    def _validate_active_lease(
        *,
        record: ExecutionRecord,
        lease_token: str,
    ) -> None:
        """Require the exact active lease before finalizing execution."""

        if record.status is not ExecutionStatus.IN_PROGRESS:
            raise ExecutionLeaseLostError(
                "Execution is not currently in progress."
            )

        if (
            record.lease_token is None
            or record.lease_token != lease_token
        ):
            raise ExecutionLeaseLostError(
                "Execution lease token does not match."
            )

        if (
            record.lease_expires_at is None
            or record.lease_expires_at <= utc_now()
        ):
            raise ExecutionLeaseLostError(
                "Execution lease has expired."
            )

    def _require_record(
        self,
        *,
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> ExecutionRecord:
        """Load a record inside an existing transaction."""

        record = self._load_by_key(
            connection=connection,
            idempotency_key=idempotency_key,
        )

        if record is None:
            raise ExecutionNotFoundError(
                "Execution record does not exist: "
                f"{idempotency_key}"
            )

        return record

    @staticmethod
    def _mark_uncertain_in_transaction(
        *,
        connection: sqlite3.Connection,
        record: ExecutionRecord,
        now: datetime,
        error_type: str,
    ) -> None:
        """Move an ambiguous operation into fail-closed state."""

        connection.execute(
            """
            UPDATE execution_journal
            SET
                status = ?,
                lease_owner = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                error_type = ?,
                updated_at = ?
            WHERE idempotency_key = ?
            """,
            (
                ExecutionStatus.UNCERTAIN.value,
                error_type,
                now.isoformat(),
                record.idempotency_key,
            ),
        )

    @staticmethod
    def _load_by_key(
        *,
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> ExecutionRecord | None:
        """Load one journal row from an existing connection."""

        row = connection.execute(
            """
            SELECT
                idempotency_key,
                incident_id,
                action_id,
                tool_name,
                idempotent,
                status,
                attempt_count,
                lease_owner,
                lease_token,
                lease_expires_at,
                result_digest,
                error_type,
                created_at,
                updated_at
            FROM execution_journal
            WHERE idempotency_key = ?
            LIMIT 1
            """,
            (
                idempotency_key,
            ),
        ).fetchone()

        if row is None:
            return None

        lease_expires_at = (
            datetime.fromisoformat(
                str(
                    row["lease_expires_at"]
                )
            )
            if row["lease_expires_at"] is not None
            else None
        )

        return ExecutionRecord(
            idempotency_key=str(
                row["idempotency_key"]
            ),
            incident_id=UUID(
                str(
                    row["incident_id"]
                )
            ),
            action_id=UUID(
                str(
                    row["action_id"]
                )
            ),
            tool_name=str(
                row["tool_name"]
            ),
            idempotent=bool(
                row["idempotent"]
            ),
            status=ExecutionStatus(
                str(
                    row["status"]
                )
            ),
            attempt_count=int(
                row["attempt_count"]
            ),
            lease_owner=(
                str(
                    row["lease_owner"]
                )
                if row["lease_owner"] is not None
                else None
            ),
            lease_token=(
                str(
                    row["lease_token"]
                )
                if row["lease_token"] is not None
                else None
            ),
            lease_expires_at=lease_expires_at,
            result_digest=(
                str(
                    row["result_digest"]
                )
                if row["result_digest"] is not None
                else None
            ),
            error_type=(
                str(
                    row["error_type"]
                )
                if row["error_type"] is not None
                else None
            ),
            created_at=datetime.fromisoformat(
                str(
                    row["created_at"]
                )
            ),
            updated_at=datetime.fromisoformat(
                str(
                    row["updated_at"]
                )
            ),
        )

    @staticmethod
    def _rollback_safely(
        connection: sqlite3.Connection,
    ) -> None:
        """Rollback an active transaction without hiding original errors."""

        try:
            connection.execute(
                "ROLLBACK"
            )

        except sqlite3.Error:
            pass