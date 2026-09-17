"""Tamper-evident audit ledger for the AegisOps control plane."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class AuditEventType(StrEnum):
    """Security and operational events recorded by AegisOps."""

    INCIDENT_CREATED = "incident_created"
    INCIDENT_STATE_CHANGED = "incident_state_changed"
    EVIDENCE_ADDED = "evidence_added"
    ACTION_PROPOSED = "action_proposed"
    POLICY_DECISION = "policy_decision"
    APPROVAL_RECORDED = "approval_recorded"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    MODEL_INVOCATION = "model_invocation"
    SECURITY_EVENT = "security_event"


class AuditEvent(BaseModel):
    """Immutable event stored in the AegisOps audit chain."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: UUID = Field(default_factory=uuid4)
    sequence: int = Field(ge=1)

    event_type: AuditEventType
    actor: str
    subject_id: str

    payload: dict[str, object] = Field(default_factory=dict)

    recorded_at: datetime = Field(default_factory=utc_now)

    previous_hash: str = Field(
        min_length=64,
        max_length=64,
    )

    event_hash: str = Field(
        min_length=64,
        max_length=64,
    )


class AuditLedger:
    """Append-only, hash-chained audit ledger."""

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """Return an immutable view of all audit events."""
        return tuple(self._events)

    @property
    def head_hash(self) -> str:
        """Return the hash of the latest audit event."""

        if not self._events:
            return self.GENESIS_HASH

        return self._events[-1].event_hash

    def append(
        self,
        *,
        event_type: AuditEventType,
        actor: str,
        subject_id: str,
        payload: dict[str, object] | None = None,
    ) -> AuditEvent:
        """Append a new event to the hash chain."""

        event_id = uuid4()
        sequence = len(self._events) + 1
        recorded_at = utc_now()
        previous_hash = self.head_hash
        event_payload = payload or {}

        hash_input = {
            "id": str(event_id),
            "sequence": sequence,
            "event_type": event_type.value,
            "actor": actor,
            "subject_id": subject_id,
            "payload": event_payload,
            "recorded_at": recorded_at.isoformat(),
            "previous_hash": previous_hash,
        }

        event_hash = self._calculate_hash(hash_input)

        event = AuditEvent(
            id=event_id,
            sequence=sequence,
            event_type=event_type,
            actor=actor,
            subject_id=subject_id,
            payload=event_payload,
            recorded_at=recorded_at,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

        self._events.append(event)

        return event

    def verify_integrity(self) -> bool:
        """Verify sequence numbers, hashes, and the complete event chain."""

        expected_previous_hash = self.GENESIS_HASH

        for expected_sequence, event in enumerate(
            self._events,
            start=1,
        ):
            if event.sequence != expected_sequence:
                return False

            if event.previous_hash != expected_previous_hash:
                return False

            hash_input = {
                "id": str(event.id),
                "sequence": event.sequence,
                "event_type": event.event_type.value,
                "actor": event.actor,
                "subject_id": event.subject_id,
                "payload": event.payload,
                "recorded_at": event.recorded_at.isoformat(),
                "previous_hash": event.previous_hash,
            }

            expected_hash = self._calculate_hash(hash_input)

            if event.event_hash != expected_hash:
                return False

            expected_previous_hash = event.event_hash

        return True

    @staticmethod
    def _calculate_hash(data: dict[str, object]) -> str:
        """Create a deterministic SHA-256 hash for an audit record."""

        canonical_data = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

        return hashlib.sha256(
            canonical_data.encode("utf-8")
        ).hexdigest()