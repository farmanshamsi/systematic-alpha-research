"""Deterministic stream health and operational gating for Day 20."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Final

from systematic_alpha.broker.reconciliation import ReconciliationResult


STREAM_STALE_AFTER: Final[timedelta] = timedelta(seconds=30)
MAX_STREAM_FUTURE_SKEW: Final[timedelta] = timedelta(seconds=5)
RECONNECT_BACKOFF_SECONDS: Final[tuple[int, ...]] = (1, 2, 4)
MAX_RECONNECT_ATTEMPTS: Final[int] = 3
MONITOR_REASON_CODES: Final[tuple[str, ...]] = (
    "stream_stale",
    "reconnect_exhausted",
    "kill_switch_latched",
)


class StreamState(str, Enum):
    HEALTHY = "healthy"
    STALE = "stale"
    RECONNECTING = "reconnecting"
    CIRCUIT_OPEN = "circuit_open"
    KILLED = "killed"


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Identifier must be non-empty text without padding.")
    if any(character.isspace() for character in value):
        raise ValueError("Identifier cannot contain whitespace.")
    return value


@dataclass(frozen=True, slots=True)
class StreamAuditEntry:
    audit_sequence: int
    stream_id: str
    action: str
    previous_state: str
    resulting_state: str
    event_id: str
    occurred_at: datetime
    reconnect_attempt: int
    next_retry_at: datetime | None
    reason_code: str
    submission_blocked: bool


@dataclass(frozen=True, slots=True)
class OperationalDecision:
    reconciliation_passed: bool
    limits_passed: bool
    stream_state: StreamState
    stream_safe: bool
    circuit_breaker_open: bool
    kill_switch_latched: bool
    active_reason_codes: tuple[str, ...]
    operational_gate_passed: bool
    day20_order_submission_authorized: bool
    can_submit_orders: bool


class StreamHealthMonitor:
    """Explicit-clock stream monitor with terminal circuit and kill states."""

    def __init__(self, stream_id: str, *, started_at: datetime) -> None:
        self.stream_id = _identifier(stream_id)
        self.started_at = _utc(started_at)
        self._state = StreamState.STALE
        self._last_received_at: datetime | None = None
        self._last_event_at: datetime | None = None
        self._event_fingerprints: dict[str, tuple[object, ...]] = {}
        self._reconnect_attempts = 0
        self._next_retry_at: datetime | None = None
        self._active_reasons: set[str] = {"stream_stale"}
        self._audit_entries: list[StreamAuditEntry] = []

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def last_received_at(self) -> datetime | None:
        return self._last_received_at

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def next_retry_at(self) -> datetime | None:
        return self._next_retry_at

    @property
    def audit_entries(self) -> tuple[StreamAuditEntry, ...]:
        return tuple(self._audit_entries)

    @property
    def active_reason_codes(self) -> tuple[str, ...]:
        return tuple(
            reason for reason in MONITOR_REASON_CODES if reason in self._active_reasons
        )

    @property
    def terminal(self) -> bool:
        return self._state in {StreamState.CIRCUIT_OPEN, StreamState.KILLED}

    def _append(
        self,
        *,
        action: str,
        previous_state: StreamState,
        occurred_at: datetime,
        event_id: str = "",
        reason_code: str = "",
    ) -> StreamAuditEntry:
        entry = StreamAuditEntry(
            audit_sequence=len(self._audit_entries) + 1,
            stream_id=self.stream_id,
            action=action,
            previous_state=previous_state.value,
            resulting_state=self._state.value,
            event_id=event_id,
            occurred_at=occurred_at,
            reconnect_attempt=self._reconnect_attempts,
            next_retry_at=self._next_retry_at,
            reason_code=reason_code,
            submission_blocked=True,
        )
        self._audit_entries.append(entry)
        return entry

    def _require_not_terminal(self) -> None:
        if self.terminal:
            raise RuntimeError("Day 20 stream state is terminal.")

    def record_message(
        self,
        *,
        event_id: str,
        event_at: datetime,
        received_at: datetime,
    ) -> StreamAuditEntry | None:
        self._require_not_terminal()
        normalized_event_id = _identifier(event_id)
        normalized_event_at = _utc(event_at)
        normalized_received_at = _utc(received_at)
        if normalized_event_at - normalized_received_at > MAX_STREAM_FUTURE_SKEW:
            raise ValueError("Stream event exceeds future-skew tolerance.")
        if self._last_received_at is not None and (
            normalized_received_at < self._last_received_at
        ):
            raise ValueError("Stream receipt time regressed.")
        if self._last_event_at is not None and (
            normalized_event_at < self._last_event_at
        ):
            raise ValueError("Stream event time regressed.")
        fingerprint = (normalized_event_id, normalized_event_at.isoformat())
        prior = self._event_fingerprints.get(normalized_event_id)
        if prior is not None:
            if prior != fingerprint:
                raise ValueError("Conflicting stream event ID reuse.")
            return None
        self._event_fingerprints[normalized_event_id] = fingerprint
        previous = self._state
        self._state = StreamState.HEALTHY
        self._last_event_at = normalized_event_at
        self._last_received_at = normalized_received_at
        self._reconnect_attempts = 0
        self._next_retry_at = None
        self._active_reasons.discard("stream_stale")
        self._active_reasons.discard("reconnect_exhausted")
        return self._append(
            action="message_received",
            previous_state=previous,
            occurred_at=normalized_received_at,
            event_id=normalized_event_id,
        )

    def evaluate(self, *, as_of: datetime) -> StreamAuditEntry | None:
        normalized_as_of = _utc(as_of)
        reference = self._last_received_at or self.started_at
        if normalized_as_of < reference:
            raise ValueError("as_of cannot precede stream reference time.")
        if self.terminal:
            return None
        if normalized_as_of - reference <= STREAM_STALE_AFTER:
            return None
        if self._state is not StreamState.HEALTHY:
            self._active_reasons.add("stream_stale")
            return None
        previous = self._state
        self._state = StreamState.STALE
        self._active_reasons.add("stream_stale")
        return self._append(
            action="stream_stale",
            previous_state=previous,
            occurred_at=normalized_as_of,
            reason_code="stream_stale",
        )

    def begin_reconnect(self, *, at: datetime) -> StreamAuditEntry:
        self._require_not_terminal()
        normalized_at = _utc(at)
        if self._state is not StreamState.STALE:
            raise RuntimeError("Reconnect requires stale stream state.")
        previous = self._state
        self._state = StreamState.RECONNECTING
        self._reconnect_attempts = 0
        self._next_retry_at = normalized_at + timedelta(
            seconds=RECONNECT_BACKOFF_SECONDS[0]
        )
        return self._append(
            action="reconnect_scheduled",
            previous_state=previous,
            occurred_at=normalized_at,
            reason_code="stream_stale",
        )

    def record_reconnect_failure(self, *, at: datetime) -> StreamAuditEntry:
        self._require_not_terminal()
        normalized_at = _utc(at)
        if self._state is not StreamState.RECONNECTING:
            raise RuntimeError("Reconnect failure requires reconnecting state.")
        if self._next_retry_at is None or normalized_at < self._next_retry_at:
            raise ValueError("Reconnect attempt occurred before scheduled retry.")
        previous = self._state
        self._reconnect_attempts += 1
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            self._state = StreamState.CIRCUIT_OPEN
            self._next_retry_at = None
            self._active_reasons.add("reconnect_exhausted")
            return self._append(
                action="circuit_opened",
                previous_state=previous,
                occurred_at=normalized_at,
                reason_code="reconnect_exhausted",
            )
        delay = RECONNECT_BACKOFF_SECONDS[self._reconnect_attempts]
        self._next_retry_at = normalized_at + timedelta(seconds=delay)
        return self._append(
            action="reconnect_failed",
            previous_state=previous,
            occurred_at=normalized_at,
            reason_code="stream_stale",
        )

    def record_reconnect_success(self, *, at: datetime) -> StreamAuditEntry:
        self._require_not_terminal()
        normalized_at = _utc(at)
        if self._state is not StreamState.RECONNECTING:
            raise RuntimeError("Reconnect success requires reconnecting state.")
        if self._next_retry_at is None or normalized_at < self._next_retry_at:
            raise ValueError("Reconnect succeeded before scheduled retry.")
        previous = self._state
        self._state = StreamState.STALE
        self._next_retry_at = None
        self._active_reasons.add("stream_stale")
        return self._append(
            action="transport_reconnected_waiting_message",
            previous_state=previous,
            occurred_at=normalized_at,
            reason_code="stream_stale",
        )

    def engage_kill_switch(self, *, at: datetime) -> StreamAuditEntry | None:
        normalized_at = _utc(at)
        if self._state is StreamState.KILLED:
            return None
        previous = self._state
        self._state = StreamState.KILLED
        self._next_retry_at = None
        self._active_reasons.add("kill_switch_latched")
        return self._append(
            action="kill_switch_latched",
            previous_state=previous,
            occurred_at=normalized_at,
            reason_code="kill_switch_latched",
        )


def evaluate_operational_gate(
    reconciliation: ReconciliationResult,
    monitor: StreamHealthMonitor,
) -> OperationalDecision:
    """Combine reconciliation and stream state without authorizing orders."""

    if not isinstance(reconciliation, ReconciliationResult):
        raise TypeError("reconciliation must be ReconciliationResult.")
    if not isinstance(monitor, StreamHealthMonitor):
        raise TypeError("monitor must be StreamHealthMonitor.")
    active_reasons = (
        *reconciliation.active_reason_codes,
        *monitor.active_reason_codes,
    )
    stream_safe = monitor.state is StreamState.HEALTHY
    gate_passed = reconciliation.reconciliation_passed and stream_safe
    return OperationalDecision(
        reconciliation_passed=reconciliation.core_reconciliation_passed,
        limits_passed=reconciliation.limits_passed,
        stream_state=monitor.state,
        stream_safe=stream_safe,
        circuit_breaker_open=monitor.state is StreamState.CIRCUIT_OPEN,
        kill_switch_latched=monitor.state is StreamState.KILLED,
        active_reason_codes=tuple(active_reasons),
        operational_gate_passed=gate_passed,
        day20_order_submission_authorized=False,
        can_submit_orders=False,
    )
