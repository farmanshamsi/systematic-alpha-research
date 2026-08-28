"""Deterministic broker-neutral synthetic order-state machine for Day 19."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


ACKNOWLEDGMENT_TIMEOUT: Final[timedelta] = timedelta(seconds=30)
UPDATE_TIMEOUT: Final[timedelta] = timedelta(seconds=120)
EVENT_STALE_AFTER: Final[timedelta] = timedelta(seconds=120)
MAX_FUTURE_SKEW: Final[timedelta] = timedelta(seconds=5)
CLIENT_ORDER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"axiom-[a-z0-9-]{8,48}"
)


class OrderStatus(str, Enum):
    INTENT_CREATED = "intent_created"
    PENDING_REVIEW = "pending_review"
    PENDING_NEW = "pending_new"
    HELD = "held"
    ACCEPTED = "accepted"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    DONE_FOR_DAY = "done_for_day"
    STOPPED = "stopped"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    REJECTED = "rejected"


STATUS_ORDER: Final[tuple[OrderStatus, ...]] = tuple(OrderStatus)
PROVIDER_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    status for status in OrderStatus if status is not OrderStatus.INTENT_CREATED
)
TERMINAL_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REPLACED,
        OrderStatus.REJECTED,
    }
)
NO_FILL_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {
        OrderStatus.PENDING_REVIEW,
        OrderStatus.PENDING_NEW,
        OrderStatus.HELD,
        OrderStatus.ACCEPTED,
        OrderStatus.ACCEPTED_FOR_BIDDING,
        OrderStatus.REJECTED,
    }
)


LEGAL_TRANSITIONS: Final[Mapping[OrderStatus, frozenset[OrderStatus]]] = (
    MappingProxyType(
        {
            OrderStatus.INTENT_CREATED: PROVIDER_STATUSES,
            OrderStatus.PENDING_REVIEW: frozenset(
                {
                    OrderStatus.HELD,
                    OrderStatus.PENDING_NEW,
                    OrderStatus.ACCEPTED,
                    OrderStatus.NEW,
                    OrderStatus.REJECTED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.PENDING_NEW: frozenset(
                {
                    OrderStatus.HELD,
                    OrderStatus.ACCEPTED,
                    OrderStatus.NEW,
                    OrderStatus.REJECTED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.HELD: frozenset(
                {
                    OrderStatus.PENDING_NEW,
                    OrderStatus.ACCEPTED,
                    OrderStatus.NEW,
                    OrderStatus.REJECTED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.ACCEPTED: frozenset(
                {
                    OrderStatus.ACCEPTED_FOR_BIDDING,
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.PENDING_REPLACE,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                    OrderStatus.HELD,
                }
            ),
            OrderStatus.ACCEPTED_FOR_BIDDING: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                }
            ),
            OrderStatus.NEW: frozenset(
                {
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.PENDING_REPLACE,
                    OrderStatus.DONE_FOR_DAY,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                    OrderStatus.STOPPED,
                    OrderStatus.SUSPENDED,
                    OrderStatus.CALCULATED,
                }
            ),
            OrderStatus.PARTIALLY_FILLED: frozenset(
                {
                    OrderStatus.FILLED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.PENDING_REPLACE,
                    OrderStatus.DONE_FOR_DAY,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REPLACED,
                    OrderStatus.STOPPED,
                    OrderStatus.SUSPENDED,
                    OrderStatus.CALCULATED,
                }
            ),
            OrderStatus.PENDING_CANCEL: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.CANCELED,
                    OrderStatus.DONE_FOR_DAY,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.PENDING_REPLACE: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.REPLACED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.DONE_FOR_DAY: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.PENDING_REPLACE,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.STOPPED: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                }
            ),
            OrderStatus.SUSPENDED: frozenset(
                {
                    OrderStatus.NEW,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                }
            ),
            OrderStatus.CALCULATED: frozenset(
                {
                    OrderStatus.FILLED,
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                }
            ),
            OrderStatus.FILLED: frozenset(),
            OrderStatus.CANCELED: frozenset(),
            OrderStatus.EXPIRED: frozenset(),
            OrderStatus.REPLACED: frozenset(),
            OrderStatus.REJECTED: frozenset(),
        }
    )
)


REASON_CODES: Final[tuple[str, ...]] = (
    "unknown_order",
    "idempotency_conflict",
    "duplicate_event_conflict",
    "broker_order_id_mismatch",
    "requested_quantity_mismatch",
    "provider_sequence_not_increasing",
    "event_time_regressed",
    "event_arrived_stale",
    "event_time_future_skew",
    "illegal_status_transition",
    "filled_quantity_decreased",
    "filled_quantity_exceeds_order",
    "status_quantity_inconsistent",
    "average_fill_price_inconsistent",
    "replacement_id_missing",
    "invalid_event",
    "acknowledgment_timeout",
    "update_timeout",
)


class OrderStateError(RuntimeError):
    """Safe-to-persist rejected order message or intent."""

    def __init__(self, reason_code: str, client_order_id: str) -> None:
        if reason_code not in REASON_CODES:
            raise ValueError("Unknown Day 19 reason code.")
        self.reason_code = reason_code
        self.client_order_id = client_order_id
        super().__init__(f"Order state rejected: {reason_code}.")


def _decimal(value: object, *, allow_zero: bool) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid decimal value.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid decimal value.") from exc
    if not number.is_finite() or number < 0 or (not allow_zero and number == 0):
        raise ValueError("Decimal value is outside the permitted range.")
    return number


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _opaque_identifier(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Identifier must be non-empty text without padding.")
    if any(character.isspace() for character in value):
        raise ValueError("Identifier cannot contain whitespace.")
    return value


@dataclass(frozen=True, slots=True)
class OrderIntent:
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    requested_quantity: Decimal
    submitted_at: datetime

    def __post_init__(self) -> None:
        if CLIENT_ORDER_ID_PATTERN.fullmatch(self.client_order_id) is None:
            raise ValueError("client_order_id does not match the Day 19 format.")
        if not isinstance(self.symbol, str) or not self.symbol.isalpha():
            raise ValueError("symbol must contain letters only.")
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")
        if self.order_type not in {
            "market",
            "limit",
            "stop",
            "stop_limit",
            "trailing_stop",
        }:
            raise ValueError("Unsupported order_type.")
        if self.time_in_force not in {"day", "gtc", "opg", "cls", "ioc", "fok"}:
            raise ValueError("Unsupported time_in_force.")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(
            self,
            "requested_quantity",
            _decimal(self.requested_quantity, allow_zero=False),
        )
        object.__setattr__(self, "submitted_at", _utc(self.submitted_at))


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    event_id: str
    provider_sequence: int
    client_order_id: str
    broker_order_id: str
    status: OrderStatus
    requested_quantity: Decimal
    filled_quantity: Decimal
    filled_average_price: Decimal | None
    event_at: datetime
    received_at: datetime
    replacement_order_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _opaque_identifier(self.event_id))
        if type(self.provider_sequence) is not int or self.provider_sequence < 0:
            raise ValueError("provider_sequence must be a non-negative integer.")
        if CLIENT_ORDER_ID_PATTERN.fullmatch(self.client_order_id) is None:
            raise ValueError("client_order_id does not match the Day 19 format.")
        object.__setattr__(
            self, "broker_order_id", _opaque_identifier(self.broker_order_id)
        )
        if not isinstance(self.status, OrderStatus):
            object.__setattr__(self, "status", OrderStatus(self.status))
        if self.status is OrderStatus.INTENT_CREATED:
            raise ValueError("Broker updates cannot use intent_created status.")
        object.__setattr__(
            self,
            "requested_quantity",
            _decimal(self.requested_quantity, allow_zero=False),
        )
        object.__setattr__(
            self,
            "filled_quantity",
            _decimal(self.filled_quantity, allow_zero=True),
        )
        if self.filled_average_price is not None:
            object.__setattr__(
                self,
                "filled_average_price",
                _decimal(self.filled_average_price, allow_zero=False),
            )
        object.__setattr__(self, "event_at", _utc(self.event_at))
        object.__setattr__(self, "received_at", _utc(self.received_at))
        if self.replacement_order_id is not None:
            object.__setattr__(
                self,
                "replacement_order_id",
                _opaque_identifier(self.replacement_order_id),
            )


@dataclass(frozen=True, slots=True)
class OrderState:
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    requested_quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal
    filled_average_price: Decimal | None
    submitted_at: datetime
    last_provider_sequence: int | None
    last_event_at: datetime | None
    last_received_at: datetime | None
    replacement_order_id: str | None
    recovery_required: bool

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class AuditEntry:
    audit_sequence: int
    scenario_id: str
    client_order_id: str
    broker_order_id: str
    event_id: str
    provider_sequence: int | None
    previous_status: str
    incoming_status: str
    resulting_status: str
    action: str
    incremental_fill: Decimal
    cumulative_filled_quantity: Decimal
    reason_code: str
    event_at: datetime
    received_at: datetime
    recovery_required: bool


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    action: str
    state: OrderState | None
    audit_entry: AuditEntry
    incremental_fill: Decimal


@dataclass(frozen=True, slots=True)
class TimeoutDiagnostic:
    scenario_id: str
    client_order_id: str
    reason_code: str
    as_of: datetime
    reference_time: datetime
    elapsed_seconds: int
    last_provider_sequence: int | None
    recovery_required: bool


def _event_fingerprint(update: OrderUpdate) -> tuple[object, ...]:
    return (
        update.event_id,
        update.provider_sequence,
        update.client_order_id,
        update.broker_order_id,
        update.status.value,
        str(update.requested_quantity),
        str(update.filled_quantity),
        (
            None
            if update.filled_average_price is None
            else str(update.filled_average_price)
        ),
        update.event_at.isoformat(),
        update.replacement_order_id,
    )


def is_transition_allowed(
    from_status: OrderStatus,
    to_status: OrderStatus,
) -> bool:
    """Return the frozen transition decision for one normalized status pair."""

    if not isinstance(from_status, OrderStatus) or not isinstance(
        to_status, OrderStatus
    ):
        raise TypeError("from_status and to_status must be OrderStatus values.")
    return to_status is not OrderStatus.INTENT_CREATED and (
        from_status is to_status or to_status in LEGAL_TRANSITIONS[from_status]
    )


class OrderStateMachine:
    """Append-only deterministic processor for one synthetic scenario."""

    def __init__(self, scenario_id: str) -> None:
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("scenario_id must be non-empty text.")
        self.scenario_id = scenario_id
        self._states: dict[str, OrderState] = {}
        self._event_fingerprints: dict[str, tuple[object, ...]] = {}
        self._audit_entries: list[AuditEntry] = []
        self._timeout_diagnostics: list[TimeoutDiagnostic] = []
        self._timeout_keys: set[tuple[str, str, int | None]] = set()
        self._audit_sequence = 0
        self._global_recovery_required = False

    @property
    def states(self) -> Mapping[str, OrderState]:
        return MappingProxyType(dict(self._states))

    @property
    def audit_entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit_entries)

    @property
    def timeout_diagnostics(self) -> tuple[TimeoutDiagnostic, ...]:
        return tuple(self._timeout_diagnostics)

    @property
    def global_recovery_required(self) -> bool:
        return self._global_recovery_required

    def _next_audit_sequence(self) -> int:
        self._audit_sequence += 1
        return self._audit_sequence

    def _append_audit(
        self,
        *,
        state: OrderState | None,
        update: OrderUpdate | None,
        previous_status: str,
        incoming_status: str,
        resulting_status: str,
        action: str,
        incremental_fill: Decimal,
        reason_code: str = "",
        event_at: datetime,
        received_at: datetime,
        safe_client_order_id: str = "",
        safe_event_id: str = "",
    ) -> AuditEntry:
        client_order_id = safe_client_order_id
        broker_order_id = ""
        if state is not None:
            client_order_id = state.client_order_id
            broker_order_id = state.broker_order_id or ""
        if update is not None:
            client_order_id = update.client_order_id
            broker_order_id = update.broker_order_id
        entry = AuditEntry(
            audit_sequence=self._next_audit_sequence(),
            scenario_id=self.scenario_id,
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            event_id=(
                update.event_id if update is not None else safe_event_id
            ),
            provider_sequence=(
                update.provider_sequence if update is not None else None
            ),
            previous_status=previous_status,
            incoming_status=incoming_status,
            resulting_status=resulting_status,
            action=action,
            incremental_fill=incremental_fill,
            cumulative_filled_quantity=(
                state.filled_quantity if state is not None else Decimal("0")
            ),
            reason_code=reason_code,
            event_at=event_at,
            received_at=received_at,
            recovery_required=(
                state.recovery_required
                if state is not None
                else self._global_recovery_required
            ),
        )
        self._audit_entries.append(entry)
        return entry

    def _reject(
        self,
        reason_code: str,
        update: OrderUpdate,
        state: OrderState | None,
    ) -> None:
        if state is None:
            self._global_recovery_required = True
        else:
            state = replace(state, recovery_required=True)
            self._states[state.client_order_id] = state
        current_status = state.status.value if state is not None else "unknown"
        self._append_audit(
            state=state,
            update=update,
            previous_status=current_status,
            incoming_status=update.status.value,
            resulting_status=current_status,
            action="rejected",
            incremental_fill=Decimal("0"),
            reason_code=reason_code,
            event_at=update.event_at,
            received_at=update.received_at,
        )
        raise OrderStateError(reason_code, update.client_order_id)

    def register_intent(self, intent: OrderIntent) -> ProcessingResult:
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent.")
        existing = self._states.get(intent.client_order_id)
        if existing is not None:
            same = (
                existing.symbol == intent.symbol
                and existing.side == intent.side
                and existing.order_type == intent.order_type
                and existing.time_in_force == intent.time_in_force
                and existing.requested_quantity == intent.requested_quantity
                and existing.submitted_at == intent.submitted_at
            )
            if not same:
                existing = replace(existing, recovery_required=True)
                self._states[intent.client_order_id] = existing
                entry = self._append_audit(
                    state=existing,
                    update=None,
                    previous_status=existing.status.value,
                    incoming_status=OrderStatus.INTENT_CREATED.value,
                    resulting_status=existing.status.value,
                    action="rejected",
                    incremental_fill=Decimal("0"),
                    reason_code="idempotency_conflict",
                    event_at=intent.submitted_at,
                    received_at=intent.submitted_at,
                )
                raise OrderStateError(
                    "idempotency_conflict", intent.client_order_id
                )
            entry = self._append_audit(
                state=existing,
                update=None,
                previous_status=existing.status.value,
                incoming_status=OrderStatus.INTENT_CREATED.value,
                resulting_status=existing.status.value,
                action="duplicate_intent_ignored",
                incremental_fill=Decimal("0"),
                event_at=intent.submitted_at,
                received_at=intent.submitted_at,
            )
            return ProcessingResult(
                action="duplicate_intent_ignored",
                state=existing,
                audit_entry=entry,
                incremental_fill=Decimal("0"),
            )

        state = OrderState(
            client_order_id=intent.client_order_id,
            broker_order_id=None,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            requested_quantity=intent.requested_quantity,
            status=OrderStatus.INTENT_CREATED,
            filled_quantity=Decimal("0"),
            filled_average_price=None,
            submitted_at=intent.submitted_at,
            last_provider_sequence=None,
            last_event_at=None,
            last_received_at=None,
            replacement_order_id=None,
            recovery_required=False,
        )
        self._states[state.client_order_id] = state
        entry = self._append_audit(
            state=state,
            update=None,
            previous_status="",
            incoming_status=OrderStatus.INTENT_CREATED.value,
            resulting_status=OrderStatus.INTENT_CREATED.value,
            action="intent_registered",
            incremental_fill=Decimal("0"),
            event_at=intent.submitted_at,
            received_at=intent.submitted_at,
        )
        return ProcessingResult(
            action="intent_registered",
            state=state,
            audit_entry=entry,
            incremental_fill=Decimal("0"),
        )

    def record_invalid_event(
        self,
        *,
        client_order_id: str,
        event_id: str,
        received_at: datetime,
    ) -> AuditEntry:
        """Record a safely identified message that failed normalization."""

        if CLIENT_ORDER_ID_PATTERN.fullmatch(client_order_id) is None:
            raise ValueError("client_order_id does not match the Day 19 format.")
        normalized_event_id = _opaque_identifier(event_id)
        normalized_received_at = _utc(received_at)
        state = self._states.get(client_order_id)
        fingerprint = ("invalid_event", client_order_id, normalized_event_id)
        prior_fingerprint = self._event_fingerprints.get(normalized_event_id)
        if prior_fingerprint == fingerprint:
            current_status = (
                state.status.value if state is not None else "unknown"
            )
            entry = self._append_audit(
                state=state,
                update=None,
                previous_status=current_status,
                incoming_status="invalid_event",
                resulting_status=current_status,
                action="duplicate_event_ignored",
                incremental_fill=Decimal("0"),
                event_at=normalized_received_at,
                received_at=normalized_received_at,
                safe_client_order_id=client_order_id,
                safe_event_id=normalized_event_id,
            )
            return entry
        if prior_fingerprint is not None:
            if state is None:
                self._global_recovery_required = True
            else:
                state = replace(state, recovery_required=True)
                self._states[client_order_id] = state
            current_status = (
                state.status.value if state is not None else "unknown"
            )
            self._append_audit(
                state=state,
                update=None,
                previous_status=current_status,
                incoming_status="invalid_event",
                resulting_status=current_status,
                action="rejected",
                incremental_fill=Decimal("0"),
                reason_code="duplicate_event_conflict",
                event_at=normalized_received_at,
                received_at=normalized_received_at,
                safe_client_order_id=client_order_id,
                safe_event_id=normalized_event_id,
            )
            raise OrderStateError("duplicate_event_conflict", client_order_id)
        self._event_fingerprints[normalized_event_id] = fingerprint
        if state is None:
            self._global_recovery_required = True
        else:
            state = replace(state, recovery_required=True)
            self._states[client_order_id] = state
        current_status = state.status.value if state is not None else "unknown"
        self._append_audit(
            state=state,
            update=None,
            previous_status=current_status,
            incoming_status="invalid_event",
            resulting_status=current_status,
            action="rejected",
            incremental_fill=Decimal("0"),
            reason_code="invalid_event",
            event_at=normalized_received_at,
            received_at=normalized_received_at,
            safe_client_order_id=client_order_id,
            safe_event_id=normalized_event_id,
        )
        raise OrderStateError("invalid_event", client_order_id)

    def _validate_update(self, update: OrderUpdate, state: OrderState) -> None:
        if state.broker_order_id is not None and (
            update.broker_order_id != state.broker_order_id
        ):
            self._reject("broker_order_id_mismatch", update, state)
        if update.requested_quantity != state.requested_quantity:
            self._reject("requested_quantity_mismatch", update, state)
        if state.last_provider_sequence is not None and (
            update.provider_sequence <= state.last_provider_sequence
        ):
            self._reject("provider_sequence_not_increasing", update, state)
        if state.last_event_at is not None and update.event_at < state.last_event_at:
            self._reject("event_time_regressed", update, state)
        if update.event_at - update.received_at > MAX_FUTURE_SKEW:
            self._reject("event_time_future_skew", update, state)
        if update.received_at - update.event_at > EVENT_STALE_AFTER:
            self._reject("event_arrived_stale", update, state)
        if not is_transition_allowed(state.status, update.status):
            self._reject("illegal_status_transition", update, state)
        if update.filled_quantity < state.filled_quantity:
            self._reject("filled_quantity_decreased", update, state)
        if update.filled_quantity > state.requested_quantity:
            self._reject("filled_quantity_exceeds_order", update, state)
        if state.terminal and update.status is state.status and (
            update.filled_quantity != state.filled_quantity
        ):
            self._reject("status_quantity_inconsistent", update, state)
        if state.terminal and update.status is state.status and (
            update.replacement_order_id != state.replacement_order_id
        ):
            self._reject("replacement_id_missing", update, state)
        if update.status is OrderStatus.PARTIALLY_FILLED and not (
            Decimal("0") < update.filled_quantity < state.requested_quantity
        ):
            self._reject("status_quantity_inconsistent", update, state)
        if update.status is OrderStatus.FILLED and (
            update.filled_quantity != state.requested_quantity
        ):
            self._reject("status_quantity_inconsistent", update, state)
        if update.status in NO_FILL_STATUSES and update.filled_quantity != 0:
            self._reject("status_quantity_inconsistent", update, state)
        if update.filled_quantity == 0 and update.filled_average_price is not None:
            self._reject("average_fill_price_inconsistent", update, state)
        if update.filled_quantity > 0 and update.filled_average_price is None:
            self._reject("average_fill_price_inconsistent", update, state)
        if update.filled_quantity == state.filled_quantity and (
            update.filled_average_price != state.filled_average_price
        ):
            self._reject("average_fill_price_inconsistent", update, state)
        if update.status is OrderStatus.REPLACED:
            if (
                update.replacement_order_id is None
                or update.replacement_order_id == update.broker_order_id
            ):
                self._reject("replacement_id_missing", update, state)

    def apply(self, update: OrderUpdate) -> ProcessingResult:
        if not isinstance(update, OrderUpdate):
            raise TypeError("update must be an OrderUpdate.")
        fingerprint = _event_fingerprint(update)
        prior_fingerprint = self._event_fingerprints.get(update.event_id)
        state = self._states.get(update.client_order_id)
        if prior_fingerprint is not None:
            if prior_fingerprint != fingerprint:
                self._reject("duplicate_event_conflict", update, state)
            current_status = (
                state.status.value if state is not None else "unknown"
            )
            entry = self._append_audit(
                state=state,
                update=update,
                previous_status=current_status,
                incoming_status=update.status.value,
                resulting_status=current_status,
                action="duplicate_event_ignored",
                incremental_fill=Decimal("0"),
                event_at=update.event_at,
                received_at=update.received_at,
            )
            return ProcessingResult(
                action="duplicate_event_ignored",
                state=state,
                audit_entry=entry,
                incremental_fill=Decimal("0"),
            )
        self._event_fingerprints[update.event_id] = fingerprint
        if state is None:
            self._reject("unknown_order", update, None)

        self._validate_update(update, state)
        incremental_fill = update.filled_quantity - state.filled_quantity
        next_state = replace(
            state,
            broker_order_id=state.broker_order_id or update.broker_order_id,
            status=update.status,
            filled_quantity=update.filled_quantity,
            filled_average_price=update.filled_average_price,
            last_provider_sequence=update.provider_sequence,
            last_event_at=update.event_at,
            last_received_at=update.received_at,
            replacement_order_id=update.replacement_order_id,
        )
        self._states[next_state.client_order_id] = next_state
        entry = self._append_audit(
            state=next_state,
            update=update,
            previous_status=state.status.value,
            incoming_status=update.status.value,
            resulting_status=next_state.status.value,
            action="applied",
            incremental_fill=incremental_fill,
            event_at=update.event_at,
            received_at=update.received_at,
        )
        return ProcessingResult(
            action="applied",
            state=next_state,
            audit_entry=entry,
            incremental_fill=incremental_fill,
        )

    def check_timeouts(self, as_of: datetime) -> tuple[TimeoutDiagnostic, ...]:
        normalized_as_of = _utc(as_of)
        created: list[TimeoutDiagnostic] = []
        for client_order_id, state in tuple(self._states.items()):
            if state.terminal:
                continue
            if state.status is OrderStatus.INTENT_CREATED:
                reference = state.submitted_at
                threshold = ACKNOWLEDGMENT_TIMEOUT
                reason_code = "acknowledgment_timeout"
            elif state.last_received_at is not None:
                reference = state.last_received_at
                threshold = UPDATE_TIMEOUT
                reason_code = "update_timeout"
            else:
                continue
            elapsed = normalized_as_of - reference
            if elapsed < timedelta(0):
                raise ValueError("as_of cannot precede the timeout reference.")
            if elapsed <= threshold:
                continue
            key = (
                client_order_id,
                reason_code,
                state.last_provider_sequence,
            )
            if key in self._timeout_keys:
                continue
            self._timeout_keys.add(key)
            state = replace(state, recovery_required=True)
            self._states[client_order_id] = state
            diagnostic = TimeoutDiagnostic(
                scenario_id=self.scenario_id,
                client_order_id=client_order_id,
                reason_code=reason_code,
                as_of=normalized_as_of,
                reference_time=reference,
                elapsed_seconds=int(elapsed.total_seconds()),
                last_provider_sequence=state.last_provider_sequence,
                recovery_required=True,
            )
            self._timeout_diagnostics.append(diagnostic)
            created.append(diagnostic)
            self._append_audit(
                state=state,
                update=None,
                previous_status=state.status.value,
                incoming_status="timeout_check",
                resulting_status=state.status.value,
                action="timeout",
                incremental_fill=Decimal("0"),
                reason_code=reason_code,
                event_at=normalized_as_of,
                received_at=normalized_as_of,
            )
        return tuple(created)


def transition_matrix_rows() -> tuple[dict[str, object], ...]:
    """Return the full frozen from/to matrix in exact status order."""

    rows: list[dict[str, object]] = []
    for from_status in STATUS_ORDER:
        for to_status in STATUS_ORDER:
            allowed = is_transition_allowed(from_status, to_status)
            rows.append(
                {
                    "from_status": from_status.value,
                    "to_status": to_status.value,
                    "allowed": allowed,
                    "terminal_from_status": from_status in TERMINAL_STATUSES,
                }
            )
    return tuple(rows)
