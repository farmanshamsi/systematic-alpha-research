"""Deterministic synthetic broker reconciliation and exposure checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Final, Mapping

from systematic_alpha.broker.order_state import (
    CLIENT_ORDER_ID_PATTERN,
    OrderState,
    OrderStatus,
)


SNAPSHOT_STALE_AFTER: Final[timedelta] = timedelta(seconds=30)
CASH_TOLERANCE: Final[Decimal] = Decimal("0.01")
PRICE_TOLERANCE: Final[Decimal] = Decimal("0.0001")
MAX_SINGLE_ORDER_QUANTITY: Final[Decimal] = Decimal("25")
MAX_ABSOLUTE_SYMBOL_POSITION: Final[Decimal] = Decimal("100")
MAX_GROSS_NOTIONAL: Final[Decimal] = Decimal("25000")
MAX_OPEN_ORDERS: Final[int] = 5
SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Z]{1,10}")

RECONCILIATION_REASON_CODES: Final[tuple[str, ...]] = (
    "local_recovery_required",
    "local_order_missing_at_broker",
    "unknown_broker_order",
    "broker_order_id_mismatch",
    "duplicate_broker_order_id",
    "order_status_mismatch",
    "requested_quantity_mismatch",
    "cumulative_fill_mismatch",
    "weighted_average_price_mismatch",
    "duplicate_fill_conflict",
    "unknown_order_fill",
    "fill_broker_order_id_mismatch",
    "fill_symbol_side_mismatch",
    "fill_quantity_mismatch",
    "broker_position_missing",
    "unexpected_broker_position",
    "position_quantity_mismatch",
    "cash_balance_mismatch",
    "snapshot_stale",
    "single_order_limit_exceeded",
    "symbol_position_limit_exceeded",
    "gross_notional_limit_exceeded",
    "open_order_limit_exceeded",
)
LIMIT_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "single_order_limit_exceeded",
        "symbol_position_limit_exceeded",
        "gross_notional_limit_exceeded",
        "open_order_limit_exceeded",
    }
)


def _decimal(
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid decimal value.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid decimal value.") from exc
    if not number.is_finite():
        raise ValueError("Decimal values must be finite.")
    if positive and number <= 0:
        raise ValueError("Decimal value must be positive.")
    if nonnegative and number < 0:
        raise ValueError("Decimal value must be non-negative.")
    return number


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


def _symbol(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Symbol must be text.")
    normalized = value.upper()
    if SYMBOL_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Symbol must contain one to ten letters.")
    return normalized


def _decimal_text(value: Decimal | None) -> str:
    return "" if value is None else str(value)


def _price_matches(left: Decimal | None, right: Decimal | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= PRICE_TOLERANCE


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    status: OrderStatus
    requested_quantity: Decimal
    filled_quantity: Decimal
    filled_average_price: Decimal | None
    snapshot_at: datetime

    def __post_init__(self) -> None:
        if CLIENT_ORDER_ID_PATTERN.fullmatch(self.client_order_id) is None:
            raise ValueError("Invalid client_order_id.")
        object.__setattr__(
            self, "broker_order_id", _identifier(self.broker_order_id)
        )
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")
        if not isinstance(self.status, OrderStatus):
            object.__setattr__(self, "status", OrderStatus(self.status))
        if self.status is OrderStatus.INTENT_CREATED:
            raise ValueError("Broker snapshots cannot use intent_created.")
        object.__setattr__(
            self,
            "requested_quantity",
            _decimal(self.requested_quantity, positive=True),
        )
        object.__setattr__(
            self,
            "filled_quantity",
            _decimal(self.filled_quantity, nonnegative=True),
        )
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("filled_quantity exceeds requested_quantity.")
        if self.filled_average_price is not None:
            object.__setattr__(
                self,
                "filled_average_price",
                _decimal(self.filled_average_price, positive=True),
            )
        if (self.filled_quantity == 0) != (
            self.filled_average_price is None
        ):
            raise ValueError("Broker fill quantity and average are inconsistent.")
        object.__setattr__(self, "snapshot_at", _utc(self.snapshot_at))


@dataclass(frozen=True, slots=True)
class BrokerFillSnapshot:
    fill_id: str
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    executed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", _identifier(self.fill_id))
        if CLIENT_ORDER_ID_PATTERN.fullmatch(self.client_order_id) is None:
            raise ValueError("Invalid client_order_id.")
        object.__setattr__(
            self, "broker_order_id", _identifier(self.broker_order_id)
        )
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")
        object.__setattr__(
            self, "quantity", _decimal(self.quantity, positive=True)
        )
        object.__setattr__(self, "price", _decimal(self.price, positive=True))
        object.__setattr__(self, "executed_at", _utc(self.executed_at))


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    symbol: str
    quantity: Decimal
    mark_price: Decimal
    snapshot_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "quantity", _decimal(self.quantity))
        object.__setattr__(
            self, "mark_price", _decimal(self.mark_price, positive=True)
        )
        object.__setattr__(self, "snapshot_at", _utc(self.snapshot_at))


@dataclass(frozen=True, slots=True)
class OpeningPosition:
    symbol: str
    quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "quantity", _decimal(self.quantity))


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    cash: Decimal
    snapshot_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash", _decimal(self.cash))
        object.__setattr__(self, "snapshot_at", _utc(self.snapshot_at))


@dataclass(frozen=True, slots=True)
class ReconciliationInput:
    local_states: tuple[OrderState, ...]
    broker_orders: tuple[BrokerOrderSnapshot, ...]
    broker_fills: tuple[BrokerFillSnapshot, ...]
    broker_positions: tuple[BrokerPositionSnapshot, ...]
    opening_positions: tuple[OpeningPosition, ...]
    opening_cash: Decimal
    broker_account: BrokerAccountSnapshot
    as_of: datetime

    def __post_init__(self) -> None:
        tuple_fields = (
            "local_states",
            "broker_orders",
            "broker_fills",
            "broker_positions",
            "opening_positions",
        )
        for field_name in tuple_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if not all(isinstance(row, OrderState) for row in self.local_states):
            raise TypeError("local_states must contain OrderState values.")
        if not all(
            isinstance(row, BrokerOrderSnapshot) for row in self.broker_orders
        ):
            raise TypeError("broker_orders contain an invalid value.")
        if not all(
            isinstance(row, BrokerFillSnapshot) for row in self.broker_fills
        ):
            raise TypeError("broker_fills contain an invalid value.")
        if not all(
            isinstance(row, BrokerPositionSnapshot)
            for row in self.broker_positions
        ):
            raise TypeError("broker_positions contain an invalid value.")
        if not all(
            isinstance(row, OpeningPosition) for row in self.opening_positions
        ):
            raise TypeError("opening_positions contain an invalid value.")
        if not isinstance(self.broker_account, BrokerAccountSnapshot):
            raise TypeError("broker_account must be BrokerAccountSnapshot.")
        object.__setattr__(self, "opening_cash", _decimal(self.opening_cash))
        object.__setattr__(self, "as_of", _utc(self.as_of))

        unique_contracts = (
            (
                "local client order",
                (row.client_order_id for row in self.local_states),
            ),
            (
                "broker client order",
                (row.client_order_id for row in self.broker_orders),
            ),
            ("broker position", (row.symbol for row in self.broker_positions)),
            ("opening position", (row.symbol for row in self.opening_positions)),
        )
        for name, values_iterator in unique_contracts:
            values = tuple(values_iterator)
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {name} key.")
        snapshots = (
            *(row.snapshot_at for row in self.broker_orders),
            *(row.snapshot_at for row in self.broker_positions),
            self.broker_account.snapshot_at,
        )
        if any(timestamp > self.as_of for timestamp in snapshots):
            raise ValueError("Broker snapshot cannot be in the future.")
        if any(row.executed_at > self.as_of for row in self.broker_fills):
            raise ValueError("Broker fill cannot be in the future.")


@dataclass(frozen=True, slots=True)
class ReconciliationDiagnostic:
    diagnostic_sequence: int
    category: str
    reason_code: str
    client_order_id: str
    broker_order_id: str
    symbol: str
    local_value: str
    broker_value: str
    required_action: str

    def __post_init__(self) -> None:
        if self.reason_code not in RECONCILIATION_REASON_CODES:
            raise ValueError("Unknown reconciliation reason code.")


@dataclass(frozen=True, slots=True)
class BalanceComparison:
    balance_type: str
    symbol: str
    opening_value: Decimal
    signed_fill_flow: Decimal
    expected_value: Decimal
    broker_value: Decimal
    difference: Decimal
    tolerance: Decimal
    reconciled: bool


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    diagnostics: tuple[ReconciliationDiagnostic, ...]
    balance_comparisons: tuple[BalanceComparison, ...]
    local_order_count: int
    broker_order_count: int
    unique_fill_count: int
    exact_duplicate_fill_count: int
    broker_position_count: int
    expected_cash: Decimal
    broker_cash: Decimal
    gross_notional: Decimal
    open_order_count: int
    core_reconciliation_passed: bool
    limits_passed: bool
    reconciliation_passed: bool

    @property
    def active_reason_codes(self) -> tuple[str, ...]:
        observed = {row.reason_code for row in self.diagnostics}
        return tuple(
            reason
            for reason in RECONCILIATION_REASON_CODES
            if reason in observed
        )


def _fill_fingerprint(fill: BrokerFillSnapshot) -> tuple[object, ...]:
    return (
        fill.client_order_id,
        fill.broker_order_id,
        fill.symbol,
        fill.side,
        str(fill.quantity),
        str(fill.price),
        fill.executed_at.isoformat(),
    )


def reconcile_snapshot(snapshot: ReconciliationInput) -> ReconciliationResult:
    """Reconcile one immutable synthetic broker snapshot without mutation."""

    if not isinstance(snapshot, ReconciliationInput):
        raise TypeError("snapshot must be ReconciliationInput.")
    diagnostics: list[ReconciliationDiagnostic] = []
    diagnostic_keys: set[tuple[str, str, str, str]] = set()

    def add(
        category: str,
        reason_code: str,
        *,
        client_order_id: str = "",
        broker_order_id: str = "",
        symbol: str = "",
        local_value: object = "",
        broker_value: object = "",
    ) -> None:
        key = (reason_code, client_order_id, broker_order_id, symbol)
        if key in diagnostic_keys:
            return
        diagnostic_keys.add(key)
        diagnostics.append(
            ReconciliationDiagnostic(
                diagnostic_sequence=len(diagnostics) + 1,
                category=category,
                reason_code=reason_code,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                symbol=symbol,
                local_value=str(local_value),
                broker_value=str(broker_value),
                required_action="block_and_reconcile",
            )
        )

    local_by_client = {
        row.client_order_id: row for row in snapshot.local_states
    }
    broker_by_client = {
        row.client_order_id: row for row in snapshot.broker_orders
    }

    broker_id_to_client: dict[str, str] = {}
    for broker_order in snapshot.broker_orders:
        prior_client = broker_id_to_client.get(broker_order.broker_order_id)
        if prior_client is not None and prior_client != broker_order.client_order_id:
            add(
                "order",
                "duplicate_broker_order_id",
                client_order_id=broker_order.client_order_id,
                broker_order_id=broker_order.broker_order_id,
                symbol=broker_order.symbol,
                local_value=prior_client,
                broker_value=broker_order.client_order_id,
            )
        broker_id_to_client[broker_order.broker_order_id] = (
            broker_order.client_order_id
        )

    for local in snapshot.local_states:
        broker = broker_by_client.get(local.client_order_id)
        if local.recovery_required:
            add(
                "order",
                "local_recovery_required",
                client_order_id=local.client_order_id,
                broker_order_id=local.broker_order_id or "",
                symbol=local.symbol,
                local_value=True,
                broker_value="manual_reconciliation_required",
            )
        if broker is None:
            add(
                "order",
                "local_order_missing_at_broker",
                client_order_id=local.client_order_id,
                broker_order_id=local.broker_order_id or "",
                symbol=local.symbol,
                local_value=local.status.value,
                broker_value="missing",
            )
            continue
        comparisons = (
            (
                "broker_order_id_mismatch",
                local.broker_order_id or "",
                broker.broker_order_id,
            ),
            ("order_status_mismatch", local.status.value, broker.status.value),
            (
                "requested_quantity_mismatch",
                local.requested_quantity,
                broker.requested_quantity,
            ),
            (
                "cumulative_fill_mismatch",
                local.filled_quantity,
                broker.filled_quantity,
            ),
        )
        for reason_code, local_value, broker_value in comparisons:
            if local_value != broker_value:
                add(
                    "order",
                    reason_code,
                    client_order_id=local.client_order_id,
                    broker_order_id=broker.broker_order_id,
                    symbol=local.symbol,
                    local_value=local_value,
                    broker_value=broker_value,
                )
        if local.symbol != broker.symbol or local.side != broker.side:
            add(
                "order",
                "fill_symbol_side_mismatch",
                client_order_id=local.client_order_id,
                broker_order_id=broker.broker_order_id,
                symbol=broker.symbol,
                local_value=f"{local.symbol}:{local.side}",
                broker_value=f"{broker.symbol}:{broker.side}",
            )
        if not _price_matches(
            local.filled_average_price, broker.filled_average_price
        ):
            add(
                "order",
                "weighted_average_price_mismatch",
                client_order_id=local.client_order_id,
                broker_order_id=broker.broker_order_id,
                symbol=local.symbol,
                local_value=_decimal_text(local.filled_average_price),
                broker_value=_decimal_text(broker.filled_average_price),
            )

    for broker in snapshot.broker_orders:
        if broker.client_order_id not in local_by_client:
            add(
                "order",
                "unknown_broker_order",
                client_order_id=broker.client_order_id,
                broker_order_id=broker.broker_order_id,
                symbol=broker.symbol,
                local_value="missing",
                broker_value=broker.status.value,
            )

    fill_fingerprints: dict[str, tuple[object, ...]] = {}
    unique_fills: list[BrokerFillSnapshot] = []
    exact_duplicates = 0
    for fill in snapshot.broker_fills:
        fingerprint = _fill_fingerprint(fill)
        prior = fill_fingerprints.get(fill.fill_id)
        if prior is None:
            fill_fingerprints[fill.fill_id] = fingerprint
            unique_fills.append(fill)
            continue
        if prior == fingerprint:
            exact_duplicates += 1
            continue
        add(
            "fill",
            "duplicate_fill_conflict",
            client_order_id=fill.client_order_id,
            broker_order_id=fill.broker_order_id,
            symbol=fill.symbol,
            local_value="first_normalized_fill",
            broker_value="conflicting_duplicate_fill",
        )

    fills_by_client: dict[str, list[BrokerFillSnapshot]] = {}
    signed_fill_by_symbol: dict[str, Decimal] = {}
    cash_flow = Decimal("0")
    for fill in unique_fills:
        fills_by_client.setdefault(fill.client_order_id, []).append(fill)
        sign = Decimal("1") if fill.side == "buy" else Decimal("-1")
        signed_fill_by_symbol[fill.symbol] = (
            signed_fill_by_symbol.get(fill.symbol, Decimal("0"))
            + sign * fill.quantity
        )
        cash_flow -= sign * fill.quantity * fill.price
        local = local_by_client.get(fill.client_order_id)
        if local is None:
            add(
                "fill",
                "unknown_order_fill",
                client_order_id=fill.client_order_id,
                broker_order_id=fill.broker_order_id,
                symbol=fill.symbol,
                local_value="missing",
                broker_value=fill.fill_id,
            )
            continue
        if fill.broker_order_id != local.broker_order_id:
            add(
                "fill",
                "fill_broker_order_id_mismatch",
                client_order_id=fill.client_order_id,
                broker_order_id=fill.broker_order_id,
                symbol=fill.symbol,
                local_value=local.broker_order_id or "",
                broker_value=fill.broker_order_id,
            )
        if fill.symbol != local.symbol or fill.side != local.side:
            add(
                "fill",
                "fill_symbol_side_mismatch",
                client_order_id=fill.client_order_id,
                broker_order_id=fill.broker_order_id,
                symbol=fill.symbol,
                local_value=f"{local.symbol}:{local.side}",
                broker_value=f"{fill.symbol}:{fill.side}",
            )

    for local in snapshot.local_states:
        fills = fills_by_client.get(local.client_order_id, [])
        total_quantity = sum(
            (row.quantity for row in fills), start=Decimal("0")
        )
        if total_quantity != local.filled_quantity:
            add(
                "fill",
                "fill_quantity_mismatch",
                client_order_id=local.client_order_id,
                broker_order_id=local.broker_order_id or "",
                symbol=local.symbol,
                local_value=local.filled_quantity,
                broker_value=total_quantity,
            )
        weighted_price = None
        if total_quantity > 0:
            weighted_price = sum(
                (row.quantity * row.price for row in fills),
                start=Decimal("0"),
            ) / total_quantity
        if not _price_matches(local.filled_average_price, weighted_price):
            add(
                "fill",
                "weighted_average_price_mismatch",
                client_order_id=local.client_order_id,
                broker_order_id=local.broker_order_id or "",
                symbol=local.symbol,
                local_value=_decimal_text(local.filled_average_price),
                broker_value=_decimal_text(weighted_price),
            )

    opening_by_symbol = {
        row.symbol: row.quantity for row in snapshot.opening_positions
    }
    expected_by_symbol = dict(opening_by_symbol)
    for symbol, flow in signed_fill_by_symbol.items():
        expected_by_symbol[symbol] = (
            expected_by_symbol.get(symbol, Decimal("0")) + flow
        )
    broker_position_by_symbol = {
        row.symbol: row for row in snapshot.broker_positions
    }
    all_symbols = tuple(
        sorted(set(expected_by_symbol) | set(broker_position_by_symbol))
    )
    balance_rows: list[BalanceComparison] = []
    for symbol in all_symbols:
        opening = opening_by_symbol.get(symbol, Decimal("0"))
        flow = signed_fill_by_symbol.get(symbol, Decimal("0"))
        expected = expected_by_symbol.get(symbol, Decimal("0"))
        broker_position = broker_position_by_symbol.get(symbol)
        actual = (
            Decimal("0")
            if broker_position is None
            else broker_position.quantity
        )
        reconciled = actual == expected
        balance_rows.append(
            BalanceComparison(
                balance_type="position",
                symbol=symbol,
                opening_value=opening,
                signed_fill_flow=flow,
                expected_value=expected,
                broker_value=actual,
                difference=actual - expected,
                tolerance=Decimal("0"),
                reconciled=reconciled,
            )
        )
        if broker_position is None and expected != 0:
            add(
                "position",
                "broker_position_missing",
                symbol=symbol,
                local_value=expected,
                broker_value="missing",
            )
        elif (
            broker_position is not None
            and symbol not in expected_by_symbol
            and actual != 0
        ):
            add(
                "position",
                "unexpected_broker_position",
                symbol=symbol,
                local_value="0",
                broker_value=actual,
            )
        elif broker_position is not None and not reconciled:
            add(
                "position",
                "position_quantity_mismatch",
                symbol=symbol,
                local_value=expected,
                broker_value=actual,
            )

    expected_cash = snapshot.opening_cash + cash_flow
    cash_difference = snapshot.broker_account.cash - expected_cash
    cash_reconciled = abs(cash_difference) <= CASH_TOLERANCE
    balance_rows.append(
        BalanceComparison(
            balance_type="cash",
            symbol="USD",
            opening_value=snapshot.opening_cash,
            signed_fill_flow=cash_flow,
            expected_value=expected_cash,
            broker_value=snapshot.broker_account.cash,
            difference=cash_difference,
            tolerance=CASH_TOLERANCE,
            reconciled=cash_reconciled,
        )
    )
    if not cash_reconciled or expected_cash < 0 or snapshot.broker_account.cash < 0:
        add(
            "cash",
            "cash_balance_mismatch",
            symbol="USD",
            local_value=expected_cash,
            broker_value=snapshot.broker_account.cash,
        )

    stale_records = (
        *(
            (
                "order",
                row.client_order_id,
                row.broker_order_id,
                row.symbol,
                row.snapshot_at,
            )
            for row in snapshot.broker_orders
        ),
        *(
            ("position", "", "", row.symbol, row.snapshot_at)
            for row in snapshot.broker_positions
        ),
        ("cash", "", "", "USD", snapshot.broker_account.snapshot_at),
    )
    for category, client_order_id, broker_order_id, symbol, timestamp in stale_records:
        age = snapshot.as_of - timestamp
        if age > SNAPSHOT_STALE_AFTER:
            add(
                category,
                "snapshot_stale",
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                symbol=symbol,
                local_value=f"max_seconds={int(SNAPSHOT_STALE_AFTER.total_seconds())}",
                broker_value=f"age_seconds={int(age.total_seconds())}",
            )

    for local in snapshot.local_states:
        if local.requested_quantity > MAX_SINGLE_ORDER_QUANTITY:
            add(
                "limit",
                "single_order_limit_exceeded",
                client_order_id=local.client_order_id,
                broker_order_id=local.broker_order_id or "",
                symbol=local.symbol,
                local_value=MAX_SINGLE_ORDER_QUANTITY,
                broker_value=local.requested_quantity,
            )
    open_order_count = sum(not row.terminal for row in snapshot.local_states)
    if open_order_count > MAX_OPEN_ORDERS:
        add(
            "limit",
            "open_order_limit_exceeded",
            local_value=MAX_OPEN_ORDERS,
            broker_value=open_order_count,
        )
    for position in snapshot.broker_positions:
        if abs(position.quantity) > MAX_ABSOLUTE_SYMBOL_POSITION:
            add(
                "limit",
                "symbol_position_limit_exceeded",
                symbol=position.symbol,
                local_value=MAX_ABSOLUTE_SYMBOL_POSITION,
                broker_value=abs(position.quantity),
            )
    gross_notional = sum(
        (
            abs(row.quantity) * row.mark_price
            for row in snapshot.broker_positions
        ),
        start=Decimal("0"),
    )
    if gross_notional > MAX_GROSS_NOTIONAL:
        add(
            "limit",
            "gross_notional_limit_exceeded",
            local_value=MAX_GROSS_NOTIONAL,
            broker_value=gross_notional,
        )

    observed_reasons = {row.reason_code for row in diagnostics}
    limits_passed = not bool(observed_reasons & LIMIT_REASON_CODES)
    core_reconciliation_passed = not bool(
        observed_reasons - LIMIT_REASON_CODES
    )
    return ReconciliationResult(
        diagnostics=tuple(diagnostics),
        balance_comparisons=tuple(balance_rows),
        local_order_count=len(snapshot.local_states),
        broker_order_count=len(snapshot.broker_orders),
        unique_fill_count=len(unique_fills),
        exact_duplicate_fill_count=exact_duplicates,
        broker_position_count=len(snapshot.broker_positions),
        expected_cash=expected_cash,
        broker_cash=snapshot.broker_account.cash,
        gross_notional=gross_notional,
        open_order_count=open_order_count,
        core_reconciliation_passed=core_reconciliation_passed,
        limits_passed=limits_passed,
        reconciliation_passed=not diagnostics,
    )


def freeze_reason_mapping(
    diagnostics: tuple[ReconciliationDiagnostic, ...],
) -> Mapping[str, int]:
    """Return immutable reason counts in the frozen vocabulary order."""

    counts = {reason: 0 for reason in RECONCILIATION_REASON_CODES}
    for row in diagnostics:
        counts[row.reason_code] += 1
    return MappingProxyType(counts)
