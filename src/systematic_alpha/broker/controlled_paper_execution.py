"""Fail-closed Day 21 Alpaca paper-order controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import time
from typing import Callable, Final, Mapping, Protocol

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from systematic_alpha.broker.day21_signal import Day21SignalSnapshot
from systematic_alpha.broker.paper_boundary import (
    ALPACA_PAPER_BASE_URL,
    AccountSnapshot,
    AlpacaPaperBroker,
    AssetSnapshot,
    MarketClockSnapshot,
    PreflightResult,
    run_paper_preflight,
    validate_day18_paper_config,
)
from systematic_alpha.data.config_loader import (
    AlpacaCredentials,
    load_alpaca_credentials,
    load_project_config,
)


DAY21_SCHEMA_VERSION: Final[str] = "day21_controlled_paper_execution_v1"
DAY21_SYMBOL: Final[str] = "SPY"
DAY21_QUANTITY: Final[Decimal] = Decimal("0.01")
DAY21_NOTIONAL_CAP: Final[Decimal] = Decimal("10.00")
MINIMUM_TIME_TO_CLOSE: Final[timedelta] = timedelta(minutes=30)
MAX_POLLS_PER_PHASE: Final[int] = 30
DAY21_AUTHORIZATION_SCOPE: Final[str] = "one_bounded_alpaca_paper_round_trip"
TERMINAL_ORDER_STATUSES: Final[frozenset[str]] = frozenset(
    {"filled", "canceled", "expired", "rejected"}
)
GATE_ORDER: Final[tuple[str, ...]] = (
    "explicit_authorization",
    "paper_preflight",
    "market_open_window",
    "day20_prerequisite",
    "post_lock_operational_data",
    "signal_available",
    "signal_nonzero",
    "entry_notional_cap",
    "no_existing_spy_position",
    "no_open_spy_order",
    "no_duplicate_signal_order",
    "kill_switch_armed",
)


class ControlledPaperExecutionError(RuntimeError):
    """Safe-to-persist controlled paper execution failure."""


@dataclass(frozen=True, slots=True)
class Day21Authorization:
    approved: bool
    scope: str
    paper_endpoint: str
    kill_switch_armed: bool
    source: str = "explicit_user_approval"

    def __post_init__(self) -> None:
        if type(self.approved) is not bool or not self.approved:
            raise ControlledPaperExecutionError(
                "Explicit Day 21 paper-order authorization is absent."
            )
        if self.scope != DAY21_AUTHORIZATION_SCOPE:
            raise ControlledPaperExecutionError(
                "Day 21 authorization scope is not exact."
            )
        if self.paper_endpoint.rstrip("/") != ALPACA_PAPER_BASE_URL:
            raise ControlledPaperExecutionError(
                "Day 21 authorization is not paper-endpoint specific."
            )
        if type(self.kill_switch_armed) is not bool or not self.kill_switch_armed:
            raise ControlledPaperExecutionError(
                "Day 21 authorization requires an armed kill switch."
            )
        if self.source != "explicit_user_approval":
            raise ControlledPaperExecutionError(
                "Day 21 authorization source is not recognized."
            )


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    requested_quantity: Decimal
    filled_quantity: Decimal
    filled_average_price: Decimal | None
    status: str
    submitted_at: datetime
    filled_at: datetime | None


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal | None
    market_value: Decimal | None
    unrealized_pl: Decimal | None


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    safe_detail: str


@dataclass(frozen=True, slots=True)
class PositionCashSnapshot:
    phase: str
    observed_at: datetime
    spy_quantity: Decimal
    cash: Decimal


@dataclass(frozen=True, slots=True)
class Day21ExecutionResult:
    schema_version: str
    signal: Day21SignalSnapshot
    gates: tuple[GateResult, ...]
    order_events: tuple[BrokerOrderSnapshot, ...]
    position_cash: tuple[PositionCashSnapshot, ...]
    entry_client_order_id: str
    flatten_client_order_id: str
    order_submission_occurred: bool
    entry_filled_quantity: Decimal
    flatten_filled_quantity: Decimal
    realized_round_trip_pnl: Decimal | None
    execution_complete: bool
    shutdown_reconciled: bool
    manual_recovery_required: bool
    outcome: str
    abort_reasons: tuple[str, ...]


class ControlledPaperBroker(Protocol):
    @property
    def paper_endpoint(self) -> str: ...

    def run_preflight(self) -> PreflightResult: ...

    def get_market_clock(self) -> MarketClockSnapshot: ...

    def get_open_orders(self, symbol: str) -> tuple[BrokerOrderSnapshot, ...]: ...

    def has_client_order_id(self, client_order_id: str) -> bool: ...

    def get_position(self, symbol: str) -> BrokerPositionSnapshot | None: ...

    def get_cash(self) -> Decimal: ...

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: str,
    ) -> BrokerOrderSnapshot: ...

    def get_order(self, broker_order_id: str) -> BrokerOrderSnapshot: ...

    def cancel_order(self, broker_order_id: str) -> None: ...


def _enum_text(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ControlledPaperExecutionError("Broker returned malformed text.")
    return value.strip().lower()


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ControlledPaperExecutionError(
                f"Broker response is missing {name}."
            )
        return value[name]
    if not hasattr(value, name):
        raise ControlledPaperExecutionError(f"Broker response is missing {name}.")
    return getattr(value, name)


def _decimal(value: object, *, name: str, allow_none: bool = False) -> Decimal | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ControlledPaperExecutionError(f"Broker {name} is malformed.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ControlledPaperExecutionError(f"Broker {name} is malformed.") from exc
    if not number.is_finite():
        raise ControlledPaperExecutionError(f"Broker {name} is non-finite.")
    return number


def _datetime(
    value: object,
    *,
    name: str,
    allow_none: bool = False,
) -> datetime | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ControlledPaperExecutionError(f"Broker {name} is malformed.")
    return value.astimezone(timezone.utc)


def _order_snapshot(response: object) -> BrokerOrderSnapshot:
    result = BrokerOrderSnapshot(
        broker_order_id=str(_field(response, "id")),
        client_order_id=str(_field(response, "client_order_id")),
        symbol=_enum_text(_field(response, "symbol")).upper(),
        side=_enum_text(_field(response, "side")),
        order_type=_enum_text(_field(response, "type")),
        time_in_force=_enum_text(_field(response, "time_in_force")),
        requested_quantity=_decimal(_field(response, "qty"), name="qty"),  # type: ignore[arg-type]
        filled_quantity=_decimal(
            _field(response, "filled_qty"), name="filled_qty"
        ),  # type: ignore[arg-type]
        filled_average_price=_decimal(
            _field(response, "filled_avg_price"),
            name="filled_avg_price",
            allow_none=True,
        ),
        status=_enum_text(_field(response, "status")),
        submitted_at=_datetime(
            _field(response, "submitted_at"), name="submitted_at"
        ),  # type: ignore[arg-type]
        filled_at=_datetime(
            _field(response, "filled_at"), name="filled_at", allow_none=True
        ),
    )
    if result.symbol != DAY21_SYMBOL:
        raise ControlledPaperExecutionError("Broker order symbol is outside Day 21.")
    if result.requested_quantity <= 0 or result.filled_quantity < 0:
        raise ControlledPaperExecutionError("Broker order quantities are invalid.")
    if result.filled_quantity > result.requested_quantity:
        raise ControlledPaperExecutionError("Broker overfill detected.")
    return result


class AlpacaControlledPaperBroker:
    """Narrow Alpaca adapter exposing only Day 21 scoped mutations."""

    def __init__(
        self,
        *,
        config: Mapping[str, object] | None = None,
        credentials: AlpacaCredentials | None = None,
        client: object | None = None,
    ) -> None:
        project_config = dict(config or load_project_config())
        endpoint = validate_day18_paper_config(project_config)
        if client is None:
            loaded = credentials or load_alpaca_credentials(project_config)
            client = TradingClient(
                loaded.api_key,
                loaded.secret_key,
                paper=True,
                raw_data=False,
                url_override=endpoint,
            )
        actual_endpoint = str(getattr(client, "_base_url", "")).rstrip("/")
        if actual_endpoint != ALPACA_PAPER_BASE_URL:
            raise ControlledPaperExecutionError(
                "Trading client is not bound to the Alpaca paper endpoint."
            )
        self._config = project_config
        self._client = client
        self._paper_endpoint = endpoint

    @property
    def paper_endpoint(self) -> str:
        return self._paper_endpoint

    def run_preflight(self) -> PreflightResult:
        read_only = AlpacaPaperBroker(config=self._config, client=self._client)
        return run_paper_preflight(read_only)

    def get_market_clock(self) -> MarketClockSnapshot:
        read_only = AlpacaPaperBroker(config=self._config, client=self._client)
        return read_only.get_market_clock_snapshot()

    def get_open_orders(self, symbol: str) -> tuple[BrokerOrderSnapshot, ...]:
        try:
            response = self._client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
        except Exception as exc:
            raise ControlledPaperExecutionError(
                "Open-order read failed."
            ) from exc
        return tuple(_order_snapshot(item) for item in response)

    def has_client_order_id(self, client_order_id: str) -> bool:
        try:
            response = self._client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    symbols=[DAY21_SYMBOL],
                    limit=500,
                )
            )
        except Exception as exc:
            raise ControlledPaperExecutionError(
                "Order-history read failed."
            ) from exc
        return any(str(_field(item, "client_order_id")) == client_order_id for item in response)

    def get_position(self, symbol: str) -> BrokerPositionSnapshot | None:
        try:
            response = self._client.get_all_positions()
        except Exception as exc:
            raise ControlledPaperExecutionError("Position read failed.") from exc
        matches = tuple(
            item
            for item in response
            if _enum_text(_field(item, "symbol")).upper() == symbol
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise ControlledPaperExecutionError("Duplicate broker position detected.")
        item = matches[0]
        return BrokerPositionSnapshot(
            symbol=symbol,
            quantity=_decimal(_field(item, "qty"), name="position qty"),  # type: ignore[arg-type]
            average_entry_price=_decimal(
                _field(item, "avg_entry_price"),
                name="average entry price",
                allow_none=True,
            ),
            market_value=_decimal(
                _field(item, "market_value"), name="market value", allow_none=True
            ),
            unrealized_pl=_decimal(
                _field(item, "unrealized_pl"),
                name="unrealized P&L",
                allow_none=True,
            ),
        )

    def get_cash(self) -> Decimal:
        try:
            response = self._client.get_account()
        except Exception as exc:
            raise ControlledPaperExecutionError("Account cash read failed.") from exc
        return _decimal(_field(response, "cash"), name="cash")  # type: ignore[return-value]

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: str,
    ) -> BrokerOrderSnapshot:
        if symbol != DAY21_SYMBOL or quantity <= 0 or quantity > DAY21_QUANTITY:
            raise ControlledPaperExecutionError("Order exceeds the Day 21 scope.")
        if side not in {"buy", "sell"}:
            raise ControlledPaperExecutionError("Order side is invalid.")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            extended_hours=False,
            client_order_id=client_order_id,
        )
        try:
            response = self._client.submit_order(request)
        except Exception as exc:
            raise ControlledPaperExecutionError("Paper order submission failed.") from exc
        return _order_snapshot(response)

    def get_order(self, broker_order_id: str) -> BrokerOrderSnapshot:
        try:
            response = self._client.get_order_by_id(broker_order_id)
        except Exception as exc:
            raise ControlledPaperExecutionError("Order refresh failed.") from exc
        return _order_snapshot(response)

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(broker_order_id)
        except Exception as exc:
            raise ControlledPaperExecutionError("Scoped order cancel failed.") from exc


def _position_quantity(position: BrokerPositionSnapshot | None) -> Decimal:
    return Decimal("0") if position is None else position.quantity


def _position_cash_snapshot(
    broker: ControlledPaperBroker,
    *,
    phase: str,
    observed_at: datetime,
) -> PositionCashSnapshot:
    return PositionCashSnapshot(
        phase=phase,
        observed_at=observed_at.astimezone(timezone.utc),
        spy_quantity=_position_quantity(broker.get_position(DAY21_SYMBOL)),
        cash=broker.get_cash(),
    )


def day21_client_order_ids(signal: Day21SignalSnapshot) -> tuple[str, str]:
    stamp = signal.bar_start.astimezone(timezone.utc).strftime("%Y%m%d%H%M")
    return (
        f"axiom-day21-spy-e-{stamp}",
        f"axiom-day21-spy-x-{stamp}",
    )


def evaluate_day21_startup_gates(
    *,
    paper_endpoint: str,
    preflight: PreflightResult,
    clock: MarketClockSnapshot,
    signal: Day21SignalSnapshot,
    authorization: Day21Authorization,
    day20_gate_passed: bool,
    open_orders: tuple[BrokerOrderSnapshot, ...],
    initial_position: BrokerPositionSnapshot | None,
    duplicate_signal_order: bool,
) -> tuple[GateResult, ...]:
    """Evaluate the exact frozen gates without performing any mutation."""

    last_close = Decimal(str(signal.last_close))
    gates = (
        GateResult("explicit_authorization", True, "exact_scope_approved"),
        GateResult(
            "paper_preflight",
            preflight.preflight_passed
            and paper_endpoint.rstrip("/") == ALPACA_PAPER_BASE_URL,
            "paper_endpoint_account_asset_checked",
        ),
        GateResult(
            "market_open_window",
            clock.is_open
            and clock.next_close - clock.timestamp >= MINIMUM_TIME_TO_CLOSE,
            "regular_hours_and_30_minutes_to_close_required",
        ),
        GateResult(
            "day20_prerequisite", day20_gate_passed, "synthetic_gate_required"
        ),
        GateResult(
            "post_lock_operational_data",
            not signal.locked_research_data_accessed
            and signal.data_start >= datetime(2026, 7, 1, tzinfo=timezone.utc),
            "data_start_on_or_after_2026_07_01",
        ),
        GateResult(
            "signal_available",
            signal.signal_available and signal.signal_fresh,
            "causal_warmup_and_maximum_20_minute_age_required",
        ),
        GateResult(
            "signal_nonzero",
            signal.position in {-1, 1},
            "position_must_be_plus_or_minus_one",
        ),
        GateResult(
            "entry_notional_cap",
            last_close * DAY21_QUANTITY <= DAY21_NOTIONAL_CAP,
            "last_close_times_0_01_not_above_10_usd",
        ),
        GateResult(
            "no_existing_spy_position",
            _position_quantity(initial_position) == 0,
            "startup_spy_quantity_must_be_zero",
        ),
        GateResult(
            "no_open_spy_order",
            not open_orders,
            "startup_open_spy_orders_must_be_zero",
        ),
        GateResult(
            "no_duplicate_signal_order",
            not duplicate_signal_order,
            "client_order_id_must_be_new",
        ),
        GateResult(
            "kill_switch_armed",
            authorization.kill_switch_armed,
            "latched_manual_control_required",
        ),
    )
    if tuple(item.gate_id for item in gates) != GATE_ORDER:
        raise RuntimeError("Day 21 gate order changed.")
    return gates


def _poll_order(
    broker: ControlledPaperBroker,
    initial: BrokerOrderSnapshot,
    *,
    sleep: Callable[[float], None],
) -> tuple[BrokerOrderSnapshot, tuple[BrokerOrderSnapshot, ...]]:
    current = initial
    events = [initial]
    for _ in range(MAX_POLLS_PER_PHASE):
        if current.status in TERMINAL_ORDER_STATUSES:
            return current, tuple(events)
        sleep(1.0)
        current = broker.get_order(current.broker_order_id)
        events.append(current)
    return current, tuple(events)


def run_controlled_paper_execution(
    broker: ControlledPaperBroker,
    *,
    signal: Day21SignalSnapshot,
    authorization: Day21Authorization,
    day20_gate_passed: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> Day21ExecutionResult:
    """Run at most one signal-driven entry and one shutdown flatten."""

    if broker.paper_endpoint.rstrip("/") != ALPACA_PAPER_BASE_URL:
        raise ControlledPaperExecutionError("Broker endpoint is not paper.")
    if not isinstance(signal, Day21SignalSnapshot):
        raise TypeError("signal must be a Day21SignalSnapshot.")
    if not isinstance(authorization, Day21Authorization):
        raise TypeError("authorization must be a Day21Authorization.")
    if type(day20_gate_passed) is not bool:
        raise TypeError("day20_gate_passed must be a boolean.")

    preflight = broker.run_preflight()
    clock = broker.get_market_clock()
    open_orders = broker.get_open_orders(DAY21_SYMBOL)
    initial_position = broker.get_position(DAY21_SYMBOL)
    entry_client_id, flatten_client_id = day21_client_order_ids(signal)
    duplicate = broker.has_client_order_id(entry_client_id)
    gates = evaluate_day21_startup_gates(
        paper_endpoint=broker.paper_endpoint,
        preflight=preflight,
        clock=clock,
        signal=signal,
        authorization=authorization,
        day20_gate_passed=day20_gate_passed,
        open_orders=open_orders,
        initial_position=initial_position,
        duplicate_signal_order=duplicate,
    )
    positions = (
        _position_cash_snapshot(
            broker, phase="startup", observed_at=clock.timestamp
        ),
    )
    failed = tuple(item.gate_id for item in gates if not item.passed)
    if failed:
        return Day21ExecutionResult(
            schema_version=DAY21_SCHEMA_VERSION,
            signal=signal,
            gates=gates,
            order_events=(),
            position_cash=positions,
            entry_client_order_id=entry_client_id,
            flatten_client_order_id=flatten_client_id,
            order_submission_occurred=False,
            entry_filled_quantity=Decimal("0"),
            flatten_filled_quantity=Decimal("0"),
            realized_round_trip_pnl=None,
            execution_complete=False,
            shutdown_reconciled=_position_quantity(initial_position) == 0 and not open_orders,
            manual_recovery_required=False,
            outcome="aborted_before_submission",
            abort_reasons=failed,
        )

    entry_side = "buy" if signal.position == 1 else "sell"
    entry = broker.submit_market_order(
        symbol=DAY21_SYMBOL,
        side=entry_side,
        quantity=DAY21_QUANTITY,
        client_order_id=entry_client_id,
    )
    entry_final, entry_events = _poll_order(broker, entry, sleep=sleep)
    all_events = list(entry_events)
    if entry_final.status not in TERMINAL_ORDER_STATUSES:
        broker.cancel_order(entry_final.broker_order_id)
        entry_final, cancel_events = _poll_order(broker, entry_final, sleep=sleep)
        all_events.extend(cancel_events[1:])

    entry_filled = entry_final.filled_quantity
    flatten_filled = Decimal("0")
    flatten_final: BrokerOrderSnapshot | None = None
    if entry_filled > 0:
        flatten_side = "sell" if entry_side == "buy" else "buy"
        flatten = broker.submit_market_order(
            symbol=DAY21_SYMBOL,
            side=flatten_side,
            quantity=entry_filled,
            client_order_id=flatten_client_id,
        )
        flatten_final, flatten_events = _poll_order(broker, flatten, sleep=sleep)
        all_events.extend(flatten_events)
        flatten_filled = flatten_final.filled_quantity

    final_clock = broker.get_market_clock()
    final_open = broker.get_open_orders(DAY21_SYMBOL)
    final_position = broker.get_position(DAY21_SYMBOL)
    positions = positions + (
        _position_cash_snapshot(
            broker, phase="shutdown", observed_at=final_clock.timestamp
        ),
    )
    flat = _position_quantity(final_position) == 0
    no_open = not final_open
    fills_match = entry_filled == flatten_filled
    shutdown_reconciled = flat and no_open and fills_match
    entry_price = entry_final.filled_average_price
    flatten_price = (
        None if flatten_final is None else flatten_final.filled_average_price
    )
    pnl: Decimal | None = None
    if (
        shutdown_reconciled
        and entry_filled > 0
        and entry_price is not None
        and flatten_price is not None
    ):
        direction = Decimal("1") if entry_side == "buy" else Decimal("-1")
        pnl = (flatten_price - entry_price) * entry_filled * direction
    execution_complete = (
        entry_filled > 0
        and flatten_filled == entry_filled
        and shutdown_reconciled
    )
    abort_reasons: list[str] = []
    if entry_filled == 0:
        abort_reasons.append(f"entry_{entry_final.status}")
    if not no_open:
        abort_reasons.append("open_spy_order_at_shutdown")
    if not flat:
        abort_reasons.append("nonzero_spy_position_at_shutdown")
    if not fills_match:
        abort_reasons.append("entry_flatten_fill_mismatch")
    return Day21ExecutionResult(
        schema_version=DAY21_SCHEMA_VERSION,
        signal=signal,
        gates=gates,
        order_events=tuple(all_events),
        position_cash=positions,
        entry_client_order_id=entry_client_id,
        flatten_client_order_id=flatten_client_id,
        order_submission_occurred=True,
        entry_filled_quantity=entry_filled,
        flatten_filled_quantity=flatten_filled,
        realized_round_trip_pnl=pnl,
        execution_complete=execution_complete,
        shutdown_reconciled=shutdown_reconciled,
        manual_recovery_required=not shutdown_reconciled,
        outcome=(
            "paper_round_trip_reconciled"
            if execution_complete
            else "paper_execution_incomplete"
        ),
        abort_reasons=tuple(abort_reasons),
    )
