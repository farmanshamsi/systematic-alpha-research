"""Fail-closed controller for the authorized Day 22 paper calibration campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import time
from typing import Callable, Final, Mapping, Protocol, Sequence, runtime_checkable

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from systematic_alpha.analysis.execution_performance_validation import build_campaign_schedule
from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    DAY21_NOTIONAL_CAP,
    DAY21_QUANTITY,
    DAY21_SYMBOL,
    MAX_POLLS_PER_PHASE,
    MINIMUM_TIME_TO_CLOSE,
    TERMINAL_ORDER_STATUSES,
    AlpacaControlledPaperBroker,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    ControlledPaperBroker,
    ControlledPaperExecutionError,
    GateResult,
    PositionCashSnapshot,
)
from systematic_alpha.broker.paper_boundary import MarketClockSnapshot, PreflightResult
from systematic_alpha.data.config_loader import (
    AlpacaCredentials,
    load_alpaca_credentials,
    load_project_config,
)


DAY22_LIVE_SCHEMA_VERSION: Final[str] = "day22_live_calibration_campaign_v1"
DAY22_CAMPAIGN_ID: Final[str] = "day22_calibration_v1"
DAY22_AUTHORIZATION_SCOPE: Final[str] = (
    "up_to_ten_bounded_alpaca_paper_calibration_round_trips"
)
DAY22_ACTIVATION_DATE: Final[date] = date(2026, 8, 3)
DAY22_SLOT_WINDOW: Final[timedelta] = timedelta(seconds=60)
DAY22_QUOTE_MAX_AGE: Final[timedelta] = timedelta(seconds=2)
DAY22_CLOCK_TOLERANCE: Final[timedelta] = timedelta(seconds=5)
DAY22_MAX_ROUND_TRIPS: Final[int] = 10
DAY22_MAX_SESSION_ROUND_TRIPS: Final[int] = 2
DAY22_PURPOSE: Final[str] = "calibration_probe"

DAY22_GATE_ORDER: Final[tuple[str, ...]] = (
    "explicit_authorization",
    "paper_endpoint",
    "frozen_campaign_slot",
    "slot_window",
    "paper_preflight",
    "market_open_window",
    "clock_synchronized",
    "day20_prerequisite",
    "campaign_not_latched",
    "campaign_total_limit",
    "campaign_session_limit",
    "no_existing_spy_position",
    "no_open_spy_order",
    "no_duplicate_entry_order",
    "kill_switch_armed",
    "fresh_valid_quote",
    "entry_notional_cap",
    "final_state_recheck",
)


class Day22CalibrationError(RuntimeError):
    """Safe-to-persist failure inside the bounded Day 22 controller."""


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise Day22CalibrationError(f"{name} is malformed.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise Day22CalibrationError(f"{name} is malformed.") from exc
    if not number.is_finite():
        raise Day22CalibrationError(f"{name} is non-finite.")
    return number


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Day22CalibrationError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _text(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise Day22CalibrationError("Broker returned malformed text.")
    return value.strip()


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise Day22CalibrationError(f"Broker response is missing {name}.")
        return value[name]
    if not hasattr(value, name):
        raise Day22CalibrationError(f"Broker response is missing {name}.")
    return getattr(value, name)


@dataclass(frozen=True, slots=True)
class Day22CampaignAuthorization:
    approved: bool
    scope: str
    paper_endpoint: str
    campaign_id: str
    activation_date: date
    maximum_entries: int
    maximum_flattens: int
    maximum_round_trips_per_session: int
    kill_switch_armed: bool
    source: str = "explicit_user_approval"

    def __post_init__(self) -> None:
        exact = (
            type(self.approved) is bool
            and self.approved
            and self.scope == DAY22_AUTHORIZATION_SCOPE
            and self.paper_endpoint.rstrip("/") == ALPACA_PAPER_BASE_URL
            and self.campaign_id == DAY22_CAMPAIGN_ID
            and self.activation_date == DAY22_ACTIVATION_DATE
            and self.maximum_entries == DAY22_MAX_ROUND_TRIPS
            and self.maximum_flattens == DAY22_MAX_ROUND_TRIPS
            and self.maximum_round_trips_per_session
            == DAY22_MAX_SESSION_ROUND_TRIPS
            and type(self.kill_switch_armed) is bool
            and self.kill_switch_armed
            and self.source == "explicit_user_approval"
        )
        if not exact:
            raise Day22CalibrationError(
                "Day 22 authorization does not match the frozen exact scope."
            )


@dataclass(frozen=True, slots=True)
class CampaignSlot:
    schedule_order: int
    campaign_id: str
    session_date: date
    scheduled_at: datetime
    scheduled_at_new_york: str
    entry_side: str
    quantity: Decimal
    maximum_notional_usd: Decimal
    purpose: str

    def __post_init__(self) -> None:
        if not 1 <= self.schedule_order <= DAY22_MAX_ROUND_TRIPS:
            raise Day22CalibrationError("Campaign schedule order is outside scope.")
        if self.campaign_id != DAY22_CAMPAIGN_ID:
            raise Day22CalibrationError("Campaign ID is outside the frozen scope.")
        if not isinstance(self.session_date, date) or isinstance(
            self.session_date, datetime
        ):
            raise Day22CalibrationError("Campaign session date is malformed.")
        object.__setattr__(
            self, "scheduled_at", _utc(self.scheduled_at, name="scheduled_at")
        )
        object.__setattr__(
            self, "quantity", _decimal(self.quantity, name="quantity")
        )
        object.__setattr__(
            self,
            "maximum_notional_usd",
            _decimal(self.maximum_notional_usd, name="maximum_notional_usd"),
        )
        if (
            self.entry_side not in {"buy", "sell"}
            or self.quantity != DAY21_QUANTITY
            or self.maximum_notional_usd != DAY21_NOTIONAL_CAP
            or self.purpose != DAY22_PURPOSE
            or self.scheduled_at.date() != self.session_date
        ):
            raise Day22CalibrationError("Campaign slot changed from the freeze.")


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    symbol: str
    quote_at: datetime
    bid_price: Decimal
    ask_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_at", _utc(self.quote_at, name="quote_at"))
        object.__setattr__(
            self, "bid_price", _decimal(self.bid_price, name="bid_price")
        )
        object.__setattr__(
            self, "ask_price", _decimal(self.ask_price, name="ask_price")
        )
        if self.symbol != DAY21_SYMBOL:
            raise Day22CalibrationError("Quote symbol is outside Day 22.")
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise Day22CalibrationError("Quote prices must be positive.")
        if self.bid_price > self.ask_price:
            raise Day22CalibrationError("Quote is crossed.")

    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / Decimal("2")


@dataclass(frozen=True, slots=True)
class CampaignLegEvidence:
    leg: str
    quote: QuoteSnapshot | None
    local_submitted_at: datetime
    order_events: tuple[BrokerOrderSnapshot, ...]

    def __post_init__(self) -> None:
        if self.leg not in {"entry", "exit"}:
            raise Day22CalibrationError("Campaign leg is malformed.")
        object.__setattr__(
            self,
            "local_submitted_at",
            _utc(self.local_submitted_at, name="local_submitted_at"),
        )
        if not self.order_events:
            raise Day22CalibrationError("Campaign leg has no order events.")


@dataclass(frozen=True, slots=True)
class Day22SlotResult:
    schema_version: str
    slot: CampaignSlot
    gates: tuple[GateResult, ...]
    entry_client_order_id: str
    flatten_client_order_id: str
    legs: tuple[CampaignLegEvidence, ...]
    position_cash: tuple[PositionCashSnapshot, ...]
    entry_submission_occurred: bool
    flatten_submission_occurred: bool
    entry_filled_quantity: Decimal
    flatten_filled_quantity: Decimal
    realized_round_trip_pnl: Decimal | None
    execution_complete: bool
    shutdown_reconciled: bool
    manual_recovery_required: bool
    outcome: str
    abort_reasons: tuple[str, ...]


@runtime_checkable
class Day22CampaignBroker(ControlledPaperBroker, Protocol):
    def get_latest_quote(self, symbol: str) -> QuoteSnapshot: ...


class AlpacaDay22CampaignBroker(AlpacaControlledPaperBroker):
    """Paper-only trading adapter plus a credential-safe latest-quote read."""

    def __init__(
        self,
        *,
        config: Mapping[str, object] | None = None,
        credentials: AlpacaCredentials | None = None,
        trading_client: object | None = None,
        data_client: object | None = None,
    ) -> None:
        project_config = dict(config or load_project_config())
        loaded = credentials
        if trading_client is None or data_client is None:
            loaded = loaded or load_alpaca_credentials(project_config)
        super().__init__(
            config=project_config,
            credentials=loaded,
            client=trading_client,
        )
        if data_client is None:
            if loaded is None:  # pragma: no cover - defensive invariant
                raise Day22CalibrationError("Alpaca credentials were not loaded.")
            data_client = StockHistoricalDataClient(
                api_key=loaded.api_key,
                secret_key=loaded.secret_key,
            )
        self._data_client = data_client
        feed_name = str(project_config["broker"].get("stock_data_feed", "iex"))
        try:
            self._feed = DataFeed(feed_name.lower())
        except ValueError as exc:
            raise Day22CalibrationError("Configured stock data feed is invalid.") from exc

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        if symbol != DAY21_SYMBOL:
            raise Day22CalibrationError("Latest-quote request is outside Day 22.")
        request = StockLatestQuoteRequest(
            symbol_or_symbols=[symbol],
            feed=self._feed,
        )
        try:
            response = self._data_client.get_stock_latest_quote(request)
            item = response[symbol]
        except Exception as exc:
            raise Day22CalibrationError("Latest-quote read failed.") from exc
        return QuoteSnapshot(
            symbol=symbol,
            quote_at=_field(item, "timestamp"),  # type: ignore[arg-type]
            bid_price=_field(item, "bid_price"),  # type: ignore[arg-type]
            ask_price=_field(item, "ask_price"),  # type: ignore[arg-type]
        )


def frozen_campaign_slots() -> tuple[CampaignSlot, ...]:
    """Return the exact authorized August 3-7 schedule."""

    rows = build_campaign_schedule(
        DAY22_ACTIVATION_DATE,
        campaign_id=DAY22_CAMPAIGN_ID,
    )
    slots = tuple(
        CampaignSlot(
            schedule_order=int(row["schedule_order"]),
            campaign_id=str(row["campaign_id"]),
            session_date=date.fromisoformat(str(row["session_date"])),
            scheduled_at=datetime.fromisoformat(str(row["scheduled_at_utc"])),
            scheduled_at_new_york=str(row["scheduled_at_new_york"]),
            entry_side=str(row["entry_side"]),
            quantity=_decimal(row["quantity"], name="quantity"),
            maximum_notional_usd=_decimal(
                row["maximum_notional_usd"], name="maximum_notional_usd"
            ),
            purpose=str(row["purpose"]),
        )
        for row in rows
    )
    expected_sides = ("buy", "sell") * 5
    if tuple(item.entry_side for item in slots) != expected_sides:
        raise RuntimeError("Frozen Day 22 side order changed.")
    return slots


def day22_client_order_ids(slot: CampaignSlot) -> tuple[str, str]:
    stamp = slot.scheduled_at.strftime("%Y%m%d%H%M")
    return (
        f"axiom-d22-c{slot.schedule_order:02d}-e-{stamp}",
        f"axiom-d22-c{slot.schedule_order:02d}-x-{stamp}",
    )


def _position_quantity(position: BrokerPositionSnapshot | None) -> Decimal:
    return Decimal("0") if position is None else position.quantity


def _position_cash_snapshot(
    broker: Day22CampaignBroker,
    *,
    phase: str,
    observed_at: datetime,
) -> PositionCashSnapshot:
    return PositionCashSnapshot(
        phase=phase,
        observed_at=_utc(observed_at, name="observed_at"),
        spy_quantity=_position_quantity(broker.get_position(DAY21_SYMBOL)),
        cash=broker.get_cash(),
    )


def _poll_order(
    broker: Day22CampaignBroker,
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


def _frozen_slot_matches(slot: CampaignSlot) -> bool:
    return slot in frozen_campaign_slots()


def _slot_window_passed(slot: CampaignSlot, now: datetime) -> bool:
    return slot.scheduled_at <= now < slot.scheduled_at + DAY22_SLOT_WINDOW


def _clock_is_synchronized(clock: MarketClockSnapshot, now: datetime) -> bool:
    return abs(clock.timestamp - now) <= DAY22_CLOCK_TOLERANCE


def _base_gates(
    *,
    broker: Day22CampaignBroker,
    slot: CampaignSlot,
    authorization: Day22CampaignAuthorization,
    day20_gate_passed: bool,
    prior_entry_attempts_total: int,
    prior_entry_attempts_session: int,
    manual_recovery_latched: bool,
    slot_already_consumed: bool,
    now: datetime,
    preflight: PreflightResult,
    clock: MarketClockSnapshot,
    open_orders: tuple[BrokerOrderSnapshot, ...],
    position: BrokerPositionSnapshot | None,
    duplicate_entry: bool,
) -> tuple[GateResult, ...]:
    return (
        GateResult("explicit_authorization", authorization.approved, "exact_scope_approved"),
        GateResult(
            "paper_endpoint",
            broker.paper_endpoint.rstrip("/") == ALPACA_PAPER_BASE_URL,
            "alpaca_paper_endpoint_required",
        ),
        GateResult(
            "frozen_campaign_slot",
            _frozen_slot_matches(slot) and not slot_already_consumed,
            "exact_unconsumed_august_3_to_7_slot_required",
        ),
        GateResult(
            "slot_window",
            _slot_window_passed(slot, now),
            "scheduled_time_inclusive_and_60_second_window_exclusive",
        ),
        GateResult(
            "paper_preflight",
            preflight.preflight_passed,
            "paper_account_and_spy_asset_checked",
        ),
        GateResult(
            "market_open_window",
            clock.is_open
            and clock.next_close - clock.timestamp >= MINIMUM_TIME_TO_CLOSE,
            "regular_hours_and_30_minutes_to_close_required",
        ),
        GateResult(
            "clock_synchronized",
            _clock_is_synchronized(clock, now),
            "broker_and_local_utc_clocks_within_5_seconds",
        ),
        GateResult("day20_prerequisite", day20_gate_passed, "day20_hash_gate_required"),
        GateResult(
            "campaign_not_latched",
            not manual_recovery_latched,
            "manual_recovery_latch_must_be_clear",
        ),
        GateResult(
            "campaign_total_limit",
            0 <= prior_entry_attempts_total < DAY22_MAX_ROUND_TRIPS,
            "fewer_than_10_prior_entry_attempts_required",
        ),
        GateResult(
            "campaign_session_limit",
            0 <= prior_entry_attempts_session < DAY22_MAX_SESSION_ROUND_TRIPS,
            "fewer_than_2_prior_session_entry_attempts_required",
        ),
        GateResult(
            "no_existing_spy_position",
            _position_quantity(position) == 0,
            "startup_spy_quantity_must_be_zero",
        ),
        GateResult(
            "no_open_spy_order",
            not open_orders,
            "startup_open_spy_orders_must_be_zero",
        ),
        GateResult(
            "no_duplicate_entry_order",
            not duplicate_entry,
            "frozen_entry_client_order_id_must_be_new",
        ),
        GateResult(
            "kill_switch_armed",
            authorization.kill_switch_armed,
            "latched_manual_control_required",
        ),
    )


def _not_evaluated_gates() -> tuple[GateResult, ...]:
    return (
        GateResult("fresh_valid_quote", False, "not_evaluated_due_to_prior_failure"),
        GateResult("entry_notional_cap", False, "not_evaluated_due_to_prior_failure"),
        GateResult("final_state_recheck", False, "not_evaluated_due_to_prior_failure"),
    )


def _empty_result(
    *,
    slot: CampaignSlot,
    gates: tuple[GateResult, ...],
    positions: tuple[PositionCashSnapshot, ...],
    outcome: str,
    abort_reasons: Sequence[str],
    manual_recovery_required: bool = False,
) -> Day22SlotResult:
    entry_id, flatten_id = day22_client_order_ids(slot)
    return Day22SlotResult(
        schema_version=DAY22_LIVE_SCHEMA_VERSION,
        slot=slot,
        gates=gates,
        entry_client_order_id=entry_id,
        flatten_client_order_id=flatten_id,
        legs=(),
        position_cash=positions,
        entry_submission_occurred=False,
        flatten_submission_occurred=False,
        entry_filled_quantity=Decimal("0"),
        flatten_filled_quantity=Decimal("0"),
        realized_round_trip_pnl=None,
        execution_complete=False,
        shutdown_reconciled=not manual_recovery_required,
        manual_recovery_required=manual_recovery_required,
        outcome=outcome,
        abort_reasons=tuple(abort_reasons),
    )


def run_day22_calibration_slot(
    broker: Day22CampaignBroker,
    *,
    slot: CampaignSlot,
    authorization: Day22CampaignAuthorization,
    day20_gate_passed: bool,
    prior_entry_attempts_total: int,
    prior_entry_attempts_session: int,
    manual_recovery_latched: bool,
    slot_already_consumed: bool,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    before_entry_submit: Callable[[], None] = lambda: None,
) -> Day22SlotResult:
    """Attempt one frozen slot and immediately flatten any confirmed fill."""

    if not isinstance(broker, Day22CampaignBroker):
        raise TypeError("broker must implement Day22CampaignBroker.")
    if not isinstance(slot, CampaignSlot):
        raise TypeError("slot must be a CampaignSlot.")
    if not isinstance(authorization, Day22CampaignAuthorization):
        raise TypeError("authorization must be Day22CampaignAuthorization.")
    for value, name in (
        (day20_gate_passed, "day20_gate_passed"),
        (manual_recovery_latched, "manual_recovery_latched"),
        (slot_already_consumed, "slot_already_consumed"),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean.")
    if (
        type(prior_entry_attempts_total) is not int
        or type(prior_entry_attempts_session) is not int
    ):
        raise TypeError("Campaign attempt counts must be integers.")
    if broker.paper_endpoint.rstrip("/") != ALPACA_PAPER_BASE_URL:
        raise Day22CalibrationError("Broker endpoint is not paper.")

    observed_now = _utc(now(), name="now")
    preflight = broker.run_preflight()
    clock = broker.get_market_clock()
    open_orders = broker.get_open_orders(DAY21_SYMBOL)
    initial_position = broker.get_position(DAY21_SYMBOL)
    entry_id, flatten_id = day22_client_order_ids(slot)
    duplicate_entry = broker.has_client_order_id(entry_id)
    positions = (
        _position_cash_snapshot(
            broker,
            phase="startup",
            observed_at=observed_now,
        ),
    )
    gates = _base_gates(
        broker=broker,
        slot=slot,
        authorization=authorization,
        day20_gate_passed=day20_gate_passed,
        prior_entry_attempts_total=prior_entry_attempts_total,
        prior_entry_attempts_session=prior_entry_attempts_session,
        manual_recovery_latched=manual_recovery_latched,
        slot_already_consumed=slot_already_consumed,
        now=observed_now,
        preflight=preflight,
        clock=clock,
        open_orders=open_orders,
        position=initial_position,
        duplicate_entry=duplicate_entry,
    )
    failed = tuple(item.gate_id for item in gates if not item.passed)
    if failed:
        return _empty_result(
            slot=slot,
            gates=gates + _not_evaluated_gates(),
            positions=positions,
            outcome="skipped_before_submission",
            abort_reasons=failed,
        )

    try:
        entry_quote = broker.get_latest_quote(DAY21_SYMBOL)
    except Exception:
        quote_gates = (
            GateResult("fresh_valid_quote", False, "latest_quote_read_failed"),
            GateResult("entry_notional_cap", False, "not_evaluated_without_quote"),
            GateResult("final_state_recheck", False, "not_evaluated_without_quote"),
        )
        return _empty_result(
            slot=slot,
            gates=gates + quote_gates,
            positions=positions,
            outcome="skipped_before_submission",
            abort_reasons=("fresh_valid_quote",),
        )
    final_clock = broker.get_market_clock()
    final_open = broker.get_open_orders(DAY21_SYMBOL)
    final_position = broker.get_position(DAY21_SYMBOL)
    duplicate_after_quote = broker.has_client_order_id(entry_id)
    submit_ready_at = _utc(now(), name="submit_ready_at")
    quote_age = submit_ready_at - entry_quote.quote_at
    fresh = timedelta(0) <= quote_age <= DAY22_QUOTE_MAX_AGE
    notional = entry_quote.mid_price * slot.quantity
    final_recheck = (
        _slot_window_passed(slot, submit_ready_at)
        and final_clock.is_open
        and final_clock.next_close - final_clock.timestamp >= MINIMUM_TIME_TO_CLOSE
        and _clock_is_synchronized(final_clock, submit_ready_at)
        and not final_open
        and _position_quantity(final_position) == 0
        and not duplicate_after_quote
    )
    quote_gates = (
        GateResult(
            "fresh_valid_quote",
            fresh,
            "noncrossed_positive_quote_no_more_than_2_seconds_old",
        ),
        GateResult(
            "entry_notional_cap",
            notional <= slot.maximum_notional_usd,
            "arrival_mid_times_0_01_not_above_10_usd",
        ),
        GateResult(
            "final_state_recheck",
            final_recheck,
            "slot_clock_position_orders_and_duplicate_rechecked",
        ),
    )
    gates = gates + quote_gates
    if tuple(item.gate_id for item in gates) != DAY22_GATE_ORDER:
        raise RuntimeError("Day 22 gate order changed.")
    failed = tuple(item.gate_id for item in quote_gates if not item.passed)
    if failed:
        return _empty_result(
            slot=slot,
            gates=gates,
            positions=positions,
            outcome="skipped_before_submission",
            abort_reasons=failed,
        )

    before_entry_submit()
    entry_local_submit = _utc(now(), name="entry_local_submit")
    if (
        entry_local_submit - entry_quote.quote_at > DAY22_QUOTE_MAX_AGE
        or entry_local_submit < entry_quote.quote_at
        or not _slot_window_passed(slot, entry_local_submit)
    ):
        revised = tuple(
            GateResult(
                item.gate_id,
                False,
                "failed_after_persisted_intent_before_broker_submission",
            )
            if item.gate_id in {"fresh_valid_quote", "final_state_recheck"}
            else item
            for item in gates
        )
        return _empty_result(
            slot=slot,
            gates=revised,
            positions=positions,
            outcome="skipped_before_submission",
            abort_reasons=("post_intent_submission_recheck",),
        )
    entry = broker.submit_market_order(
        symbol=DAY21_SYMBOL,
        side=slot.entry_side,
        quantity=slot.quantity,
        client_order_id=entry_id,
    )
    entry_final, entry_events = _poll_order(broker, entry, sleep=sleep)
    if entry_final.status not in TERMINAL_ORDER_STATUSES:
        broker.cancel_order(entry_final.broker_order_id)
        entry_final, cancel_events = _poll_order(broker, entry_final, sleep=sleep)
        entry_events = entry_events + cancel_events[1:]

    legs: list[CampaignLegEvidence] = [
        CampaignLegEvidence(
            leg="entry",
            quote=entry_quote,
            local_submitted_at=entry_local_submit,
            order_events=entry_events,
        )
    ]
    entry_filled = entry_final.filled_quantity
    flatten_filled = Decimal("0")
    flatten_submitted = False
    flatten_final: BrokerOrderSnapshot | None = None
    if entry_filled > 0:
        candidate_quote: QuoteSnapshot | None = None
        try:
            candidate_quote = broker.get_latest_quote(DAY21_SYMBOL)
        except Exception:
            pass
        flatten_local_submit = _utc(now(), name="flatten_local_submit")
        flatten_quote: QuoteSnapshot | None = None
        if candidate_quote is not None:
            flatten_age = flatten_local_submit - candidate_quote.quote_at
            flatten_quote = (
                candidate_quote
                if timedelta(0) <= flatten_age <= DAY22_QUOTE_MAX_AGE
                else None
            )
        flatten_side = "sell" if slot.entry_side == "buy" else "buy"
        flatten = broker.submit_market_order(
            symbol=DAY21_SYMBOL,
            side=flatten_side,
            quantity=entry_filled,
            client_order_id=flatten_id,
        )
        flatten_submitted = True
        flatten_final, flatten_events = _poll_order(broker, flatten, sleep=sleep)
        if flatten_final.status not in TERMINAL_ORDER_STATUSES:
            broker.cancel_order(flatten_final.broker_order_id)
            flatten_final, cancel_events = _poll_order(
                broker, flatten_final, sleep=sleep
            )
            flatten_events = flatten_events + cancel_events[1:]
        flatten_filled = flatten_final.filled_quantity
        legs.append(
            CampaignLegEvidence(
                leg="exit",
                quote=flatten_quote,
                local_submitted_at=flatten_local_submit,
                order_events=flatten_events,
            )
        )

    shutdown_at = _utc(now(), name="shutdown_at")
    shutdown_open = broker.get_open_orders(DAY21_SYMBOL)
    shutdown_position = broker.get_position(DAY21_SYMBOL)
    positions = positions + (
        _position_cash_snapshot(
            broker,
            phase="shutdown",
            observed_at=shutdown_at,
        ),
    )
    flat = _position_quantity(shutdown_position) == 0
    no_open = not shutdown_open
    fills_match = entry_filled == flatten_filled
    shutdown_reconciled = flat and no_open and fills_match
    pnl: Decimal | None = None
    if (
        shutdown_reconciled
        and entry_filled > 0
        and entry_final.filled_average_price is not None
        and flatten_final is not None
        and flatten_final.filled_average_price is not None
    ):
        direction = Decimal("1") if slot.entry_side == "buy" else Decimal("-1")
        pnl = (
            flatten_final.filled_average_price - entry_final.filled_average_price
        ) * entry_filled * direction
    execution_complete = entry_filled > 0 and shutdown_reconciled
    abort_reasons: list[str] = []
    if entry_filled == 0:
        abort_reasons.append(f"entry_{entry_final.status}")
    if not no_open:
        abort_reasons.append("open_spy_order_at_shutdown")
    if not flat:
        abort_reasons.append("nonzero_spy_position_at_shutdown")
    if not fills_match:
        abort_reasons.append("entry_flatten_fill_mismatch")
    if flatten_submitted and legs[-1].quote is None:
        abort_reasons.append("flatten_quote_unavailable")
    return Day22SlotResult(
        schema_version=DAY22_LIVE_SCHEMA_VERSION,
        slot=slot,
        gates=gates,
        entry_client_order_id=entry_id,
        flatten_client_order_id=flatten_id,
        legs=tuple(legs),
        position_cash=positions,
        entry_submission_occurred=True,
        flatten_submission_occurred=flatten_submitted,
        entry_filled_quantity=entry_filled,
        flatten_filled_quantity=flatten_filled,
        realized_round_trip_pnl=pnl,
        execution_complete=execution_complete,
        shutdown_reconciled=shutdown_reconciled,
        manual_recovery_required=not shutdown_reconciled,
        outcome=(
            "paper_calibration_round_trip_reconciled"
            if execution_complete
            else "paper_calibration_execution_incomplete"
        ),
        abort_reasons=tuple(abort_reasons),
    )


def authorized_day22_campaign() -> Day22CampaignAuthorization:
    """Construct the exact explicit authorization recorded on 2026-08-03."""

    return Day22CampaignAuthorization(
        approved=True,
        scope=DAY22_AUTHORIZATION_SCOPE,
        paper_endpoint=ALPACA_PAPER_BASE_URL,
        campaign_id=DAY22_CAMPAIGN_ID,
        activation_date=DAY22_ACTIVATION_DATE,
        maximum_entries=DAY22_MAX_ROUND_TRIPS,
        maximum_flattens=DAY22_MAX_ROUND_TRIPS,
        maximum_round_trips_per_session=DAY22_MAX_SESSION_ROUND_TRIPS,
        kill_switch_armed=True,
    )
