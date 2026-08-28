from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
)
from systematic_alpha.broker.day22_calibration_campaign import (
    DAY22_AUTHORIZATION_SCOPE,
    DAY22_CAMPAIGN_ID,
    AlpacaDay22CampaignBroker,
    CampaignSlot,
    Day22CalibrationError,
    Day22CampaignAuthorization,
    QuoteSnapshot,
    authorized_day22_campaign,
    day22_client_order_ids,
    frozen_campaign_slots,
    run_day22_calibration_slot,
)
from systematic_alpha.broker.paper_boundary import MarketClockSnapshot
from tests.day18_fixtures import safe_day18_config


NOW = datetime(2026, 8, 3, 14, 15, 10, tzinfo=timezone.utc)


def order_snapshot(
    *,
    broker_id: str,
    client_id: str,
    side: str,
    quantity: Decimal,
    status: str,
    filled_quantity: Decimal,
) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id=broker_id,
        client_order_id=client_id,
        symbol="SPY",
        side=side,
        order_type="market",
        time_in_force="day",
        requested_quantity=quantity,
        filled_quantity=filled_quantity,
        filled_average_price=(Decimal("600.02") if filled_quantity else None),
        status=status,
        submitted_at=NOW,
        filled_at=NOW + timedelta(milliseconds=200) if filled_quantity else None,
    )


class FakeCampaignBroker:
    paper_endpoint = ALPACA_PAPER_BASE_URL

    def __init__(self) -> None:
        self.position = Decimal("0")
        self.cash = Decimal("100000")
        self.open_orders: tuple[BrokerOrderSnapshot, ...] = ()
        self.duplicate = False
        self.market_open = True
        self.quote = QuoteSnapshot(
            symbol="SPY",
            quote_at=NOW - timedelta(milliseconds=100),
            bid_price=Decimal("599.98"),
            ask_price=Decimal("600.02"),
        )
        self.submissions: list[BrokerOrderSnapshot] = []
        self.orders: dict[str, BrokerOrderSnapshot] = {}
        self.quote_reads = 0

    def run_preflight(self):
        return SimpleNamespace(preflight_passed=True)

    def get_market_clock(self) -> MarketClockSnapshot:
        clock_at = self.quote.quote_at + timedelta(milliseconds=100)
        return MarketClockSnapshot(
            timestamp=clock_at,
            is_open=self.market_open,
            next_open=clock_at + timedelta(days=1),
            next_close=clock_at + timedelta(hours=5),
        )

    def get_open_orders(self, symbol: str):
        return self.open_orders

    def has_client_order_id(self, client_order_id: str) -> bool:
        return self.duplicate or any(
            item.client_order_id == client_order_id for item in self.submissions
        )

    def get_position(self, symbol: str):
        if self.position == 0:
            return None
        return BrokerPositionSnapshot("SPY", self.position, None, None, None)

    def get_cash(self) -> Decimal:
        return self.cash

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        self.quote_reads += 1
        return self.quote

    def submit_market_order(self, *, symbol, side, quantity, client_order_id):
        broker_id = f"broker-{len(self.submissions) + 1}"
        submitted = order_snapshot(
            broker_id=broker_id,
            client_id=client_order_id,
            side=side,
            quantity=quantity,
            status="new",
            filled_quantity=Decimal("0"),
        )
        self.submissions.append(submitted)
        self.orders[broker_id] = submitted
        return submitted

    def get_order(self, broker_order_id: str):
        current = self.orders[broker_order_id]
        filled = replace(
            current,
            status="filled",
            filled_quantity=current.requested_quantity,
            filled_average_price=(
                Decimal("600.02") if current.side == "buy" else Decimal("599.98")
            ),
            filled_at=NOW + timedelta(milliseconds=200),
        )
        signed = filled.filled_quantity if filled.side == "buy" else -filled.filled_quantity
        self.position += signed
        self.orders[broker_order_id] = filled
        return filled

    def cancel_order(self, broker_order_id: str) -> None:
        self.orders[broker_order_id] = replace(
            self.orders[broker_order_id], status="canceled"
        )


def run(broker: FakeCampaignBroker, **overrides: object):
    values = {
        "slot": frozen_campaign_slots()[0],
        "authorization": authorized_day22_campaign(),
        "day20_gate_passed": True,
        "prior_entry_attempts_total": 0,
        "prior_entry_attempts_session": 0,
        "manual_recovery_latched": False,
        "slot_already_consumed": False,
        "now": lambda: NOW,
        "sleep": lambda _: None,
    }
    values.update(overrides)
    return run_day22_calibration_slot(broker, **values)  # type: ignore[arg-type]


def test_frozen_schedule_and_client_ids_are_exact() -> None:
    slots = frozen_campaign_slots()
    assert len(slots) == 10
    assert [slot.entry_side for slot in slots] == ["buy", "sell"] * 5
    assert {slot.scheduled_at.hour for slot in slots} == {14, 18}
    assert day22_client_order_ids(slots[0]) == (
        "axiom-d22-c01-e-202608031415",
        "axiom-d22-c01-x-202608031415",
    )


def test_outside_exact_window_skips_without_quote_or_order() -> None:
    broker = FakeCampaignBroker()
    result = run(broker, now=lambda: NOW - timedelta(seconds=11))
    assert result.outcome == "skipped_before_submission"
    assert "slot_window" in result.abort_reasons
    assert broker.quote_reads == 0
    assert not broker.submissions


@pytest.mark.parametrize("schedule_index", [0, 1])
def test_buy_and_sell_round_trips_fill_and_reconcile(schedule_index: int) -> None:
    broker = FakeCampaignBroker()
    slot = frozen_campaign_slots()[schedule_index]
    slot_now = slot.scheduled_at + timedelta(seconds=10)
    broker.quote = replace(broker.quote, quote_at=slot_now - timedelta(milliseconds=100))
    result = run(broker, slot=slot, now=lambda: slot_now)
    assert result.entry_submission_occurred
    assert result.flatten_submission_occurred
    assert result.entry_filled_quantity == Decimal("0.01")
    assert result.flatten_filled_quantity == Decimal("0.01")
    assert result.execution_complete
    assert result.shutdown_reconciled
    assert not result.manual_recovery_required
    assert [item.side for item in broker.submissions] == [
        slot.entry_side,
        "sell" if slot.entry_side == "buy" else "buy",
    ]


def test_stale_quote_and_notional_cap_fail_closed() -> None:
    stale = FakeCampaignBroker()
    stale.quote = replace(stale.quote, quote_at=NOW - timedelta(seconds=3))
    stale_result = run(stale)
    assert "fresh_valid_quote" in stale_result.abort_reasons
    assert not stale.submissions

    expensive = FakeCampaignBroker()
    expensive.quote = replace(
        expensive.quote,
        bid_price=Decimal("1000.01"),
        ask_price=Decimal("1000.03"),
    )
    expensive_result = run(expensive)
    assert "entry_notional_cap" in expensive_result.abort_reasons
    assert not expensive.submissions


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"manual_recovery_latched": True}, "campaign_not_latched"),
        ({"prior_entry_attempts_total": 10}, "campaign_total_limit"),
        ({"prior_entry_attempts_session": 2}, "campaign_session_limit"),
        ({"slot_already_consumed": True}, "frozen_campaign_slot"),
        ({"day20_gate_passed": False}, "day20_prerequisite"),
    ],
)
def test_campaign_state_and_count_gates_fail_closed(overrides, reason) -> None:
    broker = FakeCampaignBroker()
    result = run(broker, **overrides)
    assert reason in result.abort_reasons
    assert not broker.submissions


def test_position_open_order_and_duplicate_fail_closed() -> None:
    broker = FakeCampaignBroker()
    broker.position = Decimal("0.01")
    broker.duplicate = True
    broker.open_orders = (
        order_snapshot(
            broker_id="existing",
            client_id="strategy-order",
            side="buy",
            quantity=Decimal("0.01"),
            status="new",
            filled_quantity=Decimal("0"),
        ),
    )
    result = run(broker)
    assert set(result.abort_reasons) >= {
        "no_existing_spy_position",
        "no_open_spy_order",
        "no_duplicate_entry_order",
    }
    assert not broker.submissions


def test_intent_callback_runs_before_first_submission() -> None:
    broker = FakeCampaignBroker()
    events: list[str] = []

    def before() -> None:
        assert not broker.submissions
        events.append("intent_saved")

    result = run(broker, before_entry_submit=before)
    assert result.execution_complete
    assert events == ["intent_saved"]


def test_authorization_rejects_broader_scope() -> None:
    with pytest.raises(Day22CalibrationError, match="exact scope"):
        Day22CampaignAuthorization(
            approved=True,
            scope="all_paper_orders",
            paper_endpoint=ALPACA_PAPER_BASE_URL,
            campaign_id=DAY22_CAMPAIGN_ID,
            activation_date=frozen_campaign_slots()[0].session_date,
            maximum_entries=10,
            maximum_flattens=10,
            maximum_round_trips_per_session=2,
            kill_switch_armed=True,
        )


def test_changed_slot_is_rejected() -> None:
    slot = frozen_campaign_slots()[0]
    with pytest.raises(Day22CalibrationError, match="changed"):
        CampaignSlot(
            schedule_order=slot.schedule_order,
            campaign_id=slot.campaign_id,
            session_date=slot.session_date,
            scheduled_at=slot.scheduled_at,
            scheduled_at_new_york=slot.scheduled_at_new_york,
            entry_side=slot.entry_side,
            quantity=Decimal("0.02"),
            maximum_notional_usd=slot.maximum_notional_usd,
            purpose=slot.purpose,
        )


def test_alpaca_adapter_normalizes_latest_quote() -> None:
    trading_client = SimpleNamespace(_base_url=ALPACA_PAPER_BASE_URL)
    quote = SimpleNamespace(
        timestamp=NOW,
        bid_price=Decimal("599.99"),
        ask_price=Decimal("600.01"),
    )

    class DataClient:
        def get_stock_latest_quote(self, request):
            assert request.symbol_or_symbols == ["SPY"]
            return {"SPY": quote}

    broker = AlpacaDay22CampaignBroker(
        config=safe_day18_config(),
        trading_client=trading_client,
        data_client=DataClient(),
    )
    normalized = broker.get_latest_quote("SPY")
    assert normalized.quote_at == NOW
    assert normalized.mid_price == Decimal("600.00")
