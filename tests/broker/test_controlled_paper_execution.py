from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    DAY21_AUTHORIZATION_SCOPE,
    AlpacaControlledPaperBroker,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    ControlledPaperExecutionError,
    Day21Authorization,
    run_controlled_paper_execution,
)
from systematic_alpha.broker.day21_signal import Day21SignalSnapshot
from systematic_alpha.broker.paper_boundary import MarketClockSnapshot
from tests.day18_fixtures import safe_day18_config


NOW = datetime(2026, 8, 3, 15, 35, tzinfo=timezone.utc)


def signal(**overrides: object) -> Day21SignalSnapshot:
    values = {
        "candidate_id": "ou_vwap_slow",
        "symbol": "SPY",
        "computed_at": NOW,
        "bar_start": NOW - timedelta(minutes=20),
        "bar_end": NOW - timedelta(minutes=5),
        "last_close": 600.0,
        "position": 1,
        "raw_signal": 1,
        "signal_available": True,
        "signal_fresh": True,
        "signal_age_seconds": 300.0,
        "regime_eligible": True,
        "ou_zscore": -2.5,
        "ou_half_life_bars": 10.0,
        "variance_ratio": 0.8,
        "operational_rows": 300,
        "operational_sessions": 12,
        "data_start": datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        "data_end": NOW - timedelta(minutes=20),
        "locked_research_data_accessed": False,
    }
    values.update(overrides)
    return Day21SignalSnapshot(**values)  # type: ignore[arg-type]


def authorization() -> Day21Authorization:
    return Day21Authorization(
        approved=True,
        scope=DAY21_AUTHORIZATION_SCOPE,
        paper_endpoint=ALPACA_PAPER_BASE_URL,
        kill_switch_armed=True,
    )


def order(*, phase: str, status: str, filled: str) -> BrokerOrderSnapshot:
    client = f"axiom-day21-spy-{phase}-202608031515"
    return BrokerOrderSnapshot(
        broker_order_id=f"broker-{phase}",
        client_order_id=client,
        symbol="SPY",
        side="buy" if phase == "e" else "sell",
        order_type="market",
        time_in_force="day",
        requested_quantity=Decimal("0.01"),
        filled_quantity=Decimal(filled),
        filled_average_price=(Decimal("600") if Decimal(filled) > 0 else None),
        status=status,
        submitted_at=NOW,
        filled_at=NOW if status == "filled" else None,
    )


class FakeBroker:
    paper_endpoint = ALPACA_PAPER_BASE_URL

    def __init__(self, *, market_open: bool = True) -> None:
        self.market_open = market_open
        self.next_close_offset = timedelta(hours=4)
        self.submissions: list[BrokerOrderSnapshot] = []
        self.position = Decimal("0")
        self.cash = Decimal("100000")
        self.open_orders: tuple[BrokerOrderSnapshot, ...] = ()
        self.duplicate = False
        self.refreshes: dict[str, int] = {}

    def run_preflight(self):
        return SimpleNamespace(preflight_passed=True)

    def get_market_clock(self) -> MarketClockSnapshot:
        return MarketClockSnapshot(
            timestamp=NOW,
            is_open=self.market_open,
            next_open=NOW + timedelta(days=1),
            next_close=NOW + self.next_close_offset,
        )

    def get_open_orders(self, symbol: str):
        return self.open_orders

    def has_client_order_id(self, client_order_id: str) -> bool:
        return self.duplicate

    def get_position(self, symbol: str):
        if self.position == 0:
            return None
        return BrokerPositionSnapshot(symbol, self.position, None, None, None)

    def get_cash(self) -> Decimal:
        return self.cash

    def submit_market_order(self, *, symbol, side, quantity, client_order_id):
        phase = "e" if "-e-" in client_order_id else "x"
        submitted = replace(
            order(phase=phase, status="new", filled="0"),
            side=side,
            requested_quantity=quantity,
            client_order_id=client_order_id,
        )
        self.submissions.append(submitted)
        return submitted

    def get_order(self, broker_order_id: str):
        phase = "e" if broker_order_id == "broker-e" else "x"
        current = order(phase=phase, status="filled", filled="0.01")
        if phase == "e":
            self.position = Decimal("0.01")
        else:
            self.position = Decimal("0")
        return current

    def cancel_order(self, broker_order_id: str) -> None:
        return None


def test_closed_market_aborts_without_submitting() -> None:
    broker = FakeBroker(market_open=False)
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert result.outcome == "aborted_before_submission"
    assert not result.order_submission_occurred
    assert "market_open_window" in result.abort_reasons
    assert not broker.submissions


def test_near_close_and_open_order_abort_without_submitting() -> None:
    broker = FakeBroker()
    broker.next_close_offset = timedelta(minutes=29, seconds=59)
    broker.open_orders = (order(phase="e", status="new", filled="0"),)
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert set(result.abort_reasons) >= {
        "market_open_window",
        "no_open_spy_order",
    }
    assert not broker.submissions


def test_flat_signal_aborts_without_submitting() -> None:
    broker = FakeBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(position=0),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert "signal_nonzero" in result.abort_reasons
    assert not broker.submissions


@pytest.mark.parametrize(
    ("signal_overrides", "day20_passed", "reason"),
    [
        ({"last_close": 1000.01}, True, "entry_notional_cap"),
        ({"data_start": datetime(2026, 6, 30, tzinfo=timezone.utc)}, True, "post_lock_operational_data"),
        ({}, False, "day20_prerequisite"),
    ],
)
def test_frozen_numeric_and_prerequisite_gates_fail_closed(
    signal_overrides, day20_passed, reason
) -> None:
    broker = FakeBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(**signal_overrides),
        authorization=authorization(),
        day20_gate_passed=day20_passed,
        sleep=lambda _: None,
    )
    assert reason in result.abort_reasons
    assert not broker.submissions


def test_existing_position_duplicate_and_stale_signal_fail_closed() -> None:
    broker = FakeBroker()
    broker.position = Decimal("0.01")
    broker.duplicate = True
    result = run_controlled_paper_execution(
        broker,
        signal=signal(signal_fresh=False),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert set(result.abort_reasons) >= {
        "signal_available",
        "no_existing_spy_position",
        "no_duplicate_signal_order",
    }
    assert not broker.submissions


def test_full_fill_and_flatten_reconcile() -> None:
    broker = FakeBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert result.order_submission_occurred
    assert result.entry_filled_quantity == Decimal("0.01")
    assert result.flatten_filled_quantity == Decimal("0.01")
    assert result.shutdown_reconciled
    assert result.execution_complete
    assert not result.manual_recovery_required
    assert len(broker.submissions) == 2


class RejectBroker(FakeBroker):
    def get_order(self, broker_order_id: str):
        return order(phase="e", status="rejected", filled="0")


def test_rejected_entry_never_sends_flatten() -> None:
    broker = RejectBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert result.entry_filled_quantity == 0
    assert not result.execution_complete
    assert result.abort_reasons == ("entry_rejected",)
    assert len(broker.submissions) == 1


class ResidualBroker(FakeBroker):
    def get_order(self, broker_order_id: str):
        phase = "e" if broker_order_id == "broker-e" else "x"
        if phase == "e":
            self.position = Decimal("0.01")
            return order(phase="e", status="filled", filled="0.01")
        return order(phase="x", status="rejected", filled="0")


def test_rejected_flatten_latches_manual_recovery() -> None:
    broker = ResidualBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert result.manual_recovery_required
    assert not result.shutdown_reconciled
    assert "nonzero_spy_position_at_shutdown" in result.abort_reasons
    assert "entry_flatten_fill_mismatch" in result.abort_reasons


class PartialFillBroker(FakeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.canceled = False

    def get_order(self, broker_order_id: str):
        phase = "e" if broker_order_id == "broker-e" else "x"
        if phase == "e" and not self.canceled:
            self.position = Decimal("0.005")
            return replace(
                order(phase="e", status="partially_filled", filled="0.005"),
                filled_average_price=Decimal("600"),
            )
        if phase == "e":
            return replace(
                order(phase="e", status="canceled", filled="0.005"),
                filled_average_price=Decimal("600"),
            )
        self.position = Decimal("0")
        return replace(
            order(phase="x", status="filled", filled="0.005"),
            requested_quantity=Decimal("0.005"),
            filled_quantity=Decimal("0.005"),
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self.canceled = True


def test_partial_fill_times_out_cancels_and_flattens_confirmed_quantity() -> None:
    broker = PartialFillBroker()
    result = run_controlled_paper_execution(
        broker,
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    assert broker.canceled
    assert result.entry_filled_quantity == Decimal("0.005")
    assert result.flatten_filled_quantity == Decimal("0.005")
    assert result.shutdown_reconciled
    assert result.execution_complete


def test_adapter_rejects_nonpaper_client_endpoint() -> None:
    client = SimpleNamespace(_base_url="https://api.alpaca.markets")
    with pytest.raises(ControlledPaperExecutionError, match="paper endpoint"):
        AlpacaControlledPaperBroker(config=safe_day18_config(), client=client)


def test_authorization_requires_exact_scope_and_kill_switch() -> None:
    with pytest.raises(ControlledPaperExecutionError, match="scope"):
        Day21Authorization(
            approved=True,
            scope="broad_permission",
            paper_endpoint=ALPACA_PAPER_BASE_URL,
            kill_switch_armed=True,
        )
    with pytest.raises(ControlledPaperExecutionError, match="kill switch"):
        Day21Authorization(
            approved=True,
            scope=DAY21_AUTHORIZATION_SCOPE,
            paper_endpoint=ALPACA_PAPER_BASE_URL,
            kill_switch_armed=False,
        )
