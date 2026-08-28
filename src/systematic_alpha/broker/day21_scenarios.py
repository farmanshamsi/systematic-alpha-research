"""Fixed synthetic happy path for the Day 21 paper controller."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Final

from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    DAY21_AUTHORIZATION_SCOPE,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    Day21Authorization,
    Day21ExecutionResult,
    run_controlled_paper_execution,
)
from systematic_alpha.broker.day21_signal import Day21SignalSnapshot
from systematic_alpha.broker.paper_boundary import MarketClockSnapshot


SYNTHETIC_NOW: Final[datetime] = datetime(
    2026, 8, 3, 15, 35, tzinfo=timezone.utc
)


def _signal() -> Day21SignalSnapshot:
    return Day21SignalSnapshot(
        candidate_id="ou_vwap_slow",
        symbol="SPY",
        computed_at=SYNTHETIC_NOW,
        bar_start=SYNTHETIC_NOW - timedelta(minutes=20),
        bar_end=SYNTHETIC_NOW - timedelta(minutes=5),
        last_close=600.0,
        position=1,
        raw_signal=1,
        signal_available=True,
        signal_fresh=True,
        signal_age_seconds=300.0,
        regime_eligible=True,
        ou_zscore=-2.5,
        ou_half_life_bars=10.0,
        variance_ratio=0.8,
        operational_rows=300,
        operational_sessions=12,
        data_start=datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        data_end=SYNTHETIC_NOW - timedelta(minutes=20),
    )


def _order(
    *,
    phase: str,
    side: str,
    client_order_id: str,
    status: str,
    filled: Decimal,
) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id=f"synthetic-broker-{phase}",
        client_order_id=client_order_id,
        symbol="SPY",
        side=side,
        order_type="market",
        time_in_force="day",
        requested_quantity=Decimal("0.01"),
        filled_quantity=filled,
        filled_average_price=(
            Decimal("600.01")
            if phase == "entry" and filled > 0
            else Decimal("600.02") if filled > 0 else None
        ),
        status=status,
        submitted_at=SYNTHETIC_NOW,
        filled_at=SYNTHETIC_NOW + timedelta(seconds=1) if filled > 0 else None,
    )


class _SyntheticBroker:
    paper_endpoint = ALPACA_PAPER_BASE_URL

    def __init__(self) -> None:
        self.position = Decimal("0")
        self.cash = Decimal("100000")
        self.orders: dict[str, BrokerOrderSnapshot] = {}

    def run_preflight(self):
        return SimpleNamespace(preflight_passed=True)

    def get_market_clock(self) -> MarketClockSnapshot:
        return MarketClockSnapshot(
            timestamp=SYNTHETIC_NOW,
            is_open=True,
            next_open=SYNTHETIC_NOW + timedelta(days=1),
            next_close=SYNTHETIC_NOW + timedelta(hours=4),
        )

    def get_open_orders(self, symbol: str):
        return ()

    def has_client_order_id(self, client_order_id: str) -> bool:
        return False

    def get_position(self, symbol: str):
        if self.position == 0:
            return None
        return BrokerPositionSnapshot("SPY", self.position, None, None, None)

    def get_cash(self) -> Decimal:
        return self.cash

    def submit_market_order(self, *, symbol, side, quantity, client_order_id):
        phase = "entry" if "-e-" in client_order_id else "flatten"
        snapshot = replace(
            _order(
                phase=phase,
                side=side,
                client_order_id=client_order_id,
                status="new",
                filled=Decimal("0"),
            ),
            requested_quantity=quantity,
        )
        self.orders[snapshot.broker_order_id] = snapshot
        return snapshot

    def get_order(self, broker_order_id: str):
        current = self.orders[broker_order_id]
        phase = "entry" if broker_order_id.endswith("entry") else "flatten"
        filled = replace(
            current,
            status="filled",
            filled_quantity=current.requested_quantity,
            filled_average_price=(
                Decimal("600.01") if phase == "entry" else Decimal("600.02")
            ),
            filled_at=SYNTHETIC_NOW + timedelta(seconds=1),
        )
        self.orders[broker_order_id] = filled
        self.position = (
            current.requested_quantity if phase == "entry" else Decimal("0")
        )
        return filled

    def cancel_order(self, broker_order_id: str) -> None:
        raise AssertionError("Synthetic happy path must not cancel an order.")


def run_day21_synthetic_happy_path() -> Day21ExecutionResult:
    """Run the fixed fill-and-flatten known-answer scenario."""

    result = run_controlled_paper_execution(
        _SyntheticBroker(),
        signal=_signal(),
        authorization=Day21Authorization(
            approved=True,
            scope=DAY21_AUTHORIZATION_SCOPE,
            paper_endpoint=ALPACA_PAPER_BASE_URL,
            kill_switch_armed=True,
        ),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    if not result.execution_complete or not result.shutdown_reconciled:
        raise RuntimeError("Day 21 synthetic known-answer scenario failed.")
    if result.realized_round_trip_pnl != Decimal("0.0001"):
        raise RuntimeError("Day 21 synthetic P&L arithmetic changed.")
    return result

