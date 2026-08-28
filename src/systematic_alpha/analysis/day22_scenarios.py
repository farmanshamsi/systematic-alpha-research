"""Deterministic known-answer evidence for Day 22."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Final

import exchange_calendars as xcals
import pandas as pd

from systematic_alpha.analysis.execution_performance_validation import (
    DailyPortfolioSnapshot,
    Day22AnalysisResults,
    ExecutionRecord,
    analyze_execution_performance,
)


ACTIVATION_DATE: Final[date] = date(2026, 8, 3)


def _leg(
    *,
    execution_id: str,
    round_trip_id: str,
    purpose: str,
    leg: str,
    side: str,
    decision_at: datetime,
    decision_price: str,
    bid: str,
    ask: str,
    fill: str,
) -> ExecutionRecord:
    quote_at = decision_at + timedelta(milliseconds=100)
    submitted_at = decision_at + timedelta(milliseconds=200)
    broker_submitted_at = decision_at + timedelta(milliseconds=300)
    filled_at = decision_at + timedelta(milliseconds=500)
    return ExecutionRecord(
        execution_id=execution_id,
        round_trip_id=round_trip_id,
        purpose=purpose,
        leg=leg,
        symbol="SPY",
        side=side,
        quantity=Decimal("0.01"),
        decision_at=decision_at,
        decision_price=Decimal(decision_price),
        quote_at=quote_at,
        bid_price=Decimal(bid),
        ask_price=Decimal(ask),
        submitted_at=submitted_at,
        broker_submitted_at=broker_submitted_at,
        filled_at=filled_at,
        fill_price=Decimal(fill),
        commission=Decimal("0.0001"),
    )


def synthetic_executions() -> tuple[ExecutionRecord, ...]:
    """Return two strategy and two calibration round trips."""

    base = datetime(2026, 8, 3, 14, 15, tzinfo=timezone.utc)
    return (
        _leg(
            execution_id="strategy-long-entry",
            round_trip_id="strategy-long",
            purpose="strategy_signal",
            leg="entry",
            side="buy",
            decision_at=base,
            decision_price="600.00",
            bid="599.99",
            ask="600.01",
            fill="600.02",
        ),
        _leg(
            execution_id="strategy-long-exit",
            round_trip_id="strategy-long",
            purpose="strategy_signal",
            leg="exit",
            side="sell",
            decision_at=base + timedelta(hours=1),
            decision_price="602.00",
            bid="601.99",
            ask="602.01",
            fill="601.98",
        ),
        _leg(
            execution_id="strategy-short-entry",
            round_trip_id="strategy-short",
            purpose="strategy_signal",
            leg="entry",
            side="sell",
            decision_at=base + timedelta(days=1),
            decision_price="603.00",
            bid="602.99",
            ask="603.01",
            fill="602.98",
        ),
        _leg(
            execution_id="strategy-short-exit",
            round_trip_id="strategy-short",
            purpose="strategy_signal",
            leg="exit",
            side="buy",
            decision_at=base + timedelta(days=1, hours=1),
            decision_price="603.50",
            bid="603.49",
            ask="603.51",
            fill="603.52",
        ),
        _leg(
            execution_id="probe-buy-entry",
            round_trip_id="probe-buy",
            purpose="calibration_probe",
            leg="entry",
            side="buy",
            decision_at=base + timedelta(days=2),
            decision_price="604.00",
            bid="603.99",
            ask="604.01",
            fill="604.01",
        ),
        _leg(
            execution_id="probe-buy-exit",
            round_trip_id="probe-buy",
            purpose="calibration_probe",
            leg="exit",
            side="sell",
            decision_at=base + timedelta(days=2, minutes=1),
            decision_price="604.00",
            bid="603.99",
            ask="604.01",
            fill="603.99",
        ),
        _leg(
            execution_id="probe-sell-entry",
            round_trip_id="probe-sell",
            purpose="calibration_probe",
            leg="entry",
            side="sell",
            decision_at=base + timedelta(days=3),
            decision_price="605.00",
            bid="604.99",
            ask="605.01",
            fill="604.99",
        ),
        _leg(
            execution_id="probe-sell-exit",
            round_trip_id="probe-sell",
            purpose="calibration_probe",
            leg="exit",
            side="buy",
            decision_at=base + timedelta(days=3, minutes=1),
            decision_price="605.00",
            bid="604.99",
            ask="605.01",
            fill="605.01",
        ),
    )


def synthetic_daily_snapshots() -> tuple[DailyPortfolioSnapshot, ...]:
    """Return 25 sessions whose P&L reconciles exactly to strategy fills."""

    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp(ACTIVATION_DATE), pd.Timestamp(ACTIVATION_DATE) + pd.Timedelta(days=45)
    )[:25]
    pnl_by_order = {
        0: Decimal("0.0194"),
        1: Decimal("-0.0056"),
    }
    rows: list[DailyPortfolioSnapshot] = []
    for order, session in enumerate(sessions):
        pnl = pnl_by_order.get(order, Decimal("0"))
        spy_return = 0.001 * ((order % 5) - 2)
        rows.append(
            DailyPortfolioSnapshot(
                session_date=pd.Timestamp(session).date(),
                strategy_net_pnl=pnl,
                gross_exposure=(Decimal("6.03") if order in pnl_by_order else Decimal("0")),
                net_exposure=(
                    Decimal("6.03")
                    if order == 0
                    else Decimal("-6.03") if order == 1 else Decimal("0")
                ),
                turnover_notional=(
                    Decimal("12.03") if order in pnl_by_order else Decimal("0")
                ),
                spy_return=spy_return,
            )
        )
    return tuple(rows)


def run_day22_synthetic_scenarios() -> Day22AnalysisResults:
    """Run and assert the complete Day 22 known-answer scenario."""

    results = analyze_execution_performance(
        synthetic_executions(),
        synthetic_daily_snapshots(),
        activation_date=ACTIVATION_DATE,
    )
    if (
        len(results.execution_shortfall) != 8
        or len(results.round_trip_pnl) != 4
        or len(results.daily_performance) != 25
        or len(results.campaign_schedule) != 10
        or len(results.campaign_summary) != 2
    ):
        raise RuntimeError("Day 22 synthetic row counts changed.")
    risk = results.risk_summary[0]
    if not risk["risk_metrics_available"]:
        raise RuntimeError("Day 22 synthetic risk metrics are unavailable.")
    strategy = results.campaign_summary[0]
    calibration = results.campaign_summary[1]
    if not strategy["alpha_eligible"] or calibration["alpha_eligible"]:
        raise RuntimeError("Day 22 evidence-purpose separation changed.")
    if Decimal(str(risk["cumulative_pnl"])) != Decimal("0.0138"):
        raise RuntimeError("Day 22 strategy P&L known answer changed.")
    return results

