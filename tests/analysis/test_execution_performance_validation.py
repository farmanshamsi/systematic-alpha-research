from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from systematic_alpha.analysis.day22_scenarios import (
    ACTIVATION_DATE,
    synthetic_daily_snapshots,
    synthetic_executions,
)
from systematic_alpha.analysis.execution_performance_validation import (
    DailyPortfolioSnapshot,
    ExecutionPerformanceValidationError,
    analyze_execution_performance,
    build_campaign_schedule,
)


def analyze(executions=None, snapshots=None):
    return analyze_execution_performance(
        synthetic_executions() if executions is None else executions,
        synthetic_daily_snapshots() if snapshots is None else snapshots,
        activation_date=ACTIVATION_DATE,
    )


def test_shortfall_decomposition_reconciles_for_buys_and_sells() -> None:
    results = analyze()
    assert {row["side"] for row in results.execution_shortfall} == {"buy", "sell"}
    for row in results.execution_shortfall:
        assert row["total_shortfall_bps"] == pytest.approx(
            row["delay_bps"] + row["spread_bps"] + row["residual_bps"]
        )
        assert abs(row["decomposition_error_bps"]) <= 1e-9
        assert row["decision_to_submit_ms"] == pytest.approx(200.0)
        assert row["broker_ack_ms"] == pytest.approx(100.0)
        assert row["fill_latency_ms"] == pytest.approx(300.0)


def test_buy_price_improvement_has_negative_residual() -> None:
    records = list(synthetic_executions())
    records[0] = replace(records[0], fill_price=records[0].bid_price)
    snapshots = list(synthetic_daily_snapshots())
    snapshots[0] = replace(snapshots[0], strategy_net_pnl=Decimal("0.0197"))
    results = analyze(records, snapshots)
    row = next(
        item
        for item in results.execution_shortfall
        if item["execution_id"] == "strategy-long-entry"
    )
    assert row["residual_bps"] < 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bid_price": Decimal("601"), "ask_price": Decimal("600")}, "crossed"),
        ({"symbol": "QQQ"}, "SPY"),
        ({"purpose": "forced_profit_trade"}, "purpose"),
    ],
)
def test_execution_record_rejects_malformed_scope(overrides, message) -> None:
    with pytest.raises(ExecutionPerformanceValidationError, match=message):
        replace(synthetic_executions()[0], **overrides)


def test_execution_record_rejects_stale_quote() -> None:
    record = synthetic_executions()[0]
    with pytest.raises(ExecutionPerformanceValidationError, match="stale"):
        replace(
            record,
            quote_at=record.decision_at,
            submitted_at=record.decision_at + timedelta(seconds=3),
            broker_submitted_at=record.decision_at + timedelta(seconds=3.1),
            filled_at=record.decision_at + timedelta(seconds=3.2),
        )


def test_execution_record_rejects_time_reversal() -> None:
    record = synthetic_executions()[0]
    with pytest.raises(ExecutionPerformanceValidationError, match="chronological"):
        replace(record, filled_at=record.submitted_at - timedelta(milliseconds=1))


def test_long_short_pnl_and_commissions_are_exact() -> None:
    results = analyze()
    rows = {row["round_trip_id"]: row for row in results.round_trip_pnl}
    assert rows["strategy-long"]["gross_pnl"] == Decimal("0.0196")
    assert rows["strategy-long"]["net_pnl"] == Decimal("0.0194")
    assert rows["strategy-short"]["gross_pnl"] == Decimal("-0.0054")
    assert rows["strategy-short"]["net_pnl"] == Decimal("-0.0056")


@pytest.mark.parametrize(
    "records",
    [
        synthetic_executions()[:-1],
        tuple(
            replace(item, quantity=Decimal("0.02"))
            if item.execution_id == "strategy-long-exit"
            else item
            for item in synthetic_executions()
        ),
        tuple(
            replace(item, side="buy")
            if item.execution_id == "strategy-long-exit"
            else item
            for item in synthetic_executions()
        ),
    ],
)
def test_round_trip_mismatches_fail_closed(records) -> None:
    with pytest.raises(ExecutionPerformanceValidationError, match="round trip|Round-trip"):
        analyze(records)


def test_daily_pnl_must_reconcile_to_strategy_round_trips() -> None:
    rows = list(synthetic_daily_snapshots())
    rows[0] = replace(rows[0], strategy_net_pnl=Decimal("99"))
    with pytest.raises(ExecutionPerformanceValidationError, match="reconcile"):
        analyze(snapshots=rows)


def test_risk_metrics_are_blank_at_19_and_available_at_20() -> None:
    snapshots = synthetic_daily_snapshots()
    result_19 = analyze(snapshots=snapshots[:19])
    risk_19 = result_19.risk_summary[0]
    assert not risk_19["risk_metrics_available"]
    assert risk_19["availability_reason"] == "insufficient_daily_observations"
    assert risk_19["historical_var_95"] is None
    assert result_19.daily_performance[-1]["rolling_20d_volatility"] is None

    result_20 = analyze(snapshots=snapshots[:20])
    risk_20 = result_20.risk_summary[0]
    assert risk_20["risk_metrics_available"]
    assert risk_20["historical_var_95"] is not None
    assert risk_20["historical_es_95"] >= risk_20["historical_var_95"]
    assert risk_20["beta_to_spy"] is not None
    assert result_20.daily_performance[-1]["rolling_20d_volatility"] is not None


def test_zero_spy_variance_blocks_beta_gate() -> None:
    rows = tuple(replace(item, spy_return=0.0) for item in synthetic_daily_snapshots())
    result = analyze(snapshots=rows)
    risk = result.risk_summary[0]
    assert not risk["risk_metrics_available"]
    assert risk["availability_reason"] == "spy_variance_not_positive"
    assert risk["beta_to_spy"] is None


def test_drawdown_exposure_turnover_hit_rate_and_profit_factor() -> None:
    result = analyze()
    risk = result.risk_summary[0]
    assert risk["cumulative_pnl"] == Decimal("0.0138")
    assert risk["maximum_drawdown"] < 0
    assert risk["total_turnover_notional"] == Decimal("24.06")
    assert risk["maximum_gross_exposure"] == Decimal("6.03")
    assert risk["maximum_absolute_net_exposure"] == Decimal("6.03")
    assert risk["strategy_round_trips"] == 2
    assert risk["hit_rate"] == pytest.approx(0.5)
    assert risk["profit_factor"] == pytest.approx(0.0194 / 0.0056)


def test_calibration_pnl_is_never_alpha_eligible() -> None:
    result = analyze()
    strategy, calibration = result.campaign_summary
    assert strategy["purpose"] == "strategy_signal"
    assert strategy["alpha_eligible"]
    assert strategy["round_trips"] == 2
    assert calibration["purpose"] == "calibration_probe"
    assert not calibration["alpha_eligible"]
    assert calibration["round_trips"] == 2
    assert result.risk_summary[0]["strategy_round_trips"] == 2


def test_campaign_schedule_is_fixed_unapproved_and_alternating() -> None:
    schedule = build_campaign_schedule(ACTIVATION_DATE)
    assert len(schedule) == 10
    assert [row["schedule_order"] for row in schedule] == list(range(1, 11))
    assert [row["entry_side"] for row in schedule] == ["buy", "sell"] * 5
    assert all(row["quantity"] == Decimal("0.01") for row in schedule)
    assert all(row["maximum_notional_usd"] == Decimal("10.00") for row in schedule)
    assert not any(row["authorization_granted"] for row in schedule)
    assert {row["scheduled_at_new_york"][11:16] for row in schedule} == {
        "10:15",
        "14:15",
    }
    assert len({row["session_date"] for row in schedule}) == 5


def test_duplicate_daily_session_fails_closed() -> None:
    rows = list(synthetic_daily_snapshots())
    rows[-1] = replace(rows[-1], session_date=rows[-2].session_date)
    with pytest.raises(ExecutionPerformanceValidationError, match="unique"):
        analyze(snapshots=rows)
