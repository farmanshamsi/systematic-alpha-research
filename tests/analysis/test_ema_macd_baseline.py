"""Synthetic tests for the Day 8 EMA/MACD baseline analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.ema_macd_baseline import (
    analyse_ema_macd_baseline,
    build_ema_macd_forward_signal_sample,
    build_ema_macd_signal_validation,
)
from systematic_alpha.analysis.strategy_performance import (
    calculate_performance_metrics,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdParameters,
    build_ema_macd_strategy,
)


def make_analysis_frame(
    *,
    observations: int = 80,
) -> pd.DataFrame:
    """Create deterministic multi-session synthetic SPY bars."""

    observation_number = np.arange(
        observations,
        dtype=float,
    )
    close = (
        100.0
        + 0.06 * observation_number
        + 1.25 * np.sin(observation_number / 3.0)
        + 0.45 * np.cos(observation_number / 7.0)
    )

    session_dates = pd.date_range(
        "2025-01-02",
        periods=observations // 10,
        freq="B",
    ).repeat(10)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=observations,
                freq="15min",
                tz="UTC",
            ),
            "session_date": session_dates,
            "symbol": "SPY",
            "close": close,
        }
    )
    frame["close_to_close_simple_return"] = (
        frame["close"].pct_change(fill_method=None)
    )

    return frame


def test_baseline_analysis_reuses_strategy_engine_exactly() -> None:
    frame = make_analysis_frame()
    parameters = EmaMacdParameters()

    analysis = analyse_ema_macd_baseline(
        frame,
        parameters=parameters,
    )
    direct = build_ema_macd_strategy(
        frame,
        parameters=parameters,
    )

    pd.testing.assert_frame_equal(
        analysis.strategy_bundle.observations,
        direct.observations,
    )
    pd.testing.assert_frame_equal(
        analysis.strategy_bundle.diagnostics,
        direct.diagnostics,
    )
    assert analysis.strategy_bundle.parameters == direct.parameters


def test_performance_summary_matches_existing_performance_engine() -> None:
    frame = make_analysis_frame()

    analysis = analyse_ema_macd_baseline(frame)
    observations = analysis.strategy_bundle.observations

    expected_net = calculate_performance_metrics(
        observations["net_strategy_return"],
        annualization_factor=252 * 26,
    )

    summary = analysis.performance_summary

    assert summary["series"].tolist() == [
        "buy_and_hold",
        "ema_macd_gross",
        "ema_macd_net",
    ]

    net_row = summary.loc[
        summary["series"].eq("ema_macd_net")
    ].iloc[0]

    assert net_row["cumulative_return"] == pytest.approx(
        expected_net.cumulative_return
    )
    assert net_row["annualized_return"] == pytest.approx(
        expected_net.annualized_return
    )
    assert net_row["sharpe_ratio"] == pytest.approx(
        expected_net.sharpe_ratio
    )
    assert net_row["max_drawdown"] == pytest.approx(
        expected_net.max_drawdown
    )


def test_forward_signal_sample_uses_histogram_without_ratio_offset() -> None:
    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=4,
                freq="15min",
                tz="UTC",
            ),
            "symbol": "SPY",
            "close": [100.0, 110.0, 121.0, 133.1],
            "normalized_macd_histogram": [
                -0.002,
                0.001,
                0.003,
                0.004,
            ],
        }
    )

    sample = build_ema_macd_forward_signal_sample(
        observations,
        horizon_bars=1,
    )

    assert sample["continuous_signal"].tolist() == [
        -0.002,
        0.001,
        0.003,
    ]
    assert sample.iloc[0]["forward_return"] == pytest.approx(
        0.10
    )
    assert sample.iloc[0]["first_forward_timestamp"] == (
        observations.loc[1, "timestamp"]
    )


def test_signal_validation_reports_all_frozen_horizons() -> None:
    frame = make_analysis_frame()
    observations = build_ema_macd_strategy(
        frame
    ).observations

    summary, buckets = build_ema_macd_signal_validation(
        observations
    )

    assert summary["horizon_bars"].tolist() == [
        1,
        4,
        8,
        16,
    ]
    assert len(summary) == 4
    assert set(summary["actual_signal_buckets"]) == {5}
    assert len(buckets) == 4 * 5


def test_holding_and_break_even_diagnostics_are_compact() -> None:
    analysis = analyse_ema_macd_baseline(
        make_analysis_frame()
    )

    assert len(analysis.holding_diagnostics) == 1
    assert len(analysis.cost_break_even) == 1

    holding = analysis.holding_diagnostics.iloc[0]
    break_even = analysis.cost_break_even.iloc[0]

    assert holding["symbol"] == "SPY"
    assert holding["eligible_observations"] == 80
    assert holding["non_zero_episode_count"] >= 0

    assert break_even["symbol"] == "SPY"
    assert break_even["eligible_observations"] == 80
    assert break_even["status"] in {
        "root_found",
        "non_positive_gross",
        "zero_turnover",
        "root_above_search_interval",
        "invalid_wealth_at_bound",
    }


def test_baseline_analysis_does_not_mutate_input() -> None:
    frame = make_analysis_frame()
    original = frame.copy(deep=True)

    analyse_ema_macd_baseline(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_initial_buy_and_hold_missing_return_is_handled_exactly(
) -> None:
    frame = make_analysis_frame()

    assert pd.isna(
        frame.loc[
            0,
            "close_to_close_simple_return",
        ]
    )

    analysis = analyse_ema_macd_baseline(frame)

    expected = calculate_performance_metrics(
        frame[
            "close_to_close_simple_return"
        ].fillna(0.0),
        annualization_factor=252 * 26,
    )

    buy_and_hold = analysis.performance_summary.loc[
        analysis.performance_summary["series"].eq(
            "buy_and_hold"
        )
    ].iloc[0]

    assert buy_and_hold["cumulative_return"] == pytest.approx(
        expected.cumulative_return
    )
    assert buy_and_hold["sharpe_ratio"] == pytest.approx(
        expected.sharpe_ratio
    )


def test_internal_missing_buy_and_hold_return_is_rejected() -> None:
    from systematic_alpha.analysis.ema_macd_baseline import (
        EmaMacdBaselineError,
    )

    frame = make_analysis_frame()
    frame.loc[
        10,
        "close_to_close_simple_return",
    ] = np.nan

    with pytest.raises(
        EmaMacdBaselineError,
        match="missing only for the first observation",
    ):
        analyse_ema_macd_baseline(frame)
