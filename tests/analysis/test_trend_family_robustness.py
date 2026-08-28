"""Tests for Day 10 trend-family robustness orchestration."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.trend_family_robustness as robustness
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
)
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    DEVELOPMENT_DATASET_ID,
    EMA_MACD_CONFIGURATION_ID,
    REQUIRED_RESULT_COLUMNS,
    ROBUSTNESS_FREQUENCIES,
    ROBUSTNESS_STRATEGIES,
    ROBUSTNESS_SYMBOLS,
    TREND_RATIO_CONFIGURATION_ID,
    RobustnessRunSpec,
    TrendFamilyRobustnessError,
    build_robustness_run_matrix,
    run_trend_family_robustness,
)
from systematic_alpha.strategies.ema_macd import (
    DEFAULT_COST_BPS_PER_TURNOVER as EMA_COST,
    DEFAULT_FAST_WINDOW,
    DEFAULT_NEUTRAL_BAND as EMA_BAND,
    DEFAULT_SIGNAL_WINDOW,
    DEFAULT_SLOW_WINDOW,
)
from systematic_alpha.strategies.trend_ratio import (
    DEFAULT_COST_BPS_PER_TURNOVER as TREND_COST,
    DEFAULT_LONG_WINDOW,
    DEFAULT_NEUTRAL_BAND as TREND_BAND,
    DEFAULT_SHORT_WINDOW,
    TrendRatioBundle,
)


def make_development_bars(
    *,
    session_dates: tuple[str, ...] = (
        "2025-01-02",
    ),
    symbols: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
    ),
) -> pd.DataFrame:
    """Create compact complete canonical sessions."""

    rows: list[dict[str, object]] = []
    symbol_offsets = {
        "SPY": 0.0,
        "QQQ": 100.0,
        "IWM": 200.0,
    }

    for symbol in symbols:
        sequence = 0

        for session_date in session_dates:
            timestamps = pd.date_range(
                f"{session_date} 14:30:00+00:00",
                periods=26,
                freq="15min",
            )

            for timestamp in timestamps:
                close = (
                    100.0
                    + symbol_offsets[symbol]
                    + 0.12 * sequence
                    + 0.80 * np.sin(
                        sequence / 2.5
                    )
                )
                open_price = close - 0.10

                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "open": open_price,
                        "high": close + 0.25,
                        "low": open_price - 0.25,
                        "close": close,
                        "volume": float(
                            1_000 + sequence
                        ),
                        "trade_count": (
                            100 + sequence
                        ),
                        "vwap": (
                            open_price + close
                        ) / 2.0,
                        "source": "test",
                        "feed": "sip",
                    }
                )
                sequence += 1

    return pd.DataFrame(rows)


def expected_run_keys() -> list[
    tuple[str, str, str]
]:
    """Return the frozen nested-loop run order."""

    return [
        (
            strategy,
            symbol,
            frequency,
        )
        for strategy in ROBUSTNESS_STRATEGIES
        for symbol in ROBUSTNESS_SYMBOLS
        for frequency in ROBUSTNESS_FREQUENCIES
    ]


def test_matrix_contains_exactly_18_run_specifications() -> None:
    matrix = build_robustness_run_matrix()

    assert len(matrix) == 18
    assert all(
        isinstance(specification, RobustnessRunSpec)
        for specification in matrix
    )


def test_matrix_order_is_deterministic() -> None:
    first = build_robustness_run_matrix()
    second = build_robustness_run_matrix()

    assert first == second
    assert [
        (
            specification.strategy,
            specification.symbol,
            specification.frequency,
        )
        for specification in first
    ] == expected_run_keys()


def test_matrix_keys_and_membership_are_frozen() -> None:
    matrix = build_robustness_run_matrix()
    keys = {
        (
            specification.strategy,
            specification.symbol,
            specification.frequency,
        )
        for specification in matrix
    }

    assert len(keys) == 18
    assert {
        specification.strategy
        for specification in matrix
    } == {
        "trend_ratio",
        "ema_macd",
    }
    assert {
        specification.symbol
        for specification in matrix
    } == {
        "SPY",
        "QQQ",
        "IWM",
    }
    assert {
        specification.frequency
        for specification in matrix
    } == {
        "15min",
        "30min",
        "60min",
    }


def test_matrix_uses_explicit_annualization_mapping() -> None:
    assert ANNUALIZATION_FACTORS == {
        "15min": 252 * 26,
        "30min": 252 * 13,
        "60min": 252 * 7,
    }

    for specification in (
        build_robustness_run_matrix()
    ):
        assert (
            specification.annualization_factor
            == ANNUALIZATION_FACTORS[
                specification.frequency
            ]
        )


def test_frozen_configurations_are_baseline_not_sensitivity() -> None:
    assert (
        robustness.TREND_RATIO_PARAMETERS
        .short_window
        == DEFAULT_SHORT_WINDOW
        == 8
    )
    assert (
        robustness.TREND_RATIO_PARAMETERS
        .long_window
        == DEFAULT_LONG_WINDOW
        == 32
    )
    assert (
        robustness.TREND_RATIO_PARAMETERS
        .neutral_band
        == TREND_BAND
        == 0.001
    )
    assert (
        robustness.TREND_RATIO_PARAMETERS
        .cost_bps_per_turnover
        == TREND_COST
        == 1.0
    )

    assert (
        robustness.EMA_MACD_PARAMETERS
        .fast_window
        == DEFAULT_FAST_WINDOW
        == 12
    )
    assert (
        robustness.EMA_MACD_PARAMETERS
        .slow_window
        == DEFAULT_SLOW_WINDOW
        == 26
    )
    assert (
        robustness.EMA_MACD_PARAMETERS
        .signal_window
        == DEFAULT_SIGNAL_WINDOW
        == 9
    )
    assert (
        robustness.EMA_MACD_PARAMETERS
        .neutral_band
        == EMA_BAND
        == 0.0005
    )
    assert (
        robustness.EMA_MACD_PARAMETERS
        .cost_bps_per_turnover
        == EMA_COST
        == 1.0
    )

    identifiers = {
        TREND_RATIO_CONFIGURATION_ID,
        EMA_MACD_CONFIGURATION_ID,
    }

    assert len(identifiers) == 2
    assert not any(
        token in identifier
        for identifier in identifiers
        for token in (
            "day07",
            "day09",
            "sensitivity",
        )
    )


def test_small_integration_executes_complete_matrix() -> None:
    result = run_trend_family_robustness(
        make_development_bars()
    )

    assert len(result) == 18
    assert not result.duplicated(
        [
            "strategy",
            "symbol",
            "frequency",
        ]
    ).any()
    assert list(
        result[
            [
                "strategy",
                "symbol",
                "frequency",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ) == expected_run_keys()

    expected_observations = {
        "15min": 26,
        "30min": 13,
        "60min": 7,
    }
    expected_partials = {
        "15min": 0,
        "30min": 0,
        "60min": 1,
    }

    for frequency in ROBUSTNESS_FREQUENCIES:
        selected = result.loc[
            result["frequency"].eq(frequency)
        ]

        assert selected[
            "observations"
        ].eq(
            expected_observations[
                frequency
            ]
        ).all()
        assert selected[
            "partial_bar_count"
        ].eq(
            expected_partials[
                frequency
            ]
        ).all()
        assert selected[
            "annualization_factor"
        ].eq(
            ANNUALIZATION_FACTORS[
                frequency
            ]
        ).all()

    assert result["dataset_id"].eq(
        DEVELOPMENT_DATASET_ID
    ).all()
    assert result["sessions"].eq(1).all()
    assert pd.to_datetime(
        result["start_timestamp"],
        utc=True,
    ).dt.date.min() >= pd.Timestamp(
        "2020-01-02"
    ).date()
    assert pd.to_datetime(
        result["end_timestamp"],
        utc=True,
    ).dt.date.max() <= pd.Timestamp(
        "2025-12-31"
    ).date()

    configuration_counts = result.groupby(
        "strategy",
        observed=True,
    )["configuration_id"].nunique()

    assert configuration_counts.eq(1).all()
    assert (
        result.groupby(
            "strategy",
            observed=True,
        )["configuration_id"]
        .first()
        .nunique()
        == 2
    )


def test_missing_required_symbol_is_rejected() -> None:
    bars = make_development_bars(
        symbols=("SPY", "QQQ")
    )

    with pytest.raises(
        TrendFamilyRobustnessError,
        match="symbols",
    ):
        run_trend_family_robustness(bars)


@pytest.mark.parametrize(
    "session_date",
    (
        "2019-12-31",
        "2026-01-02",
    ),
)
def test_out_of_development_period_is_rejected(
    session_date: str,
) -> None:
    bars = make_development_bars(
        session_dates=(session_date,)
    )

    with pytest.raises(
        TrendFamilyRobustnessError,
        match="development period",
    ):
        run_trend_family_robustness(bars)


def test_input_is_not_mutated_and_execution_is_deterministic() -> None:
    bars = make_development_bars()
    original = copy.deepcopy(bars)

    first = run_trend_family_robustness(
        bars
    )
    second = run_trend_family_robustness(
        bars
    )

    pd.testing.assert_frame_equal(
        bars,
        original,
    )
    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_negative_metrics_are_retained_and_factors_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_factors: list[float] = []

    def negative_metrics(
        returns,
        *,
        return_column=None,
        annualization_factor,
    ) -> PerformanceMetrics:
        observed_factors.append(
            float(annualization_factor)
        )

        return PerformanceMetrics(
            observations=len(returns),
            cumulative_return=-0.20,
            annualized_return=-0.10,
            annualized_volatility=0.25,
            sharpe_ratio=-0.75,
            max_drawdown=-0.30,
        )

    monkeypatch.setattr(
        robustness,
        "calculate_performance_metrics",
        negative_metrics,
    )

    result = run_trend_family_robustness(
        make_development_bars()
    )

    assert len(result) == 18
    assert result[
        "annualized_return"
    ].eq(-0.10).all()
    assert result["sharpe_ratio"].eq(
        -0.75
    ).all()
    assert observed_factors == [
        float(
            specification.annualization_factor
        )
        for specification in (
            build_robustness_run_matrix()
        )
    ]


def test_features_are_rebuilt_for_each_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_inputs: list[
        tuple[str, int]
    ] = []
    original_builder = (
        robustness.build_return_features
    )

    def recording_builder(
        frame: pd.DataFrame,
        *,
        expected_symbols=None,
    ):
        observed_inputs.append(
            (
                str(
                    frame[
                        "bar_frequency"
                    ].iloc[0]
                ),
                len(frame),
            )
        )

        return original_builder(
            frame,
            expected_symbols=expected_symbols,
        )

    monkeypatch.setattr(
        robustness,
        "build_return_features",
        recording_builder,
    )

    run_trend_family_robustness(
        make_development_bars()
    )

    assert observed_inputs == [
        ("15min", 78),
        ("30min", 39),
        ("60min", 21),
    ]


def test_position_delay_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = (
        robustness.build_trend_ratio_strategy
    )

    def invalid_position_builder(
        frame: pd.DataFrame,
        *,
        parameters,
    ) -> TrendRatioBundle:
        bundle = original_builder(
            frame,
            parameters=parameters,
        )
        observations = (
            bundle.observations.copy(
                deep=True
            )
        )
        observations.loc[
            observations.index[0],
            "position",
        ] = 1

        return replace(
            bundle,
            observations=observations,
        )

    monkeypatch.setattr(
        robustness,
        "build_trend_ratio_strategy",
        invalid_position_builder,
    )

    with pytest.raises(
        TrendFamilyRobustnessError,
        match="one-observation delay",
    ):
        run_trend_family_robustness(
            make_development_bars()
        )


def test_result_schema_is_complete() -> None:
    result = run_trend_family_robustness(
        make_development_bars()
    )

    assert tuple(result.columns) == (
        REQUIRED_RESULT_COLUMNS
    )
