"""Tests for reusable simple-return performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.strategy_performance import (
    StrategyPerformanceError,
    build_performance_summary,
    build_wealth_index,
    calculate_performance_metrics,
)


def test_wealth_index_compounds_returns() -> None:
    returns = pd.Series(
        [0.10, 0.10],
        dtype="float64",
    )

    wealth = build_wealth_index(returns)

    np.testing.assert_allclose(
        wealth.to_numpy(),
        np.array([1.10, 1.21]),
    )

    assert not np.isclose(
        wealth.iloc[-1],
        1.0 + returns.sum(),
    )


def test_cumulative_return_is_ending_wealth_minus_one() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(
        returns,
        annualization_factor=12.0,
    )

    expected_wealth = (
        (1.0 + returns)
        .prod()
    )

    assert np.isclose(
        metrics.cumulative_return,
        expected_wealth - 1.0,
    )


def test_annualized_geometric_return_is_correct() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(
        returns,
        annualization_factor=12.0,
    )

    ending_wealth = float(
        (1.0 + returns).prod()
    )

    expected = (
        ending_wealth
        ** (12.0 / len(returns))
        - 1.0
    )

    assert np.isclose(
        metrics.annualized_return,
        expected,
    )


def test_annualized_volatility_uses_sample_standard_deviation() -> None:
    returns = pd.Series(
        [0.01, 0.03, -0.02, 0.04],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(
        returns,
        annualization_factor=12.0,
    )

    expected = (
        returns.std(ddof=1)
        * np.sqrt(12.0)
    )

    assert np.isclose(
        metrics.annualized_volatility,
        expected,
    )


def test_sharpe_uses_requested_annualization() -> None:
    returns = pd.Series(
        [0.01, 0.03, -0.02, 0.04],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(
        returns,
        annualization_factor=12.0,
    )

    expected = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(12.0)
    )

    assert np.isclose(
        metrics.sharpe_ratio,
        expected,
    )


def test_zero_volatility_produces_nan_sharpe() -> None:
    returns = pd.Series(
        [0.10, 0.10, 0.10],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(returns)

    assert np.isnan(metrics.sharpe_ratio)
    assert metrics.annualized_volatility == 0.0


def test_maximum_drawdown_uses_running_wealth_peak() -> None:
    returns = pd.Series(
        [0.10, -0.20, 0.05, -0.10, 0.25],
        dtype="float64",
    )

    metrics = calculate_performance_metrics(returns)

    wealth = (1.0 + returns).cumprod()
    expected = float(
        (
            wealth
            / wealth.cummax()
            - 1.0
        ).min()
    )

    assert np.isclose(
        metrics.max_drawdown,
        expected,
    )
    assert metrics.max_drawdown <= 0.0


def test_two_series_summary_preserves_requested_order() -> None:
    frame = pd.DataFrame(
        {
            "gross": [0.01, 0.02, -0.01],
            "net": [0.009, 0.019, -0.011],
        }
    )

    summary = build_performance_summary(
        frame,
        ("net", "gross"),
        annualization_factor=12.0,
    )

    assert summary["series"].tolist() == [
        "net",
        "gross",
    ]

    assert summary["observations"].tolist() == [
        3,
        3,
    ]


def test_input_objects_are_not_mutated() -> None:
    series = pd.Series(
        [0.01, 0.02, -0.01],
        name="strategy_return",
        dtype="float64",
    )
    frame = pd.DataFrame(
        {
            "gross": [0.01, 0.02, -0.01],
            "net": [0.009, 0.019, -0.011],
        }
    )

    original_series = series.copy(deep=True)
    original_frame = frame.copy(deep=True)

    build_wealth_index(series)
    build_performance_summary(
        frame,
        ("gross", "net"),
    )

    pd.testing.assert_series_equal(
        series,
        original_series,
    )
    pd.testing.assert_frame_equal(
        frame,
        original_frame,
    )


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        (
            pd.Series([], dtype="float64"),
            "cannot be empty",
        ),
        (
            pd.Series([0.01], dtype="float64"),
            "At least two",
        ),
        (
            pd.Series([0.01, np.nan]),
            "missing observations",
        ),
        (
            pd.Series([0.01, np.inf]),
            "finite values",
        ),
        (
            pd.Series([0.01, "invalid"]),
            "non-numeric",
        ),
        (
            pd.Series([0.01, -1.0]),
            "strictly greater than -1.0",
        ),
        (
            pd.Series([0.01, -1.01]),
            "strictly greater than -1.0",
        ),
    ],
)
def test_invalid_return_inputs_fail_clearly(
    returns: pd.Series,
    message: str,
) -> None:
    with pytest.raises(
        StrategyPerformanceError,
        match=message,
    ):
        calculate_performance_metrics(returns)


@pytest.mark.parametrize(
    "annualization_factor",
    [
        True,
        False,
        0.0,
        -1.0,
        np.nan,
        np.inf,
        "252",
    ],
)
def test_invalid_annualization_factors_fail_clearly(
    annualization_factor: object,
) -> None:
    returns = pd.Series(
        [0.01, 0.02],
        dtype="float64",
    )

    with pytest.raises(
        StrategyPerformanceError,
        match="annualization_factor",
    ):
        calculate_performance_metrics(
            returns,
            annualization_factor=annualization_factor,
        )


def test_missing_dataframe_columns_fail_clearly() -> None:
    frame = pd.DataFrame(
        {
            "gross": [0.01, 0.02],
        }
    )

    with pytest.raises(
        StrategyPerformanceError,
        match="missing required columns",
    ):
        build_performance_summary(
            frame,
            ("gross", "net"),
        )


def test_dataframe_input_requires_a_selected_column() -> None:
    frame = pd.DataFrame(
        {
            "strategy_return": [0.01, 0.02],
        }
    )

    with pytest.raises(
        StrategyPerformanceError,
        match="return_column is required",
    ):
        calculate_performance_metrics(frame)


def test_non_pandas_return_input_fails_clearly() -> None:
    with pytest.raises(
        StrategyPerformanceError,
        match="pandas Series",
    ):
        calculate_performance_metrics([0.01, 0.02])
