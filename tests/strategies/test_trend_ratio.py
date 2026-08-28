"""Tests for the short/long average price-ratio trend strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.strategies.trend_ratio import (
    TrendRatioError,
    TrendRatioParameters,
    build_trend_ratio_strategy,
    calculate_turnover,
)


def make_strategy_bars(
    prices: list[float],
    *,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Create synthetic close-to-close strategy input."""
    close = pd.Series(prices, dtype="float64")

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30:00",
                periods=len(prices),
                freq="15min",
                tz="UTC",
            ),
            "symbol": symbol,
            "close": close,
            "close_to_close_simple_return": close.pct_change(
                fill_method=None
            ),
        }
    )


def test_constructs_long_short_and_neutral_signals() -> None:
    bars = make_strategy_bars(
        [100.0, 130.0, 130.0, 100.0]
    )
    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.05,
    )

    result = build_trend_ratio_strategy(
        bars,
        parameters=parameters,
    )

    assert result.observations["signal"].tolist() == [
        0,
        1,
        0,
        -1,
    ]


def test_long_flat_mode_never_constructs_a_short_signal() -> None:
    bars = make_strategy_bars([100.0, 130.0, 130.0, 100.0])
    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.05,
        positioning="long_flat",
    )

    result = build_trend_ratio_strategy(bars, parameters=parameters)

    assert result.observations["signal"].tolist() == [0, 1, 0, 0]
    assert not result.observations["position"].eq(-1).any()


def test_signal_affects_position_only_on_next_bar() -> None:
    bars = make_strategy_bars(
        [100.0, 100.0, 130.0, 130.0]
    )
    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.05,
    )

    observations = build_trend_ratio_strategy(
        bars,
        parameters=parameters,
    ).observations

    assert observations.loc[2, "signal"] == 1
    assert observations.loc[2, "position"] == 0
    assert observations.loc[3, "position"] == 1


def test_future_prices_do_not_change_past_signals() -> None:
    original = make_strategy_bars(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
        ]
    )
    modified = original.copy()

    modified.loc[6:, "close"] = (
        modified.loc[6:, "close"] * 5.0
    )
    modified["close_to_close_simple_return"] = (
        modified["close"].pct_change(fill_method=None)
    )

    parameters = TrendRatioParameters(
        short_window=2,
        long_window=4,
        neutral_band=0.001,
    )

    original_result = build_trend_ratio_strategy(
        original,
        parameters=parameters,
    ).observations
    modified_result = build_trend_ratio_strategy(
        modified,
        parameters=parameters,
    ).observations

    comparison_columns = [
        "short_average",
        "long_average",
        "ma_price_ratio",
        "signal",
        "position",
    ]

    pd.testing.assert_frame_equal(
        original_result.loc[:5, comparison_columns],
        modified_result.loc[:5, comparison_columns],
    )


def test_warmup_produces_no_active_position() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    )
    parameters = TrendRatioParameters(
        short_window=2,
        long_window=4,
        neutral_band=0.0,
    )

    observations = build_trend_ratio_strategy(
        bars,
        parameters=parameters,
    ).observations

    assert (
        observations.loc[:2, "signal_available"]
        .eq(False)
        .all()
    )
    assert observations.loc[3, "signal_available"]
    assert (
        observations.loc[:3, "position"]
        .eq(0)
        .all()
    )
    assert observations.loc[4, "position"] == 1
    assert (
        observations.loc[:3, "position_eligible"]
        .eq(False)
        .all()
    )


def test_turnover_counts_entry_reversal_and_exit() -> None:
    position = pd.Series(
        [0, 1, -1, 0],
        dtype="int8",
    )
    symbols = pd.Series(
        ["SPY", "SPY", "SPY", "SPY"],
        dtype="string",
    )

    turnover = calculate_turnover(
        position,
        symbols,
    )

    assert turnover.tolist() == [
        0.0,
        1.0,
        2.0,
        1.0,
    ]


def test_zero_cost_makes_net_equal_gross() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 103.0, 102.0, 100.0, 99.0]
    )
    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
        cost_bps_per_turnover=0.0,
    )

    observations = build_trend_ratio_strategy(
        bars,
        parameters=parameters,
    ).observations

    np.testing.assert_allclose(
        observations["net_strategy_return"],
        observations["gross_strategy_return"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"short_window": 0},
            "strictly positive",
        ),
        (
            {
                "short_window": 4,
                "long_window": 4,
            },
            "smaller than long_window",
        ),
        (
            {"neutral_band": -0.001},
            "non-negative",
        ),
        (
            {"cost_bps_per_turnover": -1.0},
            "non-negative",
        ),
        (
            {"short_window": 1.5},
            "must be an integer",
        ),
        (
            {"neutral_band": float("nan")},
            "must be finite",
        ),
    ],
)
def test_invalid_parameters_fail_clearly(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "short_window": 2,
        "long_window": 4,
        "neutral_band": 0.001,
        "cost_bps_per_turnover": 0.0,
    }
    arguments.update(overrides)

    with pytest.raises(
        TrendRatioError,
        match=message,
    ):
        TrendRatioParameters(**arguments)


def test_missing_required_column_fails_clearly() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 102.0, 103.0]
    ).drop(columns="close_to_close_simple_return")

    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
    )

    with pytest.raises(
        TrendRatioError,
        match="missing required columns",
    ):
        build_trend_ratio_strategy(
            bars,
            parameters=parameters,
        )


def test_malformed_timestamp_fails_clearly() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 102.0, 103.0]
    )
    bars["timestamp"] = bars["timestamp"].astype("object")
    bars.loc[2, "timestamp"] = "not-a-timestamp"

    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
    )

    with pytest.raises(
        TrendRatioError,
        match="malformed timestamps",
    ):
        build_trend_ratio_strategy(
            bars,
            parameters=parameters,
        )


def test_duplicate_symbol_timestamp_fails_clearly() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 102.0, 103.0]
    )
    bars = pd.concat(
        [bars, bars.iloc[[2]]],
        ignore_index=True,
    )

    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
    )

    with pytest.raises(
        TrendRatioError,
        match="duplicate symbol-timestamp",
    ):
        build_trend_ratio_strategy(
            bars,
            parameters=parameters,
        )


def test_nonpositive_price_fails_clearly() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 0.0, 103.0]
    )

    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
    )

    with pytest.raises(
        TrendRatioError,
        match="strictly positive",
    ):
        build_trend_ratio_strategy(
            bars,
            parameters=parameters,
        )


def test_interior_missing_return_fails_clearly() -> None:
    bars = make_strategy_bars(
        [100.0, 101.0, 102.0, 103.0]
    )
    bars.loc[2, "close_to_close_simple_return"] = np.nan

    parameters = TrendRatioParameters(
        short_window=1,
        long_window=2,
        neutral_band=0.0,
    )

    with pytest.raises(
        TrendRatioError,
        match="Missing returns",
    ):
        build_trend_ratio_strategy(
            bars,
            parameters=parameters,
        )
