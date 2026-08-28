"""Synthetic tests for recursive EMA/MACD foundations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.strategies.ema_macd import (
    EmaMacdError,
    EmaMacdParameters,
    build_ema_macd_features,
    calculate_ema_alpha,
    calculate_ema_half_life_bars,
    calculate_recursive_ema,
)


def make_bars(
    prices: list[float],
    *,
    timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Create deterministic synthetic strategy bars."""

    close = pd.Series(prices, dtype="float64")

    if timestamps is None:
        timestamps = pd.date_range(
            "2025-01-02 14:30",
            periods=len(close),
            freq="15min",
            tz="UTC",
        )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "close": close,
            "close_to_close_simple_return": close.pct_change(
                fill_method=None
            ),
        }
    )


def test_ema_alpha_and_half_life_follow_declared_formula() -> None:
    alpha = calculate_ema_alpha(12)
    half_life = calculate_ema_half_life_bars(12)

    assert alpha == pytest.approx(2.0 / 13.0)
    assert half_life == pytest.approx(
        math.log(0.5) / math.log(11.0 / 13.0)
    )


def test_recursive_ema_matches_hand_calculation() -> None:
    values = pd.Series(
        [10.0, 13.0, 13.0],
        dtype="float64",
    )

    ema = calculate_recursive_ema(
        values,
        window=2,
    )

    assert math.isnan(ema.iloc[0])
    assert ema.iloc[1] == pytest.approx(12.0)
    assert ema.iloc[2] == pytest.approx(38.0 / 3.0)


def test_feature_warmup_and_formulas_are_exact() -> None:
    prices = (
        100.0
        + np.arange(40, dtype="float64") * 0.25
        + np.sin(np.arange(40, dtype="float64") / 3.0)
    )

    observations = build_ema_macd_features(
        make_bars(prices.tolist()),
        parameters=EmaMacdParameters(),
    )

    assert observations["fast_ema"].first_valid_index() == 11
    assert observations["slow_ema"].first_valid_index() == 25
    assert observations["macd"].first_valid_index() == 25
    assert (
        observations["macd_signal_line"].first_valid_index()
        == 33
    )
    assert (
        observations["normalized_macd_histogram"]
        .first_valid_index()
        == 33
    )
    assert observations["histogram_change"].first_valid_index() == 34
    assert (
        observations["histogram_acceleration"]
        .first_valid_index()
        == 35
    )

    eligible = observations.loc[
        observations["signal_available"]
    ]

    np.testing.assert_allclose(
        eligible["macd"],
        eligible["fast_ema"] - eligible["slow_ema"],
    )
    np.testing.assert_allclose(
        eligible["macd_histogram"],
        eligible["macd"]
        - eligible["macd_signal_line"],
    )
    np.testing.assert_allclose(
        eligible["normalized_macd_histogram"],
        eligible["macd_histogram"] / eligible["close"],
    )


def test_ema_state_does_not_reset_at_session_boundary() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-01-02 20:30:00+00:00",
            "2025-01-02 20:45:00+00:00",
            "2025-01-02 21:00:00+00:00",
            "2025-01-03 14:30:00+00:00",
            "2025-01-03 14:45:00+00:00",
            "2025-01-03 15:00:00+00:00",
        ]
    )

    observations = build_ema_macd_features(
        make_bars(
            [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            timestamps=timestamps,
        ),
        parameters=EmaMacdParameters(
            fast_window=2,
            slow_window=3,
            signal_window=2,
            neutral_band=0.0,
        ),
    )

    expected_fast = (
        observations.loc[2, "fast_ema"] * (1.0 / 3.0)
        + observations.loc[3, "close"] * (2.0 / 3.0)
    )

    assert observations.loc[3, "fast_ema"] == pytest.approx(
        expected_fast
    )


def test_future_price_changes_do_not_alter_past_features() -> None:
    original = make_bars(
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
            109.0,
        ]
    )
    modified = original.copy(deep=True)

    modified.loc[8:, "close"] *= 5.0
    modified["close_to_close_simple_return"] = (
        modified["close"].pct_change(fill_method=None)
    )

    parameters = EmaMacdParameters(
        fast_window=2,
        slow_window=4,
        signal_window=2,
        neutral_band=0.0,
    )

    original_features = build_ema_macd_features(
        original,
        parameters=parameters,
    )
    modified_features = build_ema_macd_features(
        modified,
        parameters=parameters,
    )

    comparison_columns = [
        "fast_ema",
        "slow_ema",
        "macd",
        "macd_signal_line",
        "macd_histogram",
        "normalized_macd_histogram",
        "histogram_change",
        "histogram_acceleration",
        "signal_available",
    ]

    pd.testing.assert_frame_equal(
        original_features.loc[:7, comparison_columns],
        modified_features.loc[:7, comparison_columns],
    )


def test_feature_builder_does_not_mutate_input() -> None:
    bars = make_bars(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    )
    original = bars.copy(deep=True)

    build_ema_macd_features(
        bars,
        parameters=EmaMacdParameters(
            fast_window=2,
            slow_window=4,
            signal_window=2,
        ),
    )

    pd.testing.assert_frame_equal(bars, original)


def test_invalid_parameters_fail_clearly() -> None:
    with pytest.raises(EmaMacdError, match="smaller than"):
        EmaMacdParameters(
            fast_window=26,
            slow_window=26,
        )

    with pytest.raises(EmaMacdError, match="strictly positive"):
        EmaMacdParameters(signal_window=0)

    with pytest.raises(EmaMacdError, match="non-negative"):
        EmaMacdParameters(neutral_band=-0.001)

    with pytest.raises(EmaMacdError, match="must be finite"):
        EmaMacdParameters(
            cost_bps_per_turnover=float("nan")
        )

    with pytest.raises(EmaMacdError, match="must be an integer"):
        EmaMacdParameters(fast_window=2.5)


def test_invalid_feature_input_fails_without_imputation() -> None:
    bars = make_bars(
        [100.0, 101.0, 102.0, 103.0]
    )
    bars.loc[2, "close"] = np.nan

    with pytest.raises(
        EmaMacdError,
        match="contains missing observations",
    ):
        build_ema_macd_features(
            bars,
            parameters=EmaMacdParameters(
                fast_window=2,
                slow_window=3,
                signal_window=2,
            ),
        )


def test_strategy_signal_and_position_timing_are_exact() -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    bundle = build_ema_macd_strategy(
        make_bars(
            [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
        ),
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=0.0,
        ),
    )

    observations = bundle.observations

    assert observations["signal_available"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert observations["signal"].tolist() == [
        0,
        0,
        -1,
        1,
        -1,
        1,
    ]
    assert observations["position"].tolist() == [
        0,
        0,
        0,
        -1,
        1,
        -1,
    ]
    assert observations["position_eligible"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_direct_reversals_retain_turnover_two() -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    observations = build_ema_macd_strategy(
        make_bars(
            [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
        ),
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=0.0,
        ),
    ).observations

    assert observations["turnover"].tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        2.0,
    ]


def test_cost_and_strategy_return_formulas_are_exact() -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    bars = make_bars(
        [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
    )

    observations = build_ema_macd_strategy(
        bars,
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=0.0,
            cost_bps_per_turnover=1.0,
        ),
    ).observations

    expected_gross = (
        observations["position"].astype(float)
        * bars["close_to_close_simple_return"].fillna(0.0)
    )
    expected_cost = (
        observations["turnover"] / 10_000.0
    )
    expected_net = expected_gross - expected_cost

    np.testing.assert_allclose(
        observations["gross_strategy_return"],
        expected_gross,
    )
    np.testing.assert_allclose(
        observations["transaction_cost"],
        expected_cost,
    )
    np.testing.assert_allclose(
        observations["net_strategy_return"],
        expected_net,
    )


def test_default_strategy_warmup_is_33_signal_and_34_position_bars(
) -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    observation_number = np.arange(
        45,
        dtype="float64",
    )
    prices = (
        100.0
        + observation_number * 0.10
        + np.sin(observation_number / 2.5)
    )

    bundle = build_ema_macd_strategy(
        make_bars(prices.tolist()),
        parameters=EmaMacdParameters(),
    )

    observations = bundle.observations
    diagnostics = bundle.diagnostics.iloc[0]

    assert (
        observations["signal_available"].first_valid_index()
        == 0
    )
    assert (
        observations.index[
            observations["signal_available"]
        ][0]
        == 33
    )
    assert (
        observations.index[
            observations["position_eligible"]
        ][0]
        == 34
    )

    assert diagnostics["signal_warmup_observations"] == 33
    assert diagnostics["position_warmup_observations"] == 34


def test_neutral_band_can_suppress_histogram_noise() -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    bars = make_bars(
        [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
    )

    unbanded = build_ema_macd_strategy(
        bars,
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=0.0,
        ),
    ).observations

    wide_band = build_ema_macd_strategy(
        bars,
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=1.0,
        ),
    ).observations

    assert unbanded["signal"].ne(0).any()
    assert wide_band["signal"].eq(0).all()
    assert wide_band["position"].eq(0).all()
    assert wide_band["turnover"].eq(0.0).all()


def test_strategy_diagnostics_use_position_eligible_denominator(
) -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    bundle = build_ema_macd_strategy(
        make_bars(
            [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
        ),
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=2,
            signal_window=2,
            neutral_band=0.0,
        ),
    )

    diagnostics = bundle.diagnostics.iloc[0]

    assert diagnostics["observations"] == 6
    assert diagnostics["signal_available_observations"] == 4
    assert diagnostics["position_eligible_observations"] == 3
    assert diagnostics["total_turnover"] == pytest.approx(5.0)
    assert diagnostics["position_changing_bars"] == 3

    assert diagnostics["long_exposure_pct"] == pytest.approx(
        100.0 / 3.0
    )
    assert diagnostics["short_exposure_pct"] == pytest.approx(
        200.0 / 3.0
    )
    assert diagnostics["neutral_exposure_pct"] == pytest.approx(
        0.0
    )

    exposure_total = (
        diagnostics["long_exposure_pct"]
        + diagnostics["short_exposure_pct"]
        + diagnostics["neutral_exposure_pct"]
    )
    assert exposure_total == pytest.approx(100.0)


def test_strategy_builder_does_not_mutate_input() -> None:
    from systematic_alpha.strategies.ema_macd import (
        build_ema_macd_strategy,
    )

    bars = make_bars(
        [100.0, 101.0, 102.0, 101.0, 103.0, 102.0]
    )
    original = bars.copy(deep=True)

    build_ema_macd_strategy(
        bars,
        parameters=EmaMacdParameters(
            fast_window=1,
            slow_window=3,
            signal_window=2,
        ),
    )

    pd.testing.assert_frame_equal(bars, original)
