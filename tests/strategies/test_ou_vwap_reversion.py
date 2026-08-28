"""Tests for the causal OU/VWAP reversion strategy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.strategies as strategies
from systematic_alpha.strategies.ou_vwap_reversion import (
    OuVwapReversionError,
    OuVwapReversionParameters,
    _rolling_ou_statistics,
    build_ou_vwap_reversion_strategy,
)


def test_package_exports_preserve_trend_and_reversion_apis() -> None:
    assert strategies.build_ou_vwap_reversion_strategy is (
        build_ou_vwap_reversion_strategy
    )
    assert callable(strategies.build_trend_ratio_strategy)


def parameters(**overrides: object) -> OuVwapReversionParameters:
    values: dict[str, object] = {
        "configuration_id": "test",
        "reference_window": 3,
        "ou_window": 6,
        "variance_ratio_lag": 2,
        "variance_ratio_threshold": 0.99,
        "entry_threshold": 0.25,
        "exit_threshold": 0.05,
        "minimum_half_life": 0.1,
        "maximum_half_life": 100.0,
        "maximum_holding_bars": 4,
        "cost_bps_per_turnover": 1.0,
    }
    values.update(overrides)
    return OuVwapReversionParameters(**values)  # type: ignore[arg-type]


def make_bars(*, sessions: int = 8, bars_per_session: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    residual = np.zeros(sessions * bars_per_session)
    for index in range(1, len(residual)):
        residual[index] = 0.35 * residual[index - 1] + rng.normal(0.0, 0.002)
    close = 100.0 * np.exp(residual)
    timestamps: list[pd.Timestamp] = []
    session_dates: list[pd.Timestamp] = []
    close_flags: list[bool] = []
    for session in pd.bdate_range("2025-01-02", periods=sessions, tz="UTC"):
        for bar in range(bars_per_session):
            timestamps.append(session + pd.Timedelta(hours=14, minutes=30 + 15 * bar))
            session_dates.append(session)
            close_flags.append(bar == bars_per_session - 1)
    series = pd.Series(close, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "session_date": session_dates,
            "is_session_close_bar": close_flags,
            "close": series,
            "vwap": 100.0,
            "volume": 1_000.0,
            "close_to_close_simple_return": series.pct_change(fill_method=None),
        }
    )


def test_strategy_is_causal_and_delays_position_one_bar() -> None:
    bars = make_bars()
    result = build_ou_vwap_reversion_strategy(bars, parameters=parameters()).observations
    active_signal = result.index[result["signal"].ne(0)]
    assert len(active_signal) > 0
    index = int(active_signal[0])
    assert result.loc[index, "position"] == result.loc[index - 1, "signal"]
    assert result.loc[index + 1, "position"] == result.loc[index, "signal"]


def test_future_prices_do_not_change_past_features_or_signals() -> None:
    original = make_bars()
    changed = original.copy(deep=True)
    changed.loc[50:, "close"] *= 2.0
    changed["close_to_close_simple_return"] = changed["close"].pct_change(fill_method=None)
    left = build_ou_vwap_reversion_strategy(original, parameters=parameters()).observations
    right = build_ou_vwap_reversion_strategy(changed, parameters=parameters()).observations
    columns = [
        "volume_weighted_reference",
        "log_price_residual",
        "ou_phi",
        "variance_ratio",
        "signal",
        "position",
    ]
    pd.testing.assert_frame_equal(left.loc[:49, columns], right.loc[:49, columns])


def test_session_close_signal_forces_next_session_open_flat() -> None:
    result = build_ou_vwap_reversion_strategy(
        make_bars(), parameters=parameters()
    ).observations
    close_rows = result.index[result["is_session_close_bar"]]
    assert result.loc[close_rows, "signal"].eq(0).all()
    next_rows = close_rows[:-1] + 1
    assert result.loc[next_rows, "position"].eq(0).all()


def test_cost_reconciles_exactly_from_turnover() -> None:
    result = build_ou_vwap_reversion_strategy(
        make_bars(), parameters=parameters(cost_bps_per_turnover=2.0)
    ).observations
    np.testing.assert_allclose(result["transaction_cost"], result["turnover"] * 2.0 / 10_000.0)
    np.testing.assert_allclose(
        result["net_strategy_return"],
        result["gross_strategy_return"] - result["transaction_cost"],
    )


def test_execution_reset_starts_selected_boundary_flat_without_cost() -> None:
    bars = make_bars()
    reset_timestamp = bars.loc[40, "timestamp"]
    result = build_ou_vwap_reversion_strategy(
        bars,
        parameters=parameters(),
        execution_reset_timestamps=(reset_timestamp,),
    ).observations
    row = result.loc[result["timestamp"].eq(reset_timestamp)].iloc[0]
    assert row["position"] == 0
    assert row["turnover"] == 0.0
    assert row["transaction_cost"] == 0.0


def test_exact_ar1_to_ou_parameter_mapping_known_answer() -> None:
    intercept = 0.03
    phi = 0.65
    delta = 0.25
    values = [-0.4]
    for _ in range(8):
        values.append(intercept + phi * values[-1])

    row = _rolling_ou_statistics(pd.Series(values), window=8).iloc[-1]
    kappa = -math.log(float(row["ou_phi"])) / delta
    theta = float(row["ou_intercept"]) / (1.0 - float(row["ou_phi"]))

    assert row["ou_phi"] == pytest.approx(phi)
    assert row["ou_intercept"] == pytest.approx(intercept)
    assert math.exp(-kappa * delta) == pytest.approx(phi)
    assert theta == pytest.approx(intercept / (1.0 - phi))
    assert theta * (1.0 - phi) == pytest.approx(intercept)


def test_rolling_ols_stationary_standard_deviation_and_dof_known_answer() -> None:
    values = pd.Series([0.2, 0.4, 0.35, 0.5, 0.45, 0.55, 0.52])
    window = 6
    x = values.iloc[:-1].to_numpy(dtype="float64")
    y = values.iloc[1:].to_numpy(dtype="float64")
    expected_phi = np.sum((x - x.mean()) * (y - y.mean())) / np.sum(
        (x - x.mean()) ** 2
    )
    expected_intercept = y.mean() - expected_phi * x.mean()
    residuals = y - expected_intercept - expected_phi * x
    expected_innovation_variance = np.sum(residuals**2) / (window - 2)
    expected_stationary_std = math.sqrt(
        expected_innovation_variance / (1.0 - expected_phi**2)
    )

    row = _rolling_ou_statistics(values, window=window).iloc[-1]

    assert row["ou_phi"] == pytest.approx(expected_phi)
    assert row["ou_intercept"] == pytest.approx(expected_intercept)
    assert row["ou_innovation_std"] ** 2 == pytest.approx(
        expected_innovation_variance
    )
    assert row["ou_stationary_std"] == pytest.approx(expected_stationary_std)


def test_half_life_identity_is_reported_in_bar_units() -> None:
    phi = 0.8
    values = [2.0]
    for _ in range(7):
        values.append(0.1 + phi * values[-1])

    row = _rolling_ou_statistics(pd.Series(values), window=7).iloc[-1]

    assert row["ou_phi"] == pytest.approx(phi)
    assert row["ou_half_life_bars"] == pytest.approx(
        -math.log(2.0) / math.log(phi)
    )


@pytest.mark.parametrize("phi", [-0.5, 1.1])
def test_incompatible_phi_masks_ou_level_diagnostics(phi: float) -> None:
    values = [0.2]
    for _ in range(6):
        values.append(0.05 + phi * values[-1])

    row = _rolling_ou_statistics(pd.Series(values), window=6).iloc[-1]

    assert row["ou_phi"] == pytest.approx(phi)
    assert math.isnan(float(row["ou_equilibrium"]))
    assert math.isnan(float(row["ou_stationary_std"]))
    assert math.isnan(float(row["ou_half_life_bars"]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reference_window": 1}, "at least two"),
        ({"ou_window": 2}, "at least three"),
        ({"variance_ratio_lag": 6}, "smaller than ou_window"),
        ({"variance_ratio_threshold": 1.0}, "strictly between"),
        ({"entry_threshold": 0.0}, "strictly positive"),
        ({"exit_threshold": 0.3}, "smaller than entry"),
        ({"maximum_half_life": 0.05}, "must not be smaller"),
        ({"cost_bps_per_turnover": -1.0}, "non-negative"),
    ],
)
def test_invalid_parameters_fail_closed(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(OuVwapReversionError, match=message):
        parameters(**overrides)
