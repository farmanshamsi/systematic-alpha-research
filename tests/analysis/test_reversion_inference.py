"""Tests for the Day 17 walk-forward and inference engine."""

from __future__ import annotations

import pandas as pd
import pytest

from systematic_alpha.analysis.reversion_inference import (
    AGGREGATE_PERFORMANCE_COLUMNS,
    COST_SENSITIVITY_COLUMNS,
    FOLD_PERFORMANCE_COLUMNS,
    INFERENCE_COLUMNS,
    RETURN_PANEL_COLUMNS,
    SIGNAL_DIAGNOSTIC_COLUMNS,
    ReversionInferenceError,
    _apply_ou_performance_timing,
    run_reversion_inference,
)
from tests.day17_fixtures import make_day17_development_bars


@pytest.fixture(scope="module")
def results():
    return run_reversion_inference(
        make_day17_development_bars(), bootstrap_replications=25
    )


def test_exact_output_contract_and_row_counts(results) -> None:
    assert tuple(results.signal_diagnostics.columns) == SIGNAL_DIAGNOSTIC_COLUMNS
    assert tuple(results.fold_performance.columns) == FOLD_PERFORMANCE_COLUMNS
    assert tuple(results.aggregate_performance.columns) == AGGREGATE_PERFORMANCE_COLUMNS
    assert tuple(results.inference_results.columns) == INFERENCE_COLUMNS
    assert tuple(results.cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert tuple(results.session_return_panel.columns) == RETURN_PANEL_COLUMNS
    assert len(results.signal_diagnostics) == 36
    assert len(results.fold_performance) == 48
    assert len(results.aggregate_performance) == 12
    assert len(results.inference_results) == 12
    assert len(results.cost_sensitivity) == 48
    assert len(results.session_return_panel) == 72


def test_execution_resets_and_overnight_flatness_hold(results) -> None:
    diagnostics = results.signal_diagnostics
    assert diagnostics["initial_position"].eq(0).all()
    assert diagnostics["initial_turnover"].eq(0.0).all()
    assert diagnostics["overnight_position_violations"].eq(0).all()


def test_causal_timing_preserves_ou_statistics_and_raw_signals() -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-02 14:30Z", "2025-01-02 14:45Z"],
                utc=True,
            ),
            "symbol": ["SPY", "SPY"],
            "session_date": ["2025-01-02", "2025-01-02"],
            "open": [100.0, 110.0],
            "close": [101.0, 121.0],
            "signal": [1, -1],
            "signal_available": [True, True],
            "ou_phi": [0.8, 0.81],
            "ou_equilibrium": [0.001, 0.0011],
            "ou_stationary_std": [0.02, 0.021],
            "ou_half_life_bars": [3.1, 3.3],
            "ou_zscore": [-2.1, -1.7],
            "variance_ratio": [0.8, 0.82],
            "signal_score": [2.1, 1.7],
            "position": [99, 99],
            "gross_strategy_return": [99.0, 99.0],
        }
    )
    protected = (
        "signal",
        "signal_available",
        "ou_phi",
        "ou_equilibrium",
        "ou_stationary_std",
        "ou_half_life_bars",
        "ou_zscore",
        "variance_ratio",
        "signal_score",
    )

    timed = _apply_ou_performance_timing(source, fold_id="synthetic")

    pd.testing.assert_frame_equal(
        timed.loc[:, protected],
        source.loc[:, protected],
        check_exact=True,
    )
    assert timed["position"].tolist() == [0, 1]
    assert timed.loc[0, "gross_strategy_return"] == 0.0
    assert timed.loc[1, "gross_strategy_return"] == pytest.approx(
        121.0 / 110.0 - 1.0
    )
    assert timed.loc[1, "close_turnover"] == 1.0
    assert timed.loc[1, "ending_position"] == 0


def test_cost_stress_uses_identical_positions_and_is_monotone(results) -> None:
    for _, group in results.cost_sensitivity.groupby(
        ["configuration_id", "series"], sort=False
    ):
        ordered = group.sort_values("cost_bps_per_turnover")
        changes = ordered["cumulative_return"].diff().dropna()
        assert changes.le(1.0e-12).all()


def test_inference_declares_three_trials_and_fixed_hac_contract(results) -> None:
    inference = results.inference_results
    assert inference["declared_trials"].eq(3).all()
    assert inference["hac_lags"].eq(5).all()
    assert inference["bootstrap_replications"].eq(25).all()
    assert inference["bootstrap_block_length"].eq(5).all()
    assert inference["information_coefficient_observations"].ge(0).all()
    assert inference["information_coefficient_observations"].gt(0).any()


def test_locked_period_is_rejected() -> None:
    bars = make_day17_development_bars()
    bars.loc[0, "timestamp"] = pd.Timestamp("2026-01-02 14:30", tz="UTC")
    with pytest.raises(ReversionInferenceError, match="development-only"):
        run_reversion_inference(bars, bootstrap_replications=5)
