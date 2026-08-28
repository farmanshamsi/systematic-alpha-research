"""Result-table contracts for Day 14 cointegration feasibility."""

from __future__ import annotations

import pandas as pd

import systematic_alpha.analysis.cointegration_feasibility as feasibility


EXPECTED_SCHEMAS = {
    "pair_input_diagnostics": (
        "pair_id",
        "y_symbol",
        "x_symbol",
        "daily_observations",
        "intraday_observations",
        "daily_start_session",
        "daily_end_session",
        "intraday_start_timestamp",
        "intraday_end_timestamp",
        "forward_fill_used",
        "locked_period_accessed",
    ),
    "series_integration_diagnostics": (
        "symbol",
        "test",
        "regression",
        "autolag",
        "observations",
        "adf_statistic",
        "p_value",
        "used_lag",
        "critical_value_5pct",
        "reject_unit_root",
        "plausibly_i1",
    ),
    "cointegration_diagnostics": (
        "pair_id",
        "y_symbol",
        "x_symbol",
        "observations",
        "alpha",
        "beta",
        "beta_interpretable",
        "ols_r_squared",
        "engle_granger_statistic",
        "engle_granger_p_value",
        "holm_adjusted_p_value",
        "holm_reject",
        "residual_adf_statistic",
        "residual_adf_p_value",
        "residual_adf_used_lag",
        "residual_adf_reject",
    ),
    "fold_stability_diagnostics": (
        "pair_id",
        "fold_id",
        "train_start",
        "train_end_exclusive",
        "test_start",
        "test_end_exclusive",
        "train_observations",
        "test_observations",
        "train_alpha",
        "train_beta",
        "beta_relative_deviation",
        "beta_sign_stable",
        "test_residual_adf_statistic",
        "test_residual_adf_p_value",
        "test_residual_stationary",
    ),
    "ou_diagnostics": (
        "pair_id",
        "attempted",
        "intraday_observations",
        "consecutive_transitions",
        "ar_intercept",
        "phi",
        "kappa_per_bar",
        "theta",
        "innovation_sigma",
        "diffusion_sigma",
        "half_life_bars",
        "phi_valid",
        "parameters_finite",
        "half_life_valid",
        "ou_pass",
        "rejection_reason",
    ),
    "pair_eligibility": (
        "pair_id",
        "y_symbol",
        "x_symbol",
        "y_plausibly_i1",
        "x_plausibly_i1",
        "holm_cointegration_pass",
        "beta_pass",
        "fold_beta_stability_pass",
        "stationary_fold_count",
        "fold_stationarity_pass",
        "ou_attempted",
        "ou_pass",
        "eligible",
        "rejection_reasons",
    ),
}


def test_exact_result_schemas_are_frozen() -> None:
    assert feasibility.PAIR_INPUT_DIAGNOSTIC_COLUMNS == (
        EXPECTED_SCHEMAS["pair_input_diagnostics"]
    )
    assert feasibility.SERIES_INTEGRATION_COLUMNS == (
        EXPECTED_SCHEMAS["series_integration_diagnostics"]
    )
    assert feasibility.COINTEGRATION_DIAGNOSTIC_COLUMNS == (
        EXPECTED_SCHEMAS["cointegration_diagnostics"]
    )
    assert feasibility.FOLD_STABILITY_COLUMNS == (
        EXPECTED_SCHEMAS["fold_stability_diagnostics"]
    )
    assert feasibility.OU_DIAGNOSTIC_COLUMNS == (
        EXPECTED_SCHEMAS["ou_diagnostics"]
    )
    assert feasibility.PAIR_ELIGIBILITY_COLUMNS == (
        EXPECTED_SCHEMAS["pair_eligibility"]
    )


def test_result_object_defensively_copies_every_table() -> None:
    source_tables = {
        name: pd.DataFrame(
            [["original"] * len(columns)],
            columns=columns,
        )
        for name, columns in EXPECTED_SCHEMAS.items()
    }

    results = feasibility.CointegrationFeasibilityResults(
        **source_tables
    )

    for name, source in source_tables.items():
        source.iloc[0, 0] = "changed"

        retained = getattr(results, name)
        copied = getattr(
            results,
            f"copy_{name}",
        )()

        assert retained.iloc[0, 0] == "original"
        assert copied.equals(retained)
        assert copied is not retained

        copied.iloc[0, 0] = "mutated"
        assert retained.iloc[0, 0] == "original"
