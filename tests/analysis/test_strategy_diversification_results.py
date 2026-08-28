"""Result-table contracts for Day 15 strategy diversification."""

from __future__ import annotations

import numpy as np
import pandas as pd

import systematic_alpha.analysis.strategy_diversification as diversification
from tests.analysis.test_strategy_diversification_statistics import (
    make_weakly_correlated_panel,
)


def test_exact_schemas_and_frozen_row_counts() -> None:
    results = diversification.analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )

    assert tuple(results.sleeve_input_diagnostics.columns) == (
        diversification.SLEEVE_INPUT_DIAGNOSTIC_COLUMNS
    )
    assert tuple(results.full_sample_pairwise_correlations.columns) == (
        diversification.FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS
    )
    assert tuple(results.full_sample_covariance_diagnostics.columns) == (
        diversification.FULL_SAMPLE_COVARIANCE_DIAGNOSTIC_COLUMNS
    )
    assert tuple(results.fold_pairwise_correlations.columns) == (
        diversification.FOLD_PAIRWISE_CORRELATION_COLUMNS
    )
    assert tuple(results.fold_covariance_diagnostics.columns) == (
        diversification.FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS
    )
    assert tuple(results.ensemble_feasibility.columns) == (
        diversification.ENSEMBLE_FEASIBILITY_COLUMNS
    )

    assert len(results.sleeve_input_diagnostics) == 6
    assert len(results.full_sample_pairwise_correlations) == 15
    assert len(results.full_sample_covariance_diagnostics) == 1
    assert len(results.fold_pairwise_correlations) == 120
    assert len(results.fold_covariance_diagnostics) == 8
    assert len(results.ensemble_feasibility) == 1

    counts = results.fold_pairwise_correlations.groupby(
        ["fold_id", "sample"]
    ).size()
    assert len(counts) == 8
    assert counts.eq(15).all()
    assert set(results.fold_covariance_diagnostics["sample"]) == {
        "train",
        "test",
    }


def test_result_container_defensively_copies_every_table() -> None:
    source = diversification.analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )
    tables = {
        name: getattr(source, name).copy(deep=True)
        for name in (
            "session_return_panel",
            "sleeve_input_diagnostics",
            "full_sample_pairwise_correlations",
            "full_sample_covariance_diagnostics",
            "fold_pairwise_correlations",
            "fold_covariance_diagnostics",
            "ensemble_feasibility",
        )
    }
    retained = diversification.StrategyDiversificationResults(**tables)

    for name, original in tables.items():
        first_value = original.iloc[0, 0]
        replacement = (
            "changed"
            if isinstance(first_value, str)
            else not first_value
            if isinstance(first_value, (bool, np.bool_))
            else 99
        )
        copied_replacement = (
            "mutated"
            if isinstance(first_value, str)
            else not first_value
            if isinstance(first_value, (bool, np.bool_))
            else 101
        )
        original.iloc[0, 0] = replacement
        result_table = getattr(retained, name)
        copy_method = getattr(retained, f"copy_{name}")
        copied = copy_method()

        assert result_table.iloc[0, 0] != replacement
        assert copied.equals(result_table)
        assert copied is not result_table
        copied.iloc[0, 0] = copied_replacement
        assert result_table.iloc[0, 0] != copied_replacement


def test_results_contain_no_selection_optimization_or_profitability_fields() -> None:
    results = diversification.analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )
    forbidden_tokens = (
        "ranking",
        "winner",
        "optimal",
        "optimization",
        "optimisation",
        "allocation",
        "profit",
        "pnl",
        "sharpe",
    )
    diagnostic_tables = (
        results.sleeve_input_diagnostics,
        results.full_sample_pairwise_correlations,
        results.full_sample_covariance_diagnostics,
        results.fold_pairwise_correlations,
        results.fold_covariance_diagnostics,
        results.ensemble_feasibility,
    )

    for table in diagnostic_tables:
        for column in table.columns:
            assert not any(
                token in column.lower()
                for token in forbidden_tokens
            )


def test_training_can_pass_while_realised_test_diversification_fails() -> None:
    panel = make_weakly_correlated_panel()
    rng = np.random.default_rng(151)

    for year in range(2022, 2026):
        mask = panel.index.year == year
        common = rng.normal(0.0, 0.001, int(mask.sum()))
        panel.loc[mask, :] = common[:, None]

    results = diversification.analyze_strategy_diversification_panel(panel)
    gate = results.ensemble_feasibility.iloc[0]

    assert bool(gate["maximum_absolute_correlation_gate"])
    assert bool(gate["median_effective_rank_gate"])
    assert bool(gate["median_pc1_share_gate"])
    assert not bool(gate["realised_test_diversification_gate"])
    assert not bool(gate["ensemble_feasible"])
    assert np.isclose(
        gate["median_test_equal_weight_diversification_ratio"],
        1.0,
    )
