"""Deterministic report contracts for Day 16."""

from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.day16_portfolio_validation_report as reporting
import systematic_alpha.analysis.portfolio_allocation_validation as allocation
from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS
from tests.day16_fixtures import make_day16_panel


@pytest.fixture(scope="module")
def day16_results() -> allocation.PortfolioAllocationResults:
    return allocation.analyze_portfolio_allocation_panel(make_day16_panel())


@pytest.fixture(scope="module")
def day16_report(
    day16_results: allocation.PortfolioAllocationResults,
) -> reporting.Day16PortfolioValidationReport:
    return reporting.build_day16_portfolio_validation_report(day16_results)


def clone_results(
    results: allocation.PortfolioAllocationResults,
    **replacements: object,
) -> allocation.PortfolioAllocationResults:
    """Copy a result object while replacing selected evidence tables."""

    values: dict[str, object] = {
        "allocation_weights": results.copy_allocation_weights(),
        "allocation_diagnostics": results.copy_allocation_diagnostics(),
        "fold_portfolio_performance": (
            results.copy_fold_portfolio_performance()
        ),
        "aggregate_portfolio_performance": (
            results.copy_aggregate_portfolio_performance()
        ),
        "portfolio_return_panel": results.copy_portfolio_return_panel(),
        "minimum_variance_covariances": (
            results.copy_minimum_variance_covariances()
        ),
        "evaluation_complete": results.evaluation_complete,
    }
    values.update(replacements)
    return allocation.PortfolioAllocationResults(**values)  # type: ignore[arg-type]


def test_report_is_frozen_and_defensively_copies_every_artifact_table(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    assert is_dataclass(day16_report)
    assert day16_report.__dataclass_params__.frozen
    assert not hasattr(day16_report, "__dict__")
    for name in (
        "allocation_weights",
        "allocation_diagnostics",
        "fold_portfolio_performance",
        "aggregate_portfolio_performance",
        "portfolio_return_panel",
    ):
        retained = getattr(day16_report, name)
        copied = getattr(day16_report, f"copy_{name}")()
        original = retained.iloc[0, 0]
        replacement = (
            original + pd.Timedelta(days=1)
            if isinstance(original, pd.Timestamp)
            else "changed"
        )
        copied.iloc[0, 0] = replacement
        assert retained.iloc[0, 0] == original

    manifest = day16_report.copy_manifest()
    manifest["report_id"] = "changed"
    assert day16_report.copy_manifest()["report_id"] == (
        "day16_portfolio_validation"
    )


def test_report_tables_preserve_exact_schemas_counts_and_order(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    contracts = (
        (
            day16_report.allocation_weights,
            allocation.ALLOCATION_WEIGHT_COLUMNS,
            72,
        ),
        (
            day16_report.allocation_diagnostics,
            allocation.ALLOCATION_DIAGNOSTIC_COLUMNS,
            12,
        ),
        (
            day16_report.fold_portfolio_performance,
            allocation.FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
            12,
        ),
        (
            day16_report.aggregate_portfolio_performance,
            allocation.AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
            3,
        ),
    )
    for table, columns, rows in contracts:
        assert tuple(table.columns) == columns
        assert len(table) == rows
    assert tuple(day16_report.portfolio_return_panel.columns) == (
        allocation.PORTFOLIO_RETURN_PANEL_COLUMNS
    )
    assert tuple(
        day16_report.aggregate_portfolio_performance["allocation_rule"]
    ) == allocation.ALLOCATION_RULES
    expected_sleeves = SLEEVE_IDS * 12
    assert tuple(day16_report.allocation_weights["sleeve_id"]) == expected_sleeves


def test_nonapplicable_covariance_and_solver_fields_are_explicit_neutral_values(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    diagnostics = day16_report.allocation_diagnostics
    not_applicable = diagnostics.loc[
        ~diagnostics["allocation_rule"].eq("constrained_minimum_variance")
    ]
    assert set(not_applicable["covariance_estimator"]) == {"not_applicable"}
    assert set(not_applicable["shrinkage_coefficient"]) == {""}
    assert set(not_applicable["solver_status"]) == {"not_applicable"}
    applicable = diagnostics.loc[
        diagnostics["allocation_rule"].eq("constrained_minimum_variance")
    ]
    assert set(applicable["covariance_estimator"]) == {
        "LedoitWolf(assume_centered=False)"
    }
    assert set(applicable["solver_status"]) == {"success"}
    assert np.isfinite(
        applicable["shrinkage_coefficient"].astype(float).to_numpy()
    ).all()


def test_manifest_freezes_provenance_methods_costs_and_mechanical_gates(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    manifest = day16_report.copy_manifest()
    assert reporting.DAY16_ARTIFACT_VERSION == "day16_portfolio_validation_v1"
    assert tuple(manifest["artifact_filenames"]) == (
        reporting.APPROVED_DAY16_ARTIFACT_NAMES
    )
    assert manifest["development_only"] is True
    assert manifest["development_start"] == "2020-01-02"
    assert manifest["development_end"] == "2025-12-31"
    assert manifest["locked_period_accessed"] is False
    assert manifest["day14_eligible_cointegration_pairs"] == 0
    assert tuple(manifest["sleeve_order"]) == SLEEVE_IDS
    assert tuple(manifest["allocation_rule_order"]) == allocation.ALLOCATION_RULES
    assert len(manifest["fold_definitions"]) == 4
    assert manifest["canonical_expected_portfolio_return_rows"] == 1003
    assert manifest["allocation_cost_rate"] == 1.0 / 10_000.0
    assert manifest["annualization_sessions"] == 252.0
    assert manifest["risk_free_rate"] == 0.0
    assert manifest["historical_var_return_quantile"] == 0.05
    assert manifest["maximum_weight"] == 0.35
    assert manifest["weight_tolerance"] == 1e-12
    assert manifest["minimum_variance_solver"] == {
        "method": "SLSQP",
        "initial_weights": [1.0 / 6.0] * 6,
        "analytical_gradient": "2 * covariance * weights",
        "ftol": 1e-12,
        "maxiter": 10_000,
        "constraint_tolerance": 1e-10,
        "random_initialization": False,
    }
    assert len(manifest["minimum_variance_covariance_estimates"]) == 4
    for record in manifest["minimum_variance_covariance_estimates"]:
        assert tuple(record["sleeve_order"]) == SLEEVE_IDS
        matrix = np.asarray(record["covariance_matrix"], dtype="float64")
        assert matrix.shape == (6, 6)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-15)
        assert 0.0 <= record["shrinkage_coefficient"] <= 1.0

    assert all(manifest["mechanical_gates"].values())
    assert manifest["evaluation_complete"] is True
    assert manifest["row_counts"] == {
        "allocation_weights": 72,
        "allocation_diagnostics": 12,
        "fold_portfolio_performance": 12,
        "aggregate_portfolio_performance": 3,
        "portfolio_return_panel": len(day16_report.portfolio_return_panel),
    }

    false_flags = (
        "locked_period_accessed",
        "forward_fill_used",
        "backward_fill_used",
        "interpolation_used",
        "strategy_cost_counted_twice",
        "expected_return_inputs_used",
        "cost_aware_optimization_used",
        "intrafold_rebalancing_used",
        "ranking_performed",
        "winner_selection_performed",
        "sleeve_removal_performed",
        "leverage_used",
        "short_allocation_used",
        "borrowing_used",
        "profitability_gate_used",
        "paper_or_live_orders_submitted",
    )
    assert all(manifest[name] is False for name in false_flags)
    assert manifest["artifact_sha256"] == {}


def test_manifest_contains_no_timestamps_or_absolute_paths(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    text = reporting._manifest_bytes(day16_report.copy_manifest()).decode()
    assert "/Users/" not in text
    assert "T00:00:00" not in text
    assert "timestamp" not in text.lower()


def test_report_wording_is_neutral_complete_and_has_one_final_newline(
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    markdown = day16_report.report
    lowered = markdown.lower()
    for required in (
        "statistical diversification",
        "economic performance",
        "allocation rules were predeclared",
        "training rows only",
        "holds those weights fixed throughout that fold",
        "negative results are valid",
        "constrained minimum variance is an optimization rule",
        "no rule was selected using realized performance",
        "day 14 zero-pair cointegration result is retained",
        "locked january–june 2026 period was not accessed",
        "evaluation_complete field is true",
    ):
        assert required in lowered
    for forbidden in ("best", "winner", "optimal strategy", "deployment"):
        assert forbidden not in lowered
    for fold_id in ("wf_2022", "wf_2023", "wf_2024", "wf_2025"):
        assert markdown.count(fold_id) >= 6
    for rule in allocation.ALLOCATION_RULES:
        assert markdown.count(rule) >= 9
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_negative_metrics_do_not_control_evaluation_complete() -> None:
    results = allocation.analyze_portfolio_allocation_panel(
        make_day16_panel(mean_return=-0.0015)
    )
    report = reporting.build_day16_portfolio_validation_report(results)
    assert report.copy_manifest()["evaluation_complete"] is True
    assert (report.aggregate_portfolio_performance["cumulative_return"] < 0).any()
    assert (report.aggregate_portfolio_performance["sharpe_ratio"] < 0).any()


def test_incomplete_analysis_is_rejected_before_reporting(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    incomplete = clone_results(day16_results, evaluation_complete=False)
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="mechanically complete",
    ):
        reporting.build_day16_portfolio_validation_report(incomplete)


def test_schema_and_row_order_corruption_fail_closed(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    weights = day16_results.copy_allocation_weights().rename(
        columns={"weight": "changed"}
    )
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="unexpected schema",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(day16_results, allocation_weights=weights)
        )

    reordered = day16_results.copy_allocation_diagnostics().iloc[::-1]
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="frozen fold/rule order",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(day16_results, allocation_diagnostics=reordered)
        )


def test_locked_return_row_and_asymmetric_covariance_fail_closed(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    returns = day16_results.copy_portfolio_return_panel()
    returns.loc[returns.index[-1], "session_date"] = pd.Timestamp(
        "2026-01-02", tz="UTC"
    )
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="Locked or later",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(day16_results, portfolio_return_panel=returns)
        )

    covariance = day16_results.copy_minimum_variance_covariances()
    covariance.loc[0, SLEEVE_IDS[1]] += 0.01
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="finite and symmetric",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(
                day16_results,
                minimum_variance_covariances=covariance,
            )
        )


def test_cross_table_metric_cost_and_constraint_corruption_fail_closed(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    performance = day16_results.copy_fold_portfolio_performance()
    performance.loc[0, "cumulative_return"] += 0.01
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="metric cumulative_return does not reconcile",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(
                day16_results,
                fold_portfolio_performance=performance,
            )
        )

    diagnostics = day16_results.copy_allocation_diagnostics()
    diagnostics.loc[0, "allocation_cost"] += 0.01
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="turnover or cost does not reconcile",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(
                day16_results,
                allocation_diagnostics=diagnostics,
            )
    )

    weights = day16_results.copy_allocation_weights()
    weights["constraint_valid"] = weights["constraint_valid"].astype(object)
    weights.loc[0, "constraint_valid"] = "True"
    with pytest.raises(
        reporting.Day16PortfolioValidationReportError,
        match="explicit true booleans",
    ):
        reporting.build_day16_portfolio_validation_report(
            clone_results(day16_results, allocation_weights=weights)
        )


@pytest.mark.parametrize("invalid", [None, 7, object()])
def test_invalid_result_types_fail_closed(invalid: object) -> None:
    with pytest.raises(TypeError, match="PortfolioAllocationResults"):
        reporting.build_day16_portfolio_validation_report(invalid)  # type: ignore[arg-type]
