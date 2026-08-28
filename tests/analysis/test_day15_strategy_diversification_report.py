"""Report contracts for Day 15 strategy diversification."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import numpy as np

import systematic_alpha.analysis.strategy_diversification as diversification
from systematic_alpha.analysis.day15_strategy_diversification_report import (
    APPROVED_DAY15_ARTIFACT_NAMES,
    DAY15_ARTIFACT_VERSION,
    Day15StrategyDiversificationReport,
    build_day15_strategy_diversification_report,
)
from systematic_alpha.analysis.trend_family_robustness import (
    DEVELOPMENT_DATASET_ID,
)
from tests.analysis.test_strategy_diversification_statistics import (
    make_nearly_identical_panel,
    make_weakly_correlated_panel,
)


def build_report(
    *,
    feasible: bool = True,
) -> Day15StrategyDiversificationReport:
    """Build one deterministic synthetic Day 15 report."""

    panel = (
        make_weakly_correlated_panel()
        if feasible
        else make_nearly_identical_panel()
    )
    results = diversification.analyze_strategy_diversification_panel(panel)
    return build_day15_strategy_diversification_report(results)


def test_report_container_has_exact_fields_and_defensive_copies() -> None:
    report = build_report()

    assert isinstance(report, Day15StrategyDiversificationReport)
    assert is_dataclass(report)
    assert report.__dataclass_params__.frozen
    assert not hasattr(report, "__dict__")
    assert tuple(field.name for field in fields(report)) == (
        "sleeve_input_diagnostics",
        "full_sample_pairwise_correlations",
        "fold_pairwise_correlations",
        "fold_covariance_diagnostics",
        "ensemble_feasibility",
        "manifest",
        "report",
    )

    for name in (
        "sleeve_input_diagnostics",
        "full_sample_pairwise_correlations",
        "fold_pairwise_correlations",
        "fold_covariance_diagnostics",
        "ensemble_feasibility",
    ):
        retained = getattr(report, name)
        copied = getattr(report, f"copy_{name}")()
        assert copied.equals(retained)
        assert copied is not retained
        original = retained.iloc[0, 0]
        replacement = (
            "changed"
            if isinstance(original, str)
            else not original
            if isinstance(original, (bool, np.bool_))
            else 999
        )
        copied.iloc[0, 0] = replacement
        assert retained.iloc[0, 0] == original

    manifest = report.copy_manifest()
    manifest["report_id"] = "changed"
    assert report.copy_manifest()["report_id"] == (
        "day15_strategy_diversification"
    )


def test_report_tables_preserve_exact_phase1_schemas_and_counts() -> None:
    report = build_report()

    expected = (
        (
            report.sleeve_input_diagnostics,
            diversification.SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
            6,
        ),
        (
            report.full_sample_pairwise_correlations,
            diversification.FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS,
            15,
        ),
        (
            report.fold_pairwise_correlations,
            diversification.FOLD_PAIRWISE_CORRELATION_COLUMNS,
            120,
        ),
        (
            report.fold_covariance_diagnostics,
            diversification.FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS,
            8,
        ),
        (
            report.ensemble_feasibility,
            diversification.ENSEMBLE_FEASIBILITY_COLUMNS,
            1,
        ),
    )
    for table, columns, rows in expected:
        assert tuple(table.columns) == columns
        assert len(table) == rows

    assert tuple(report.sleeve_input_diagnostics["sleeve_id"]) == (
        diversification.SLEEVE_IDS
    )
    assert len(report.fold_pairwise_correlations) == 4 * 2 * 15


def test_manifest_freezes_provenance_tolerances_thresholds_and_safety() -> None:
    manifest = build_report().copy_manifest()

    assert DAY15_ARTIFACT_VERSION == (
        "day15_strategy_diversification_v1"
    )
    assert tuple(manifest["artifact_filenames"]) == (
        APPROVED_DAY15_ARTIFACT_NAMES
    )
    assert manifest["report_id"] == "day15_strategy_diversification"
    assert manifest["artifact_version"] == DAY15_ARTIFACT_VERSION
    assert manifest["schema_version"] == 1
    assert manifest["development_only"] is True
    assert manifest["dataset_id"] == DEVELOPMENT_DATASET_ID
    assert manifest["frequency"] == "15min"
    assert manifest["development_start"] == "2020-01-02"
    assert manifest["development_end"] == "2025-12-31"
    assert manifest["locked_period_accessed"] is False
    assert manifest["covariance_estimator"] == (
        "ordinary sample covariance"
    )
    assert manifest["covariance_ddof"] == 1
    assert "exact common session dates" in manifest["alignment_method"]

    assert manifest["sleeve_count"] == 6
    assert tuple(
        sleeve["sleeve_id"] for sleeve in manifest["sleeve_universe"]
    ) == diversification.SLEEVE_IDS
    assert manifest["strategy_configuration_ids"] == (
        diversification.CONFIGURATION_IDS
    )
    assert len(manifest["fold_definitions"]) == 4
    assert tuple(
        fold["fold_id"] for fold in manifest["fold_definitions"]
    ) == ("wf_2022", "wf_2023", "wf_2024", "wf_2025")

    assert manifest["numerical_tolerances"] == {
        "variance_tolerance": diversification.VARIANCE_TOLERANCE,
        "psd_tolerance": diversification.PSD_TOLERANCE,
    }
    assert manifest["feasibility_thresholds"] == {
        "minimum_training_sessions": (
            diversification.MIN_TRAINING_SESSIONS
        ),
        "minimum_test_sessions": diversification.MIN_TEST_SESSIONS,
        "maximum_absolute_correlation": (
            diversification.MAX_ABSOLUTE_CORRELATION
        ),
        "minimum_median_effective_rank": (
            diversification.MIN_MEDIAN_EFFECTIVE_RANK
        ),
        "maximum_median_pc1_share": (
            diversification.MAX_MEDIAN_PC1_SHARE
        ),
        "minimum_median_test_diversification_ratio": (
            diversification.MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO
        ),
    }

    false_flags = (
        "locked_period_accessed",
        "forward_fill_used",
        "backward_fill_used",
        "interpolation_used",
        "covariance_repair_used",
        "covariance_shrinkage_used",
        "optimisation_performed",
        "ranking_performed",
        "winner_selection_performed",
        "sleeve_removal_performed",
        "leverage_used",
        "profitability_claimed",
    )
    assert all(manifest[name] is False for name in false_flags)


def test_report_wording_is_neutral_complete_and_has_one_final_newline() -> None:
    markdown = build_report().report
    lowered = markdown.lower()

    for required in (
        "statistical diversification",
        "economic performance",
        "portfolio allocation",
        "low correlation does not automatically create alpha",
        "equal weights of 1/6 are used only as a neutral diagnostic",
        "no portfolio weights were optimised",
        "no strategy was ranked or removed",
        "no profitability improvement is claimed",
        "locked january–june 2026 period was not accessed",
        "a false feasibility outcome is valid",
    ):
        assert required in lowered

    for forbidden in (
        "optimal portfolio",
        "best combination",
        "guaranteed risk reduction",
        "expected outperformance",
        "profitable ensemble",
        "winner",
        "allocation recommendation",
    ):
        assert forbidden not in lowered

    for section in range(1, 10):
        assert f"## {section}." in markdown
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_feasible_and_infeasible_outcomes_render_without_reinterpretation() -> None:
    feasible = build_report(feasible=True)
    infeasible = build_report(feasible=False)

    assert feasible.copy_manifest()["ensemble_feasible"] is True
    assert infeasible.copy_manifest()["ensemble_feasible"] is False
    assert "outcome is **true**" in feasible.report
    assert "exhibit sufficient return-stream diversification" in (
        feasible.report
    )
    assert "outcome is **false**" in infeasible.report
    assert "exhibit insufficient return-stream diversification" in (
        infeasible.report
    )
