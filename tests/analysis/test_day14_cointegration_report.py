"""Report contracts for Day 14 cointegration feasibility."""

from __future__ import annotations

from dataclasses import is_dataclass

from systematic_alpha.analysis.cointegration_feasibility import (
    run_cointegration_feasibility,
)
from systematic_alpha.analysis.day14_cointegration_report import (
    APPROVED_DAY14_ARTIFACT_NAMES,
    DAY14_ARTIFACT_VERSION,
    Day14CointegrationReport,
    build_day14_cointegration_report,
)
from tests.analysis.test_cointegration_statistics import (
    make_cointegrated_bars,
)


def test_day14_artifact_contract_is_frozen() -> None:
    assert DAY14_ARTIFACT_VERSION == (
        "cointegration_ou_feasibility_v1"
    )

    assert APPROVED_DAY14_ARTIFACT_NAMES == (
        "pair_input_diagnostics.csv",
        "series_integration_diagnostics.csv",
        "cointegration_diagnostics.csv",
        "fold_stability_diagnostics.csv",
        "ou_diagnostics.csv",
        "pair_eligibility.csv",
        "manifest.json",
        "report.md",
    )


def test_report_is_immutable_and_neutral() -> None:
    results = run_cointegration_feasibility(
        make_cointegrated_bars()
    )
    report = build_day14_cointegration_report(
        results
    )

    assert isinstance(
        report,
        Day14CointegrationReport,
    )
    assert is_dataclass(report)
    assert report.__dataclass_params__.frozen
    assert not hasattr(report, "__dict__")

    assert len(
        report.pair_input_diagnostics
    ) == 3
    assert len(
        report.series_integration_diagnostics
    ) == 6
    assert len(
        report.cointegration_diagnostics
    ) == 3
    assert len(
        report.fold_stability_diagnostics
    ) == 12
    assert len(
        report.ou_diagnostics
    ) == 3
    assert len(
        report.pair_eligibility
    ) == 3

    manifest = report.copy_manifest()

    assert manifest["report_id"] == (
        "day14_cointegration_feasibility"
    )
    assert manifest["development_only"] is True
    assert (
        manifest["locked_period_accessed"]
        is False
    )
    assert manifest["tuning_performed"] is False
    assert manifest["ranking_performed"] is False
    assert (
        manifest["winner_selection_performed"]
        is False
    )
    assert manifest["candidate_pair_count"] == 3
    assert manifest["eligible_pair_count"] == 1

    assert (
        "# Day 14 — Cointegration and OU Feasibility"
        in report.report
    )
    assert "No profitability criterion" in report.report
    assert "SPY_QQQ" in report.report
