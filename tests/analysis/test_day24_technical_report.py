"""Tests for the frozen Day 24 CQF technical-report payload."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

import systematic_alpha.analysis.day24_technical_report as day24


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report_artifact() -> dict[str, object]:
    return day24.build_report_artifact(PROJECT_ROOT)


def test_source_manifests_verify_before_reporting() -> None:
    verified = day24.verify_source_manifests(PROJECT_ROOT)
    assert len(verified) == 18
    assert all(item["verified_files"] > 0 for item in verified)
    assert all(len(str(item["manifest_sha256"])) == 64 for item in verified)


def test_manifest_verifier_fails_closed_on_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = PROJECT_ROOT / "artifacts/day11"
    destination = tmp_path / "artifacts/day11"
    shutil.copytree(source, destination)
    monkeypatch.setattr(day24, "MANIFEST_DIRECTORIES", ("artifacts/day11",))
    with (destination / "aggregate_results.csv").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        day24.verify_source_manifests(tmp_path)


def test_report_has_exact_visible_structure_and_six_charts(
    report_artifact: dict[str, object],
) -> None:
    manifest = report_artifact["manifest"]
    blocks = manifest["blocks"]
    headings = [
        block["body"].splitlines()[0]
        for block in blocks
        if block["type"] == "markdown"
        and re.fullmatch(r"#{1,2} .+", block["body"].splitlines()[0])
    ]
    assert headings[0].startswith("# Axiom:")
    assert headings[1:] == [
        "## 2. Technical summary",
        "## 3. Key economic results",
        "## 4. Research scope, data, timing, and metric definitions",
        "## 5. Price-ratio trend model and evidence",
        "## 6. EMA/MACD trend model and evidence",
        "## 7. Pair-feasibility rejection and OU/VWAP reversion design",
        "## 8. Reversion profitability, cost sensitivity, and inference",
        "## 9. Diversification and allocation evidence",
        "## 10. Event-driven parity and numerical/statistical methods",
        "## 11. Broker architecture, order state, reconciliation, and safety",
        "## 12. Execution benchmarking and performance reporting",
        "## 13. Reproducibility, CI, scheduling, and container limitations",
        "## 14. Limitations and threats to validity",
        "## 15. Conclusion and recommended next steps",
        "## 16. References",
    ]
    narrative = " ".join(
        block["body"] for block in blocks if block["type"] == "markdown"
    )
    assert len(re.findall(r"\b[\w.+/%-]+\b", narrative)) >= 8_000
    assert narrative.count("https://doi.org/") >= 10
    assert "I have not established deployable profitability." in narrative
    assert "bootstrap interval crossed zero" in narrative
    assert (
        "### 14.2 The predeclared Phase II development test did not improve the base case"
        in narrative
    )
    assert "Improved profitability may be claimed only if" in narrative
    assert "improved profitability has not been demonstrated" in narrative
    assert "cut long-flat turnover by 10.27%" in narrative
    assert "cost-margin gate did not remove a single baseline entry" in narrative
    assert "environment: conda" not in narrative
    assert "## 17." not in narrative
    assert narrative.count("I ") >= 100
    assert len(manifest["charts"]) == 6
    assert [chart["id"] for chart in manifest["charts"]] == [
        "trend_walk_forward_returns",
        "cross_market_annualized_returns",
        "reversion_cost_sensitivity",
        "annual_trend_vs_spy",
        "annual_ou_vs_equal_weight",
        "annual_short_sleeve_effect",
    ]


def test_chart_datasets_reconcile_to_frozen_rows(
    report_artifact: dict[str, object],
) -> None:
    datasets = report_artifact["snapshot"]["datasets"]
    assert len(datasets["trend_walk_forward"]) == 12
    assert len(datasets["cross_market"]) == 27
    assert len(datasets["reversion_costs"]) == 12
    assert len(datasets["annual_trend_comparison"]) == 16
    assert len(datasets["annual_reversion_comparison"]) == 8
    assert len(datasets["annual_short_effect_chart"]) == 8
    assert len(datasets["annual_ou_concentration"]) == 4
    assert all(
        row["excess_return"] > 0
        for row in datasets["annual_trend_comparison"]
        if row["series_type"] == "strategy" and row["year"] == 2022
    )
    assert all(
        row["excess_return"] < 0
        for row in datasets["annual_trend_comparison"]
        if row["series_type"] == "strategy" and row["year"] != 2022
    )
    slow_one_bps = next(
        row
        for row in datasets["reversion_costs"]
        if row["configuration_id"] == "ou_vwap_slow" and row["cost_bps"] == 1.0
    )
    assert slow_one_bps["cumulative_return"] == pytest.approx(0.0603485555378)
    assert all(
        chart["referenceLines"][0]["value"] == 0
        for chart in report_artifact["manifest"]["charts"]
    )


def test_headline_claims_are_honest_and_auditable(
    report_artifact: dict[str, object],
) -> None:
    outcomes = report_artifact["snapshot"]["datasets"]["model_outcomes"]
    outcome_by_id = {row["model_id"]: row for row in outcomes}
    assert outcome_by_id["ou_vwap_slow_equal_weight_locked"][
        "cumulative_return"
    ] == pytest.approx(0.00431073822018)
    assert outcome_by_id["price_ratio_long_short_neutral_locked"][
        "cumulative_return"
    ] == pytest.approx(-0.0190027556733)
    assert outcome_by_id["ema_macd_long_short_neutral_locked"][
        "cumulative_return"
    ] == pytest.approx(-0.049354694753)
    assert outcome_by_id["price_ratio_long_flat_development"][
        "cumulative_return"
    ] == pytest.approx(-0.00451722985778)
    assert (
        "inconclusive"
        in outcome_by_id["ou_vwap_slow_equal_weight_development"]["decision"]
    )
    claims = day24.claim_inventory(PROJECT_ROOT)
    assert len(claims) == 14
    assert claims[-2]["claim_id"] == "annual_regime_diagnostic"
    assert claims[-1]["claim_id"] == "profitability_not_established"
    assert claims[-1]["status"] == "complete"
    assert sum(claim["status"] == "provisional" for claim in claims) == 1


def test_live_boundary_and_source_paths_fail_closed(
    report_artifact: dict[str, object],
) -> None:
    snapshot = report_artifact["snapshot"]
    assert snapshot["status"] == "partial"
    assert snapshot["accessIssues"] == [
        {
            "id": "live_campaign_pending",
            "dataset": "execution",
            "message": "Empirical Day 22 paper fills and slippage are not yet available; displayed execution rows are synthetic known-answer fixtures only.",
        }
    ]
    for source in report_artifact["sources"]:
        path = Path(source["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert len(source["sha256"]) == 64
        assert source["query"]["sql"].startswith("SELECT * FROM reviewed_file(")
    assert {
        "day25_locked_performance",
        "day25_locked_methodology",
        "day26_comparison",
        "day26_aggregate",
        "day26_inference",
        "day26_methodology",
        "annual_regime_comparison",
        "annual_short_effect",
        "annual_ou_concentration",
        "annual_regime_methodology",
    }.issubset({source["id"] for source in report_artifact["sources"]})


def test_chart_and_claim_inventory_schemas_are_frozen() -> None:
    charts = day24.chart_inventory(PROJECT_ROOT)
    claims = day24.claim_inventory(PROJECT_ROOT)
    assert tuple(charts[0]) == day24.CHART_COLUMNS
    assert tuple(claims[0]) == day24.CLAIM_COLUMNS
    assert [row["row_count"] for row in charts] == [12, 27, 12, 16, 8, 8]


def test_report_equations_use_complete_latex_group_contract(
    report_artifact: dict[str, object],
) -> None:
    blocks = report_artifact["manifest"]["blocks"]
    narrative = "\n".join(
        block["body"] for block in blocks if block["type"] == "markdown"
    )
    expected_markers = {
        day24.report_math_placeholder(group_id)
        for group_id, _columns, _equations in day24.REPORT_MATH_GROUPS
    }
    observed_markers = set(re.findall(r"\[\[AXIOM_LATEX_[A-Z0-9_]+\]\]", narrative))
    assert observed_markers == expected_markers
    assert all(narrative.count(marker) == 1 for marker in expected_markers)
    assert len(day24.REPORT_MATH_GROUPS) == 11
    assert sum(len(item[2]) for item in day24.REPORT_MATH_GROUPS) == 40
    assert "~~~text" not in narrative
