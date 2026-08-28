"""Deterministic contracts for the Day 29 accounting experiment."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.day29_fixed_holdings_portfolio_experiment import (
    ACCOUNTING_VERSION,
    AGGREGATE_PERFORMANCE_COLUMNS,
    APPROVED_ARTIFACT_NAMES,
    CORRECTED_WEIGHT_PATH_COLUMNS,
    DAY16_COMPARATOR_FILES,
    DAY25_COMPARATOR_FILES,
    ENDING_DRIFT_COLUMNS,
    EXPERIMENT_VERSION,
    FOLD_PERFORMANCE_COLUMNS,
    FOLD_TURNOVER_COLUMNS,
    INVARIANCE_COLUMNS,
    OUTPUT_DIRECTORY_BASENAME,
    RETURN_COMPARISON_COLUMNS,
    WEALTH_IDENTITY_COLUMNS,
    WEALTH_IDENTITY_TOLERANCE,
    Day29FixedHoldingsExperimentError,
    build_day29_experiment,
    load_comparator_snapshot,
    sha256_file,
    validate_method_invariants,
    verify_comparator_snapshot,
    write_day29_artifacts,
)
from systematic_alpha.analysis.portfolio_allocation_validation import (
    FIXED_HOLDINGS_ACCOUNTING_VERSION,
    MAXIMUM_WEIGHT,
    FixedHoldingsPortfolioAllocationResults,
    PortfolioAllocationResults,
    analyze_portfolio_allocation_panel,
    analyze_portfolio_allocation_panel_fixed_holdings,
)
from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS
from tests.day16_fixtures import make_day16_panel


FIXED_TIMESTAMP = "2026-08-15T00:00:00+00:00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_file(tmp_path: Path) -> dict[str, str]:
    comparator = tmp_path / "comparator.csv"
    comparator.write_text("frozen\n", encoding="utf-8")
    return {comparator.resolve().as_posix(): _sha256(comparator)}


def _build(
    panel: pd.DataFrame | None = None,
    *,
    snapshot: dict[str, str] | None = None,
):
    return build_day29_experiment(
        make_day16_panel() if panel is None else panel,
        source_dataset_path="synthetic-development.parquet",
        source_sha256="0" * 64,
        comparator_snapshot={} if snapshot is None else snapshot,
        generation_timestamp=FIXED_TIMESTAMP,
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_day16_panel()


@pytest.fixture(scope="module")
def results(panel: pd.DataFrame):
    return _build(panel)


def _replace_portfolio_frame(
    original: PortfolioAllocationResults,
    **updates: pd.DataFrame,
) -> PortfolioAllocationResults:
    values = {
        "allocation_weights": original.copy_allocation_weights(),
        "allocation_diagnostics": original.copy_allocation_diagnostics(),
        "fold_portfolio_performance": original.copy_fold_portfolio_performance(),
        "aggregate_portfolio_performance": (
            original.copy_aggregate_portfolio_performance()
        ),
        "portfolio_return_panel": original.copy_portfolio_return_panel(),
        "minimum_variance_covariances": (
            original.copy_minimum_variance_covariances()
        ),
        "evaluation_complete": True,
    }
    values.update(updates)
    return PortfolioAllocationResults(**values)


def test_versions_output_schemas_and_no_selection_columns(results) -> None:
    assert EXPERIMENT_VERSION == "day29_fixed_holdings_portfolio_experiment_v1"
    assert ACCOUNTING_VERSION == "fixed_holdings_fold_rebalance_v1"
    assert ACCOUNTING_VERSION == FIXED_HOLDINGS_ACCOUNTING_VERSION
    assert tuple(results.target_and_covariance_invariance.columns) == (
        INVARIANCE_COLUMNS
    )
    assert tuple(results.fold_performance_comparison.columns) == (
        FOLD_PERFORMANCE_COLUMNS
    )
    assert tuple(results.aggregate_performance_comparison.columns) == (
        AGGREGATE_PERFORMANCE_COLUMNS
    )
    assert tuple(results.fold_turnover_comparison.columns) == FOLD_TURNOVER_COLUMNS
    assert tuple(results.ending_weight_drift.columns) == ENDING_DRIFT_COLUMNS
    assert tuple(results.corrected_weight_path.columns) == (
        CORRECTED_WEIGHT_PATH_COLUMNS
    )
    assert tuple(results.wealth_identity_checks.columns) == WEALTH_IDENTITY_COLUMNS
    assert tuple(results.portfolio_return_comparison.columns) == (
        RETURN_COMPARISON_COLUMNS
    )
    for frame in (
        results.target_and_covariance_invariance,
        results.fold_performance_comparison,
        results.aggregate_performance_comparison,
        results.fold_turnover_comparison,
        results.ending_weight_drift,
        results.corrected_weight_path,
        results.wealth_identity_checks,
        results.portfolio_return_comparison,
    ):
        assert not any(
            token in column.lower()
            for column in frame.columns
            for token in ("winner", "rank", "promotion")
        )


def test_targets_covariances_shrinkage_and_folds_are_exact_invariants(results) -> None:
    evidence = results.target_and_covariance_invariance
    assert evidence["exact_equal"].all()
    assert set(evidence["invariant_type"]) == {
        "target_weight",
        "ledoit_wolf_covariance",
        "ledoit_wolf_shrinkage",
        "fold_definition",
    }
    numeric = pd.to_numeric(
        evidence.loc[
            evidence["invariant_type"].ne("fold_definition"),
            "corrected_minus_historical",
        ]
    )
    np.testing.assert_array_equal(numeric.to_numpy(), 0.0)
    assert len(evidence.loc[evidence["invariant_type"].eq("target_weight")]) == 72
    assert len(
        evidence.loc[evidence["invariant_type"].eq("ledoit_wolf_covariance")]
    ) == 144
    assert len(evidence.loc[evidence["invariant_type"].eq("fold_definition")]) == 4


def test_sign_convention_is_corrected_minus_historical(results) -> None:
    aggregate = results.aggregate_performance_comparison
    np.testing.assert_allclose(
        aggregate["corrected_minus_historical_cumulative_gross_return"],
        aggregate["corrected_cumulative_gross_return"]
        - aggregate["historical_cumulative_gross_return"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        aggregate["corrected_minus_historical_cumulative_net_return_1bp"],
        aggregate["corrected_cumulative_net_return_1bp"]
        - aggregate["historical_cumulative_net_return_1bp"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        aggregate["cost_effect"],
        aggregate["corrected_minus_historical_cumulative_net_return_1bp"]
        - aggregate["return_accounting_effect"],
        rtol=0.0,
        atol=0.0,
    )


def test_fold_boundary_turnover_uses_prior_ending_drift_and_equal_weight_moves(
    results,
) -> None:
    drift = results.ending_weight_drift
    turnover = results.fold_turnover_comparison
    prior = drift.loc[
        drift["fold_id"].eq("wf_2022")
        & drift["allocation_rule"].eq("equal_weight")
    ].sort_values("sleeve_order")
    target = drift.loc[
        drift["fold_id"].eq("wf_2023")
        & drift["allocation_rule"].eq("equal_weight")
    ].sort_values("sleeve_order")
    expected = float(
        np.abs(
            target["target_weight"].to_numpy()
            - prior["ending_drifted_weight"].to_numpy()
        ).sum()
    )
    row = turnover.loc[
        turnover["fold_id"].eq("wf_2023")
        & turnover["allocation_rule"].eq("equal_weight")
    ].iloc[0]
    assert row["corrected_turnover"] == pytest.approx(expected, abs=1e-15)
    assert row["corrected_turnover"] > 0.0
    assert row["historical_turnover"] == pytest.approx(0.0, abs=1e-15)
    assert row["corrected_previous_reference"] == (
        "prior_fold_ending_drifted_weights"
    )


def test_cost_is_only_on_first_fold_session(results) -> None:
    returns = results.portfolio_return_comparison
    for (_, _), group in returns.groupby(
        ["fold_id", "allocation_rule"], sort=False
    ):
        assert group.iloc[0]["is_first_fold_session"]
        assert group.iloc[0]["historical_cost_charged"] >= 0.0
        assert group.iloc[0]["corrected_cost_charged"] >= 0.0
        assert not group.iloc[1:]["is_first_fold_session"].any()
        assert group.iloc[1:]["historical_cost_charged"].eq(0.0).all()
        assert group.iloc[1:]["corrected_cost_charged"].eq(0.0).all()
        np.testing.assert_allclose(
            group["historical_gross_return"]
            - group["historical_net_return_1bp"],
            group["historical_cost_charged"],
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            group["corrected_gross_return"]
            - group["corrected_net_return_1bp"],
            group["corrected_cost_charged"],
            rtol=0.0,
            atol=1e-15,
        )


def test_gross_wealth_identity_is_recorded_per_fold_and_rule(results) -> None:
    checks = results.wealth_identity_checks
    assert len(checks) == 12
    assert checks["identity_within_tolerance"].all()
    assert checks["tolerance"].eq(WEALTH_IDENTITY_TOLERANCE).all()
    assert checks["absolute_gross_wealth_identity_residual"].max() <= (
        WEALTH_IDENTITY_TOLERANCE
    )
    np.testing.assert_allclose(
        checks["recursive_gross_terminal_wealth"],
        checks["holdings_gross_terminal_wealth"],
        rtol=WEALTH_IDENTITY_TOLERANCE,
        atol=WEALTH_IDENTITY_TOLERANCE,
    )


def test_ending_and_intrafold_drift_summaries_are_exact(results) -> None:
    drift = results.ending_weight_drift
    path = results.corrected_weight_path
    row = drift.iloc[0]
    subset = path.loc[
        path["fold_id"].eq(row["fold_id"])
        & path["allocation_rule"].eq(row["allocation_rule"])
        & path["sleeve_id"].eq(row["sleeve_id"])
    ]
    assert row["ending_drifted_weight"] == pytest.approx(
        subset.iloc[-1]["post_return_weight"], abs=1e-15
    )
    assert row["ending_minus_target_weight"] == pytest.approx(
        row["ending_drifted_weight"] - row["target_weight"], abs=1e-15
    )
    expected_max = max(
        subset["pre_return_minus_target"].abs().max(),
        subset["post_return_minus_target"].abs().max(),
    )
    assert row["maximum_absolute_intrafold_drift"] == pytest.approx(
        expected_max, abs=1e-15
    )
    assert row["minimum_observed_pre_return_weight"] == pytest.approx(
        subset["pre_return_weight"].min(), abs=1e-15
    )
    assert row["maximum_observed_pre_return_weight"] == pytest.approx(
        subset["pre_return_weight"].max(), abs=1e-15
    )


def test_drift_above_target_cap_is_retained_and_labelled() -> None:
    panel = make_day16_panel()
    mask = panel.index.year == 2022
    panel.loc[mask, SLEEVE_IDS[0]] = 0.25
    evidence = _build(panel)
    above = evidence.corrected_weight_path.loc[
        evidence.corrected_weight_path["post_return_weight"].gt(MAXIMUM_WEIGHT)
    ]
    assert not above.empty
    assert above["post_return_weight_above_target_cap"].all()
    assert above["target_cap_status"].eq(
        "expected_fixed_holdings_drift_above_target_cap"
    ).all()
    assert above["post_return_weight"].max() > MAXIMUM_WEIGHT


def test_deterministic_order_repeated_execution_and_defensive_copies(panel) -> None:
    first = _build(panel)
    second = _build(panel)
    pd.testing.assert_frame_equal(
        first.aggregate_performance_comparison,
        second.aggregate_performance_comparison,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        first.corrected_weight_path,
        second.corrected_weight_path,
        check_exact=True,
    )
    assert first.source_and_method_metadata == second.source_and_method_metadata
    assert tuple(first.aggregate_performance_comparison["allocation_rule"]) == (
        "equal_weight",
        "inverse_volatility",
        "constrained_minimum_variance",
    )
    ordered = first.corrected_weight_path.sort_values(
        ["session_date", "fold_id", "allocation_rule", "sleeve_order"],
        kind="stable",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(first.corrected_weight_path, ordered)
    copied = first.copy_corrected_weight_path()
    copied.loc[0, "pre_return_weight"] = 99.0
    assert first.corrected_weight_path.loc[0, "pre_return_weight"] != 99.0


def test_inputs_are_not_mutated_and_future_returns_do_not_change_history(panel) -> None:
    original = panel.copy(deep=True)
    baseline = _build(panel)
    pd.testing.assert_frame_equal(panel, original, check_exact=True)
    mutated = panel.copy(deep=True)
    mutated.loc[mutated.index[-1], SLEEVE_IDS[-1]] += 0.20
    future = _build(mutated)
    cutoff = mutated.index[-1]
    baseline_earlier = baseline.corrected_weight_path.loc[
        baseline.corrected_weight_path["session_date"].lt(cutoff)
    ].reset_index(drop=True)
    future_earlier = future.corrected_weight_path.loc[
        future.corrected_weight_path["session_date"].lt(cutoff)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_earlier, future_earlier, check_exact=True)
    baseline_returns = baseline.portfolio_return_comparison.loc[
        baseline.portfolio_return_comparison["session_date"].lt(cutoff)
    ].reset_index(drop=True)
    future_returns = future.portfolio_return_comparison.loc[
        future.portfolio_return_comparison["session_date"].lt(cutoff)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_returns, future_returns, check_exact=True)
    pd.testing.assert_frame_equal(
        baseline.target_and_covariance_invariance,
        future.target_and_covariance_invariance,
        check_exact=True,
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_impossible_returns_fail_closed(invalid: float) -> None:
    panel = make_day16_panel()
    panel.iloc[0, 0] = invalid
    match = "finite" if invalid != -1.0 else "strictly greater than -1"
    with pytest.raises(Day29FixedHoldingsExperimentError, match=match):
        _build(panel)


def test_duplicate_unordered_2026_and_sleeve_order_fail_closed() -> None:
    panel = make_day16_panel()
    duplicate = pd.concat([panel.iloc[[0]], panel]).sort_index(kind="stable")
    with pytest.raises(Day29FixedHoldingsExperimentError, match="unique"):
        _build(duplicate)
    unordered = panel.iloc[::-1]
    with pytest.raises(Day29FixedHoldingsExperimentError, match="monotonic"):
        _build(unordered)
    future = panel.copy(deep=True)
    changed_index = list(future.index)
    changed_index[-1] = pd.Timestamp("2026-01-02", tz="UTC")
    future.index = pd.DatetimeIndex(changed_index, name="session_date")
    with pytest.raises(Day29FixedHoldingsExperimentError, match="2026"):
        _build(future)
    reordered = panel.loc[:, list(reversed(SLEEVE_IDS))]
    with pytest.raises(Day29FixedHoldingsExperimentError, match="ordering"):
        _build(reordered)


def test_mismatched_target_covariance_and_fold_fail_closed(panel) -> None:
    historical = analyze_portfolio_allocation_panel(panel)
    corrected = analyze_portfolio_allocation_panel_fixed_holdings(panel)

    altered_weights = corrected.portfolio_results.copy_allocation_weights()
    altered_weights.loc[0, "weight"] += 1e-6
    altered_portfolio = _replace_portfolio_frame(
        corrected.portfolio_results,
        allocation_weights=altered_weights,
    )
    with pytest.raises(Day29FixedHoldingsExperimentError, match="Target weights"):
        validate_method_invariants(
            historical,
            replace(corrected, portfolio_results=altered_portfolio),
        )

    altered_covariance = corrected.portfolio_results.copy_minimum_variance_covariances()
    altered_covariance.loc[0, SLEEVE_IDS[0]] += 1e-12
    altered_portfolio = _replace_portfolio_frame(
        corrected.portfolio_results,
        minimum_variance_covariances=altered_covariance,
    )
    with pytest.raises(Day29FixedHoldingsExperimentError, match="covariance"):
        validate_method_invariants(
            historical,
            replace(corrected, portfolio_results=altered_portfolio),
        )

    altered_returns = corrected.portfolio_results.copy_portfolio_return_panel()
    altered_returns.loc[0, "fold_id"] = "wf_2023"
    altered_portfolio = _replace_portfolio_frame(
        corrected.portfolio_results,
        portfolio_return_panel=altered_returns,
    )
    with pytest.raises(Day29FixedHoldingsExperimentError, match="Fold definitions"):
        validate_method_invariants(
            historical,
            replace(corrected, portfolio_results=altered_portfolio),
        )


def _write_authenticated_comparator(
    directory: Path,
    filenames: tuple[str, ...],
) -> None:
    directory.mkdir()
    hashes: dict[str, str] = {}
    for filename in filenames:
        if filename == "manifest.json":
            continue
        path = directory / filename
        if filename.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text(f"{filename}\n", encoding="utf-8")
        hashes[filename] = _sha256(path)
    (directory / "manifest.json").write_text(
        json.dumps({"artifact_sha256": hashes}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_comparator_manifests_authenticate_and_mutation_fails(tmp_path: Path) -> None:
    day16 = tmp_path / "day16"
    day25 = tmp_path / "day25"
    _write_authenticated_comparator(day16, DAY16_COMPARATOR_FILES)
    _write_authenticated_comparator(day25, DAY25_COMPARATOR_FILES)
    snapshot = load_comparator_snapshot(
        day16_directory=day16,
        day25_directory=day25,
    )
    verify_comparator_snapshot(snapshot)
    target = day16 / "allocation_weights.csv"
    target.write_text("changed\n", encoding="utf-8")
    with pytest.raises(Day29FixedHoldingsExperimentError, match="changed"):
        verify_comparator_snapshot(snapshot)


def test_writer_hashes_manifest_is_deterministic_and_never_overwrites(
    tmp_path: Path,
    panel: pd.DataFrame,
) -> None:
    snapshot = _snapshot_file(tmp_path)
    results = _build(panel, snapshot=snapshot)
    first = tmp_path / "first" / OUTPUT_DIRECTORY_BASENAME
    second = tmp_path / "second" / OUTPUT_DIRECTORY_BASENAME
    first_paths = write_day29_artifacts(results, first)
    second_paths = write_day29_artifacts(results, second)
    assert tuple(path.name for path in first_paths) == APPROVED_ARTIFACT_NAMES
    assert tuple(path.name for path in second_paths) == APPROVED_ARTIFACT_NAMES
    for filename in APPROVED_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["artifact_sha256"].items():
        assert sha256_file(first / filename) == expected
    assert manifest["comparator_hashes_before"] == snapshot
    assert manifest["comparator_hashes_after"] == snapshot
    assert manifest["comparator_hashes_unchanged"] is True
    assert manifest["no_2026_observations_accessed"] is True
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        write_day29_artifacts(results, first)
    with pytest.raises(Day29FixedHoldingsExperimentError, match="basename"):
        write_day29_artifacts(results, tmp_path / "day16")
    assert not any(
        path.suffix.lower() in {".md", ".html", ".ipynb", ".png", ".svg"}
        for path in first.iterdir()
    )


def test_frozen_historical_day16_behavior_is_unchanged(panel) -> None:
    before = analyze_portfolio_allocation_panel(panel)
    _build(panel)
    after = analyze_portfolio_allocation_panel(panel)
    for method in (
        "copy_allocation_weights",
        "copy_allocation_diagnostics",
        "copy_fold_portfolio_performance",
        "copy_aggregate_portfolio_performance",
        "copy_portfolio_return_panel",
        "copy_minimum_variance_covariances",
    ):
        pd.testing.assert_frame_equal(
            getattr(before, method)(),
            getattr(after, method)(),
            check_exact=True,
        )
