"""Deterministic contracts for the Day 28 corrected OU timing evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from systematic_alpha.analysis.day28_ou_causal_timing import (
    ANNUAL_CONCENTRATION_COLUMNS,
    APPROVED_ARTIFACT_NAMES,
    CORRECTED_AGGREGATE_COLUMNS,
    CORRECTED_COST_COLUMNS,
    CORRECTED_FOLD_COLUMNS,
    CORRECTED_INFERENCE_COLUMNS,
    CORRECTED_TIMING,
    DAY17_COMPARATOR_FILES,
    DAY26_COMPARATOR_FILES,
    HISTORICAL_TIMING,
    PHASE2_OU_COMPARISON_COLUMNS,
    SPECIFICATION_VERSION,
    TIMING_COMPARISON_COLUMNS,
    Day28OuCausalTimingError,
    assemble_day28_results,
    audit_day28_development_input,
    load_historical_comparators,
    verify_comparator_snapshot,
    write_day28_artifacts,
)
from systematic_alpha.analysis.phase2_profitability import Phase2ProfitabilityResults
from systematic_alpha.analysis.reversion_inference import (
    CONFIGURATION_IDS,
    ReversionInferenceResults,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY17 = PROJECT_ROOT / "artifacts/day17"
DAY26 = PROJECT_ROOT / "artifacts/day26"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in ("SPY", "QQQ", "IWM"):
        for timestamp, price in (
            (pd.Timestamp("2025-01-02 14:30Z"), 100.0),
            (pd.Timestamp("2025-01-02 14:45Z"), 101.0),
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.5,
                    "volume": 1_000.0,
                    "trade_count": 100,
                    "vwap": price + 0.25,
                    "source": "alpaca",
                    "feed": "sip",
                }
            )
    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def historical():
    return load_historical_comparators(
        day17_directory=DAY17,
        day26_directory=DAY26,
    )


@pytest.fixture(scope="module")
def assembled(historical):
    reversion = ReversionInferenceResults(
        signal_diagnostics=pd.DataFrame(),
        fold_performance=pd.read_csv(DAY17 / "fold_performance.csv"),
        aggregate_performance=historical.day17_aggregate,
        inference_results=pd.read_csv(DAY17 / "inference_results.csv"),
        cost_sensitivity=historical.day17_cost_sensitivity,
        session_return_panel=historical.day17_session_returns,
    )
    phase2 = Phase2ProfitabilityResults(
        data_quality={},
        aggregate_performance=historical.day26_aggregate,
        fold_performance=pd.DataFrame(),
        cost_sensitivity=pd.DataFrame(),
        comparison=historical.day26_comparison,
        inference=historical.day26_inference,
        session_returns=historical.day26_session_returns,
    )
    return assemble_day28_results(
        reversion_results=reversion,
        phase2_results=phase2,
        historical=historical,
        data_audit={
            "source_dataset_path": "development.parquet",
            "source_sha256": "0" * 64,
            "timestamp_min": "2020-01-02T14:30:00+00:00",
            "timestamp_max": "2025-12-31T20:45:00+00:00",
            "session_min": "2020-01-02",
            "session_max": "2025-12-31",
            "symbols": ["IWM", "QQQ", "SPY"],
            "frequency": "15min",
            "contains_2026": False,
        },
    )


def test_development_audit_rejects_injected_2026_before_calculation() -> None:
    source = _development_rows()
    source.loc[0, "timestamp"] = pd.Timestamp("2026-01-02 14:30Z")
    with pytest.raises(Day28OuCausalTimingError, match="prohibited 2026"):
        audit_day28_development_input(
            source,
            source_dataset_path="development.parquet",
            source_sha256="0" * 64,
        )


def test_development_audit_records_exact_scope_and_frequency() -> None:
    audit = audit_day28_development_input(
        _development_rows(),
        source_dataset_path="development.parquet",
        source_sha256="0" * 64,
    )
    assert audit["session_min"] == "2025-01-02"
    assert audit["session_max"] == "2025-01-02"
    assert audit["frequency"] == "15min"
    assert audit["contains_2026"] is False
    assert audit["symbols"] == ["IWM", "QQQ", "SPY"]


def test_exact_output_schemas_and_all_frozen_configurations(assembled) -> None:
    assert tuple(assembled.corrected_fold_performance.columns) == CORRECTED_FOLD_COLUMNS
    assert (
        tuple(assembled.corrected_aggregate_performance.columns)
        == CORRECTED_AGGREGATE_COLUMNS
    )
    assert tuple(assembled.corrected_cost_sensitivity.columns) == CORRECTED_COST_COLUMNS
    assert (
        tuple(assembled.corrected_inference_results.columns)
        == CORRECTED_INFERENCE_COLUMNS
    )
    assert (
        tuple(assembled.historical_vs_corrected_timing.columns)
        == TIMING_COMPARISON_COLUMNS
    )
    assert (
        tuple(assembled.corrected_phase2_ou_comparison.columns)
        == PHASE2_OU_COMPARISON_COLUMNS
    )
    assert (
        tuple(assembled.annual_concentration_diagnostics.columns)
        == ANNUAL_CONCENTRATION_COLUMNS
    )
    assert (
        tuple(
            assembled.corrected_aggregate_performance[
                "configuration_id"
            ].drop_duplicates()
        )
        == CONFIGURATION_IDS
    )


def test_timing_labels_are_explicit_and_no_selection_columns_exist(assembled) -> None:
    assert assembled.corrected_aggregate_performance["timing_convention"].eq(
        CORRECTED_TIMING
    ).all()
    assert set(assembled.historical_vs_corrected_timing["timing_convention"]) == {
        HISTORICAL_TIMING,
        CORRECTED_TIMING,
    }
    for frame in (
        assembled.corrected_fold_performance,
        assembled.corrected_aggregate_performance,
        assembled.corrected_cost_sensitivity,
        assembled.corrected_inference_results,
        assembled.historical_vs_corrected_timing,
        assembled.corrected_phase2_ou_comparison,
        assembled.annual_concentration_diagnostics,
    ):
        assert not any(
            token in column.lower()
            for column in frame.columns
            for token in ("winner", "rank", "promotion")
        )


def test_historical_comparators_are_manifest_authenticated_and_immutable(
    tmp_path: Path,
) -> None:
    day17_copy = tmp_path / "day17"
    day26_copy = tmp_path / "day26"
    day17_copy.mkdir()
    day26_copy.mkdir()
    for filename in DAY17_COMPARATOR_FILES:
        shutil.copy2(DAY17 / filename, day17_copy / filename)
    for filename in DAY26_COMPARATOR_FILES:
        shutil.copy2(DAY26 / filename, day26_copy / filename)
    evidence = load_historical_comparators(
        day17_directory=day17_copy,
        day26_directory=day26_copy,
    )
    before = dict(evidence.source_hashes)
    verify_comparator_snapshot(before)
    target = day17_copy / "aggregate_performance.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(Day28OuCausalTimingError, match="Immutable comparator changed"):
        verify_comparator_snapshot(before)


def test_phase2_comparison_preserves_frozen_pair_and_signal_impact(assembled) -> None:
    frame = assembled.corrected_phase2_ou_comparison
    assert tuple(frame["configuration_id"].drop_duplicates()) == (
        "ou_vwap_slow_baseline",
        "ou_vwap_slow_cost_margin_phase2",
    )
    assert set(frame["phase_role"]) == {"baseline", "cost_margin_candidate"}
    candidate = frame.loc[frame["phase_role"].eq("cost_margin_candidate")]
    assert candidate["corrected_signal_entry_count_change_vs_baseline"].eq(0.0).all()
    assert candidate[
        "corrected_execution_path_difference_sessions_vs_baseline"
    ].eq(0).all()
    assert assembled.manifest["phase2_expected_convergence_threshold"] == 0.001


def test_writer_is_isolated_no_overwrite_and_byte_deterministic(
    tmp_path: Path, assembled
) -> None:
    comparator_hashes_before = {
        path: _sha256(Path(path)) for path in assembled.comparator_snapshot
    }
    first = tmp_path / "first/day28_ou_causal_timing"
    second = tmp_path / "second/day28_ou_causal_timing"
    first_paths = write_day28_artifacts(assembled, first)
    second_paths = write_day28_artifacts(assembled, second)
    assert tuple(path.name for path in first_paths) == APPROVED_ARTIFACT_NAMES
    assert tuple(path.name for path in second_paths) == APPROVED_ARTIFACT_NAMES
    for filename in APPROVED_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    with pytest.raises(FileExistsError):
        write_day28_artifacts(assembled, first)
    with pytest.raises(Day28OuCausalTimingError, match="basename"):
        write_day28_artifacts(assembled, tmp_path / "day17")
    assert comparator_hashes_before == {
        path: _sha256(Path(path)) for path in assembled.comparator_snapshot
    }


def test_manifest_records_version_scope_settings_and_hashes(
    tmp_path: Path, assembled
) -> None:
    output = tmp_path / "day28_ou_causal_timing"
    write_day28_artifacts(assembled, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SPECIFICATION_VERSION
    assert manifest["timing_version"] == CORRECTED_TIMING
    assert manifest["configuration_ids"] == list(CONFIGURATION_IDS)
    assert manifest["data"]["session_min"] == "2020-01-02"
    assert manifest["data"]["session_max"] == "2025-12-31"
    assert manifest["inference"] == {
        "bootstrap_block_length": 5,
        "bootstrap_replications": 2000,
        "bootstrap_seed": 1701,
        "declared_dsr_trials": 3,
        "hac_lags": 5,
    }
    assert len(manifest["comparator_sources"]) == 10
    assert len(manifest["artifacts"]) == 7
    for artifact in manifest["artifacts"]:
        path = output / artifact["filename"]
        assert artifact["sha256"] == _sha256(path)
    assert manifest["locked_2026_accessed"] is False
    assert manifest["ranking_performed"] is False
    assert manifest["winner_selected"] is False
    assert manifest["promotion_performed"] is False
