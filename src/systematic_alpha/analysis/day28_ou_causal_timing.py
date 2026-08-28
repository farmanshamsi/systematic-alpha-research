"""Versioned development-only evidence for corrected OU execution timing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, Mapping

import numpy as np
import pandas as pd

from systematic_alpha.analysis.phase2_profitability import (
    AGGREGATE_COLUMNS as PHASE2_AGGREGATE_COLUMNS,
    BOOTSTRAP_BLOCK_LENGTH as PHASE2_BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATIONS as PHASE2_BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED as PHASE2_BOOTSTRAP_SEED,
    COMPARISON_COLUMNS as PHASE2_COMPARISON_COLUMNS,
    HAC_LAGS as PHASE2_HAC_LAGS,
    INFERENCE_COLUMNS as PHASE2_INFERENCE_COLUMNS,
    OU_CONFIGURATION_IDS,
    OU_EXPECTED_CONVERGENCE_THRESHOLD,
    OU_SERIES,
    Phase2ProfitabilityResults,
    SESSION_RETURN_COLUMNS as PHASE2_SESSION_RETURN_COLUMNS,
    run_phase2_profitability,
)
from systematic_alpha.analysis.reversion_inference import (
    AGGREGATE_PERFORMANCE_COLUMNS,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIGURATION_IDS,
    COST_SENSITIVITY_COLUMNS,
    COST_STRESS_BPS,
    FOLD_PERFORMANCE_COLUMNS,
    HAC_LAGS,
    INFERENCE_COLUMNS,
    REPORTED_SERIES,
    REQUIRED_SYMBOLS,
    RETURN_PANEL_COLUMNS,
    ReversionInferenceResults,
    run_reversion_inference,
)
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds


SPECIFICATION_VERSION: Final[str] = "day28_ou_causal_timing_v1"
HISTORICAL_TIMING: Final[str] = "historical_close_to_close"
CORRECTED_TIMING: Final[str] = "corrected_next_open_overnight_flat"
AUTHORIZED_START: Final[pd.Timestamp] = pd.Timestamp("2020-01-02", tz="UTC")
AUTHORIZED_END: Final[pd.Timestamp] = pd.Timestamp(
    "2025-12-31 23:59:59.999999999", tz="UTC"
)
EXPECTED_FOLD_IDS: Final[tuple[str, ...]] = (
    "wf_2022",
    "wf_2023",
    "wf_2024",
    "wf_2025",
)
EXPECTED_SOURCE_VALUES: Final[tuple[str, ...]] = ("alpaca",)
EXPECTED_FEED_VALUES: Final[tuple[str, ...]] = ("sip",)
OUTPUT_DIRECTORY_BASENAME: Final[str] = "day28_ou_causal_timing"

CORRECTED_FOLD_FILENAME: Final[str] = "corrected_fold_performance.csv"
CORRECTED_AGGREGATE_FILENAME: Final[str] = "corrected_aggregate_performance.csv"
CORRECTED_COST_FILENAME: Final[str] = "corrected_cost_sensitivity.csv"
CORRECTED_INFERENCE_FILENAME: Final[str] = "corrected_inference_results.csv"
TIMING_COMPARISON_FILENAME: Final[str] = "historical_vs_corrected_timing.csv"
PHASE2_COMPARISON_FILENAME: Final[str] = "corrected_phase2_ou_comparison.csv"
ANNUAL_CONCENTRATION_FILENAME: Final[str] = "annual_concentration_diagnostics.csv"
MANIFEST_FILENAME: Final[str] = "manifest.json"

CSV_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    CORRECTED_FOLD_FILENAME,
    CORRECTED_AGGREGATE_FILENAME,
    CORRECTED_COST_FILENAME,
    CORRECTED_INFERENCE_FILENAME,
    TIMING_COMPARISON_FILENAME,
    PHASE2_COMPARISON_FILENAME,
    ANNUAL_CONCENTRATION_FILENAME,
)
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    *CSV_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
)

CORRECTED_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    "configuration_id",
    "fold_id",
    "series",
    "cost_bps_per_turnover",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_net_return_1bp",
    "annualized_net_return_1bp",
    "annualized_volatility_1bp",
    "sharpe_ratio_1bp",
    "maximum_drawdown_1bp",
    "turnover",
)

CORRECTED_AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    "configuration_id",
    "series",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_gross_return",
    "cumulative_net_return_1bp",
    "annualized_volatility_1bp",
    "sharpe_ratio_1bp",
    "maximum_drawdown_1bp",
    "turnover",
    "nonzero_net_sessions_1bp",
)

CORRECTED_COST_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    *COST_SENSITIVITY_COLUMNS,
)

CORRECTED_INFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    *INFERENCE_COLUMNS,
)

COMPARISON_METRICS: Final[tuple[str, ...]] = (
    "cumulative_gross_return",
    "cumulative_net_return_1bp",
    "annualized_volatility_1bp",
    "sharpe_ratio_1bp",
    "maximum_drawdown_1bp",
    "turnover",
    "nonzero_net_sessions_1bp",
)

TIMING_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "series",
    "timing_convention",
    *COMPARISON_METRICS,
    *(f"change_from_historical_{name}" for name in COMPARISON_METRICS),
)

PHASE2_OU_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    "configuration_id",
    "phase_role",
    "series",
    "cost_bps_per_turnover",
    "historical_cumulative_net_return_1bp",
    "corrected_cumulative_net_return_1bp",
    "change_cumulative_net_return_1bp",
    "historical_turnover",
    "corrected_turnover",
    "change_turnover",
    "historical_signal_entry_count",
    "corrected_signal_entry_count",
    "corrected_signal_entry_count_change_vs_baseline",
    "historical_execution_path_difference_sessions_vs_baseline",
    "corrected_execution_path_difference_sessions_vs_baseline",
    "historical_hac_t_statistic",
    "corrected_hac_t_statistic",
    "historical_bootstrap_mean_ci_lower",
    "historical_bootstrap_mean_ci_upper",
    "corrected_bootstrap_mean_ci_lower",
    "corrected_bootstrap_mean_ci_upper",
    "historical_bootstrap_sharpe_ci_lower",
    "historical_bootstrap_sharpe_ci_upper",
    "corrected_bootstrap_sharpe_ci_lower",
    "corrected_bootstrap_sharpe_ci_upper",
)

ANNUAL_CONCENTRATION_COLUMNS: Final[tuple[str, ...]] = (
    "timing_convention",
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "year",
    "sessions",
    "nonzero_net_sessions_1bp",
    "cumulative_net_return_1bp",
    "mean_session_return_1bp",
    "annualized_volatility_1bp",
    "log_return_contribution",
    "absolute_log_return_share",
    "absolute_log_return_hhi",
)

DAY17_COMPARATOR_FILES: Final[tuple[str, ...]] = (
    "aggregate_performance.csv",
    "cost_sensitivity.csv",
    "session_return_panel.csv",
    "manifest.json",
)
DAY26_COMPARATOR_FILES: Final[tuple[str, ...]] = (
    "aggregate_performance.csv",
    "comparison.csv",
    "inference.csv",
    "session_returns.csv",
    "methodology.json",
    "manifest.json",
)


class Day28OuCausalTimingError(ValueError):
    """Raised when the Day 28 evidence contract cannot be satisfied safely."""


@dataclass(frozen=True, slots=True)
class HistoricalComparatorEvidence:
    """Immutable in-memory snapshot of the saved Day 17 and Day 26 evidence."""

    day17_aggregate: pd.DataFrame
    day17_cost_sensitivity: pd.DataFrame
    day17_session_returns: pd.DataFrame
    day26_aggregate: pd.DataFrame
    day26_comparison: pd.DataFrame
    day26_inference: pd.DataFrame
    day26_session_returns: pd.DataFrame
    day26_methodology: Mapping[str, object]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "day17_aggregate",
            "day17_cost_sensitivity",
            "day17_session_returns",
            "day26_aggregate",
            "day26_comparison",
            "day26_inference",
            "day26_session_returns",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, value.copy(deep=True).reset_index(drop=True))
        object.__setattr__(self, "day26_methodology", dict(self.day26_methodology))
        object.__setattr__(self, "source_hashes", dict(self.source_hashes))


@dataclass(frozen=True, slots=True)
class Day28OuCausalTimingResults:
    """All Day 28 tables plus a comparator snapshot and manifest metadata."""

    corrected_fold_performance: pd.DataFrame
    corrected_aggregate_performance: pd.DataFrame
    corrected_cost_sensitivity: pd.DataFrame
    corrected_inference_results: pd.DataFrame
    historical_vs_corrected_timing: pd.DataFrame
    corrected_phase2_ou_comparison: pd.DataFrame
    annual_concentration_diagnostics: pd.DataFrame
    manifest: Mapping[str, object]
    comparator_snapshot: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "corrected_fold_performance",
            "corrected_aggregate_performance",
            "corrected_cost_sensitivity",
            "corrected_inference_results",
            "historical_vs_corrected_timing",
            "corrected_phase2_ou_comparison",
            "annual_concentration_diagnostics",
        ):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            object.__setattr__(self, name, value.copy(deep=True).reset_index(drop=True))
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(self, "comparator_snapshot", dict(self.comparator_snapshot))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_day28_development_input(
    bars: pd.DataFrame,
    *,
    source_dataset_path: str,
    source_sha256: str,
) -> dict[str, object]:
    """Fail closed before calculation if the development contract is violated."""

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise Day28OuCausalTimingError("Development bars must be non-empty.")
    if not isinstance(source_dataset_path, str) or not source_dataset_path.strip():
        raise Day28OuCausalTimingError("source_dataset_path must be non-empty.")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise Day28OuCausalTimingError("source_sha256 must be lowercase SHA-256 hex.")
    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "source",
        "feed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise Day28OuCausalTimingError(f"Development bars are missing: {missing}.")
    try:
        timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise Day28OuCausalTimingError("Development timestamps are malformed.") from exc
    if timestamps.dt.year.eq(2026).any() or timestamps.max() > AUTHORIZED_END:
        raise Day28OuCausalTimingError("Development input contains a prohibited 2026 row.")
    if timestamps.min() < AUTHORIZED_START:
        raise Day28OuCausalTimingError("Development input predates the authorized start.")

    symbols = bars["symbol"].astype("string").str.strip().str.upper()
    observed_symbols = tuple(sorted(symbols.dropna().unique().tolist()))
    if observed_symbols != tuple(sorted(REQUIRED_SYMBOLS)):
        raise Day28OuCausalTimingError(
            f"Expected exactly {REQUIRED_SYMBOLS}; received {observed_symbols}."
        )
    if pd.DataFrame({"symbol": symbols, "timestamp": timestamps}).duplicated().any():
        raise Day28OuCausalTimingError("Duplicate symbol-timestamp rows are prohibited.")

    sources = tuple(sorted(bars["source"].astype(str).unique().tolist()))
    feeds = tuple(sorted(bars["feed"].astype(str).unique().tolist()))
    if sources != EXPECTED_SOURCE_VALUES or feeds != EXPECTED_FEED_VALUES:
        raise Day28OuCausalTimingError(
            f"Frozen source/feed must be {EXPECTED_SOURCE_VALUES}/{EXPECTED_FEED_VALUES}."
        )

    order = pd.DataFrame({"symbol": symbols, "timestamp": timestamps}).sort_values(
        ["symbol", "timestamp"], kind="stable"
    )
    prior = order.groupby("symbol", observed=True)["timestamp"].shift()
    same_session = order["timestamp"].dt.normalize().eq(prior.dt.normalize())
    intervals = (
        order.loc[same_session, "timestamp"] - prior.loc[same_session]
    ).dt.total_seconds()
    if intervals.empty or not intervals.eq(15.0 * 60.0).all():
        raise Day28OuCausalTimingError("Frozen bar frequency must be exactly 15 minutes.")

    folds = build_walk_forward_folds()
    fold_ids = tuple(fold.fold_id for fold in folds)
    if fold_ids != EXPECTED_FOLD_IDS:
        raise Day28OuCausalTimingError("Frozen chronological fold identifiers changed.")
    for previous, current in zip(folds, folds[1:]):
        if previous.test_end_exclusive != current.test_start:
            raise Day28OuCausalTimingError("Walk-forward test folds are not chronological.")
    for fold in folds:
        if fold.train_end_exclusive > fold.test_start:
            raise Day28OuCausalTimingError("A walk-forward training interval overlaps test.")
    if CONFIGURATION_IDS != ("ou_vwap_fast", "ou_vwap_base", "ou_vwap_slow"):
        raise Day28OuCausalTimingError("Frozen OU configuration identifiers changed.")
    if len(set(CONFIGURATION_IDS)) != len(CONFIGURATION_IDS):
        raise Day28OuCausalTimingError("Frozen OU configurations are not unique.")

    return {
        "source_dataset_path": source_dataset_path,
        "source_sha256": source_sha256,
        "rows": int(len(bars)),
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "session_min": timestamps.min().normalize().date().isoformat(),
        "session_max": timestamps.max().normalize().date().isoformat(),
        "symbols": list(observed_symbols),
        "frequency": "15min",
        "source_values": list(sources),
        "feed_values": list(feeds),
        "contains_2026": False,
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Day28OuCausalTimingError(f"Comparator JSON is unreadable: {path}.") from exc
    if not isinstance(value, dict):
        raise Day28OuCausalTimingError(f"Comparator JSON must be an object: {path}.")
    return value


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    if tuple(frame.columns) != columns:
        raise Day28OuCausalTimingError(f"{name} comparator schema changed.")


def _validate_manifest_hashes(
    directory: Path,
    manifest: Mapping[str, object],
    filenames: tuple[str, ...],
    *,
    day: int,
) -> None:
    if day == 17:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise Day28OuCausalTimingError("Day 17 comparator manifest is malformed.")
        declared = {
            str(row.get("filename")): str(row.get("sha256"))
            for row in artifacts
            if isinstance(row, dict)
        }
    else:
        hashes = manifest.get("hashes")
        if not isinstance(hashes, dict):
            raise Day28OuCausalTimingError("Day 26 comparator manifest is malformed.")
        declared = {str(key): str(value) for key, value in hashes.items()}
    for filename in filenames:
        if filename == "manifest.json":
            continue
        observed = _sha256_file(directory / filename)
        if declared.get(filename) != observed:
            raise Day28OuCausalTimingError(
                f"Day {day} comparator hash mismatch for {filename}."
            )


def load_historical_comparators(
    *,
    day17_directory: str | Path,
    day26_directory: str | Path,
) -> HistoricalComparatorEvidence:
    """Read saved comparators only after their original manifests authenticate them."""

    day17 = Path(day17_directory)
    day26 = Path(day26_directory)
    for directory, filenames in (
        (day17, DAY17_COMPARATOR_FILES),
        (day26, DAY26_COMPARATOR_FILES),
    ):
        if not directory.is_dir():
            raise Day28OuCausalTimingError(f"Comparator directory is missing: {directory}.")
        missing = [name for name in filenames if not (directory / name).is_file()]
        if missing:
            raise Day28OuCausalTimingError(
                f"Comparator files are missing from {directory}: {missing}."
            )

    day17_manifest = _read_json(day17 / "manifest.json")
    day26_manifest = _read_json(day26 / "manifest.json")
    _validate_manifest_hashes(
        day17, day17_manifest, DAY17_COMPARATOR_FILES, day=17
    )
    _validate_manifest_hashes(
        day26, day26_manifest, DAY26_COMPARATOR_FILES, day=26
    )

    day17_aggregate = pd.read_csv(day17 / "aggregate_performance.csv")
    day17_cost = pd.read_csv(day17 / "cost_sensitivity.csv")
    day17_sessions = pd.read_csv(day17 / "session_return_panel.csv")
    day26_aggregate = pd.read_csv(day26 / "aggregate_performance.csv")
    day26_comparison = pd.read_csv(day26 / "comparison.csv")
    day26_inference = pd.read_csv(day26 / "inference.csv")
    day26_sessions = pd.read_csv(day26 / "session_returns.csv")
    methodology = _read_json(day26 / "methodology.json")

    _require_columns(day17_aggregate, AGGREGATE_PERFORMANCE_COLUMNS, name="Day 17 aggregate")
    _require_columns(day17_cost, COST_SENSITIVITY_COLUMNS, name="Day 17 cost")
    _require_columns(day17_sessions, RETURN_PANEL_COLUMNS, name="Day 17 sessions")
    _require_columns(day26_aggregate, PHASE2_AGGREGATE_COLUMNS, name="Day 26 aggregate")
    _require_columns(day26_comparison, PHASE2_COMPARISON_COLUMNS, name="Day 26 comparison")
    _require_columns(day26_inference, PHASE2_INFERENCE_COLUMNS, name="Day 26 inference")
    _require_columns(day26_sessions, PHASE2_SESSION_RETURN_COLUMNS, name="Day 26 sessions")

    observed_configs = tuple(day17_aggregate["configuration_id"].drop_duplicates())
    if observed_configs != CONFIGURATION_IDS:
        raise Day28OuCausalTimingError("Day 17 comparator configurations changed.")
    for configuration_id in CONFIGURATION_IDS:
        rows = day17_aggregate.loc[
            day17_aggregate["configuration_id"].eq(configuration_id)
        ]
        if tuple(rows["series"]) != REPORTED_SERIES:
            raise Day28OuCausalTimingError("Day 17 comparator series changed.")

    phase2_ou = day26_aggregate.loc[
        day26_aggregate["configuration_id"].isin(OU_CONFIGURATION_IDS)
    ]
    if tuple(phase2_ou["configuration_id"].drop_duplicates()) != OU_CONFIGURATION_IDS:
        raise Day28OuCausalTimingError("Day 26 OU comparator configurations changed.")
    ou_methodology = methodology.get("ou")
    if not isinstance(ou_methodology, dict):
        raise Day28OuCausalTimingError("Day 26 OU methodology is missing.")
    if (
        ou_methodology.get("baseline_configuration") != "ou_vwap_slow"
        or float(ou_methodology.get("expected_convergence_threshold", math.nan))
        != OU_EXPECTED_CONVERGENCE_THRESHOLD
    ):
        raise Day28OuCausalTimingError("Frozen Phase II OU signal logic changed.")

    source_hashes: dict[str, str] = {}
    for prefix, directory, filenames in (
        ("day17", day17, DAY17_COMPARATOR_FILES),
        ("day26", day26, DAY26_COMPARATOR_FILES),
    ):
        for filename in filenames:
            source_hashes[(directory / filename).resolve().as_posix()] = _sha256_file(
                directory / filename
            )
    return HistoricalComparatorEvidence(
        day17_aggregate=day17_aggregate,
        day17_cost_sensitivity=day17_cost,
        day17_session_returns=day17_sessions,
        day26_aggregate=day26_aggregate,
        day26_comparison=day26_comparison,
        day26_inference=day26_inference,
        day26_session_returns=day26_sessions,
        day26_methodology=methodology,
        source_hashes=source_hashes,
    )


def verify_comparator_snapshot(snapshot: Mapping[str, str]) -> None:
    """Prove every comparator byte string is unchanged since it was loaded."""

    for raw_path, expected_hash in snapshot.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise Day28OuCausalTimingError(f"Immutable comparator changed: {path}.")


def _nonzero_session_counts(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for configuration_id in CONFIGURATION_IDS:
        sample = frame.loc[frame["configuration_id"].eq(configuration_id)]
        for series in REPORTED_SERIES:
            records.append(
                {
                    "configuration_id": configuration_id,
                    "series": series,
                    "nonzero_net_sessions_1bp": int(
                        sample[series].astype(float).ne(0.0).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _aggregate_evidence(
    *,
    aggregate: pd.DataFrame,
    costs: pd.DataFrame,
    sessions: pd.DataFrame,
    timing: str,
) -> pd.DataFrame:
    gross = costs.loc[costs["cost_bps_per_turnover"].eq(0.0)].copy()
    net = costs.loc[costs["cost_bps_per_turnover"].eq(1.0)].copy()
    selected = net.loc[
        :,
        [
            "configuration_id",
            "series",
            "test_sessions",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "cumulative_return",
        ],
    ].rename(
        columns={
            "cumulative_return": "cumulative_net_return_1bp",
            "annualized_volatility": "annualized_volatility_1bp",
            "sharpe_ratio": "sharpe_ratio_1bp",
            "maximum_drawdown": "maximum_drawdown_1bp",
        }
    )
    selected = selected.merge(
        gross.loc[:, ["configuration_id", "series", "cumulative_return"]].rename(
            columns={"cumulative_return": "cumulative_gross_return"}
        ),
        on=["configuration_id", "series"],
        validate="one_to_one",
    )
    selected = selected.merge(
        aggregate.loc[
            :,
            [
                "configuration_id",
                "series",
                "start_session",
                "end_session",
                "turnover",
            ],
        ],
        on=["configuration_id", "series"],
        validate="one_to_one",
    )
    selected = selected.merge(
        _nonzero_session_counts(sessions),
        on=["configuration_id", "series"],
        validate="one_to_one",
    )
    selected.insert(0, "timing_convention", timing)
    selected = selected.loc[:, CORRECTED_AGGREGATE_COLUMNS]
    expected = [(configuration, series) for configuration in CONFIGURATION_IDS for series in REPORTED_SERIES]
    observed = list(selected.loc[:, ["configuration_id", "series"]].itertuples(index=False, name=None))
    if observed != expected:
        raise Day28OuCausalTimingError("Aggregate configuration/series order changed.")
    return selected


def _corrected_fold(results: ReversionInferenceResults) -> pd.DataFrame:
    frame = results.fold_performance.copy(deep=True).rename(
        columns={
            "cumulative_return": "cumulative_net_return_1bp",
            "annualized_return": "annualized_net_return_1bp",
            "annualized_volatility": "annualized_volatility_1bp",
            "sharpe_ratio": "sharpe_ratio_1bp",
            "maximum_drawdown": "maximum_drawdown_1bp",
        }
    )
    frame.insert(0, "timing_convention", CORRECTED_TIMING)
    frame.insert(4, "cost_bps_per_turnover", 1.0)
    return frame.loc[:, CORRECTED_FOLD_COLUMNS]


def _timing_comparison(
    historical: pd.DataFrame, corrected: pd.DataFrame
) -> pd.DataFrame:
    key = ["configuration_id", "series"]
    baseline = historical.loc[:, [*key, *COMPARISON_METRICS]].copy()
    pieces: list[pd.DataFrame] = []
    for source in (historical, corrected):
        frame = source.loc[:, [*key, "timing_convention", *COMPARISON_METRICS]].copy()
        merged = frame.merge(
            baseline,
            on=key,
            suffixes=("", "_historical"),
            validate="many_to_one",
        )
        for metric in COMPARISON_METRICS:
            merged[f"change_from_historical_{metric}"] = (
                merged[metric].astype(float) - merged[f"{metric}_historical"].astype(float)
            )
        pieces.append(merged.loc[:, TIMING_COMPARISON_COLUMNS])
    combined = pd.concat(pieces, ignore_index=True)
    timing_order = {HISTORICAL_TIMING: 0, CORRECTED_TIMING: 1}
    combined["_configuration_order"] = combined["configuration_id"].map(
        {value: index for index, value in enumerate(CONFIGURATION_IDS)}
    )
    combined["_series_order"] = combined["series"].map(
        {value: index for index, value in enumerate(REPORTED_SERIES)}
    )
    combined["_timing_order"] = combined["timing_convention"].map(timing_order)
    return combined.sort_values(
        ["_configuration_order", "_series_order", "_timing_order"], kind="stable"
    ).drop(columns=["_configuration_order", "_series_order", "_timing_order"]).reset_index(drop=True)


def _execution_path_differences(session_returns: pd.DataFrame) -> dict[str, int]:
    baseline_id, phase2_id = OU_CONFIGURATION_IDS
    differences: dict[str, int] = {}
    for series in OU_SERIES:
        baseline = session_returns.loc[
            session_returns["configuration_id"].eq(baseline_id)
            & session_returns["series"].eq(series)
        ].drop(columns="configuration_id")
        candidate = session_returns.loc[
            session_returns["configuration_id"].eq(phase2_id)
            & session_returns["series"].eq(series)
        ].drop(columns="configuration_id")
        merged = baseline.merge(
            candidate,
            on=["fold_id", "series", "session_date"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if len(merged) != len(baseline) or len(merged) != len(candidate):
            raise Day28OuCausalTimingError("Phase II baseline/candidate sessions differ.")
        changed = (
            ~np.isclose(
                merged["gross_return_baseline"],
                merged["gross_return_candidate"],
                rtol=0.0,
                atol=1.0e-15,
            )
            | ~np.isclose(
                merged["net_return_1bp_baseline"],
                merged["net_return_1bp_candidate"],
                rtol=0.0,
                atol=1.0e-15,
            )
            | ~np.isclose(
                merged["turnover_baseline"],
                merged["turnover_candidate"],
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        differences[series] = int(changed.sum())
    return differences


def _phase2_comparison(
    corrected: Phase2ProfitabilityResults,
    historical: HistoricalComparatorEvidence,
) -> pd.DataFrame:
    corrected_aggregate = corrected.aggregate_performance.loc[
        corrected.aggregate_performance["configuration_id"].isin(OU_CONFIGURATION_IDS)
    ]
    corrected_inference = corrected.inference.loc[
        corrected.inference["configuration_id"].isin(OU_CONFIGURATION_IDS)
    ]
    historical_aggregate = historical.day26_aggregate.loc[
        historical.day26_aggregate["configuration_id"].isin(OU_CONFIGURATION_IDS)
    ]
    historical_inference = historical.day26_inference.loc[
        historical.day26_inference["configuration_id"].isin(OU_CONFIGURATION_IDS)
    ]
    corrected_path = _execution_path_differences(corrected.session_returns)
    historical_path = _execution_path_differences(historical.day26_session_returns)
    records: list[dict[str, object]] = []
    baseline_corrected_entries: dict[str, float] = {}
    for series in OU_SERIES:
        row = corrected_aggregate.loc[
            corrected_aggregate["configuration_id"].eq(OU_CONFIGURATION_IDS[0])
            & corrected_aggregate["series"].eq(series)
        ]
        if len(row) != 1:
            raise Day28OuCausalTimingError("Corrected Phase II baseline row is missing.")
        baseline_corrected_entries[series] = float(row.iloc[0]["trade_count"])

    for configuration_id in OU_CONFIGURATION_IDS:
        for series in OU_SERIES:
            corrected_row = corrected_aggregate.loc[
                corrected_aggregate["configuration_id"].eq(configuration_id)
                & corrected_aggregate["series"].eq(series)
            ]
            historical_row = historical_aggregate.loc[
                historical_aggregate["configuration_id"].eq(configuration_id)
                & historical_aggregate["series"].eq(series)
            ]
            corrected_inf = corrected_inference.loc[
                corrected_inference["configuration_id"].eq(configuration_id)
                & corrected_inference["series"].eq(series)
            ]
            historical_inf = historical_inference.loc[
                historical_inference["configuration_id"].eq(configuration_id)
                & historical_inference["series"].eq(series)
            ]
            if any(len(value) != 1 for value in (corrected_row, historical_row, corrected_inf, historical_inf)):
                raise Day28OuCausalTimingError("Phase II OU comparator rows are incomplete.")
            current = corrected_row.iloc[0]
            old = historical_row.iloc[0]
            current_inf = corrected_inf.iloc[0]
            old_inf = historical_inf.iloc[0]
            is_baseline = configuration_id == OU_CONFIGURATION_IDS[0]
            records.append(
                {
                    "timing_convention": CORRECTED_TIMING,
                    "configuration_id": configuration_id,
                    "phase_role": "baseline" if is_baseline else "cost_margin_candidate",
                    "series": series,
                    "cost_bps_per_turnover": 1.0,
                    "historical_cumulative_net_return_1bp": old["cumulative_return"],
                    "corrected_cumulative_net_return_1bp": current["cumulative_return"],
                    "change_cumulative_net_return_1bp": float(current["cumulative_return"]) - float(old["cumulative_return"]),
                    "historical_turnover": old["turnover"],
                    "corrected_turnover": current["turnover"],
                    "change_turnover": float(current["turnover"]) - float(old["turnover"]),
                    "historical_signal_entry_count": old["trade_count"],
                    "corrected_signal_entry_count": current["trade_count"],
                    "corrected_signal_entry_count_change_vs_baseline": float(current["trade_count"]) - baseline_corrected_entries[series],
                    "historical_execution_path_difference_sessions_vs_baseline": 0 if is_baseline else historical_path[series],
                    "corrected_execution_path_difference_sessions_vs_baseline": 0 if is_baseline else corrected_path[series],
                    "historical_hac_t_statistic": old_inf["hac_t_statistic"],
                    "corrected_hac_t_statistic": current_inf["hac_t_statistic"],
                    "historical_bootstrap_mean_ci_lower": old_inf["bootstrap_mean_ci_lower"],
                    "historical_bootstrap_mean_ci_upper": old_inf["bootstrap_mean_ci_upper"],
                    "corrected_bootstrap_mean_ci_lower": current_inf["bootstrap_mean_ci_lower"],
                    "corrected_bootstrap_mean_ci_upper": current_inf["bootstrap_mean_ci_upper"],
                    "historical_bootstrap_sharpe_ci_lower": old_inf["bootstrap_sharpe_ci_lower"],
                    "historical_bootstrap_sharpe_ci_upper": old_inf["bootstrap_sharpe_ci_upper"],
                    "corrected_bootstrap_sharpe_ci_lower": current_inf["bootstrap_sharpe_ci_lower"],
                    "corrected_bootstrap_sharpe_ci_upper": current_inf["bootstrap_sharpe_ci_upper"],
                }
            )
    return pd.DataFrame.from_records(records).loc[:, PHASE2_OU_COMPARISON_COLUMNS]


def _annual_concentration(session_returns: pd.DataFrame) -> pd.DataFrame:
    sample = session_returns.loc[
        session_returns["configuration_id"].eq("ou_vwap_slow")
    ].copy(deep=True)
    sample["session_date"] = pd.to_datetime(sample["session_date"], utc=True, errors="raise")
    values = sample["equal_weight"].astype(float)
    if values.le(-1.0).any() or not np.isfinite(values).all():
        raise Day28OuCausalTimingError("Slow equal-weight session returns are inadmissible.")
    sample["year"] = sample["session_date"].dt.year
    records: list[dict[str, object]] = []
    for year, group in sample.groupby("year", sort=True):
        returns = group["equal_weight"].astype(float).reset_index(drop=True)
        metrics = calculate_performance_metrics(returns, annualization_factor=252.0)
        records.append(
            {
                "timing_convention": CORRECTED_TIMING,
                "configuration_id": "ou_vwap_slow",
                "series": "equal_weight",
                "cost_bps_per_turnover": 1.0,
                "year": int(year),
                "sessions": int(len(returns)),
                "nonzero_net_sessions_1bp": int(returns.ne(0.0).sum()),
                "cumulative_net_return_1bp": metrics.cumulative_return,
                "mean_session_return_1bp": float(returns.mean()),
                "annualized_volatility_1bp": metrics.annualized_volatility,
                "log_return_contribution": float(np.log1p(returns).sum()),
            }
        )
    frame = pd.DataFrame.from_records(records)
    absolute = frame["log_return_contribution"].abs()
    denominator = float(absolute.sum())
    frame["absolute_log_return_share"] = (
        absolute / denominator if denominator > 0.0 else 0.0
    )
    frame["absolute_log_return_hhi"] = float(
        np.square(frame["absolute_log_return_share"]).sum()
    )
    return frame.loc[:, ANNUAL_CONCENTRATION_COLUMNS]


def assemble_day28_results(
    *,
    reversion_results: ReversionInferenceResults,
    phase2_results: Phase2ProfitabilityResults,
    historical: HistoricalComparatorEvidence,
    data_audit: Mapping[str, object],
) -> Day28OuCausalTimingResults:
    """Transform accepted engines and authenticated comparators into Day 28 tables."""

    if not isinstance(reversion_results, ReversionInferenceResults):
        raise TypeError("reversion_results must be ReversionInferenceResults.")
    if not isinstance(phase2_results, Phase2ProfitabilityResults):
        raise TypeError("phase2_results must be Phase2ProfitabilityResults.")
    if not isinstance(historical, HistoricalComparatorEvidence):
        raise TypeError("historical must be HistoricalComparatorEvidence.")

    corrected_aggregate = _aggregate_evidence(
        aggregate=reversion_results.aggregate_performance,
        costs=reversion_results.cost_sensitivity,
        sessions=reversion_results.session_return_panel,
        timing=CORRECTED_TIMING,
    )
    historical_aggregate = _aggregate_evidence(
        aggregate=historical.day17_aggregate,
        costs=historical.day17_cost_sensitivity,
        sessions=historical.day17_session_returns,
        timing=HISTORICAL_TIMING,
    )
    corrected_cost = reversion_results.cost_sensitivity.copy(deep=True)
    corrected_cost.insert(0, "timing_convention", CORRECTED_TIMING)
    corrected_cost = corrected_cost.loc[:, CORRECTED_COST_COLUMNS]
    corrected_inference = reversion_results.inference_results.copy(deep=True)
    corrected_inference.insert(0, "timing_convention", CORRECTED_TIMING)
    corrected_inference = corrected_inference.loc[:, CORRECTED_INFERENCE_COLUMNS]
    phase2_comparison = _phase2_comparison(phase2_results, historical)
    annual = _annual_concentration(reversion_results.session_return_panel)

    if tuple(corrected_aggregate["configuration_id"].drop_duplicates()) != CONFIGURATION_IDS:
        raise Day28OuCausalTimingError("Not all frozen OU configurations were retained.")
    forbidden_tokens = ("winner", "rank", "promotion")
    tables = (
        _corrected_fold(reversion_results),
        corrected_aggregate,
        corrected_cost,
        corrected_inference,
        _timing_comparison(historical_aggregate, corrected_aggregate),
        phase2_comparison,
        annual,
    )
    if any(
        token in column.lower()
        for table in tables
        for column in table.columns
        for token in forbidden_tokens
    ):
        raise Day28OuCausalTimingError("Selection fields are prohibited.")

    folds = build_walk_forward_folds()
    manifest = {
        "schema_version": SPECIFICATION_VERSION,
        "timing_version": CORRECTED_TIMING,
        "historical_timing_label": HISTORICAL_TIMING,
        "claim_boundary": "development_evidence_only_no_selection_or_promotion",
        "data": dict(data_audit),
        "configuration_ids": list(CONFIGURATION_IDS),
        "reported_series": list(REPORTED_SERIES),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start.date().isoformat(),
                "train_end_exclusive": fold.train_end_exclusive.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end_exclusive": fold.test_end_exclusive.date().isoformat(),
            }
            for fold in folds
        ],
        "cost_stress_bps": list(COST_STRESS_BPS),
        "inference": {
            "hac_lags": HAC_LAGS,
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "declared_dsr_trials": len(CONFIGURATION_IDS),
        },
        "phase2_inference": {
            "hac_lags": PHASE2_HAC_LAGS,
            "bootstrap_replications": PHASE2_BOOTSTRAP_REPLICATIONS,
            "bootstrap_block_length": PHASE2_BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": PHASE2_BOOTSTRAP_SEED,
        },
        "phase2_ou_configuration_ids": list(OU_CONFIGURATION_IDS),
        "phase2_expected_convergence_threshold": OU_EXPECTED_CONVERGENCE_THRESHOLD,
        "comparator_sources": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(historical.source_hashes.items())
        ],
        "locked_2026_accessed": False,
        "ranking_performed": False,
        "winner_selected": False,
        "promotion_performed": False,
        "report_generated": False,
        "broker_accessed": False,
    }
    return Day28OuCausalTimingResults(
        corrected_fold_performance=tables[0],
        corrected_aggregate_performance=tables[1],
        corrected_cost_sensitivity=tables[2],
        corrected_inference_results=tables[3],
        historical_vs_corrected_timing=tables[4],
        corrected_phase2_ou_comparison=tables[5],
        annual_concentration_diagnostics=tables[6],
        manifest=manifest,
        comparator_snapshot=historical.source_hashes,
    )


def run_day28_ou_causal_timing(
    bars: pd.DataFrame,
    *,
    source_dataset_path: str,
    source_sha256: str,
    day17_comparator_directory: str | Path,
    day26_comparator_directory: str | Path,
) -> Day28OuCausalTimingResults:
    """Run the two accepted corrected-timing engines once on authorized data."""

    audit = audit_day28_development_input(
        bars,
        source_dataset_path=source_dataset_path,
        source_sha256=source_sha256,
    )
    historical = load_historical_comparators(
        day17_directory=day17_comparator_directory,
        day26_directory=day26_comparator_directory,
    )
    reversion = run_reversion_inference(bars)
    phase2 = run_phase2_profitability(
        bars,
        source_dataset_id=Path(source_dataset_path).name,
        source_sha256=source_sha256,
    )
    verify_comparator_snapshot(historical.source_hashes)
    return assemble_day28_results(
        reversion_results=reversion,
        phase2_results=phase2,
        historical=historical,
        data_audit=audit,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n", float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_day28_artifacts(
    results: Day28OuCausalTimingResults,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Create one isolated deterministic Day 28 bundle without overwrite."""

    if not isinstance(results, Day28OuCausalTimingResults):
        raise TypeError("results must be Day28OuCausalTimingResults.")
    output = Path(directory)
    if output.name != OUTPUT_DIRECTORY_BASENAME:
        raise Day28OuCausalTimingError(
            f"Artifact directory basename must be {OUTPUT_DIRECTORY_BASENAME!r}."
        )
    if output.exists():
        raise FileExistsError(f"Day 28 artifact directory already exists: {output}.")
    output_resolved = output.resolve()
    for comparator_path in results.comparator_snapshot:
        comparator_resolved = Path(comparator_path).resolve()
        if output_resolved == comparator_resolved or output_resolved in comparator_resolved.parents:
            raise Day28OuCausalTimingError("Day 28 output overlaps a comparator path.")

    verify_comparator_snapshot(results.comparator_snapshot)
    tables = {
        CORRECTED_FOLD_FILENAME: results.corrected_fold_performance,
        CORRECTED_AGGREGATE_FILENAME: results.corrected_aggregate_performance,
        CORRECTED_COST_FILENAME: results.corrected_cost_sensitivity,
        CORRECTED_INFERENCE_FILENAME: results.corrected_inference_results,
        TIMING_COMPARISON_FILENAME: results.historical_vs_corrected_timing,
        PHASE2_COMPARISON_FILENAME: results.corrected_phase2_ou_comparison,
        ANNUAL_CONCENTRATION_FILENAME: results.annual_concentration_diagnostics,
    }
    payloads = {name: _csv_bytes(frame) for name, frame in tables.items()}
    manifest = dict(results.manifest)
    manifest["artifacts"] = [
        {
            "filename": filename,
            "rows": int(len(tables[filename])),
            "bytes": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in CSV_ARTIFACT_NAMES
    ]
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    if tuple(payloads) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Day 28 artifact allow-list changed.")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for filename in APPROVED_ARTIFACT_NAMES:
            (stage / filename).write_bytes(payloads[filename])
        os.replace(stage, output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verify_comparator_snapshot(results.comparator_snapshot)
    observed = tuple(sorted(path.name for path in output.iterdir()))
    if observed != tuple(sorted(APPROVED_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 28 artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_ARTIFACT_NAMES)
