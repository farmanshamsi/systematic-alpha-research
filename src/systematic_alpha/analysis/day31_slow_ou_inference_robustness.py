"""Development-only inference robustness for the corrected slow OU path.

Day 31 holds the Day 28 slow OU strategy and its causal execution path fixed.
Only transaction-cost, HAC-lag, circular-block-length, and leave-one-year-out
inference sensitivities are evaluated.  Nothing in this module ranks, tunes, or
promotes a strategy.
"""

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
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from systematic_alpha.analysis.day28_ou_causal_timing import (
    APPROVED_ARTIFACT_NAMES as DAY28_APPROVED_ARTIFACT_NAMES,
    CORRECTED_AGGREGATE_COLUMNS,
    CORRECTED_COST_COLUMNS,
    CORRECTED_INFERENCE_COLUMNS,
    CORRECTED_TIMING,
    SPECIFICATION_VERSION as DAY28_SPECIFICATION_VERSION,
    audit_day28_development_input,
)
from systematic_alpha.analysis.reversion_inference import (
    ANNUALIZATION_FACTOR,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIGURATIONS,
    CONFIGURATION_IDS,
    HAC_LAGS,
    REQUIRED_SYMBOLS,
    ReversionInferenceResults,
    _apply_ou_performance_timing,
    _deflated_benchmark,
    _sharpe_probability,
    _validate_and_prepare_bars,
    run_reversion_inference,
)
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds
from systematic_alpha.strategies.ou_vwap_reversion import build_ou_vwap_reversion_strategy


EXPERIMENT_VERSION: Final[str] = "day31_slow_ou_inference_robustness_v1"
DAY28_TIMING_VERSION: Final[str] = CORRECTED_TIMING
PRIMARY_CONFIGURATION_ID: Final[str] = "ou_vwap_slow"
PRIMARY_SERIES: Final[str] = "equal_weight"
PRIMARY_COST_BPS: Final[float] = 1.0
PRIMARY_HAC_LAG: Final[int] = HAC_LAGS
PRIMARY_BOOTSTRAP_BLOCK_LENGTH: Final[int] = 5
PRIMARY_BOOTSTRAP_SEED: Final[int] = (
    BOOTSTRAP_SEED
    + CONFIGURATION_IDS.index(PRIMARY_CONFIGURATION_ID) * 4
    + 3
)
TRANSACTION_COST_BPS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 5.0)
HAC_LAGS_SENSITIVITY: Final[tuple[int, ...]] = (1, 5, 10, 20)
BOOTSTRAP_BLOCK_LENGTHS: Final[tuple[int, ...]] = (5, 10, 20, 40)
LEAVE_ONE_YEAR_OUT_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024, 2025)
DECLARED_DSR_TRIALS: Final[int] = 3
REPRODUCTION_TOLERANCE: Final[float] = 5.0e-10
PATH_TOLERANCE: Final[float] = 5.0e-15
OUTPUT_DIRECTORY_BASENAME: Final[str] = "day31_slow_ou_inference_robustness"

SOURCE_METADATA_FILENAME: Final[str] = "source_and_method_metadata.json"
PRIMARY_REPRODUCTION_FILENAME: Final[str] = "primary_day28_reproduction.csv"
COST_SENSITIVITY_FILENAME: Final[str] = "transaction_cost_sensitivity.csv"
HAC_SENSITIVITY_FILENAME: Final[str] = "hac_lag_sensitivity.csv"
BOOTSTRAP_SENSITIVITY_FILENAME: Final[str] = "block_bootstrap_sensitivity.csv"
LEAVE_ONE_YEAR_OUT_FILENAME: Final[str] = "leave_one_year_out.csv"
PSR_DSR_FILENAME: Final[str] = "psr_dsr_disclosure.csv"
MANIFEST_FILENAME: Final[str] = "manifest.json"

CSV_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    PRIMARY_REPRODUCTION_FILENAME,
    COST_SENSITIVITY_FILENAME,
    HAC_SENSITIVITY_FILENAME,
    BOOTSTRAP_SENSITIVITY_FILENAME,
    LEAVE_ONE_YEAR_OUT_FILENAME,
    PSR_DSR_FILENAME,
)
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SOURCE_METADATA_FILENAME,
    *CSV_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
)

BAR_EXECUTION_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "timestamp",
    "session_date",
    "symbol",
    "gross_return",
    "turnover",
)
SESSION_PATH_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "session_date",
    "gross_return",
    "turnover",
    "net_return_1bp",
)
PRIMARY_REPRODUCTION_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "timing_version",
    "cost_bps_per_turnover",
    "metric",
    "day28_saved_value",
    "day31_recomputed_value",
    "absolute_difference",
    "tolerance",
    "within_tolerance",
)
COST_SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "timing_version",
    "cost_bps_per_turnover",
    "observations",
    "cumulative_net_return",
    "annualized_volatility",
    "annualized_sharpe_ratio",
    "maximum_drawdown",
    "total_turnover",
    "total_cost",
)
HAC_SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "hac_lag",
    "observations",
    "mean_session_return",
    "long_run_variance",
    "hac_standard_error",
    "hac_t_statistic",
    "is_primary_lag",
)
BOOTSTRAP_SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "block_length",
    "replications",
    "seed",
    "mean_ci_lower",
    "mean_ci_upper",
    "sharpe_ci_lower",
    "sharpe_ci_upper",
    "mean_interval_includes_zero",
    "sharpe_interval_includes_zero",
    "is_primary_block_length",
)
LEAVE_ONE_YEAR_OUT_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "excluded_year",
    "included_years",
    "observations",
    "start_session",
    "end_session",
    "cumulative_net_return",
    "annualized_volatility",
    "annualized_sharpe_ratio",
    "maximum_drawdown",
    "hac_lag",
    "hac_t_statistic",
)
PSR_DSR_DISCLOSURE_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_version",
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "observations",
    "annualized_sharpe_ratio",
    "sample_skewness",
    "sample_kurtosis",
    "psr_benchmark_per_period",
    "probabilistic_sharpe_probability",
    "deflated_sharpe_benchmark_per_period",
    "deflated_sharpe_probability",
    "declared_trials",
    "trial_scope",
    "excluded_research_families",
    "globally_corrected_dsr_claimed",
    "multiplicity_disclosure",
)


class Day31SlowOuRobustnessError(ValueError):
    """Raised when the frozen Day 31 robustness contract fails closed."""


def _copy_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class Day28Evidence:
    """Authenticated immutable Day 28 comparator evidence."""

    aggregate: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    inference: pd.DataFrame
    manifest: Mapping[str, object]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate", _copy_frame(self.aggregate, name="aggregate"))
        object.__setattr__(
            self,
            "cost_sensitivity",
            _copy_frame(self.cost_sensitivity, name="cost_sensitivity"),
        )
        object.__setattr__(self, "inference", _copy_frame(self.inference, name="inference"))
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(self, "source_hashes", dict(self.source_hashes))


@dataclass(frozen=True, slots=True)
class FrozenSlowOuPath:
    """Defensive bar and session paths for the frozen slow OU execution."""

    bar_execution_path: pd.DataFrame
    session_return_path: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bar_execution_path",
            _copy_frame(self.bar_execution_path, name="bar_execution_path"),
        )
        object.__setattr__(
            self,
            "session_return_path",
            _copy_frame(self.session_return_path, name="session_return_path"),
        )

    def copy_bar_execution_path(self) -> pd.DataFrame:
        return self.bar_execution_path.copy(deep=True)

    def copy_session_return_path(self) -> pd.DataFrame:
        return self.session_return_path.copy(deep=True)


@dataclass(frozen=True, slots=True)
class Day31SlowOuRobustnessResults:
    """All deterministic Day 31 tables and authenticated metadata."""

    source_and_method_metadata: Mapping[str, object]
    primary_day28_reproduction: pd.DataFrame
    transaction_cost_sensitivity: pd.DataFrame
    hac_lag_sensitivity: pd.DataFrame
    block_bootstrap_sensitivity: pd.DataFrame
    leave_one_year_out: pd.DataFrame
    psr_dsr_disclosure: pd.DataFrame
    comparator_snapshot: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "primary_day28_reproduction",
            "transaction_cost_sensitivity",
            "hac_lag_sensitivity",
            "block_bootstrap_sensitivity",
            "leave_one_year_out",
            "psr_dsr_disclosure",
        ):
            object.__setattr__(self, name, _copy_frame(getattr(self, name), name=name))
        object.__setattr__(
            self, "source_and_method_metadata", dict(self.source_and_method_metadata)
        )
        object.__setattr__(self, "comparator_snapshot", dict(self.comparator_snapshot))


def sha256_file(path: Path) -> str:
    """Hash one existing file without changing it."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Day31SlowOuRobustnessError(f"Unreadable JSON evidence: {path}.") from exc
    if not isinstance(value, dict):
        raise Day31SlowOuRobustnessError(f"JSON evidence must be an object: {path}.")
    return value


def load_day28_evidence(directory: Path) -> Day28Evidence:
    """Authenticate every Day 28 artifact before reading comparator tables."""

    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Day 28 evidence directory is missing: {directory}.")
    missing = [name for name in DAY28_APPROVED_ARTIFACT_NAMES if not (directory / name).is_file()]
    if missing:
        raise Day31SlowOuRobustnessError(f"Day 28 evidence is incomplete: {missing}.")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != DAY28_SPECIFICATION_VERSION:
        raise Day31SlowOuRobustnessError("Day 28 schema version changed.")
    if manifest.get("timing_version") != DAY28_TIMING_VERSION:
        raise Day31SlowOuRobustnessError("Day 28 timing version changed.")
    inference = manifest.get("inference")
    expected_inference = {
        "hac_lags": PRIMARY_HAC_LAG,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_block_length": PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "declared_dsr_trials": DECLARED_DSR_TRIALS,
    }
    if inference != expected_inference:
        raise Day31SlowOuRobustnessError("Day 28 inference contract changed.")
    declared_rows = manifest.get("artifacts")
    if not isinstance(declared_rows, list):
        raise Day31SlowOuRobustnessError("Day 28 artifact manifest is malformed.")
    declared = {
        str(row.get("filename")): str(row.get("sha256"))
        for row in declared_rows
        if isinstance(row, dict)
    }
    for filename in DAY28_APPROVED_ARTIFACT_NAMES:
        if filename == "manifest.json":
            continue
        observed = sha256_file(directory / filename)
        if declared.get(filename) != observed:
            raise Day31SlowOuRobustnessError(
                f"Day 28 manifest hash mismatch for {filename}."
            )
    aggregate = pd.read_csv(directory / "corrected_aggregate_performance.csv")
    costs = pd.read_csv(directory / "corrected_cost_sensitivity.csv")
    inference_table = pd.read_csv(directory / "corrected_inference_results.csv")
    if tuple(aggregate.columns) != CORRECTED_AGGREGATE_COLUMNS:
        raise Day31SlowOuRobustnessError("Day 28 aggregate schema changed.")
    if tuple(costs.columns) != CORRECTED_COST_COLUMNS:
        raise Day31SlowOuRobustnessError("Day 28 cost schema changed.")
    if tuple(inference_table.columns) != CORRECTED_INFERENCE_COLUMNS:
        raise Day31SlowOuRobustnessError("Day 28 inference schema changed.")
    snapshot = {
        (directory / filename).resolve().as_posix(): sha256_file(directory / filename)
        for filename in DAY28_APPROVED_ARTIFACT_NAMES
    }
    return Day28Evidence(
        aggregate=aggregate,
        cost_sensitivity=costs,
        inference=inference_table,
        manifest=manifest,
        source_hashes=dict(sorted(snapshot.items())),
    )


def verify_comparator_snapshot(snapshot: Mapping[str, str]) -> None:
    """Require every Day 28 comparator to remain byte-identical in place."""

    if not isinstance(snapshot, Mapping) or not snapshot:
        raise Day31SlowOuRobustnessError("Comparator snapshot must be non-empty.")
    for raw_path, expected_hash in sorted(snapshot.items()):
        if sha256_file(Path(raw_path)) != expected_hash:
            raise Day31SlowOuRobustnessError(
                f"Immutable Day 28 comparator changed: {raw_path}."
            )


def _finite_returns(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype="float64").copy()
    if array.ndim != 1 or len(array) < 2:
        raise Day31SlowOuRobustnessError("Return input requires at least two values.")
    if not np.isfinite(array).all() or np.less_equal(array, -1.0).any():
        raise Day31SlowOuRobustnessError(
            "Returns must be finite and strictly greater than -1."
        )
    return array


def apply_transaction_costs(
    gross_returns: Sequence[float] | np.ndarray | pd.Series,
    turnover: Sequence[float] | np.ndarray | pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> np.ndarray:
    """Deduct actual row turnover, never a constant daily fee."""

    gross = _finite_returns(gross_returns)
    turns = np.asarray(turnover, dtype="float64").copy()
    if turns.shape != gross.shape or not np.isfinite(turns).all() or (turns < 0.0).any():
        raise Day31SlowOuRobustnessError("Turnover must be finite, nonnegative, and aligned.")
    if not math.isfinite(float(cost_bps_per_turnover)) or float(cost_bps_per_turnover) < 0.0:
        raise Day31SlowOuRobustnessError("Transaction cost must be finite and nonnegative.")
    net = gross - turns * float(cost_bps_per_turnover) / 10_000.0
    if not np.isfinite(net).all() or np.less_equal(net, -1.0).any():
        raise Day31SlowOuRobustnessError("Cost-adjusted returns are inadmissible.")
    return net


def _validate_bar_execution_path(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("bar_execution_path must be a pandas DataFrame.")
    if tuple(frame.columns) != BAR_EXECUTION_COLUMNS or frame.empty:
        raise Day31SlowOuRobustnessError("Bar execution-path schema is invalid.")
    result = frame.copy(deep=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["session_date"] = pd.to_datetime(
        result["session_date"], utc=True, errors="raise"
    ).dt.normalize()
    result["symbol"] = result["symbol"].astype(str)
    if tuple(sorted(result["symbol"].unique())) != tuple(sorted(REQUIRED_SYMBOLS)):
        raise Day31SlowOuRobustnessError("Bar path must contain the frozen universe.")
    if result.duplicated(["symbol", "timestamp"]).any():
        raise Day31SlowOuRobustnessError("Bar path contains duplicate observations.")
    ordered = result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if not result.reset_index(drop=True).equals(ordered):
        raise Day31SlowOuRobustnessError("Bar path must be deterministically ordered.")
    if result["timestamp"].ge(pd.Timestamp("2026-01-01", tz="UTC")).any():
        raise Day31SlowOuRobustnessError("Bar path contains prohibited 2026 data.")
    gross = pd.to_numeric(result["gross_return"], errors="raise").to_numpy(dtype="float64")
    turns = pd.to_numeric(result["turnover"], errors="raise").to_numpy(dtype="float64")
    _finite_returns(gross)
    if not np.isfinite(turns).all() or (turns < 0.0).any():
        raise Day31SlowOuRobustnessError("Bar turnover is inadmissible.")
    result["gross_return"] = gross
    result["turnover"] = turns
    return result


def aggregate_equal_weight_execution_path(
    bar_execution_path: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Apply bar costs, compound within symbol/session, then equal weight."""

    bars = _validate_bar_execution_path(bar_execution_path)
    bars["net_return"] = apply_transaction_costs(
        bars["gross_return"],
        bars["turnover"],
        cost_bps_per_turnover=cost_bps_per_turnover,
    )
    records: list[dict[str, object]] = []
    for (fold_id, session_date, symbol), group in bars.groupby(
        ["fold_id", "session_date", "symbol"], observed=True, sort=True
    ):
        records.append(
            {
                "fold_id": fold_id,
                "session_date": session_date,
                "symbol": symbol,
                "gross_return": float(np.prod(1.0 + group["gross_return"]) - 1.0),
                "net_return": float(np.prod(1.0 + group["net_return"]) - 1.0),
                "turnover": float(group["turnover"].sum()),
            }
        )
    symbol_sessions = pd.DataFrame.from_records(records)
    counts = symbol_sessions.groupby(["fold_id", "session_date"], observed=True)[
        "symbol"
    ].nunique()
    if not counts.eq(len(REQUIRED_SYMBOLS)).all():
        raise Day31SlowOuRobustnessError("Session path is incomplete across symbols.")
    session = (
        symbol_sessions.groupby(["fold_id", "session_date"], observed=True, sort=True)[
            ["gross_return", "net_return", "turnover"]
        ]
        .mean()
        .reset_index()
    )
    return session.sort_values("session_date", kind="stable").reset_index(drop=True)


def build_frozen_slow_ou_path(bars: pd.DataFrame) -> FrozenSlowOuPath:
    """Reconstruct only the frozen slow execution path with Day 28 code."""

    source = bars.copy(deep=True)
    features = _validate_and_prepare_bars(source)
    parameters = next(
        item for item in CONFIGURATIONS if item.configuration_id == PRIMARY_CONFIGURATION_ID
    )
    feature_sessions = pd.to_datetime(features["session_date"], utc=True).dt.normalize()
    pieces: list[pd.DataFrame] = []
    for fold in build_walk_forward_folds():
        history = features.loc[feature_sessions.lt(fold.test_end_exclusive)].copy(deep=True)
        test_source = features.loc[
            feature_sessions.ge(fold.test_start)
            & feature_sessions.lt(fold.test_end_exclusive)
        ]
        reset_timestamps = tuple(
            pd.Timestamp(value)
            for value in test_source.groupby("symbol", observed=True)["timestamp"].min()
        )
        bundle = build_ou_vwap_reversion_strategy(
            history.reset_index(drop=True),
            parameters=parameters,
            execution_reset_timestamps=reset_timestamps,
        )
        result_sessions = pd.to_datetime(
            bundle.observations["session_date"], utc=True
        ).dt.normalize()
        test = bundle.observations.loc[
            result_sessions.ge(fold.test_start)
            & result_sessions.lt(fold.test_end_exclusive)
        ].copy(deep=True)
        timed = _apply_ou_performance_timing(test, fold_id=fold.fold_id)
        piece = timed.loc[
            :, ["timestamp", "session_date", "symbol", "gross_strategy_return", "turnover"]
        ].rename(columns={"gross_strategy_return": "gross_return"})
        piece.insert(0, "fold_id", fold.fold_id)
        pieces.append(piece.loc[:, BAR_EXECUTION_COLUMNS])
    bar_path = pd.concat(pieces, ignore_index=True).sort_values(
        ["timestamp", "symbol"], kind="stable"
    ).reset_index(drop=True)
    primary = aggregate_equal_weight_execution_path(
        bar_path, cost_bps_per_turnover=PRIMARY_COST_BPS
    ).rename(columns={"net_return": "net_return_1bp"})
    return FrozenSlowOuPath(
        bar_execution_path=bar_path,
        session_return_path=primary.loc[:, SESSION_PATH_COLUMNS],
    )


def calculate_hac_statistics(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    lag: int,
) -> dict[str, float | int]:
    """Calculate Bartlett/Newey-West mean inference with denominator T."""

    returns = _finite_returns(values)
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
        raise TypeError("lag must be an integer.")
    normalized_lag = int(lag)
    if normalized_lag < 0 or normalized_lag >= len(returns):
        raise Day31SlowOuRobustnessError("HAC lag must be between zero and T-1.")
    observations = len(returns)
    mean = float(np.mean(returns))
    demeaned = returns - mean
    autocovariances = [
        float(np.dot(demeaned[offset:], demeaned[: observations - offset]) / observations)
        for offset in range(normalized_lag + 1)
    ]
    long_run_variance = autocovariances[0]
    for offset in range(1, normalized_lag + 1):
        weight = 1.0 - offset / float(normalized_lag + 1)
        long_run_variance += 2.0 * weight * autocovariances[offset]
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        raise Day31SlowOuRobustnessError("HAC long-run variance must be positive.")
    standard_error = float(math.sqrt(long_run_variance / observations))
    return {
        "observations": observations,
        "mean_session_return": mean,
        "long_run_variance": float(long_run_variance),
        "hac_standard_error": standard_error,
        "hac_t_statistic": float(mean / standard_error),
    }


def circular_block_indices(
    *,
    observations: int,
    block_length: int,
    starts: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Construct one deterministic circular-block sample index vector."""

    if observations < 1 or block_length < 1:
        raise Day31SlowOuRobustnessError("Bootstrap dimensions must be positive.")
    start_array = np.asarray(starts, dtype="int64").copy()
    blocks_required = int(math.ceil(observations / block_length))
    if start_array.ndim != 1 or len(start_array) != blocks_required:
        raise Day31SlowOuRobustnessError("Bootstrap starts do not cover the sample.")
    if (start_array < 0).any() or (start_array >= observations).any():
        raise Day31SlowOuRobustnessError("Bootstrap starts are out of bounds.")
    offsets = np.arange(block_length, dtype="int64")
    return ((start_array[:, None] + offsets[None, :]) % observations).reshape(-1)[
        :observations
    ]


def _annualized_sharpe(values: np.ndarray) -> float:
    deviation = float(np.std(values, ddof=1))
    return (
        float(np.mean(values) / deviation * math.sqrt(ANNUALIZATION_FACTOR))
        if deviation > 0.0
        else float("nan")
    )


def circular_block_bootstrap_intervals(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    block_length: int,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = PRIMARY_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Return deterministic percentile intervals using the Day 28 convention."""

    returns = _finite_returns(values)
    if block_length < 1 or block_length > len(returns):
        raise Day31SlowOuRobustnessError("Block length must be between one and T.")
    if replications < 1:
        raise Day31SlowOuRobustnessError("Bootstrap replications must be positive.")
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(returns) / block_length))
    means = np.empty(replications, dtype="float64")
    sharpes = np.empty(replications, dtype="float64")
    for replication in range(replications):
        starts = rng.integers(0, len(returns), size=blocks)
        indices = circular_block_indices(
            observations=len(returns), block_length=block_length, starts=starts
        )
        sample = returns[indices]
        means[replication] = float(np.mean(sample))
        sharpes[replication] = _annualized_sharpe(sample)
    finite_sharpes = sharpes[np.isfinite(sharpes)]
    if len(finite_sharpes) == 0:
        raise Day31SlowOuRobustnessError("Bootstrap Sharpe samples are all nonfinite.")
    mean_bounds = np.quantile(means, [0.025, 0.975])
    sharpe_bounds = np.quantile(finite_sharpes, [0.025, 0.975])
    return {
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
        "mean_ci_lower": float(mean_bounds[0]),
        "mean_ci_upper": float(mean_bounds[1]),
        "sharpe_ci_lower": float(sharpe_bounds[0]),
        "sharpe_ci_upper": float(sharpe_bounds[1]),
    }


def calculate_leave_one_year_out(
    session_returns: pd.Series,
) -> pd.DataFrame:
    """Remove each complete predeclared year without reordering observations."""

    if not isinstance(session_returns, pd.Series):
        raise TypeError("session_returns must be a pandas Series.")
    if not isinstance(session_returns.index, pd.DatetimeIndex) or session_returns.index.tz is None:
        raise Day31SlowOuRobustnessError("Session returns require a timezone-aware index.")
    if not session_returns.index.is_monotonic_increasing or session_returns.index.has_duplicates:
        raise Day31SlowOuRobustnessError("Session returns must be unique and chronological.")
    values = _finite_returns(session_returns)
    retained = pd.Series(values, index=session_returns.index.copy(), dtype="float64")
    observed_years = tuple(sorted(retained.index.year.unique().tolist()))
    if observed_years != LEAVE_ONE_YEAR_OUT_YEARS:
        raise Day31SlowOuRobustnessError("Primary path must contain exactly 2022-2025.")
    records: list[dict[str, object]] = []
    for excluded_year in LEAVE_ONE_YEAR_OUT_YEARS:
        sample = retained.loc[retained.index.year != excluded_year]
        if (sample.index.year == excluded_year).any():
            raise RuntimeError("Leave-one-year-out membership failed.")
        metrics = calculate_performance_metrics(
            sample.reset_index(drop=True), annualization_factor=ANNUALIZATION_FACTOR
        )
        hac = calculate_hac_statistics(sample, lag=PRIMARY_HAC_LAG)
        records.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "cost_bps_per_turnover": PRIMARY_COST_BPS,
                "excluded_year": excluded_year,
                "included_years": "|".join(
                    str(year) for year in observed_years if year != excluded_year
                ),
                "observations": int(len(sample)),
                "start_session": sample.index.min().date().isoformat(),
                "end_session": sample.index.max().date().isoformat(),
                "cumulative_net_return": metrics.cumulative_return,
                "annualized_volatility": metrics.annualized_volatility,
                "annualized_sharpe_ratio": metrics.sharpe_ratio,
                "maximum_drawdown": metrics.max_drawdown,
                "hac_lag": PRIMARY_HAC_LAG,
                "hac_t_statistic": hac["hac_t_statistic"],
            }
        )
    return pd.DataFrame.from_records(records).loc[:, LEAVE_ONE_YEAR_OUT_COLUMNS]


def _single_row(frame: pd.DataFrame, mask: pd.Series, *, label: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise Day31SlowOuRobustnessError(f"Expected one {label} row; found {len(selected)}.")
    return selected.iloc[0]


def _metric_reproduction_rows(
    pairs: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric, (saved, recomputed) in pairs.items():
        difference = abs(float(recomputed) - float(saved))
        tolerance = 0.0 if metric in {"observations", "declared_trials", "turnover"} else REPRODUCTION_TOLERANCE
        within = difference <= tolerance + tolerance * abs(float(saved))
        records.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "timing_version": DAY28_TIMING_VERSION,
                "cost_bps_per_turnover": PRIMARY_COST_BPS,
                "metric": metric,
                "day28_saved_value": float(saved),
                "day31_recomputed_value": float(recomputed),
                "absolute_difference": difference,
                "tolerance": tolerance,
                "within_tolerance": within,
            }
        )
    result = pd.DataFrame.from_records(records).loc[:, PRIMARY_REPRODUCTION_COLUMNS]
    if not result["within_tolerance"].all():
        failed = result.loc[~result["within_tolerance"], "metric"].tolist()
        raise Day31SlowOuRobustnessError(
            f"Day 28 primary reproduction failed for: {failed}."
        )
    return result


def build_day31_results(
    *,
    frozen_path: FrozenSlowOuPath,
    reversion_results: ReversionInferenceResults,
    day28_evidence: Day28Evidence,
    data_audit: Mapping[str, object],
    source_dataset_path: str,
    source_sha256: str,
    generation_timestamp: str,
) -> Day31SlowOuRobustnessResults:
    """Authenticate the primary path, then calculate predeclared sensitivities."""

    if not isinstance(frozen_path, FrozenSlowOuPath):
        raise TypeError("frozen_path must be a FrozenSlowOuPath.")
    if not isinstance(reversion_results, ReversionInferenceResults):
        raise TypeError("reversion_results must be ReversionInferenceResults.")
    if not isinstance(day28_evidence, Day28Evidence):
        raise TypeError("day28_evidence must be Day28Evidence.")
    verify_comparator_snapshot(day28_evidence.source_hashes)
    bar_path = _validate_bar_execution_path(frozen_path.copy_bar_execution_path())
    session = frozen_path.copy_session_return_path()
    if tuple(session.columns) != SESSION_PATH_COLUMNS:
        raise Day31SlowOuRobustnessError("Frozen session-path schema changed.")
    session["session_date"] = pd.to_datetime(session["session_date"], utc=True, errors="raise")
    session = session.sort_values("session_date", kind="stable").reset_index(drop=True)
    if session["session_date"].duplicated().any() or session["session_date"].ge(
        pd.Timestamp("2026-01-01", tz="UTC")
    ).any():
        raise Day31SlowOuRobustnessError("Frozen session chronology is invalid.")

    engine_panel = reversion_results.session_return_panel.loc[
        reversion_results.session_return_panel["configuration_id"].eq(
            PRIMARY_CONFIGURATION_ID
        ),
        ["fold_id", "session_date", PRIMARY_SERIES],
    ].copy(deep=True)
    engine_panel["session_date"] = pd.to_datetime(
        engine_panel["session_date"], utc=True, errors="raise"
    )
    engine_panel = engine_panel.sort_values("session_date", kind="stable").reset_index(drop=True)
    if not session[["fold_id", "session_date"]].equals(
        engine_panel[["fold_id", "session_date"]]
    ):
        raise Day31SlowOuRobustnessError("Reconstructed and Day 28 session identities differ.")
    if not np.allclose(
        session["net_return_1bp"].to_numpy(dtype="float64"),
        engine_panel[PRIMARY_SERIES].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=PATH_TOLERANCE,
    ):
        raise Day31SlowOuRobustnessError("Reconstructed Day 28 primary path differs.")

    primary = pd.Series(
        engine_panel[PRIMARY_SERIES].to_numpy(dtype="float64", copy=True),
        index=pd.DatetimeIndex(engine_panel["session_date"], name="session_date"),
    )
    _finite_returns(primary)
    cost_records: list[dict[str, object]] = []
    cost_paths: dict[float, pd.DataFrame] = {}
    total_turnover = float(bar_path["turnover"].sum() / len(REQUIRED_SYMBOLS))
    for cost in TRANSACTION_COST_BPS:
        cost_path = aggregate_equal_weight_execution_path(
            bar_path, cost_bps_per_turnover=cost
        )
        cost_paths[cost] = cost_path
        metrics = calculate_performance_metrics(
            cost_path["net_return"], annualization_factor=ANNUALIZATION_FACTOR
        )
        cost_records.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "timing_version": DAY28_TIMING_VERSION,
                "cost_bps_per_turnover": cost,
                "observations": int(len(cost_path)),
                "cumulative_net_return": metrics.cumulative_return,
                "annualized_volatility": metrics.annualized_volatility,
                "annualized_sharpe_ratio": metrics.sharpe_ratio,
                "maximum_drawdown": metrics.max_drawdown,
                "total_turnover": total_turnover,
                "total_cost": total_turnover * cost / 10_000.0,
            }
        )
    cost_table = pd.DataFrame.from_records(cost_records).loc[:, COST_SENSITIVITY_COLUMNS]
    for row in cost_table.itertuples(index=False):
        saved_cost = _single_row(
            day28_evidence.cost_sensitivity,
            day28_evidence.cost_sensitivity["configuration_id"].eq(
                PRIMARY_CONFIGURATION_ID
            )
            & day28_evidence.cost_sensitivity["series"].eq(PRIMARY_SERIES)
            & day28_evidence.cost_sensitivity["cost_bps_per_turnover"].eq(
                row.cost_bps_per_turnover
            ),
            label=f"saved {row.cost_bps_per_turnover:g}bp cost sensitivity",
        )
        comparisons = (
            (row.observations, saved_cost["test_sessions"]),
            (row.cumulative_net_return, saved_cost["cumulative_return"]),
            (row.annualized_volatility, saved_cost["annualized_volatility"]),
            (row.annualized_sharpe_ratio, saved_cost["sharpe_ratio"]),
            (row.maximum_drawdown, saved_cost["maximum_drawdown"]),
        )
        if not all(
            math.isclose(
                float(observed),
                float(saved),
                rel_tol=REPRODUCTION_TOLERANCE,
                abs_tol=REPRODUCTION_TOLERANCE,
            )
            for observed, saved in comparisons
        ):
            raise Day31SlowOuRobustnessError(
                f"Day 28 {row.cost_bps_per_turnover:g}bp cost row was not reproduced."
            )
    primary_metrics = calculate_performance_metrics(
        primary.reset_index(drop=True), annualization_factor=ANNUALIZATION_FACTOR
    )
    primary_deviation = float(np.std(primary.to_numpy(dtype="float64"), ddof=1))
    naive_t_statistic = float(
        primary.mean() / (primary_deviation / math.sqrt(len(primary)))
    )
    gross_metrics = calculate_performance_metrics(
        cost_paths[0.0]["net_return"], annualization_factor=ANNUALIZATION_FACTOR
    )

    hac_records: list[dict[str, object]] = []
    for lag in HAC_LAGS_SENSITIVITY:
        statistics = calculate_hac_statistics(primary, lag=lag)
        hac_records.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "cost_bps_per_turnover": PRIMARY_COST_BPS,
                "hac_lag": lag,
                **statistics,
                "is_primary_lag": lag == PRIMARY_HAC_LAG,
            }
        )
    hac_table = pd.DataFrame.from_records(hac_records).loc[:, HAC_SENSITIVITY_COLUMNS]
    primary_hac = _single_row(
        hac_table, hac_table["hac_lag"].eq(PRIMARY_HAC_LAG), label="primary HAC"
    )

    bootstrap_records: list[dict[str, object]] = []
    for block_length in BOOTSTRAP_BLOCK_LENGTHS:
        interval = circular_block_bootstrap_intervals(
            primary,
            block_length=block_length,
            replications=BOOTSTRAP_REPLICATIONS,
            seed=PRIMARY_BOOTSTRAP_SEED,
        )
        bootstrap_records.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "cost_bps_per_turnover": PRIMARY_COST_BPS,
                **interval,
                "mean_interval_includes_zero": bool(
                    interval["mean_ci_lower"] <= 0.0 <= interval["mean_ci_upper"]
                ),
                "sharpe_interval_includes_zero": bool(
                    interval["sharpe_ci_lower"] <= 0.0 <= interval["sharpe_ci_upper"]
                ),
                "is_primary_block_length": block_length
                == PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
            }
        )
    bootstrap_table = pd.DataFrame.from_records(bootstrap_records).loc[
        :, BOOTSTRAP_SENSITIVITY_COLUMNS
    ]
    primary_bootstrap = _single_row(
        bootstrap_table,
        bootstrap_table["block_length"].eq(PRIMARY_BOOTSTRAP_BLOCK_LENGTH),
        label="primary bootstrap",
    )

    inference_rows = reversion_results.inference_results.loc[
        reversion_results.inference_results["series"].eq(PRIMARY_SERIES)
    ].copy(deep=True)
    if tuple(inference_rows["configuration_id"]) != CONFIGURATION_IDS:
        raise Day31SlowOuRobustnessError("Frozen three-configuration DSR scope changed.")
    saved_trial_rows = day28_evidence.inference.loc[
        day28_evidence.inference["series"].eq(PRIMARY_SERIES)
    ]
    for configuration_id in CONFIGURATION_IDS:
        observed = _single_row(
            inference_rows,
            inference_rows["configuration_id"].eq(configuration_id),
            label=f"{configuration_id} trial",
        )
        saved = _single_row(
            saved_trial_rows,
            saved_trial_rows["configuration_id"].eq(configuration_id),
            label=f"saved {configuration_id} trial",
        )
        if not math.isclose(
            float(observed["annualized_sharpe_ratio"]),
            float(saved["annualized_sharpe_ratio"]),
            rel_tol=REPRODUCTION_TOLERANCE,
            abs_tol=REPRODUCTION_TOLERANCE,
        ):
            raise Day31SlowOuRobustnessError("Day 28 DSR trial Sharpes were not reproduced.")
    trial_per_period = inference_rows["annualized_sharpe_ratio"].to_numpy(
        dtype="float64"
    ) / math.sqrt(ANNUALIZATION_FACTOR)
    annualized_sharpe = _annualized_sharpe(primary.to_numpy(dtype="float64"))
    per_period_sharpe = annualized_sharpe / math.sqrt(ANNUALIZATION_FACTOR)
    sample_skewness = float(skew(primary, bias=False))
    sample_kurtosis = float(kurtosis(primary, fisher=False, bias=False))
    psr = _sharpe_probability(
        per_period_sharpe=per_period_sharpe,
        benchmark=0.0,
        observations=len(primary),
        sample_skewness=sample_skewness,
        sample_kurtosis=sample_kurtosis,
    )
    dsr_benchmark = _deflated_benchmark(trial_per_period)
    dsr = _sharpe_probability(
        per_period_sharpe=per_period_sharpe,
        benchmark=dsr_benchmark,
        observations=len(primary),
        sample_skewness=sample_skewness,
        sample_kurtosis=sample_kurtosis,
    )
    psr_dsr = pd.DataFrame.from_records(
        [
            {
                "experiment_version": EXPERIMENT_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "cost_bps_per_turnover": PRIMARY_COST_BPS,
                "observations": len(primary),
                "annualized_sharpe_ratio": annualized_sharpe,
                "sample_skewness": sample_skewness,
                "sample_kurtosis": sample_kurtosis,
                "psr_benchmark_per_period": 0.0,
                "probabilistic_sharpe_probability": psr,
                "deflated_sharpe_benchmark_per_period": dsr_benchmark,
                "deflated_sharpe_probability": dsr,
                "declared_trials": DECLARED_DSR_TRIALS,
                "trial_scope": "ou_vwap_fast|ou_vwap_base|ou_vwap_slow",
                "excluded_research_families": (
                    "trend|ema_macd|portfolio|other_axiom_research_choices"
                ),
                "globally_corrected_dsr_claimed": False,
                "multiplicity_disclosure": (
                    "local_three_configuration_diagnostic_likely_understates_"
                    "total_research_multiplicity"
                ),
            }
        ]
    ).loc[:, PSR_DSR_DISCLOSURE_COLUMNS]

    saved_aggregate = _single_row(
        day28_evidence.aggregate,
        day28_evidence.aggregate["configuration_id"].eq(PRIMARY_CONFIGURATION_ID)
        & day28_evidence.aggregate["series"].eq(PRIMARY_SERIES),
        label="saved primary aggregate",
    )
    saved_inference = _single_row(
        day28_evidence.inference,
        day28_evidence.inference["configuration_id"].eq(PRIMARY_CONFIGURATION_ID)
        & day28_evidence.inference["series"].eq(PRIMARY_SERIES),
        label="saved primary inference",
    )
    saved_cost_one = _single_row(
        day28_evidence.cost_sensitivity,
        day28_evidence.cost_sensitivity["configuration_id"].eq(PRIMARY_CONFIGURATION_ID)
        & day28_evidence.cost_sensitivity["series"].eq(PRIMARY_SERIES)
        & day28_evidence.cost_sensitivity["cost_bps_per_turnover"].eq(PRIMARY_COST_BPS),
        label="saved primary cost",
    )
    primary_reproduction = _metric_reproduction_rows(
        {
            "observations": (saved_inference["observations"], len(primary)),
            "cumulative_gross_return": (
                saved_aggregate["cumulative_gross_return"],
                gross_metrics.cumulative_return,
            ),
            "cumulative_net_return_1bp": (
                saved_aggregate["cumulative_net_return_1bp"],
                primary_metrics.cumulative_return,
            ),
            "annualized_volatility_1bp": (
                saved_aggregate["annualized_volatility_1bp"],
                primary_metrics.annualized_volatility,
            ),
            "sharpe_ratio_1bp": (
                saved_aggregate["sharpe_ratio_1bp"], primary_metrics.sharpe_ratio
            ),
            "maximum_drawdown_1bp": (
                saved_aggregate["maximum_drawdown_1bp"], primary_metrics.max_drawdown
            ),
            "turnover": (saved_aggregate["turnover"], total_turnover),
            "cost_table_cumulative_return_1bp": (
                saved_cost_one["cumulative_return"], primary_metrics.cumulative_return
            ),
            "mean_session_return": (
                saved_inference["mean_session_return"], float(primary.mean())
            ),
            "naive_t_statistic": (
                saved_inference["naive_t_statistic"], naive_t_statistic
            ),
            "hac_t_statistic_5": (
                saved_inference["hac_t_statistic"], primary_hac["hac_t_statistic"]
            ),
            "annualized_sharpe_ratio": (
                saved_inference["annualized_sharpe_ratio"], annualized_sharpe
            ),
            "bootstrap_mean_ci_lower": (
                saved_inference["bootstrap_mean_ci_lower"],
                primary_bootstrap["mean_ci_lower"],
            ),
            "bootstrap_mean_ci_upper": (
                saved_inference["bootstrap_mean_ci_upper"],
                primary_bootstrap["mean_ci_upper"],
            ),
            "bootstrap_sharpe_ci_lower": (
                saved_inference["bootstrap_sharpe_ci_lower"],
                primary_bootstrap["sharpe_ci_lower"],
            ),
            "bootstrap_sharpe_ci_upper": (
                saved_inference["bootstrap_sharpe_ci_upper"],
                primary_bootstrap["sharpe_ci_upper"],
            ),
            "sample_skewness": (saved_inference["sample_skewness"], sample_skewness),
            "sample_kurtosis": (saved_inference["sample_kurtosis"], sample_kurtosis),
            "probabilistic_sharpe_probability": (
                saved_inference["probabilistic_sharpe_probability"], psr
            ),
            "deflated_sharpe_benchmark": (
                saved_inference["deflated_sharpe_benchmark"], dsr_benchmark
            ),
            "deflated_sharpe_probability": (
                saved_inference["deflated_sharpe_probability"], dsr
            ),
            "declared_trials": (
                saved_inference["declared_trials"], DECLARED_DSR_TRIALS
            ),
        }
    )
    leave_one_year_out = calculate_leave_one_year_out(primary)
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "day28_schema_version": DAY28_SPECIFICATION_VERSION,
        "timing_version": DAY28_TIMING_VERSION,
        "claim_boundary": "development_sensitivity_only_no_selection_or_promotion",
        "source_dataset_path": source_dataset_path,
        "source_sha256": source_sha256,
        "data_audit": dict(data_audit),
        "generation_timestamp": generation_timestamp,
        "configuration_id": PRIMARY_CONFIGURATION_ID,
        "series": PRIMARY_SERIES,
        "annualization_factor": ANNUALIZATION_FACTOR,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "cost_sensitivity_bps": list(TRANSACTION_COST_BPS),
        "primary_hac_lag": PRIMARY_HAC_LAG,
        "hac_lag_sensitivity": list(HAC_LAGS_SENSITIVITY),
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "primary_bootstrap_block_length": PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
        "bootstrap_block_lengths": list(BOOTSTRAP_BLOCK_LENGTHS),
        "day28_bootstrap_seed": BOOTSTRAP_SEED,
        "effective_primary_bootstrap_seed": PRIMARY_BOOTSTRAP_SEED,
        "declared_dsr_trials": DECLARED_DSR_TRIALS,
        "dsr_scope": list(CONFIGURATION_IDS),
        "leave_one_year_out_years": list(LEAVE_ONE_YEAR_OUT_YEARS),
        "day28_comparator_sources": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(day28_evidence.source_hashes.items())
        ],
        "primary_path_sessions": len(primary),
        "primary_path_start": primary.index.min().date().isoformat(),
        "primary_path_end": primary.index.max().date().isoformat(),
        "locked_2026_accessed": False,
        "holdout_runner_accessed": False,
        "report_generated": False,
        "notebook_created": False,
        "chart_created": False,
        "broker_or_network_accessed": False,
        "parameter_selection_performed": False,
        "promotion_performed": False,
        "globally_corrected_dsr_claimed": False,
        "commit_or_push_performed": False,
    }
    verify_comparator_snapshot(day28_evidence.source_hashes)
    return Day31SlowOuRobustnessResults(
        source_and_method_metadata=metadata,
        primary_day28_reproduction=primary_reproduction,
        transaction_cost_sensitivity=cost_table,
        hac_lag_sensitivity=hac_table,
        block_bootstrap_sensitivity=bootstrap_table,
        leave_one_year_out=leave_one_year_out,
        psr_dsr_disclosure=psr_dsr,
        comparator_snapshot=day28_evidence.source_hashes,
    )


def run_day31_slow_ou_robustness(
    bars: pd.DataFrame,
    *,
    source_dataset_path: str,
    source_sha256: str,
    day28_directory: Path,
    generation_timestamp: str,
) -> Day31SlowOuRobustnessResults:
    """Run the frozen engine and slow-path reconstruction on authorized bars."""

    audit = audit_day28_development_input(
        bars,
        source_dataset_path=source_dataset_path,
        source_sha256=source_sha256,
    )
    evidence = load_day28_evidence(day28_directory)
    reversion = run_reversion_inference(bars)
    frozen_path = build_frozen_slow_ou_path(bars)
    return build_day31_results(
        frozen_path=frozen_path,
        reversion_results=reversion,
        day28_evidence=evidence,
        data_audit=audit,
        source_dataset_path=source_dataset_path,
        source_sha256=source_sha256,
        generation_timestamp=generation_timestamp,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n", float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def describe_existing_bundle(directory: Path) -> str:
    if not directory.exists():
        return "directory does not exist"
    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        entries.append(
            f"{path.name}:{sha256_file(path) if path.is_file() else 'directory'}"
        )
    return ", ".join(entries) if entries else "empty directory"


def write_day31_artifacts(
    results: Day31SlowOuRobustnessResults,
    directory: Path,
) -> tuple[Path, ...]:
    """Atomically create one allow-listed bundle, refusing every overwrite."""

    if not isinstance(results, Day31SlowOuRobustnessResults):
        raise TypeError("results must be Day31SlowOuRobustnessResults.")
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path.")
    if directory.name != OUTPUT_DIRECTORY_BASENAME:
        raise Day31SlowOuRobustnessError(
            f"Artifact directory basename must be {OUTPUT_DIRECTORY_BASENAME!r}."
        )
    if directory.exists():
        raise FileExistsError(
            f"Day 31 artifact directory already exists: {describe_existing_bundle(directory)}"
        )
    verify_comparator_snapshot(results.comparator_snapshot)
    tables = {
        PRIMARY_REPRODUCTION_FILENAME: results.primary_day28_reproduction,
        COST_SENSITIVITY_FILENAME: results.transaction_cost_sensitivity,
        HAC_SENSITIVITY_FILENAME: results.hac_lag_sensitivity,
        BOOTSTRAP_SENSITIVITY_FILENAME: results.block_bootstrap_sensitivity,
        LEAVE_ONE_YEAR_OUT_FILENAME: results.leave_one_year_out,
        PSR_DSR_FILENAME: results.psr_dsr_disclosure,
    }
    payloads: dict[str, bytes] = {
        SOURCE_METADATA_FILENAME: _json_bytes(results.source_and_method_metadata),
        **{filename: _csv_bytes(table) for filename, table in tables.items()},
    }
    manifest = {
        **dict(results.source_and_method_metadata),
        "artifacts": [
            {
                "filename": filename,
                "rows": (
                    int(len(tables[filename]))
                    if filename in tables
                    else None
                ),
                "bytes": len(payloads[filename]),
                "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
            }
            for filename in APPROVED_ARTIFACT_NAMES
            if filename != MANIFEST_FILENAME
        ],
        "artifact_file_list": list(APPROVED_ARTIFACT_NAMES),
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    if tuple(payloads) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Day 31 artifact allow-list changed.")
    if any(Path(name).suffix not in {".csv", ".json"} for name in payloads):
        raise RuntimeError("Day 31 produced a prohibited artifact type.")

    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.stage-", dir=directory.parent)
    )
    try:
        for filename in APPROVED_ARTIFACT_NAMES:
            (stage / filename).write_bytes(payloads[filename])
        os.replace(stage, directory)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verify_comparator_snapshot(results.comparator_snapshot)
    observed = tuple(sorted(path.name for path in directory.iterdir()))
    if observed != tuple(sorted(APPROVED_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 31 artifact allow-list changed.")
    return tuple(directory / filename for filename in APPROVED_ARTIFACT_NAMES)
