"""Development-only evidence for fixed-holdings portfolio accounting.

This module compares the frozen Day 16 daily constant-mix accounting path with
the corrected Phase 3 fixed-holdings path.  Target estimators, covariance
estimates, folds, constraints, annualization, and cost rate are invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd

from systematic_alpha.analysis.portfolio_allocation_validation import (
    ALLOCATION_COST_RATE,
    ALLOCATION_RULES,
    FIXED_HOLDINGS_ACCOUNTING_VERSION,
    MAXIMUM_WEIGHT,
    TRADING_SESSIONS_PER_YEAR,
    FixedHoldingsPortfolioAllocationResults,
    PortfolioAllocationResults,
    PortfolioAllocationValidationError,
    analyze_portfolio_allocation_panel,
    analyze_portfolio_allocation_panel_fixed_holdings,
    calculate_portfolio_metrics,
)
from systematic_alpha.analysis.strategy_diversification import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FREQUENCY,
    SLEEVE_IDS,
)
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds


EXPERIMENT_VERSION: Final[str] = (
    "day29_fixed_holdings_portfolio_experiment_v1"
)
ACCOUNTING_VERSION: Final[str] = FIXED_HOLDINGS_ACCOUNTING_VERSION
OUTPUT_DIRECTORY_BASENAME: Final[str] = (
    "day29_fixed_holdings_portfolio_experiment"
)
WEALTH_IDENTITY_TOLERANCE: Final[float] = 1e-12

SOURCE_METADATA_FILENAME: Final[str] = "source_and_method_metadata.json"
INVARIANCE_FILENAME: Final[str] = "target_and_covariance_invariance.csv"
FOLD_PERFORMANCE_FILENAME: Final[str] = "fold_performance_comparison.csv"
AGGREGATE_PERFORMANCE_FILENAME: Final[str] = (
    "aggregate_performance_comparison.csv"
)
FOLD_TURNOVER_FILENAME: Final[str] = "fold_turnover_comparison.csv"
ENDING_DRIFT_FILENAME: Final[str] = "ending_weight_drift.csv"
WEIGHT_PATH_FILENAME: Final[str] = "corrected_weight_path.csv"
WEALTH_IDENTITY_FILENAME: Final[str] = "wealth_identity_checks.csv"
RETURN_COMPARISON_FILENAME: Final[str] = "portfolio_return_comparison.csv"
MANIFEST_FILENAME: Final[str] = "manifest.json"

CSV_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    INVARIANCE_FILENAME,
    FOLD_PERFORMANCE_FILENAME,
    AGGREGATE_PERFORMANCE_FILENAME,
    FOLD_TURNOVER_FILENAME,
    ENDING_DRIFT_FILENAME,
    WEIGHT_PATH_FILENAME,
    WEALTH_IDENTITY_FILENAME,
    RETURN_COMPARISON_FILENAME,
)
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SOURCE_METADATA_FILENAME,
    *CSV_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
)

DAY16_COMPARATOR_FILES: Final[tuple[str, ...]] = (
    "allocation_weights.csv",
    "allocation_diagnostics.csv",
    "fold_portfolio_performance.csv",
    "aggregate_portfolio_performance.csv",
    "portfolio_return_panel.csv",
    "manifest.json",
)
DAY25_COMPARATOR_FILES: Final[tuple[str, ...]] = (
    "sleeve_session_returns.csv",
    "allocation_weights.csv",
    "allocation_diagnostics.csv",
    "fold_portfolio_performance.csv",
    "aggregate_portfolio_performance.csv",
    "portfolio_return_panel.csv",
    "methodology.json",
    "manifest.json",
)

INVARIANCE_COLUMNS: Final[tuple[str, ...]] = (
    "invariant_type",
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "row_sleeve_id",
    "column_sleeve_id",
    "value_name",
    "historical_numeric_value",
    "corrected_numeric_value",
    "corrected_minus_historical",
    "historical_text_value",
    "corrected_text_value",
    "exact_equal",
)

FOLD_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "observations",
    "start_session",
    "end_session",
    "target_weights",
    "historical_ending_weights",
    "corrected_ending_weights",
    "historical_cumulative_gross_return",
    "corrected_cumulative_gross_return",
    "corrected_minus_historical_cumulative_gross_return",
    "historical_cumulative_net_return_1bp",
    "corrected_cumulative_net_return_1bp",
    "corrected_minus_historical_cumulative_net_return_1bp",
    "historical_annualized_net_volatility_1bp",
    "corrected_annualized_net_volatility_1bp",
    "corrected_minus_historical_annualized_net_volatility_1bp",
    "historical_annualized_net_sharpe_1bp",
    "corrected_annualized_net_sharpe_1bp",
    "corrected_minus_historical_annualized_net_sharpe_1bp",
    "historical_maximum_net_drawdown_1bp",
    "corrected_maximum_net_drawdown_1bp",
    "corrected_minus_historical_maximum_net_drawdown_1bp",
    "historical_turnover",
    "corrected_turnover",
    "corrected_minus_historical_turnover",
    "historical_transaction_cost",
    "corrected_transaction_cost",
    "corrected_minus_historical_transaction_cost",
    "historical_gross_terminal_wealth",
    "corrected_gross_terminal_wealth",
    "corrected_minus_historical_gross_terminal_wealth",
    "historical_net_terminal_wealth",
    "corrected_net_terminal_wealth",
    "corrected_minus_historical_net_terminal_wealth",
    "corrected_gross_wealth_identity_residual",
)

AGGREGATE_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "allocation_rule",
    "folds",
    "sessions",
    "start_session",
    "end_session",
    "historical_cumulative_gross_return",
    "corrected_cumulative_gross_return",
    "corrected_minus_historical_cumulative_gross_return",
    "historical_cumulative_net_return_1bp",
    "corrected_cumulative_net_return_1bp",
    "corrected_minus_historical_cumulative_net_return_1bp",
    "historical_annualized_net_volatility_1bp",
    "corrected_annualized_net_volatility_1bp",
    "corrected_minus_historical_annualized_net_volatility_1bp",
    "historical_annualized_net_sharpe_1bp",
    "corrected_annualized_net_sharpe_1bp",
    "corrected_minus_historical_annualized_net_sharpe_1bp",
    "historical_maximum_net_drawdown_1bp",
    "corrected_maximum_net_drawdown_1bp",
    "corrected_minus_historical_maximum_net_drawdown_1bp",
    "historical_total_turnover",
    "corrected_total_turnover",
    "corrected_minus_historical_total_turnover",
    "historical_total_transaction_cost",
    "corrected_total_transaction_cost",
    "corrected_minus_historical_total_transaction_cost",
    "return_accounting_effect",
    "turnover_effect",
    "cost_effect",
)

FOLD_TURNOVER_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "historical_turnover",
    "corrected_turnover",
    "corrected_minus_historical_turnover",
    "historical_transaction_cost",
    "corrected_transaction_cost",
    "corrected_minus_historical_transaction_cost",
    "historical_previous_reference",
    "corrected_previous_reference",
)

ENDING_DRIFT_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "sleeve_order",
    "target_weight",
    "ending_drifted_weight",
    "ending_minus_target_weight",
    "absolute_ending_drift",
    "maximum_absolute_intrafold_drift",
    "minimum_observed_pre_return_weight",
    "maximum_observed_pre_return_weight",
    "fold_l1_ending_drift",
    "fold_maximum_sleeve_drift",
    "any_drifted_weight_above_target_cap",
    "target_cap_status",
)

CORRECTED_WEIGHT_PATH_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "sleeve_order",
    "target_weight",
    "pre_return_weight",
    "post_return_weight",
    "pre_return_minus_target",
    "post_return_minus_target",
    "pre_return_weight_above_target_cap",
    "post_return_weight_above_target_cap",
    "target_cap_status",
)

WEALTH_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "sessions",
    "recursive_gross_terminal_wealth",
    "holdings_gross_terminal_wealth",
    "gross_wealth_identity_residual",
    "absolute_gross_wealth_identity_residual",
    "tolerance",
    "identity_within_tolerance",
)

RETURN_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "fold_id",
    "allocation_rule",
    "historical_gross_return",
    "corrected_gross_return",
    "corrected_minus_historical_gross_return",
    "historical_net_return_1bp",
    "corrected_net_return_1bp",
    "corrected_minus_historical_net_return_1bp",
    "historical_cost_charged",
    "corrected_cost_charged",
    "is_first_fold_session",
)


class Day29FixedHoldingsExperimentError(ValueError):
    """Raised when the Day 29 evidence contract fails closed."""


def _copy_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class Day29FixedHoldingsExperimentResults:
    """Defensively retained Day 29 evidence tables and metadata."""

    source_and_method_metadata: Mapping[str, object]
    target_and_covariance_invariance: pd.DataFrame
    fold_performance_comparison: pd.DataFrame
    aggregate_performance_comparison: pd.DataFrame
    fold_turnover_comparison: pd.DataFrame
    ending_weight_drift: pd.DataFrame
    corrected_weight_path: pd.DataFrame
    wealth_identity_checks: pd.DataFrame
    portfolio_return_comparison: pd.DataFrame
    comparator_snapshot: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "target_and_covariance_invariance",
            "fold_performance_comparison",
            "aggregate_performance_comparison",
            "fold_turnover_comparison",
            "ending_weight_drift",
            "corrected_weight_path",
            "wealth_identity_checks",
            "portfolio_return_comparison",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "source_and_method_metadata",
            dict(self.source_and_method_metadata),
        )
        object.__setattr__(self, "comparator_snapshot", dict(self.comparator_snapshot))

    def copy_target_and_covariance_invariance(self) -> pd.DataFrame:
        return self.target_and_covariance_invariance.copy(deep=True)

    def copy_fold_performance_comparison(self) -> pd.DataFrame:
        return self.fold_performance_comparison.copy(deep=True)

    def copy_aggregate_performance_comparison(self) -> pd.DataFrame:
        return self.aggregate_performance_comparison.copy(deep=True)

    def copy_fold_turnover_comparison(self) -> pd.DataFrame:
        return self.fold_turnover_comparison.copy(deep=True)

    def copy_ending_weight_drift(self) -> pd.DataFrame:
        return self.ending_weight_drift.copy(deep=True)

    def copy_corrected_weight_path(self) -> pd.DataFrame:
        return self.corrected_weight_path.copy(deep=True)

    def copy_wealth_identity_checks(self) -> pd.DataFrame:
        return self.wealth_identity_checks.copy(deep=True)

    def copy_portfolio_return_comparison(self) -> pd.DataFrame:
        return self.portfolio_return_comparison.copy(deep=True)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one existing file."""

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
        raise Day29FixedHoldingsExperimentError(
            f"Required JSON comparator is unreadable: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise Day29FixedHoldingsExperimentError(
            f"Required JSON comparator must contain an object: {path}."
        )
    return value


def load_comparator_snapshot(
    *,
    day16_directory: Path,
    day25_directory: Path,
) -> dict[str, str]:
    """Authenticate and hash the exact machine-readable preservation set."""

    for name, directory in (
        ("day16_directory", day16_directory),
        ("day25_directory", day25_directory),
    ):
        if not isinstance(directory, Path):
            raise TypeError(f"{name} must be a pathlib.Path.")
        if not directory.is_dir():
            raise FileNotFoundError(f"Comparator directory does not exist: {directory}.")

    manifests = {
        day16_directory: _read_json(day16_directory / "manifest.json"),
        day25_directory: _read_json(day25_directory / "manifest.json"),
    }
    file_sets = {
        day16_directory: DAY16_COMPARATOR_FILES,
        day25_directory: DAY25_COMPARATOR_FILES,
    }
    snapshot: dict[str, str] = {}
    for directory, filenames in file_sets.items():
        declared = manifests[directory].get("artifact_sha256")
        if not isinstance(declared, dict):
            raise Day29FixedHoldingsExperimentError(
                f"Comparator manifest lacks artifact hashes: {directory}."
            )
        for filename in filenames:
            path = (directory / filename).resolve()
            current = sha256_file(path)
            if filename != "manifest.json":
                expected = declared.get(filename)
                if expected != current:
                    raise Day29FixedHoldingsExperimentError(
                        f"Comparator manifest hash mismatch: {path}."
                    )
            snapshot[path.as_posix()] = current
    return dict(sorted(snapshot.items()))


def verify_comparator_snapshot(snapshot: Mapping[str, str]) -> None:
    """Require all comparator bytes to remain unchanged in place."""

    if not isinstance(snapshot, Mapping) or not snapshot:
        raise Day29FixedHoldingsExperimentError(
            "Comparator snapshot must be a non-empty mapping."
        )
    for raw_path, expected in sorted(snapshot.items()):
        path = Path(raw_path)
        if sha256_file(path) != expected:
            raise Day29FixedHoldingsExperimentError(
                f"Immutable comparator changed: {path}."
            )


def hash_return_panel(panel: pd.DataFrame) -> str:
    """Hash a canonical, deterministic serialization of the input panel."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")
    serializable = panel.copy(deep=True).reset_index()
    payload = serializable.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_development_panel(
    panel: pd.DataFrame,
    *,
    require_exact_development_range: bool,
) -> pd.DataFrame:
    """Validate Day 29 input scope before either accounting path is run."""

    if not isinstance(require_exact_development_range, bool):
        raise TypeError("require_exact_development_range must be a boolean.")
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")
    if tuple(panel.columns) != SLEEVE_IDS:
        raise Day29FixedHoldingsExperimentError(
            "Sleeve labels and ordering must exactly match the frozen contract."
        )
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise Day29FixedHoldingsExperimentError(
            "The return panel requires a DatetimeIndex."
        )
    if panel.index.name != "session_date" or panel.index.tz is None:
        raise Day29FixedHoldingsExperimentError(
            "The return panel requires a timezone-aware session_date index."
        )
    if not panel.index.is_monotonic_increasing:
        raise Day29FixedHoldingsExperimentError(
            "Return-panel timestamps must be monotonic."
        )
    if panel.index.has_duplicates:
        raise Day29FixedHoldingsExperimentError(
            "Return-panel observations must be unique."
        )
    try:
        numeric = panel.copy(deep=True).apply(pd.to_numeric, errors="raise").astype(
            "float64"
        )
    except (TypeError, ValueError) as exc:
        raise Day29FixedHoldingsExperimentError(
            "Return-panel values must be numeric."
        ) from exc
    values = numeric.to_numpy(dtype="float64", copy=True)
    if not np.isfinite(values).all():
        raise Day29FixedHoldingsExperimentError(
            "Return-panel values must be finite and complete."
        )
    if np.less_equal(values, -1.0).any():
        raise Day29FixedHoldingsExperimentError(
            "Return-panel values must be strictly greater than -1."
        )
    if numeric.index.min() < DEVELOPMENT_START:
        raise Day29FixedHoldingsExperimentError(
            "Return panel begins before the authorized development start."
        )
    if numeric.index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise Day29FixedHoldingsExperimentError(
            "Return panel contains a prohibited 2026-or-later observation."
        )
    if require_exact_development_range and (
        numeric.index.min() != DEVELOPMENT_START
        or numeric.index.max()
        != pd.Timestamp("2025-12-31", tz="UTC")
    ):
        raise Day29FixedHoldingsExperimentError(
            "Canonical Day 29 input must span exactly 2020-01-02 through "
            "2025-12-31."
        )
    numeric.index = numeric.index.copy()
    numeric.index.name = "session_date"
    return numeric


def _ordered_weights(results: PortfolioAllocationResults) -> pd.DataFrame:
    return results.copy_allocation_weights().sort_values(
        ["fold_id", "allocation_rule", "sleeve_order"],
        kind="stable",
    ).reset_index(drop=True)


def _ordered_covariances(results: PortfolioAllocationResults) -> pd.DataFrame:
    return results.copy_minimum_variance_covariances().sort_values(
        ["fold_id", "sleeve_order"], kind="stable"
    ).reset_index(drop=True)


def _fold_signature(return_panel: pd.DataFrame, fold_id: str) -> str:
    subset = return_panel.loc[return_panel["fold_id"].eq(fold_id)]
    if subset.empty:
        raise Day29FixedHoldingsExperimentError(
            f"Missing fold in portfolio return panel: {fold_id}."
        )
    sessions = pd.to_datetime(subset["session_date"], utc=True)
    return "|".join(
        (
            str(int(len(subset))),
            sessions.min().isoformat(),
            sessions.max().isoformat(),
        )
    )


def validate_method_invariants(
    historical: PortfolioAllocationResults,
    corrected: FixedHoldingsPortfolioAllocationResults,
) -> pd.DataFrame:
    """Fail closed unless every non-accounting input is exactly invariant."""

    if not isinstance(historical, PortfolioAllocationResults):
        raise TypeError("historical must be PortfolioAllocationResults.")
    if not isinstance(corrected, FixedHoldingsPortfolioAllocationResults):
        raise TypeError("corrected must be FixedHoldingsPortfolioAllocationResults.")
    if corrected.accounting_version != ACCOUNTING_VERSION:
        raise Day29FixedHoldingsExperimentError(
            "Corrected accounting version does not match Day 29."
        )

    historical_weights = _ordered_weights(historical)
    corrected_weights = _ordered_weights(corrected.portfolio_results)
    key_columns = ["fold_id", "allocation_rule", "sleeve_id", "sleeve_order"]
    for column in key_columns:
        if not historical_weights[column].equals(corrected_weights[column]):
            raise Day29FixedHoldingsExperimentError(
                "Target-weight labels or ordering differ between methods."
            )
    if not np.array_equal(
        historical_weights["weight"].to_numpy(dtype="float64"),
        corrected_weights["weight"].to_numpy(dtype="float64"),
    ):
        raise Day29FixedHoldingsExperimentError(
            "Target weights differ between accounting methods."
        )

    historical_covariance = _ordered_covariances(historical)
    corrected_covariance = _ordered_covariances(corrected.portfolio_results)
    if not historical_covariance.equals(corrected_covariance):
        raise Day29FixedHoldingsExperimentError(
            "Ledoit-Wolf covariance estimates differ between methods."
        )

    historical_diagnostics = historical.copy_allocation_diagnostics().sort_values(
        ["fold_id", "allocation_rule"], kind="stable"
    ).reset_index(drop=True)
    corrected_diagnostics = (
        corrected.portfolio_results.copy_allocation_diagnostics()
        .sort_values(["fold_id", "allocation_rule"], kind="stable")
        .reset_index(drop=True)
    )
    invariant_diagnostic_columns = (
        "fold_id",
        "allocation_rule",
        "training_sessions",
        "test_sessions",
        "covariance_estimator",
        "shrinkage_coefficient",
        "solver_status",
        "weight_sum",
        "minimum_weight",
        "maximum_weight",
        "gross_weight",
        "herfindahl_concentration",
        "effective_sleeve_count",
        "constraint_valid",
    )
    if not historical_diagnostics.loc[:, invariant_diagnostic_columns].equals(
        corrected_diagnostics.loc[:, invariant_diagnostic_columns]
    ):
        raise Day29FixedHoldingsExperimentError(
            "Fold, estimator, shrinkage, optimizer, or target diagnostics differ."
        )

    historical_returns = historical.copy_portfolio_return_panel()
    corrected_returns = corrected.portfolio_results.copy_portfolio_return_panel()
    for fold in build_walk_forward_folds():
        if _fold_signature(historical_returns, fold.fold_id) != _fold_signature(
            corrected_returns, fold.fold_id
        ):
            raise Day29FixedHoldingsExperimentError(
                "Fold definitions differ between accounting methods."
            )

    records: list[dict[str, object]] = []
    for historical_row, corrected_row in zip(
        historical_weights.itertuples(index=False),
        corrected_weights.itertuples(index=False),
        strict=True,
    ):
        delta = float(corrected_row.weight - historical_row.weight)
        records.append(
            {
                "invariant_type": "target_weight",
                "fold_id": historical_row.fold_id,
                "allocation_rule": historical_row.allocation_rule,
                "sleeve_id": historical_row.sleeve_id,
                "row_sleeve_id": "",
                "column_sleeve_id": "",
                "value_name": "weight",
                "historical_numeric_value": float(historical_row.weight),
                "corrected_numeric_value": float(corrected_row.weight),
                "corrected_minus_historical": delta,
                "historical_text_value": "",
                "corrected_text_value": "",
                "exact_equal": delta == 0.0,
            }
        )

    corrected_covariance_by_key = corrected_covariance.set_index(
        ["fold_id", "sleeve_id"]
    )
    for row in historical_covariance.itertuples(index=False):
        corrected_row = corrected_covariance_by_key.loc[(row.fold_id, row.sleeve_id)]
        for column_sleeve in SLEEVE_IDS:
            historical_value = float(getattr(row, column_sleeve))
            corrected_value = float(corrected_row[column_sleeve])
            delta = corrected_value - historical_value
            records.append(
                {
                    "invariant_type": "ledoit_wolf_covariance",
                    "fold_id": row.fold_id,
                    "allocation_rule": "constrained_minimum_variance",
                    "sleeve_id": "",
                    "row_sleeve_id": row.sleeve_id,
                    "column_sleeve_id": column_sleeve,
                    "value_name": "covariance",
                    "historical_numeric_value": historical_value,
                    "corrected_numeric_value": corrected_value,
                    "corrected_minus_historical": delta,
                    "historical_text_value": "",
                    "corrected_text_value": "",
                    "exact_equal": delta == 0.0,
                }
            )

    for fold in build_walk_forward_folds():
        historical_signature = _fold_signature(historical_returns, fold.fold_id)
        corrected_signature = _fold_signature(corrected_returns, fold.fold_id)
        records.append(
            {
                "invariant_type": "fold_definition",
                "fold_id": fold.fold_id,
                "allocation_rule": "",
                "sleeve_id": "",
                "row_sleeve_id": "",
                "column_sleeve_id": "",
                "value_name": "test_sessions_start_end",
                "historical_numeric_value": "",
                "corrected_numeric_value": "",
                "corrected_minus_historical": "",
                "historical_text_value": historical_signature,
                "corrected_text_value": corrected_signature,
                "exact_equal": historical_signature == corrected_signature,
            }
        )

    for fold_id in tuple(fold.fold_id for fold in build_walk_forward_folds()):
        historical_row = historical_diagnostics.loc[
            historical_diagnostics["fold_id"].eq(fold_id)
            & historical_diagnostics["allocation_rule"].eq(
                "constrained_minimum_variance"
            )
        ].iloc[0]
        corrected_row = corrected_diagnostics.loc[
            corrected_diagnostics["fold_id"].eq(fold_id)
            & corrected_diagnostics["allocation_rule"].eq(
                "constrained_minimum_variance"
            )
        ].iloc[0]
        historical_value = float(historical_row["shrinkage_coefficient"])
        corrected_value = float(corrected_row["shrinkage_coefficient"])
        delta = corrected_value - historical_value
        records.append(
            {
                "invariant_type": "ledoit_wolf_shrinkage",
                "fold_id": fold_id,
                "allocation_rule": "constrained_minimum_variance",
                "sleeve_id": "",
                "row_sleeve_id": "",
                "column_sleeve_id": "",
                "value_name": "shrinkage_coefficient",
                "historical_numeric_value": historical_value,
                "corrected_numeric_value": corrected_value,
                "corrected_minus_historical": delta,
                "historical_text_value": "",
                "corrected_text_value": "",
                "exact_equal": delta == 0.0,
            }
        )
    result = pd.DataFrame.from_records(records, columns=INVARIANCE_COLUMNS)
    if not result["exact_equal"].all():
        raise Day29FixedHoldingsExperimentError(
            "At least one Day 29 method invariant failed."
        )
    return result


def _target_vector(
    weights: pd.DataFrame,
    *,
    fold_id: str,
    rule: str,
) -> np.ndarray:
    rows = weights.loc[
        weights["fold_id"].eq(fold_id)
        & weights["allocation_rule"].eq(rule)
    ].sort_values("sleeve_order", kind="stable")
    if tuple(rows["sleeve_id"]) != SLEEVE_IDS:
        raise Day29FixedHoldingsExperimentError(
            "Target weights do not follow the frozen sleeve order."
        )
    return rows["weight"].to_numpy(dtype="float64", copy=True)


def _json_vector(values: np.ndarray) -> str:
    return json.dumps(
        {sleeve: float(value) for sleeve, value in zip(SLEEVE_IDS, values, strict=True)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _metric_triplet(
    historical_returns: pd.Series,
    corrected_returns: pd.Series,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        calculate_portfolio_metrics(historical_returns),
        calculate_portfolio_metrics(corrected_returns),
    )


def _assemble_accounting_tables(
    panel: pd.DataFrame,
    historical: PortfolioAllocationResults,
    corrected: FixedHoldingsPortfolioAllocationResults,
) -> tuple[pd.DataFrame, ...]:
    weights = _ordered_weights(historical)
    historical_net_panel = historical.copy_portfolio_return_panel()
    corrected_net_panel = corrected.portfolio_results.copy_portfolio_return_panel()
    corrected_path = corrected.copy_fixed_holdings_weight_path().sort_values(
        ["session_date", "fold_id", "allocation_rule", "sleeve_order"],
        kind="stable",
    ).reset_index(drop=True)
    ending = corrected.copy_ending_fold_weights().sort_values(
        ["fold_id", "allocation_rule", "sleeve_order"], kind="stable"
    ).reset_index(drop=True)
    historical_diagnostics = historical.copy_allocation_diagnostics().set_index(
        ["fold_id", "allocation_rule"]
    )
    corrected_diagnostics = (
        corrected.portfolio_results.copy_allocation_diagnostics().set_index(
            ["fold_id", "allocation_rule"]
        )
    )

    fold_records: list[dict[str, object]] = []
    turnover_records: list[dict[str, object]] = []
    drift_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    wealth_records: list[dict[str, object]] = []
    return_records: list[dict[str, object]] = []
    historical_gross_by_rule: dict[str, list[pd.Series]] = {
        rule: [] for rule in ALLOCATION_RULES
    }
    corrected_gross_by_rule: dict[str, list[pd.Series]] = {
        rule: [] for rule in ALLOCATION_RULES
    }

    fold_ids = tuple(fold.fold_id for fold in build_walk_forward_folds())
    for fold_number, fold_id in enumerate(fold_ids):
        session_rows = historical_net_panel.loc[
            historical_net_panel["fold_id"].eq(fold_id)
        ]
        corrected_session_rows = corrected_net_panel.loc[
            corrected_net_panel["fold_id"].eq(fold_id)
        ]
        historical_sessions = pd.DatetimeIndex(
            pd.to_datetime(session_rows["session_date"], utc=True),
            name="session_date",
        )
        corrected_sessions = pd.DatetimeIndex(
            pd.to_datetime(corrected_session_rows["session_date"], utc=True),
            name="session_date",
        )
        if not historical_sessions.equals(corrected_sessions):
            raise Day29FixedHoldingsExperimentError(
                "Historical and corrected test sessions differ."
            )
        test_panel = panel.loc[historical_sessions, list(SLEEVE_IDS)].copy(deep=True)
        for rule in ALLOCATION_RULES:
            target = _target_vector(weights, fold_id=fold_id, rule=rule)
            historical_gross = pd.Series(
                test_panel.to_numpy(dtype="float64", copy=True) @ target,
                index=historical_sessions,
                name=rule,
            )
            rule_path = corrected_path.loc[
                corrected_path["fold_id"].eq(fold_id)
                & corrected_path["allocation_rule"].eq(rule)
            ].copy(deep=True)
            if len(rule_path) != len(historical_sessions) * len(SLEEVE_IDS):
                raise Day29FixedHoldingsExperimentError(
                    "Corrected weight path is incomplete."
                )
            pre_matrix = rule_path.pivot(
                index="session_date",
                columns="sleeve_id",
                values="pre_return_weight",
            ).loc[historical_sessions, list(SLEEVE_IDS)]
            post_matrix = rule_path.pivot(
                index="session_date",
                columns="sleeve_id",
                values="post_return_weight",
            ).loc[historical_sessions, list(SLEEVE_IDS)]
            corrected_gross_values = np.sum(
                pre_matrix.to_numpy(dtype="float64")
                * test_panel.to_numpy(dtype="float64"),
                axis=1,
            )
            corrected_gross = pd.Series(
                corrected_gross_values,
                index=historical_sessions,
                name=rule,
            )
            historical_net = pd.Series(
                session_rows[rule].to_numpy(dtype="float64", copy=True),
                index=historical_sessions,
                name=rule,
            )
            corrected_net = pd.Series(
                corrected_session_rows[rule].to_numpy(dtype="float64", copy=True),
                index=historical_sessions,
                name=rule,
            )
            historical_turnover = float(
                historical_diagnostics.loc[(fold_id, rule), "allocation_turnover"]
            )
            corrected_turnover = float(
                corrected_diagnostics.loc[(fold_id, rule), "allocation_turnover"]
            )
            historical_cost = float(
                historical_diagnostics.loc[(fold_id, rule), "allocation_cost"]
            )
            corrected_cost = float(
                corrected_diagnostics.loc[(fold_id, rule), "allocation_cost"]
            )
            expected_historical_net = historical_gross.to_numpy(copy=True)
            expected_corrected_net = corrected_gross.to_numpy(copy=True)
            expected_historical_net[0] -= historical_cost
            expected_corrected_net[0] -= corrected_cost
            if not np.allclose(
                expected_historical_net,
                historical_net.to_numpy(),
                rtol=0.0,
                atol=1e-15,
            ) or not np.allclose(
                expected_corrected_net,
                corrected_net.to_numpy(),
                rtol=0.0,
                atol=1e-15,
            ):
                raise Day29FixedHoldingsExperimentError(
                    "Allocation cost was not charged only on the first fold session."
                )

            recursive_wealth = float(np.prod(1.0 + corrected_gross.to_numpy()))
            holdings_wealth = float(
                target
                @ np.prod(
                    1.0 + test_panel.to_numpy(dtype="float64", copy=True), axis=0
                )
            )
            wealth_residual = recursive_wealth - holdings_wealth
            within_tolerance = math.isclose(
                recursive_wealth,
                holdings_wealth,
                rel_tol=WEALTH_IDENTITY_TOLERANCE,
                abs_tol=WEALTH_IDENTITY_TOLERANCE,
            )
            if not within_tolerance:
                raise Day29FixedHoldingsExperimentError(
                    "Corrected gross terminal-wealth identity failed."
                )
            wealth_records.append(
                {
                    "fold_id": fold_id,
                    "allocation_rule": rule,
                    "sessions": int(len(historical_sessions)),
                    "recursive_gross_terminal_wealth": recursive_wealth,
                    "holdings_gross_terminal_wealth": holdings_wealth,
                    "gross_wealth_identity_residual": wealth_residual,
                    "absolute_gross_wealth_identity_residual": abs(wealth_residual),
                    "tolerance": WEALTH_IDENTITY_TOLERANCE,
                    "identity_within_tolerance": True,
                }
            )

            ending_rows = ending.loc[
                ending["fold_id"].eq(fold_id)
                & ending["allocation_rule"].eq(rule)
            ].sort_values("sleeve_order", kind="stable")
            corrected_ending = ending_rows["ending_weight"].to_numpy(
                dtype="float64", copy=True
            )
            if tuple(ending_rows["sleeve_id"]) != SLEEVE_IDS:
                raise Day29FixedHoldingsExperimentError(
                    "Corrected ending weights have changed sleeve order."
                )

            historical_gross_metrics, corrected_gross_metrics = _metric_triplet(
                historical_gross, corrected_gross
            )
            historical_net_metrics, corrected_net_metrics = _metric_triplet(
                historical_net, corrected_net
            )
            fold_records.append(
                {
                    "fold_id": fold_id,
                    "allocation_rule": rule,
                    "observations": int(len(historical_sessions)),
                    "start_session": historical_sessions.min(),
                    "end_session": historical_sessions.max(),
                    "target_weights": _json_vector(target),
                    "historical_ending_weights": _json_vector(target),
                    "corrected_ending_weights": _json_vector(corrected_ending),
                    "historical_cumulative_gross_return": historical_gross_metrics[
                        "cumulative_return"
                    ],
                    "corrected_cumulative_gross_return": corrected_gross_metrics[
                        "cumulative_return"
                    ],
                    "corrected_minus_historical_cumulative_gross_return": (
                        corrected_gross_metrics["cumulative_return"]
                        - historical_gross_metrics["cumulative_return"]
                    ),
                    "historical_cumulative_net_return_1bp": historical_net_metrics[
                        "cumulative_return"
                    ],
                    "corrected_cumulative_net_return_1bp": corrected_net_metrics[
                        "cumulative_return"
                    ],
                    "corrected_minus_historical_cumulative_net_return_1bp": (
                        corrected_net_metrics["cumulative_return"]
                        - historical_net_metrics["cumulative_return"]
                    ),
                    "historical_annualized_net_volatility_1bp": historical_net_metrics[
                        "annualized_volatility"
                    ],
                    "corrected_annualized_net_volatility_1bp": corrected_net_metrics[
                        "annualized_volatility"
                    ],
                    "corrected_minus_historical_annualized_net_volatility_1bp": (
                        corrected_net_metrics["annualized_volatility"]
                        - historical_net_metrics["annualized_volatility"]
                    ),
                    "historical_annualized_net_sharpe_1bp": historical_net_metrics[
                        "sharpe_ratio"
                    ],
                    "corrected_annualized_net_sharpe_1bp": corrected_net_metrics[
                        "sharpe_ratio"
                    ],
                    "corrected_minus_historical_annualized_net_sharpe_1bp": (
                        corrected_net_metrics["sharpe_ratio"]
                        - historical_net_metrics["sharpe_ratio"]
                    ),
                    "historical_maximum_net_drawdown_1bp": historical_net_metrics[
                        "maximum_drawdown"
                    ],
                    "corrected_maximum_net_drawdown_1bp": corrected_net_metrics[
                        "maximum_drawdown"
                    ],
                    "corrected_minus_historical_maximum_net_drawdown_1bp": (
                        corrected_net_metrics["maximum_drawdown"]
                        - historical_net_metrics["maximum_drawdown"]
                    ),
                    "historical_turnover": historical_turnover,
                    "corrected_turnover": corrected_turnover,
                    "corrected_minus_historical_turnover": (
                        corrected_turnover - historical_turnover
                    ),
                    "historical_transaction_cost": historical_cost,
                    "corrected_transaction_cost": corrected_cost,
                    "corrected_minus_historical_transaction_cost": (
                        corrected_cost - historical_cost
                    ),
                    "historical_gross_terminal_wealth": (
                        historical_gross_metrics["cumulative_return"] + 1.0
                    ),
                    "corrected_gross_terminal_wealth": recursive_wealth,
                    "corrected_minus_historical_gross_terminal_wealth": (
                        recursive_wealth
                        - (historical_gross_metrics["cumulative_return"] + 1.0)
                    ),
                    "historical_net_terminal_wealth": (
                        historical_net_metrics["cumulative_return"] + 1.0
                    ),
                    "corrected_net_terminal_wealth": (
                        corrected_net_metrics["cumulative_return"] + 1.0
                    ),
                    "corrected_minus_historical_net_terminal_wealth": (
                        corrected_net_metrics["cumulative_return"]
                        - historical_net_metrics["cumulative_return"]
                    ),
                    "corrected_gross_wealth_identity_residual": wealth_residual,
                }
            )

            turnover_records.append(
                {
                    "fold_id": fold_id,
                    "allocation_rule": rule,
                    "historical_turnover": historical_turnover,
                    "corrected_turnover": corrected_turnover,
                    "corrected_minus_historical_turnover": (
                        corrected_turnover - historical_turnover
                    ),
                    "historical_transaction_cost": historical_cost,
                    "corrected_transaction_cost": corrected_cost,
                    "corrected_minus_historical_transaction_cost": (
                        corrected_cost - historical_cost
                    ),
                    "historical_previous_reference": (
                        "zero_vector" if fold_number == 0 else "prior_fold_target"
                    ),
                    "corrected_previous_reference": (
                        "zero_vector"
                        if fold_number == 0
                        else "prior_fold_ending_drifted_weights"
                    ),
                }
            )

            absolute_pre = np.abs(pre_matrix.to_numpy() - target)
            absolute_post = np.abs(post_matrix.to_numpy() - target)
            maximum_drift = np.maximum(
                absolute_pre.max(axis=0), absolute_post.max(axis=0)
            )
            ending_difference = corrected_ending - target
            fold_l1 = float(np.abs(ending_difference).sum())
            fold_maximum = float(np.abs(ending_difference).max())
            above_cap = bool(
                np.greater(pre_matrix.to_numpy(), MAXIMUM_WEIGHT).any()
                or np.greater(post_matrix.to_numpy(), MAXIMUM_WEIGHT).any()
            )
            cap_status = (
                "expected_fixed_holdings_drift_above_target_cap"
                if above_cap
                else "all_drifted_weights_at_or_below_target_cap"
            )
            for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
                index = sleeve_order - 1
                drift_records.append(
                    {
                        "fold_id": fold_id,
                        "allocation_rule": rule,
                        "sleeve_id": sleeve_id,
                        "sleeve_order": sleeve_order,
                        "target_weight": float(target[index]),
                        "ending_drifted_weight": float(corrected_ending[index]),
                        "ending_minus_target_weight": float(ending_difference[index]),
                        "absolute_ending_drift": float(
                            abs(ending_difference[index])
                        ),
                        "maximum_absolute_intrafold_drift": float(
                            maximum_drift[index]
                        ),
                        "minimum_observed_pre_return_weight": float(
                            pre_matrix.iloc[:, index].min()
                        ),
                        "maximum_observed_pre_return_weight": float(
                            pre_matrix.iloc[:, index].max()
                        ),
                        "fold_l1_ending_drift": fold_l1,
                        "fold_maximum_sleeve_drift": fold_maximum,
                        "any_drifted_weight_above_target_cap": above_cap,
                        "target_cap_status": cap_status,
                    }
                )

            for row in rule_path.itertuples(index=False):
                target_weight = float(target[int(row.sleeve_order) - 1])
                pre_above = bool(row.pre_return_weight > MAXIMUM_WEIGHT)
                post_above = bool(row.post_return_weight > MAXIMUM_WEIGHT)
                path_records.append(
                    {
                        "session_date": row.session_date,
                        "fold_id": row.fold_id,
                        "allocation_rule": row.allocation_rule,
                        "sleeve_id": row.sleeve_id,
                        "sleeve_order": int(row.sleeve_order),
                        "target_weight": target_weight,
                        "pre_return_weight": float(row.pre_return_weight),
                        "post_return_weight": float(row.post_return_weight),
                        "pre_return_minus_target": float(
                            row.pre_return_weight - target_weight
                        ),
                        "post_return_minus_target": float(
                            row.post_return_weight - target_weight
                        ),
                        "pre_return_weight_above_target_cap": pre_above,
                        "post_return_weight_above_target_cap": post_above,
                        "target_cap_status": (
                            "expected_fixed_holdings_drift_above_target_cap"
                            if pre_above or post_above
                            else "at_or_below_target_cap"
                        ),
                    }
                )

            for row_number, session_date in enumerate(historical_sessions):
                return_records.append(
                    {
                        "session_date": session_date,
                        "fold_id": fold_id,
                        "allocation_rule": rule,
                        "historical_gross_return": float(
                            historical_gross.iloc[row_number]
                        ),
                        "corrected_gross_return": float(
                            corrected_gross.iloc[row_number]
                        ),
                        "corrected_minus_historical_gross_return": float(
                            corrected_gross.iloc[row_number]
                            - historical_gross.iloc[row_number]
                        ),
                        "historical_net_return_1bp": float(
                            historical_net.iloc[row_number]
                        ),
                        "corrected_net_return_1bp": float(
                            corrected_net.iloc[row_number]
                        ),
                        "corrected_minus_historical_net_return_1bp": float(
                            corrected_net.iloc[row_number]
                            - historical_net.iloc[row_number]
                        ),
                        "historical_cost_charged": (
                            historical_cost if row_number == 0 else 0.0
                        ),
                        "corrected_cost_charged": (
                            corrected_cost if row_number == 0 else 0.0
                        ),
                        "is_first_fold_session": row_number == 0,
                    }
                )
            historical_gross_by_rule[rule].append(historical_gross)
            corrected_gross_by_rule[rule].append(corrected_gross)

    return_comparison = pd.DataFrame.from_records(
        return_records, columns=RETURN_COMPARISON_COLUMNS
    ).sort_values(
        ["session_date", "fold_id", "allocation_rule"], kind="stable"
    ).reset_index(drop=True)
    aggregate_records: list[dict[str, object]] = []
    for rule in ALLOCATION_RULES:
        historical_gross = pd.concat(historical_gross_by_rule[rule])
        corrected_gross = pd.concat(corrected_gross_by_rule[rule])
        historical_net = pd.Series(
            historical_net_panel[rule].to_numpy(dtype="float64", copy=True),
            index=pd.to_datetime(historical_net_panel["session_date"], utc=True),
        )
        corrected_net = pd.Series(
            corrected_net_panel[rule].to_numpy(dtype="float64", copy=True),
            index=pd.to_datetime(corrected_net_panel["session_date"], utc=True),
        )
        historical_gross_metrics, corrected_gross_metrics = _metric_triplet(
            historical_gross, corrected_gross
        )
        historical_net_metrics, corrected_net_metrics = _metric_triplet(
            historical_net, corrected_net
        )
        historical_turnover = float(
            historical_diagnostics.xs(rule, level="allocation_rule")[
                "allocation_turnover"
            ].sum()
        )
        corrected_turnover = float(
            corrected_diagnostics.xs(rule, level="allocation_rule")[
                "allocation_turnover"
            ].sum()
        )
        historical_cost = float(
            historical_diagnostics.xs(rule, level="allocation_rule")[
                "allocation_cost"
            ].sum()
        )
        corrected_cost = float(
            corrected_diagnostics.xs(rule, level="allocation_rule")[
                "allocation_cost"
            ].sum()
        )
        gross_effect = float(
            corrected_gross_metrics["cumulative_return"]
            - historical_gross_metrics["cumulative_return"]
        )
        net_effect = float(
            corrected_net_metrics["cumulative_return"]
            - historical_net_metrics["cumulative_return"]
        )
        aggregate_records.append(
            {
                "allocation_rule": rule,
                "folds": len(fold_ids),
                "sessions": int(len(historical_net)),
                "start_session": historical_net.index.min(),
                "end_session": historical_net.index.max(),
                "historical_cumulative_gross_return": historical_gross_metrics[
                    "cumulative_return"
                ],
                "corrected_cumulative_gross_return": corrected_gross_metrics[
                    "cumulative_return"
                ],
                "corrected_minus_historical_cumulative_gross_return": gross_effect,
                "historical_cumulative_net_return_1bp": historical_net_metrics[
                    "cumulative_return"
                ],
                "corrected_cumulative_net_return_1bp": corrected_net_metrics[
                    "cumulative_return"
                ],
                "corrected_minus_historical_cumulative_net_return_1bp": net_effect,
                "historical_annualized_net_volatility_1bp": historical_net_metrics[
                    "annualized_volatility"
                ],
                "corrected_annualized_net_volatility_1bp": corrected_net_metrics[
                    "annualized_volatility"
                ],
                "corrected_minus_historical_annualized_net_volatility_1bp": (
                    corrected_net_metrics["annualized_volatility"]
                    - historical_net_metrics["annualized_volatility"]
                ),
                "historical_annualized_net_sharpe_1bp": historical_net_metrics[
                    "sharpe_ratio"
                ],
                "corrected_annualized_net_sharpe_1bp": corrected_net_metrics[
                    "sharpe_ratio"
                ],
                "corrected_minus_historical_annualized_net_sharpe_1bp": (
                    corrected_net_metrics["sharpe_ratio"]
                    - historical_net_metrics["sharpe_ratio"]
                ),
                "historical_maximum_net_drawdown_1bp": historical_net_metrics[
                    "maximum_drawdown"
                ],
                "corrected_maximum_net_drawdown_1bp": corrected_net_metrics[
                    "maximum_drawdown"
                ],
                "corrected_minus_historical_maximum_net_drawdown_1bp": (
                    corrected_net_metrics["maximum_drawdown"]
                    - historical_net_metrics["maximum_drawdown"]
                ),
                "historical_total_turnover": historical_turnover,
                "corrected_total_turnover": corrected_turnover,
                "corrected_minus_historical_total_turnover": (
                    corrected_turnover - historical_turnover
                ),
                "historical_total_transaction_cost": historical_cost,
                "corrected_total_transaction_cost": corrected_cost,
                "corrected_minus_historical_total_transaction_cost": (
                    corrected_cost - historical_cost
                ),
                "return_accounting_effect": gross_effect,
                "turnover_effect": corrected_turnover - historical_turnover,
                "cost_effect": net_effect - gross_effect,
            }
        )

    return (
        pd.DataFrame.from_records(fold_records, columns=FOLD_PERFORMANCE_COLUMNS),
        pd.DataFrame.from_records(
            aggregate_records, columns=AGGREGATE_PERFORMANCE_COLUMNS
        ),
        pd.DataFrame.from_records(turnover_records, columns=FOLD_TURNOVER_COLUMNS),
        pd.DataFrame.from_records(drift_records, columns=ENDING_DRIFT_COLUMNS),
        pd.DataFrame.from_records(path_records, columns=CORRECTED_WEIGHT_PATH_COLUMNS)
        .sort_values(
            ["session_date", "fold_id", "allocation_rule", "sleeve_order"],
            kind="stable",
        )
        .reset_index(drop=True),
        pd.DataFrame.from_records(wealth_records, columns=WEALTH_IDENTITY_COLUMNS),
        return_comparison,
    )


def build_day29_experiment(
    panel: pd.DataFrame,
    *,
    source_dataset_path: str,
    source_sha256: str,
    comparator_snapshot: Mapping[str, str],
    generation_timestamp: str,
    require_canonical_counts: bool = False,
    require_exact_development_range: bool = False,
    historical_results: PortfolioAllocationResults | None = None,
    corrected_results: FixedHoldingsPortfolioAllocationResults | None = None,
) -> Day29FixedHoldingsExperimentResults:
    """Run both frozen accounting paths and assemble comparison evidence."""

    if not isinstance(source_dataset_path, str) or not source_dataset_path:
        raise TypeError("source_dataset_path must be a non-empty string.")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise Day29FixedHoldingsExperimentError(
            "source_sha256 must be a lowercase SHA-256 digest."
        )
    if not isinstance(generation_timestamp, str) or not generation_timestamp:
        raise TypeError("generation_timestamp must be a non-empty string.")
    validated = audit_development_panel(
        panel,
        require_exact_development_range=require_exact_development_range,
    )
    historical = (
        analyze_portfolio_allocation_panel(
            validated,
            require_canonical_counts=require_canonical_counts,
        )
        if historical_results is None
        else historical_results
    )
    corrected = (
        analyze_portfolio_allocation_panel_fixed_holdings(
            validated,
            require_canonical_counts=require_canonical_counts,
        )
        if corrected_results is None
        else corrected_results
    )
    invariance = validate_method_invariants(historical, corrected)
    (
        fold_performance,
        aggregate_performance,
        fold_turnover,
        ending_drift,
        corrected_weight_path,
        wealth_identity,
        return_comparison,
    ) = _assemble_accounting_tables(validated, historical, corrected)
    if tuple(aggregate_performance["allocation_rule"]) != ALLOCATION_RULES:
        raise Day29FixedHoldingsExperimentError(
            "Aggregate allocation-rule ordering changed."
        )
    fold_definitions = [
        {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.date().isoformat(),
            "train_end_exclusive": fold.train_end_exclusive.date().isoformat(),
            "test_start": fold.test_start.date().isoformat(),
            "test_end_exclusive": fold.test_end_exclusive.date().isoformat(),
        }
        for fold in build_walk_forward_folds()
    ]
    metadata: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "accounting_version": ACCOUNTING_VERSION,
        "generation_timestamp": generation_timestamp,
        "source_dataset_path": source_dataset_path,
        "source_dataset_sha256": source_sha256,
        "return_panel_construction": (
            "run_strategy_diversification(validated_canonical_bars)."
            "copy_session_return_panel()"
        ),
        "return_panel_sha256": hash_return_panel(validated),
        "timestamp_min": validated.index.min().isoformat(),
        "timestamp_max": validated.index.max().isoformat(),
        "row_count": int(len(validated)),
        "sleeves": list(SLEEVE_IDS),
        "upstream_frequency": FREQUENCY,
        "portfolio_frequency": "session",
        "folds": fold_definitions,
        "allocation_rules": list(ALLOCATION_RULES),
        "historical_method": "frozen_day16_daily_constant_mix",
        "corrected_method": ACCOUNTING_VERSION,
        "allocation_cost_rate": ALLOCATION_COST_RATE,
        "annualization_sessions": TRADING_SESSIONS_PER_YEAR,
        "maximum_target_weight": MAXIMUM_WEIGHT,
        "covariance_estimator": "LedoitWolf(assume_centered=False)",
        "comparator_source_hashes": dict(sorted(comparator_snapshot.items())),
        "development_only": True,
        "parameter_selection_performed": False,
        "winner_selected": False,
    }
    return Day29FixedHoldingsExperimentResults(
        source_and_method_metadata=metadata,
        target_and_covariance_invariance=invariance,
        fold_performance_comparison=fold_performance,
        aggregate_performance_comparison=aggregate_performance,
        fold_turnover_comparison=fold_turnover,
        ending_weight_drift=ending_drift,
        corrected_weight_path=corrected_weight_path,
        wealth_identity_checks=wealth_identity,
        portfolio_return_comparison=return_comparison,
        comparator_snapshot=comparator_snapshot,
    )


def _normalized_saved_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _assert_saved_csv_matches(
    actual: pd.DataFrame,
    saved_path: Path,
    *,
    numeric_columns: tuple[str, ...],
) -> None:
    saved = pd.read_csv(saved_path)
    if tuple(saved.columns) != tuple(actual.columns) or len(saved) != len(actual):
        raise Day29FixedHoldingsExperimentError(
            f"Frozen Day 16 comparator schema or row count changed: {saved_path}."
        )
    for column in actual.columns:
        if column in numeric_columns:
            left = pd.to_numeric(saved[column], errors="raise").to_numpy(
                dtype="float64"
            )
            right = pd.to_numeric(actual[column], errors="raise").to_numpy(
                dtype="float64"
            )
            if not np.allclose(
                left,
                right,
                rtol=1e-12,
                atol=1e-15,
                equal_nan=True,
            ):
                raise Day29FixedHoldingsExperimentError(
                    f"Frozen Day 16 numeric comparator differs: {saved_path}:{column}."
                )
        elif column in {"session_date", "start_session", "end_session"}:
            left = pd.to_datetime(saved[column], utc=True, errors="raise")
            right = pd.to_datetime(actual[column], utc=True, errors="raise")
            if not pd.DatetimeIndex(left).equals(pd.DatetimeIndex(right)):
                raise Day29FixedHoldingsExperimentError(
                    f"Frozen Day 16 date comparator differs: {saved_path}:{column}."
                )
        else:
            left = tuple(_normalized_saved_value(value) for value in saved[column])
            right = tuple(_normalized_saved_value(value) for value in actual[column])
            if left != right:
                raise Day29FixedHoldingsExperimentError(
                    f"Frozen Day 16 comparator differs: {saved_path}:{column}."
                )


def validate_historical_day16_comparators(
    historical: PortfolioAllocationResults,
    *,
    day16_directory: Path,
) -> None:
    """Prove the rebuilt historical path matches saved Day 16 evidence."""

    if not isinstance(day16_directory, Path):
        raise TypeError("day16_directory must be a pathlib.Path.")
    frame_contracts = (
        (
            historical.copy_allocation_weights(),
            "allocation_weights.csv",
            (
                "sleeve_order",
                "training_sessions",
                "weight",
                "weight_sum",
                "maximum_weight",
                "gross_weight",
            ),
        ),
        (
            historical.copy_allocation_diagnostics(),
            "allocation_diagnostics.csv",
            (
                "training_sessions",
                "test_sessions",
                "shrinkage_coefficient",
                "allocation_turnover",
                "allocation_cost",
                "weight_sum",
                "minimum_weight",
                "maximum_weight",
                "gross_weight",
                "herfindahl_concentration",
                "effective_sleeve_count",
            ),
        ),
        (
            historical.copy_fold_portfolio_performance(),
            "fold_portfolio_performance.csv",
            tuple(
                column
                for column in historical.fold_portfolio_performance.columns
                if column
                not in ("fold_id", "allocation_rule", "start_session", "end_session")
            ),
        ),
        (
            historical.copy_aggregate_portfolio_performance(),
            "aggregate_portfolio_performance.csv",
            tuple(
                column
                for column in historical.aggregate_portfolio_performance.columns
                if column
                not in ("allocation_rule", "start_session", "end_session")
            ),
        ),
        (
            historical.copy_portfolio_return_panel(),
            "portfolio_return_panel.csv",
            ALLOCATION_RULES,
        ),
    )
    for frame, filename, numeric_columns in frame_contracts:
        _assert_saved_csv_matches(
            frame,
            day16_directory / filename,
            numeric_columns=tuple(numeric_columns),
        )

    manifest = _read_json(day16_directory / "manifest.json")
    covariance_entries = manifest.get("minimum_variance_covariance_estimates")
    if not isinstance(covariance_entries, list):
        raise Day29FixedHoldingsExperimentError(
            "Day 16 manifest lacks covariance comparator evidence."
        )
    covariance = historical.copy_minimum_variance_covariances()
    diagnostics = historical.copy_allocation_diagnostics()
    for entry in covariance_entries:
        if not isinstance(entry, dict):
            raise Day29FixedHoldingsExperimentError(
                "Day 16 covariance comparator entry is invalid."
            )
        fold_id = str(entry["fold_id"])
        matrix = np.asarray(entry["covariance_matrix"], dtype="float64")
        actual_rows = covariance.loc[covariance["fold_id"].eq(fold_id)].sort_values(
            "sleeve_order", kind="stable"
        )
        actual_matrix = actual_rows.loc[:, list(SLEEVE_IDS)].to_numpy(
            dtype="float64"
        )
        if not np.allclose(matrix, actual_matrix, rtol=1e-12, atol=1e-15):
            raise Day29FixedHoldingsExperimentError(
                f"Frozen Day 16 covariance differs for {fold_id}."
            )
        actual_shrinkage = float(
            diagnostics.loc[
                diagnostics["fold_id"].eq(fold_id)
                & diagnostics["allocation_rule"].eq(
                    "constrained_minimum_variance"
                ),
                "shrinkage_coefficient",
            ].iloc[0]
        )
        if actual_shrinkage != float(entry["shrinkage_coefficient"]):
            raise Day29FixedHoldingsExperimentError(
                f"Frozen Day 16 shrinkage differs for {fold_id}."
            )


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Day29FixedHoldingsExperimentError(
            "Day 29 JSON metadata is not deterministically serializable."
        ) from exc


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def describe_existing_bundle(path: Path) -> str:
    """Return deterministic file names and hashes for an existing target."""

    if not path.exists():
        return ""
    if not path.is_dir():
        return f"{path}:not_a_directory"
    descriptions = []
    for item in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        descriptions.append(
            f"{item.name}:{sha256_file(item) if item.is_file() else 'directory'}"
        )
    return ", ".join(descriptions)


def write_day29_artifacts(
    results: Day29FixedHoldingsExperimentResults,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write one non-overwriting, machine-readable Day 29 evidence bundle."""

    if not isinstance(results, Day29FixedHoldingsExperimentResults):
        raise TypeError("results must be Day29FixedHoldingsExperimentResults.")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path.")
    if output_directory.name != OUTPUT_DIRECTORY_BASENAME:
        raise Day29FixedHoldingsExperimentError(
            f"Day 29 output-directory basename must be {OUTPUT_DIRECTORY_BASENAME}."
        )
    if output_directory.exists():
        raise FileExistsError(
            "Day 29 output directory already exists; refusing overwrite. "
            + describe_existing_bundle(output_directory)
        )
    verify_comparator_snapshot(results.comparator_snapshot)

    payloads: dict[str, bytes] = {
        SOURCE_METADATA_FILENAME: _json_bytes(results.source_and_method_metadata),
        INVARIANCE_FILENAME: _csv_bytes(
            results.target_and_covariance_invariance
        ),
        FOLD_PERFORMANCE_FILENAME: _csv_bytes(
            results.fold_performance_comparison
        ),
        AGGREGATE_PERFORMANCE_FILENAME: _csv_bytes(
            results.aggregate_performance_comparison
        ),
        FOLD_TURNOVER_FILENAME: _csv_bytes(results.fold_turnover_comparison),
        ENDING_DRIFT_FILENAME: _csv_bytes(results.ending_weight_drift),
        WEIGHT_PATH_FILENAME: _csv_bytes(results.corrected_weight_path),
        WEALTH_IDENTITY_FILENAME: _csv_bytes(results.wealth_identity_checks),
        RETURN_COMPARISON_FILENAME: _csv_bytes(
            results.portfolio_return_comparison
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    for filename in APPROVED_ARTIFACT_NAMES[:-1]:
        path = output_directory / filename
        path.write_bytes(payloads[filename])
        written.append(path)

    verify_comparator_snapshot(results.comparator_snapshot)
    comparator_after = {
        path: sha256_file(Path(path)) for path in sorted(results.comparator_snapshot)
    }
    artifact_hashes = {
        path.name: sha256_file(path) for path in written
    }
    manifest: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "accounting_version": ACCOUNTING_VERSION,
        "generation_timestamp": results.source_and_method_metadata[
            "generation_timestamp"
        ],
        "source_paths_and_hashes": {
            "development_dataset": {
                "path": results.source_and_method_metadata["source_dataset_path"],
                "sha256": results.source_and_method_metadata[
                    "source_dataset_sha256"
                ],
            },
            "development_return_panel": {
                "construction": results.source_and_method_metadata[
                    "return_panel_construction"
                ],
                "sha256": results.source_and_method_metadata[
                    "return_panel_sha256"
                ],
            },
        },
        "timestamp_boundary_checks": {
            "minimum": results.source_and_method_metadata["timestamp_min"],
            "maximum": results.source_and_method_metadata["timestamp_max"],
            "authorized_start": "2020-01-02",
            "authorized_end": "2025-12-31",
            "contains_2026_or_later": False,
        },
        "row_count": results.source_and_method_metadata["row_count"],
        "sleeves": results.source_and_method_metadata["sleeves"],
        "frequency": results.source_and_method_metadata["upstream_frequency"],
        "folds": results.source_and_method_metadata["folds"],
        "allocation_rules": results.source_and_method_metadata[
            "allocation_rules"
        ],
        "allocation_cost_rate": ALLOCATION_COST_RATE,
        "artifact_filenames": list(APPROVED_ARTIFACT_NAMES),
        "artifact_sha256": artifact_hashes,
        "comparator_hashes_before": dict(results.comparator_snapshot),
        "comparator_hashes_after": comparator_after,
        "comparator_hashes_unchanged": (
            dict(results.comparator_snapshot) == comparator_after
        ),
        "no_2026_observations_accessed": True,
        "no_holdout_runner_accessed": True,
        "no_broker_or_network_accessed": True,
        "no_report_generator_executed": True,
        "no_notebook_created": True,
        "no_parameter_selection_performed": True,
        "no_commit_or_push_performed": True,
    }
    manifest_path = output_directory / MANIFEST_FILENAME
    manifest_path.write_bytes(_json_bytes(manifest))
    written.append(manifest_path)
    if tuple(path.name for path in written) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Day 29 artifact contract is incomplete.")
    return tuple(written)
