"""Deterministic Day 17 reversion and inference report bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Final, Mapping

import pandas as pd

from systematic_alpha.analysis.reversion_inference import (
    AGGREGATE_PERFORMANCE_COLUMNS,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIGURATIONS,
    CONFIGURATION_IDS,
    COST_SENSITIVITY_COLUMNS,
    COST_STRESS_BPS,
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FOLD_PERFORMANCE_COLUMNS,
    HAC_LAGS,
    INFERENCE_COLUMNS,
    REPORTED_SERIES,
    REQUIRED_SYMBOLS,
    RETURN_PANEL_COLUMNS,
    SIGNAL_DIAGNOSTIC_COLUMNS,
    ReversionInferenceResults,
)
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds


SIGNAL_DIAGNOSTICS_FILENAME: Final[str] = "signal_diagnostics.csv"
FOLD_PERFORMANCE_FILENAME: Final[str] = "fold_performance.csv"
AGGREGATE_PERFORMANCE_FILENAME: Final[str] = "aggregate_performance.csv"
INFERENCE_RESULTS_FILENAME: Final[str] = "inference_results.csv"
COST_SENSITIVITY_FILENAME: Final[str] = "cost_sensitivity.csv"
SESSION_RETURN_PANEL_FILENAME: Final[str] = "session_return_panel.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"

APPROVED_DAY17_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SIGNAL_DIAGNOSTICS_FILENAME,
    FOLD_PERFORMANCE_FILENAME,
    AGGREGATE_PERFORMANCE_FILENAME,
    INFERENCE_RESULTS_FILENAME,
    COST_SENSITIVITY_FILENAME,
    SESSION_RETURN_PANEL_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)


class Day17ReportError(ValueError):
    """Raised when a Day 17 report violates its frozen contract."""


def _copy_table(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if tuple(frame.columns) != columns:
        raise Day17ReportError(
            f"{name} columns must exactly match the frozen schema."
        )
    if frame.empty:
        raise Day17ReportError(f"{name} cannot be empty.")
    return frame.copy(deep=True).reset_index(drop=True)


def _expected_keys() -> dict[str, list[tuple[object, ...]]]:
    folds = tuple(item.fold_id for item in build_walk_forward_folds())
    return {
        "signal_diagnostics": [
            (configuration, fold, symbol)
            for configuration in CONFIGURATION_IDS
            for fold in folds
            for symbol in REQUIRED_SYMBOLS
        ],
        "fold_performance": [
            (configuration, fold, series)
            for configuration in CONFIGURATION_IDS
            for fold in folds
            for series in REPORTED_SERIES
        ],
        "aggregate_performance": [
            (configuration, series)
            for configuration in CONFIGURATION_IDS
            for series in REPORTED_SERIES
        ],
        "inference_results": [
            (configuration, series)
            for configuration in CONFIGURATION_IDS
            for series in REPORTED_SERIES
        ],
        "cost_sensitivity": [
            (configuration, series, cost)
            for configuration in CONFIGURATION_IDS
            for series in REPORTED_SERIES
            for cost in COST_STRESS_BPS
        ],
    }


def _validate_order(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    expected: list[tuple[object, ...]],
    name: str,
) -> None:
    observed = list(frame.loc[:, columns].itertuples(index=False, name=None))
    if observed != expected:
        raise Day17ReportError(f"{name} row order or key coverage is invalid.")


def _format(value: object, *, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _render_report(
    aggregate: pd.DataFrame,
    inference: pd.DataFrame,
    costs: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# Day 17 OU/VWAP Reversion and Statistical Inference",
        "",
        "## Scope",
        "",
        "This is a development-only evaluation of three predeclared sensitivity "
        "calibrations on SPY, QQQ, and IWM. It uses rolling volume-weighted "
        "price residuals, rolling OU diagnostics, a variance-ratio regime gate, "
        "one-bar execution delay, forced overnight flatness, and explicit costs.",
        "",
        "No calibration or asset is ranked, selected, or authorized for paper "
        "trading. Locked 2026 data were not accessed.",
        "",
        "## Aggregate walk-forward performance",
        "",
        "| Configuration | Series | Cumulative return | Annualized return | "
        "Sharpe | Maximum drawdown | Turnover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.configuration_id} | {row.series} | "
            f"{_format(row.cumulative_return)} | {_format(row.annualized_return)} | "
            f"{_format(row.sharpe_ratio)} | {_format(row.maximum_drawdown)} | "
            f"{_format(row.turnover, digits=2)} |"
        )
    lines.extend(
        [
            "",
            "## Statistical inference",
            "",
            "| Configuration | Series | HAC t-stat | Mean bootstrap 95% CI | "
            "IC | PSR | DSR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inference.itertuples(index=False):
        interval = (
            f"[{_format(row.bootstrap_mean_ci_lower)}, "
            f"{_format(row.bootstrap_mean_ci_upper)}]"
        )
        lines.append(
            f"| {row.configuration_id} | {row.series} | "
            f"{_format(row.hac_t_statistic)} | {interval} | "
            f"{_format(row.information_coefficient)} | "
            f"{_format(row.probabilistic_sharpe_probability)} | "
            f"{_format(row.deflated_sharpe_probability)} |"
        )
    base_cost = costs.loc[costs["cost_bps_per_turnover"].eq(1.0)]
    severe_cost = costs.loc[costs["cost_bps_per_turnover"].eq(5.0)]
    merged = base_cost.merge(
        severe_cost,
        on=["configuration_id", "series"],
        suffixes=("_1bp", "_5bp"),
        validate="one_to_one",
    )
    lines.extend(
        [
            "",
            "## Cost stress",
            "",
            "| Configuration | Series | Cumulative return at 1 bp | "
            "Cumulative return at 5 bp |",
            "|---|---|---:|---:|",
        ]
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.configuration_id} | {row.series} | "
            f"{_format(row.cumulative_return_1bp)} | "
            f"{_format(row.cumulative_return_5bp)} |"
        )
    lines.extend(
        [
            "",
            "## Execution and leakage controls",
            "",
            f"- Initial non-flat positions: "
            f"{int(diagnostics['initial_position'].ne(0).sum())}.",
            f"- Initial non-zero turnover rows: "
            f"{int(diagnostics['initial_turnover'].ne(0.0).sum())}.",
            f"- Overnight position violations: "
            f"{int(diagnostics['overnight_position_violations'].sum())}.",
            "- Training history warms rolling indicators, while execution state "
            "resets at every test boundary.",
            "- Undefined statistics remain reported as N/A.",
            "",
            "## Interpretation boundary",
            "",
            "The evidence tests whether the proposed reversion mechanism is "
            "statistically and economically defensible. Profitability is not an "
            "acceptance condition. Weak or negative findings are retained and do "
            "not trigger calibration replacement.",
            "",
        ]
    )
    return "\n".join(lines)


def _configuration_manifest() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in CONFIGURATIONS:
        records.append(
            {
                "configuration_id": item.configuration_id,
                "reference_window": item.reference_window,
                "ou_window": item.ou_window,
                "variance_ratio_lag": item.variance_ratio_lag,
                "variance_ratio_threshold": item.variance_ratio_threshold,
                "entry_threshold": item.entry_threshold,
                "exit_threshold": item.exit_threshold,
                "minimum_half_life": item.minimum_half_life,
                "maximum_half_life": item.maximum_half_life,
                "maximum_holding_bars": item.maximum_holding_bars,
                "cost_bps_per_turnover": item.cost_bps_per_turnover,
            }
        )
    return records


@dataclass(frozen=True, slots=True)
class Day17ReversionInferenceReport:
    signal_diagnostics: pd.DataFrame
    fold_performance: pd.DataFrame
    aggregate_performance: pd.DataFrame
    inference_results: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    session_return_panel: pd.DataFrame
    report: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "signal_diagnostics",
            "fold_performance",
            "aggregate_performance",
            "inference_results",
            "cost_sensitivity",
            "session_return_panel",
        ):
            object.__setattr__(
                self, name, getattr(self, name).copy(deep=True).reset_index(drop=True)
            )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def build_day17_reversion_inference_report(
    results: ReversionInferenceResults,
) -> Day17ReversionInferenceReport:
    """Validate results and build the deterministic in-memory report."""

    if not isinstance(results, ReversionInferenceResults):
        raise TypeError("results must be ReversionInferenceResults.")
    diagnostics = _copy_table(
        results.signal_diagnostics,
        name="signal_diagnostics",
        columns=SIGNAL_DIAGNOSTIC_COLUMNS,
    )
    fold = _copy_table(
        results.fold_performance,
        name="fold_performance",
        columns=FOLD_PERFORMANCE_COLUMNS,
    )
    aggregate = _copy_table(
        results.aggregate_performance,
        name="aggregate_performance",
        columns=AGGREGATE_PERFORMANCE_COLUMNS,
    )
    inference = _copy_table(
        results.inference_results,
        name="inference_results",
        columns=INFERENCE_COLUMNS,
    )
    costs = _copy_table(
        results.cost_sensitivity,
        name="cost_sensitivity",
        columns=COST_SENSITIVITY_COLUMNS,
    )
    panel = _copy_table(
        results.session_return_panel,
        name="session_return_panel",
        columns=RETURN_PANEL_COLUMNS,
    )
    expected = _expected_keys()
    _validate_order(
        diagnostics,
        columns=("configuration_id", "fold_id", "symbol"),
        expected=expected["signal_diagnostics"],
        name="signal_diagnostics",
    )
    _validate_order(
        fold,
        columns=("configuration_id", "fold_id", "series"),
        expected=expected["fold_performance"],
        name="fold_performance",
    )
    _validate_order(
        aggregate,
        columns=("configuration_id", "series"),
        expected=expected["aggregate_performance"],
        name="aggregate_performance",
    )
    _validate_order(
        inference,
        columns=("configuration_id", "series"),
        expected=expected["inference_results"],
        name="inference_results",
    )
    _validate_order(
        costs,
        columns=("configuration_id", "series", "cost_bps_per_turnover"),
        expected=expected["cost_sensitivity"],
        name="cost_sensitivity",
    )
    if panel.duplicated(["configuration_id", "session_date"]).any():
        raise Day17ReportError("Session return panel keys must be unique.")
    panel_dates = pd.to_datetime(panel["session_date"], utc=True, errors="raise")
    if panel_dates.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise Day17ReportError("Session return panel crossed the development boundary.")
    if diagnostics["overnight_position_violations"].sum() != 0:
        raise Day17ReportError("Overnight positions are forbidden.")
    if diagnostics["initial_position"].ne(0).any() or diagnostics[
        "initial_turnover"
    ].ne(0.0).any():
        raise Day17ReportError("Every fold must begin flat and cost-free.")
    replication_values = inference["bootstrap_replications"].unique().tolist()
    if len(replication_values) != 1 or int(replication_values[0]) <= 0:
        raise Day17ReportError(
            "Inference rows must share one positive bootstrap replication count."
        )
    observed_replications = int(replication_values[0])

    report = _render_report(aggregate, inference, costs, diagnostics)
    manifest: dict[str, object] = {
        "schema_version": "day17-reversion-inference-v1",
        "analysis": "OU/VWAP reversion and statistical inference",
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        "locked_2026_accessed": False,
        "symbols": list(REQUIRED_SYMBOLS),
        "reported_series": list(REPORTED_SERIES),
        "configurations": _configuration_manifest(),
        "inference_contract": {
            "hac_lags": HAC_LAGS,
            "bootstrap_replications": observed_replications,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "declared_trials": len(CONFIGURATIONS),
        },
        "cost_stress_bps": list(COST_STRESS_BPS),
        "row_counts": {
            "signal_diagnostics": len(diagnostics),
            "fold_performance": len(fold),
            "aggregate_performance": len(aggregate),
            "inference_results": len(inference),
            "cost_sensitivity": len(costs),
            "session_return_panel": len(panel),
        },
        "selection_performed": False,
        "ranking_performed": False,
        "paper_trading_authorized": False,
    }
    return Day17ReversionInferenceReport(
        signal_diagnostics=diagnostics,
        fold_performance=fold,
        aggregate_performance=aggregate,
        inference_results=inference,
        cost_sensitivity=costs,
        session_return_panel=panel,
        report=report,
        manifest=manifest,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode("utf-8")


def _payloads(report: Day17ReversionInferenceReport) -> dict[str, bytes]:
    return {
        SIGNAL_DIAGNOSTICS_FILENAME: _csv_bytes(report.signal_diagnostics),
        FOLD_PERFORMANCE_FILENAME: _csv_bytes(report.fold_performance),
        AGGREGATE_PERFORMANCE_FILENAME: _csv_bytes(report.aggregate_performance),
        INFERENCE_RESULTS_FILENAME: _csv_bytes(report.inference_results),
        COST_SENSITIVITY_FILENAME: _csv_bytes(report.cost_sensitivity),
        SESSION_RETURN_PANEL_FILENAME: _csv_bytes(report.session_return_panel),
        REPORT_FILENAME: report.report.encode("utf-8"),
    }


def _manifest_bytes(
    report: Day17ReversionInferenceReport,
    payloads: Mapping[str, bytes],
) -> bytes:
    manifest = dict(report.manifest)
    manifest["artifacts"] = [
        {
            "filename": filename,
            "bytes": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in APPROVED_DAY17_ARTIFACT_NAMES
        if filename != MANIFEST_FILENAME
    ]
    return (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_day17_reversion_inference_artifacts(
    report: Day17ReversionInferenceReport,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly eight artifacts using sibling staging and replacement."""

    if not isinstance(report, Day17ReversionInferenceReport):
        raise TypeError("report must be Day17ReversionInferenceReport.")
    destination = Path(output_directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        payloads = _payloads(report)
        payloads[MANIFEST_FILENAME] = _manifest_bytes(report, payloads)
        if tuple(payloads) != APPROVED_DAY17_ARTIFACT_NAMES:
            raise RuntimeError("Day 17 payload allow-list mismatch.")
        for filename, payload in payloads.items():
            (staged / filename).write_bytes(payload)
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup-", dir=destination.parent
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(staged, destination)
        staged = Path()
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
            backup = None
        raise
    finally:
        if staged and staged.exists() and staged != Path():
            shutil.rmtree(staged)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)

    names = tuple(sorted(path.name for path in destination.iterdir()))
    if names != tuple(sorted(APPROVED_DAY17_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 17 artifact allow-list verification failed.")
    return tuple(destination / name for name in APPROVED_DAY17_ARTIFACT_NAMES)
