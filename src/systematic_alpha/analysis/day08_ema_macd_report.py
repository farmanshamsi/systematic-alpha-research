"""Compact artifact writer for the Day 8 EMA/MACD baseline."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_alpha.analysis.ema_macd_baseline import (
    DAY08_ANNUALIZATION_FACTOR,
    DAY08_FORWARD_HORIZONS,
    EmaMacdBaselineAnalysis,
)


ARTIFACT_VERSION: Final[str] = "ema_macd_baseline_v1"

FROZEN_DEVELOPMENT_START: Final[date] = date(2020, 1, 2)
FROZEN_DEVELOPMENT_END: Final[date] = date(2025, 12, 31)

APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "metadata.json",
    "performance_summary.csv",
    "strategy_diagnostics.csv",
    "holding_diagnostics.csv",
    "cost_break_even.csv",
    "signal_validation.csv",
    "signal_buckets.csv",
    "cumulative_wealth.png",
    "position_and_signal.png",
    "findings.md",
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{64}$"
)


class Day08ReportError(ValueError):
    """Raised when Day 8 artifacts cannot be written safely."""


def _parse_frozen_date(
    value: str | date,
    *,
    name: str,
) -> date:
    """Parse one development-period boundary."""

    if isinstance(value, date):
        return value

    if not isinstance(value, str) or not value.strip():
        raise Day08ReportError(
            f"{name} must be an ISO date string or date object."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise Day08ReportError(
            f"{name} must use YYYY-MM-DD format."
        ) from exc


def _validate_metadata_inputs(
    *,
    dataset_identifier: str,
    dataset_manifest_sha256: str,
    development_start: str | date,
    development_end: str | date,
) -> tuple[date, date]:
    """Validate immutable data-lineage metadata."""

    if (
        not isinstance(dataset_identifier, str)
        or not dataset_identifier.strip()
    ):
        raise Day08ReportError(
            "dataset_identifier must be a non-empty string."
        )

    if not isinstance(dataset_manifest_sha256, str):
        raise Day08ReportError(
            "dataset_manifest_sha256 must be a SHA-256 string."
        )

    normalized_sha256 = dataset_manifest_sha256.strip().lower()

    if not _SHA256_PATTERN.fullmatch(normalized_sha256):
        raise Day08ReportError(
            "dataset_manifest_sha256 must contain exactly "
            "64 lowercase hexadecimal characters."
        )

    parsed_start = _parse_frozen_date(
        development_start,
        name="development_start",
    )
    parsed_end = _parse_frozen_date(
        development_end,
        name="development_end",
    )

    if parsed_start != FROZEN_DEVELOPMENT_START:
        raise Day08ReportError(
            "Day 8 development_start must remain 2020-01-02."
        )

    if parsed_end != FROZEN_DEVELOPMENT_END:
        raise Day08ReportError(
            "Day 8 development_end must remain 2025-12-31."
        )

    if parsed_start > parsed_end:
        raise Day08ReportError(
            "development_start cannot follow development_end."
        )

    return parsed_start, parsed_end


def _validate_analysis(
    analysis: EmaMacdBaselineAnalysis,
) -> None:
    """Validate the compact baseline analysis contract."""

    if not isinstance(analysis, EmaMacdBaselineAnalysis):
        raise Day08ReportError(
            "analysis must be an EmaMacdBaselineAnalysis object."
        )

    observations = analysis.strategy_bundle.observations

    if observations.empty:
        raise Day08ReportError(
            "EMA/MACD observations must not be empty."
        )

    required_observation_columns = (
        "timestamp",
        "symbol",
        "close",
        "normalized_macd_histogram",
        "signal",
        "position",
        "turnover",
        "gross_strategy_return",
        "net_strategy_return",
    )
    missing_columns = [
        column
        for column in required_observation_columns
        if column not in observations.columns
    ]

    if missing_columns:
        raise Day08ReportError(
            "EMA/MACD observations are missing required columns: "
            f"{missing_columns}."
        )

    if observations["symbol"].nunique(dropna=False) != 1:
        raise Day08ReportError(
            "Day 8 reporting requires exactly one symbol."
        )


def _json_safe(value: object) -> object:
    """Convert common NumPy and pandas scalar values to JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    raise TypeError(
        f"Unsupported JSON metadata type: {type(value).__name__}."
    )


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write deterministic UTF-8 JSON."""

    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one deterministic compact CSV."""

    if not isinstance(frame, pd.DataFrame):
        raise Day08ReportError(
            f"{path.name} source must be a pandas DataFrame."
        )

    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )


def _compound_wealth(
    returns: pd.Series,
) -> pd.Series:
    """Compound one complete simple-return series."""

    numeric = pd.to_numeric(
        returns,
        errors="raise",
    ).astype(float)

    if numeric.isna().any():
        raise Day08ReportError(
            "Wealth-curve returns must not contain missing values."
        )

    if not np.isfinite(numeric.to_numpy()).all():
        raise Day08ReportError(
            "Wealth-curve returns must contain finite values."
        )

    if numeric.le(-1.0).any():
        raise Day08ReportError(
            "Simple returns must be greater than -1.0."
        )

    return (1.0 + numeric).cumprod()


def _write_cumulative_wealth_figure(
    observations: pd.DataFrame,
    path: Path,
) -> None:
    """Write buy-and-hold, gross and net cumulative wealth."""

    timestamps = pd.to_datetime(
        observations["timestamp"],
        utc=True,
        errors="raise",
    )

    buy_and_hold_returns = observations[
        "close_to_close_simple_return"
    ].copy()

    missing_locations = np.flatnonzero(
        buy_and_hold_returns.isna().to_numpy()
    )

    if len(missing_locations):
        if not np.array_equal(
            missing_locations,
            np.array([0]),
        ):
            raise Day08ReportError(
                "Buy-and-hold returns may be missing only at "
                "the first observation."
            )
        buy_and_hold_returns.iloc[0] = 0.0

    wealth = pd.DataFrame(
        {
            "buy_and_hold": _compound_wealth(
                buy_and_hold_returns
            ),
            "ema_macd_gross": _compound_wealth(
                observations["gross_strategy_return"]
            ),
            "ema_macd_net": _compound_wealth(
                observations["net_strategy_return"]
            ),
        }
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))

    axis.plot(
        timestamps,
        wealth["buy_and_hold"],
        label="SPY buy-and-hold",
    )
    axis.plot(
        timestamps,
        wealth["ema_macd_gross"],
        label="EMA/MACD gross",
    )
    axis.plot(
        timestamps,
        wealth["ema_macd_net"],
        label="EMA/MACD net",
    )

    axis.set_title(
        "Day 8 EMA/MACD baseline cumulative wealth"
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Cumulative wealth")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_position_signal_figure(
    observations: pd.DataFrame,
    path: Path,
) -> None:
    """Write normalised histogram and delayed target position."""

    timestamps = pd.to_datetime(
        observations["timestamp"],
        utc=True,
        errors="raise",
    )

    figure, signal_axis = plt.subplots(
        figsize=(10, 5.5)
    )
    position_axis = signal_axis.twinx()

    signal_axis.plot(
        timestamps,
        observations["normalized_macd_histogram"],
        label="Normalised MACD histogram",
    )
    signal_axis.axhline(
        0.0,
        linewidth=0.8,
    )

    position_axis.step(
        timestamps,
        observations["position"],
        where="post",
        label="Delayed target position",
        alpha=0.55,
    )

    signal_axis.set_title(
        "Day 8 EMA/MACD signal and one-bar-delayed position"
    )
    signal_axis.set_xlabel("Timestamp")
    signal_axis.set_ylabel(
        "Normalised MACD histogram"
    )
    position_axis.set_ylabel("Target position")
    position_axis.set_yticks([-1, 0, 1])
    signal_axis.grid(True, alpha=0.25)

    signal_handles, signal_labels = (
        signal_axis.get_legend_handles_labels()
    )
    position_handles, position_labels = (
        position_axis.get_legend_handles_labels()
    )

    signal_axis.legend(
        signal_handles + position_handles,
        signal_labels + position_labels,
        loc="best",
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _format_markdown_value(value: object) -> str:
    """Format one compact markdown-table value."""

    if pd.isna(value):
        return "NA"

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"

    return str(value).replace("|", r"\|")


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render a compact DataFrame without optional dependencies."""

    if frame.empty:
        return "_No observations available._"

    columns = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(value)
                for value in row
            )
            + " |"
        )

    return "\n".join(lines)


def _performance_value(
    performance_summary: pd.DataFrame,
    *,
    series: str,
    column: str,
) -> float:
    """Read one scalar performance result."""

    selected = performance_summary.loc[
        performance_summary["series"].eq(series),
        column,
    ]

    if len(selected) != 1:
        raise Day08ReportError(
            f"Expected one {series!r} performance row."
        )

    return float(selected.iloc[0])


def _write_findings(
    analysis: EmaMacdBaselineAnalysis,
    path: Path,
) -> None:
    """Write a conservative development-only findings report."""

    performance = analysis.performance_summary

    gross_return = _performance_value(
        performance,
        series="ema_macd_gross",
        column="cumulative_return",
    )
    net_return = _performance_value(
        performance,
        series="ema_macd_net",
        column="cumulative_return",
    )
    gross_sharpe = _performance_value(
        performance,
        series="ema_macd_gross",
        column="sharpe_ratio",
    )
    net_sharpe = _performance_value(
        performance,
        series="ema_macd_net",
        column="sharpe_ratio",
    )

    cost_effect = gross_return - net_return

    report = f"""# Day 8 EMA/MACD Baseline Findings

## Scope

This report evaluates one predeclared EMA/MACD baseline on the
development sample only. It is not parameter optimisation, walk-forward evidence,
cross-asset robustness, frequency robustness, locked-period evidence or
paper-trading evidence.

The frozen model uses 12-bar and 26-bar recursive EMAs, a 9-bar MACD signal
line, a normalised-histogram neutral band of 0.0005, one-bar-delayed target
positions and a cost assumption of one basis point per turnover unit.

## Performance summary

{_markdown_table(analysis.performance_summary)}

The gross cumulative return was {gross_return:.4%}, while the net cumulative
return was {net_return:.4%}. The difference attributable to the stated cost
model was {cost_effect:.4%} in cumulative-return units.

The gross Sharpe ratio was {gross_sharpe:.4f}; the net Sharpe ratio was
{net_sharpe:.4f}. These values are descriptive development results and
do not establish alpha, statistical significance, robustness or deployability.

## Strategy diagnostics

{_markdown_table(analysis.strategy_bundle.diagnostics)}

## Holding-period diagnostics

{_markdown_table(analysis.holding_diagnostics)}

## Model-implied cost break-even

{_markdown_table(analysis.cost_break_even)}

The break-even estimate is generated by compounded wealth under the declared
turnover convention. It is not an estimate of actual broker, spread, market
impact or slippage costs.

## Continuous-signal validation

{_markdown_table(analysis.signal_validation)}

Information coefficients and signal-bucket patterns are diagnostic only. They
must not be used to redefine the Day 8 baseline after observing these results.

## Interpretation constraints

A positive result would justify later bounded sensitivity and walk-forward
testing, not deployment. A negative result remains valid evidence and should
not be hidden by changing the frozen baseline.

The locked final period was not accessed and no parameter was selected using
locked-period information.
"""

    path.write_text(
        report,
        encoding="utf-8",
    )


def _build_metadata(
    analysis: EmaMacdBaselineAnalysis,
    *,
    dataset_identifier: str,
    dataset_manifest_sha256: str,
    development_start: date,
    development_end: date,
) -> dict[str, object]:
    """Build the immutable Day 8 metadata payload."""

    parameters = analysis.strategy_bundle.parameters
    observations = analysis.strategy_bundle.observations

    symbol = str(
        observations["symbol"].drop_duplicates().iloc[0]
    )

    return {
        "artifact_version": ARTIFACT_VERSION,
        "dataset_identifier": dataset_identifier.strip(),
        "dataset_manifest_sha256": (
            dataset_manifest_sha256.strip().lower()
        ),
        "development_sample_start": development_start,
        "development_sample_end": development_end,
        "locked_period_accessed": False,
        "parameter_selected_using_locked_period": False,
        "symbol": symbol,
        "timeframe": "15-minute",
        "price_column": parameters.price_column,
        "return_column": parameters.return_column,
        "ema_method": "recursive_adjust_false",
        "ema_seed": "first_available_observation",
        "ema_alpha_formula": "2 / (window + 1)",
        "fast_window": parameters.fast_window,
        "slow_window": parameters.slow_window,
        "signal_window": parameters.signal_window,
        "neutral_band": parameters.neutral_band,
        "continuous_signal": "normalized_macd_histogram",
        "signal_rule": (
            "+1 above neutral band; -1 below negative neutral "
            "band; 0 otherwise"
        ),
        "position_timing": "signal shifted by one bar",
        "overnight_positions_allowed": True,
        "ema_state_resets_at_session_boundary": False,
        "cost_bps_per_turnover": (
            parameters.cost_bps_per_turnover
        ),
        "turnover_definition": (
            "absolute change in one-bar-delayed target position"
        ),
        "direct_reversal_turnover": 2,
        "annualization_factor": DAY08_ANNUALIZATION_FACTOR,
        "risk_free_rate": 0.0,
        "signal_validation_horizons": (
            DAY08_FORWARD_HORIZONS
        ),
        "signal_bucket_count": 5,
        "cost_break_even_method": (
            "compounded simple-return wealth with bounded "
            "deterministic numerical root search"
        ),
        "full_bar_level_artifacts_written": False,
        "locked_final_period_start": "2026-01-02",
        "locked_final_period_end": "2026-06-30",
        "strategy_parameters": asdict(parameters),
    }


def _reject_unapproved_existing_files(
    output_directory: Path,
) -> None:
    """Reject unrelated pre-existing files in the artifact directory."""

    if not output_directory.exists():
        return

    unexpected = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
        and path.name not in APPROVED_ARTIFACT_NAMES
    )

    if unexpected:
        raise Day08ReportError(
            "Output directory contains unapproved files: "
            f"{unexpected}."
        )

    nested_directories = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_dir()
    )

    if nested_directories:
        raise Day08ReportError(
            "Output directory must not contain nested directories: "
            f"{nested_directories}."
        )


def _validate_written_artifacts(
    output_directory: Path,
) -> tuple[Path, ...]:
    """Prove the artifact directory contains exactly the approved files."""

    actual_names = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    )
    expected_names = sorted(APPROVED_ARTIFACT_NAMES)

    if actual_names != expected_names:
        raise Day08ReportError(
            "Day 8 artifact set does not match the approved contract. "
            f"Expected {expected_names}; received {actual_names}."
        )

    output_paths = tuple(
        output_directory / name
        for name in APPROVED_ARTIFACT_NAMES
    )

    empty_files = [
        path.name
        for path in output_paths
        if not path.exists() or path.stat().st_size <= 0
    ]

    if empty_files:
        raise Day08ReportError(
            f"Day 8 artifacts are missing or empty: {empty_files}."
        )

    forbidden_suffixes = {
        ".parquet",
        ".feather",
        ".pickle",
        ".pkl",
    }
    forbidden_files = [
        path.name
        for path in output_directory.iterdir()
        if path.suffix.lower() in forbidden_suffixes
    ]

    if forbidden_files:
        raise Day08ReportError(
            "Forbidden full-data artifacts were created: "
            f"{forbidden_files}."
        )

    return output_paths


def calculate_file_sha256(path: Path) -> str:
    """Calculate one artifact SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_ema_macd_baseline_artifacts(
    analysis: EmaMacdBaselineAnalysis,
    *,
    output_directory: str | Path,
    dataset_identifier: str,
    dataset_manifest_sha256: str,
    development_start: str | date = FROZEN_DEVELOPMENT_START,
    development_end: str | date = FROZEN_DEVELOPMENT_END,
) -> tuple[Path, ...]:
    """Write exactly the approved compact Day 8 artifacts."""

    _validate_analysis(analysis)

    parsed_start, parsed_end = _validate_metadata_inputs(
        dataset_identifier=dataset_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_start=development_start,
        development_end=development_end,
    )

    directory = Path(output_directory)

    _reject_unapproved_existing_files(directory)
    directory.mkdir(parents=True, exist_ok=True)

    observations = (
        analysis.strategy_bundle.observations
    )

    metadata = _build_metadata(
        analysis,
        dataset_identifier=dataset_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_start=parsed_start,
        development_end=parsed_end,
    )

    _write_json(
        directory / "metadata.json",
        metadata,
    )
    _write_csv(
        analysis.performance_summary,
        directory / "performance_summary.csv",
    )
    _write_csv(
        analysis.strategy_bundle.diagnostics,
        directory / "strategy_diagnostics.csv",
    )
    _write_csv(
        analysis.holding_diagnostics,
        directory / "holding_diagnostics.csv",
    )
    _write_csv(
        analysis.cost_break_even,
        directory / "cost_break_even.csv",
    )
    _write_csv(
        analysis.signal_validation,
        directory / "signal_validation.csv",
    )
    _write_csv(
        analysis.signal_buckets,
        directory / "signal_buckets.csv",
    )

    _write_cumulative_wealth_figure(
        observations,
        directory / "cumulative_wealth.png",
    )
    _write_position_signal_figure(
        observations,
        directory / "position_and_signal.png",
    )
    _write_findings(
        analysis,
        directory / "findings.md",
    )

    return _validate_written_artifacts(directory)
