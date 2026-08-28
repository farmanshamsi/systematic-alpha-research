"""Compact artifact writing for Day 7 trend-ratio sensitivity analysis."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_alpha.analysis.trend_ratio_sensitivity import (
    BASELINE_COST_BPS_PER_TURNOVER,
    BREAK_EVEN_COST_LOWER_BPS,
    BREAK_EVEN_COST_UPPER_BPS,
    BREAK_EVEN_STATUS_ROOT_FOUND,
    DAY07_ANNUALIZATION_FACTOR,
    FORWARD_RETURN_HORIZONS,
    LONG_WINDOWS,
    NEUTRAL_BANDS,
    SHORT_WINDOWS,
    TrendRatioSensitivityTables,
)


DAY07_ARTIFACT_VERSION: Final[str] = (
    "trend_ratio_sensitivity_v1"
)

APPROVED_DAY07_ARTIFACT_FILENAMES: Final[frozenset[str]] = (
    frozenset(
        {
            "metadata.json",
            "parameter_results.csv",
            "annual_results.csv",
            "regime_results.csv",
            "holding_diagnostics.csv",
            "signal_validation.csv",
            "neighborhood_stability.csv",
            "net_sharpe_surface.png",
            "net_return_surface.png",
            "turnover_surface.png",
            "cost_break_even_surface.png",
            "stability_surface.png",
            "findings.md",
        }
    )
)

REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "artifact_version",
    "permitted_dataset_identifier",
    "dataset_manifest_sha256",
    "development_sample_start",
    "development_sample_end",
    "locked_period_accessed",
    "symbol",
    "timeframe",
    "price_column",
    "return_column",
    "averaging_method",
    "short_window_grid",
    "long_window_grid",
    "neutral_band_grid",
    "transaction_cost_convention",
    "baseline_cost_bps_per_turnover",
    "annualization_factor",
    "risk_free_rate_convention",
    "signal_timing",
    "position_timing",
    "holding_episode_definition",
    "whipsaw_definition",
    "cost_break_even_method",
    "volatility_regime_definition",
    "signal_validation_horizons",
    "neighborhood_definition",
    "parameter_selected_using_locked_period",
)

ABSOLUTE_PATH_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?:^|[\s\"'])/"
        r"(?:Users|home|tmp|var|mnt|opt)/",
        flags=re.IGNORECASE,
    ),
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
)


def _records_from_frame(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a compact frame to JSON-safe records."""

    payload = frame.to_json(
        orient="records",
        date_format="iso",
        double_precision=15,
    )
    records = json.loads(payload)

    if not isinstance(records, list):
        raise RuntimeError(
            "DataFrame record conversion did not produce a list."
        )

    return records


def _contains_absolute_local_path(value: object) -> bool:
    """Return whether nested text contains a machine-local absolute path."""

    if isinstance(value, Mapping):
        return any(
            _contains_absolute_local_path(key)
            or _contains_absolute_local_path(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_absolute_local_path(item)
            for item in value
        )

    if not isinstance(value, str):
        return False

    return any(
        pattern.search(value) is not None
        for pattern in ABSOLUTE_PATH_PATTERNS
    )


def build_day07_metadata(
    *,
    permitted_dataset_identifier: str,
    dataset_manifest_sha256: str,
    regime_definition: pd.DataFrame,
    symbol: str = "SPY",
    timeframe: str = "15-minute",
    price_column: str = "close",
    return_column: str = "close_to_close_simple_return",
) -> dict[str, Any]:
    """Build the frozen Day 7 metadata payload."""

    normalized_dataset_identifier = (
        str(permitted_dataset_identifier).strip()
    )
    normalized_hash = str(dataset_manifest_sha256).strip().lower()
    normalized_symbol = str(symbol).strip().upper()

    if not normalized_dataset_identifier:
        raise ValueError(
            "permitted_dataset_identifier must not be empty."
        )

    if _contains_absolute_local_path(
        normalized_dataset_identifier
    ):
        raise ValueError(
            "permitted_dataset_identifier must not contain "
            "an absolute local path."
        )

    if re.fullmatch(r"[0-9a-f]{64}", normalized_hash) is None:
        raise ValueError(
            "dataset_manifest_sha256 must be a 64-character "
            "lowercase hexadecimal SHA-256 value."
        )

    if normalized_symbol != "SPY":
        raise ValueError(
            "Day 7 primary analysis must use SPY."
        )

    if not isinstance(regime_definition, pd.DataFrame):
        raise TypeError(
            "regime_definition must be a pandas DataFrame."
        )

    if regime_definition.empty:
        raise ValueError(
            "regime_definition must contain the reused Day 5 "
            "volatility-regime definition."
        )

    metadata: dict[str, Any] = {
        "artifact_version": DAY07_ARTIFACT_VERSION,
        "permitted_dataset_identifier": (
            normalized_dataset_identifier
        ),
        "dataset_manifest_sha256": normalized_hash,
        "development_sample_start": "2020-01-02",
        "development_sample_end": "2025-12-31",
        "locked_period_accessed": False,
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "price_column": price_column,
        "return_column": return_column,
        "averaging_method": "arithmetic_simple_moving_average",
        "short_window_grid": list(SHORT_WINDOWS),
        "long_window_grid": list(LONG_WINDOWS),
        "neutral_band_grid": list(NEUTRAL_BANDS),
        "transaction_cost_convention": (
            "net_return = gross_return - turnover * "
            "cost_bps_per_turnover / 10000; "
            "direct reversal turnover equals 2"
        ),
        "baseline_cost_bps_per_turnover": (
            BASELINE_COST_BPS_PER_TURNOVER
        ),
        "annualization_factor": DAY07_ANNUALIZATION_FACTOR,
        "risk_free_rate_convention": "zero",
        "signal_timing": (
            "signal uses information available through bar t"
        ),
        "position_timing": (
            "position at bar t equals signal from bar t-1"
        ),
        "holding_episode_definition": (
            "maximal consecutive run of an identical non-zero "
            "position across the complete chronological sample; "
            "not reset at session or calendar-year boundaries"
        ),
        "whipsaw_definition": (
            "non-zero episode lasting no more than four bars, "
            "followed by the next opposite non-zero episode "
            "starting within four positional bars"
        ),
        "cost_break_even_method": {
            "objective": (
                "sum(log(1 + gross_return - turnover * "
                "cost_bps / 10000)) = 0"
            ),
            "solver": "scipy.optimize.brentq",
            "search_lower_bps": BREAK_EVEN_COST_LOWER_BPS,
            "search_upper_bps": BREAK_EVEN_COST_UPPER_BPS,
        },
        "volatility_regime_definition": (
            _records_from_frame(regime_definition)
        ),
        "signal_validation_horizons": list(
            FORWARD_RETURN_HORIZONS
        ),
        "neighborhood_definition": (
            "axis-adjacent configurations differing by exactly "
            "one declared grid step in one parameter while the "
            "other two parameters remain unchanged"
        ),
        "parameter_selected_using_locked_period": False,
    }

    validate_day07_metadata(metadata)

    return metadata


def validate_day07_metadata(
    metadata: Mapping[str, Any],
) -> None:
    """Validate Day 7 metadata and locked-period safeguards."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")

    missing_keys = [
        key
        for key in REQUIRED_METADATA_KEYS
        if key not in metadata
    ]

    if missing_keys:
        raise ValueError(
            "Day 7 metadata is missing required keys: "
            f"{missing_keys}."
        )

    if metadata["artifact_version"] != DAY07_ARTIFACT_VERSION:
        raise ValueError(
            "Unexpected Day 7 artifact version."
        )

    if metadata["development_sample_start"] != "2020-01-02":
        raise ValueError(
            "Day 7 development sample must start on 2020-01-02."
        )

    if metadata["development_sample_end"] != "2025-12-31":
        raise ValueError(
            "Day 7 development sample must end on 2025-12-31."
        )

    if metadata["locked_period_accessed"] is not False:
        raise ValueError(
            "locked_period_accessed must be false."
        )

    if (
        metadata["parameter_selected_using_locked_period"]
        is not False
    ):
        raise ValueError(
            "parameter_selected_using_locked_period must be false."
        )

    if list(metadata["short_window_grid"]) != list(
        SHORT_WINDOWS
    ):
        raise ValueError(
            "short_window_grid does not match the frozen grid."
        )

    if list(metadata["long_window_grid"]) != list(
        LONG_WINDOWS
    ):
        raise ValueError(
            "long_window_grid does not match the frozen grid."
        )

    if list(metadata["neutral_band_grid"]) != list(
        NEUTRAL_BANDS
    ):
        raise ValueError(
            "neutral_band_grid does not match the frozen grid."
        )

    if list(metadata["signal_validation_horizons"]) != list(
        FORWARD_RETURN_HORIZONS
    ):
        raise ValueError(
            "signal_validation_horizons do not match the "
            "frozen horizons."
        )

    manifest_hash = str(
        metadata["dataset_manifest_sha256"]
    ).strip().lower()

    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ValueError(
            "dataset_manifest_sha256 must be a valid SHA-256."
        )

    if _contains_absolute_local_path(metadata):
        raise ValueError(
            "Metadata must not contain absolute local paths."
        )


def _validate_tables(
    tables: TrendRatioSensitivityTables,
) -> None:
    """Validate compact full-grid inputs before artifact writing."""

    if not isinstance(tables, TrendRatioSensitivityTables):
        raise TypeError(
            "tables must be a TrendRatioSensitivityTables object."
        )

    if len(tables.parameter_results) != 36:
        raise ValueError(
            "parameter_results must contain exactly 36 configurations."
        )

    if (
        tables.parameter_results["configuration_id"].nunique()
        != 36
    ):
        raise ValueError(
            "parameter_results must contain 36 unique "
            "configuration identifiers."
        )

    required_nonempty = {
        "annual_results": tables.annual_results,
        "regime_results": tables.regime_results,
        "regime_definition": tables.regime_definition,
        "holding_diagnostics": tables.holding_diagnostics,
        "signal_validation": tables.signal_validation,
        "signal_buckets": tables.signal_buckets,
        "neighborhood_stability": (
            tables.neighborhood_stability
        ),
    }

    empty_names = [
        name
        for name, frame in required_nonempty.items()
        if not isinstance(frame, pd.DataFrame) or frame.empty
    ]

    if empty_names:
        raise ValueError(
            "Day 7 artifact tables must not be empty: "
            f"{empty_names}."
        )

    if len(tables.holding_diagnostics) != 36:
        raise ValueError(
            "holding_diagnostics must contain 36 rows."
        )

    if len(tables.neighborhood_stability) != 36:
        raise ValueError(
            "neighborhood_stability must contain 36 rows."
        )

    prohibited_columns = {
        "gross_strategy_return",
        "net_strategy_return",
        "position",
        "signal",
        "transaction_cost",
    }

    for name, frame in {
        "parameter_results": tables.parameter_results,
        **required_nonempty,
    }.items():
        present = sorted(
            prohibited_columns.intersection(frame.columns)
        )

        if present:
            raise ValueError(
                f"{name} contains prohibited bar-level columns: "
                f"{present}."
            )


def _build_signal_artifact(
    tables: TrendRatioSensitivityTables,
) -> pd.DataFrame:
    """Combine horizon summaries and signal buckets in one tidy CSV."""

    summary = tables.signal_validation.copy(deep=True)
    buckets = tables.signal_buckets.copy(deep=True)

    summary.insert(0, "record_type", "horizon_summary")
    buckets.insert(0, "record_type", "signal_bucket")

    combined = pd.concat(
        [summary, buckets],
        ignore_index=True,
        sort=False,
    )

    sort_columns = [
        column
        for column in (
            "configuration_id",
            "symbol",
            "horizon_bars",
            "record_type",
            "signal_bucket",
        )
        if column in combined.columns
    ]

    return combined.sort_values(
        sort_columns,
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def _surface_limits(
    parameter_results: pd.DataFrame,
    *,
    value_column: str,
) -> tuple[float, float]:
    """Return finite shared limits for all neutral-band panels."""

    values = pd.to_numeric(
        parameter_results[value_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return 0.0, 1.0

    lower = float(finite_values.min())
    upper = float(finite_values.max())

    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1e-6)
        return lower - padding, upper + padding

    return lower, upper


def _write_surface_figure(
    parameter_results: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    annotation_format: str,
) -> None:
    """Write four neutral-band panels for one sensitivity metric."""

    required_columns = {
        "short_window",
        "long_window",
        "neutral_band",
        value_column,
    }

    missing = sorted(
        required_columns.difference(
            parameter_results.columns
        )
    )

    if missing:
        raise ValueError(
            f"Surface input is missing columns: {missing}."
        )

    vmin, vmax = _surface_limits(
        parameter_results,
        value_column=value_column,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        constrained_layout=True,
    )
    flattened_axes = axes.ravel()
    image = None

    for axis, neutral_band in zip(
        flattened_axes,
        NEUTRAL_BANDS,
    ):
        panel = parameter_results.loc[
            parameter_results["neutral_band"].eq(
                neutral_band
            )
        ]

        pivot = (
            panel.pivot(
                index="long_window",
                columns="short_window",
                values=value_column,
            )
            .reindex(
                index=LONG_WINDOWS,
                columns=SHORT_WINDOWS,
            )
        )

        matrix = pivot.to_numpy(dtype=float)
        masked_matrix = np.ma.masked_invalid(matrix)

        image = axis.imshow(
            masked_matrix,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        axis.set_title(
            f"Neutral band = {neutral_band:.4f}"
        )
        axis.set_xlabel("Short window")
        axis.set_ylabel("Long window")
        axis.set_xticks(
            range(len(SHORT_WINDOWS)),
            labels=[str(value) for value in SHORT_WINDOWS],
        )
        axis.set_yticks(
            range(len(LONG_WINDOWS)),
            labels=[str(value) for value in LONG_WINDOWS],
        )

        for row_index, long_window in enumerate(
            LONG_WINDOWS
        ):
            for column_index, short_window in enumerate(
                SHORT_WINDOWS
            ):
                value = pivot.loc[
                    long_window,
                    short_window,
                ]

                text = (
                    "NA"
                    if pd.isna(value)
                    else format(
                        float(value),
                        annotation_format,
                    )
                )

                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    figure.suptitle(title)

    if image is not None:
        figure.colorbar(
            image,
            ax=flattened_axes.tolist(),
            shrink=0.82,
        )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Figure was not written correctly: {output_path.name}."
        )


def build_day07_findings(
    tables: TrendRatioSensitivityTables,
) -> str:
    """Generate a conservative development-only findings summary."""

    parameter_results = tables.parameter_results.copy(deep=True)
    stability = tables.neighborhood_stability.copy(deep=True)

    positive_net_count = int(
        parameter_results["net_cumulative_return"].gt(0.0).sum()
    )
    positive_break_even_count = int(
        (
            parameter_results["break_even_status"].eq(
                BREAK_EVEN_STATUS_ROOT_FOUND
            )
            & parameter_results["break_even_cost_bps"].gt(0.0)
        ).sum()
    )

    finite_sharpe = parameter_results.dropna(
        subset=["net_sharpe_ratio"]
    )
    finite_stability = stability.dropna(
        subset=["median_neighbor_net_sharpe"]
    )

    if finite_sharpe.empty:
        isolated_sentence = (
            "No configuration produced a finite net Sharpe estimate."
        )
    else:
        isolated = finite_sharpe.sort_values(
            "net_sharpe_ratio",
            ascending=False,
            kind="stable",
        ).iloc[0]

        isolated_sentence = (
            "The highest isolated net Sharpe occurred at "
            f"`{isolated['configuration_id']}` with net Sharpe "
            f"{isolated['net_sharpe_ratio']:.3f}. This is an "
            "in-sample diagnostic, not a selected final parameter."
        )

    if finite_stability.empty:
        stability_sentence = (
            "No finite neighbouring net-Sharpe median was available."
        )
    else:
        stable = finite_stability.sort_values(
            "median_neighbor_net_sharpe",
            ascending=False,
            kind="stable",
        ).iloc[0]

        stability_sentence = (
            "The highest median neighbouring net Sharpe occurred "
            f"around `{stable['configuration_id']}` at "
            f"{stable['median_neighbor_net_sharpe']:.3f}; its own "
            f"net Sharpe was {stable['net_sharpe_ratio']:.3f}."
        )

    median_turnover = float(
        parameter_results["total_turnover"].median()
    )

    findings = f"""# Day 7 Findings — Trend-Ratio Sensitivity

## Scope

This report covers the predeclared 36-configuration SPY 15-minute development grid only. The locked 2026-01-02 through 2026-06-30 period was not accessed.

## Development evidence

- Configurations with positive cumulative net return: {positive_net_count} of 36.
- Configurations with a positive model-implied break-even cost: {positive_break_even_count} of 36.
- Median total turnover across the grid: {median_turnover:.1f} turnover units.

{isolated_sentence}

{stability_sentence}

## Interpretation constraints

The results are development-only sensitivity evidence. They do not establish alpha, statistical superiority, robustness, deployability, realised execution costs, or final parameter validity. An isolated high-performing cell should be treated as weaker evidence than a coherent neighbouring plateau.

## Next evidence gate

Any configuration retained for further study must later pass parameter-stability review, walk-forward evaluation, anti-overfitting controls, and the separately locked final test.
"""

    if _contains_absolute_local_path(findings):
        raise RuntimeError(
            "Generated findings unexpectedly contain an absolute path."
        )

    return findings


def write_day07_artifacts(
    tables: TrendRatioSensitivityTables,
    *,
    metadata: Mapping[str, Any],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Write only the approved compact Day 7 artifact set."""

    _validate_tables(tables)
    validate_day07_metadata(metadata)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_unapproved = sorted(
        path.name
        for path in output_path.iterdir()
        if path.is_file()
        and path.name
        not in APPROVED_DAY07_ARTIFACT_FILENAMES
    )

    if existing_unapproved:
        raise ValueError(
            "Output directory contains unapproved files: "
            f"{existing_unapproved}."
        )

    table_map = {
        "parameter_results.csv": (
            tables.parameter_results.copy(deep=True)
        ),
        "annual_results.csv": (
            tables.annual_results.copy(deep=True)
        ),
        "regime_results.csv": (
            tables.regime_results.copy(deep=True)
        ),
        "holding_diagnostics.csv": (
            tables.holding_diagnostics.copy(deep=True)
        ),
        "signal_validation.csv": (
            _build_signal_artifact(tables)
        ),
        "neighborhood_stability.csv": (
            tables.neighborhood_stability.copy(deep=True)
        ),
    }

    for filename, frame in table_map.items():
        frame.to_csv(
            output_path / filename,
            index=False,
            float_format="%.12g",
            na_rep="",
        )

    metadata_text = json.dumps(
        dict(metadata),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )

    if _contains_absolute_local_path(metadata_text):
        raise ValueError(
            "Serialized metadata contains an absolute local path."
        )

    (output_path / "metadata.json").write_text(
        metadata_text + "\n",
        encoding="utf-8",
    )

    findings = build_day07_findings(tables)
    (output_path / "findings.md").write_text(
        findings,
        encoding="utf-8",
    )

    _write_surface_figure(
        tables.parameter_results,
        value_column="net_sharpe_ratio",
        title="Day 7 Net Sharpe Sensitivity",
        output_path=output_path / "net_sharpe_surface.png",
        annotation_format=".2f",
    )
    _write_surface_figure(
        tables.parameter_results,
        value_column="net_cumulative_return",
        title="Day 7 Net Cumulative Return Sensitivity",
        output_path=output_path / "net_return_surface.png",
        annotation_format=".2%",
    )
    _write_surface_figure(
        tables.parameter_results,
        value_column="total_turnover",
        title="Day 7 Turnover Sensitivity",
        output_path=output_path / "turnover_surface.png",
        annotation_format=".0f",
    )
    _write_surface_figure(
        tables.parameter_results,
        value_column="break_even_cost_bps",
        title="Day 7 Cost Break-Even Sensitivity",
        output_path=output_path / "cost_break_even_surface.png",
        annotation_format=".2f",
    )
    _write_surface_figure(
        tables.neighborhood_stability,
        value_column="median_neighbor_net_sharpe",
        title="Day 7 Median Neighbour Net Sharpe",
        output_path=output_path / "stability_surface.png",
        annotation_format=".2f",
    )

    written_files = tuple(
        sorted(
            (
                path
                for path in output_path.iterdir()
                if path.is_file()
            ),
            key=lambda path: path.name,
        )
    )

    written_names = {
        path.name
        for path in written_files
    }

    if written_names != APPROVED_DAY07_ARTIFACT_FILENAMES:
        missing = sorted(
            APPROVED_DAY07_ARTIFACT_FILENAMES
            - written_names
        )
        unexpected = sorted(
            written_names
            - APPROVED_DAY07_ARTIFACT_FILENAMES
        )
        raise RuntimeError(
            "Day 7 artifact set does not match the approved set. "
            f"Missing={missing}, unexpected={unexpected}."
        )

    for path in written_files:
        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Day 7 artifact is empty: {path.name}."
            )

    return written_files
