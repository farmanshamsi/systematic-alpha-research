"""Compact artifact writer for Day 9 EMA/MACD sensitivity analysis."""

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

from systematic_alpha.analysis.ema_macd_baseline import (
    DAY08_FORWARD_HORIZONS,
)
from systematic_alpha.analysis.ema_macd_sensitivity import (
    BASELINE_FAST_WINDOW,
    BASELINE_NEUTRAL_BAND,
    BASELINE_SIGNAL_WINDOW,
    BASELINE_SLOW_WINDOW,
    DAY09_ANNUALIZATION_FACTOR,
    DAY09_COST_BPS_PER_TURNOVER,
    EXPECTED_CONFIGURATION_COUNT,
    FAST_WINDOWS,
    NEUTRAL_BANDS,
    SIGNAL_WINDOWS,
    SLOW_WINDOWS,
    EmaMacdSensitivityTables,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    BREAK_EVEN_STATUS_ROOT_FOUND,
)


DAY09_ARTIFACT_VERSION: Final[str] = (
    "ema_macd_sensitivity_v1"
)

APPROVED_DAY09_ARTIFACT_FILENAMES: Final[
    frozenset[str]
] = frozenset(
    {
        "metadata.json",
        "parameter_results.csv",
        "annual_results.csv",
        "regime_results.csv",
        "holding_diagnostics.csv",
        "signal_validation.csv",
        "neighborhood_stability.csv",
        "net_sharpe_slices.png",
        "net_return_slices.png",
        "turnover_slices.png",
        "cost_break_even_slices.png",
        "stability_slices.png",
        "findings.md",
    }
)

REQUIRED_DAY09_METADATA_KEYS: Final[
    tuple[str, ...]
] = (
    "artifact_version",
    "permitted_dataset_identifier",
    "dataset_manifest_sha256",
    "development_sample_start",
    "development_sample_end",
    "locked_period_accessed",
    "parameter_selected_using_locked_period",
    "full_bar_level_artifacts_written",
    "symbol",
    "timeframe",
    "price_column",
    "return_column",
    "ema_method",
    "ema_seed",
    "ema_alpha_formula",
    "fast_window_grid",
    "slow_window_grid",
    "signal_window_grid",
    "neutral_band_grid",
    "configuration_count",
    "day08_baseline_configuration",
    "continuous_signal",
    "signal_rule",
    "signal_timing",
    "position_timing",
    "overnight_positions_allowed",
    "ema_state_resets_at_session_boundary",
    "transaction_cost_convention",
    "baseline_cost_bps_per_turnover",
    "turnover_definition",
    "direct_reversal_turnover",
    "annualization_factor",
    "risk_free_rate_convention",
    "holding_episode_definition",
    "whipsaw_definition",
    "cost_break_even_method",
    "volatility_regime_definition",
    "signal_validation_horizons",
    "signal_bucket_count",
    "neighborhood_definition",
    "composite_stability_score_used",
    "figure_slice_definition",
)

ABSOLUTE_PATH_PATTERNS: Final[
    tuple[re.Pattern[str], ...]
] = (
    re.compile(
        r"(?:^|[\s\"'])/"
        r"(?:Users|home|tmp|var|mnt|opt)/",
        flags=re.IGNORECASE,
    ),
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
)

PROHIBITED_OBSERVATION_COLUMNS: Final[
    frozenset[str]
] = frozenset(
    {
        "timestamp",
        "close",
        "fast_ema",
        "slow_ema",
        "macd",
        "macd_signal_line",
        "macd_histogram",
        "normalized_macd_histogram",
        "signal",
        "position",
        "transaction_cost",
        "gross_strategy_return",
        "net_strategy_return",
    }
)

CONFIGURATION_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "fast_window",
    "slow_window",
    "signal_window",
    "neutral_band",
)


class Day09ReportError(ValueError):
    """Raised when Day 9 artifacts cannot be written safely."""


def _records_from_frame(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a compact DataFrame into JSON-safe records."""

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


def _contains_absolute_local_path(
    value: object,
) -> bool:
    """Return whether nested text contains a local absolute path."""

    if isinstance(value, Mapping):
        return any(
            _contains_absolute_local_path(key)
            or _contains_absolute_local_path(item)
            for key, item in value.items()
        )

    if isinstance(
        value,
        (list, tuple, set, frozenset),
    ):
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


def build_day09_metadata(
    *,
    permitted_dataset_identifier: str,
    dataset_manifest_sha256: str,
    regime_definition: pd.DataFrame,
    symbol: str = "SPY",
    timeframe: str = "15-minute",
    price_column: str = "close",
    return_column: str = (
        "close_to_close_simple_return"
    ),
) -> dict[str, Any]:
    """Build immutable Day 9 development metadata."""

    if (
        not isinstance(
            permitted_dataset_identifier,
            str,
        )
        or not permitted_dataset_identifier.strip()
    ):
        raise Day09ReportError(
            "permitted_dataset_identifier must be a "
            "non-empty string."
        )

    normalized_identifier = (
        permitted_dataset_identifier.strip()
    )

    if _contains_absolute_local_path(
        normalized_identifier
    ):
        raise Day09ReportError(
            "permitted_dataset_identifier must not contain "
            "an absolute local path."
        )

    if not isinstance(
        dataset_manifest_sha256,
        str,
    ):
        raise Day09ReportError(
            "dataset_manifest_sha256 must be a SHA-256 string."
        )

    normalized_sha256 = (
        dataset_manifest_sha256.strip().lower()
    )

    if (
        re.fullmatch(
            r"[0-9a-f]{64}",
            normalized_sha256,
        )
        is None
    ):
        raise Day09ReportError(
            "dataset_manifest_sha256 must contain exactly "
            "64 hexadecimal characters."
        )

    normalized_symbol = str(symbol).strip().upper()

    if normalized_symbol != "SPY":
        raise Day09ReportError(
            "Day 9 primary sensitivity analysis must use SPY."
        )

    if not isinstance(
        regime_definition,
        pd.DataFrame,
    ):
        raise TypeError(
            "regime_definition must be a pandas DataFrame."
        )

    if regime_definition.empty:
        raise Day09ReportError(
            "regime_definition must contain the reused "
            "Day 5 volatility-regime definition."
        )

    metadata: dict[str, Any] = {
        "artifact_version": DAY09_ARTIFACT_VERSION,
        "permitted_dataset_identifier": (
            normalized_identifier
        ),
        "dataset_manifest_sha256": normalized_sha256,
        "development_sample_start": "2020-01-02",
        "development_sample_end": "2025-12-31",
        "locked_period_accessed": False,
        "parameter_selected_using_locked_period": False,
        "full_bar_level_artifacts_written": False,
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "price_column": price_column,
        "return_column": return_column,
        "ema_method": "recursive_adjust_false",
        "ema_seed": "first_available_observation",
        "ema_alpha_formula": "2 / (window + 1)",
        "fast_window_grid": list(FAST_WINDOWS),
        "slow_window_grid": list(SLOW_WINDOWS),
        "signal_window_grid": list(SIGNAL_WINDOWS),
        "neutral_band_grid": list(NEUTRAL_BANDS),
        "configuration_count": (
            EXPECTED_CONFIGURATION_COUNT
        ),
        "day08_baseline_configuration": {
            "fast_window": BASELINE_FAST_WINDOW,
            "slow_window": BASELINE_SLOW_WINDOW,
            "signal_window": BASELINE_SIGNAL_WINDOW,
            "neutral_band": BASELINE_NEUTRAL_BAND,
        },
        "continuous_signal": (
            "normalized_macd_histogram"
        ),
        "signal_rule": (
            "+1 above the neutral band; -1 below the "
            "negative neutral band; 0 otherwise"
        ),
        "signal_timing": (
            "signal calculated from information available "
            "at the close of bar t"
        ),
        "position_timing": "signal shifted by one bar",
        "overnight_positions_allowed": True,
        "ema_state_resets_at_session_boundary": False,
        "transaction_cost_convention": (
            "cost bps per absolute target-position change"
        ),
        "baseline_cost_bps_per_turnover": (
            DAY09_COST_BPS_PER_TURNOVER
        ),
        "turnover_definition": (
            "absolute change in one-bar-delayed "
            "target position"
        ),
        "direct_reversal_turnover": 2,
        "annualization_factor": (
            DAY09_ANNUALIZATION_FACTOR
        ),
        "risk_free_rate_convention": "zero",
        "holding_episode_definition": (
            "consecutive run of identical non-zero "
            "positions; neutral runs are excluded"
        ),
        "whipsaw_definition": (
            "non-zero position episode lasting no more "
            "than four 15-minute bars and followed within "
            "the next four bars by an opposite non-zero "
            "position"
        ),
        "cost_break_even_method": (
            "compounded simple-return wealth with a "
            "bounded deterministic numerical root search"
        ),
        "volatility_regime_definition": (
            _records_from_frame(regime_definition)
        ),
        "signal_validation_horizons": list(
            DAY08_FORWARD_HORIZONS
        ),
        "signal_bucket_count": 5,
        "neighborhood_definition": (
            "one declared grid step along exactly one of "
            "fast window, slow window, signal window or "
            "neutral band; the other three values remain "
            "unchanged"
        ),
        "composite_stability_score_used": False,
        "figure_slice_definition": (
            "rows are signal-line windows, columns are "
            "neutral bands, and each panel is a slow-window "
            "by fast-window matrix"
        ),
    }

    validate_day09_metadata(metadata)

    return metadata


def validate_day09_metadata(
    metadata: Mapping[str, Any],
) -> None:
    """Validate Day 9 metadata and locked-period controls."""

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping."
        )

    missing_keys = [
        key
        for key in REQUIRED_DAY09_METADATA_KEYS
        if key not in metadata
    ]

    if missing_keys:
        raise Day09ReportError(
            "Day 9 metadata are missing required keys: "
            f"{missing_keys}."
        )

    if (
        metadata["artifact_version"]
        != DAY09_ARTIFACT_VERSION
    ):
        raise Day09ReportError(
            "Unexpected Day 9 artifact version."
        )

    if (
        metadata["development_sample_start"]
        != "2020-01-02"
    ):
        raise Day09ReportError(
            "Day 9 development sample must start "
            "on 2020-01-02."
        )

    if (
        metadata["development_sample_end"]
        != "2025-12-31"
    ):
        raise Day09ReportError(
            "Day 9 development sample must end "
            "on 2025-12-31."
        )

    false_flags = (
        "locked_period_accessed",
        "parameter_selected_using_locked_period",
        "full_bar_level_artifacts_written",
        "composite_stability_score_used",
    )

    for key in false_flags:
        if metadata[key] is not False:
            raise Day09ReportError(
                f"{key} must be false."
            )

    frozen_axes = {
        "fast_window_grid": list(FAST_WINDOWS),
        "slow_window_grid": list(SLOW_WINDOWS),
        "signal_window_grid": list(
            SIGNAL_WINDOWS
        ),
        "neutral_band_grid": list(
            NEUTRAL_BANDS
        ),
        "signal_validation_horizons": list(
            DAY08_FORWARD_HORIZONS
        ),
    }

    for key, expected in frozen_axes.items():
        if list(metadata[key]) != expected:
            raise Day09ReportError(
                f"{key} does not match the frozen Day 9 "
                "definition."
            )

    if (
        metadata["configuration_count"]
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "configuration_count must remain 108."
        )

    expected_baseline = {
        "fast_window": BASELINE_FAST_WINDOW,
        "slow_window": BASELINE_SLOW_WINDOW,
        "signal_window": BASELINE_SIGNAL_WINDOW,
        "neutral_band": BASELINE_NEUTRAL_BAND,
    }

    if (
        dict(
            metadata[
                "day08_baseline_configuration"
            ]
        )
        != expected_baseline
    ):
        raise Day09ReportError(
            "Day 8 baseline configuration metadata "
            "does not match the frozen baseline."
        )

    manifest_hash = str(
        metadata["dataset_manifest_sha256"]
    ).strip().lower()

    if (
        re.fullmatch(
            r"[0-9a-f]{64}",
            manifest_hash,
        )
        is None
    ):
        raise Day09ReportError(
            "dataset_manifest_sha256 must be a valid "
            "SHA-256 value."
        )

    if _contains_absolute_local_path(metadata):
        raise Day09ReportError(
            "Metadata must not contain absolute local paths."
        )


def _validate_compact_table(
    frame: pd.DataFrame,
    *,
    name: str,
    required_nonempty: bool = True,
) -> None:
    """Validate one compact summary table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if required_nonempty and frame.empty:
        raise Day09ReportError(
            f"{name} must not be empty."
        )

    prohibited = sorted(
        PROHIBITED_OBSERVATION_COLUMNS.intersection(
            frame.columns
        )
    )

    if prohibited:
        raise Day09ReportError(
            f"{name} contains prohibited observation-level "
            f"columns: {prohibited}."
        )


def _validate_tables(
    tables: EmaMacdSensitivityTables,
) -> None:
    """Validate complete compact full-grid inputs."""

    if not isinstance(
        tables,
        EmaMacdSensitivityTables,
    ):
        raise TypeError(
            "tables must be an "
            "EmaMacdSensitivityTables object."
        )

    table_map = {
        "parameter_results": tables.parameter_results,
        "annual_results": tables.annual_results,
        "annual_consistency": (
            tables.annual_consistency
        ),
        "regime_results": tables.regime_results,
        "regime_definition": (
            tables.regime_definition
        ),
        "holding_diagnostics": (
            tables.holding_diagnostics
        ),
        "signal_validation": (
            tables.signal_validation
        ),
        "signal_buckets": tables.signal_buckets,
        "neighborhood_stability": (
            tables.neighborhood_stability
        ),
    }

    for name, frame in table_map.items():
        _validate_compact_table(
            frame,
            name=name,
        )

    parameter_results = tables.parameter_results

    if (
        len(parameter_results)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "parameter_results must contain exactly "
            "108 configurations."
        )

    if (
        parameter_results[
            "configuration_id"
        ].nunique()
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "parameter_results must contain 108 unique "
            "configuration identifiers."
        )

    if (
        len(tables.holding_diagnostics)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "holding_diagnostics must contain 108 rows."
        )

    if (
        len(tables.annual_consistency)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "annual_consistency must contain 108 rows."
        )

    if (
        len(tables.neighborhood_stability)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise Day09ReportError(
            "neighborhood_stability must contain 108 rows."
        )

    expected_signal_rows = (
        EXPECTED_CONFIGURATION_COUNT
        * len(DAY08_FORWARD_HORIZONS)
    )

    if (
        len(tables.signal_validation)
        != expected_signal_rows
    ):
        raise Day09ReportError(
            "signal_validation must contain one row for "
            "every configuration and forward horizon."
        )

    expected_ids = set(
        parameter_results["configuration_id"]
    )

    for name in (
        "annual_results",
        "annual_consistency",
        "regime_results",
        "holding_diagnostics",
        "signal_validation",
        "signal_buckets",
        "neighborhood_stability",
    ):
        frame = getattr(tables, name)

        if "configuration_id" not in frame.columns:
            raise Day09ReportError(
                f"{name} must contain configuration_id."
            )

        actual_ids = set(
            frame["configuration_id"].dropna()
        )

        if actual_ids != expected_ids:
            raise Day09ReportError(
                f"{name} does not preserve the complete "
                "Day 9 configuration set."
            )


def _build_signal_artifact(
    tables: EmaMacdSensitivityTables,
) -> pd.DataFrame:
    """Combine horizon and bucket diagnostics compactly."""

    summary = tables.signal_validation.copy(
        deep=True
    )
    buckets = tables.signal_buckets.copy(
        deep=True
    )

    summary.insert(
        0,
        "record_type",
        "horizon_summary",
    )
    buckets.insert(
        0,
        "record_type",
        "signal_bucket",
    )

    combined = pd.concat(
        [summary, buckets],
        ignore_index=True,
        sort=False,
    )

    sort_columns = [
        column
        for column in (
            "fast_window",
            "slow_window",
            "signal_window",
            "neutral_band",
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


def _write_csv(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a deterministic compact CSV."""

    frame.to_csv(
        output_path,
        index=False,
        float_format="%.12g",
        na_rep="",
        lineterminator="\n",
    )


def _finite_limits(
    values: pd.Series,
) -> tuple[float | None, float | None]:
    """Return common finite figure limits."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    finite = numeric.dropna()

    if finite.empty:
        return None, None

    minimum = float(finite.min())
    maximum = float(finite.max())

    if math.isclose(
        minimum,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return None, None

    return minimum, maximum


def _write_slice_figure(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    annotation_format: str,
) -> None:
    """Write 12 deterministic fast/slow parameter slices."""

    required_columns = {
        *CONFIGURATION_COLUMNS,
        value_column,
    }
    missing = sorted(
        required_columns.difference(frame.columns)
    )

    if missing:
        raise Day09ReportError(
            f"Figure input is missing columns: {missing}."
        )

    vmin, vmax = _finite_limits(
        frame[value_column]
    )

    figure, axes = plt.subplots(
        len(SIGNAL_WINDOWS),
        len(NEUTRAL_BANDS),
        figsize=(15, 10.5),
        constrained_layout=True,
        squeeze=False,
    )

    image = None

    for row_index, signal_window in enumerate(
        SIGNAL_WINDOWS
    ):
        for column_index, neutral_band in enumerate(
            NEUTRAL_BANDS
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            panel = frame.loc[
                frame["signal_window"].eq(
                    signal_window
                )
                & frame["neutral_band"].eq(
                    neutral_band
                )
            ]

            pivot = (
                panel.pivot(
                    index="slow_window",
                    columns="fast_window",
                    values=value_column,
                )
                .reindex(
                    index=SLOW_WINDOWS,
                    columns=FAST_WINDOWS,
                )
            )

            matrix = pivot.to_numpy(dtype=float)
            masked = np.ma.masked_invalid(matrix)

            image_kwargs: dict[str, object] = {
                "origin": "lower",
                "aspect": "auto",
            }

            if vmin is not None and vmax is not None:
                image_kwargs.update(
                    {
                        "vmin": vmin,
                        "vmax": vmax,
                    }
                )

            image = axis.imshow(
                masked,
                **image_kwargs,
            )

            axis.set_title(
                f"Signal={signal_window}, "
                f"band={neutral_band:.5f}"
            )
            axis.set_xlabel("Fast EMA")
            axis.set_ylabel("Slow EMA")
            axis.set_xticks(
                range(len(FAST_WINDOWS)),
                labels=[
                    str(value)
                    for value in FAST_WINDOWS
                ],
            )
            axis.set_yticks(
                range(len(SLOW_WINDOWS)),
                labels=[
                    str(value)
                    for value in SLOW_WINDOWS
                ],
            )

            for slow_index, slow_window in enumerate(
                SLOW_WINDOWS
            ):
                for fast_index, fast_window in enumerate(
                    FAST_WINDOWS
                ):
                    value = pivot.loc[
                        slow_window,
                        fast_window,
                    ]

                    label = (
                        "NA"
                        if pd.isna(value)
                        else format(
                            float(value),
                            annotation_format,
                        )
                    )

                    axis.text(
                        fast_index,
                        slow_index,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

    figure.suptitle(title)

    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.80,
        )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    if (
        not output_path.exists()
        or output_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Figure was not written correctly: "
            f"{output_path.name}."
        )


def build_day09_findings(
    tables: EmaMacdSensitivityTables,
) -> str:
    """Build conservative development-only findings."""

    parameter_results = (
        tables.parameter_results.copy(deep=True)
    )
    stability = (
        tables.neighborhood_stability.copy(
            deep=True
        )
    )

    positive_net_count = int(
        parameter_results[
            "net_cumulative_return"
        ].gt(0.0).sum()
    )

    positive_break_even_count = int(
        (
            parameter_results[
                "break_even_status"
            ].eq(BREAK_EVEN_STATUS_ROOT_FOUND)
            & parameter_results[
                "break_even_cost_bps"
            ].gt(0.0)
        ).sum()
    )

    median_turnover = float(
        parameter_results[
            "total_turnover"
        ].median()
    )

    baseline = parameter_results.loc[
        parameter_results["fast_window"].eq(
            BASELINE_FAST_WINDOW
        )
        & parameter_results["slow_window"].eq(
            BASELINE_SLOW_WINDOW
        )
        & parameter_results["signal_window"].eq(
            BASELINE_SIGNAL_WINDOW
        )
        & parameter_results["neutral_band"].eq(
            BASELINE_NEUTRAL_BAND
        )
    ]

    if len(baseline) != 1:
        raise Day09ReportError(
            "The Day 8 baseline must appear exactly once "
            "in Day 9 parameter results."
        )

    baseline_row = baseline.iloc[0]

    finite_sharpe = parameter_results.dropna(
        subset=["net_sharpe_ratio"]
    )

    if finite_sharpe.empty:
        isolated_sentence = (
            "No configuration produced a finite net "
            "Sharpe estimate."
        )
    else:
        isolated = finite_sharpe.sort_values(
            [
                "net_sharpe_ratio",
                "configuration_id",
            ],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]

        isolated_sentence = (
            "The highest isolated development net Sharpe "
            f"occurred at `{isolated['configuration_id']}` "
            f"with a value of "
            f"{isolated['net_sharpe_ratio']:.3f}. "
            "This cell is reported diagnostically and is "
            "not selected as a final parameter."
        )

    finite_stability = stability.dropna(
        subset=[
            "median_neighbor_net_sharpe"
        ]
    )

    if finite_stability.empty:
        stability_sentence = (
            "No finite neighbouring net-Sharpe median "
            "was available."
        )
    else:
        stable = finite_stability.sort_values(
            [
                "median_neighbor_net_sharpe",
                "configuration_id",
            ],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]

        stability_sentence = (
            "The highest median neighbouring development "
            "net Sharpe occurred around "
            f"`{stable['configuration_id']}` at "
            f"{stable['median_neighbor_net_sharpe']:.3f}; "
            "its own net Sharpe was "
            f"{stable['net_sharpe_ratio']:.3f}. "
            "This is neighbourhood evidence, not a "
            "parameter-selection decision."
        )

    findings = f"""# Day 9 Findings — EMA/MACD Sensitivity

## Scope

This report evaluates the predeclared 108-configuration SPY 15-minute
development grid only. It is parameter sensitivity, not parameter
optimisation. The locked final period from 2026-01-02 through 2026-06-30
was not accessed.

The grid varies only the fast EMA window, slow EMA window, MACD signal-line
window and normalised-histogram neutral band. Signal construction,
one-bar-delayed position timing, overnight carry and the one-basis-point
turnover-cost convention remain unchanged from Day 8.

## Development evidence

- Configurations with positive cumulative net return: {positive_net_count} of 108.
- Configurations with a positive model-implied break-even cost: {positive_break_even_count} of 108.
- Median total turnover across the grid: {median_turnover:.1f} turnover units.
- Day 8 baseline net cumulative return: {float(baseline_row['net_cumulative_return']):.4%}.
- Day 8 baseline net Sharpe: {float(baseline_row['net_sharpe_ratio']):.4f}.

{isolated_sentence}

{stability_sentence}

## Interpretation constraints

These results are development-only sensitivity evidence.
They do not establish alpha, statistical superiority, robustness,
deployability, realised execution costs or final parameter validity.

A broad coherent neighbourhood is stronger evidence than an isolated
high-performing cell, but even a stable development plateau must later
pass walk-forward evaluation, explicit anti-overfitting controls and the
separately locked final test.

No parameter was selected using locked-period information.
"""

    if _contains_absolute_local_path(findings):
        raise RuntimeError(
            "Generated findings contain an absolute path."
        )

    return findings


def _reject_unapproved_existing_entries(
    output_directory: Path,
) -> None:
    """Reject unrelated files and nested directories."""

    if not output_directory.exists():
        return

    unexpected_files = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
        and path.name
        not in APPROVED_DAY09_ARTIFACT_FILENAMES
    )

    if unexpected_files:
        raise Day09ReportError(
            "Output directory contains unapproved files: "
            f"{unexpected_files}."
        )

    nested_directories = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_dir()
    )

    if nested_directories:
        raise Day09ReportError(
            "Output directory must not contain nested "
            f"directories: {nested_directories}."
        )


def _validate_written_artifacts(
    output_directory: Path,
) -> tuple[Path, ...]:
    """Validate the exact final artifact set."""

    actual = {
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    }

    if actual != APPROVED_DAY09_ARTIFACT_FILENAMES:
        raise Day09ReportError(
            "Written Day 9 artifact set does not match "
            "the approved contract. "
            f"Missing: "
            f"{sorted(APPROVED_DAY09_ARTIFACT_FILENAMES - actual)}; "
            f"unexpected: "
            f"{sorted(actual - APPROVED_DAY09_ARTIFACT_FILENAMES)}."
        )

    paths = tuple(
        output_directory / name
        for name in sorted(
            APPROVED_DAY09_ARTIFACT_FILENAMES
        )
    )

    empty = [
        path.name
        for path in paths
        if (
            not path.exists()
            or path.stat().st_size <= 0
        )
    ]

    if empty:
        raise Day09ReportError(
            "Day 9 artifacts are missing or empty: "
            f"{empty}."
        )

    forbidden_suffixes = {
        ".parquet",
        ".feather",
        ".pickle",
        ".pkl",
    }
    forbidden = [
        path.name
        for path in output_directory.iterdir()
        if path.suffix.lower()
        in forbidden_suffixes
    ]

    if forbidden:
        raise Day09ReportError(
            "Forbidden full-data artifacts were created: "
            f"{forbidden}."
        )

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in paths
        if path.suffix.lower()
        in {".json", ".csv", ".md"}
    )

    if _contains_absolute_local_path(text):
        raise Day09ReportError(
            "Written text artifacts contain an absolute "
            "local path."
        )

    return paths


def write_day09_ema_macd_sensitivity_artifacts(
    tables: EmaMacdSensitivityTables,
    *,
    metadata: Mapping[str, Any],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Write exactly the approved compact Day 9 outputs."""

    _validate_tables(tables)
    validate_day09_metadata(metadata)

    directory = Path(output_directory)

    _reject_unapproved_existing_entries(
        directory
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
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
            tables.holding_diagnostics.copy(
                deep=True
            )
        ),
        "signal_validation.csv": (
            _build_signal_artifact(tables)
        ),
        "neighborhood_stability.csv": (
            tables.neighborhood_stability.copy(
                deep=True
            )
        ),
    }

    for filename, frame in table_map.items():
        _write_csv(
            frame,
            directory / filename,
        )

    metadata_text = json.dumps(
        dict(metadata),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )

    if _contains_absolute_local_path(
        metadata_text
    ):
        raise Day09ReportError(
            "Serialized metadata contain an absolute "
            "local path."
        )

    (
        directory / "metadata.json"
    ).write_text(
        metadata_text + "\n",
        encoding="utf-8",
    )

    (
        directory / "findings.md"
    ).write_text(
        build_day09_findings(tables),
        encoding="utf-8",
    )

    _write_slice_figure(
        tables.parameter_results,
        value_column="net_sharpe_ratio",
        title=(
            "Day 9 EMA/MACD Net Sharpe "
            "Sensitivity"
        ),
        output_path=(
            directory / "net_sharpe_slices.png"
        ),
        annotation_format=".2f",
    )
    _write_slice_figure(
        tables.parameter_results,
        value_column="net_cumulative_return",
        title=(
            "Day 9 EMA/MACD Net Cumulative "
            "Return Sensitivity"
        ),
        output_path=(
            directory / "net_return_slices.png"
        ),
        annotation_format=".1%",
    )
    _write_slice_figure(
        tables.parameter_results,
        value_column="total_turnover",
        title=(
            "Day 9 EMA/MACD Turnover Sensitivity"
        ),
        output_path=(
            directory / "turnover_slices.png"
        ),
        annotation_format=".0f",
    )
    _write_slice_figure(
        tables.parameter_results,
        value_column="break_even_cost_bps",
        title=(
            "Day 9 EMA/MACD Cost Break-Even "
            "Sensitivity"
        ),
        output_path=(
            directory
            / "cost_break_even_slices.png"
        ),
        annotation_format=".2f",
    )
    _write_slice_figure(
        tables.neighborhood_stability,
        value_column=(
            "median_neighbor_net_sharpe"
        ),
        title=(
            "Day 9 EMA/MACD Median Neighbour "
            "Net Sharpe"
        ),
        output_path=(
            directory / "stability_slices.png"
        ),
        annotation_format=".2f",
    )

    return _validate_written_artifacts(
        directory
    )
