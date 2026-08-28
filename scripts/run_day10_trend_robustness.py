"""Execute and report the frozen Day 10 trend robustness matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_alpha.analysis.eda_features import (
    REQUIRED_COLUMNS as CANONICAL_BAR_COLUMNS,
)
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    CONFIGURATION_IDS,
    DEVELOPMENT_DATASET_ID,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    EXPECTED_PORTFOLIO_OBSERVATIONS,
    REQUIRED_RESULT_COLUMNS,
    ROBUSTNESS_FREQUENCIES,
    ROBUSTNESS_STRATEGIES,
    ROBUSTNESS_SYMBOLS,
    build_robustness_run_matrix,
    run_trend_family_robustness,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
)
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day10"
)

EXPECTED_SOURCE_ROW_COUNT: int = 117_192
EXPECTED_SESSION_SIZE_COUNTS: dict[int, int] = {
    14: 36,
    26: 4_488,
}
EXPECTED_FREQUENCY_ROW_COUNTS: dict[
    str,
    int,
] = dict(EXPECTED_PORTFOLIO_OBSERVATIONS)

APPROVED_ARTIFACT_NAMES: Final[
    tuple[str, ...]
] = (
    "matrix.csv",
    "summary.csv",
    "manifest.json",
    "report.md",
    "sharpe.png",
    "drawdown.png",
    "turnover.png",
)

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "strategy",
    "symbol",
    "frequency",
    "reference_symbol",
    "reference_frequency",
    "sharpe_ratio",
    "sharpe_delta",
    "annualized_return",
    "annualized_return_delta",
    "maximum_drawdown",
    "maximum_drawdown_delta",
    "turnover",
    "turnover_ratio_to_reference",
    "average_exposure",
    "sign_matches_reference_return",
    "sign_matches_reference_sharpe",
)

PERMANENT_FLAT_EXPOSURE_THRESHOLD: Final[
    float
] = 99.5
NEAR_PERMANENT_DIRECTIONAL_THRESHOLD: Final[
    float
] = 95.0
EXTREME_TURNOVER_RATIO_LOW: Final[float] = (
    0.25
)
EXTREME_TURNOVER_RATIO_HIGH: Final[float] = (
    4.0
)
MATERIAL_DRAWDOWN_INCREASE: Final[float] = (
    0.10
)
LOW_ACTIVE_OBSERVATION_PROPORTION: Final[
    float
] = 0.20

DIAGNOSTIC_THRESHOLDS: Final[
    dict[str, float]
] = {
    "permanent_flat_exposure_pct": (
        PERMANENT_FLAT_EXPOSURE_THRESHOLD
    ),
    "near_permanent_directional_exposure_pct": (
        NEAR_PERMANENT_DIRECTIONAL_THRESHOLD
    ),
    "extreme_turnover_ratio_low": (
        EXTREME_TURNOVER_RATIO_LOW
    ),
    "extreme_turnover_ratio_high": (
        EXTREME_TURNOVER_RATIO_HIGH
    ),
    "material_drawdown_increase": (
        MATERIAL_DRAWDOWN_INCREASE
    ),
    "low_active_observation_proportion": (
        LOW_ACTIVE_OBSERVATION_PROPORTION
    ),
}

FLAG_COLUMNS: Final[tuple[str, ...]] = (
    "strategy",
    "symbol",
    "frequency",
    "permanently_flat",
    "near_permanent_long",
    "near_permanent_short",
    "zero_trades",
    "undefined_sharpe",
    "extreme_turnover_change",
    "return_sign_reversal",
    "sharpe_sign_reversal",
    "materially_larger_drawdown",
    "low_active_observation_count",
)

STRATEGY_DISPLAY_NAMES: Final[
    dict[str, str]
] = {
    "trend_ratio": "Trend Ratio",
    "ema_macd": "EMA/MACD",
}


class Day10RunnerError(ValueError):
    """Raised when the Day 10 runner cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class Day10RunResult:
    """In-memory results and written Day 10 artifact paths."""

    matrix: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object]
    artifact_directory: Path
    artifact_paths: tuple[Path, ...]


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the Day 10 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen development-only Day 10 "
            "trend-family robustness matrix."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Canonical 15-minute development Parquet "
            "or CSV path."
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="Directory for the seven Day 10 artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing approved Day 10 "
            "artifact set."
        ),
    )

    return parser.parse_args(argv)


def _read_canonical_dataset(
    path: Path,
) -> pd.DataFrame:
    """Read one explicitly selected canonical dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical dataset does not exist: {path}"
        )

    if not path.is_file():
        raise Day10RunnerError(
            "Canonical dataset path must be a file."
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise Day10RunnerError(
        "Canonical dataset must be Parquet or CSV."
    )


def validate_canonical_input(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the audited development dataset before execution."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    if bars.empty:
        raise Day10RunnerError(
            "Canonical input must not be empty."
        )

    missing_columns = sorted(
        set(CANONICAL_BAR_COLUMNS).difference(
            bars.columns
        )
    )

    if missing_columns:
        raise Day10RunnerError(
            "Canonical input is missing required columns: "
            f"{missing_columns}."
        )

    result = bars.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise Day10RunnerError(
            "Canonical input contains malformed "
            "timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise Day10RunnerError(
            "Canonical timestamps cannot be missing."
        )

    normalized_symbols = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    actual_symbols = set(
        normalized_symbols.dropna()
    )
    required_symbols = set(
        ROBUSTNESS_SYMBOLS
    )
    missing_symbols = sorted(
        required_symbols - actual_symbols
    )
    unexpected_symbols = sorted(
        actual_symbols - required_symbols
    )

    if missing_symbols:
        raise Day10RunnerError(
            "Canonical input is missing required symbols: "
            f"{missing_symbols}."
        )

    if unexpected_symbols:
        raise Day10RunnerError(
            "Canonical input contains unexpected symbols: "
            f"{unexpected_symbols}."
        )

    local_dates = (
        result["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if (
        local_dates.min() < DEVELOPMENT_START
        or local_dates.max() > DEVELOPMENT_END
    ):
        raise Day10RunnerError(
            "Canonical input must remain within the "
            "development period from 2020-01-02 through "
            "2025-12-31; no 2026 observations are allowed."
        )

    if len(result) != EXPECTED_SOURCE_ROW_COUNT:
        raise Day10RunnerError(
            "Canonical input row count must be exactly "
            f"{EXPECTED_SOURCE_ROW_COUNT:,}; received "
            f"{len(result):,}."
        )

    session_labels = local_dates.dt.strftime(
        "%Y-%m-%d"
    )
    session_sizes = (
        result.assign(
            _session_date=session_labels
        )
        .groupby(
            [
                "symbol",
                "_session_date",
            ],
            observed=True,
            sort=False,
        )
        .size()
    )
    session_size_counts = {
        int(size): int(count)
        for size, count in (
            session_sizes.value_counts()
            .sort_index()
            .items()
        )
    }

    if (
        session_size_counts
        != EXPECTED_SESSION_SIZE_COUNTS
    ):
        raise Day10RunnerError(
            "Canonical session counts differ from the "
            "audited contract. Expected "
            f"{EXPECTED_SESSION_SIZE_COUNTS}; received "
            f"{session_size_counts}."
        )

    try:
        validated = aggregate_session_bars(
            result,
            "15min",
        )
    except SessionAggregationError as exc:
        raise Day10RunnerError(
            "Canonical 15-minute validation failed: "
            f"{exc}"
        ) from exc

    if len(validated) != len(result):
        raise RuntimeError(
            "15-minute validation changed the source "
            "row count."
        )

    return validated


def _validate_matrix_structure(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Validate schema, keys and frozen run ordering."""

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(
            "matrix must be a pandas DataFrame."
        )

    if tuple(matrix.columns) != (
        REQUIRED_RESULT_COLUMNS
    ):
        raise Day10RunnerError(
            "Matrix columns do not match the complete "
            "Day 10 result schema."
        )

    if len(matrix) != 18:
        raise Day10RunnerError(
            "Day 10 matrix must contain exactly 18 rows."
        )

    key_columns = [
        "strategy",
        "symbol",
        "frequency",
    ]

    if matrix.duplicated(key_columns).any():
        raise Day10RunnerError(
            "Matrix strategy/symbol/frequency keys must "
            "be unique."
        )

    actual_keys = list(
        matrix[
            key_columns
        ].itertuples(
            index=False,
            name=None,
        )
    )
    expected_keys = [
        (
            specification.strategy,
            specification.symbol,
            specification.frequency,
        )
        for specification in (
            build_robustness_run_matrix()
        )
    ]

    if actual_keys != expected_keys:
        raise Day10RunnerError(
            "Matrix rows do not follow the frozen "
            "deterministic order."
        )

    if not matrix["dataset_id"].eq(
        DEVELOPMENT_DATASET_ID
    ).all():
        raise Day10RunnerError(
            "Matrix dataset lineage is not frozen."
        )

    expected_ids = matrix[
        "strategy"
    ].map(CONFIGURATION_IDS)

    if not matrix[
        "configuration_id"
    ].eq(expected_ids).all():
        raise Day10RunnerError(
            "Matrix strategy configuration identifiers "
            "are not frozen."
        )

    expected_factors = matrix[
        "frequency"
    ].map(ANNUALIZATION_FACTORS)

    if not matrix[
        "annualization_factor"
    ].eq(expected_factors).all():
        raise Day10RunnerError(
            "Matrix annualization factors are not frozen."
        )

    starts = pd.to_datetime(
        matrix["start_timestamp"],
        utc=True,
        errors="raise",
    )
    ends = pd.to_datetime(
        matrix["end_timestamp"],
        utc=True,
        errors="raise",
    )
    start_dates = (
        starts.dt.tz_convert(
            "America/New_York"
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )
    end_dates = (
        ends.dt.tz_convert(
            "America/New_York"
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if (
        start_dates.lt(DEVELOPMENT_START).any()
        or end_dates.gt(DEVELOPMENT_END).any()
    ):
        raise Day10RunnerError(
            "Matrix results extend outside the "
            "development period."
        )

    return matrix.copy(deep=True)


def validate_matrix_output(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the complete canonical 18-run output."""

    result = _validate_matrix_structure(matrix)
    symbol_count = len(ROBUSTNESS_SYMBOLS)

    for frequency in ROBUSTNESS_FREQUENCIES:
        total_rows = (
            EXPECTED_FREQUENCY_ROW_COUNTS[
                frequency
            ]
        )

        if total_rows % symbol_count != 0:
            raise RuntimeError(
                "Frequency row count cannot be divided "
                "across the frozen symbol universe."
            )

        expected_observations = (
            total_rows // symbol_count
        )
        selected = result.loc[
            result["frequency"].eq(
                frequency
            )
        ]

        if not selected[
            "observations"
        ].eq(expected_observations).all():
            raise Day10RunnerError(
                f"{frequency} matrix observations do not "
                "match the audited row count."
            )

        expected_partials = (
            selected["sessions"]
            if frequency == "60min"
            else 0
        )

        if not selected[
            "partial_bar_count"
        ].eq(expected_partials).all():
            raise Day10RunnerError(
                f"{frequency} partial-bar counts do not "
                "match the frozen policy."
            )

    return result


def _safe_delta(
    value: object,
    reference: object,
) -> float:
    """Return a finite-aware arithmetic difference."""

    left = float(value)
    right = float(reference)

    if (
        not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return float("nan")

    return left - right


def _safe_ratio(
    value: object,
    reference: object,
) -> float:
    """Return a finite ratio or the project-style NaN."""

    numerator = float(value)
    denominator = float(reference)

    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0.0
    ):
        return float("nan")

    return numerator / denominator


def _matching_sign(
    value: object,
    reference: object,
) -> object:
    """Compare signs while preserving undefined values."""

    left = float(value)
    right = float(reference)

    if (
        not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return pd.NA

    return bool(
        np.sign(left) == np.sign(right)
    )


def build_summary(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each run with its strategy's SPY 15-minute row."""

    validated = _validate_matrix_structure(
        matrix
    )
    records: list[dict[str, object]] = []

    for strategy in ROBUSTNESS_STRATEGIES:
        strategy_rows = validated.loc[
            validated["strategy"].eq(strategy)
        ]
        reference = strategy_rows.loc[
            strategy_rows["symbol"].eq("SPY")
            & strategy_rows[
                "frequency"
            ].eq("15min")
        ]

        if len(reference) != 1:
            raise Day10RunnerError(
                "Each strategy requires exactly one SPY "
                "15-minute reference row."
            )

        reference_row = reference.iloc[0]

        for row in strategy_rows.itertuples(
            index=False
        ):
            records.append(
                {
                    "strategy": row.strategy,
                    "symbol": row.symbol,
                    "frequency": row.frequency,
                    "reference_symbol": "SPY",
                    "reference_frequency": (
                        "15min"
                    ),
                    "sharpe_ratio": (
                        row.sharpe_ratio
                    ),
                    "sharpe_delta": _safe_delta(
                        row.sharpe_ratio,
                        reference_row[
                            "sharpe_ratio"
                        ],
                    ),
                    "annualized_return": (
                        row.annualized_return
                    ),
                    "annualized_return_delta": (
                        _safe_delta(
                            row.annualized_return,
                            reference_row[
                                "annualized_return"
                            ],
                        )
                    ),
                    "maximum_drawdown": (
                        row.maximum_drawdown
                    ),
                    "maximum_drawdown_delta": (
                        _safe_delta(
                            row.maximum_drawdown,
                            reference_row[
                                "maximum_drawdown"
                            ],
                        )
                    ),
                    "turnover": row.turnover,
                    "turnover_ratio_to_reference": (
                        _safe_ratio(
                            row.turnover,
                            reference_row[
                                "turnover"
                            ],
                        )
                    ),
                    "average_exposure": (
                        row.average_exposure
                    ),
                    "sign_matches_reference_return": (
                        _matching_sign(
                            row.annualized_return,
                            reference_row[
                                "annualized_return"
                            ],
                        )
                    ),
                    "sign_matches_reference_sharpe": (
                        _matching_sign(
                            row.sharpe_ratio,
                            reference_row[
                                "sharpe_ratio"
                            ],
                        )
                    ),
                }
            )

    summary = pd.DataFrame.from_records(
        records,
        columns=SUMMARY_COLUMNS,
    )
    summary[
        "sign_matches_reference_return"
    ] = summary[
        "sign_matches_reference_return"
    ].astype("boolean")
    summary[
        "sign_matches_reference_sharpe"
    ] = summary[
        "sign_matches_reference_sharpe"
    ].astype("boolean")

    return summary


def build_degenerate_flags(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build descriptive flags without rejecting any run."""

    validated = _validate_matrix_structure(
        matrix
    )
    comparisons = summary.set_index(
        [
            "strategy",
            "symbol",
            "frequency",
        ]
    )
    records: list[dict[str, object]] = []

    for row in validated.itertuples(
        index=False
    ):
        key = (
            row.strategy,
            row.symbol,
            row.frequency,
        )
        comparison = comparisons.loc[key]
        turnover_ratio = float(
            comparison[
                "turnover_ratio_to_reference"
            ]
        )
        extreme_turnover = (
            math.isfinite(turnover_ratio)
            and (
                turnover_ratio
                < EXTREME_TURNOVER_RATIO_LOW
                or turnover_ratio
                > EXTREME_TURNOVER_RATIO_HIGH
            )
        )
        active_proportion = (
            float(row.active_observations)
            / float(row.observations)
        )
        return_sign = comparison[
            "sign_matches_reference_return"
        ]
        sharpe_sign = comparison[
            "sign_matches_reference_sharpe"
        ]

        records.append(
            {
                "strategy": row.strategy,
                "symbol": row.symbol,
                "frequency": row.frequency,
                "permanently_flat": (
                    float(row.flat_exposure)
                    >= (
                        PERMANENT_FLAT_EXPOSURE_THRESHOLD
                    )
                    if pd.notna(
                        row.flat_exposure
                    )
                    else False
                ),
                "near_permanent_long": (
                    float(row.long_exposure)
                    >= (
                        NEAR_PERMANENT_DIRECTIONAL_THRESHOLD
                    )
                    if pd.notna(
                        row.long_exposure
                    )
                    else False
                ),
                "near_permanent_short": (
                    float(row.short_exposure)
                    >= (
                        NEAR_PERMANENT_DIRECTIONAL_THRESHOLD
                    )
                    if pd.notna(
                        row.short_exposure
                    )
                    else False
                ),
                "zero_trades": int(
                    row.trade_count
                ) == 0,
                "undefined_sharpe": pd.isna(
                    row.sharpe_ratio
                ),
                "extreme_turnover_change": (
                    extreme_turnover
                ),
                "return_sign_reversal": (
                    not bool(return_sign)
                    if pd.notna(return_sign)
                    else False
                ),
                "sharpe_sign_reversal": (
                    not bool(sharpe_sign)
                    if pd.notna(sharpe_sign)
                    else False
                ),
                "materially_larger_drawdown": (
                    float(
                        comparison[
                            "maximum_drawdown_delta"
                        ]
                    )
                    <= -MATERIAL_DRAWDOWN_INCREASE
                ),
                "low_active_observation_count": (
                    active_proportion
                    < (
                        LOW_ACTIVE_OBSERVATION_PROPORTION
                    )
                ),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=FLAG_COLUMNS,
    )


def _format_markdown_value(
    value: object,
) -> str:
    """Format one deterministic compact Markdown value."""

    if pd.isna(value):
        return "NA"

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"

    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render a small DataFrame without optional dependencies."""

    columns = list(frame.columns)
    header = (
        "| "
        + " | ".join(columns)
        + " |"
    )
    separator = (
        "| "
        + " | ".join("---" for _ in columns)
        + " |"
    )
    rows = [
        "| "
        + " | ".join(
            _format_markdown_value(value)
            for value in row
        )
        + " |"
        for row in frame.itertuples(
            index=False,
            name=None,
        )
    ]

    return "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    )


def build_report(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    """Build the concise deterministic Day 10 report."""

    flags = build_degenerate_flags(
        matrix,
        summary,
    )
    flag_columns = list(
        FLAG_COLUMNS[3:]
    )
    flag_counts = (
        flags[flag_columns]
        .sum()
        .astype(int)
        .rename_axis("diagnostic_flag")
        .reset_index(name="flagged_runs")
    )
    cross_asset = (
        matrix.groupby(
            [
                "strategy",
                "symbol",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            mean_annualized_return=(
                "annualized_return",
                "mean",
            ),
            mean_sharpe_ratio=(
                "sharpe_ratio",
                "mean",
            ),
            worst_maximum_drawdown=(
                "maximum_drawdown",
                "min",
            ),
            mean_turnover=(
                "turnover",
                "mean",
            ),
        )
        .reset_index()
    )
    cross_frequency = (
        matrix.groupby(
            [
                "strategy",
                "frequency",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            mean_annualized_return=(
                "annualized_return",
                "mean",
            ),
            mean_sharpe_ratio=(
                "sharpe_ratio",
                "mean",
            ),
            worst_maximum_drawdown=(
                "maximum_drawdown",
                "min",
            ),
            mean_turnover=(
                "turnover",
                "mean",
            ),
        )
        .reset_index()
    )

    return f"""# Day 10 Trend-Family Robustness

## 1. Objective

Evaluate whether the two frozen trend-family baselines retain coherent
behaviour across SPY, QQQ and IWM at 15-, 30- and 60-minute frequencies.
Profitability was not an acceptance criterion.

## 2. Frozen experimental design

The experiment contains exactly 18 runs: two strategies, three symbols and
three frequencies. Annualisation factors are 6,552 for 15-minute bars, 3,276
for 30-minute bars and 1,764 for 60-minute bars. All comparisons are
descriptive and no run was removed because its return or Sharpe was negative.

## 3. Data and aggregation contract

The source is the canonical 117,192-row development dataset from 2020-01-02
through 2025-12-31. Aggregation remained inside symbol and trading-session
boundaries. 60-minute partial closing bars were retained.
The 2026 locked test period was not accessed.

## 4. Strategy configuration lock

Trend Ratio uses the Day 6 baseline configuration `{CONFIGURATION_IDS['trend_ratio']}`.
EMA/MACD uses the Day 8 baseline configuration `{CONFIGURATION_IDS['ema_macd']}`.
No Day 9 best configuration was selected.
Fixed-bar parameters imply longer clock-time horizons at lower frequencies;
the windows were not converted into time-equivalent alternatives.

## 5. Cross-asset results

{_markdown_table(cross_asset)}

## 6. Cross-frequency results

{_markdown_table(cross_frequency)}

## 7. Economic-coherence assessment

Each strategy-symbol-frequency run was retained regardless of economic sign.
The assessment compares direction, drawdown, turnover and exposure with each
strategy's SPY 15-minute reference.
The rules below are diagnostic rules, not performance acceptance criteria.

- Permanently flat: flat exposure at least {PERMANENT_FLAT_EXPOSURE_THRESHOLD:.1f}%.
- Near-permanent direction: long or short exposure at least {NEAR_PERMANENT_DIRECTIONAL_THRESHOLD:.1f}%.
- Zero trades: no position-changing bars.
- Undefined Sharpe: zero return volatility under the project convention.
- Extreme turnover change: ratio below {EXTREME_TURNOVER_RATIO_LOW:.2f} or above {EXTREME_TURNOVER_RATIO_HIGH:.2f}.
- Sign reversal: return or Sharpe sign differs from the strategy reference.
- Materially larger drawdown: drawdown worsens by at least {MATERIAL_DRAWDOWN_INCREASE:.2f}.
- Low active count: active observations below {LOW_ACTIVE_OBSERVATION_PROPORTION:.0%} of observations.

## 8. Degenerate-behaviour flags

{_markdown_table(flag_counts)}

Flags identify cases for investigation; they do not reject or rank
configurations.

## 9. Limitations

These results omit locked-period evidence, execution slippage beyond the
frozen turnover-cost assumption, market impact and walk-forward parameter
re-estimation. Lower-frequency results also mix full bars with the retained
half-hour session-close bar in the 60-minute series.

## 10. Day 10 conclusion

This report provides development-sample robustness evidence, not locked-test results.
It documents cross-asset and cross-frequency behaviour without
optimising, selecting or accepting configurations based on profitability.
"""


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one deterministic CSV artifact."""

    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )


def _write_comparison_chart(
    matrix: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    y_label: str,
    path: Path,
) -> None:
    """Write one deterministic grouped comparison chart."""

    colors = {
        "15min": "#4C78A8",
        "30min": "#F2B134",
        "60min": "#8E6C8A",
    }
    x_locations = np.arange(
        len(ROBUSTNESS_SYMBOLS),
        dtype=float,
    )
    width = 0.24

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
        }
    ):
        figure, axes = plt.subplots(
            1,
            len(ROBUSTNESS_STRATEGIES),
            figsize=(12, 5),
            sharey=True,
            constrained_layout=True,
        )

        for axis, strategy in zip(
            axes,
            ROBUSTNESS_STRATEGIES,
        ):
            strategy_rows = matrix.loc[
                matrix["strategy"].eq(strategy)
            ]

            for index, frequency in enumerate(
                ROBUSTNESS_FREQUENCIES
            ):
                selected = (
                    strategy_rows.loc[
                        strategy_rows[
                            "frequency"
                        ].eq(frequency)
                    ]
                    .set_index("symbol")
                    .reindex(
                        ROBUSTNESS_SYMBOLS
                    )
                )
                values = pd.to_numeric(
                    selected[value_column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                offset = (
                    index
                    - (
                        len(
                            ROBUSTNESS_FREQUENCIES
                        )
                        - 1
                    )
                    / 2.0
                ) * width

                axis.bar(
                    x_locations + offset,
                    values,
                    width=width,
                    label=frequency,
                    color=colors[frequency],
                )

            axis.set_title(
                STRATEGY_DISPLAY_NAMES[
                    strategy
                ]
            )
            axis.set_xticks(
                x_locations,
                labels=ROBUSTNESS_SYMBOLS,
            )
            axis.set_xlabel("Symbol")
            axis.axhline(
                0.0,
                color="#555555",
                linewidth=0.8,
            )

        axes[0].set_ylabel(y_label)
        axes[0].legend(
            title="Frequency",
            frameon=False,
        )
        figure.suptitle(title)
        figure.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
            metadata={
                "Software": "cqf-al",
            },
        )
        plt.close(figure)


def _sha256(path: Path) -> str:
    """Calculate one streaming artifact SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _generation_timestamp(
    supplied: str | None,
) -> str:
    """Normalize the one explicitly variable timestamp."""

    if supplied is None:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    try:
        timestamp = pd.Timestamp(supplied)
    except (TypeError, ValueError) as exc:
        raise Day10RunnerError(
            "generation_timestamp must be a valid "
            "timestamp."
        ) from exc

    if timestamp.tzinfo is None:
        raise Day10RunnerError(
            "generation_timestamp must include a "
            "timezone."
        )

    return (
        timestamp.tz_convert("UTC")
        .isoformat()
        .replace("+00:00", "Z")
    )


def _source_git_commit(
    supplied: str | None,
) -> str:
    """Resolve explicit or repository source provenance."""

    if supplied is not None:
        normalized = supplied.strip()

        if not normalized:
            raise Day10RunnerError(
                "source_git_commit cannot be empty."
            )

        return normalized

    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=find_project_root(),
            text=True,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise Day10RunnerError(
            "Could not resolve the source Git commit."
        ) from exc


def _build_manifest(
    *,
    generation_timestamp: str,
    source_row_count: int,
    source_git_commit: str,
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    """Build deterministic Day 10 artifact provenance."""

    return {
        "day": 10,
        "analysis_name": (
            "trend_family_robustness"
        ),
        "generation_timestamp": (
            generation_timestamp
        ),
        "source_dataset_identifier": (
            DEVELOPMENT_DATASET_ID
        ),
        "source_row_count": source_row_count,
        "development_start": (
            DEVELOPMENT_START.date().isoformat()
        ),
        "development_end": (
            DEVELOPMENT_END.date().isoformat()
        ),
        "symbols": list(ROBUSTNESS_SYMBOLS),
        "frequencies": list(
            ROBUSTNESS_FREQUENCIES
        ),
        "strategies": list(
            ROBUSTNESS_STRATEGIES
        ),
        "annualization_factors": dict(
            ANNUALIZATION_FACTORS
        ),
        "strategy_configuration_identifiers": (
            dict(CONFIGURATION_IDS)
        ),
        "matrix_run_count": 18,
        "output_row_counts_by_frequency": (
            dict(
                EXPECTED_FREQUENCY_ROW_COUNTS
            )
        ),
        "partial_bar_policy": (
            "Retain the final two-input 60-minute "
            "closing bar and mark it partial; 15- and "
            "30-minute bars are complete."
        ),
        "fixed_bar_window_policy": (
            "Keep frozen strategy windows measured in "
            "bars; do not convert them to clock-time "
            "equivalents across frequencies."
        ),
        "source_git_commit": source_git_commit,
        "locked_2026_period_accessed": False,
        "diagnostic_thresholds": dict(
            DIAGNOSTIC_THRESHOLDS
        ),
        "artifact_sha256": dict(
            sorted(
                artifact_hashes.items()
            )
        ),
    }


def _validate_artifact_directory(
    directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Apply conservative output overwrite controls."""

    if directory.exists() and not directory.is_dir():
        raise Day10RunnerError(
            "Artifact path exists but is not a directory."
        )

    if not directory.exists():
        return

    entries = list(directory.iterdir())

    if entries and not overwrite:
        raise Day10RunnerError(
            "Artifact directory is non-empty; pass "
            "--overwrite to replace the approved Day 10 "
            "artifact set."
        )

    nested = sorted(
        path.name
        for path in entries
        if path.is_dir()
    )

    if nested:
        raise Day10RunnerError(
            "Artifact directory contains nested "
            f"directories: {nested}."
        )

    unexpected = sorted(
        path.name
        for path in entries
        if path.name
        not in APPROVED_ARTIFACT_NAMES
    )

    if unexpected:
        raise Day10RunnerError(
            "Artifact directory contains unapproved "
            f"files: {unexpected}."
        )


def _write_artifacts(
    *,
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    report: str,
    artifact_directory: Path,
    overwrite: bool,
    generation_timestamp: str,
    source_row_count: int,
    source_git_commit: str,
) -> tuple[
    dict[str, object],
    tuple[Path, ...],
]:
    """Stage and atomically replace the approved artifacts."""

    _validate_artifact_directory(
        artifact_directory,
        overwrite=overwrite,
    )
    artifact_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".day10-",
        dir=artifact_directory.parent,
    ) as temporary:
        staging = Path(temporary)
        _write_csv(
            matrix,
            staging / "matrix.csv",
        )
        _write_csv(
            summary,
            staging / "summary.csv",
        )
        (
            staging / "report.md"
        ).write_text(
            report,
            encoding="utf-8",
        )
        _write_comparison_chart(
            matrix,
            value_column="sharpe_ratio",
            title=(
                "Day 10 Sharpe Ratio Robustness"
            ),
            y_label="Sharpe ratio",
            path=staging / "sharpe.png",
        )
        _write_comparison_chart(
            matrix,
            value_column=(
                "maximum_drawdown"
            ),
            title=(
                "Day 10 Maximum Drawdown Robustness"
            ),
            y_label="Maximum drawdown",
            path=staging / "drawdown.png",
        )
        _write_comparison_chart(
            matrix,
            value_column="turnover",
            title=(
                "Day 10 Turnover Robustness"
            ),
            y_label="Turnover",
            path=staging / "turnover.png",
        )

        hashed_names = [
            name
            for name in (
                APPROVED_ARTIFACT_NAMES
            )
            if name != "manifest.json"
        ]
        artifact_hashes = {
            name: _sha256(staging / name)
            for name in hashed_names
        }
        manifest = _build_manifest(
            generation_timestamp=(
                generation_timestamp
            ),
            source_row_count=source_row_count,
            source_git_commit=(
                source_git_commit
            ),
            artifact_hashes=artifact_hashes,
        )
        (
            staging / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        staged_names = {
            path.name
            for path in staging.iterdir()
            if path.is_file()
        }

        if staged_names != set(
            APPROVED_ARTIFACT_NAMES
        ):
            raise RuntimeError(
                "Staged Day 10 artifact set is "
                "incomplete."
            )

        for name in APPROVED_ARTIFACT_NAMES:
            os.replace(
                staging / name,
                artifact_directory / name,
            )

    paths = tuple(
        artifact_directory / name
        for name in APPROVED_ARTIFACT_NAMES
    )

    if any(
        not path.exists()
        or path.stat().st_size <= 0
        for path in paths
    ):
        raise RuntimeError(
            "A Day 10 artifact is missing or empty."
        )

    return manifest, paths


def execute_day10(
    *,
    dataset_path: str | Path,
    artifact_directory: str | Path,
    overwrite: bool = False,
    generation_timestamp: str | None = None,
    source_git_commit: str | None = None,
) -> Day10RunResult:
    """Validate, execute and write the complete Day 10 study."""

    source_path = Path(dataset_path)
    output_directory = Path(
        artifact_directory
    )
    _validate_artifact_directory(
        output_directory,
        overwrite=overwrite,
    )

    source = _read_canonical_dataset(
        source_path
    )
    validated = validate_canonical_input(
        source
    )

    specifications = (
        build_robustness_run_matrix()
    )

    if len(specifications) != 18:
        raise RuntimeError(
            "The public robustness matrix must contain "
            "18 runs."
        )

    matrix = validate_matrix_output(
        run_trend_family_robustness(
            validated
        )
    )
    summary = build_summary(matrix)
    report = build_report(
        matrix,
        summary,
    )
    normalized_timestamp = (
        _generation_timestamp(
            generation_timestamp
        )
    )
    normalized_commit = (
        _source_git_commit(
            source_git_commit
        )
    )
    manifest, paths = _write_artifacts(
        matrix=matrix,
        summary=summary,
        report=report,
        artifact_directory=(
            output_directory
        ),
        overwrite=overwrite,
        generation_timestamp=(
            normalized_timestamp
        ),
        source_row_count=len(validated),
        source_git_commit=normalized_commit,
    )

    return Day10RunResult(
        matrix=matrix,
        summary=summary,
        manifest=manifest,
        artifact_directory=(
            output_directory
        ),
        artifact_paths=paths,
    )


def _resolve_from_project_root(
    path: Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve one CLI path relative to the repository."""

    if path.is_absolute():
        return path

    return project_root / path


def _display_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    """Display a repository-relative or absolute CLI path."""

    try:
        return path.relative_to(
            project_root
        ).as_posix()
    except ValueError:
        return path.as_posix()


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the Day 10 command-line workflow."""

    arguments = parse_args(argv)
    project_root = find_project_root()
    dataset_path = _resolve_from_project_root(
        arguments.dataset_path,
        project_root=project_root,
    )
    artifact_directory = (
        _resolve_from_project_root(
            arguments.artifact_directory,
            project_root=project_root,
        )
    )
    result = execute_day10(
        dataset_path=dataset_path,
        artifact_directory=(
            artifact_directory
        ),
        overwrite=arguments.overwrite,
    )

    print(
        "===== DAY 10 TREND-FAMILY "
        "ROBUSTNESS ====="
    )
    print(
        "Source dataset:",
        _display_path(
            dataset_path,
            project_root=project_root,
        ),
    )
    print(
        "Matrix runs:",
        len(result.matrix),
    )
    print(
        "Frequency row counts:",
        EXPECTED_FREQUENCY_ROW_COUNTS,
    )
    print(
        "Configuration identifiers:",
        CONFIGURATION_IDS,
    )
    print("Artifacts:")

    for path in result.artifact_paths:
        print(
            _display_path(
                path,
                project_root=project_root,
            )
        )

    print(
        "Locked 2026 data accessed:",
        False,
    )
    print(
        "DAY 10 TREND-FAMILY ROBUSTNESS "
        "PASSED"
    )


if __name__ == "__main__":
    main()
