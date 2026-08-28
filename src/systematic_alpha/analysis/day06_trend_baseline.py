"""Compact reporting artifacts for the Day 6 trend-ratio baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import make_column_validator
from systematic_alpha.analysis.strategy_performance import (
    build_performance_summary,
    build_wealth_index,
)


ANNUALIZATION_FACTOR: Final[int] = 252 * 26

SUMMARY_SERIES: Final[tuple[str, ...]] = (
    "spy_buy_and_hold",
    "trend_ratio_gross",
    "trend_ratio_net",
)

PERFORMANCE_RETURN_COLUMNS: Final[dict[str, str]] = {
    "spy_buy_and_hold": "close_to_close_simple_return",
    "trend_ratio_gross": "gross_strategy_return",
    "trend_ratio_net": "net_strategy_return",
}

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "series",
    "observations",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
)

SIGNAL_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "observations",
    "signal_warmup_observations",
    "position_warmup_observations",
    "long_signals",
    "short_signals",
    "neutral_signals",
    "long_exposure_pct",
    "short_exposure_pct",
    "flat_exposure_pct",
    "total_turnover",
    "position_changes",
)

ANNUAL_EXPOSURE_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "observations",
    "long_exposure_pct",
    "short_exposure_pct",
    "flat_exposure_pct",
    "total_turnover",
    "position_changes",
)

ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "metadata.json",
    "summary.csv",
    "signal_diagnostics.csv",
    "annual_exposure.csv",
    "cumulative_wealth.png",
    "annual_exposure.png",
    "findings.md",
)

REQUIRED_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_version",
        "dataset_identifier",
        "dataset_manifest_sha256",
        "symbol",
        "frequency",
        "sample_start",
        "sample_end",
        "observations",
        "development_window_start",
        "development_window_end",
        "locked_period_accessed",
        "short_window",
        "long_window",
        "neutral_band",
        "price_column",
        "return_column",
        "transaction_cost_convention",
        "cost_bps_per_turnover",
        "annualization_factor",
        "risk_free_rate_convention",
        "signal_timing",
        "position_timing",
        "overnight_position_rule",
        "session_boundary_rule",
        "signal_warmup_observations",
        "position_warmup_observations",
    }
)


class Day06TrendBaselineError(ValueError):
    """Raised when Day 6 baseline artifacts cannot be built safely."""


_require_columns = make_column_validator(
    Day06TrendBaselineError
)


def _prepare_performance_returns(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Select the three return series used in baseline reporting."""
    if not isinstance(observations, pd.DataFrame):
        raise Day06TrendBaselineError(
            "Baseline observations must be a pandas DataFrame."
        )

    if observations.empty:
        raise Day06TrendBaselineError(
            "Baseline observations cannot be empty."
        )

    _require_columns(
        observations,
        PERFORMANCE_RETURN_COLUMNS.values(),
        context="Day 6 baseline observations",
    )

    performance = pd.DataFrame(
        {
            series: observations[column].copy()
            for series, column in (
                PERFORMANCE_RETURN_COLUMNS.items()
            )
        },
        index=observations.index.copy(),
    )

    buy_and_hold = performance["spy_buy_and_hold"]
    missing_locations = np.flatnonzero(
        buy_and_hold.isna().to_numpy()
    )

    if len(missing_locations):
        if not np.array_equal(
            missing_locations,
            np.array([0]),
        ):
            raise Day06TrendBaselineError(
                "SPY buy-and-hold returns may be missing only "
                "for the first observation."
            )

        performance.iloc[
            0,
            performance.columns.get_loc(
                "spy_buy_and_hold"
            ),
        ] = 0.0

    return performance


def build_baseline_summary(
    observations: pd.DataFrame,
    *,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Build the ordered buy-and-hold, gross and net summary."""
    performance = _prepare_performance_returns(
        observations
    )

    summary = build_performance_summary(
        performance,
        SUMMARY_SERIES,
        annualization_factor=annualization_factor,
    )

    return summary.loc[
        :,
        SUMMARY_COLUMNS,
    ].copy()


def build_annual_exposure(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize annual exposure using eligible positions only."""
    if not isinstance(observations, pd.DataFrame):
        raise Day06TrendBaselineError(
            "Baseline observations must be a pandas DataFrame."
        )

    if observations.empty:
        raise Day06TrendBaselineError(
            "Baseline observations cannot be empty."
        )

    _require_columns(
        observations,
        (
            "timestamp",
            "position",
            "position_eligible",
            "turnover",
        ),
        context="Day 6 annual exposure input",
    )

    frame = observations[
        [
            "timestamp",
            "position",
            "position_eligible",
            "turnover",
        ]
    ].copy()

    try:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise Day06TrendBaselineError(
            "Annual exposure input contains malformed timestamps."
        ) from exc

    if frame["timestamp"].isna().any():
        raise Day06TrendBaselineError(
            "Annual exposure input contains missing timestamps."
        )

    frame["position"] = pd.to_numeric(
        frame["position"],
        errors="coerce",
    )

    if frame["position"].isna().any():
        raise Day06TrendBaselineError(
            "Annual exposure positions must be numeric and nonmissing."
        )

    if not frame["position"].isin((-1, 0, 1)).all():
        raise Day06TrendBaselineError(
            "Annual exposure positions must belong to {-1, 0, 1}."
        )

    eligible_values = frame["position_eligible"]

    if (
        eligible_values.isna().any()
        or not eligible_values.map(
            lambda value: isinstance(
                value,
                (bool, np.bool_),
            )
        ).all()
    ):
        raise Day06TrendBaselineError(
            "position_eligible must contain nonmissing booleans."
        )

    frame["position_eligible"] = (
        eligible_values.astype(bool)
    )

    frame["turnover"] = pd.to_numeric(
        frame["turnover"],
        errors="coerce",
    )

    turnover_values = frame["turnover"].to_numpy(
        dtype="float64"
    )

    if (
        frame["turnover"].isna().any()
        or not np.isfinite(turnover_values).all()
        or frame["turnover"].lt(0.0).any()
    ):
        raise Day06TrendBaselineError(
            "Turnover must be finite, nonmissing and non-negative."
        )

    if not frame["position_eligible"].any():
        raise Day06TrendBaselineError(
            "Annual exposure requires at least one eligible "
            "position observation."
        )

    frame["year"] = frame["timestamp"].dt.year.astype(
        "int64"
    )

    records: list[dict[str, float | int]] = []

    for year, group in frame.groupby(
        "year",
        observed=True,
        sort=True,
    ):
        eligible = group.loc[
            group["position_eligible"]
        ]
        eligible_observations = len(eligible)

        def exposure_percentage(
            position: int,
        ) -> float:
            if eligible_observations == 0:
                return float("nan")

            return float(
                100.0
                * eligible["position"].eq(position).sum()
                / eligible_observations
            )

        records.append(
            {
                "year": int(year),
                "observations": int(len(group)),
                "long_exposure_pct": (
                    exposure_percentage(1)
                ),
                "short_exposure_pct": (
                    exposure_percentage(-1)
                ),
                "flat_exposure_pct": (
                    exposure_percentage(0)
                ),
                "total_turnover": float(
                    group["turnover"].sum()
                ),
                "position_changes": int(
                    group["turnover"].gt(0.0).sum()
                ),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=ANNUAL_EXPOSURE_COLUMNS,
    )


def _prepare_signal_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and select the existing trend diagnostic columns."""
    if not isinstance(diagnostics, pd.DataFrame):
        raise Day06TrendBaselineError(
            "Signal diagnostics must be a pandas DataFrame."
        )

    if len(diagnostics) != 1:
        raise Day06TrendBaselineError(
            "The SPY baseline requires exactly one diagnostics row."
        )

    _require_columns(
        diagnostics,
        SIGNAL_DIAGNOSTIC_COLUMNS,
        context="Day 6 signal diagnostics",
    )

    return diagnostics.loc[
        :,
        SIGNAL_DIAGNOSTIC_COLUMNS,
    ].copy()


def _absolute_path_strings(
    value: object,
) -> list[str]:
    """Find absolute path values before serializing metadata."""
    if isinstance(value, Path):
        return (
            [str(value)]
            if value.is_absolute()
            else []
        )

    if isinstance(value, str):
        if (
            Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
        ):
            return [value]

        return []

    if isinstance(value, Mapping):
        paths: list[str] = []

        for nested in value.values():
            paths.extend(
                _absolute_path_strings(nested)
            )

        return paths

    if isinstance(value, (list, tuple)):
        paths = []

        for nested in value:
            paths.extend(
                _absolute_path_strings(nested)
            )

        return paths

    return []


def _validate_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Require complete, development-only, portable metadata."""
    missing = REQUIRED_METADATA_FIELDS.difference(
        metadata
    )

    if missing:
        raise Day06TrendBaselineError(
            "Day 6 metadata is missing required fields: "
            f"{sorted(missing)}"
        )

    if metadata["locked_period_accessed"] is not False:
        raise Day06TrendBaselineError(
            "Day 6 metadata must confirm "
            "locked_period_accessed is false."
        )

    absolute_paths = _absolute_path_strings(
        metadata
    )

    if absolute_paths:
        raise Day06TrendBaselineError(
            "Day 6 metadata cannot contain absolute paths."
        )

    result = dict(metadata)

    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise Day06TrendBaselineError(
            "Day 6 metadata must be JSON serializable."
        ) from exc

    return result


def build_findings_markdown(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    """Build concise and conservative baseline findings."""
    _require_columns(
        summary,
        SUMMARY_COLUMNS,
        context="Day 6 performance summary",
    )

    diagnostic = _prepare_signal_diagnostics(
        diagnostics
    ).iloc[0]

    indexed = summary.set_index("series")
    missing_series = set(SUMMARY_SERIES).difference(
        indexed.index
    )

    if missing_series:
        raise Day06TrendBaselineError(
            "Day 6 performance summary is missing series: "
            f"{sorted(missing_series)}"
        )

    buy_and_hold = indexed.loc[
        "spy_buy_and_hold"
    ]
    gross = indexed.loc[
        "trend_ratio_gross"
    ]
    net = indexed.loc[
        "trend_ratio_net"
    ]

    cost_impact = float(
        gross["cumulative_return"]
        - net["cumulative_return"]
    )
    costs_material = abs(cost_impact) >= 0.01

    exposure_values = {
        "long": float(
            diagnostic["long_exposure_pct"]
        ),
        "short": float(
            diagnostic["short_exposure_pct"]
        ),
        "neutral": float(
            diagnostic["flat_exposure_pct"]
        ),
    }

    dominant_exposure = max(
        exposure_values,
        key=exposure_values.__getitem__,
    )

    turnover = float(
        diagnostic["total_turnover"]
    )
    turnover_assessment = (
        "appears economically relevant to interpreting the "
        "baseline because it generated cumulative placeholder costs"
        if turnover > 0.0
        else "did not affect this baseline because no turnover occurred"
    )

    comparison = (
        "outperformed"
        if float(net["cumulative_return"])
        > float(buy_and_hold["cumulative_return"])
        else "underperformed"
    )

    materiality = (
        "materially changed"
        if costs_material
        else "did not materially change"
    )

    return f"""# Day 6 Trend-Ratio Baseline Findings

This is one predeclared illustrative baseline. No parameter optimization or
parameter grid was used. All results are development-only, and the locked
period was not accessed.

## Descriptive findings

- SPY buy-and-hold cumulative return: {float(buy_and_hold["cumulative_return"]):.6f}.
- Trend-ratio gross cumulative return: {float(gross["cumulative_return"]):.6f}.
- Trend-ratio net cumulative return: {float(net["cumulative_return"]):.6f}.
- At the stated one-basis-point placeholder cost, costs {materiality} the
  cumulative result under a descriptive one-percentage-point threshold. The
  gross-minus-net cumulative-return difference was {cost_impact:.6f}.
- Total turnover was {turnover:.6f}; turnover {turnover_assessment}.
- Eligible positions were mostly {dominant_exposure}: long
  {exposure_values["long"]:.2f}%, short {exposure_values["short"]:.2f}% and
  neutral {exposure_values["neutral"]:.2f}%.
- On cumulative return, the net baseline {comparison} SPY buy-and-hold. This
  is a descriptive comparison and is not evidence of statistical superiority.

## Limitations

Transaction costs are a placeholder, not a complete execution model. This
single baseline does not establish alpha, robustness, statistical
significance or deployability. Parameter sensitivity and genuine
out-of-sample validation remain future work.
"""


def _save_figure(
    figure: plt.Figure,
    path: Path,
) -> None:
    """Save and close a deterministic, compact figure."""
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={
            "Title": path.stem,
            "Description": (
                "Development-only Day 6 trend-ratio baseline"
            ),
        },
    )
    plt.close(figure)


def create_cumulative_wealth_figure(
    observations: pd.DataFrame,
    path: Path,
) -> None:
    """Plot buy-and-hold, gross and net compounded wealth."""
    _require_columns(
        observations,
        ("timestamp",),
        context="Day 6 cumulative wealth input",
    )

    timestamps = pd.to_datetime(
        observations["timestamp"].copy(),
        utc=True,
        errors="raise",
    )
    performance = _prepare_performance_returns(
        observations
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 5.4)
    )

    labels = {
        "spy_buy_and_hold": "SPY buy-and-hold",
        "trend_ratio_gross": "Trend ratio gross",
        "trend_ratio_net": (
            "Trend ratio net (1 bp placeholder)"
        ),
    }

    for series in SUMMARY_SERIES:
        wealth = build_wealth_index(
            performance[series]
        )
        axis.plot(
            timestamps,
            wealth,
            label=labels[series],
            linewidth=1.25,
        )

    axis.set_title(
        "Development-only cumulative wealth"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Growth of 1.0")
    axis.set_xlim(
        timestamps.iloc[0],
        timestamps.iloc[-1],
    )
    axis.grid(
        alpha=0.25,
        linewidth=0.6,
    )
    axis.legend(
        frameon=False
    )

    _save_figure(
        figure,
        path,
    )


def create_annual_exposure_figure(
    annual_exposure: pd.DataFrame,
    path: Path,
) -> None:
    """Plot annual eligible-position exposure percentages."""
    _require_columns(
        annual_exposure,
        ANNUAL_EXPOSURE_COLUMNS,
        context="Day 6 annual exposure figure input",
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 5.4)
    )

    years = annual_exposure[
        "year"
    ].astype(str)
    long_values = annual_exposure[
        "long_exposure_pct"
    ]
    short_values = annual_exposure[
        "short_exposure_pct"
    ]
    flat_values = annual_exposure[
        "flat_exposure_pct"
    ]

    axis.bar(
        years,
        long_values,
        label="Long",
        color="#2c7fb8",
    )
    axis.bar(
        years,
        short_values,
        bottom=long_values,
        label="Short",
        color="#d95f0e",
    )
    axis.bar(
        years,
        flat_values,
        bottom=long_values + short_values,
        label="Neutral",
        color="#969696",
    )

    axis.set_title(
        "Development-only annual eligible-position exposure"
    )
    axis.set_xlabel("Calendar year")
    axis.set_ylabel("Exposure (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.6,
    )
    axis.legend(
        frameon=False,
        ncol=3,
    )

    _save_figure(
        figure,
        path,
    )


def write_trend_baseline_artifacts(
    *,
    output_dir: Path,
    observations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write seven fixed artifacts, replacing same-name outputs."""
    summary = build_baseline_summary(
        observations
    )
    annual_exposure = build_annual_exposure(
        observations
    )
    signal_diagnostics = (
        _prepare_signal_diagnostics(
            diagnostics
        )
    )
    metadata_payload = _validate_metadata(
        metadata
    )
    findings = build_findings_markdown(
        summary,
        signal_diagnostics,
    )

    destination = Path(output_dir)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        filename: destination / filename
        for filename in ARTIFACT_FILENAMES
    }

    paths["metadata.json"].write_text(
        json.dumps(
            metadata_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary.to_csv(
        paths["summary.csv"],
        index=False,
    )

    signal_diagnostics.to_csv(
        paths["signal_diagnostics.csv"],
        index=False,
    )

    annual_exposure.to_csv(
        paths["annual_exposure.csv"],
        index=False,
    )

    create_cumulative_wealth_figure(
        observations,
        paths["cumulative_wealth.png"],
    )

    create_annual_exposure_figure(
        annual_exposure,
        paths["annual_exposure.png"],
    )

    paths["findings.md"].write_text(
        findings,
        encoding="utf-8",
    )

    return paths
