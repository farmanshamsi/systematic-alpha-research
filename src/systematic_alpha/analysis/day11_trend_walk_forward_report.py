"""Deterministic reporting for Day 11 trend walk-forward validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_alpha.analysis.trend_family_walk_forward import (
    AGGREGATE_RESULT_COLUMNS,
    CONFIGURATION_IDS,
    DEVELOPMENT_START,
    EMA_MACD_PARAMETERS,
    FOLD_RESULT_COLUMNS,
    TREND_RATIO_PARAMETERS,
    WALK_FORWARD_FREQUENCY,
    WALK_FORWARD_STRATEGIES,
    WALK_FORWARD_SYMBOL,
    TrendFamilyWalkForwardResults,
    build_walk_forward_folds,
)


DAY11_ARTIFACT_VERSION: Final[str] = (
    "trend_family_walk_forward_v1"
)
APPROVED_DAY11_ARTIFACT_NAMES: Final[
    tuple[str, ...]
] = (
    "fold_results.csv",
    "aggregate_results.csv",
    "manifest.json",
    "report.md",
    "net_return.png",
    "sharpe.png",
    "drawdown.png",
)
STRATEGY_DISPLAY_NAMES: Final[
    dict[str, str]
] = {
    "trend_ratio": "Trend Ratio",
    "ema_macd": "EMA/MACD",
}
FIGURE_COLORS: Final[dict[str, str]] = {
    "trend_ratio": "#4C78A8",
    "ema_macd": "#8E6C8A",
}


class Day11ReportError(ValueError):
    """Raised when Day 11 artifacts cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class Day11DatasetAudit:
    """Canonical development-dataset lineage used by Day 11."""

    dataset_id: str
    dataset_path: str
    manifest_sha256: str
    canonical_row_count: int
    spy_row_count: int
    spy_session_count: int
    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class Day11ArtifactResult:
    """In-memory report content and written artifact paths."""

    fold_results: pd.DataFrame
    aggregate_results: pd.DataFrame
    manifest: dict[str, object]
    report: str
    artifact_directory: Path
    artifact_paths: tuple[Path, ...]


def _as_utc_timestamp(
    value: object,
    *,
    field_name: str,
) -> pd.Timestamp:
    """Normalize one required audit timestamp to UTC."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise Day11ReportError(
            f"{field_name} must be a valid timestamp."
        ) from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def validate_dataset_audit(
    audit: Day11DatasetAudit,
) -> Day11DatasetAudit:
    """Validate canonical lineage without reading another dataset."""

    if not isinstance(audit, Day11DatasetAudit):
        raise TypeError(
            "audit must be a Day11DatasetAudit."
        )

    if not audit.dataset_id.strip():
        raise Day11ReportError(
            "dataset_id must not be empty."
        )

    path = Path(audit.dataset_path)

    if path.is_absolute():
        raise Day11ReportError(
            "dataset_path must be repository-relative."
        )

    if (
        len(audit.manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in (
                audit.manifest_sha256.lower()
            )
        )
    ):
        raise Day11ReportError(
            "manifest_sha256 must be a 64-character "
            "hexadecimal digest."
        )

    for field_name in (
        "canonical_row_count",
        "spy_row_count",
        "spy_session_count",
    ):
        value = getattr(audit, field_name)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise Day11ReportError(
                f"{field_name} must be a positive integer."
            )

    minimum = _as_utc_timestamp(
        audit.minimum_timestamp,
        field_name="minimum_timestamp",
    )
    maximum = _as_utc_timestamp(
        audit.maximum_timestamp,
        field_name="maximum_timestamp",
    )

    if minimum > maximum:
        raise Day11ReportError(
            "minimum_timestamp must not exceed "
            "maximum_timestamp."
        )

    local_dates = pd.DatetimeIndex(
        [minimum, maximum]
    ).tz_convert(
        "America/New_York"
    ).normalize()

    if (
        local_dates[0].date().isoformat()
        != "2020-01-02"
        or local_dates[1].date().isoformat()
        != "2025-12-31"
    ):
        raise Day11ReportError(
            "Dataset audit must cover only the complete "
            "2020-01-02 through 2025-12-31 development "
            "window."
        )

    if maximum >= pd.Timestamp(
        "2026-01-01",
        tz="UTC",
    ):
        raise Day11ReportError(
            "Dataset audit must not contain 2026 data."
        )

    return Day11DatasetAudit(
        dataset_id=audit.dataset_id.strip(),
        dataset_path=path.as_posix(),
        manifest_sha256=(
            audit.manifest_sha256.lower()
        ),
        canonical_row_count=(
            audit.canonical_row_count
        ),
        spy_row_count=audit.spy_row_count,
        spy_session_count=(
            audit.spy_session_count
        ),
        minimum_timestamp=minimum,
        maximum_timestamp=maximum,
    )


def validate_day11_results(
    results: TrendFamilyWalkForwardResults,
) -> TrendFamilyWalkForwardResults:
    """Validate report inputs against the public Day 11 contract."""

    if not isinstance(
        results,
        TrendFamilyWalkForwardResults,
    ):
        raise TypeError(
            "results must be a "
            "TrendFamilyWalkForwardResults object."
        )

    folds = results.fold_results.copy(
        deep=True
    )
    aggregate = results.aggregate_results.copy(
        deep=True
    )

    if tuple(folds.columns) != FOLD_RESULT_COLUMNS:
        raise Day11ReportError(
            "Fold results do not match the stable Day 11 "
            "schema."
        )

    if (
        tuple(aggregate.columns)
        != AGGREGATE_RESULT_COLUMNS
    ):
        raise Day11ReportError(
            "Aggregate results do not match the stable "
            "Day 11 schema."
        )

    expected_keys = [
        (
            strategy,
            fold.fold_id,
        )
        for strategy in WALK_FORWARD_STRATEGIES
        for fold in build_walk_forward_folds()
    ]
    actual_keys = list(
        folds[
            [
                "strategy",
                "fold_id",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if actual_keys != expected_keys:
        raise Day11ReportError(
            "Fold rows must contain exactly four ordered "
            "folds for each frozen strategy."
        )

    if (
        len(folds) != 8
        or len(aggregate) != 2
        or aggregate["strategy"].tolist()
        != list(WALK_FORWARD_STRATEGIES)
    ):
        raise Day11ReportError(
            "Day 11 artifacts require eight fold rows and "
            "two aggregate rows."
        )

    for frame in (folds, aggregate):
        if not frame["symbol"].eq(
            WALK_FORWARD_SYMBOL
        ).all():
            raise Day11ReportError(
                "Day 11 artifacts are restricted to SPY."
            )

        if not frame["frequency"].eq(
            WALK_FORWARD_FREQUENCY
        ).all():
            raise Day11ReportError(
                "Day 11 artifacts are restricted to "
                "15-minute results."
            )

        expected_ids = frame[
            "strategy"
        ].map(CONFIGURATION_IDS)

        if not frame[
            "configuration_id"
        ].eq(expected_ids).all():
            raise Day11ReportError(
                "Frozen strategy configuration identifiers "
                "changed."
            )

    timestamp_columns = (
        "train_start_timestamp",
        "train_end_timestamp",
        "test_start_timestamp",
        "test_end_timestamp",
    )

    for column in timestamp_columns:
        values = pd.to_datetime(
            folds[column],
            utc=True,
            errors="raise",
        )

        if (
            values.min() < DEVELOPMENT_START
            or values.max()
            >= pd.Timestamp(
                "2026-01-01",
                tz="UTC",
            )
        ):
            raise Day11ReportError(
                f"{column} extends outside the permitted "
                "development window."
            )

        folds[column] = values

    for column in (
        "test_start_timestamp",
        "test_end_timestamp",
    ):
        values = pd.to_datetime(
            aggregate[column],
            utc=True,
            errors="raise",
        )

        if values.max() >= pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ):
            raise Day11ReportError(
                "Aggregate results contain a 2026 "
                "timestamp."
            )

        aggregate[column] = values

    for strategy in WALK_FORWARD_STRATEGIES:
        strategy_folds = folds.loc[
            folds["strategy"].eq(strategy)
        ]
        strategy_aggregate = aggregate.loc[
            aggregate["strategy"].eq(
                strategy
            )
        ]

        if len(strategy_aggregate) != 1:
            raise Day11ReportError(
                "Each strategy requires one aggregate row."
            )

        row = strategy_aggregate.iloc[0]

        if (
            int(row["folds"]) != 4
            or int(row["test_sessions"])
            != int(
                strategy_folds[
                    "test_sessions"
                ].sum()
            )
            or int(row["test_observations"])
            != int(
                strategy_folds[
                    "test_observations"
                ].sum()
            )
        ):
            raise Day11ReportError(
                "Aggregate counts do not reconcile to the "
                "four out-of-sample folds."
            )

    if (
        not folds[
            "initial_test_position"
        ].eq(0).all()
        or not folds[
            "initial_test_turnover"
        ].eq(0.0).all()
        or not folds[
            "purge_sessions"
        ].eq(0).all()
        or not folds[
            "embargo_sessions"
        ].eq(0).all()
    ):
        raise Day11ReportError(
            "Fold reset or purge/embargo integrity checks "
            "failed."
        )

    forbidden_columns = [
        column
        for column in folds.columns
        if any(
            token in column.lower()
            for token in (
                "rank",
                "winner",
                "selected",
                "optimal",
                "sensitivity",
                "training_return",
                "train_return",
            )
        )
    ]

    if forbidden_columns:
        raise Day11ReportError(
            "Fold artifacts contain prohibited selection "
            "or training-return fields: "
            f"{forbidden_columns}."
        )

    finite_columns = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "turnover",
    )

    for frame in (folds, aggregate):
        for column in finite_columns:
            if not pd.to_numeric(
                frame[column],
                errors="coerce",
            ).map(math.isfinite).all():
                raise Day11ReportError(
                    f"Defined {column} values must be "
                    "finite."
                )

        sharpe = pd.to_numeric(
            frame["sharpe_ratio"],
            errors="coerce",
        ).dropna()

        if not sharpe.map(math.isfinite).all():
            raise Day11ReportError(
                "Defined Sharpe-like ratios must be finite."
            )

    return TrendFamilyWalkForwardResults(
        fold_results=folds,
        aggregate_results=aggregate,
    )


def _format_markdown_value(
    value: object,
) -> str:
    """Format one deterministic compact Markdown value."""

    if pd.isna(value):
        return "NA"

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"

    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render one compact DataFrame without optional dependencies."""

    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
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


def _configuration_table() -> pd.DataFrame:
    """Build a readable table of both frozen baselines."""

    return pd.DataFrame(
        [
            {
                "strategy": "Trend Ratio",
                "configuration_id": (
                    CONFIGURATION_IDS[
                        "trend_ratio"
                    ]
                ),
                "parameters": json.dumps(
                    asdict(
                        TREND_RATIO_PARAMETERS
                    ),
                    sort_keys=True,
                ),
            },
            {
                "strategy": "EMA/MACD",
                "configuration_id": (
                    CONFIGURATION_IDS[
                        "ema_macd"
                    ]
                ),
                "parameters": json.dumps(
                    asdict(
                        EMA_MACD_PARAMETERS
                    ),
                    sort_keys=True,
                ),
            },
        ]
    )


def _fold_report_table(
    folds: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    """Select readable implemented fold fields for Markdown."""

    return folds.loc[
        folds["strategy"].eq(strategy),
        [
            "fold_id",
            "train_start_timestamp",
            "train_end_timestamp",
            "test_start_timestamp",
            "test_end_timestamp",
            "train_sessions",
            "test_sessions",
            "test_observations",
            "annualization_factor",
            "initial_test_position",
            "initial_test_turnover",
            "warmup_observations",
            "active_observations",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "turnover",
            "trade_count",
            "average_exposure",
            "long_exposure",
            "short_exposure",
            "flat_exposure",
        ],
    ].reset_index(drop=True)


def _stability_sentence(
    folds: pd.DataFrame,
    *,
    strategy: str,
) -> str:
    """Describe fold variation without ranking strategies."""

    selected = folds.loc[
        folds["strategy"].eq(strategy)
    ]
    returns = selected["cumulative_return"]
    positive = int(returns.gt(0.0).sum())
    lowest = float(returns.min())
    highest = float(returns.max())
    turnover = float(
        selected["turnover"].sum()
    )
    worst_drawdown = float(
        selected["maximum_drawdown"].min()
    )

    return (
        f"{STRATEGY_DISPLAY_NAMES[strategy]} had "
        f"{positive} positive net-return folds out of four; "
        f"fold net returns ranged from {lowest:.4f} to "
        f"{highest:.4f}. Total out-of-sample turnover was "
        f"{turnover:.1f}, and the most negative fold "
        f"drawdown was {worst_drawdown:.4f}."
    )


def build_day11_report(
    results: TrendFamilyWalkForwardResults,
    audit: Day11DatasetAudit,
) -> str:
    """Build the deterministic Day 11 research report."""

    validated = validate_day11_results(
        results
    )
    lineage = validate_dataset_audit(audit)
    folds = validated.fold_results
    aggregate = validated.aggregate_results
    trend_table = _fold_report_table(
        folds,
        strategy="trend_ratio",
    )
    ema_table = _fold_report_table(
        folds,
        strategy="ema_macd",
    )
    aggregate_table = aggregate[
        [
            "strategy",
            "configuration_id",
            "folds",
            "test_sessions",
            "test_observations",
            "annualization_factor",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "turnover",
            "trade_count",
            "average_exposure",
            "long_exposure",
            "short_exposure",
            "flat_exposure",
        ]
    ]

    return f"""# Day 11 — Trend-Family Walk-Forward Validation

## 1. Title and scope

Day 11 evaluates SPY at the primary 15-minute frequency using development
data only. The 2026 locked final-test period was not accessed.

## 2. Scientific purpose

The purpose is to test the temporal stability of the two frozen trend-family
baselines through chronological out-of-sample folds. Day 11 does not tune,
rank, optimise or select parameters, does not use a Day 7 or Day 9
sensitivity winner, and has no profitability gate.

## 3. Locked scope

- Symbol: SPY only.
- Frequency: 15-minute bars only.
- Sample: development observations through 2025-12-31 only.
- Strategies: frozen Day 6 Trend Ratio and frozen Day 8 EMA/MACD baselines.
- Parameter selection and sensitivity-result consultation: none.
- Profitability as an acceptance condition: explicitly excluded.

## 4. Fold design

- `wf_2022`: train 2020-01-02 through 2021-12-31; test calendar year 2022.
- `wf_2023`: train 2020-01-02 through 2022-12-31; test calendar year 2023.
- `wf_2024`: train 2020-01-02 through 2023-12-31; test calendar year 2024.
- `wf_2025`: train 2020-01-02 through 2024-12-31; test calendar year 2025.

Training windows expand from one fixed origin and test years do not overlap.

## 5. Leakage controls

Whole trading sessions define every partition.
Training history is available only to warm indicators.
Training returns, turnover, costs and drawdowns are excluded from test metrics.
Each test fold resets position and delayed execution state to neutral.
The existing one-observation execution delay remains active within the fold.
Train and test sessions never overlap.

Purge and embargo are both zero sessions. They are unnecessary here because
the strategies are fixed rather than fitted in each fold, labels do not span
the boundary, test years do not overlap, execution is delayed by one
observation, and test accounting is reset and isolated. The 2026 locked period
was not accessed.

## 6. Frozen configurations

{_markdown_table(_configuration_table())}

## 7. Dataset audit

- Canonical loader input: `{lineage.dataset_path}`.
- Dataset identifier: `{lineage.dataset_id}`.
- Full canonical development rows: {lineage.canonical_row_count}.
- SPY rows: {lineage.spy_row_count}.
- SPY sessions: {lineage.spy_session_count}.
- Minimum timestamp: {lineage.minimum_timestamp.isoformat()}.
- Maximum timestamp: {lineage.maximum_timestamp.isoformat()}.
- Development window confirmed: 2020-01-02 through 2025-12-31.

## 8. Per-fold results

All performance columns below use test observations only. Net returns already
include the frozen one-basis-point turnover cost. The compact public engine
does not serialize a competing reconstructed gross-return series or per-bar
training data; turnover is retained to document cost intensity.

### Trend Ratio

{_markdown_table(trend_table)}

### EMA/MACD

{_markdown_table(ema_table)}

## 9. Aggregate out-of-sample results

These metrics are recomputed from the chronologically concatenated test
observations from 2022 through 2025. Fold Sharpe ratios and drawdowns are not
averaged.

{_markdown_table(aggregate_table)}

## 10. Stability interpretation

{_stability_sentence(folds, strategy="trend_ratio")}

{_stability_sentence(folds, strategy="ema_macd")}

Variation across calendar years shows that neither frozen rule has uniform
temporal behaviour. Net performance reflects the same frozen transaction-cost
assumption in every fold. Turnover therefore provides descriptive evidence
about cost pressure, while the fold and aggregate drawdowns document different
loss paths.
These comparisons do not declare a winning strategy.
The report does not select either strategy for deployment.

## 11. Acceptance criteria

Day 11 passes on chronological integrity, absence of leakage, deterministic
output, correct fold resets, frozen configurations, valid test-only accounting,
reproducibility and full test coverage.
Profitability is not an acceptance condition.
Negative returns or Sharpe-like ratios remain valid research outcomes.

## 12. Limitations and next step

Evidence is limited to SPY, 15-minute bars, four calendar-year test folds, two
frozen trend-family baselines and the development period. It does not establish
locked-period performance, deployability, market-impact tolerance or parameter
superiority. Any later locked evaluation must remain separate and must not
feed back into configuration choice.
"""


def _normalize_generation_timestamp(
    value: str | None,
) -> str:
    """Normalize the explicitly variable generation timestamp."""

    if value is None:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    timestamp = _as_utc_timestamp(
        value,
        field_name="generation_timestamp",
    )

    return (
        timestamp.isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    """Calculate a streaming artifact SHA-256 digest."""

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


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one deterministic compact CSV artifact."""

    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )


def _write_fold_figure(
    folds: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    y_label: str,
    path: Path,
) -> None:
    """Write one deterministic test-fold comparison figure."""

    x_locations = np.arange(4, dtype=float)
    width = 0.34

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
        }
    ):
        figure, axis = plt.subplots(
            figsize=(10, 5.5),
            constrained_layout=True,
        )

        for index, strategy in enumerate(
            WALK_FORWARD_STRATEGIES
        ):
            selected = folds.loc[
                folds["strategy"].eq(strategy)
            ]
            values = pd.to_numeric(
                selected[value_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            offset = (
                index
                - (
                    len(WALK_FORWARD_STRATEGIES)
                    - 1
                )
                / 2.0
            ) * width

            axis.bar(
                x_locations + offset,
                values,
                width=width,
                label=(
                    STRATEGY_DISPLAY_NAMES[
                        strategy
                    ]
                ),
                color=FIGURE_COLORS[strategy],
            )

        axis.set_xticks(
            x_locations,
            labels=[
                str(year)
                for year in range(2022, 2026)
            ],
        )
        axis.set_xlabel("Out-of-sample test year")
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.axhline(
            0.0,
            color="#555555",
            linewidth=0.8,
        )
        axis.legend(
            title="Frozen strategy",
            frameon=False,
        )
        figure.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
            metadata={
                "Software": "cqf-al",
            },
        )
        plt.close(figure)


def _fold_definitions() -> list[
    dict[str, object]
]:
    """Serialize the frozen fold boundaries deterministically."""

    return [
        {
            "fold_id": fold.fold_id,
            "train_start": (
                fold.train_start.date().isoformat()
            ),
            "train_end_inclusive": (
                (
                    fold.train_end_exclusive
                    - pd.Timedelta(days=1)
                )
                .date()
                .isoformat()
            ),
            "test_start": (
                fold.test_start.date().isoformat()
            ),
            "test_end_inclusive": (
                (
                    fold.test_end_exclusive
                    - pd.Timedelta(days=1)
                )
                .date()
                .isoformat()
            ),
            "purge_sessions": (
                fold.purge_sessions
            ),
            "embargo_sessions": (
                fold.embargo_sessions
            ),
        }
        for fold in build_walk_forward_folds()
    ]


def build_day11_manifest(
    results: TrendFamilyWalkForwardResults,
    audit: Day11DatasetAudit,
    *,
    generation_timestamp: str,
    source_git_commit: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build deterministic Day 11 metadata and integrity evidence."""

    validated = validate_day11_results(
        results
    )
    lineage = validate_dataset_audit(audit)
    folds = validated.fold_results
    aggregate = validated.aggregate_results

    return {
        "day": 11,
        "analysis_name": (
            "trend_family_walk_forward"
        ),
        "artifact_version": (
            DAY11_ARTIFACT_VERSION
        ),
        "generation_timestamp": (
            _normalize_generation_timestamp(
                generation_timestamp
            )
        ),
        "source_dataset_identifier": (
            lineage.dataset_id
        ),
        "source_dataset_path": (
            lineage.dataset_path
        ),
        "source_manifest_sha256": (
            lineage.manifest_sha256
        ),
        "source_git_commit": (
            source_git_commit
        ),
        "canonical_row_count": (
            lineage.canonical_row_count
        ),
        "spy_row_count": (
            lineage.spy_row_count
        ),
        "spy_session_count": (
            lineage.spy_session_count
        ),
        "minimum_timestamp": (
            lineage.minimum_timestamp.isoformat()
        ),
        "maximum_timestamp": (
            lineage.maximum_timestamp.isoformat()
        ),
        "development_start": "2020-01-02",
        "development_end": "2025-12-31",
        "locked_2026_period_accessed": False,
        "symbol": WALK_FORWARD_SYMBOL,
        "frequency": (
            WALK_FORWARD_FREQUENCY
        ),
        "strategies": list(
            WALK_FORWARD_STRATEGIES
        ),
        "strategy_configuration_identifiers": (
            dict(CONFIGURATION_IDS)
        ),
        "strategy_configurations": {
            "trend_ratio": asdict(
                TREND_RATIO_PARAMETERS
            ),
            "ema_macd": asdict(
                EMA_MACD_PARAMETERS
            ),
        },
        "fold_definitions": (
            _fold_definitions()
        ),
        "fold_result_rows": int(
            len(folds)
        ),
        "aggregate_result_rows": int(
            len(aggregate)
        ),
        "aggregate_observation_counts": {
            row.strategy: int(
                row.test_observations
            )
            for row in (
                aggregate.itertuples(
                    index=False
                )
            )
        },
        "annualization_method": (
            "252 multiplied by observed test observations "
            "divided by observed complete test sessions"
        ),
        "indicator_warmup_policy": (
            "Training history may warm indicators; only "
            "test observations enter performance."
        ),
        "execution_reset_policy": (
            "Position and delayed execution reset to "
            "neutral at every test-fold boundary."
        ),
        "execution_delay_observations": 1,
        "purge_sessions": 0,
        "embargo_sessions": 0,
        "parameter_selection_performed": False,
        "profitability_acceptance_gate": False,
        "integrity_checks": {
            "four_folds_per_strategy": True,
            "deterministic_order": True,
            "whole_sessions": True,
            "non_overlapping_test_years": True,
            "train_test_session_overlap": False,
            "training_returns_in_metrics": False,
            "neutral_fold_reset": True,
            "delayed_execution_reset": True,
            "frozen_configurations": True,
            "aggregate_counts_reconciled": True,
            "locked_2026_period_accessed": False,
        },
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
    """Apply conservative Day 11 overwrite controls."""

    if directory.exists() and not directory.is_dir():
        raise Day11ReportError(
            "Artifact path exists but is not a directory."
        )

    if not directory.exists():
        return

    entries = list(directory.iterdir())

    if entries and not overwrite:
        raise Day11ReportError(
            "Artifact directory is non-empty; pass "
            "--overwrite to replace the approved Day 11 "
            "artifact set."
        )

    nested = sorted(
        path.name
        for path in entries
        if path.is_dir()
    )

    if nested:
        raise Day11ReportError(
            "Artifact directory contains nested "
            f"directories: {nested}."
        )

    unexpected = sorted(
        path.name
        for path in entries
        if path.name
        not in APPROVED_DAY11_ARTIFACT_NAMES
    )

    if unexpected:
        raise Day11ReportError(
            "Artifact directory contains unapproved "
            f"files: {unexpected}."
        )


def write_day11_artifacts(
    results: TrendFamilyWalkForwardResults,
    audit: Day11DatasetAudit,
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
    generation_timestamp: str | None = None,
    source_git_commit: str,
) -> Day11ArtifactResult:
    """Write the exact approved Day 11 artifact set atomically."""

    validated = validate_day11_results(
        results
    )
    lineage = validate_dataset_audit(audit)
    report = build_day11_report(
        validated,
        lineage,
    )
    directory = Path(artifact_directory)
    normalized_timestamp = (
        _normalize_generation_timestamp(
            generation_timestamp
        )
    )
    normalized_commit = (
        source_git_commit.strip()
    )

    if not normalized_commit:
        raise Day11ReportError(
            "source_git_commit must not be empty."
        )

    _validate_artifact_directory(
        directory,
        overwrite=overwrite,
    )
    directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".day11-",
        dir=directory.parent,
    ) as temporary:
        staging = Path(temporary)
        _write_csv(
            validated.fold_results,
            staging / "fold_results.csv",
        )
        _write_csv(
            validated.aggregate_results,
            staging / "aggregate_results.csv",
        )
        (
            staging / "report.md"
        ).write_text(
            report,
            encoding="utf-8",
        )
        _write_fold_figure(
            validated.fold_results,
            value_column="cumulative_return",
            title=(
                "Day 11 Out-of-Sample Net Return "
                "by Test Fold"
            ),
            y_label="Net cumulative return",
            path=staging / "net_return.png",
        )
        _write_fold_figure(
            validated.fold_results,
            value_column="sharpe_ratio",
            title=(
                "Day 11 Out-of-Sample Sharpe-like "
                "Ratio by Test Fold"
            ),
            y_label="Sharpe-like ratio",
            path=staging / "sharpe.png",
        )
        _write_fold_figure(
            validated.fold_results,
            value_column="maximum_drawdown",
            title=(
                "Day 11 Out-of-Sample Maximum "
                "Drawdown by Test Fold"
            ),
            y_label="Maximum drawdown",
            path=staging / "drawdown.png",
        )

        hashed_names = [
            name
            for name in (
                APPROVED_DAY11_ARTIFACT_NAMES
            )
            if name != "manifest.json"
        ]
        artifact_hashes = {
            name: _sha256(
                staging / name
            )
            for name in hashed_names
        }
        manifest = build_day11_manifest(
            validated,
            lineage,
            generation_timestamp=(
                normalized_timestamp
            ),
            source_git_commit=(
                normalized_commit
            ),
            artifact_hashes=(
                artifact_hashes
            ),
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
            APPROVED_DAY11_ARTIFACT_NAMES
        ):
            raise RuntimeError(
                "Staged Day 11 artifact set is incomplete."
            )

        for name in (
            APPROVED_DAY11_ARTIFACT_NAMES
        ):
            os.replace(
                staging / name,
                directory / name,
            )

    paths = tuple(
        directory / name
        for name in (
            APPROVED_DAY11_ARTIFACT_NAMES
        )
    )

    if any(
        not path.exists()
        or path.stat().st_size <= 0
        for path in paths
    ):
        raise RuntimeError(
            "A Day 11 artifact is missing or empty."
        )

    return Day11ArtifactResult(
        fold_results=(
            validated.fold_results
        ),
        aggregate_results=(
            validated.aggregate_results
        ),
        manifest=manifest,
        report=report,
        artifact_directory=directory,
        artifact_paths=paths,
    )
