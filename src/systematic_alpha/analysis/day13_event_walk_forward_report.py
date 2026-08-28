"""Build neutral development-only evidence from Day 13 results."""

from __future__ import annotations

from copy import deepcopy
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

from systematic_alpha.analysis.trend_family_event_walk_forward import (
    AGGREGATE_SUMMARY_COLUMNS,
    EVENT_COUNT_COLUMNS,
    FOLD_SUMMARY_COLUMNS,
    PARITY_COLUMNS,
    PARITY_MAPPINGS,
    PARITY_TOLERANCE,
    PERFORMANCE_COLUMNS,
    POSITION_DIAGNOSTIC_COLUMNS,
    STRATEGY_ORDER,
    TrendFamilyEventWalkForwardResults,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    CONFIGURATION_IDS,
)


DAY13_ARTIFACT_VERSION: Final[str] = (
    "trend_family_event_walk_forward_v1"
)
DAY13_ARTIFACT_DIRECTORY_NAME: Final[str] = "day13"
FOLD_SUMMARY_FILENAME: Final[str] = "fold_summary.csv"
EVENT_COUNTS_FILENAME: Final[str] = "event_counts.csv"
POSITION_DIAGNOSTICS_FILENAME: Final[str] = (
    "position_diagnostics.csv"
)
PERFORMANCE_FILENAME: Final[str] = "performance.csv"
VECTORIZED_PARITY_FILENAME: Final[str] = (
    "vectorized_parity.csv"
)
AGGREGATE_SUMMARY_FILENAME: Final[str] = (
    "aggregate_summary.csv"
)
MANIFEST_FILENAME: Final[str] = "manifest.json"
REPORT_FILENAME: Final[str] = "report.md"
APPROVED_DAY13_ARTIFACT_NAMES: Final[
    tuple[str, ...]
] = (
    FOLD_SUMMARY_FILENAME,
    EVENT_COUNTS_FILENAME,
    POSITION_DIAGNOSTICS_FILENAME,
    PERFORMANCE_FILENAME,
    VECTORIZED_PARITY_FILENAME,
    AGGREGATE_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
)
FOLD_ORDER: Final[tuple[str, ...]] = (
    "wf_2022",
    "wf_2023",
    "wf_2024",
    "wf_2025",
)
SERIES_ORDER: Final[tuple[str, ...]] = (
    "gross",
    "net",
)
EXPECTED_ROW_COUNTS: Final[
    dict[str, int]
] = {
    "fold_summary": 8,
    "event_counts": 8,
    "position_diagnostics": 8,
    "performance": 16,
    "vectorized_parity": 64,
    "aggregate_summary": 4,
}


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Return a defensive zero-based copy of one evidence table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    return frame.copy(deep=True).reset_index(drop=True)


def _freeze_manifest_value(
    value: object,
) -> object:
    """Recursively freeze mutable manifest containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_manifest_value(
                    item
                )
                for key, item in deepcopy(
                    dict(value)
                ).items()
            }
        )

    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_manifest_value(item)
            for item in deepcopy(value)
        )

    return deepcopy(value)


def _copy_manifest_value(
    value: object,
) -> object:
    """Return mutable copies of recursively frozen manifest values."""

    if isinstance(value, Mapping):
        return {
            str(key): _copy_manifest_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            _copy_manifest_value(item)
            for item in value
        ]

    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class Day13EventWalkForwardReport:
    """Copied deterministic Day 13 tables, manifest, and Markdown."""

    fold_summary: pd.DataFrame
    event_counts: pd.DataFrame
    position_diagnostics: pd.DataFrame
    performance: pd.DataFrame
    vectorized_parity: pd.DataFrame
    aggregate_summary: pd.DataFrame
    manifest: Mapping[str, object]
    report: str

    def __post_init__(self) -> None:
        """Defensively retain every mutable report component."""

        for name in (
            "fold_summary",
            "event_counts",
            "position_diagnostics",
            "performance",
            "vectorized_parity",
            "aggregate_summary",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(
                    getattr(self, name),
                    name=name,
                ),
            )

        if not isinstance(self.manifest, Mapping):
            raise TypeError(
                "manifest must be a mapping."
            )

        frozen_manifest = _freeze_manifest_value(
            self.manifest
        )
        object.__setattr__(
            self,
            "manifest",
            frozen_manifest,
        )

        if not isinstance(self.report, str):
            raise TypeError(
                "report must be Markdown text."
            )

    def copy_fold_summary(self) -> pd.DataFrame:
        """Return a mutable fold-summary copy."""

        return self.fold_summary.copy(deep=True)

    def copy_event_counts(self) -> pd.DataFrame:
        """Return a mutable event-count copy."""

        return self.event_counts.copy(deep=True)

    def copy_position_diagnostics(
        self,
    ) -> pd.DataFrame:
        """Return a mutable position-diagnostic copy."""

        return self.position_diagnostics.copy(
            deep=True
        )

    def copy_performance(self) -> pd.DataFrame:
        """Return a mutable fold-performance copy."""

        return self.performance.copy(deep=True)

    def copy_vectorized_parity(
        self,
    ) -> pd.DataFrame:
        """Return a mutable vectorised-parity copy."""

        return self.vectorized_parity.copy(
            deep=True
        )

    def copy_aggregate_summary(
        self,
    ) -> pd.DataFrame:
        """Return a mutable aggregate-summary copy."""

        return self.aggregate_summary.copy(
            deep=True
        )

    def copy_manifest(self) -> dict[str, object]:
        """Return a mutable recursive copy of the manifest."""

        copied = _copy_manifest_value(
            self.manifest
        )

        if not isinstance(copied, dict):
            raise RuntimeError(
                "Frozen manifest did not copy to a dictionary."
            )

        return copied


def _validated_table(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Require one exact deterministic Day 13 evidence schema."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if tuple(frame.columns) != columns:
        raise ValueError(
            f"{name} does not match its frozen Day 13 schema."
        )

    expected_rows = EXPECTED_ROW_COUNTS[
        name
    ]

    if len(frame) != expected_rows:
        raise ValueError(
            f"{name} must contain exactly {expected_rows} rows."
        )

    return frame.copy(
        deep=True
    ).reset_index(drop=True)


def _expected_fold_keys() -> list[
    tuple[str, str]
]:
    """Return the frozen strategy-major fold ordering."""

    return [
        (
            strategy,
            fold_id,
        )
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_ORDER
    ]


def _require_key_order(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    expected: list[tuple[object, ...]],
    name: str,
) -> None:
    """Require unique rows in one exact deterministic key order."""

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing ordering columns: {missing}."
        )

    actual = list(
        frame.loc[
            :,
            columns,
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if actual != expected:
        raise ValueError(
            f"{name} does not follow the frozen Day 13 row order."
        )

    if frame.duplicated(
        list(columns)
    ).any():
        raise ValueError(
            f"{name} contains duplicate scientific keys."
        )


def _require_finite(
    value: object,
    *,
    name: str,
) -> float:
    """Require one finite real report value."""

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(normalized):
        raise ValueError(
            f"{name} must be finite."
        )

    return normalized


def _require_close(
    value: object,
    expected: float,
    *,
    name: str,
) -> None:
    """Require one reset identity within the frozen tolerance."""

    normalized = _require_finite(
        value,
        name=name,
    )

    if not math.isclose(
        normalized,
        expected,
        rel_tol=PARITY_TOLERANCE,
        abs_tol=PARITY_TOLERANCE,
    ):
        raise ValueError(
            f"{name} violates the neutral fold-reset contract."
        )


def _validate_fold_runs(
    results: TrendFamilyEventWalkForwardResults,
) -> None:
    """Require exactly eight strategy-major immutable replay runs."""

    keys = [
        (
            run.strategy,
            run.fold_id,
        )
        for run in results.fold_runs
    ]

    if keys != _expected_fold_keys():
        raise ValueError(
            "fold_runs must contain exactly eight frozen "
            "strategy/fold replays in deterministic order."
        )


def _validate_fold_summary(
    frame: pd.DataFrame,
    parity: pd.DataFrame,
) -> None:
    """Validate fold scope, boundaries, counts, and parity status."""

    _require_key_order(
        frame,
        columns=(
            "strategy",
            "fold_id",
        ),
        expected=[
            tuple(key)
            for key in _expected_fold_keys()
        ],
        name="fold_summary",
    )
    development_end = pd.Timestamp(
        "2026-01-01",
        tz="UTC",
    )

    for row in frame.itertuples(
        index=False
    ):
        if (
            row.strategy
            not in STRATEGY_ORDER
            or row.fold_id
            not in FOLD_ORDER
        ):
            raise ValueError(
                "fold_summary contains an unsupported "
                "strategy or fold."
            )

        if (
            row.configuration_id
            != CONFIGURATION_IDS[
                row.strategy
            ]
            or row.symbol != "SPY"
            or row.frequency != "15min"
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} metadata "
                "does not match the frozen configuration."
            )

        if any(
            int(value) <= 0
            for value in (
                row.train_observations,
                row.test_observations,
                row.train_sessions,
                row.test_sessions,
                row.indicator_history_observations,
            )
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} requires "
                "positive observation and session counts."
            )

        train_start = pd.Timestamp(
            row.train_start_timestamp
        )
        train_end = pd.Timestamp(
            row.train_end_exclusive
        )
        test_start = pd.Timestamp(
            row.test_start_timestamp
        )
        test_end = pd.Timestamp(
            row.test_end_exclusive
        )
        evaluation_start = pd.Timestamp(
            row.evaluation_start
        )
        evaluation_end = pd.Timestamp(
            row.evaluation_end_exclusive
        )
        timestamps = (
            train_start,
            train_end,
            test_start,
            test_end,
            evaluation_start,
            evaluation_end,
        )

        if not all(
            timestamp.tzinfo is not None
            for timestamp in timestamps
        ):
            raise ValueError(
                "Fold boundaries must be timezone-aware."
            )

        timestamps = tuple(
            timestamp.tz_convert("UTC")
            for timestamp in timestamps
        )
        (
            train_start,
            train_end,
            test_start,
            test_end,
            evaluation_start,
            evaluation_end,
        ) = timestamps
        year = int(
            row.fold_id.removeprefix(
                "wf_"
            )
        )

        if (
            train_start
            != pd.Timestamp(
                "2020-01-02",
                tz="UTC",
            )
            or train_end
            != pd.Timestamp(
                f"{year}-01-01",
                tz="UTC",
            )
            or test_start != train_end
            or test_end
            != pd.Timestamp(
                f"{year + 1}-01-01",
                tz="UTC",
            )
            or not (
                test_start
                <= evaluation_start
                < evaluation_end
                <= test_end
                <= development_end
            )
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} boundaries "
                "violate the frozen expanding protocol."
            )

        if evaluation_end >= development_end:
            raise ValueError(
                f"{row.strategy}/{row.fold_id} evaluation "
                "evidence cannot enter the locked period."
            )

        _require_close(
            row.initial_position,
            0.0,
            name=(
                f"{row.strategy}/{row.fold_id} "
                "initial_position"
            ),
        )
        _require_close(
            row.initial_equity,
            1.0,
            name=(
                f"{row.strategy}/{row.fold_id} "
                "initial_equity"
            ),
        )

        if int(
            row.parity_comparisons
        ) != 8:
            raise ValueError(
                "Every fold requires eight parity comparisons."
            )

        group = parity.loc[
            parity[
                "strategy"
            ].eq(row.strategy)
            & parity[
                "fold_id"
            ].eq(row.fold_id)
        ]
        observed_status = bool(
            group[
                "passed"
            ].astype(bool).all()
        )

        if (
            len(group) != 8
            or bool(
                row.parity_passed
            )
            != observed_status
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} parity "
                "summary is inconsistent."
            )


def _validate_event_counts(
    frame: pd.DataFrame,
) -> None:
    """Require actual event records to reconcile with replay rows."""

    _require_key_order(
        frame,
        columns=(
            "strategy",
            "fold_id",
        ),
        expected=[
            tuple(key)
            for key in _expected_fold_keys()
        ],
        name="event_counts",
    )

    for row in frame.itertuples(
        index=False
    ):
        counts = (
            row.market_bar_events,
            row.signal_events,
            row.order_events,
            row.fill_events,
            row.portfolio_snapshots,
            row.total_events,
            row.observations,
        )

        if any(
            int(value) < 0
            for value in counts
        ):
            raise ValueError(
                "Event counts must be nonnegative."
            )

        if (
            int(
                row.market_bar_events
            )
            != int(row.observations)
            or int(
                row.signal_events
            )
            != int(row.observations)
            or int(
                row.portfolio_snapshots
            )
            != int(row.observations)
            or int(row.total_events)
            != sum(
                int(value)
                for value in (
                    row.market_bar_events,
                    row.signal_events,
                    row.order_events,
                    row.fill_events,
                    row.portfolio_snapshots,
                )
            )
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} event counts "
                "do not reconcile."
            )


def _validate_position_diagnostics(
    frame: pd.DataFrame,
) -> None:
    """Require neutral execution, pending state, and equity resets."""

    _require_key_order(
        frame,
        columns=(
            "strategy",
            "fold_id",
        ),
        expected=[
            tuple(key)
            for key in _expected_fold_keys()
        ],
        name="position_diagnostics",
    )
    reset_values = (
        (
            "initial_previous_position",
            0.0,
        ),
        (
            "initial_position",
            0.0,
        ),
        (
            "initial_turnover",
            0.0,
        ),
        (
            "initial_transaction_cost",
            0.0,
        ),
        (
            "initial_previous_equity",
            1.0,
        ),
        (
            "initial_cash_balance",
            1.0,
        ),
        (
            "initial_holdings_value",
            0.0,
        ),
        (
            "initial_ending_equity",
            1.0,
        ),
    )

    for row in frame.itertuples(
        index=False
    ):
        for name, expected in reset_values:
            _require_close(
                getattr(row, name),
                expected,
                name=(
                    f"{row.strategy}/{row.fold_id} "
                    f"{name}"
                ),
            )

        if (
            bool(
                row.initial_position_eligible
            )
            or bool(
                row.initial_fill_executed
            )
        ):
            raise ValueError(
                f"{row.strategy}/{row.fold_id} did not "
                "clear pending execution state."
            )

        for name in (
            "total_turnover",
            "total_fractional_transaction_cost",
            "total_transaction_cost_amount",
        ):
            if _require_finite(
                getattr(row, name),
                name=name,
            ) < 0.0:
                raise ValueError(
                    f"{name} must be nonnegative."
                )

        if _require_finite(
            row.final_equity,
            name="final_equity",
        ) <= 0.0:
            raise ValueError(
                "final_equity must be positive."
            )


def _validate_parity(
    frame: pd.DataFrame,
) -> None:
    """Validate all sixty-four exact and numerical comparisons."""

    expected = [
        (
            strategy,
            fold_id,
            replay_column,
            comparison_type,
        )
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_ORDER
        for (
            replay_column,
            _,
            comparison_type,
        ) in PARITY_MAPPINGS
    ]
    _require_key_order(
        frame,
        columns=(
            "strategy",
            "fold_id",
            "comparison",
            "comparison_type",
        ),
        expected=[
            tuple(key)
            for key in expected
        ],
        name="vectorized_parity",
    )

    for row in frame.itertuples(
        index=False
    ):
        row_count = int(
            row.row_count
        )
        mismatch_count = int(
            row.mismatch_count
        )
        maximum_difference = (
            _require_finite(
                row.maximum_absolute_difference,
                name=(
                    "maximum_absolute_difference"
                ),
            )
        )
        expected_tolerance = (
            0.0
            if row.comparison_type
            == "exact"
            else PARITY_TOLERANCE
        )

        if (
            row_count < 0
            or mismatch_count < 0
            or mismatch_count
            > row_count
            or maximum_difference < 0.0
        ):
            raise ValueError(
                "Parity counts and differences must be "
                "nonnegative and internally consistent."
            )

        if not math.isclose(
            float(row.tolerance),
            expected_tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Parity tolerance does not match comparison type."
            )

        if bool(
            row.passed
        ) != (
            mismatch_count == 0
        ):
            raise ValueError(
                "Parity passed status must match mismatch_count."
            )

    group_sizes = frame.groupby(
        [
            "strategy",
            "fold_id",
            "comparison_type",
        ],
        observed=True,
        sort=False,
    ).size()

    for strategy, fold_id in (
        _expected_fold_keys()
    ):
        if (
            int(
                group_sizes.loc[
                    (
                        strategy,
                        fold_id,
                        "exact",
                    )
                ]
            )
            != 4
            or int(
                group_sizes.loc[
                    (
                        strategy,
                        fold_id,
                        "numeric",
                    )
                ]
            )
            != 4
        ):
            raise ValueError(
                "Every fold requires four exact and four "
                "numeric parity comparisons."
            )


def _validate_performance(
    fold_performance: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    """Validate complete neutral gross/net performance evidence."""

    expected_fold = [
        (
            strategy,
            fold_id,
            series,
        )
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_ORDER
        for series in SERIES_ORDER
    ]
    _require_key_order(
        fold_performance,
        columns=(
            "strategy",
            "fold_id",
            "series",
        ),
        expected=[
            tuple(key)
            for key in expected_fold
        ],
        name="performance",
    )
    expected_aggregate = [
        (
            strategy,
            series,
        )
        for strategy in STRATEGY_ORDER
        for series in SERIES_ORDER
    ]
    _require_key_order(
        aggregate,
        columns=(
            "strategy",
            "series",
        ),
        expected=[
            tuple(key)
            for key in expected_aggregate
        ],
        name="aggregate_summary",
    )

    for frame in (
        fold_performance,
        aggregate,
    ):
        for row in frame.itertuples(
            index=False
        ):
            if (
                int(row.observations) <= 0
                or int(row.sessions) <= 0
                or _require_finite(
                    row.annualization_factor,
                    name=(
                        "annualization_factor"
                    ),
                )
                <= 0.0
            ):
                raise ValueError(
                    "Performance counts and annualization "
                    "must be positive."
                )

            for name in (
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "maximum_drawdown",
                "final_wealth",
            ):
                _require_finite(
                    getattr(row, name),
                    name=name,
                )

            sharpe = float(
                row.sharpe_ratio
            )

            if math.isinf(sharpe):
                raise ValueError(
                    "sharpe_ratio cannot be infinite."
                )

            if float(
                row.annualized_volatility
            ) < 0.0:
                raise ValueError(
                    "annualized_volatility cannot be negative."
                )

            if float(
                row.maximum_drawdown
            ) > PARITY_TOLERANCE:
                raise ValueError(
                    "maximum_drawdown cannot be positive."
                )

            if float(
                row.final_wealth
            ) <= 0.0:
                raise ValueError(
                    "final_wealth must be positive."
                )

            _require_close(
                row.final_wealth,
                1.0
                + float(
                    row.cumulative_return
                ),
                name="final_wealth",
            )

    if not aggregate[
        "folds"
    ].eq(4).all():
        raise ValueError(
            "Aggregate evidence must contain four folds."
        )


def _validate_cross_table_counts(
    fold_summary: pd.DataFrame,
    event_counts: pd.DataFrame,
    position_diagnostics: pd.DataFrame,
) -> None:
    """Require matching run keys and observation counts across tables."""

    summary = fold_summary.set_index(
        [
            "strategy",
            "fold_id",
        ]
    )
    events = event_counts.set_index(
        [
            "strategy",
            "fold_id",
        ]
    )
    positions = (
        position_diagnostics.set_index(
            [
                "strategy",
                "fold_id",
            ]
        )
    )

    if (
        not summary.index.equals(
            events.index
        )
        or not summary.index.equals(
            positions.index
        )
        or not summary[
            "test_observations"
        ].astype("int64").equals(
            events[
                "observations"
            ].astype("int64")
        )
    ):
        raise ValueError(
            "Fold summaries, events, and reset diagnostics "
            "must describe the same eight replay runs."
        )


def _format_markdown_value(
    value: object,
) -> str:
    """Format one deterministic compact Markdown cell."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"

        return f"{value:.6g}"

    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render one compact DataFrame without optional dependencies."""

    headers = [
        str(column)
        for column in frame.columns
    ]
    lines = [
        "| "
        + " | ".join(headers)
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |",
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(
                    value
                )
                for value in row
            )
            + " |"
        )

    return "\n".join(lines)


def _render_report(
    *,
    fold_summary: pd.DataFrame,
    event_counts: pd.DataFrame,
    position_diagnostics: pd.DataFrame,
    performance: pd.DataFrame,
    parity: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> str:
    """Render neutral deterministic Day 13 research Markdown."""

    fold_table = fold_summary[
        [
            "strategy",
            "fold_id",
            "train_observations",
            "test_observations",
            "train_sessions",
            "test_sessions",
            "evaluation_start",
            "evaluation_end_exclusive",
            "initial_position",
            "initial_equity",
            "parity_passed",
        ]
    ]
    event_table = event_counts[
        [
            "strategy",
            "fold_id",
            "market_bar_events",
            "signal_events",
            "order_events",
            "fill_events",
            "portfolio_snapshots",
            "total_events",
        ]
    ]
    reset_table = position_diagnostics[
        [
            "strategy",
            "fold_id",
            "initial_previous_position",
            "initial_position",
            "initial_position_eligible",
            "initial_turnover",
            "initial_transaction_cost",
            "initial_previous_equity",
            "initial_cash_balance",
            "initial_holdings_value",
            "initial_ending_equity",
            "initial_fill_executed",
        ]
    ]
    parity_summary = (
        parity.groupby(
            [
                "strategy",
                "fold_id",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            comparisons=(
                "comparison",
                "size",
            ),
            passed=(
                "passed",
                "sum",
            ),
            mismatches=(
                "mismatch_count",
                "sum",
            ),
            maximum_absolute_difference=(
                "maximum_absolute_difference",
                "max",
            ),
        )
        .reset_index()
    )
    fold_performance = performance[
        [
            "strategy",
            "fold_id",
            "series",
            "annualization_factor",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "final_wealth",
        ]
    ]
    aggregate_table = aggregate[
        [
            "strategy",
            "series",
            "folds",
            "observations",
            "sessions",
            "annualization_factor",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "final_wealth",
        ]
    ]

    return f"""# Day 13 — Event-Driven Trend-Family Walk-Forward

## 1. Scope and frozen protocol

This is development data only: SPY only, 15-minute bars only, using the
frozen Trend Ratio and EMA/MACD baselines over four fixed expanding folds.
There was no tuning, no ranking, and no winner selection. No 2026 data was
accessed, and this evidence is not final locked-period performance.

## 2. Scientific question

The study asks whether the frozen event-driven executions reproduce the
independent vectorised walk-forward reference through chronological test
folds without carrying execution or accounting state across fold boundaries.

## 3. Fold definitions

Training expands from 2020 through the year before each test year.
Training history warms indicators, while each evaluation contains one complete
calendar-year test fold.

{_markdown_table(fold_table)}

## 4. Event-driven replay and warm-up design

Execution events are test-only. Signals are formed on completed bars and use
a one-observation execution delay. Ordinary sessions within a fold preserve
state; training observations supply feature history but never execution
records.

## 5. Fold-boundary reset and leakage controls

Every fold starts from a neutral position. Pending execution state is cleared
at every fold start, and equity reset to 1.0 is enforced. These diagnostics
demonstrate no training P&L leakage and no fold-boundary position leakage.

{_markdown_table(reset_table)}

## 6. Event counts and execution diagnostics

{_markdown_table(event_table)}

## 7. Vectorised/event-driven parity

All eight comparisons remain visible for every run, including any observed
failure, mismatch count, and maximum absolute difference.

{_markdown_table(parity_summary)}

## 8. Fold-level gross and net performance

Gross and net evidence is presented neutrally using observed-session
annualisation.

{_markdown_table(fold_performance)}

## 9. Aggregate walk-forward evidence

Chronological out-of-sample aggregation concatenates the four independently
reset return streams. Aggregate metrics are recomputed rather than averaged;
portfolio and pending-order state are not linked between folds.

{_markdown_table(aggregate_table)}

## 10. Interpretation and limitations

The evidence covers two frozen strategies, SPY, 15-minute bars, four
development-period folds, normalized-notional fills, and the existing cost
model. It does not establish locked-period performance, deployability,
market-impact behavior, partial-fill behavior, or live-broker readiness.

## 11. Reproducibility and manifest

The exact table schemas, row order, reset diagnostics, parity evidence, and
artifact SHA-256 hashes are recorded in `{MANIFEST_FILENAME}`. The evidence
files are `{FOLD_SUMMARY_FILENAME}`, `{EVENT_COUNTS_FILENAME}`,
`{POSITION_DIAGNOSTICS_FILENAME}`, `{PERFORMANCE_FILENAME}`,
`{VECTORIZED_PARITY_FILENAME}`, and `{AGGREGATE_SUMMARY_FILENAME}`.

## 12. Day 13 conclusion

Day 13 acceptance concerns deterministic orchestration, fold isolation,
accounting integrity, and transparent parity diagnostics. Performance signs
and magnitudes are descriptive research outcomes.
"""


def _csv_bytes(
    frame: pd.DataFrame,
) -> bytes:
    """Serialize one table with stable repository-compatible settings."""

    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def _report_bytes(
    report: str,
) -> bytes:
    """Serialize Markdown with one stable final newline."""

    return (
        report.rstrip()
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(
    payload: bytes,
) -> str:
    """Calculate one deterministic SHA-256 digest."""

    return hashlib.sha256(
        payload
    ).hexdigest()


def _artifact_payloads(
    report: Day13EventWalkForwardReport,
) -> dict[str, bytes]:
    """Build every non-manifest artifact payload in fixed order."""

    return {
        FOLD_SUMMARY_FILENAME: _csv_bytes(
            report.fold_summary
        ),
        EVENT_COUNTS_FILENAME: _csv_bytes(
            report.event_counts
        ),
        POSITION_DIAGNOSTICS_FILENAME: (
            _csv_bytes(
                report.position_diagnostics
            )
        ),
        PERFORMANCE_FILENAME: _csv_bytes(
            report.performance
        ),
        VECTORIZED_PARITY_FILENAME: (
            _csv_bytes(
                report.vectorized_parity
            )
        ),
        AGGREGATE_SUMMARY_FILENAME: (
            _csv_bytes(
                report.aggregate_summary
            )
        ),
        REPORT_FILENAME: _report_bytes(
            report.report
        ),
    }


def _manifest(
    *,
    fold_summary: pd.DataFrame,
    parity: pd.DataFrame,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build deterministic development-only Day 13 provenance."""

    configuration_ids = {
        strategy: CONFIGURATION_IDS[
            strategy
        ]
        for strategy in STRATEGY_ORDER
    }

    return {
        "report_id": (
            "day13_event_walk_forward"
        ),
        "artifact_version": (
            DAY13_ARTIFACT_VERSION
        ),
        "schema_version": 1,
        "artifact_filenames": list(
            APPROVED_DAY13_ARTIFACT_NAMES
        ),
        "development_only": True,
        "symbol": "SPY",
        "frequency": "15min",
        "strategies": list(
            STRATEGY_ORDER
        ),
        "folds": list(FOLD_ORDER),
        "configuration_ids": (
            configuration_ids
        ),
        "evaluation_start": pd.Timestamp(
            fold_summary[
                "evaluation_start"
            ].min()
        ).isoformat(),
        "evaluation_end_exclusive": pd.Timestamp(
            fold_summary[
                "evaluation_end_exclusive"
            ].max()
        ).isoformat(),
        "locked_period_accessed": False,
        "tuning_performed": False,
        "ranking_performed": False,
        "winner_selection_performed": False,
        "row_counts": dict(
            EXPECTED_ROW_COUNTS
        ),
        "replay_run_count": 8,
        "parity_comparison_count": int(
            len(parity)
        ),
        "parity_passed_count": int(
            parity[
                "passed"
            ].astype(bool).sum()
        ),
        "parity_failed_count": int(
            (
                ~parity[
                    "passed"
                ].astype(bool)
            ).sum()
        ),
        "artifact_sha256": {
            name: artifact_hashes[name]
            for name in (
                APPROVED_DAY13_ARTIFACT_NAMES
            )
            if name != MANIFEST_FILENAME
        },
    }


def build_day13_event_walk_forward_report(
    results: TrendFamilyEventWalkForwardResults,
) -> Day13EventWalkForwardReport:
    """Build deterministic neutral evidence from Day 13 results."""

    if not isinstance(
        results,
        TrendFamilyEventWalkForwardResults,
    ):
        raise TypeError(
            "results must be a "
            "TrendFamilyEventWalkForwardResults."
        )

    _validate_fold_runs(results)
    fold_summary = _validated_table(
        results.fold_summary,
        name="fold_summary",
        columns=FOLD_SUMMARY_COLUMNS,
    )
    event_counts = _validated_table(
        results.event_counts,
        name="event_counts",
        columns=EVENT_COUNT_COLUMNS,
    )
    position_diagnostics = (
        _validated_table(
            results.position_diagnostics,
            name=(
                "position_diagnostics"
            ),
            columns=(
                POSITION_DIAGNOSTIC_COLUMNS
            ),
        )
    )
    performance = _validated_table(
        results.performance,
        name="performance",
        columns=PERFORMANCE_COLUMNS,
    )
    parity = _validated_table(
        results.vectorized_parity,
        name="vectorized_parity",
        columns=PARITY_COLUMNS,
    )
    aggregate = _validated_table(
        results.aggregate_summary,
        name="aggregate_summary",
        columns=(
            AGGREGATE_SUMMARY_COLUMNS
        ),
    )
    _validate_parity(parity)
    _validate_fold_summary(
        fold_summary,
        parity,
    )
    _validate_event_counts(
        event_counts
    )
    _validate_position_diagnostics(
        position_diagnostics
    )
    _validate_performance(
        performance,
        aggregate,
    )
    _validate_cross_table_counts(
        fold_summary,
        event_counts,
        position_diagnostics,
    )
    markdown = _render_report(
        fold_summary=fold_summary,
        event_counts=event_counts,
        position_diagnostics=(
            position_diagnostics
        ),
        performance=performance,
        parity=parity,
        aggregate=aggregate,
    )
    preliminary = (
        Day13EventWalkForwardReport(
            fold_summary=fold_summary,
            event_counts=event_counts,
            position_diagnostics=(
                position_diagnostics
            ),
            performance=performance,
            vectorized_parity=parity,
            aggregate_summary=aggregate,
            manifest={},
            report=markdown,
        )
    )
    payloads = _artifact_payloads(
        preliminary
    )
    artifact_hashes = {
        name: _sha256_bytes(payload)
        for name, payload in (
            payloads.items()
        )
    }
    manifest = _manifest(
        fold_summary=fold_summary,
        parity=parity,
        artifact_hashes=(
            artifact_hashes
        ),
    )

    return Day13EventWalkForwardReport(
        fold_summary=fold_summary,
        event_counts=event_counts,
        position_diagnostics=(
            position_diagnostics
        ),
        performance=performance,
        vectorized_parity=parity,
        aggregate_summary=aggregate,
        manifest=manifest,
        report=markdown,
    )


def _validate_output_directory(
    directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Apply conservative replacement controls before writing."""

    if directory.exists() and not directory.is_dir():
        raise ValueError(
            "Day 13 output path exists but is not a directory."
        )

    if not directory.exists():
        return

    entries = list(
        directory.iterdir()
    )

    if entries and not overwrite:
        raise ValueError(
            "Day 13 output directory is non-empty; "
            "explicit overwrite is required."
        )

    nested = sorted(
        path.name
        for path in entries
        if path.is_dir()
    )

    if nested:
        raise ValueError(
            "Day 13 output contains nested directories: "
            f"{nested}."
        )

    unexpected = sorted(
        path.name
        for path in entries
        if path.name
        not in APPROVED_DAY13_ARTIFACT_NAMES
    )

    if unexpected:
        raise ValueError(
            "Day 13 output contains unapproved files: "
            f"{unexpected}."
        )


def _manifest_bytes(
    manifest: Mapping[str, object],
) -> bytes:
    """Serialize the final manifest canonically."""

    return (
        json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _replace_directory(
    *,
    staged: Path,
    destination: Path,
) -> None:
    """Replace one complete output directory with rollback protection."""

    backup: Path | None = None

    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=".day13-backup-",
                dir=destination.parent,
            )
        )
        backup.rmdir()
        os.replace(
            destination,
            backup,
        )

    try:
        os.replace(
            staged,
            destination,
        )
    except Exception:
        if (
            backup is not None
            and backup.exists()
            and not destination.exists()
        ):
            os.replace(
                backup,
                destination,
            )

        raise
    else:
        if (
            backup is not None
            and backup.exists()
        ):
            shutil.rmtree(backup)


def write_day13_event_walk_forward_artifacts(
    report: Day13EventWalkForwardReport,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact approved Day 13 evidence set safely."""

    if not isinstance(
        report,
        Day13EventWalkForwardReport,
    ):
        raise TypeError(
            "report must be a "
            "Day13EventWalkForwardReport."
        )

    if not isinstance(
        output_directory,
        (str, Path),
    ):
        raise TypeError(
            "output_directory must be a path."
        )

    if not isinstance(overwrite, bool):
        raise TypeError(
            "overwrite must be a boolean."
        )

    directory = Path(
        output_directory
    )
    _validate_output_directory(
        directory,
        overwrite=overwrite,
    )
    directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payloads = _artifact_payloads(
        report
    )
    artifact_hashes = {
        name: _sha256_bytes(payload)
        for name, payload in (
            payloads.items()
        )
    }
    manifest = report.copy_manifest()
    manifest[
        "artifact_sha256"
    ] = artifact_hashes
    expected_hash_names = set(
        APPROVED_DAY13_ARTIFACT_NAMES
    ).difference(
        {
            MANIFEST_FILENAME,
        }
    )

    if set(
        artifact_hashes
    ) != expected_hash_names:
        raise RuntimeError(
            "Day 13 artifact hash set is incomplete."
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix=".day13-stage-",
            dir=directory.parent,
        ) as temporary:
            staged = (
                Path(temporary)
                / DAY13_ARTIFACT_DIRECTORY_NAME
            )
            staged.mkdir()

            for name, payload in (
                payloads.items()
            ):
                path = staged / name

                if name == REPORT_FILENAME:
                    path.write_text(
                        payload.decode(
                            "utf-8"
                        ),
                        encoding="utf-8",
                    )
                else:
                    path.write_bytes(
                        payload
                    )

            (
                staged / MANIFEST_FILENAME
            ).write_text(
                _manifest_bytes(
                    manifest
                ).decode("utf-8"),
                encoding="utf-8",
            )
            staged_names = {
                path.name
                for path in staged.iterdir()
                if path.is_file()
            }

            if staged_names != set(
                APPROVED_DAY13_ARTIFACT_NAMES
            ):
                raise RuntimeError(
                    "Staged Day 13 artifact set is incomplete."
                )

            for name, digest in (
                artifact_hashes.items()
            ):
                if _sha256_bytes(
                    (
                        staged
                        / name
                    ).read_bytes()
                ) != digest:
                    raise RuntimeError(
                        f"Staged artifact hash mismatch: {name}."
                    )

            _replace_directory(
                staged=staged,
                destination=directory,
            )
    except OSError:
        return ()

    paths = tuple(
        directory / name
        for name in (
            APPROVED_DAY13_ARTIFACT_NAMES
        )
    )

    if any(
        not path.is_file()
        or path.stat().st_size <= 0
        for path in paths
    ):
        raise RuntimeError(
            "Written Day 13 artifact set is incomplete."
        )

    return paths
