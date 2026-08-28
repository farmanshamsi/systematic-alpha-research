"""Run frozen Day 11 folds through the public Day 12 replay API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    build_wealth_index,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_event_replay import (
    FillEvent,
    MarketBarEvent,
    PortfolioSnapshot,
    ReplayEvent,
    SignalEvent,
    TargetPositionOrderEvent,
    TrendFamilyEventReplayResult,
    run_trend_family_event_replay,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    WalkForwardFold,
    _build_strategy_observations,
    _partition_fold,
    _prepare_development_features,
    _reset_test_execution,
    build_walk_forward_folds,
)


STRATEGY_ORDER: Final[tuple[str, ...]] = (
    "trend_ratio",
    "ema_macd",
)
PARITY_TOLERANCE: Final[float] = 1e-12
FREQUENCY: Final[str] = "15min"
EXPECTED_FOLD_IDS: Final[tuple[str, ...]] = (
    "wf_2022",
    "wf_2023",
    "wf_2024",
    "wf_2025",
)
PARITY_MAPPINGS: Final[
    tuple[tuple[str, str, str], ...]
] = (
    (
        "target_position",
        "signal",
        "exact",
    ),
    (
        "signal_available",
        "signal_available",
        "exact",
    ),
    (
        "executed_position",
        "position",
        "exact",
    ),
    (
        "position_eligible",
        "position_eligible",
        "exact",
    ),
    (
        "turnover",
        "turnover",
        "numeric",
    ),
    (
        "transaction_cost",
        "transaction_cost",
        "numeric",
    ),
    (
        "gross_strategy_return",
        "gross_strategy_return",
        "numeric",
    ),
    (
        "net_strategy_return",
        "net_strategy_return",
        "numeric",
    ),
)
FOLD_SUMMARY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "fold_id",
    "symbol",
    "frequency",
    "configuration_id",
    "train_start_timestamp",
    "train_end_exclusive",
    "test_start_timestamp",
    "test_end_exclusive",
    "evaluation_start",
    "evaluation_end_exclusive",
    "train_observations",
    "test_observations",
    "train_sessions",
    "test_sessions",
    "indicator_history_observations",
    "initial_position",
    "initial_equity",
    "parity_comparisons",
    "parity_passed",
)
EVENT_COUNT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "fold_id",
    "market_bar_events",
    "signal_events",
    "order_events",
    "fill_events",
    "portfolio_snapshots",
    "total_events",
    "observations",
)
POSITION_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = (
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
    "total_turnover",
    "total_fractional_transaction_cost",
    "total_transaction_cost_amount",
    "final_equity",
)
PERFORMANCE_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "fold_id",
    "series",
    "observations",
    "sessions",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_wealth",
)
PARITY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "fold_id",
    "comparison",
    "comparison_type",
    "row_count",
    "maximum_absolute_difference",
    "mismatch_count",
    "tolerance",
    "passed",
)
AGGREGATE_SUMMARY_COLUMNS: Final[
    tuple[str, ...]
] = (
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
)


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Return a defensive, zero-based copy of one result table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    return frame.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class EventWalkForwardFoldRun:
    """One frozen-strategy fold replay and its vectorised reference."""

    strategy: str
    fold_id: str
    replay_result: TrendFamilyEventReplayResult
    vectorized_observations: pd.DataFrame

    def __post_init__(self) -> None:
        """Defensively retain the vectorised reference observations."""

        object.__setattr__(
            self,
            "vectorized_observations",
            _copy_frame(
                self.vectorized_observations,
                name="vectorized_observations",
            ),
        )

    def copy_vectorized_observations(
        self,
    ) -> pd.DataFrame:
        """Return a mutable copy of the vectorised observations."""

        return self.vectorized_observations.copy(deep=True)


@dataclass(frozen=True, slots=True)
class TrendFamilyEventWalkForwardResults:
    """Immutable fold runs and copied deterministic Day 13 tables."""

    fold_runs: tuple[EventWalkForwardFoldRun, ...]
    fold_summary: pd.DataFrame
    event_counts: pd.DataFrame
    position_diagnostics: pd.DataFrame
    performance: pd.DataFrame
    vectorized_parity: pd.DataFrame
    aggregate_summary: pd.DataFrame

    def __post_init__(self) -> None:
        """Defensively retain every mutable result table."""

        object.__setattr__(
            self,
            "fold_runs",
            tuple(self.fold_runs),
        )

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

    def copy_fold_summary(self) -> pd.DataFrame:
        """Return a mutable copy of the fold summary."""

        return self.fold_summary.copy(deep=True)

    def copy_event_counts(self) -> pd.DataFrame:
        """Return a mutable copy of the event-count table."""

        return self.event_counts.copy(deep=True)

    def copy_position_diagnostics(
        self,
    ) -> pd.DataFrame:
        """Return a mutable copy of the position diagnostics."""

        return self.position_diagnostics.copy(deep=True)

    def copy_performance(self) -> pd.DataFrame:
        """Return a mutable copy of the performance table."""

        return self.performance.copy(deep=True)

    def copy_vectorized_parity(
        self,
    ) -> pd.DataFrame:
        """Return a mutable copy of the parity table."""

        return self.vectorized_parity.copy(deep=True)

    def copy_aggregate_summary(
        self,
    ) -> pd.DataFrame:
        """Return a mutable copy of the aggregate summary."""

        return self.aggregate_summary.copy(deep=True)


def _normalize_utc_timestamp(
    value: object,
    *,
    name: str,
) -> pd.Timestamp:
    """Require one timezone-aware timestamp normalized to UTC."""

    if (
        not isinstance(value, pd.Timestamp)
        or value.tzinfo is None
    ):
        raise ValueError(
            f"{name} must be a timezone-aware pandas Timestamp."
        )

    return value.tz_convert("UTC")


def _validate_folds(
    folds: tuple[WalkForwardFold, ...],
) -> None:
    """Require the frozen four-fold order without recreating folds."""

    actual = tuple(
        fold.fold_id
        for fold in folds
    )

    if actual != EXPECTED_FOLD_IDS:
        raise ValueError(
            "Day 13 requires the frozen Day 11 folds in "
            f"this order: {EXPECTED_FOLD_IDS}."
        )


def _build_fold_input(
    features: pd.DataFrame,
    *,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build expanding training history plus one complete test year."""

    train, test = _partition_fold(
        features,
        fold,
    )
    fold_input = (
        pd.concat(
            [
                train,
                test,
            ],
            ignore_index=True,
        )
        .copy(deep=True)
        .reset_index(drop=True)
    )

    if (
        fold_input.empty
        or not fold_input[
            "timestamp"
        ].is_monotonic_increasing
        or fold_input[
            "timestamp"
        ].duplicated().any()
    ):
        raise ValueError(
            f"{fold.fold_id} input must be nonempty, unique, "
            "and chronological."
        )

    timestamps = pd.to_datetime(
        fold_input["timestamp"],
        utc=True,
        errors="raise",
    )

    if (
        timestamps.lt(
            fold.train_start
        ).any()
        or timestamps.ge(
            fold.test_end_exclusive
        ).any()
    ):
        raise ValueError(
            f"{fold.fold_id} input contains observations "
            "outside its expanding train-plus-test interval."
        )

    return train, test, fold_input


def _map_evaluation_boundaries(
    test: pd.DataFrame,
    *,
    fold: WalkForwardFold,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map calendar fold edges to actual 15-minute bar boundaries."""

    if test.empty:
        raise ValueError(
            f"{fold.fold_id} test frame cannot be empty."
        )

    raw_timestamps = test["timestamp"]

    if not raw_timestamps.map(
        lambda value: (
            isinstance(value, pd.Timestamp)
            and value.tzinfo is not None
        )
    ).all():
        raise ValueError(
            f"{fold.fold_id} test timestamps must be "
            "timezone-aware pandas Timestamps."
        )

    timestamps = pd.to_datetime(
        raw_timestamps,
        utc=True,
        errors="raise",
    )

    if (
        not timestamps.is_monotonic_increasing
        or timestamps.duplicated().any()
    ):
        raise ValueError(
            f"{fold.fold_id} test timestamps must be unique "
            "and chronological."
        )

    start = _normalize_utc_timestamp(
        pd.Timestamp(
            timestamps.iloc[0]
        ),
        name="evaluation_start",
    )
    end = _normalize_utc_timestamp(
        pd.Timestamp(
            timestamps.iloc[-1]
        )
        + pd.Timedelta(minutes=15),
        name="evaluation_end_exclusive",
    )
    first_session = test[
        "session_date"
    ].iloc[0]
    first_session_start = pd.Timestamp(
        timestamps.loc[
            test[
                "session_date"
            ].eq(first_session)
        ].iloc[0]
    )

    if start != first_session_start:
        raise ValueError(
            f"{fold.fold_id} evaluation must start at the "
            "first bar of a complete session."
        )

    if (
        start >= end
        or start < fold.test_start
        or end > fold.test_end_exclusive
    ):
        raise ValueError(
            f"{fold.fold_id} bar boundaries do not fit its "
            "calendar test interval."
        )

    return start, end


def _event_timestamp(
    event: ReplayEvent,
) -> pd.Timestamp:
    """Return the timestamp shared by one validated replay event."""

    if isinstance(
        event,
        TargetPositionOrderEvent,
    ):
        return event.submitted_timestamp

    return event.timestamp


def _validate_replay_evidence(
    replay: TrendFamilyEventReplayResult,
) -> None:
    """Require test-only evidence and a neutral fold-boundary reset."""

    observations = replay.observations

    if observations.empty:
        raise ValueError(
            "A Day 13 fold replay cannot be empty."
        )

    timestamps = pd.to_datetime(
        observations["timestamp"],
        utc=True,
        errors="raise",
    )
    inside = (
        timestamps.ge(
            replay.evaluation_start
        )
        & timestamps.lt(
            replay.evaluation_end_exclusive
        )
    )

    if not inside.all():
        raise ValueError(
            "Replay observations must contain test-period "
            "rows only."
        )

    if not all(
        replay.evaluation_start
        <= _event_timestamp(event)
        < replay.evaluation_end_exclusive
        for event in replay.events
    ):
        raise ValueError(
            "Replay events must contain test-period records "
            "only."
        )

    first = observations.iloc[0]
    exact_reset = (
        int(
            first[
                "previous_executed_position"
            ]
        )
        == 0
        and int(
            first[
                "executed_position"
            ]
        )
        == 0
        and not bool(
            first[
                "position_eligible"
            ]
        )
        and int(
            first[
                "position_change"
            ]
        )
        == 0
        and not bool(
            first[
                "fill_executed"
            ]
        )
    )
    reset_values = (
        (
            "turnover",
            0.0,
        ),
        (
            "transaction_cost",
            0.0,
        ),
        (
            "previous_equity",
            1.0,
        ),
        (
            "cash_balance",
            1.0,
        ),
        (
            "holdings_value",
            0.0,
        ),
        (
            "ending_equity",
            1.0,
        ),
    )

    if (
        not exact_reset
        or not all(
            math.isclose(
                float(first[name]),
                expected,
                rel_tol=PARITY_TOLERANCE,
                abs_tol=PARITY_TOLERANCE,
            )
            for name, expected in (
                reset_values
            )
        )
    ):
        raise ValueError(
            "Replay fold did not reset position, pending "
            "execution, and equity to the neutral state."
        )

    if len(observations) >= 2:
        executed = observations[
            "executed_position"
        ].iloc[1:].reset_index(drop=True)
        delayed_targets = observations[
            "target_position"
        ].iloc[:-1].reset_index(drop=True)

        if not executed.equals(
            delayed_targets
        ):
            raise ValueError(
                "Replay fold violates one-observation "
                "delayed execution."
            )


def _build_vectorized_reference(
    fold_input: pd.DataFrame,
    *,
    strategy: str,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    """Build the independent Day 11 test-only execution reference."""

    (
        observations,
        cost_bps_per_turnover,
    ) = _build_strategy_observations(
        fold_input.copy(deep=True),
        strategy=strategy,
    )
    reset = _reset_test_execution(
        observations,
        fold=fold,
        cost_bps_per_turnover=(
            cost_bps_per_turnover
        ),
    )

    return reset.copy(
        deep=True
    ).reset_index(drop=True)


def _build_parity_records(
    *,
    strategy: str,
    fold_id: str,
    replay: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[dict[str, object]]:
    """Compare Day 12 replay fields with independent Day 11 fields."""

    if len(replay) != len(reference):
        raise ValueError(
            f"{strategy}/{fold_id} replay and vectorised "
            "references must have equal row counts."
        )

    records: list[
        dict[str, object]
    ] = []

    for (
        replay_column,
        reference_column,
        comparison_type,
    ) in PARITY_MAPPINGS:
        if (
            replay_column
            not in replay.columns
            or reference_column
            not in reference.columns
        ):
            raise ValueError(
                f"{strategy}/{fold_id} parity requires "
                f"{replay_column} and {reference_column}."
            )

        actual = replay[
            replay_column
        ].reset_index(drop=True)
        expected = reference[
            reference_column
        ].reset_index(drop=True)

        if comparison_type == "exact":
            unequal = actual.ne(expected)
            mismatch_count = int(
                unequal.sum()
            )
            passed = bool(
                mismatch_count == 0
                and actual.equals(expected)
            )
            maximum_difference = (
                0.0
                if mismatch_count == 0
                else 1.0
            )
            tolerance = 0.0
        else:
            actual_numeric = pd.to_numeric(
                actual,
                errors="raise",
            ).to_numpy(dtype="float64")
            expected_numeric = pd.to_numeric(
                expected,
                errors="raise",
            ).to_numpy(dtype="float64")
            differences = np.abs(
                actual_numeric
                - expected_numeric
            )
            close = np.isclose(
                actual_numeric,
                expected_numeric,
                rtol=PARITY_TOLERANCE,
                atol=PARITY_TOLERANCE,
                equal_nan=False,
            )
            mismatch_count = int(
                (~close).sum()
            )
            maximum_difference = float(
                differences.max(
                    initial=0.0
                )
            )
            tolerance = (
                PARITY_TOLERANCE
            )
            passed = bool(
                mismatch_count == 0
            )

        records.append(
            {
                "strategy": strategy,
                "fold_id": fold_id,
                "comparison": (
                    replay_column
                ),
                "comparison_type": (
                    comparison_type
                ),
                "row_count": int(
                    len(actual)
                ),
                "maximum_absolute_difference": (
                    maximum_difference
                ),
                "mismatch_count": (
                    mismatch_count
                ),
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    return records


def _metrics_record(
    metrics: PerformanceMetrics,
) -> dict[str, object]:
    """Map shared metrics to deterministic Day 13 column names."""

    return {
        "cumulative_return": (
            metrics.cumulative_return
        ),
        "annualized_return": (
            metrics.annualized_return
        ),
        "annualized_volatility": (
            metrics.annualized_volatility
        ),
        "sharpe_ratio": (
            metrics.sharpe_ratio
        ),
        "maximum_drawdown": (
            metrics.max_drawdown
        ),
    }


def _performance_record(
    *,
    strategy: str,
    fold_id: str | None,
    series: str,
    returns: pd.Series,
    sessions: int,
    folds: int | None = None,
) -> dict[str, object]:
    """Recompute shared metrics with observed session density."""

    observations = int(
        len(returns)
    )

    if observations <= 0 or sessions <= 0:
        raise ValueError(
            "Performance requires positive observations "
            "and sessions."
        )

    annualization_factor = float(
        252.0
        * observations
        / sessions
    )
    metrics = (
        calculate_performance_metrics(
            returns,
            annualization_factor=(
                annualization_factor
            ),
        )
    )
    wealth = build_wealth_index(
        returns
    )
    record: dict[str, object] = {
        "strategy": strategy,
        "series": series,
        "observations": observations,
        "sessions": sessions,
        "annualization_factor": (
            annualization_factor
        ),
        **_metrics_record(metrics),
        "final_wealth": float(
            wealth.iloc[-1]
        ),
    }

    if fold_id is not None:
        record["fold_id"] = fold_id

    if folds is not None:
        record["folds"] = folds

    return record


def _build_event_count_record(
    *,
    strategy: str,
    fold_id: str,
    replay: TrendFamilyEventReplayResult,
) -> dict[str, object]:
    """Count immutable replay event records by validated type."""

    events = replay.events

    return {
        "strategy": strategy,
        "fold_id": fold_id,
        "market_bar_events": sum(
            isinstance(
                event,
                MarketBarEvent,
            )
            for event in events
        ),
        "signal_events": sum(
            isinstance(
                event,
                SignalEvent,
            )
            for event in events
        ),
        "order_events": sum(
            isinstance(
                event,
                TargetPositionOrderEvent,
            )
            for event in events
        ),
        "fill_events": sum(
            isinstance(
                event,
                FillEvent,
            )
            for event in events
        ),
        "portfolio_snapshots": sum(
            isinstance(
                event,
                PortfolioSnapshot,
            )
            for event in events
        ),
        "total_events": int(
            len(events)
        ),
        "observations": int(
            len(replay.observations)
        ),
    }


def _build_position_diagnostic_record(
    *,
    strategy: str,
    fold_id: str,
    observations: pd.DataFrame,
) -> dict[str, object]:
    """Summarize fold-boundary reset and replay ledger accounting."""

    first = observations.iloc[0]

    return {
        "strategy": strategy,
        "fold_id": fold_id,
        "initial_previous_position": int(
            first[
                "previous_executed_position"
            ]
        ),
        "initial_position": int(
            first[
                "executed_position"
            ]
        ),
        "initial_position_eligible": bool(
            first[
                "position_eligible"
            ]
        ),
        "initial_turnover": float(
            first[
                "turnover"
            ]
        ),
        "initial_transaction_cost": float(
            first[
                "transaction_cost"
            ]
        ),
        "initial_previous_equity": float(
            first[
                "previous_equity"
            ]
        ),
        "initial_cash_balance": float(
            first[
                "cash_balance"
            ]
        ),
        "initial_holdings_value": float(
            first[
                "holdings_value"
            ]
        ),
        "initial_ending_equity": float(
            first[
                "ending_equity"
            ]
        ),
        "initial_fill_executed": bool(
            first[
                "fill_executed"
            ]
        ),
        "total_turnover": float(
            observations[
                "turnover"
            ].sum()
        ),
        "total_fractional_transaction_cost": float(
            observations[
                "transaction_cost"
            ].sum()
        ),
        "total_transaction_cost_amount": float(
            observations[
                "transaction_cost_amount"
            ].sum()
        ),
        "final_equity": float(
            observations[
                "ending_equity"
            ].iloc[-1]
        ),
    }


def _build_fold_summary_record(
    *,
    strategy: str,
    fold: WalkForwardFold,
    train: pd.DataFrame,
    test: pd.DataFrame,
    replay: TrendFamilyEventReplayResult,
    parity_records: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Summarize one frozen fold without performance selection."""

    first = replay.observations.iloc[
        0
    ]

    return {
        "strategy": strategy,
        "fold_id": fold.fold_id,
        "symbol": replay.symbol,
        "frequency": replay.frequency,
        "configuration_id": (
            replay.configuration_id
        ),
        "train_start_timestamp": (
            fold.train_start
        ),
        "train_end_exclusive": (
            fold.train_end_exclusive
        ),
        "test_start_timestamp": (
            fold.test_start
        ),
        "test_end_exclusive": (
            fold.test_end_exclusive
        ),
        "evaluation_start": (
            replay.evaluation_start
        ),
        "evaluation_end_exclusive": (
            replay.evaluation_end_exclusive
        ),
        "train_observations": int(
            len(train)
        ),
        "test_observations": int(
            len(test)
        ),
        "train_sessions": int(
            train[
                "session_date"
            ].nunique()
        ),
        "test_sessions": int(
            test[
                "session_date"
            ].nunique()
        ),
        "indicator_history_observations": int(
            len(train)
        ),
        "initial_position": int(
            first[
                "executed_position"
            ]
        ),
        "initial_equity": float(
            first[
                "previous_equity"
            ]
        ),
        "parity_comparisons": int(
            len(parity_records)
        ),
        "parity_passed": bool(
            all(
                bool(record["passed"])
                for record in (
                    parity_records
                )
            )
        ),
    }


def run_trend_family_event_walk_forward(
    bars: pd.DataFrame,
) -> TrendFamilyEventWalkForwardResults:
    """Run the frozen Day 13 event-driven walk-forward study."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")

    features = _prepare_development_features(
        bars.copy(deep=True)
    )
    folds = tuple(
        build_walk_forward_folds()
    )
    _validate_folds(folds)
    fold_runs: list[
        EventWalkForwardFoldRun
    ] = []
    fold_summary_records: list[
        dict[str, object]
    ] = []
    event_count_records: list[
        dict[str, object]
    ] = []
    position_records: list[
        dict[str, object]
    ] = []
    performance_records: list[
        dict[str, object]
    ] = []
    parity_records: list[
        dict[str, object]
    ] = []
    aggregate_inputs: dict[
        str,
        list[pd.DataFrame],
    ] = {
        strategy: []
        for strategy in (
            STRATEGY_ORDER
        )
    }

    for strategy in STRATEGY_ORDER:
        for fold in folds:
            train, test, fold_input = (
                _build_fold_input(
                    features,
                    fold=fold,
                )
            )
            (
                evaluation_start,
                evaluation_end_exclusive,
            ) = _map_evaluation_boundaries(
                test,
                fold=fold,
            )
            replay = (
                run_trend_family_event_replay(
                    fold_input.copy(
                        deep=True
                    ),
                    strategy=strategy,
                    frequency=FREQUENCY,
                    evaluation_start=(
                        evaluation_start
                    ),
                    evaluation_end_exclusive=(
                        evaluation_end_exclusive
                    ),
                )
            )
            _validate_replay_evidence(
                replay
            )
            reference = (
                _build_vectorized_reference(
                    fold_input,
                    strategy=strategy,
                    fold=fold,
                )
            )
            current_parity = (
                _build_parity_records(
                    strategy=strategy,
                    fold_id=fold.fold_id,
                    replay=(
                        replay.observations
                    ),
                    reference=reference,
                )
            )
            parity_records.extend(
                current_parity
            )
            fold_runs.append(
                EventWalkForwardFoldRun(
                    strategy=strategy,
                    fold_id=fold.fold_id,
                    replay_result=replay,
                    vectorized_observations=(
                        reference
                    ),
                )
            )
            fold_summary_records.append(
                _build_fold_summary_record(
                    strategy=strategy,
                    fold=fold,
                    train=train,
                    test=test,
                    replay=replay,
                    parity_records=(
                        current_parity
                    ),
                )
            )
            event_count_records.append(
                _build_event_count_record(
                    strategy=strategy,
                    fold_id=fold.fold_id,
                    replay=replay,
                )
            )
            position_records.append(
                _build_position_diagnostic_record(
                    strategy=strategy,
                    fold_id=fold.fold_id,
                    observations=(
                        replay.observations
                    ),
                )
            )
            sessions = int(
                replay.observations[
                    "session_date"
                ].nunique()
            )

            for series, column in (
                (
                    "gross",
                    "gross_strategy_return",
                ),
                (
                    "net",
                    "net_strategy_return",
                ),
            ):
                performance_records.append(
                    _performance_record(
                        strategy=strategy,
                        fold_id=(
                            fold.fold_id
                        ),
                        series=series,
                        returns=(
                            replay.observations[
                                column
                            ]
                        ),
                        sessions=sessions,
                    )
                )

            aggregate_inputs[
                strategy
            ].append(
                replay.observations.copy(
                    deep=True
                )
            )

    aggregate_records: list[
        dict[str, object]
    ] = []

    for strategy in STRATEGY_ORDER:
        combined = (
            pd.concat(
                aggregate_inputs[
                    strategy
                ],
                ignore_index=True,
            )
            .sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )
        sessions = int(
            combined[
                "session_date"
            ].nunique()
        )

        for series, column in (
            (
                "gross",
                "gross_strategy_return",
            ),
            (
                "net",
                "net_strategy_return",
            ),
        ):
            aggregate_records.append(
                _performance_record(
                    strategy=strategy,
                    fold_id=None,
                    series=series,
                    returns=combined[
                        column
                    ],
                    sessions=sessions,
                    folds=len(folds),
                )
            )

    return TrendFamilyEventWalkForwardResults(
        fold_runs=tuple(
            fold_runs
        ),
        fold_summary=(
            pd.DataFrame.from_records(
                fold_summary_records,
                columns=(
                    FOLD_SUMMARY_COLUMNS
                ),
            )
        ),
        event_counts=(
            pd.DataFrame.from_records(
                event_count_records,
                columns=(
                    EVENT_COUNT_COLUMNS
                ),
            )
        ),
        position_diagnostics=(
            pd.DataFrame.from_records(
                position_records,
                columns=(
                    POSITION_DIAGNOSTIC_COLUMNS
                ),
            )
        ),
        performance=(
            pd.DataFrame.from_records(
                performance_records,
                columns=(
                    PERFORMANCE_COLUMNS
                ),
            )
        ),
        vectorized_parity=(
            pd.DataFrame.from_records(
                parity_records,
                columns=(
                    PARITY_COLUMNS
                ),
            )
        ),
        aggregate_summary=(
            pd.DataFrame.from_records(
                aggregate_records,
                columns=(
                    AGGREGATE_SUMMARY_COLUMNS
                ),
            )
        ),
    )
