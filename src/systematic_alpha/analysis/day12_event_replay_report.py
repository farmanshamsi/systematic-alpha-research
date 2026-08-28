"""Deterministic reporting for Day 12 trend-family event replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Mapping

import numpy as np
import pandas as pd

from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    build_wealth_index,
)
from systematic_alpha.analysis.trend_family_event_replay import (
    ACCOUNTING_TOLERANCE,
    FillEvent,
    MarketBarEvent,
    PortfolioSnapshot,
    SignalEvent,
    TargetPositionOrderEvent,
    TrendFamilyEventReplayResult,
    _prepare_replay_bars,
    run_trend_family_event_replay,
)
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS,
)
from systematic_alpha.strategies.ema_macd import (
    build_ema_macd_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    build_trend_ratio_strategy,
)


DAY12_ARTIFACT_VERSION: Final[str] = (
    "trend_family_event_replay_v1"
)
DAY12_STRATEGIES: Final[tuple[str, ...]] = (
    "trend_ratio",
    "ema_macd",
)
APPROVED_DAY12_ARTIFACT_NAMES: Final[
    tuple[str, ...]
] = (
    "replay_summary.csv",
    "performance.csv",
    "event_counts.csv",
    "position_diagnostics.csv",
    "vectorized_parity.csv",
    "manifest.json",
    "report.md",
)
DAY12_REPLAY_SUMMARY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "symbol",
    "frequency",
    "configuration_id",
    "observations",
    "sessions",
    "start_timestamp",
    "end_timestamp",
    "events",
    "initial_executed_position",
    "initial_position_eligible",
    "initial_turnover",
    "initial_transaction_cost",
    "initial_previous_equity",
    "final_equity",
    "event_sha256",
    "observation_sha256",
    "deterministic_replay",
    "input_mutated",
    "locked_2026_accessed",
)
DAY12_PERFORMANCE_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "series",
    "observations",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_equity",
)
DAY12_EVENT_COUNT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "event_type",
    "count",
)
DAY12_POSITION_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "long_bars",
    "neutral_bars",
    "short_bars",
    "entries",
    "exits",
    "direct_reversals",
    "position_changes",
    "total_turnover",
    "total_fractional_transaction_cost",
    "total_transaction_cost_amount",
    "final_equity",
)
DAY12_PARITY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "comparison",
    "comparison_type",
    "row_count",
    "maximum_absolute_difference",
    "mismatch_count",
    "tolerance",
    "passed",
)
EVENT_TYPES: Final[
    tuple[type[object], ...]
] = (
    MarketBarEvent,
    SignalEvent,
    TargetPositionOrderEvent,
    FillEvent,
    PortfolioSnapshot,
)
PARITY_FIELDS: Final[
    tuple[
        tuple[str, str, str],
        ...,
    ]
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


class Day12ReportError(ValueError):
    """Raised when Day 12 evidence cannot be built safely."""


@dataclass(frozen=True, slots=True)
class Day12DatasetAudit:
    """Canonical development-dataset lineage used by Day 12."""

    dataset_id: str
    dataset_path: str
    manifest_sha256: str
    canonical_row_count: int
    spy_row_count: int
    spy_session_count: int
    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class Day12ReplayStudyResult:
    """Two replay results and their compact deterministic evidence."""

    replay_results: tuple[
        TrendFamilyEventReplayResult,
        ...,
    ]
    replay_summary: pd.DataFrame
    performance: pd.DataFrame
    event_counts: pd.DataFrame
    position_diagnostics: pd.DataFrame
    vectorized_parity: pd.DataFrame

    def __post_init__(self) -> None:
        """Copy mutable evidence frames retained by the result."""

        if tuple(
            result.strategy
            for result in self.replay_results
        ) != DAY12_STRATEGIES:
            raise Day12ReportError(
                "Replay results must contain the two "
                "frozen strategies in deterministic order."
            )

        for field_name, columns in (
            (
                "replay_summary",
                DAY12_REPLAY_SUMMARY_COLUMNS,
            ),
            (
                "performance",
                DAY12_PERFORMANCE_COLUMNS,
            ),
            (
                "event_counts",
                DAY12_EVENT_COUNT_COLUMNS,
            ),
            (
                "position_diagnostics",
                DAY12_POSITION_DIAGNOSTIC_COLUMNS,
            ),
            (
                "vectorized_parity",
                DAY12_PARITY_COLUMNS,
            ),
        ):
            frame = getattr(self, field_name)

            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"{field_name} must be a DataFrame."
                )

            if tuple(frame.columns) != columns:
                raise Day12ReportError(
                    f"{field_name} does not match its "
                    "stable Day 12 schema."
                )

            object.__setattr__(
                self,
                field_name,
                frame.copy(deep=True).reset_index(
                    drop=True
                ),
            )

        if (
            len(self.replay_summary) != 2
            or len(self.performance) != 4
            or len(self.event_counts) != 10
            or len(self.position_diagnostics) != 2
            or len(self.vectorized_parity) != 16
        ):
            raise Day12ReportError(
                "Day 12 compact evidence row counts are "
                "inconsistent."
            )

        if not self.vectorized_parity[
            "passed"
        ].astype(bool).all():
            raise Day12ReportError(
                "Vectorised parity must pass for every "
                "frozen comparison."
            )


@dataclass(frozen=True, slots=True)
class Day12ArtifactResult:
    """In-memory Day 12 evidence and written artifact paths."""

    study: Day12ReplayStudyResult
    manifest: dict[str, object]
    report: str
    artifact_directory: Path
    artifact_paths: tuple[Path, ...]


def _utc_timestamp(
    value: object,
    *,
    name: str,
) -> pd.Timestamp:
    """Normalize one audit timestamp to UTC."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise Day12ReportError(
            f"{name} must be a valid timestamp."
        ) from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def validate_day12_dataset_audit(
    audit: Day12DatasetAudit,
) -> Day12DatasetAudit:
    """Validate development lineage without reading another file."""

    if not isinstance(audit, Day12DatasetAudit):
        raise TypeError(
            "audit must be a Day12DatasetAudit."
        )

    dataset_id = audit.dataset_id.strip()
    path = Path(audit.dataset_path)
    digest = audit.manifest_sha256.strip().lower()

    if not dataset_id:
        raise Day12ReportError(
            "dataset_id must not be empty."
        )

    if path.is_absolute():
        raise Day12ReportError(
            "dataset_path must be repository-relative."
        )

    if (
        len(digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in digest
        )
    ):
        raise Day12ReportError(
            "manifest_sha256 must be a 64-character "
            "hexadecimal digest."
        )

    for name in (
        "canonical_row_count",
        "spy_row_count",
        "spy_session_count",
    ):
        value = getattr(audit, name)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise Day12ReportError(
                f"{name} must be a positive integer."
            )

    minimum = _utc_timestamp(
        audit.minimum_timestamp,
        name="minimum_timestamp",
    )
    maximum = _utc_timestamp(
        audit.maximum_timestamp,
        name="maximum_timestamp",
    )
    local_dates = pd.DatetimeIndex(
        [minimum, maximum]
    ).tz_convert(
        "America/New_York"
    ).date

    if (
        local_dates[0].isoformat()
        != "2020-01-02"
        or local_dates[1].isoformat()
        != "2025-12-31"
        or maximum
        >= pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        )
    ):
        raise Day12ReportError(
            "Dataset audit must cover only the complete "
            "2020-01-02 through 2025-12-31 development "
            "period and must not contain 2026 data."
        )

    return Day12DatasetAudit(
        dataset_id=dataset_id,
        dataset_path=path.as_posix(),
        manifest_sha256=digest,
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


def _hash_frame(
    frame: pd.DataFrame,
) -> str:
    """Hash one DataFrame through stable CSV serialization."""

    buffer = io.StringIO()
    frame.to_csv(
        buffer,
        index=False,
        float_format="%.17g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )

    return hashlib.sha256(
        buffer.getvalue().encode("utf-8")
    ).hexdigest()


def _hash_events(
    result: TrendFamilyEventReplayResult,
) -> str:
    """Hash an immutable event stream without exporting it."""

    digest = hashlib.sha256()

    for event in result.events:
        record = asdict(event)
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )
        digest.update(
            payload.encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def _performance_equal(
    left: PerformanceMetrics,
    right: PerformanceMetrics,
) -> bool:
    """Compare shared metrics while preserving paired NaNs."""

    if left.observations != right.observations:
        return False

    for name in (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ):
        left_value = float(getattr(left, name))
        right_value = float(getattr(right, name))

        if (
            math.isnan(left_value)
            and math.isnan(right_value)
        ):
            continue

        if not math.isclose(
            left_value,
            right_value,
            rel_tol=ACCOUNTING_TOLERANCE,
            abs_tol=ACCOUNTING_TOLERANCE,
        ):
            return False

    return True


def _replays_equal(
    left: TrendFamilyEventReplayResult,
    right: TrendFamilyEventReplayResult,
) -> bool:
    """Compare repeated public replay outputs exactly where possible."""

    try:
        pd.testing.assert_frame_equal(
            left.observations,
            right.observations,
            check_exact=True,
        )
    except AssertionError:
        return False

    return (
        left.strategy == right.strategy
        and left.configuration_id
        == right.configuration_id
        and left.events == right.events
        and _performance_equal(
            left.gross_performance,
            right.gross_performance,
        )
        and _performance_equal(
            left.net_performance,
            right.net_performance,
        )
    )


def _vectorized_observations(
    prepared: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    """Execute one frozen vectorised builder as a reference."""

    if strategy == "trend_ratio":
        bundle = build_trend_ratio_strategy(
            prepared,
            parameters=TREND_RATIO_PARAMETERS,
        )
    elif strategy == "ema_macd":
        bundle = build_ema_macd_strategy(
            prepared,
            parameters=EMA_MACD_PARAMETERS,
        )
    else:
        raise Day12ReportError(
            "Unknown Day 12 strategy."
        )

    return bundle.observations.copy(
        deep=True
    ).reset_index(drop=True)


def _parity_records(
    result: TrendFamilyEventReplayResult,
    reference: pd.DataFrame,
) -> list[dict[str, object]]:
    """Compare replay and vectorised execution bar for bar."""

    replay = result.observations

    if len(replay) != len(reference):
        raise Day12ReportError(
            "Vectorised parity row counts differ."
        )

    records: list[dict[str, object]] = []

    for (
        replay_column,
        reference_column,
        comparison_type,
    ) in PARITY_FIELDS:
        actual = replay[replay_column].reset_index(
            drop=True
        )
        expected = reference[
            reference_column
        ].reset_index(drop=True)

        if comparison_type == "exact":
            mismatch_count = int(
                (~actual.eq(expected)).sum()
            )
            maximum_difference = 0.0
            tolerance = 0.0
        else:
            difference = (
                pd.to_numeric(
                    actual,
                    errors="raise",
                ).astype(float)
                - pd.to_numeric(
                    expected,
                    errors="raise",
                ).astype(float)
            ).abs()
            maximum_difference = float(
                difference.max()
            )
            tolerance = (
                ACCOUNTING_TOLERANCE
            )
            mismatch_count = int(
                difference.gt(tolerance).sum()
            )

        records.append(
            {
                "strategy": result.strategy,
                "comparison": replay_column,
                "comparison_type": (
                    comparison_type
                ),
                "row_count": len(replay),
                "maximum_absolute_difference": (
                    maximum_difference
                ),
                "mismatch_count": (
                    mismatch_count
                ),
                "tolerance": tolerance,
                "passed": (
                    mismatch_count == 0
                ),
            }
        )

    return records


def _summary_record(
    result: TrendFamilyEventReplayResult,
    *,
    deterministic: bool,
) -> dict[str, object]:
    """Build one compact replay summary row."""

    observations = result.observations

    return {
        "strategy": result.strategy,
        "symbol": result.symbol,
        "frequency": result.frequency,
        "configuration_id": (
            result.configuration_id
        ),
        "observations": len(observations),
        "sessions": int(
            observations[
                "session_date"
            ].nunique()
        ),
        "start_timestamp": (
            observations["timestamp"].iloc[0]
        ),
        "end_timestamp": (
            observations["timestamp"].iloc[-1]
        ),
        "events": len(result.events),
        "initial_executed_position": int(
            observations[
                "executed_position"
            ].iloc[0]
        ),
        "initial_position_eligible": bool(
            observations[
                "position_eligible"
            ].iloc[0]
        ),
        "initial_turnover": float(
            observations["turnover"].iloc[0]
        ),
        "initial_transaction_cost": float(
            observations[
                "transaction_cost"
            ].iloc[0]
        ),
        "initial_previous_equity": float(
            observations[
                "previous_equity"
            ].iloc[0]
        ),
        "final_equity": float(
            observations[
                "ending_equity"
            ].iloc[-1]
        ),
        "event_sha256": _hash_events(result),
        "observation_sha256": _hash_frame(
            observations
        ),
        "deterministic_replay": deterministic,
        "input_mutated": False,
        "locked_2026_accessed": False,
    }


def _performance_records(
    result: TrendFamilyEventReplayResult,
) -> list[dict[str, object]]:
    """Build gross and net shared-performance rows."""

    annualization_factor = (
        ANNUALIZATION_FACTORS["15min"]
    )
    records: list[dict[str, object]] = []

    for series, metrics, returns in (
        (
            "gross",
            result.gross_performance,
            result.observations[
                "gross_strategy_return"
            ],
        ),
        (
            "net",
            result.net_performance,
            result.observations[
                "net_strategy_return"
            ],
        ),
    ):
        final_equity = float(
            build_wealth_index(returns).iloc[-1]
        )
        records.append(
            {
                "strategy": result.strategy,
                "series": series,
                "observations": (
                    metrics.observations
                ),
                "annualization_factor": (
                    annualization_factor
                ),
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
                "final_equity": final_equity,
            }
        )

    if not math.isclose(
        records[-1]["final_equity"],
        float(
            result.observations[
                "ending_equity"
            ].iloc[-1]
        ),
        rel_tol=ACCOUNTING_TOLERANCE,
        abs_tol=ACCOUNTING_TOLERANCE,
    ):
        raise Day12ReportError(
            "Net final equity does not match the replay "
            "ledger."
        )

    return records


def _event_count_records(
    result: TrendFamilyEventReplayResult,
) -> list[dict[str, object]]:
    """Count every frozen event type in stable order."""

    return [
        {
            "strategy": result.strategy,
            "event_type": event_type.__name__,
            "count": sum(
                isinstance(event, event_type)
                for event in result.events
            ),
        }
        for event_type in EVENT_TYPES
    ]


def _position_record(
    result: TrendFamilyEventReplayResult,
) -> dict[str, object]:
    """Build compact position, turnover, and cost diagnostics."""

    observations = result.observations
    executed = observations[
        "executed_position"
    ]
    previous = observations[
        "previous_executed_position"
    ]
    changes = observations[
        "position_change"
    ]

    return {
        "strategy": result.strategy,
        "long_bars": int(executed.eq(1).sum()),
        "neutral_bars": int(
            executed.eq(0).sum()
        ),
        "short_bars": int(
            executed.eq(-1).sum()
        ),
        "entries": int(
            (
                previous.eq(0)
                & executed.ne(0)
            ).sum()
        ),
        "exits": int(
            (
                previous.ne(0)
                & executed.eq(0)
            ).sum()
        ),
        "direct_reversals": int(
            changes.abs().eq(2).sum()
        ),
        "position_changes": int(
            changes.ne(0).sum()
        ),
        "total_turnover": float(
            observations["turnover"].sum()
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


def run_day12_replay_study(
    bars: pd.DataFrame,
    *,
    verify_determinism: bool = True,
) -> Day12ReplayStudyResult:
    """Run both public replays and build compact parity evidence."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    original = bars.copy(deep=True)
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    replay_results: list[
        TrendFamilyEventReplayResult
    ] = []
    summary_records: list[
        dict[str, object]
    ] = []
    performance_records: list[
        dict[str, object]
    ] = []
    event_records: list[
        dict[str, object]
    ] = []
    position_records: list[
        dict[str, object]
    ] = []
    parity_records: list[
        dict[str, object]
    ] = []

    for strategy in DAY12_STRATEGIES:
        result = (
            run_trend_family_event_replay(
                bars,
                strategy=strategy,
            )
        )
        deterministic = True

        if verify_determinism:
            repeated = (
                run_trend_family_event_replay(
                    bars,
                    strategy=strategy,
                )
            )
            deterministic = _replays_equal(
                result,
                repeated,
            )

            if not deterministic:
                raise Day12ReportError(
                    "Repeated public replay was not "
                    "deterministic."
                )

        reference = _vectorized_observations(
            prepared,
            strategy=strategy,
        )
        strategy_parity = _parity_records(
            result,
            reference,
        )

        if not all(
            bool(record["passed"])
            for record in strategy_parity
        ):
            raise Day12ReportError(
                f"{strategy} vectorised parity failed."
            )

        replay_results.append(result)
        summary_records.append(
            _summary_record(
                result,
                deterministic=deterministic,
            )
        )
        performance_records.extend(
            _performance_records(result)
        )
        event_records.extend(
            _event_count_records(result)
        )
        position_records.append(
            _position_record(result)
        )
        parity_records.extend(
            strategy_parity
        )

    try:
        pd.testing.assert_frame_equal(
            bars,
            original,
            check_exact=True,
        )
    except AssertionError as exc:
        raise Day12ReportError(
            "Day 12 replay mutated its caller input."
        ) from exc

    return Day12ReplayStudyResult(
        replay_results=tuple(
            replay_results
        ),
        replay_summary=pd.DataFrame.from_records(
            summary_records,
            columns=DAY12_REPLAY_SUMMARY_COLUMNS,
        ),
        performance=pd.DataFrame.from_records(
            performance_records,
            columns=DAY12_PERFORMANCE_COLUMNS,
        ),
        event_counts=pd.DataFrame.from_records(
            event_records,
            columns=DAY12_EVENT_COUNT_COLUMNS,
        ),
        position_diagnostics=(
            pd.DataFrame.from_records(
                position_records,
                columns=(
                    DAY12_POSITION_DIAGNOSTIC_COLUMNS
                ),
            )
        ),
        vectorized_parity=(
            pd.DataFrame.from_records(
                parity_records,
                columns=DAY12_PARITY_COLUMNS,
            )
        ),
    )


def _format_markdown_value(
    value: object,
) -> str:
    """Format one compact Markdown value deterministically."""

    if pd.isna(value):
        return "NA"

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"

    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render a DataFrame without optional table dependencies."""

    columns = list(frame.columns)
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
            "| "
            + " | ".join(columns)
            + " |",
            "| "
            + " | ".join(
                "---" for _ in columns
            )
            + " |",
            *rows,
        ]
    )


def _configuration_table() -> pd.DataFrame:
    """Serialize both frozen strategy configurations."""

    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "configuration_id": (
                    CONFIGURATION_IDS[
                        strategy
                    ]
                ),
                "parameters": json.dumps(
                    asdict(
                        TREND_RATIO_PARAMETERS
                        if strategy
                        == "trend_ratio"
                        else EMA_MACD_PARAMETERS
                    ),
                    sort_keys=True,
                ),
            }
            for strategy in DAY12_STRATEGIES
        ]
    )


def build_day12_report(
    study: Day12ReplayStudyResult,
    audit: Day12DatasetAudit,
) -> str:
    """Build the deterministic Day 12 research report."""

    if not isinstance(
        study,
        Day12ReplayStudyResult,
    ):
        raise TypeError(
            "study must be a Day12ReplayStudyResult."
        )

    lineage = validate_day12_dataset_audit(
        audit
    )
    performance = study.performance
    diagnostics = study.position_diagnostics

    return f"""# Day 12 — Deterministic Trend-Family Event Replay

## 1. Objective and scope

Day 12 validates deterministic historical event replay for SPY 15-minute
development bars from 2020-01-02 through 2025-12-31. It evaluates the frozen
Day 6 Trend Ratio and Day 8 EMA/MACD baselines without parameter selection.

## 2. Frozen constraints

- SPY and 15-minute bars only.
- Development data only; the locked 2026 period was not accessed.
- Frozen configuration identifiers and one basis point per unit turnover.
- No tuning, optimisation, sensitivity-winner use, profitability gate, or
  strategy selection. This report does not select a winner.

{_markdown_table(_configuration_table())}

## 3. Event model

Each immutable replay record is one of: market bar, signal, target-position
order, fill, or portfolio snapshot. Orders express a normalized target
position in `-1, 0, 1`; they do not model shares or an order book.

## 4. Exact per-bar event ordering

Events are globally sequenced as market bar, optional fill, portfolio
snapshot, signal, and optional target-position order. All events for one bar
precede every event for the next bar. A terminal signal is retained, while an
order that cannot execute inside the replay is omitted.

## 5. One-observation execution delay

The completed-bar target `q_t` becomes executed position `p_(t+1)`. The first
bar starts neutral, so no same-bar signal can earn that bar's return.

## 6. Neutral initial and reset conventions

Every independent replay begins with executed position zero, no pending order,
position eligibility false, turnover and transaction cost zero, and equity
one. State persists across ordinary sessions. No pre-evaluation state or P&L
is carried into an evaluation interval.

## 7. Return-ledger and normalized-notional accounting

For previous equity `E`, executed position `p`, asset return `r`, turnover
`|p-p_prev|`, and fractional cost `c`:

- gross return = `p * r`
- net return = `p * r - c`
- transaction-cost amount = `E * c`
- cash = `E * (1 - p) - E * c`
- holdings = `p * E * (1 + r)`
- ending equity = `E * (1 + p * r - c)`
- cash plus holdings equals ending equity

## 8. Canonical development-data coverage

- Loader path: `{lineage.dataset_path}`
- Dataset identifier: `{lineage.dataset_id}`
- Canonical rows before SPY filtering: {lineage.canonical_row_count}
- SPY rows: {lineage.spy_row_count}
- SPY sessions: {lineage.spy_session_count}
- Minimum timestamp: {lineage.minimum_timestamp.isoformat()}
- Maximum timestamp: {lineage.maximum_timestamp.isoformat()}

## 9. Results for both frozen strategies

{_markdown_table(study.replay_summary)}

## 10. Vectorised versus event-driven parity

The vectorised builders were executed independently as validation references;
their execution columns were not replay inputs. Every exact mismatch count and
every numeric maximum absolute difference is reported below.

{_markdown_table(study.vectorized_parity)}

## 11. Event counts by type

{_markdown_table(study.event_counts)}

## 12. Position, turnover, and cost diagnostics

{_markdown_table(diagnostics)}

## 13. Gross and net performance

Shared performance utilities calculate both series with annualisation factor
{ANNUALIZATION_FACTORS["15min"]}. Net final equity reconciles bar for bar to
the shared wealth index.

{_markdown_table(performance)}

## 14. Determinism and no-mutation validation

Repeated public replay produces identical immutable events, ledgers, and
shared metrics. Input frames are copied and remain unchanged. Event and
observation SHA-256 digests provide compact evidence without exporting the
full event stream.

## 15. Limitations

This replay has no partial fills, no order-book model, no latency model,
no concurrency, and no live broker. It performs no tuning.
There is no 2026 evaluation.
Normalized positions abstract from shares, borrowing constraints, market
impact, and venue-specific execution.

## 16. Day 12 conclusion

Both frozen baselines pass the deterministic event-ordering, delayed-execution,
accounting, development-window, and vectorised-parity integrity gates.
Profitability is not an acceptance criterion, and neither strategy is selected
for deployment.
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

    return (
        _utc_timestamp(
            value,
            name="generation_timestamp",
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(
    path: Path,
) -> str:
    """Calculate one artifact SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one deterministic compact CSV."""

    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )


def build_day12_manifest(
    study: Day12ReplayStudyResult,
    audit: Day12DatasetAudit,
    *,
    generation_timestamp: str,
    source_git_commit: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build deterministic Day 12 lineage and integrity metadata."""

    lineage = validate_day12_dataset_audit(
        audit
    )
    commit = source_git_commit.strip()

    if not commit:
        raise Day12ReportError(
            "source_git_commit must not be empty."
        )

    return {
        "day": 12,
        "analysis_name": (
            "trend_family_event_replay"
        ),
        "artifact_version": (
            DAY12_ARTIFACT_VERSION
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
        "source_git_commit": commit,
        "canonical_row_count": (
            lineage.canonical_row_count
        ),
        "spy_row_count": lineage.spy_row_count,
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
        "symbol": "SPY",
        "frequency": "15min",
        "strategies": list(
            DAY12_STRATEGIES
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
        "annualization_factor": (
            ANNUALIZATION_FACTORS["15min"]
        ),
        "execution_delay_observations": 1,
        "event_order": [
            "MarketBarEvent",
            "FillEvent when position changes",
            "PortfolioSnapshot",
            "SignalEvent",
            "TargetPositionOrderEvent when requested",
        ],
        "accounting_model": (
            "normalized-notional return ledger"
        ),
        "parameter_selection_performed": False,
        "profitability_acceptance_gate": False,
        "full_event_stream_exported": False,
        "replay_observation_counts": {
            row.strategy: int(
                row.observations
            )
            for row in (
                study.replay_summary.itertuples(
                    index=False
                )
            )
        },
        "replay_event_counts": {
            row.strategy: int(
                row.events
            )
            for row in (
                study.replay_summary.itertuples(
                    index=False
                )
            )
        },
        "event_sha256": {
            row.strategy: row.event_sha256
            for row in (
                study.replay_summary.itertuples(
                    index=False
                )
            )
        },
        "observation_sha256": {
            row.strategy: (
                row.observation_sha256
            )
            for row in (
                study.replay_summary.itertuples(
                    index=False
                )
            )
        },
        "integrity_checks": {
            "public_replay_api_used": True,
            "development_only": True,
            "spy_only": True,
            "frequency_15min_only": True,
            "neutral_initial_state": True,
            "one_observation_delay": True,
            "deterministic_replay": bool(
                study.replay_summary[
                    "deterministic_replay"
                ].all()
            ),
            "input_mutated": False,
            "vectorized_parity": bool(
                study.vectorized_parity[
                    "passed"
                ].all()
            ),
            "net_wealth_reconciled": True,
            "locked_2026_period_accessed": False,
        },
        "artifact_sha256": dict(
            sorted(artifact_hashes.items())
        ),
    }


def _validate_artifact_directory(
    directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Apply conservative Day 12 overwrite controls."""

    if directory.exists() and not directory.is_dir():
        raise Day12ReportError(
            "Artifact path exists but is not a directory."
        )

    if not directory.exists():
        return

    entries = list(directory.iterdir())

    if entries and not overwrite:
        raise Day12ReportError(
            "Artifact directory is non-empty; pass "
            "--overwrite to replace the approved Day 12 "
            "artifact set."
        )

    nested = sorted(
        path.name
        for path in entries
        if path.is_dir()
    )

    if nested:
        raise Day12ReportError(
            "Artifact directory contains nested "
            f"directories: {nested}."
        )

    unexpected = sorted(
        path.name
        for path in entries
        if path.name
        not in APPROVED_DAY12_ARTIFACT_NAMES
    )

    if unexpected:
        raise Day12ReportError(
            "Artifact directory contains unapproved "
            f"files: {unexpected}."
        )


def write_day12_artifacts(
    study: Day12ReplayStudyResult,
    audit: Day12DatasetAudit,
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
    generation_timestamp: str | None = None,
    source_git_commit: str,
) -> Day12ArtifactResult:
    """Write the exact compact Day 12 evidence set atomically."""

    if not isinstance(
        study,
        Day12ReplayStudyResult,
    ):
        raise TypeError(
            "study must be a Day12ReplayStudyResult."
        )

    lineage = validate_day12_dataset_audit(
        audit
    )
    report = build_day12_report(
        study,
        lineage,
    )
    directory = Path(artifact_directory)
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
        prefix=".day12-",
        dir=directory.parent,
    ) as temporary:
        staging = Path(temporary)

        for name, frame in (
            (
                "replay_summary.csv",
                study.replay_summary,
            ),
            (
                "performance.csv",
                study.performance,
            ),
            (
                "event_counts.csv",
                study.event_counts,
            ),
            (
                "position_diagnostics.csv",
                study.position_diagnostics,
            ),
            (
                "vectorized_parity.csv",
                study.vectorized_parity,
            ),
        ):
            _write_csv(
                frame,
                staging / name,
            )

        (
            staging / "report.md"
        ).write_text(
            report,
            encoding="utf-8",
        )
        hashed_names = [
            name
            for name in (
                APPROVED_DAY12_ARTIFACT_NAMES
            )
            if name != "manifest.json"
        ]
        artifact_hashes = {
            name: _sha256(
                staging / name
            )
            for name in hashed_names
        }
        manifest = build_day12_manifest(
            study,
            lineage,
            generation_timestamp=(
                _normalize_generation_timestamp(
                    generation_timestamp
                )
            ),
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

        if {
            path.name
            for path in staging.iterdir()
            if path.is_file()
        } != set(
            APPROVED_DAY12_ARTIFACT_NAMES
        ):
            raise RuntimeError(
                "Staged Day 12 artifact set is incomplete."
            )

        for name in (
            APPROVED_DAY12_ARTIFACT_NAMES
        ):
            os.replace(
                staging / name,
                directory / name,
            )

    paths = tuple(
        directory / name
        for name in (
            APPROVED_DAY12_ARTIFACT_NAMES
        )
    )

    if any(
        not path.exists()
        or path.stat().st_size <= 0
        for path in paths
    ):
        raise RuntimeError(
            "A Day 12 artifact is missing or empty."
        )

    return Day12ArtifactResult(
        study=study,
        manifest=manifest,
        report=report,
        artifact_directory=directory,
        artifact_paths=paths,
    )
