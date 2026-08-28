"""Development-only walk-forward evaluation for frozen trend baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import pandas as pd

from systematic_alpha.analysis.eda_features import (
    build_return_features,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS as DAY10_CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS as DAY10_EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS as DAY10_TREND_RATIO_PARAMETERS,
)
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)
from systematic_alpha.strategies.ema_macd import (
    build_ema_macd_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    calculate_turnover,
    build_trend_ratio_strategy,
)


DEVELOPMENT_START: Final[pd.Timestamp] = (
    pd.Timestamp("2020-01-02", tz="UTC")
)
DEVELOPMENT_END_EXCLUSIVE: Final[
    pd.Timestamp
] = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)
LOCKED_PERIOD_START: Final[pd.Timestamp] = (
    pd.Timestamp("2026-01-02", tz="UTC")
)

WALK_FORWARD_STRATEGIES: Final[
    tuple[str, ...]
] = (
    "trend_ratio",
    "ema_macd",
)
WALK_FORWARD_SYMBOL: Final[str] = "SPY"
WALK_FORWARD_FREQUENCY: Final[str] = "15min"
REQUIRED_INPUT_SYMBOLS: Final[
    tuple[str, ...]
] = (
    "SPY",
    "QQQ",
    "IWM",
)
EXPECTED_DEVELOPMENT_YEARS: Final[
    frozenset[int]
] = frozenset(
    range(2020, 2026)
)
TRADING_SESSIONS_PER_YEAR: Final[float] = 252.0

TREND_RATIO_PARAMETERS = (
    DAY10_TREND_RATIO_PARAMETERS
)
EMA_MACD_PARAMETERS = (
    DAY10_EMA_MACD_PARAMETERS
)
CONFIGURATION_IDS: Final[
    dict[str, str]
] = dict(
    DAY10_CONFIGURATION_IDS
)

FOLD_RESULT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "symbol",
    "frequency",
    "fold_id",
    "configuration_id",
    "train_start_timestamp",
    "train_end_timestamp",
    "test_start_timestamp",
    "test_end_timestamp",
    "train_sessions",
    "test_sessions",
    "train_observations",
    "test_observations",
    "annualization_factor",
    "purge_sessions",
    "embargo_sessions",
    "indicator_history_observations",
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
    "average_exposure",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "trade_count",
)

AGGREGATE_RESULT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "symbol",
    "frequency",
    "configuration_id",
    "folds",
    "test_start_timestamp",
    "test_end_timestamp",
    "test_sessions",
    "test_observations",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "average_exposure",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "trade_count",
)


class TrendFamilyWalkForwardError(ValueError):
    """Raised when Day 11 walk-forward evaluation is unsafe."""


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One immutable expanding-window calendar fold."""

    fold_id: str
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp
    purge_sessions: int = 0
    embargo_sessions: int = 0

    def __post_init__(self) -> None:
        """Validate chronological and leakage-control boundaries."""

        timestamps = (
            self.train_start,
            self.train_end_exclusive,
            self.test_start,
            self.test_end_exclusive,
        )

        if not all(
            isinstance(value, pd.Timestamp)
            and value.tzinfo is not None
            for value in timestamps
        ):
            raise TypeError(
                "Walk-forward boundaries must be "
                "timezone-aware pandas Timestamps."
            )

        if (
            self.train_start
            != DEVELOPMENT_START
            or self.train_end_exclusive
            != self.test_start
            or not (
                self.train_start
                < self.train_end_exclusive
                < self.test_end_exclusive
            )
        ):
            raise TrendFamilyWalkForwardError(
                "Walk-forward folds must use one expanding "
                "training origin and adjacent test years."
            )

        if (
            self.test_start.month,
            self.test_start.day,
        ) != (
            1,
            1,
        ) or (
            self.test_end_exclusive.month,
            self.test_end_exclusive.day,
        ) != (
            1,
            1,
        ):
            raise TrendFamilyWalkForwardError(
                "Test folds must use whole calendar years."
            )

        if (
            self.test_end_exclusive.year
            != self.test_start.year + 1
        ):
            raise TrendFamilyWalkForwardError(
                "Each test fold must contain exactly one "
                "calendar year."
            )

        if (
            self.purge_sessions != 0
            or self.embargo_sessions != 0
        ):
            raise TrendFamilyWalkForwardError(
                "Day 11 uses no purge or embargo because "
                "no fitting or overlapping forward labels "
                "are present."
            )


@dataclass(frozen=True, slots=True)
class TrendFamilyWalkForwardResults:
    """Compact per-fold and aggregate out-of-sample results."""

    fold_results: pd.DataFrame
    aggregate_results: pd.DataFrame


def build_walk_forward_folds() -> tuple[
    WalkForwardFold,
    ...,
]:
    """Build the four deterministic expanding calendar folds."""

    folds = tuple(
        WalkForwardFold(
            fold_id=f"wf_{test_year}",
            train_start=DEVELOPMENT_START,
            train_end_exclusive=pd.Timestamp(
                f"{test_year}-01-01",
                tz="UTC",
            ),
            test_start=pd.Timestamp(
                f"{test_year}-01-01",
                tz="UTC",
            ),
            test_end_exclusive=pd.Timestamp(
                f"{test_year + 1}-01-01",
                tz="UTC",
            ),
        )
        for test_year in range(2022, 2026)
    )

    if len(folds) != 4 or len(
        {
            fold.fold_id
            for fold in folds
        }
    ) != 4:
        raise RuntimeError(
            "Day 11 must contain four unique folds."
        )

    return folds


def _validate_frozen_contract() -> None:
    """Reject runtime changes to the frozen strategy contract."""

    if WALK_FORWARD_STRATEGIES != (
        "trend_ratio",
        "ema_macd",
    ):
        raise TrendFamilyWalkForwardError(
            "Only the frozen Trend Ratio and EMA/MACD "
            "strategies are supported."
        )

    if WALK_FORWARD_SYMBOL != "SPY":
        raise TrendFamilyWalkForwardError(
            "Only SPY is supported by the initial Day 11 "
            "walk-forward contract."
        )

    if WALK_FORWARD_FREQUENCY != "15min":
        raise TrendFamilyWalkForwardError(
            "Only the 15min frequency is supported by the "
            "initial Day 11 walk-forward contract."
        )

    if (
        TREND_RATIO_PARAMETERS
        != DAY10_TREND_RATIO_PARAMETERS
    ):
        raise TrendFamilyWalkForwardError(
            "The frozen Day 6 Trend Ratio configuration "
            "cannot be changed."
        )

    if (
        EMA_MACD_PARAMETERS
        != DAY10_EMA_MACD_PARAMETERS
    ):
        raise TrendFamilyWalkForwardError(
            "The frozen Day 8 EMA/MACD configuration "
            "cannot be changed."
        )

    if CONFIGURATION_IDS != (
        DAY10_CONFIGURATION_IDS
    ):
        raise TrendFamilyWalkForwardError(
            "Frozen strategy configuration identifiers "
            "cannot be changed."
        )


def _normalize_timestamps(
    values: pd.Series,
) -> pd.Series:
    """Normalize timestamps to UTC without mutating input."""

    try:
        normalized = pd.to_datetime(
            values.copy(deep=True),
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendFamilyWalkForwardError(
            "Development bars contain malformed "
            "timestamps."
        ) from exc

    if normalized.isna().any():
        raise TrendFamilyWalkForwardError(
            "Development timestamps cannot be missing."
        )

    return normalized


def _validate_date_and_symbol_scope(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate development dates and the canonical symbol universe."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    if bars.empty:
        raise TrendFamilyWalkForwardError(
            "bars must not be empty."
        )

    required = {
        "timestamp",
        "symbol",
    }
    missing = sorted(
        required.difference(bars.columns)
    )

    if missing:
        raise TrendFamilyWalkForwardError(
            "Development bars are missing required "
            f"columns: {missing}."
        )

    result = bars.copy(deep=True)
    result["timestamp"] = (
        _normalize_timestamps(
            result["timestamp"]
        )
    )
    normalized_symbols = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        normalized_symbols.isna().any()
        or normalized_symbols.eq("").any()
    ):
        raise TrendFamilyWalkForwardError(
            "Development symbols cannot be missing or "
            "empty."
        )

    actual_symbols = set(
        normalized_symbols.astype(str)
    )
    expected_symbols = set(
        REQUIRED_INPUT_SYMBOLS
    )

    if actual_symbols != expected_symbols:
        raise TrendFamilyWalkForwardError(
            "Development bars must contain exactly the "
            "canonical symbols. "
            f"Expected symbols: "
            f"{sorted(expected_symbols)}; "
            f"actual symbols: "
            f"{sorted(actual_symbols)}."
        )

    if not result["symbol"].astype(
        "string"
    ).eq(normalized_symbols).all():
        raise TrendFamilyWalkForwardError(
            "Development symbols must already use "
            "canonical uppercase identifiers."
        )

    local_dates = (
        result["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    start = DEVELOPMENT_START.tz_localize(
        None
    )
    end_exclusive = (
        DEVELOPMENT_END_EXCLUSIVE
        .tz_localize(None)
    )
    outside = (
        local_dates.lt(start)
        | local_dates.ge(end_exclusive)
    )

    if outside.any():
        outside_timestamps = result.loc[
            outside,
            "timestamp",
        ]
        locked_rows = int(
            outside_timestamps.ge(
                LOCKED_PERIOD_START
            ).sum()
        )

        raise TrendFamilyWalkForwardError(
            "Development bars contain observations outside "
            "the development period from 2020-01-02 "
            "through 2025-12-31. "
            f"Outside rows: {int(outside.sum())}; "
            f"locked rows: {locked_rows}."
        )

    return result


def _validate_complete_coverage(
    bars: pd.DataFrame,
) -> None:
    """Validate complete boundary, year, and symbol-session coverage."""

    session_dates = pd.to_datetime(
        bars["session_date"],
        utc=True,
        errors="raise",
    ).dt.normalize()
    expected_start = DEVELOPMENT_START
    expected_end = (
        DEVELOPMENT_END_EXCLUSIVE
        - pd.Timedelta(days=1)
    )

    if (
        session_dates.min()
        != expected_start
        or session_dates.max()
        != expected_end
    ):
        raise TrendFamilyWalkForwardError(
            "Complete development coverage must begin "
            "on 2020-01-02 and end on 2025-12-31."
        )

    coverage = pd.DataFrame(
        {
            "symbol": bars["symbol"],
            "session_date": session_dates,
        }
    ).drop_duplicates()
    calendars = {
        symbol: frozenset(
            group["session_date"]
        )
        for symbol, group in coverage.groupby(
            "symbol",
            observed=True,
            sort=False,
        )
    }

    if len(set(calendars.values())) != 1:
        raise TrendFamilyWalkForwardError(
            "Complete development coverage requires the "
            "same whole-session calendar for every "
            "symbol."
        )

    for symbol, group in coverage.groupby(
        "symbol",
        observed=True,
        sort=False,
    ):
        years = frozenset(
            group["session_date"].dt.year
        )

        if years != EXPECTED_DEVELOPMENT_YEARS:
            raise TrendFamilyWalkForwardError(
                "Complete development coverage requires "
                "at least one whole session in every year "
                f"from 2020 through 2025 for {symbol}."
            )


def _prepare_development_features(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate canonical sessions and build 15-minute returns."""

    scoped = _validate_date_and_symbol_scope(
        bars
    )

    try:
        aggregated = aggregate_session_bars(
            scoped,
            WALK_FORWARD_FREQUENCY,
        )
    except SessionAggregationError as exc:
        raise TrendFamilyWalkForwardError(
            "Whole-session validation failed: "
            f"{exc}"
        ) from exc

    _validate_complete_coverage(
        aggregated
    )
    features = build_return_features(
        aggregated,
        expected_symbols=(
            REQUIRED_INPUT_SYMBOLS
        ),
    ).bars
    spy = (
        features.loc[
            features["symbol"].eq(
                WALK_FORWARD_SYMBOL
            )
        ]
        .copy(deep=True)
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if spy.empty:
        raise RuntimeError(
            "Validated development data contain no SPY "
            "observations."
        )

    return spy


def _session_timestamps(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return normalized UTC session dates."""

    try:
        sessions = pd.to_datetime(
            frame["session_date"],
            utc=True,
            errors="raise",
        ).dt.normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise TrendFamilyWalkForwardError(
            "Strategy data require valid session_date "
            "values."
        ) from exc

    if sessions.isna().any():
        raise TrendFamilyWalkForwardError(
            "Strategy session dates cannot be missing."
        )

    return sessions


def _partition_fold(
    features: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Partition one fold using whole exchange sessions."""

    sessions = _session_timestamps(
        features
    )
    train_mask = (
        sessions.ge(fold.train_start)
        & sessions.lt(
            fold.train_end_exclusive
        )
    )
    test_mask = (
        sessions.ge(fold.test_start)
        & sessions.lt(
            fold.test_end_exclusive
        )
    )
    train = (
        features.loc[train_mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    test = (
        features.loc[test_mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    if train.empty or test.empty:
        raise TrendFamilyWalkForwardError(
            f"{fold.fold_id} lacks complete train or test "
            "coverage."
        )

    train_sessions = set(
        _session_timestamps(train)
    )
    test_sessions = set(
        _session_timestamps(test)
    )

    if train_sessions.intersection(
        test_sessions
    ):
        raise TrendFamilyWalkForwardError(
            f"{fold.fold_id} splits a whole session "
            "between train and test."
        )

    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    if not combined[
        "timestamp"
    ].is_monotonic_increasing:
        raise TrendFamilyWalkForwardError(
            f"{fold.fold_id} is not chronologically "
            "ordered."
        )

    return train, test


def _build_strategy_observations(
    frame: pd.DataFrame,
    *,
    strategy: str,
) -> tuple[
    pd.DataFrame,
    float,
]:
    """Execute one frozen strategy over training and test history."""

    if strategy == "trend_ratio":
        parameters = (
            TREND_RATIO_PARAMETERS
        )
        bundle = build_trend_ratio_strategy(
            frame,
            parameters=parameters,
        )
    elif strategy == "ema_macd":
        parameters = EMA_MACD_PARAMETERS
        bundle = build_ema_macd_strategy(
            frame,
            parameters=parameters,
        )
    else:
        raise RuntimeError(
            f"Unsupported frozen strategy: {strategy}."
        )

    observations = (
        bundle.observations.copy(deep=True)
    )

    if len(observations) != len(frame):
        raise RuntimeError(
            "A frozen strategy changed the observation "
            "count."
        )

    return (
        observations,
        float(
            parameters.cost_bps_per_turnover
        ),
    )


def _reset_test_execution(
    observations: pd.DataFrame,
    *,
    fold: WalkForwardFold,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Reset position and execution state at one test boundary."""

    required = {
        "timestamp",
        "symbol",
        "signal",
        "position",
        "position_eligible",
        "turnover",
        "gross_strategy_return",
        "net_strategy_return",
        "close_to_close_simple_return",
    }
    missing = sorted(
        required.difference(
            observations.columns
        )
    )

    if missing:
        raise TrendFamilyWalkForwardError(
            "Strategy observations are missing reset "
            f"columns: {missing}."
        )

    sessions = _session_timestamps(
        observations
    )
    test_mask = (
        sessions.ge(fold.test_start)
        & sessions.lt(
            fold.test_end_exclusive
        )
    )
    test = (
        observations.loc[test_mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    if test.empty:
        raise RuntimeError(
            f"{fold.fold_id} produced no test strategy "
            "observations."
        )

    numeric_signal = pd.to_numeric(
        test["signal"],
        errors="coerce",
    )

    if (
        numeric_signal.isna().any()
        or not numeric_signal.isin(
            (-1, 0, 1)
        ).all()
    ):
        raise TrendFamilyWalkForwardError(
            "Strategy signals must contain only -1, 0, "
            "or 1."
        )

    test["position"] = (
        numeric_signal.shift(
            1,
            fill_value=0,
        )
        .astype("int8")
    )

    if "signal_available" in test.columns:
        signal_available = test[
            "signal_available"
        ].astype(bool)
    else:
        signal_available = test[
            "position_eligible"
        ].astype(bool)

    test["position_eligible"] = (
        signal_available.shift(
            1,
            fill_value=False,
        ).astype(bool)
    )
    test["turnover"] = calculate_turnover(
        test["position"],
        test["symbol"],
    )
    raw_return = pd.to_numeric(
        test[
            "close_to_close_simple_return"
        ],
        errors="coerce",
    ).fillna(0.0)
    test[
        "gross_strategy_return"
    ] = (
        test["position"].astype("float64")
        * raw_return.astype("float64")
    )
    test["transaction_cost"] = (
        test["turnover"].astype("float64")
        * cost_bps_per_turnover
        / 10_000.0
    )
    test["net_strategy_return"] = (
        test[
            "gross_strategy_return"
        ]
        - test["transaction_cost"]
    )

    if (
        int(test.iloc[0]["position"]) != 0
        or float(
            test.iloc[0]["turnover"]
        ) != 0.0
    ):
        raise RuntimeError(
            "Fold-boundary execution did not reset to "
            "neutral."
        )

    expected_position = (
        test["signal"]
        .shift(
            1,
            fill_value=0,
        )
        .astype("int8")
    )

    if not test[
        "position"
    ].eq(expected_position).all():
        raise RuntimeError(
            "Test positions do not preserve the existing "
            "one-observation execution delay."
        )

    return test


def _observed_annualization_factor(
    *,
    observations: int,
    sessions: int,
) -> float:
    """Annualize using observed complete-session bar density."""

    if observations <= 0 or sessions <= 0:
        raise TrendFamilyWalkForwardError(
            "Annualization requires positive test "
            "observations and sessions."
        )

    return float(
        TRADING_SESSIONS_PER_YEAR
        * observations
        / sessions
    )


def _calculate_exposures(
    observations: pd.DataFrame,
) -> tuple[
    int,
    int,
    float,
    float,
    float,
    float,
]:
    """Calculate exposure over position-eligible test rows."""

    eligible_mask = observations[
        "position_eligible"
    ].astype(bool)
    eligible = observations.loc[
        eligible_mask,
        "position",
    ]
    active = int(eligible_mask.sum())
    warmup = int(
        len(observations) - active
    )

    if eligible.empty:
        undefined = float("nan")

        return (
            warmup,
            active,
            undefined,
            undefined,
            undefined,
            undefined,
        )

    long_exposure = float(
        100.0 * eligible.eq(1).mean()
    )
    short_exposure = float(
        100.0 * eligible.eq(-1).mean()
    )
    flat_exposure = float(
        100.0 * eligible.eq(0).mean()
    )

    return (
        warmup,
        active,
        long_exposure + short_exposure,
        long_exposure,
        short_exposure,
        flat_exposure,
    )


def _performance_record(
    metrics: PerformanceMetrics,
) -> dict[str, float]:
    """Map shared metrics to the Day 11 naming contract."""

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
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": (
            metrics.max_drawdown
        ),
    }


def _build_fold_record(
    *,
    strategy: str,
    fold: WalkForwardFold,
    train: pd.DataFrame,
    test_source: pd.DataFrame,
    test_observations: pd.DataFrame,
) -> dict[str, object]:
    """Build one compact test-only fold record."""

    train_sessions = int(
        train["session_date"].nunique()
    )
    test_sessions = int(
        test_source[
            "session_date"
        ].nunique()
    )
    annualization_factor = (
        _observed_annualization_factor(
            observations=len(
                test_observations
            ),
            sessions=test_sessions,
        )
    )
    metrics = calculate_performance_metrics(
        test_observations[
            "net_strategy_return"
        ],
        annualization_factor=(
            annualization_factor
        ),
    )

    if metrics.observations != len(
        test_observations
    ):
        raise RuntimeError(
            "Fold performance metrics changed the test "
            "observation count."
        )

    (
        warmup,
        active,
        average_exposure,
        long_exposure,
        short_exposure,
        flat_exposure,
    ) = _calculate_exposures(
        test_observations
    )

    return {
        "strategy": strategy,
        "symbol": WALK_FORWARD_SYMBOL,
        "frequency": (
            WALK_FORWARD_FREQUENCY
        ),
        "fold_id": fold.fold_id,
        "configuration_id": (
            CONFIGURATION_IDS[strategy]
        ),
        "train_start_timestamp": (
            train["timestamp"].min()
        ),
        "train_end_timestamp": (
            train["timestamp"].max()
        ),
        "test_start_timestamp": (
            test_source["timestamp"].min()
        ),
        "test_end_timestamp": (
            test_source["timestamp"].max()
        ),
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "train_observations": int(
            len(train)
        ),
        "test_observations": int(
            len(test_observations)
        ),
        "annualization_factor": (
            annualization_factor
        ),
        "purge_sessions": (
            fold.purge_sessions
        ),
        "embargo_sessions": (
            fold.embargo_sessions
        ),
        "indicator_history_observations": (
            int(len(train))
        ),
        "initial_test_position": int(
            test_observations.iloc[
                0
            ]["position"]
        ),
        "initial_test_turnover": float(
            test_observations.iloc[
                0
            ]["turnover"]
        ),
        "warmup_observations": warmup,
        "active_observations": active,
        **_performance_record(metrics),
        "turnover": float(
            test_observations[
                "turnover"
            ].sum()
        ),
        "average_exposure": (
            average_exposure
        ),
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "flat_exposure": flat_exposure,
        "trade_count": int(
            test_observations[
                "turnover"
            ].gt(0.0).sum()
        ),
    }


def _build_aggregate_record(
    *,
    strategy: str,
    fold_observations: list[
        pd.DataFrame
    ],
) -> dict[str, object]:
    """Recompute aggregate metrics from test-fold rows only."""

    combined = pd.concat(
        fold_observations,
        ignore_index=True,
    )
    test_sessions = int(
        combined[
            "session_date"
        ].nunique()
    )
    annualization_factor = (
        _observed_annualization_factor(
            observations=len(combined),
            sessions=test_sessions,
        )
    )
    metrics = calculate_performance_metrics(
        combined[
            "net_strategy_return"
        ],
        annualization_factor=(
            annualization_factor
        ),
    )

    if metrics.observations != len(combined):
        raise RuntimeError(
            "Aggregate performance metrics changed the "
            "test observation count."
        )

    (
        _,
        _,
        average_exposure,
        long_exposure,
        short_exposure,
        flat_exposure,
    ) = _calculate_exposures(combined)

    return {
        "strategy": strategy,
        "symbol": WALK_FORWARD_SYMBOL,
        "frequency": (
            WALK_FORWARD_FREQUENCY
        ),
        "configuration_id": (
            CONFIGURATION_IDS[strategy]
        ),
        "folds": len(
            fold_observations
        ),
        "test_start_timestamp": (
            combined["timestamp"].min()
        ),
        "test_end_timestamp": (
            combined["timestamp"].max()
        ),
        "test_sessions": test_sessions,
        "test_observations": int(
            len(combined)
        ),
        "annualization_factor": (
            annualization_factor
        ),
        **_performance_record(metrics),
        "turnover": float(
            combined["turnover"].sum()
        ),
        "average_exposure": (
            average_exposure
        ),
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "flat_exposure": flat_exposure,
        "trade_count": int(
            combined["turnover"]
            .gt(0.0)
            .sum()
        ),
    }


def _validate_metrics(
    frame: pd.DataFrame,
) -> None:
    """Validate defined metrics without rejecting losses."""

    always_finite = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "turnover",
    )

    for column in always_finite:
        if not frame[
            column
        ].map(math.isfinite).all():
            raise RuntimeError(
                f"Defined {column} values must be finite."
            )

    defined_sharpe = frame[
        "sharpe_ratio"
    ].dropna()

    if not defined_sharpe.map(
        math.isfinite
    ).all():
        raise RuntimeError(
            "Defined Sharpe ratios must be finite."
        )


def _validate_results(
    fold_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
) -> None:
    """Validate deterministic Day 11 output invariants."""

    if tuple(fold_results.columns) != (
        FOLD_RESULT_COLUMNS
    ):
        raise RuntimeError(
            "Day 11 fold result schema changed."
        )

    if tuple(
        aggregate_results.columns
    ) != AGGREGATE_RESULT_COLUMNS:
        raise RuntimeError(
            "Day 11 aggregate result schema changed."
        )

    expected_fold_keys = [
        (
            strategy,
            fold.fold_id,
        )
        for strategy in (
            WALK_FORWARD_STRATEGIES
        )
        for fold in (
            build_walk_forward_folds()
        )
    ]
    actual_fold_keys = list(
        fold_results[
            [
                "strategy",
                "fold_id",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if (
        len(fold_results) != 8
        or actual_fold_keys
        != expected_fold_keys
    ):
        raise RuntimeError(
            "Day 11 must retain eight ordered fold runs."
        )

    if aggregate_results[
        "strategy"
    ].tolist() != list(
        WALK_FORWARD_STRATEGIES
    ):
        raise RuntimeError(
            "Day 11 aggregate strategy order changed."
        )

    aggregate_ids = aggregate_results[
        "strategy"
    ].map(CONFIGURATION_IDS)

    if not aggregate_results[
        "configuration_id"
    ].eq(aggregate_ids).all():
        raise RuntimeError(
            "Aggregate configuration identifiers changed."
        )

    if (
        len(aggregate_results) != 2
        or not fold_results[
            "symbol"
        ].eq(WALK_FORWARD_SYMBOL).all()
        or not fold_results[
            "frequency"
        ].eq(
            WALK_FORWARD_FREQUENCY
        ).all()
    ):
        raise RuntimeError(
            "Day 11 result scope changed."
        )

    expected_ids = fold_results[
        "strategy"
    ].map(CONFIGURATION_IDS)

    if not fold_results[
        "configuration_id"
    ].eq(expected_ids).all():
        raise RuntimeError(
            "Frozen configuration identifiers changed."
        )

    expected_factor = (
        TRADING_SESSIONS_PER_YEAR
        * fold_results[
            "test_observations"
        ]
        / fold_results["test_sessions"]
    )

    if not fold_results[
        "annualization_factor"
    ].eq(expected_factor).all():
        raise RuntimeError(
            "Fold annualization does not use observed "
            "test sessions."
        )

    aggregate_factor = (
        TRADING_SESSIONS_PER_YEAR
        * aggregate_results[
            "test_observations"
        ]
        / aggregate_results[
            "test_sessions"
        ]
    )

    if not aggregate_results[
        "annualization_factor"
    ].eq(aggregate_factor).all():
        raise RuntimeError(
            "Aggregate annualization does not use observed "
            "test sessions."
        )

    fold_totals = (
        fold_results.groupby(
            "strategy",
            observed=True,
            sort=False,
        )
        .agg(
            test_sessions=(
                "test_sessions",
                "sum",
            ),
            test_observations=(
                "test_observations",
                "sum",
            ),
            folds=(
                "fold_id",
                "size",
            ),
        )
        .reset_index()
    )
    aggregate_counts = aggregate_results[
        [
            "strategy",
            "test_sessions",
            "test_observations",
            "folds",
        ]
    ].reset_index(drop=True)

    if not aggregate_counts.equals(
        fold_totals
    ):
        raise RuntimeError(
            "Aggregate counts do not reconcile to the four "
            "test folds."
        )

    if (
        not fold_results[
            "purge_sessions"
        ].eq(0).all()
        or not fold_results[
            "embargo_sessions"
        ].eq(0).all()
    ):
        raise RuntimeError(
            "Day 11 purge or embargo policy changed."
        )

    if (
        not fold_results[
            "initial_test_position"
        ].eq(0).all()
        or not fold_results[
            "initial_test_turnover"
        ].eq(0.0).all()
    ):
        raise RuntimeError(
            "Test execution state must reset at every "
            "fold."
        )

    if not fold_results[
        "warmup_observations"
    ].add(
        fold_results[
            "active_observations"
        ]
    ).eq(
        fold_results[
            "test_observations"
        ]
    ).all():
        raise RuntimeError(
            "Test warmup and active observations do not "
            "reconcile."
        )

    _validate_metrics(fold_results)
    _validate_metrics(
        aggregate_results
    )


def run_trend_family_walk_forward(
    bars: pd.DataFrame,
) -> TrendFamilyWalkForwardResults:
    """Run the frozen SPY 15-minute expanding walk-forward study."""

    _validate_frozen_contract()
    features = (
        _prepare_development_features(
            bars
        )
    )
    folds = build_walk_forward_folds()
    fold_records: list[
        dict[str, object]
    ] = []
    aggregate_inputs: dict[
        str,
        list[pd.DataFrame],
    ] = {
        strategy: []
        for strategy in (
            WALK_FORWARD_STRATEGIES
        )
    }

    for strategy in (
        WALK_FORWARD_STRATEGIES
    ):
        for fold in folds:
            train, test_source = (
                _partition_fold(
                    features,
                    fold,
                )
            )
            combined = pd.concat(
                [
                    train,
                    test_source,
                ],
                ignore_index=True,
            )
            (
                observations,
                cost_bps_per_turnover,
            ) = _build_strategy_observations(
                combined,
                strategy=strategy,
            )
            test_observations = (
                _reset_test_execution(
                    observations,
                    fold=fold,
                    cost_bps_per_turnover=(
                        cost_bps_per_turnover
                    ),
                )
            )
            fold_records.append(
                _build_fold_record(
                    strategy=strategy,
                    fold=fold,
                    train=train,
                    test_source=test_source,
                    test_observations=(
                        test_observations
                    ),
                )
            )
            aggregate_inputs[
                strategy
            ].append(
                test_observations
            )

    fold_results = (
        pd.DataFrame.from_records(
            fold_records,
            columns=(
                FOLD_RESULT_COLUMNS
            ),
        )
    )
    aggregate_results = (
        pd.DataFrame.from_records(
            [
                _build_aggregate_record(
                    strategy=strategy,
                    fold_observations=(
                        aggregate_inputs[
                            strategy
                        ]
                    ),
                )
                for strategy in (
                    WALK_FORWARD_STRATEGIES
                )
            ],
            columns=(
                AGGREGATE_RESULT_COLUMNS
            ),
        )
    )
    _validate_results(
        fold_results,
        aggregate_results,
    )

    return TrendFamilyWalkForwardResults(
        fold_results=fold_results,
        aggregate_results=(
            aggregate_results
        ),
    )
