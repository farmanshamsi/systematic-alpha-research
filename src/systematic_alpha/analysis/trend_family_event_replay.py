"""Immutable contracts for deterministic trend-family event replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from numbers import Integral, Real
from typing import Final, TypeAlias

import pandas as pd

from systematic_alpha.analysis.eda_features import (
    EdaFeatureError,
    REQUIRED_COLUMNS as CANONICAL_BAR_COLUMNS,
    build_return_features,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    build_wealth_index,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    CONFIGURATION_IDS,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS,
)
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdError,
    build_ema_macd_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    TrendRatioError,
    build_trend_ratio_strategy,
)


ACCOUNTING_TOLERANCE: Final[float] = 1e-12
SUPPORTED_SYMBOL: Final[str] = "SPY"
SUPPORTED_FREQUENCY: Final[str] = "15min"
FROZEN_SIGNAL_OBSERVATION_COLUMNS: Final[
    tuple[str, ...]
] = (
    "timestamp",
    "symbol",
    "signal",
    "signal_available",
)
REPLAY_LEDGER_COLUMNS: Final[
    tuple[str, ...]
] = (
    "bar_index",
    "timestamp",
    "session_date",
    "symbol",
    "strategy",
    "configuration_id",
    "close",
    "asset_return",
    "signal_available",
    "target_position",
    "position_eligible",
    "previous_executed_position",
    "executed_position",
    "position_change",
    "turnover",
    "cost_bps_per_turnover",
    "transaction_cost",
    "transaction_cost_amount",
    "gross_strategy_return",
    "net_strategy_return",
    "previous_equity",
    "gross_ending_equity",
    "cash_balance",
    "holdings_value",
    "ending_equity",
    "order_submitted",
    "fill_executed",
)


class TrendFamilyEventReplayError(ValueError):
    """Raised when a Day 12 replay contract is invalid."""


def _validate_nonnegative_index(
    value: object,
    *,
    name: str,
) -> int:
    """Validate an event sequence or bar index."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a nonnegative integer."
        )

    return int(value)


def _normalize_timestamp(
    value: object,
    *,
    name: str,
) -> pd.Timestamp:
    """Require a timezone-aware timestamp and convert it to UTC."""

    if (
        not isinstance(value, pd.Timestamp)
        or value.tzinfo is None
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a timezone-aware pandas Timestamp."
        )

    return value.tz_convert("UTC")


def _normalize_symbol(
    value: object,
) -> str:
    """Normalize and validate the frozen replay symbol."""

    if not isinstance(value, str):
        raise TrendFamilyEventReplayError(
            "symbol must be a string containing SPY."
        )

    normalized = value.strip().upper()

    if normalized != SUPPORTED_SYMBOL:
        raise TrendFamilyEventReplayError(
            "Only SPY is supported by the Day 12 replay contract."
        )

    return normalized


def _validate_iso_session_date(
    value: object,
) -> str:
    """Require one exact ISO calendar-date string."""

    if not isinstance(value, str):
        raise TrendFamilyEventReplayError(
            "session_date must be an ISO date string."
        )

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise TrendFamilyEventReplayError(
            "session_date must be a valid ISO date."
        ) from exc

    if parsed.isoformat() != value:
        raise TrendFamilyEventReplayError(
            "session_date must use exact YYYY-MM-DD format."
        )

    return value


def _validate_finite_real(
    value: object,
    *,
    name: str,
) -> float:
    """Validate one finite non-boolean real value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a finite real number."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise TrendFamilyEventReplayError(
            f"{name} must be finite."
        )

    return normalized


def _validate_nonnegative_real(
    value: object,
    *,
    name: str,
) -> float:
    """Validate one finite nonnegative real value."""

    normalized = _validate_finite_real(
        value,
        name=name,
    )

    if normalized < 0.0:
        raise TrendFamilyEventReplayError(
            f"{name} must be nonnegative."
        )

    return normalized


def _validate_positive_real(
    value: object,
    *,
    name: str,
) -> float:
    """Validate one finite strictly positive real value."""

    normalized = _validate_finite_real(
        value,
        name=name,
    )

    if normalized <= 0.0:
        raise TrendFamilyEventReplayError(
            f"{name} must be strictly positive."
        )

    return normalized


def _validate_nonnegative_integer(
    value: object,
    *,
    name: str,
) -> int:
    """Validate one finite nonnegative integer value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a nonnegative integer."
        )

    return int(value)


def _validate_position(
    value: object,
    *,
    name: str,
) -> int:
    """Require a normalized target or executed position."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in (-1, 0, 1)
    ):
        raise TrendFamilyEventReplayError(
            f"{name} position must belong to {{-1, 0, 1}}."
        )

    return int(value)


def _validate_position_change(
    value: object,
) -> int:
    """Require an integral normalized-position change."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise TrendFamilyEventReplayError(
            "position_change must be an integer."
        )

    return int(value)


def _validate_boolean(
    value: object,
    *,
    name: str,
) -> bool:
    """Require a real Python boolean."""

    if not isinstance(value, bool):
        raise TrendFamilyEventReplayError(
            f"{name} must be a boolean."
        )

    return value


def _validate_nonempty_string(
    value: object,
    *,
    name: str,
) -> str:
    """Require a nonempty string without changing it."""

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a nonempty string."
        )

    return value


def _validate_strategy_configuration(
    *,
    strategy: object,
    configuration_id: object,
) -> tuple[str, str]:
    """Require one existing frozen strategy/configuration pair."""

    if (
        not isinstance(strategy, str)
        or strategy not in CONFIGURATION_IDS
    ):
        raise TrendFamilyEventReplayError(
            "strategy must be one of the frozen Trend Ratio "
            "or EMA/MACD strategy identifiers."
        )

    if not isinstance(configuration_id, str):
        raise TrendFamilyEventReplayError(
            "configuration_id must match the frozen strategy."
        )

    expected = CONFIGURATION_IDS[strategy]

    if configuration_id != expected:
        raise TrendFamilyEventReplayError(
            "configuration_id does not match the frozen "
            f"{strategy} configuration."
        )

    return strategy, configuration_id


def _require_close(
    actual: object,
    expected: float,
    *,
    name: str,
) -> float:
    """Require one accounting identity within the fixed tolerance."""

    normalized = _validate_finite_real(
        actual,
        name=name,
    )

    if not math.isclose(
        normalized,
        float(expected),
        rel_tol=ACCOUNTING_TOLERANCE,
        abs_tol=ACCOUNTING_TOLERANCE,
    ):
        raise TrendFamilyEventReplayError(
            f"{name} violates the frozen accounting identity."
        )

    return normalized


def _prepare_replay_bars(
    bars: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Validate SPY source bars and rebuild 15-minute return features."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    if bars.empty:
        raise TrendFamilyEventReplayError(
            "Replay bars must not be empty."
        )

    if (
        not isinstance(frequency, str)
        or frequency != SUPPORTED_FREQUENCY
    ):
        raise TrendFamilyEventReplayError(
            "Day 12 replay frequency must be exactly 15min."
        )

    missing = sorted(
        set(CANONICAL_BAR_COLUMNS).difference(
            bars.columns
        )
    )

    if missing:
        raise TrendFamilyEventReplayError(
            "Replay bars are missing required canonical "
            f"columns: {missing}."
        )

    source = bars.copy(deep=True)

    try:
        parsed_timestamps = source[
            "timestamp"
        ].map(pd.Timestamp)
    except (TypeError, ValueError) as exc:
        raise TrendFamilyEventReplayError(
            "Replay bars contain malformed timestamps."
        ) from exc

    if parsed_timestamps.isna().any():
        raise TrendFamilyEventReplayError(
            "Replay timestamps cannot be missing."
        )

    if not parsed_timestamps.map(
        lambda value: value.tzinfo
        is not None
    ).all():
        raise TrendFamilyEventReplayError(
            "Replay timestamps must be timezone-aware."
        )

    source["timestamp"] = pd.to_datetime(
        parsed_timestamps,
        utc=True,
        errors="raise",
    )

    normalized_symbols = (
        source["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        normalized_symbols.isna().any()
        or normalized_symbols.eq("").any()
    ):
        raise TrendFamilyEventReplayError(
            "Replay symbols cannot be missing or empty."
        )

    actual_symbols = set(
        normalized_symbols.astype(str)
    )

    if actual_symbols != {SUPPORTED_SYMBOL}:
        raise TrendFamilyEventReplayError(
            "Day 12 replay bars must contain SPY only."
        )

    source["symbol"] = normalized_symbols

    if source.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise TrendFamilyEventReplayError(
            "Replay bars contain duplicate symbol-timestamp "
            "observations."
        )

    if not source[
        "timestamp"
    ].is_monotonic_increasing:
        raise TrendFamilyEventReplayError(
            "Replay bars must already be in chronological "
            "timestamp order."
        )

    local_dates = (
        source["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if local_dates.dt.year.ge(2026).any():
        raise TrendFamilyEventReplayError(
            "Replay bars must not contain any 2026 "
            "observation."
        )

    if (
        local_dates.min()
        < DEVELOPMENT_START
        or local_dates.max()
        > DEVELOPMENT_END
    ):
        raise TrendFamilyEventReplayError(
            "Replay bars must remain within the development "
            "period from 2020-01-02 through 2025-12-31."
        )

    try:
        aggregated = aggregate_session_bars(
            source,
            SUPPORTED_FREQUENCY,
        )
        prepared = build_return_features(
            aggregated,
            expected_symbols=(
                SUPPORTED_SYMBOL,
            ),
        ).bars
    except (
        EdaFeatureError,
        SessionAggregationError,
    ) as exc:
        raise TrendFamilyEventReplayError(
            "Replay-bar preparation failed: "
            f"{exc}"
        ) from exc

    if len(prepared) != len(source):
        raise RuntimeError(
            "15-minute replay preparation changed the "
            "observation count."
        )

    if (
        not prepared["symbol"].eq(
            SUPPORTED_SYMBOL
        ).all()
        or not prepared[
            "bar_frequency"
        ].eq(
            SUPPORTED_FREQUENCY
        ).all()
        or not prepared[
            "timestamp"
        ].is_monotonic_increasing
    ):
        raise RuntimeError(
            "Prepared replay bars violate the frozen "
            "SPY 15-minute ordering contract."
        )

    return prepared.copy(deep=True)


def _validate_prepared_signal_input(
    prepared_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate prepared input without rebuilding any features."""

    if not isinstance(
        prepared_bars,
        pd.DataFrame,
    ):
        raise TypeError(
            "prepared_bars must be a pandas DataFrame."
        )

    if prepared_bars.empty:
        raise TrendFamilyEventReplayError(
            "prepared_bars must not be empty."
        )

    required = {
        "timestamp",
        "session_date",
        "symbol",
        "close",
        "close_to_close_simple_return",
        "bar_frequency",
    }
    missing = sorted(
        required.difference(
            prepared_bars.columns
        )
    )

    if missing:
        raise TrendFamilyEventReplayError(
            "prepared_bars are missing required columns: "
            f"{missing}."
        )

    result = prepared_bars.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendFamilyEventReplayError(
            "prepared_bars contain malformed timestamps."
        ) from exc

    if (
        result["timestamp"].isna().any()
        or not result[
            "timestamp"
        ].is_monotonic_increasing
    ):
        raise TrendFamilyEventReplayError(
            "prepared_bars must be chronologically ordered."
        )

    if result.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise TrendFamilyEventReplayError(
            "prepared_bars contain duplicate timestamps."
        )

    if (
        not result["symbol"].eq(
            SUPPORTED_SYMBOL
        ).all()
        or not result[
            "bar_frequency"
        ].eq(
            SUPPORTED_FREQUENCY
        ).all()
    ):
        raise TrendFamilyEventReplayError(
            "prepared_bars must contain SPY 15min "
            "observations only."
        )

    return result


def _build_frozen_signal_observations(
    prepared_bars: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    """Release causal completed-bar decisions from one frozen builder."""

    if (
        not isinstance(strategy, str)
        or strategy not in CONFIGURATION_IDS
    ):
        raise TrendFamilyEventReplayError(
            "strategy must be trend_ratio or ema_macd."
        )

    prepared = (
        _validate_prepared_signal_input(
            prepared_bars
        )
    )

    try:
        if strategy == "trend_ratio":
            bundle = build_trend_ratio_strategy(
                prepared,
                parameters=(
                    TREND_RATIO_PARAMETERS
                ),
            )
        else:
            bundle = build_ema_macd_strategy(
                prepared,
                parameters=EMA_MACD_PARAMETERS,
            )
    except (
        EmaMacdError,
        TrendRatioError,
    ) as exc:
        raise TrendFamilyEventReplayError(
            "Frozen signal construction failed: "
            f"{exc}"
        ) from exc

    observations = getattr(
        bundle,
        "observations",
        None,
    )

    if not isinstance(
        observations,
        pd.DataFrame,
    ):
        raise TrendFamilyEventReplayError(
            "Frozen strategy output must contain an "
            "observations DataFrame."
        )

    required = {
        "timestamp",
        "symbol",
        "signal",
        "signal_available",
    }
    missing = sorted(
        required.difference(
            observations.columns
        )
    )

    if missing:
        raise TrendFamilyEventReplayError(
            "Frozen strategy observations are missing "
            f"signal columns: {missing}."
        )

    if len(observations) != len(
        prepared
    ):
        raise TrendFamilyEventReplayError(
            "Frozen strategy changed the prepared "
            "observation count."
        )

    observed_timestamps = pd.to_datetime(
        observations["timestamp"],
        utc=True,
        errors="coerce",
    )

    if (
        observed_timestamps.isna().any()
        or not observed_timestamps.reset_index(
            drop=True
        ).equals(
            prepared[
                "timestamp"
            ].reset_index(drop=True)
        )
        or not observations[
            "symbol"
        ].reset_index(drop=True).eq(
            prepared[
                "symbol"
            ].reset_index(drop=True)
        ).all()
    ):
        raise TrendFamilyEventReplayError(
            "Frozen strategy observations do not align "
            "with prepared bars."
        )

    raw_signal = observations[
        "signal"
    ]
    valid_signal_types = raw_signal.map(
        lambda value: (
            isinstance(value, Integral)
            and not isinstance(value, bool)
        )
    )
    numeric_signal = pd.to_numeric(
        raw_signal,
        errors="coerce",
    )

    if (
        not valid_signal_types.all()
        or numeric_signal.isna().any()
        or not numeric_signal.isin(
            (-1, 0, 1)
        ).all()
    ):
        raise TrendFamilyEventReplayError(
            "Frozen strategy signal target positions "
            "must belong to {-1, 0, 1}."
        )

    availability = observations[
        "signal_available"
    ]

    if (
        availability.isna().any()
        or not pd.api.types.is_bool_dtype(
            availability.dtype
        )
    ):
        raise TrendFamilyEventReplayError(
            "signal_available must contain boolean values."
        )

    unavailable_non_neutral = (
        ~availability.astype(bool)
        & numeric_signal.ne(0)
    )

    if unavailable_non_neutral.any():
        raise TrendFamilyEventReplayError(
            "Unavailable frozen signals must have neutral "
            "target positions."
        )

    result = observations.loc[
        :,
        FROZEN_SIGNAL_OBSERVATION_COLUMNS,
    ].copy(deep=True).reset_index(
        drop=True
    )
    result["timestamp"] = (
        observed_timestamps.reset_index(
            drop=True
        )
    )
    result["symbol"] = prepared[
        "symbol"
    ].reset_index(drop=True)
    result["signal"] = (
        numeric_signal.astype("int8")
        .reset_index(drop=True)
    )
    result["signal_available"] = (
        availability.astype(bool)
        .reset_index(drop=True)
    )

    return result


@dataclass(frozen=True, slots=True)
class MarketBarEvent:
    """One validated completed SPY market bar."""

    event_sequence: int
    bar_index: int
    timestamp: pd.Timestamp
    session_date: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    vwap: float
    source: str
    feed: str
    asset_return: float

    def __post_init__(self) -> None:
        """Validate immutable market-bar fields."""

        _validate_nonnegative_index(
            self.event_sequence,
            name="event_sequence",
        )
        _validate_nonnegative_index(
            self.bar_index,
            name="bar_index",
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_timestamp(
                self.timestamp,
                name="timestamp",
            ),
        )
        _validate_iso_session_date(
            self.session_date
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )

        open_price = _validate_positive_real(
            self.open,
            name="open",
        )
        high_price = _validate_positive_real(
            self.high,
            name="high",
        )
        low_price = _validate_positive_real(
            self.low,
            name="low",
        )
        close_price = _validate_positive_real(
            self.close,
            name="close",
        )
        _validate_nonnegative_real(
            self.volume,
            name="volume",
        )
        _validate_nonnegative_integer(
            self.trade_count,
            name="trade_count",
        )
        _validate_positive_real(
            self.vwap,
            name="vwap",
        )
        _validate_nonempty_string(
            self.source,
            name="source",
        )
        _validate_nonempty_string(
            self.feed,
            name="feed",
        )
        _validate_finite_real(
            self.asset_return,
            name="asset_return",
        )

        if not (
            low_price
            <= open_price
            <= high_price
            and low_price
            <= close_price
            <= high_price
        ):
            raise TrendFamilyEventReplayError(
                "Market-bar OHLC values are inconsistent."
            )


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """One frozen-strategy decision from a completed market bar."""

    event_sequence: int
    bar_index: int
    timestamp: pd.Timestamp
    symbol: str
    strategy: str
    configuration_id: str
    target_position: int
    signal_available: bool

    def __post_init__(self) -> None:
        """Validate the causal target-position decision."""

        _validate_nonnegative_index(
            self.event_sequence,
            name="event_sequence",
        )
        _validate_nonnegative_index(
            self.bar_index,
            name="bar_index",
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_timestamp(
                self.timestamp,
                name="timestamp",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )
        _validate_strategy_configuration(
            strategy=self.strategy,
            configuration_id=(
                self.configuration_id
            ),
        )
        target = _validate_position(
            self.target_position,
            name="target",
        )
        available = _validate_boolean(
            self.signal_available,
            name="signal_available",
        )

        if not available and target != 0:
            raise TrendFamilyEventReplayError(
                "An unavailable signal must have a neutral "
                "target position."
            )


@dataclass(frozen=True, slots=True)
class TargetPositionOrderEvent:
    """One target-position order scheduled for the next bar."""

    event_sequence: int
    submitted_bar_index: int
    execute_bar_index: int
    submitted_timestamp: pd.Timestamp
    symbol: str
    strategy: str
    configuration_id: str
    current_executed_position: int
    target_position: int

    def __post_init__(self) -> None:
        """Validate next-observation target execution."""

        _validate_nonnegative_index(
            self.event_sequence,
            name="event_sequence",
        )
        submitted = _validate_nonnegative_index(
            self.submitted_bar_index,
            name="submitted_bar_index",
        )
        execute = _validate_nonnegative_index(
            self.execute_bar_index,
            name="execute_bar_index",
        )
        object.__setattr__(
            self,
            "submitted_timestamp",
            _normalize_timestamp(
                self.submitted_timestamp,
                name="submitted_timestamp",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )
        _validate_strategy_configuration(
            strategy=self.strategy,
            configuration_id=(
                self.configuration_id
            ),
        )
        current = _validate_position(
            self.current_executed_position,
            name="current_executed",
        )
        target = _validate_position(
            self.target_position,
            name="target",
        )

        if execute != submitted + 1:
            raise TrendFamilyEventReplayError(
                "An order must execute exactly one observation "
                "after submission."
            )

        if target == current:
            raise TrendFamilyEventReplayError(
                "An order must request an actual position change."
            )


@dataclass(frozen=True, slots=True)
class FillEvent:
    """One normalized-position fill on the next market bar."""

    event_sequence: int
    bar_index: int
    timestamp: pd.Timestamp
    symbol: str
    strategy: str
    configuration_id: str
    submitted_bar_index: int
    previous_position: int
    executed_position: int
    position_change: int
    turnover: float
    cost_bps_per_turnover: float
    transaction_cost: float

    def __post_init__(self) -> None:
        """Validate fill timing, turnover, and fractional cost."""

        _validate_nonnegative_index(
            self.event_sequence,
            name="event_sequence",
        )
        bar_index = _validate_nonnegative_index(
            self.bar_index,
            name="bar_index",
        )
        submitted = _validate_nonnegative_index(
            self.submitted_bar_index,
            name="submitted_bar_index",
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_timestamp(
                self.timestamp,
                name="timestamp",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )
        _validate_strategy_configuration(
            strategy=self.strategy,
            configuration_id=(
                self.configuration_id
            ),
        )
        previous = _validate_position(
            self.previous_position,
            name="previous",
        )
        executed = _validate_position(
            self.executed_position,
            name="executed",
        )
        change = _validate_position_change(
            self.position_change
        )
        turnover = _validate_nonnegative_real(
            self.turnover,
            name="turnover",
        )
        cost_bps = _validate_nonnegative_real(
            self.cost_bps_per_turnover,
            name="cost_bps_per_turnover",
        )
        transaction_cost = (
            _validate_nonnegative_real(
                self.transaction_cost,
                name="transaction_cost",
            )
        )

        if bar_index != submitted + 1:
            raise TrendFamilyEventReplayError(
                "Fill bar_index must be the next observation "
                "after submitted_bar_index."
            )

        expected_change = executed - previous

        if change != expected_change:
            raise TrendFamilyEventReplayError(
                "position_change must equal executed_position "
                "minus previous_position."
            )

        if change == 0:
            raise TrendFamilyEventReplayError(
                "A fill must represent an actual position change."
            )

        _require_close(
            turnover,
            abs(expected_change),
            name="turnover",
        )
        _require_close(
            transaction_cost,
            turnover * cost_bps / 10_000.0,
            name="transaction_cost",
        )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """One normalized-notional portfolio record after a completed bar.

    Accounting identities use ``ACCOUNTING_TOLERANCE`` as both the
    relative and absolute floating-point tolerance.
    """

    event_sequence: int
    bar_index: int
    timestamp: pd.Timestamp
    symbol: str
    strategy: str
    configuration_id: str
    position_eligible: bool
    previous_position: int
    executed_position: int
    position_change: int
    turnover: float
    asset_return: float
    gross_strategy_return: float
    transaction_cost: float
    transaction_cost_amount: float
    net_strategy_return: float
    previous_equity: float
    gross_ending_equity: float
    cash_balance: float
    holdings_value: float
    ending_equity: float

    def __post_init__(self) -> None:
        """Validate all frozen return-ledger identities."""

        _validate_nonnegative_index(
            self.event_sequence,
            name="event_sequence",
        )
        _validate_nonnegative_index(
            self.bar_index,
            name="bar_index",
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_timestamp(
                self.timestamp,
                name="timestamp",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )
        _validate_strategy_configuration(
            strategy=self.strategy,
            configuration_id=(
                self.configuration_id
            ),
        )
        _validate_boolean(
            self.position_eligible,
            name="position_eligible",
        )
        previous_position = _validate_position(
            self.previous_position,
            name="previous",
        )
        executed_position = _validate_position(
            self.executed_position,
            name="executed",
        )
        position_change = (
            _validate_position_change(
                self.position_change
            )
        )
        turnover = _validate_nonnegative_real(
            self.turnover,
            name="turnover",
        )
        asset_return = _validate_finite_real(
            self.asset_return,
            name="asset_return",
        )
        gross_return = _validate_finite_real(
            self.gross_strategy_return,
            name="gross_strategy_return",
        )
        transaction_cost = (
            _validate_nonnegative_real(
                self.transaction_cost,
                name="transaction_cost",
            )
        )
        transaction_cost_amount = (
            _validate_nonnegative_real(
                self.transaction_cost_amount,
                name="transaction_cost_amount",
            )
        )
        net_return = _validate_finite_real(
            self.net_strategy_return,
            name="net_strategy_return",
        )
        previous_equity = _validate_positive_real(
            self.previous_equity,
            name="previous_equity",
        )
        gross_ending_equity = (
            _validate_finite_real(
                self.gross_ending_equity,
                name="gross_ending_equity",
            )
        )
        cash_balance = _validate_finite_real(
            self.cash_balance,
            name="cash_balance",
        )
        holdings_value = _validate_finite_real(
            self.holdings_value,
            name="holdings_value",
        )
        ending_equity = _validate_finite_real(
            self.ending_equity,
            name="ending_equity",
        )

        expected_change = (
            executed_position
            - previous_position
        )

        if position_change != expected_change:
            raise TrendFamilyEventReplayError(
                "position_change must equal executed_position "
                "minus previous_position."
            )

        _require_close(
            turnover,
            abs(expected_change),
            name="turnover",
        )
        _require_close(
            gross_return,
            executed_position * asset_return,
            name="gross_strategy_return",
        )
        _require_close(
            net_return,
            gross_return - transaction_cost,
            name="net_strategy_return",
        )
        _require_close(
            transaction_cost_amount,
            previous_equity * transaction_cost,
            name="transaction_cost_amount",
        )
        _require_close(
            gross_ending_equity,
            previous_equity
            * (1.0 + gross_return),
            name="gross_ending_equity",
        )

        expected_cash = (
            previous_equity
            * (1.0 - executed_position)
            - transaction_cost_amount
        )
        expected_holdings = (
            executed_position
            * previous_equity
            * (1.0 + asset_return)
        )
        _require_close(
            cash_balance,
            expected_cash,
            name="cash_balance",
        )
        _require_close(
            holdings_value,
            expected_holdings,
            name="holdings_value",
        )
        _require_close(
            ending_equity,
            cash_balance + holdings_value,
            name=(
                "cash_balance plus holdings_value "
                "ending_equity"
            ),
        )
        _require_close(
            ending_equity,
            previous_equity
            * (1.0 + net_return),
            name="ending_equity",
        )


ReplayEvent: TypeAlias = (
    MarketBarEvent
    | SignalEvent
    | TargetPositionOrderEvent
    | FillEvent
    | PortfolioSnapshot
)


def _performance_metrics_equal(
    actual: PerformanceMetrics,
    expected: PerformanceMetrics,
) -> bool:
    """Compare shared metrics while treating paired NaNs as equal."""

    if actual.observations != expected.observations:
        return False

    for name in (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ):
        actual_value = float(
            getattr(actual, name)
        )
        expected_value = float(
            getattr(expected, name)
        )

        if (
            math.isnan(actual_value)
            and math.isnan(expected_value)
        ):
            continue

        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=ACCOUNTING_TOLERANCE,
            abs_tol=ACCOUNTING_TOLERANCE,
        ):
            return False

    return True


@dataclass(frozen=True, slots=True)
class TrendFamilyEventReplayResult:
    """Immutable metadata and copied observations for one replay.

    The dataclass prevents attribute replacement. Use
    :meth:`copy_observations` when callers need a mutable working copy
    without changing the frame retained by this result.
    """

    strategy: str
    symbol: str
    frequency: str
    configuration_id: str
    evaluation_start: pd.Timestamp
    evaluation_end_exclusive: pd.Timestamp
    events: tuple[ReplayEvent, ...]
    observations: pd.DataFrame
    gross_performance: PerformanceMetrics
    net_performance: PerformanceMetrics

    def __post_init__(self) -> None:
        """Validate cross-field replay and performance consistency."""

        _validate_strategy_configuration(
            strategy=self.strategy,
            configuration_id=(
                self.configuration_id
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_symbol(self.symbol),
        )

        if self.frequency != SUPPORTED_FREQUENCY:
            raise TrendFamilyEventReplayError(
                "Replay result frequency must be exactly "
                "15min."
            )

        start = _normalize_timestamp(
            self.evaluation_start,
            name="evaluation_start",
        )
        end = _normalize_timestamp(
            self.evaluation_end_exclusive,
            name="evaluation_end_exclusive",
        )

        if start >= end:
            raise TrendFamilyEventReplayError(
                "evaluation_start must be strictly before "
                "evaluation_end_exclusive."
            )

        object.__setattr__(
            self,
            "evaluation_start",
            start,
        )
        object.__setattr__(
            self,
            "evaluation_end_exclusive",
            end,
        )

        if not isinstance(self.events, tuple):
            raise TrendFamilyEventReplayError(
                "events must be an immutable tuple."
            )

        event_types = (
            MarketBarEvent,
            FillEvent,
            PortfolioSnapshot,
            SignalEvent,
            TargetPositionOrderEvent,
        )

        if (
            not self.events
            or not all(
                isinstance(event, event_types)
                for event in self.events
            )
        ):
            raise TrendFamilyEventReplayError(
                "events must contain validated replay "
                "event records."
            )

        if [
            event.event_sequence
            for event in self.events
        ] != list(range(len(self.events))):
            raise TrendFamilyEventReplayError(
                "Replay event_sequence values must be "
                "global, gap-free, and zero-based."
            )

        def event_timestamp(
            event: ReplayEvent,
        ) -> pd.Timestamp:
            if isinstance(
                event,
                TargetPositionOrderEvent,
            ):
                return event.submitted_timestamp

            return event.timestamp

        if not all(
            (
                start
                <= event_timestamp(event)
                < end
                and event.symbol == self.symbol
                and (
                    isinstance(
                        event,
                        MarketBarEvent,
                    )
                    or (
                        event.strategy
                        == self.strategy
                        and event.configuration_id
                        == self.configuration_id
                    )
                )
            )
            for event in self.events
        ):
            raise TrendFamilyEventReplayError(
                "Replay events must lie inside the "
                "evaluation interval and match result "
                "metadata."
            )

        if not isinstance(
            self.observations,
            pd.DataFrame,
        ):
            raise TypeError(
                "observations must be a pandas DataFrame."
            )

        if self.observations.empty:
            raise TrendFamilyEventReplayError(
                "Replay observations must not be empty."
            )

        if tuple(
            self.observations.columns
        ) != REPLAY_LEDGER_COLUMNS:
            raise TrendFamilyEventReplayError(
                "Replay observations do not match the "
                "frozen ledger schema."
            )

        observations = self.observations.copy(
            deep=True
        ).reset_index(drop=True)
        raw_timestamps = observations[
            "timestamp"
        ]

        if not raw_timestamps.map(
            lambda value: (
                isinstance(value, pd.Timestamp)
                and value.tzinfo is not None
            )
        ).all():
            raise TrendFamilyEventReplayError(
                "Observation timestamps must be "
                "timezone-aware pandas Timestamps."
            )

        observations["timestamp"] = (
            pd.to_datetime(
                raw_timestamps,
                utc=True,
                errors="raise",
            )
        )

        if (
            not observations[
                "timestamp"
            ].is_monotonic_increasing
            or observations[
                "timestamp"
            ].duplicated().any()
        ):
            raise TrendFamilyEventReplayError(
                "Replay observations must be strictly "
                "chronological."
            )

        inside = (
            observations[
                "timestamp"
            ].ge(start)
            & observations[
                "timestamp"
            ].lt(end)
        )

        if not inside.all():
            raise TrendFamilyEventReplayError(
                "Replay observations must lie entirely "
                "inside the evaluation interval."
            )

        expected_bar_index = pd.Series(
            range(len(observations)),
            dtype="int64",
        )

        if not observations[
            "bar_index"
        ].reset_index(drop=True).eq(
            expected_bar_index
        ).all():
            raise TrendFamilyEventReplayError(
                "Replay observation bar indexes must be "
                "zero-based and gap-free."
            )

        if (
            not observations[
                "strategy"
            ].eq(self.strategy).all()
            or not observations[
                "symbol"
            ].eq(self.symbol).all()
            or not observations[
                "configuration_id"
            ].eq(
                self.configuration_id
            ).all()
        ):
            raise TrendFamilyEventReplayError(
                "Replay observation metadata do not match "
                "the result contract."
            )

        snapshots = tuple(
            event
            for event in self.events
            if isinstance(
                event,
                PortfolioSnapshot,
            )
        )

        if len(snapshots) != len(
            observations
        ):
            raise TrendFamilyEventReplayError(
                "Portfolio snapshot count must match the "
                "observation count."
            )

        for required_type in (
            MarketBarEvent,
            SignalEvent,
        ):
            if sum(
                isinstance(
                    event,
                    required_type,
                )
                for event in self.events
            ) != len(observations):
                raise TrendFamilyEventReplayError(
                    "Each replay observation requires one "
                    "market, portfolio, and signal event."
                )

        for snapshot, row in zip(
            snapshots,
            observations.itertuples(
                index=False
            ),
            strict=True,
        ):
            if (
                snapshot.bar_index
                != row.bar_index
                or snapshot.timestamp
                != row.timestamp
                or not math.isclose(
                    snapshot.ending_equity,
                    float(
                        row.ending_equity
                    ),
                    rel_tol=(
                        ACCOUNTING_TOLERANCE
                    ),
                    abs_tol=(
                        ACCOUNTING_TOLERANCE
                    ),
                )
            ):
                raise TrendFamilyEventReplayError(
                    "Portfolio snapshots do not align with "
                    "replay observations."
                )

        if not isinstance(
            self.gross_performance,
            PerformanceMetrics,
        ) or not isinstance(
            self.net_performance,
            PerformanceMetrics,
        ):
            raise TypeError(
                "Replay performance fields must use "
                "PerformanceMetrics."
            )

        annualization_factor = (
            ANNUALIZATION_FACTORS[
                SUPPORTED_FREQUENCY
            ]
        )
        expected_gross = (
            calculate_performance_metrics(
                observations[
                    "gross_strategy_return"
                ],
                annualization_factor=(
                    annualization_factor
                ),
            )
        )
        expected_net = (
            calculate_performance_metrics(
                observations[
                    "net_strategy_return"
                ],
                annualization_factor=(
                    annualization_factor
                ),
            )
        )

        if not _performance_metrics_equal(
            self.gross_performance,
            expected_gross,
        ) or not _performance_metrics_equal(
            self.net_performance,
            expected_net,
        ):
            raise TrendFamilyEventReplayError(
                "Replay performance metrics do not match "
                "the observation return series."
            )

        net_wealth = build_wealth_index(
            observations[
                "net_strategy_return"
            ]
        )

        if not all(
            math.isclose(
                float(actual),
                float(expected),
                rel_tol=ACCOUNTING_TOLERANCE,
                abs_tol=ACCOUNTING_TOLERANCE,
            )
            for actual, expected in zip(
                observations[
                    "ending_equity"
                ],
                net_wealth,
                strict=True,
            )
        ):
            raise TrendFamilyEventReplayError(
                "Replay ending equity does not match the "
                "shared net wealth index."
            )

        object.__setattr__(
            self,
            "observations",
            observations,
        )

    def copy_observations(
        self,
    ) -> pd.DataFrame:
        """Return a deep mutable copy of replay observations."""

        return self.observations.copy(
            deep=True
        )


@dataclass(frozen=True, slots=True)
class _ReplayState:
    """Immutable normalized-notional replay state before one bar."""

    executed_position: int
    pending_target_position: int
    pending_signal_available: bool
    pending_order: (
        TargetPositionOrderEvent | None
    )
    equity: float

    def __post_init__(self) -> None:
        """Validate pending execution and portfolio state."""

        executed = _validate_position(
            self.executed_position,
            name="executed",
        )
        pending = _validate_position(
            self.pending_target_position,
            name="pending target",
        )
        available = _validate_boolean(
            self.pending_signal_available,
            name="pending_signal_available",
        )
        _validate_positive_real(
            self.equity,
            name="equity",
        )

        if not available and pending != 0:
            raise TrendFamilyEventReplayError(
                "An unavailable pending signal must have a "
                "neutral target position."
            )

        order = self.pending_order

        if order is not None and not isinstance(
            order,
            TargetPositionOrderEvent,
        ):
            raise TrendFamilyEventReplayError(
                "pending_order must be a target-position "
                "order or None."
            )

        target_changes = pending != executed

        if target_changes and order is None:
            raise TrendFamilyEventReplayError(
                "A changed pending target requires a pending "
                "order for the following observation."
            )

        if not target_changes and order is not None:
            raise TrendFamilyEventReplayError(
                "An unchanged pending target cannot have a "
                "pending order."
            )

        if order is not None:
            if (
                order.current_executed_position
                != executed
                or order.target_position
                != pending
                or order.execute_bar_index
                != order.submitted_bar_index + 1
            ):
                raise TrendFamilyEventReplayError(
                    "pending_order position or execution "
                    "timing does not agree with replay state."
                )


def _validate_replay_core_inputs(
    prepared_bars: pd.DataFrame,
    signal_observations: pd.DataFrame,
    *,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy and validate aligned replay bars and decisions."""

    if not isinstance(
        prepared_bars,
        pd.DataFrame,
    ):
        raise TypeError(
            "prepared_bars must be a pandas DataFrame."
        )

    if not isinstance(
        signal_observations,
        pd.DataFrame,
    ):
        raise TypeError(
            "signal_observations must be a pandas DataFrame."
        )

    if (
        prepared_bars.empty
        or signal_observations.empty
    ):
        raise TrendFamilyEventReplayError(
            "Replay inputs must not be empty."
        )

    if (
        not isinstance(strategy, str)
        or strategy not in CONFIGURATION_IDS
    ):
        raise TrendFamilyEventReplayError(
            "strategy must be trend_ratio or ema_macd."
        )

    required_bars = {
        "timestamp",
        "session_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "source",
        "feed",
        "close_to_close_simple_return",
        "bar_frequency",
    }
    missing_bars = sorted(
        required_bars.difference(
            prepared_bars.columns
        )
    )

    if missing_bars:
        raise TrendFamilyEventReplayError(
            "prepared_bars are missing required replay "
            f"columns: {missing_bars}."
        )

    missing_signals = sorted(
        set(
            FROZEN_SIGNAL_OBSERVATION_COLUMNS
        ).difference(
            signal_observations.columns
        )
    )

    if missing_signals:
        raise TrendFamilyEventReplayError(
            "signal_observations are missing required "
            f"columns: {missing_signals}."
        )

    if len(prepared_bars) != len(
        signal_observations
    ):
        raise TrendFamilyEventReplayError(
            "Prepared bars and signal observations must "
            "have the same row count."
        )

    prepared = prepared_bars.copy(
        deep=True
    ).reset_index(drop=True)
    signals = signal_observations.copy(
        deep=True
    ).reset_index(drop=True)

    try:
        prepared_timestamps = pd.to_datetime(
            prepared["timestamp"],
            utc=True,
            errors="raise",
        )
        signal_timestamps = pd.to_datetime(
            signals["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendFamilyEventReplayError(
            "Replay inputs contain malformed timestamps."
        ) from exc

    if (
        prepared_timestamps.isna().any()
        or signal_timestamps.isna().any()
    ):
        raise TrendFamilyEventReplayError(
            "Replay timestamps cannot be missing."
        )

    prepared["timestamp"] = (
        prepared_timestamps
    )
    signals["timestamp"] = signal_timestamps

    if not prepared[
        "timestamp"
    ].is_monotonic_increasing:
        raise TrendFamilyEventReplayError(
            "prepared_bars must be in chronological order."
        )

    if prepared.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise TrendFamilyEventReplayError(
            "prepared_bars contain duplicate timestamps."
        )

    local_dates = (
        prepared["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if (
        local_dates.dt.year.ge(2026).any()
        or local_dates.min()
        < DEVELOPMENT_START
        or local_dates.max()
        > DEVELOPMENT_END
    ):
        raise TrendFamilyEventReplayError(
            "Replay-core bars must remain within the "
            "2020-01-02 through 2025-12-31 development "
            "period and cannot contain 2026 observations."
        )

    if not prepared[
        "timestamp"
    ].equals(signals["timestamp"]):
        raise TrendFamilyEventReplayError(
            "Replay timestamps do not align."
        )

    if (
        not prepared["symbol"].eq(
            SUPPORTED_SYMBOL
        ).all()
        or not prepared[
            "bar_frequency"
        ].eq(
            SUPPORTED_FREQUENCY
        ).all()
    ):
        raise TrendFamilyEventReplayError(
            "prepared_bars must contain SPY 15min "
            "observations only."
        )

    if not signals["symbol"].eq(
        prepared["symbol"]
    ).all():
        raise TrendFamilyEventReplayError(
            "Replay symbols do not align."
        )

    raw_signal = signals["signal"]
    valid_signal_types = raw_signal.map(
        lambda value: (
            isinstance(value, Integral)
            and not isinstance(value, bool)
        )
    )
    numeric_signal = pd.to_numeric(
        raw_signal,
        errors="coerce",
    )

    if (
        not valid_signal_types.all()
        or numeric_signal.isna().any()
        or not numeric_signal.isin(
            (-1, 0, 1)
        ).all()
    ):
        raise TrendFamilyEventReplayError(
            "Replay signal target positions must belong "
            "to {-1, 0, 1}."
        )

    availability = signals[
        "signal_available"
    ]

    if (
        availability.isna().any()
        or not pd.api.types.is_bool_dtype(
            availability.dtype
        )
    ):
        raise TrendFamilyEventReplayError(
            "signal_available must contain boolean values."
        )

    unavailable_non_neutral = (
        ~availability.astype(bool)
        & numeric_signal.ne(0)
    )

    if unavailable_non_neutral.any():
        raise TrendFamilyEventReplayError(
            "Unavailable replay signals must have neutral "
            "target positions."
        )

    try:
        asset_return = pd.to_numeric(
            prepared[
                "close_to_close_simple_return"
            ],
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendFamilyEventReplayError(
            "Prepared asset returns must be numeric."
        ) from exc
    missing_return = asset_return.isna()

    if (
        missing_return.iloc[1:].any()
        or (
            ~missing_return
            & ~asset_return.map(
                lambda value: math.isfinite(
                    float(value)
                )
            )
        ).any()
    ):
        raise TrendFamilyEventReplayError(
            "Prepared asset returns must be finite except "
            "for the first observation."
        )

    prepared[
        "close_to_close_simple_return"
    ] = asset_return.astype("float64")
    signals["signal"] = numeric_signal.astype(
        "int8"
    )
    signals["signal_available"] = (
        availability.astype(bool)
    )

    return prepared, signals


def _run_event_replay_core(
    prepared_bars: pd.DataFrame,
    signal_observations: pd.DataFrame,
    *,
    strategy: str,
) -> tuple[
    tuple[ReplayEvent, ...],
    pd.DataFrame,
]:
    """Replay completed-bar targets through a one-bar-delay ledger.

    Events for each bar are dispatched as market, optional fill,
    portfolio, signal, and optional order. The terminal signal is
    retained, but its unexecutable order is deliberately omitted.
    """

    prepared, signals = (
        _validate_replay_core_inputs(
            prepared_bars,
            signal_observations,
            strategy=strategy,
        )
    )
    configuration_id = CONFIGURATION_IDS[
        strategy
    ]
    parameters = (
        TREND_RATIO_PARAMETERS
        if strategy == "trend_ratio"
        else EMA_MACD_PARAMETERS
    )
    cost_bps = float(
        parameters.cost_bps_per_turnover
    )
    events: list[ReplayEvent] = []
    records: list[dict[str, object]] = []
    state = _ReplayState(
        executed_position=0,
        pending_target_position=0,
        pending_signal_available=False,
        pending_order=None,
        equity=1.0,
    )
    final_bar_index = len(prepared) - 1

    for bar_index in range(len(prepared)):
        bar = prepared.iloc[bar_index]
        signal = signals.iloc[bar_index]
        timestamp = pd.Timestamp(
            bar["timestamp"]
        )
        raw_asset_return = bar[
            "close_to_close_simple_return"
        ]
        asset_return = (
            0.0
            if pd.isna(raw_asset_return)
            else float(raw_asset_return)
        )

        market_event = MarketBarEvent(
            event_sequence=len(events),
            bar_index=bar_index,
            timestamp=timestamp,
            session_date=str(
                bar["session_date"]
            ),
            symbol=str(bar["symbol"]),
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar["volume"]),
            trade_count=int(
                bar["trade_count"]
            ),
            vwap=float(bar["vwap"]),
            source=str(bar["source"]),
            feed=str(bar["feed"]),
            asset_return=asset_return,
        )
        events.append(market_event)

        previous_position = (
            state.executed_position
        )
        executed_position = (
            state.pending_target_position
        )
        position_eligible = (
            state.pending_signal_available
        )
        position_change = (
            executed_position
            - previous_position
        )
        turnover = float(
            abs(position_change)
        )
        transaction_cost = (
            turnover
            * cost_bps
            / 10_000.0
        )
        fill_executed = position_change != 0

        if fill_executed:
            pending_order = (
                state.pending_order
            )

            if (
                pending_order is None
                or pending_order.execute_bar_index
                != bar_index
            ):
                raise TrendFamilyEventReplayError(
                    "Pending order does not execute on the "
                    "following replay observation."
                )

            events.append(
                FillEvent(
                    event_sequence=len(events),
                    bar_index=bar_index,
                    timestamp=timestamp,
                    symbol=str(bar["symbol"]),
                    strategy=strategy,
                    configuration_id=(
                        configuration_id
                    ),
                    submitted_bar_index=(
                        pending_order
                        .submitted_bar_index
                    ),
                    previous_position=(
                        previous_position
                    ),
                    executed_position=(
                        executed_position
                    ),
                    position_change=(
                        position_change
                    ),
                    turnover=turnover,
                    cost_bps_per_turnover=(
                        cost_bps
                    ),
                    transaction_cost=(
                        transaction_cost
                    ),
                )
            )
        elif state.pending_order is not None:
            raise TrendFamilyEventReplayError(
                "A pending order must produce one changed-"
                "position fill."
            )

        previous_equity = float(
            state.equity
        )
        gross_strategy_return = (
            float(executed_position)
            * asset_return
        )
        net_strategy_return = (
            gross_strategy_return
            - transaction_cost
        )
        transaction_cost_amount = (
            previous_equity
            * transaction_cost
        )
        gross_ending_equity = (
            previous_equity
            * (
                1.0
                + gross_strategy_return
            )
        )
        cash_balance = (
            previous_equity
            * (
                1.0
                - executed_position
            )
            - transaction_cost_amount
        )
        holdings_value = (
            float(executed_position)
            * previous_equity
            * (1.0 + asset_return)
        )
        ending_equity = (
            previous_equity
            * (
                1.0
                + net_strategy_return
            )
        )

        accounting_values = (
            transaction_cost,
            transaction_cost_amount,
            gross_strategy_return,
            net_strategy_return,
            gross_ending_equity,
            cash_balance,
            holdings_value,
            ending_equity,
        )

        if not all(
            math.isfinite(value)
            for value in accounting_values
        ):
            raise TrendFamilyEventReplayError(
                "Replay accounting produced a non-finite "
                "value."
            )

        if ending_equity <= 0.0:
            raise TrendFamilyEventReplayError(
                "Replay ending equity must remain strictly "
                "positive."
            )

        snapshot = PortfolioSnapshot(
            event_sequence=len(events),
            bar_index=bar_index,
            timestamp=timestamp,
            symbol=str(bar["symbol"]),
            strategy=strategy,
            configuration_id=(
                configuration_id
            ),
            position_eligible=(
                position_eligible
            ),
            previous_position=(
                previous_position
            ),
            executed_position=(
                executed_position
            ),
            position_change=position_change,
            turnover=turnover,
            asset_return=asset_return,
            gross_strategy_return=(
                gross_strategy_return
            ),
            transaction_cost=(
                transaction_cost
            ),
            transaction_cost_amount=(
                transaction_cost_amount
            ),
            net_strategy_return=(
                net_strategy_return
            ),
            previous_equity=previous_equity,
            gross_ending_equity=(
                gross_ending_equity
            ),
            cash_balance=cash_balance,
            holdings_value=holdings_value,
            ending_equity=ending_equity,
        )
        events.append(snapshot)

        target_position = int(
            signal["signal"]
        )
        signal_available = bool(
            signal["signal_available"]
        )
        signal_event = SignalEvent(
            event_sequence=len(events),
            bar_index=bar_index,
            timestamp=timestamp,
            symbol=str(bar["symbol"]),
            strategy=strategy,
            configuration_id=(
                configuration_id
            ),
            target_position=target_position,
            signal_available=(
                signal_available
            ),
        )
        events.append(signal_event)

        order: (
            TargetPositionOrderEvent | None
        ) = None

        if (
            bar_index < final_bar_index
            and target_position
            != executed_position
        ):
            order = TargetPositionOrderEvent(
                event_sequence=len(events),
                submitted_bar_index=(
                    bar_index
                ),
                execute_bar_index=(
                    bar_index + 1
                ),
                submitted_timestamp=timestamp,
                symbol=str(bar["symbol"]),
                strategy=strategy,
                configuration_id=(
                    configuration_id
                ),
                current_executed_position=(
                    executed_position
                ),
                target_position=(
                    target_position
                ),
            )
            events.append(order)

        order_submitted = order is not None
        records.append(
            {
                "bar_index": bar_index,
                "timestamp": timestamp,
                "session_date": str(
                    bar["session_date"]
                ),
                "symbol": str(bar["symbol"]),
                "strategy": strategy,
                "configuration_id": (
                    configuration_id
                ),
                "close": float(bar["close"]),
                "asset_return": asset_return,
                "signal_available": (
                    signal_available
                ),
                "target_position": (
                    target_position
                ),
                "position_eligible": (
                    position_eligible
                ),
                "previous_executed_position": (
                    previous_position
                ),
                "executed_position": (
                    executed_position
                ),
                "position_change": (
                    position_change
                ),
                "turnover": turnover,
                "cost_bps_per_turnover": (
                    cost_bps
                ),
                "transaction_cost": (
                    transaction_cost
                ),
                "transaction_cost_amount": (
                    transaction_cost_amount
                ),
                "gross_strategy_return": (
                    gross_strategy_return
                ),
                "net_strategy_return": (
                    net_strategy_return
                ),
                "previous_equity": (
                    previous_equity
                ),
                "gross_ending_equity": (
                    gross_ending_equity
                ),
                "cash_balance": cash_balance,
                "holdings_value": (
                    holdings_value
                ),
                "ending_equity": (
                    ending_equity
                ),
                "order_submitted": (
                    order_submitted
                ),
                "fill_executed": (
                    fill_executed
                ),
            }
        )

        if bar_index < final_bar_index:
            state = _ReplayState(
                executed_position=(
                    executed_position
                ),
                pending_target_position=(
                    target_position
                ),
                pending_signal_available=(
                    signal_available
                ),
                pending_order=order,
                equity=ending_equity,
            )

    ledger = pd.DataFrame.from_records(
        records,
        columns=REPLAY_LEDGER_COLUMNS,
    )
    position_columns = (
        "target_position",
        "previous_executed_position",
        "executed_position",
        "position_change",
    )
    boolean_columns = (
        "signal_available",
        "position_eligible",
        "order_submitted",
        "fill_executed",
    )

    ledger["bar_index"] = ledger[
        "bar_index"
    ].astype("int64")

    for column in position_columns:
        ledger[column] = ledger[
            column
        ].astype("int8")

    for column in boolean_columns:
        ledger[column] = ledger[
            column
        ].astype(bool)

    ledger["timestamp"] = pd.to_datetime(
        ledger["timestamp"],
        utc=True,
        errors="raise",
    )

    return (
        tuple(events),
        ledger.reset_index(drop=True),
    )


def _normalize_evaluation_boundary(
    value: object,
    *,
    name: str,
) -> pd.Timestamp:
    """Require one timezone-aware boundary and normalize it to UTC."""

    if (
        not isinstance(value, pd.Timestamp)
        or value.tzinfo is None
    ):
        raise TrendFamilyEventReplayError(
            f"{name} must be a timezone-aware pandas "
            "Timestamp."
        )

    return value.tz_convert("UTC")


def _resolve_evaluation_window(
    prepared: pd.DataFrame,
    *,
    evaluation_start: (
        pd.Timestamp | None
    ),
    evaluation_end_exclusive: (
        pd.Timestamp | None
    ),
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
    pd.Series,
]:
    """Resolve one half-open interval using complete session edges."""

    session_groups = prepared.groupby(
        "session_date",
        observed=True,
        sort=False,
    )
    session_starts = tuple(
        pd.Timestamp(
            group["timestamp"].iloc[0]
        )
        for _, group in session_groups
    )

    if not session_starts:
        raise TrendFamilyEventReplayError(
            "Prepared replay data contain no sessions."
        )

    first_timestamp = pd.Timestamp(
        prepared["timestamp"].iloc[0]
    )
    last_timestamp = pd.Timestamp(
        prepared["timestamp"].iloc[-1]
    )
    terminal_boundary = (
        last_timestamp
        + pd.Timedelta(minutes=15)
    )
    start = (
        first_timestamp
        if evaluation_start is None
        else _normalize_evaluation_boundary(
            evaluation_start,
            name="evaluation_start",
        )
    )
    end = (
        terminal_boundary
        if evaluation_end_exclusive is None
        else _normalize_evaluation_boundary(
            evaluation_end_exclusive,
            name=(
                "evaluation_end_exclusive"
            ),
        )
    )

    if start >= end:
        raise TrendFamilyEventReplayError(
            "evaluation_start must precede "
            "evaluation_end_exclusive and the evaluation "
            "window cannot be empty."
        )

    if (
        start < first_timestamp
        or start > last_timestamp
        or end <= first_timestamp
        or end > terminal_boundary
    ):
        raise TrendFamilyEventReplayError(
            "Evaluation boundaries cannot extend outside "
            "the prepared replay data."
        )

    valid_starts = frozenset(
        session_starts
    )
    valid_ends = frozenset(
        (
            *session_starts,
            terminal_boundary,
        )
    )

    if start not in valid_starts:
        raise TrendFamilyEventReplayError(
            "evaluation_start must equal the first bar of "
            "a complete trading session."
        )

    if end not in valid_ends:
        raise TrendFamilyEventReplayError(
            "evaluation_end_exclusive must equal the next "
            "session start or the first 15-minute boundary "
            "after the final session."
        )

    mask = (
        prepared["timestamp"].ge(start)
        & prepared["timestamp"].lt(end)
    )

    if not mask.any():
        raise TrendFamilyEventReplayError(
            "The evaluation window cannot be empty."
        )

    selected = prepared.loc[
        mask
    ]
    selected_counts = selected.groupby(
        "session_date",
        observed=True,
        sort=False,
    ).size()
    complete_counts = prepared.groupby(
        "session_date",
        observed=True,
        sort=False,
    ).size()

    if not selected_counts.eq(
        complete_counts.loc[
            selected_counts.index
        ]
    ).all():
        raise TrendFamilyEventReplayError(
            "Evaluation boundaries cannot split a trading "
            "session."
        )

    return start, end, mask


def run_trend_family_event_replay(
    bars: pd.DataFrame,
    *,
    strategy: str,
    frequency: str = SUPPORTED_FREQUENCY,
    evaluation_start: (
        pd.Timestamp | None
    ) = None,
    evaluation_end_exclusive: (
        pd.Timestamp | None
    ) = None,
) -> TrendFamilyEventReplayResult:
    """Run one frozen SPY strategy over a reset evaluation window."""

    if frequency != SUPPORTED_FREQUENCY:
        raise TrendFamilyEventReplayError(
            "Day 12 public replay frequency must be "
            "exactly 15min."
        )

    if (
        not isinstance(strategy, str)
        or strategy not in CONFIGURATION_IDS
    ):
        raise TrendFamilyEventReplayError(
            "strategy must be trend_ratio or ema_macd."
        )

    prepared = _prepare_replay_bars(
        bars,
        frequency=frequency,
    )
    signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )
    start, end, mask = (
        _resolve_evaluation_window(
            prepared,
            evaluation_start=(
                evaluation_start
            ),
            evaluation_end_exclusive=(
                evaluation_end_exclusive
            ),
        )
    )
    evaluation_bars = (
        prepared.loc[mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    evaluation_signals = (
        signals.loc[mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    events, observations = (
        _run_event_replay_core(
            evaluation_bars,
            evaluation_signals,
            strategy=strategy,
        )
    )
    annualization_factor = (
        ANNUALIZATION_FACTORS[
            SUPPORTED_FREQUENCY
        ]
    )
    gross_performance = (
        calculate_performance_metrics(
            observations[
                "gross_strategy_return"
            ],
            annualization_factor=(
                annualization_factor
            ),
        )
    )
    net_performance = (
        calculate_performance_metrics(
            observations[
                "net_strategy_return"
            ],
            annualization_factor=(
                annualization_factor
            ),
        )
    )

    return TrendFamilyEventReplayResult(
        strategy=strategy,
        symbol=SUPPORTED_SYMBOL,
        frequency=frequency,
        configuration_id=(
            CONFIGURATION_IDS[strategy]
        ),
        evaluation_start=start,
        evaluation_end_exclusive=end,
        events=events,
        observations=observations,
        gross_performance=(
            gross_performance
        ),
        net_performance=net_performance,
    )
