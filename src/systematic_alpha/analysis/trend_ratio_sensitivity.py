"""Deterministic foundations for Day 7 trend-ratio sensitivity analysis.

This module defines the frozen parameter grid, descriptive moving-average lag
diagnostics, and the axis-adjacent parameter-neighborhood relationship.

It contains no market-data loading, strategy execution, optimization, or
artifact-writing logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
import math
from numbers import Real
from typing import Final

import pandas as pd
from scipy.optimize import brentq

from systematic_alpha.analysis.dependence_diagnostics import (
    build_volatility_regimes,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.strategies.trend_ratio import (
    TrendRatioBundle,
    TrendRatioParameters,
    build_trend_ratio_strategy,
)


SHORT_WINDOWS: Final[tuple[int, ...]] = (4, 8, 16)
LONG_WINDOWS: Final[tuple[int, ...]] = (32, 64, 96)
NEUTRAL_BANDS: Final[tuple[float, ...]] = (
    0.0,
    0.0005,
    0.0010,
    0.0020,
)

BASELINE_SHORT_WINDOW: Final[int] = 8
BASELINE_LONG_WINDOW: Final[int] = 32
BASELINE_NEUTRAL_BAND: Final[float] = 0.0010

BASELINE_COST_BPS_PER_TURNOVER: Final[float] = 1.0
FORWARD_RETURN_HORIZONS: Final[tuple[int, ...]] = (1, 4, 8, 16)

WHIPSAW_MAX_EPISODE_BARS: Final[int] = 4
WHIPSAW_MAX_FOLLOWING_BARS: Final[int] = 4

BREAK_EVEN_COST_LOWER_BPS: Final[float] = 0.0
BREAK_EVEN_COST_UPPER_BPS: Final[float] = 100.0
DAY07_ANNUALIZATION_FACTOR: Final[int] = 252 * 26

BREAK_EVEN_STATUS_ROOT_FOUND: Final[str] = "root_found"
BREAK_EVEN_STATUS_NON_POSITIVE_GROSS: Final[str] = "non_positive_gross"
BREAK_EVEN_STATUS_ZERO_TURNOVER: Final[str] = "zero_turnover"
BREAK_EVEN_STATUS_ROOT_ABOVE_INTERVAL: Final[str] = (
    "root_above_search_interval"
)
BREAK_EVEN_STATUS_INVALID_WEALTH_AT_BOUND: Final[str] = (
    "invalid_wealth_at_bound"
)


@dataclass(frozen=True, order=True, slots=True)
class TrendRatioConfiguration:
    """One immutable short/long-window and neutral-band configuration."""

    short_window: int
    long_window: int
    neutral_band: float

    def __post_init__(self) -> None:
        if isinstance(self.short_window, bool) or not isinstance(
            self.short_window,
            int,
        ):
            raise TypeError("short_window must be an integer.")

        if isinstance(self.long_window, bool) or not isinstance(
            self.long_window,
            int,
        ):
            raise TypeError("long_window must be an integer.")

        if self.short_window <= 0:
            raise ValueError("short_window must be positive.")

        if self.long_window <= 0:
            raise ValueError("long_window must be positive.")

        if self.short_window >= self.long_window:
            raise ValueError(
                "short_window must be strictly smaller than long_window."
            )

        if isinstance(self.neutral_band, bool) or not isinstance(
            self.neutral_band,
            (int, float),
        ):
            raise TypeError("neutral_band must be numeric.")

        neutral_band = float(self.neutral_band)

        if not math.isfinite(neutral_band):
            raise ValueError("neutral_band must be finite.")

        if neutral_band < 0.0:
            raise ValueError("neutral_band must be non-negative.")

        object.__setattr__(self, "neutral_band", neutral_band)

    @property
    def configuration_id(self) -> str:
        """Return a stable identifier suitable for compact artifacts."""

        band_token = f"{self.neutral_band:.4f}".replace(".", "p")
        return (
            f"s{self.short_window:03d}"
            f"_l{self.long_window:03d}"
            f"_d{band_token}"
        )


@dataclass(frozen=True, slots=True)
class FilterLagDiagnostics:
    """Approximate simple-moving-average lag diagnostics in bars."""

    short_filter_lag_bars: float
    long_filter_lag_bars: float
    lag_spread_bars: float


def _validate_strictly_increasing(
    values: Sequence[int | float],
    *,
    name: str,
) -> tuple[int | float, ...]:
    """Validate an ordered grid axis without changing its declared order."""

    normalized = tuple(values)

    if not normalized:
        raise ValueError(f"{name} must not be empty.")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")

    if any(
        current >= following
        for current, following in zip(normalized, normalized[1:])
    ):
        raise ValueError(f"{name} must be strictly increasing.")

    return normalized


def build_parameter_grid(
    *,
    short_windows: Sequence[int] = SHORT_WINDOWS,
    long_windows: Sequence[int] = LONG_WINDOWS,
    neutral_bands: Sequence[float] = NEUTRAL_BANDS,
) -> tuple[TrendRatioConfiguration, ...]:
    """Build the deterministic Cartesian Day 7 parameter grid.

    Every declared short window must be smaller than every declared long
    window. Invalid declared axes raise rather than being silently filtered.
    """

    validated_short_windows = tuple(
        int(value)
        for value in _validate_strictly_increasing(
            short_windows,
            name="short_windows",
        )
    )
    validated_long_windows = tuple(
        int(value)
        for value in _validate_strictly_increasing(
            long_windows,
            name="long_windows",
        )
    )
    validated_neutral_bands = tuple(
        float(value)
        for value in _validate_strictly_increasing(
            neutral_bands,
            name="neutral_bands",
        )
    )

    if validated_short_windows[0] <= 0:
        raise ValueError("short_windows must contain only positive values.")

    if validated_long_windows[0] <= 0:
        raise ValueError("long_windows must contain only positive values.")

    if validated_neutral_bands[0] < 0.0:
        raise ValueError(
            "neutral_bands must contain only non-negative values."
        )

    if max(validated_short_windows) >= min(validated_long_windows):
        raise ValueError(
            "Every declared short window must be smaller than every "
            "declared long window."
        )

    configurations = tuple(
        TrendRatioConfiguration(
            short_window=short_window,
            long_window=long_window,
            neutral_band=neutral_band,
        )
        for short_window, long_window, neutral_band in product(
            validated_short_windows,
            validated_long_windows,
            validated_neutral_bands,
        )
    )

    configuration_ids = tuple(
        configuration.configuration_id
        for configuration in configurations
    )

    if len(set(configuration_ids)) != len(configuration_ids):
        raise RuntimeError(
            "Parameter-grid construction produced duplicate configurations."
        )

    return configurations


def calculate_filter_lag(
    configuration: TrendRatioConfiguration,
) -> FilterLagDiagnostics:
    """Calculate descriptive SMA lag using L_N = (N - 1) / 2."""

    short_lag = (configuration.short_window - 1) / 2.0
    long_lag = (configuration.long_window - 1) / 2.0

    return FilterLagDiagnostics(
        short_filter_lag_bars=short_lag,
        long_filter_lag_bars=long_lag,
        lag_spread_bars=long_lag - short_lag,
    )


def configurations_are_neighbors(
    left: TrendRatioConfiguration,
    right: TrendRatioConfiguration,
    *,
    short_windows: Sequence[int] = SHORT_WINDOWS,
    long_windows: Sequence[int] = LONG_WINDOWS,
    neutral_bands: Sequence[float] = NEUTRAL_BANDS,
) -> bool:
    """Return whether two configurations are one declared grid step apart.

    Neighbors differ along exactly one parameter axis by exactly one position
    in that axis. The other two parameters must be identical.
    """

    validated_short_windows = tuple(short_windows)
    validated_long_windows = tuple(long_windows)
    validated_neutral_bands = tuple(float(value) for value in neutral_bands)

    short_index = {
        value: index
        for index, value in enumerate(validated_short_windows)
    }
    long_index = {
        value: index
        for index, value in enumerate(validated_long_windows)
    }
    band_index = {
        value: index
        for index, value in enumerate(validated_neutral_bands)
    }

    try:
        left_coordinates = (
            short_index[left.short_window],
            long_index[left.long_window],
            band_index[left.neutral_band],
        )
        right_coordinates = (
            short_index[right.short_window],
            long_index[right.long_window],
            band_index[right.neutral_band],
        )
    except KeyError as exc:
        raise ValueError(
            "Both configurations must belong to the declared parameter grid."
        ) from exc

    coordinate_distances = tuple(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            left_coordinates,
            right_coordinates,
        )
    )

    return sum(coordinate_distances) == 1


def build_neighborhood_map(
    configurations: Sequence[TrendRatioConfiguration] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build a deterministic symmetric neighborhood map."""

    parameter_grid = (
        build_parameter_grid()
        if configurations is None
        else tuple(configurations)
    )

    configuration_ids = tuple(
        configuration.configuration_id
        for configuration in parameter_grid
    )

    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("configurations must not contain duplicates.")

    neighborhoods: dict[str, tuple[str, ...]] = {}

    for configuration in parameter_grid:
        neighbor_ids = sorted(
            candidate.configuration_id
            for candidate in parameter_grid
            if candidate != configuration
            and configurations_are_neighbors(configuration, candidate)
        )

        neighborhoods[configuration.configuration_id] = tuple(neighbor_ids)

    return neighborhoods


@dataclass(frozen=True, slots=True)
class HoldingEpisode:
    """One maximal consecutive run of an identical non-zero position."""

    episode_id: int
    position: int
    start_position: int
    end_position: int
    start_timestamp: object
    end_timestamp: object
    duration_bars: int
    start_session: object | None
    end_session: object | None
    crosses_session_boundary: bool


@dataclass(frozen=True, slots=True)
class WhipsawClassification:
    """Whipsaw classification for one non-zero holding episode."""

    episode_id: int
    is_whipsaw: bool
    following_episode_id: int | None
    following_start_gap_bars: int | None


@dataclass(frozen=True, slots=True)
class HoldingDiagnostics:
    """Aggregate holding-period and whipsaw diagnostics."""

    eligible_observations: int
    non_zero_episode_count: int
    long_episode_count: int
    short_episode_count: int
    median_holding_duration_bars: float
    mean_holding_duration_bars: float
    holding_duration_25th_percentile_bars: float
    holding_duration_75th_percentile_bars: float
    maximum_holding_duration_bars: float
    overnight_carry_episode_count: int
    session_crossing_episode_proportion: float
    whipsaw_count: int
    whipsaw_rate_per_1000_eligible_observations: float
    whipsaw_episode_proportion: float


def _validate_position_series(positions: pd.Series) -> pd.Series:
    """Return a numeric copy containing only -1, 0, 1, or missing values."""

    if not isinstance(positions, pd.Series):
        raise TypeError("positions must be a pandas Series.")

    if pd.api.types.is_bool_dtype(positions.dtype):
        raise TypeError("positions must not use a Boolean dtype.")

    try:
        numeric_positions = pd.to_numeric(
            positions.copy(deep=True),
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("positions must contain numeric values.") from exc

    non_missing_positions = numeric_positions.dropna()

    if not non_missing_positions.isin((-1, 0, 1)).all():
        raise ValueError(
            "positions may contain only -1, 0, 1, or missing values."
        )

    return numeric_positions.astype(float)


def _validate_session_labels(
    positions: pd.Series,
    session_labels: pd.Series | None,
) -> pd.Series | None:
    """Validate optional session labels without modifying the caller's data."""

    if session_labels is None:
        return None

    if not isinstance(session_labels, pd.Series):
        raise TypeError("session_labels must be a pandas Series.")

    if not session_labels.index.equals(positions.index):
        raise ValueError(
            "session_labels must have the same index as positions."
        )

    validated_labels = session_labels.copy(deep=True)
    invested_mask = positions.notna() & positions.ne(0.0)

    if validated_labels.loc[invested_mask].isna().any():
        raise ValueError(
            "session_labels must be present for every non-zero position."
        )

    return validated_labels


def _build_episode(
    *,
    episode_id: int,
    position: int,
    start_position: int,
    end_position: int,
    positions: pd.Series,
    session_labels: pd.Series | None,
) -> HoldingEpisode:
    """Construct one immutable episode from positional boundaries."""

    start_session: object | None = None
    end_session: object | None = None
    crosses_session_boundary = False

    if session_labels is not None:
        episode_sessions = session_labels.iloc[
            start_position : end_position + 1
        ]
        unique_sessions = tuple(pd.unique(episode_sessions))

        start_session = episode_sessions.iloc[0]
        end_session = episode_sessions.iloc[-1]
        crosses_session_boundary = len(unique_sessions) > 1

    return HoldingEpisode(
        episode_id=episode_id,
        position=position,
        start_position=start_position,
        end_position=end_position,
        start_timestamp=positions.index[start_position],
        end_timestamp=positions.index[end_position],
        duration_bars=end_position - start_position + 1,
        start_session=start_session,
        end_session=end_session,
        crosses_session_boundary=crosses_session_boundary,
    )


def identify_holding_episodes(
    positions: pd.Series,
    *,
    session_labels: pd.Series | None = None,
) -> tuple[HoldingEpisode, ...]:
    """Identify maximal consecutive runs of identical non-zero positions.

    Neutral and missing observations are not holding episodes and terminate any
    active episode. Episodes are built over the complete supplied sequence and
    therefore do not reset at calendar-year or session boundaries.
    """

    validated_positions = _validate_position_series(positions)
    validated_sessions = _validate_session_labels(
        validated_positions,
        session_labels,
    )

    episodes: list[HoldingEpisode] = []
    active_position = 0
    active_start_position: int | None = None

    for observation_position, value in enumerate(
        validated_positions.to_numpy()
    ):
        normalized_position = (
            0
            if pd.isna(value)
            else int(value)
        )

        if normalized_position == 0:
            if active_start_position is not None:
                episodes.append(
                    _build_episode(
                        episode_id=len(episodes) + 1,
                        position=active_position,
                        start_position=active_start_position,
                        end_position=observation_position - 1,
                        positions=validated_positions,
                        session_labels=validated_sessions,
                    )
                )
                active_position = 0
                active_start_position = None

            continue

        if active_start_position is None:
            active_position = normalized_position
            active_start_position = observation_position
            continue

        if normalized_position != active_position:
            episodes.append(
                _build_episode(
                    episode_id=len(episodes) + 1,
                    position=active_position,
                    start_position=active_start_position,
                    end_position=observation_position - 1,
                    positions=validated_positions,
                    session_labels=validated_sessions,
                )
            )
            active_position = normalized_position
            active_start_position = observation_position

    if active_start_position is not None:
        episodes.append(
            _build_episode(
                episode_id=len(episodes) + 1,
                position=active_position,
                start_position=active_start_position,
                end_position=len(validated_positions) - 1,
                positions=validated_positions,
                session_labels=validated_sessions,
            )
        )

    return tuple(episodes)


def classify_whipsaw_episodes(
    episodes: Sequence[HoldingEpisode],
    *,
    max_episode_bars: int = WHIPSAW_MAX_EPISODE_BARS,
    max_following_bars: int = WHIPSAW_MAX_FOLLOWING_BARS,
) -> tuple[WhipsawClassification, ...]:
    """Classify episodes using the frozen Day 7 whipsaw definition.

    An episode is a whipsaw when it lasts no more than ``max_episode_bars`` and
    the next non-zero episode is opposite in sign and starts between one and
    ``max_following_bars`` positional bars after the current episode ends.
    """

    if isinstance(max_episode_bars, bool) or not isinstance(
        max_episode_bars,
        int,
    ):
        raise TypeError("max_episode_bars must be an integer.")

    if isinstance(max_following_bars, bool) or not isinstance(
        max_following_bars,
        int,
    ):
        raise TypeError("max_following_bars must be an integer.")

    if max_episode_bars <= 0:
        raise ValueError("max_episode_bars must be positive.")

    if max_following_bars <= 0:
        raise ValueError("max_following_bars must be positive.")

    episode_sequence = tuple(episodes)

    if any(
        current.episode_id >= following.episode_id
        or current.start_position > current.end_position
        or current.end_position >= following.start_position
        for current, following in zip(
            episode_sequence,
            episode_sequence[1:],
        )
    ):
        raise ValueError(
            "episodes must be strictly ordered, non-overlapping episodes."
        )

    classifications: list[WhipsawClassification] = []

    for episode_index, episode in enumerate(episode_sequence):
        following_episode = (
            episode_sequence[episode_index + 1]
            if episode_index + 1 < len(episode_sequence)
            else None
        )

        following_gap: int | None = None
        is_whipsaw = False

        if following_episode is not None:
            following_gap = (
                following_episode.start_position - episode.end_position
            )
            is_whipsaw = (
                episode.duration_bars <= max_episode_bars
                and following_episode.position == -episode.position
                and 1 <= following_gap <= max_following_bars
            )

        classifications.append(
            WhipsawClassification(
                episode_id=episode.episode_id,
                is_whipsaw=is_whipsaw,
                following_episode_id=(
                    None
                    if following_episode is None
                    else following_episode.episode_id
                ),
                following_start_gap_bars=following_gap,
            )
        )

    return tuple(classifications)


def calculate_holding_diagnostics(
    positions: pd.Series,
    *,
    session_labels: pd.Series | None = None,
) -> HoldingDiagnostics:
    """Calculate aggregate holding-period and whipsaw diagnostics.

    Whipsaw rates use all non-missing position observations as the eligible
    observation denominator.
    """

    validated_positions = _validate_position_series(positions)
    episodes = identify_holding_episodes(
        validated_positions,
        session_labels=session_labels,
    )
    whipsaw_classifications = classify_whipsaw_episodes(episodes)

    eligible_observations = int(validated_positions.notna().sum())
    durations = pd.Series(
        [episode.duration_bars for episode in episodes],
        dtype=float,
    )

    if durations.empty:
        median_duration = float("nan")
        mean_duration = float("nan")
        duration_25th_percentile = float("nan")
        duration_75th_percentile = float("nan")
        maximum_duration = float("nan")
    else:
        median_duration = float(durations.median())
        mean_duration = float(durations.mean())
        duration_25th_percentile = float(durations.quantile(0.25))
        duration_75th_percentile = float(durations.quantile(0.75))
        maximum_duration = float(durations.max())

    long_episode_count = sum(
        episode.position == 1
        for episode in episodes
    )
    short_episode_count = sum(
        episode.position == -1
        for episode in episodes
    )
    session_crossing_count = sum(
        episode.crosses_session_boundary
        for episode in episodes
    )
    whipsaw_count = sum(
        classification.is_whipsaw
        for classification in whipsaw_classifications
    )

    episode_count = len(episodes)

    session_crossing_proportion = (
        float(session_crossing_count / episode_count)
        if episode_count
        else float("nan")
    )
    whipsaw_episode_proportion = (
        float(whipsaw_count / episode_count)
        if episode_count
        else float("nan")
    )
    whipsaw_rate = (
        float(whipsaw_count * 1000.0 / eligible_observations)
        if eligible_observations
        else float("nan")
    )

    return HoldingDiagnostics(
        eligible_observations=eligible_observations,
        non_zero_episode_count=episode_count,
        long_episode_count=long_episode_count,
        short_episode_count=short_episode_count,
        median_holding_duration_bars=median_duration,
        mean_holding_duration_bars=mean_duration,
        holding_duration_25th_percentile_bars=duration_25th_percentile,
        holding_duration_75th_percentile_bars=duration_75th_percentile,
        maximum_holding_duration_bars=maximum_duration,
        overnight_carry_episode_count=session_crossing_count,
        session_crossing_episode_proportion=(
            session_crossing_proportion
        ),
        whipsaw_count=whipsaw_count,
        whipsaw_rate_per_1000_eligible_observations=whipsaw_rate,
        whipsaw_episode_proportion=whipsaw_episode_proportion,
    )


@dataclass(frozen=True, slots=True)
class CostBreakEvenResult:
    """Compounded transaction-cost break-even calculation result."""

    status: str
    break_even_cost_bps: float | None
    eligible_observations: int
    gross_cumulative_return: float
    total_turnover: float
    search_lower_bps: float
    search_upper_bps: float
    effective_upper_bps: float | None
    objective_at_lower: float
    objective_at_effective_upper: float | None


def _validate_break_even_inputs(
    gross_returns: pd.Series,
    turnover: pd.Series,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate and copy aligned gross-return and turnover observations."""

    if not isinstance(gross_returns, pd.Series):
        raise TypeError("gross_returns must be a pandas Series.")

    if not isinstance(turnover, pd.Series):
        raise TypeError("turnover must be a pandas Series.")

    if not gross_returns.index.equals(turnover.index):
        raise ValueError(
            "gross_returns and turnover must have identical indexes."
        )

    if pd.api.types.is_bool_dtype(gross_returns.dtype):
        raise TypeError("gross_returns must not use a Boolean dtype.")

    if pd.api.types.is_bool_dtype(turnover.dtype):
        raise TypeError("turnover must not use a Boolean dtype.")

    try:
        numeric_gross_returns = pd.to_numeric(
            gross_returns.copy(deep=True),
            errors="raise",
        ).astype(float)
        numeric_turnover = pd.to_numeric(
            turnover.copy(deep=True),
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "gross_returns and turnover must contain numeric values."
        ) from exc

    paired = pd.concat(
        {
            "gross_return": numeric_gross_returns,
            "turnover": numeric_turnover,
        },
        axis=1,
    ).dropna(how="any")

    if paired.empty:
        raise ValueError(
            "No paired non-missing gross-return and turnover observations."
        )

    gross_values = tuple(
        float(value)
        for value in paired["gross_return"].to_numpy()
    )
    turnover_values = tuple(
        float(value)
        for value in paired["turnover"].to_numpy()
    )

    if not all(math.isfinite(value) for value in gross_values):
        raise ValueError("gross_returns must contain only finite values.")

    if not all(math.isfinite(value) for value in turnover_values):
        raise ValueError("turnover must contain only finite values.")

    if any(value < 0.0 for value in turnover_values):
        raise ValueError("turnover must be non-negative.")

    if any(value <= -1.0 for value in gross_values):
        raise ValueError(
            "Every gross return must be greater than -1.0."
        )

    return gross_values, turnover_values


def _log_wealth_at_cost(
    *,
    gross_values: Sequence[float],
    turnover_values: Sequence[float],
    cost_bps: float,
) -> float | None:
    """Return compounded log wealth or None when wealth becomes invalid."""

    net_returns = tuple(
        gross_return
        - turnover_value * cost_bps / 10_000.0
        for gross_return, turnover_value in zip(
            gross_values,
            turnover_values,
        )
    )

    if any(net_return <= -1.0 for net_return in net_returns):
        return None

    return math.fsum(
        math.log1p(net_return)
        for net_return in net_returns
    )


def calculate_cost_break_even(
    gross_returns: pd.Series,
    turnover: pd.Series,
) -> CostBreakEvenResult:
    """Calculate compounded break-even cost per turnover unit.

    The frozen Day 7 search interval is 0 to 100 basis points per turnover
    unit. The objective is compounded log wealth:

        sum(log(1 + gross_return - turnover * cost_bps / 10_000))

    A root corresponds to compounded cumulative net return equal to zero.
    """

    gross_values, turnover_values = _validate_break_even_inputs(
        gross_returns,
        turnover,
    )

    search_lower_bps = BREAK_EVEN_COST_LOWER_BPS
    search_upper_bps = BREAK_EVEN_COST_UPPER_BPS

    gross_log_wealth = _log_wealth_at_cost(
        gross_values=gross_values,
        turnover_values=turnover_values,
        cost_bps=0.0,
    )

    if gross_log_wealth is None:
        raise RuntimeError(
            "Validated gross returns unexpectedly produced invalid wealth."
        )

    gross_cumulative_return = math.expm1(gross_log_wealth)
    total_turnover = math.fsum(turnover_values)

    if math.isclose(
        total_turnover,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return CostBreakEvenResult(
            status=BREAK_EVEN_STATUS_ZERO_TURNOVER,
            break_even_cost_bps=None,
            eligible_observations=len(gross_values),
            gross_cumulative_return=gross_cumulative_return,
            total_turnover=total_turnover,
            search_lower_bps=search_lower_bps,
            search_upper_bps=search_upper_bps,
            effective_upper_bps=None,
            objective_at_lower=gross_log_wealth,
            objective_at_effective_upper=None,
        )

    if gross_log_wealth <= 0.0:
        return CostBreakEvenResult(
            status=BREAK_EVEN_STATUS_NON_POSITIVE_GROSS,
            break_even_cost_bps=None,
            eligible_observations=len(gross_values),
            gross_cumulative_return=gross_cumulative_return,
            total_turnover=total_turnover,
            search_lower_bps=search_lower_bps,
            search_upper_bps=search_upper_bps,
            effective_upper_bps=None,
            objective_at_lower=gross_log_wealth,
            objective_at_effective_upper=None,
        )

    upper_objective = _log_wealth_at_cost(
        gross_values=gross_values,
        turnover_values=turnover_values,
        cost_bps=search_upper_bps,
    )

    effective_upper_bps = search_upper_bps
    upper_bound_was_invalid = upper_objective is None

    if upper_bound_was_invalid:
        feasible_cost_limits = tuple(
            10_000.0 * (1.0 + gross_return) / turnover_value
            for gross_return, turnover_value in zip(
                gross_values,
                turnover_values,
            )
            if turnover_value > 0.0
        )

        if not feasible_cost_limits:
            raise RuntimeError(
                "Positive total turnover produced no positive turnover rows."
            )

        wealth_failure_cost_bps = min(feasible_cost_limits)

        effective_upper_bps = math.nextafter(
            min(search_upper_bps, wealth_failure_cost_bps),
            search_lower_bps,
        )

        if effective_upper_bps <= search_lower_bps:
            return CostBreakEvenResult(
                status=BREAK_EVEN_STATUS_INVALID_WEALTH_AT_BOUND,
                break_even_cost_bps=None,
                eligible_observations=len(gross_values),
                gross_cumulative_return=gross_cumulative_return,
                total_turnover=total_turnover,
                search_lower_bps=search_lower_bps,
                search_upper_bps=search_upper_bps,
                effective_upper_bps=effective_upper_bps,
                objective_at_lower=gross_log_wealth,
                objective_at_effective_upper=None,
            )

        upper_objective = _log_wealth_at_cost(
            gross_values=gross_values,
            turnover_values=turnover_values,
            cost_bps=effective_upper_bps,
        )

    if upper_objective is None:
        return CostBreakEvenResult(
            status=BREAK_EVEN_STATUS_INVALID_WEALTH_AT_BOUND,
            break_even_cost_bps=None,
            eligible_observations=len(gross_values),
            gross_cumulative_return=gross_cumulative_return,
            total_turnover=total_turnover,
            search_lower_bps=search_lower_bps,
            search_upper_bps=search_upper_bps,
            effective_upper_bps=effective_upper_bps,
            objective_at_lower=gross_log_wealth,
            objective_at_effective_upper=None,
        )

    if upper_objective > 0.0:
        status = (
            BREAK_EVEN_STATUS_INVALID_WEALTH_AT_BOUND
            if upper_bound_was_invalid
            else BREAK_EVEN_STATUS_ROOT_ABOVE_INTERVAL
        )

        return CostBreakEvenResult(
            status=status,
            break_even_cost_bps=None,
            eligible_observations=len(gross_values),
            gross_cumulative_return=gross_cumulative_return,
            total_turnover=total_turnover,
            search_lower_bps=search_lower_bps,
            search_upper_bps=search_upper_bps,
            effective_upper_bps=effective_upper_bps,
            objective_at_lower=gross_log_wealth,
            objective_at_effective_upper=upper_objective,
        )

    if math.isclose(
        upper_objective,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        break_even_cost_bps = effective_upper_bps
    else:

        def objective(cost_bps: float) -> float:
            value = _log_wealth_at_cost(
                gross_values=gross_values,
                turnover_values=turnover_values,
                cost_bps=cost_bps,
            )

            if value is None:
                raise ValueError(
                    "Wealth became invalid inside the root bracket."
                )

            return value

        break_even_cost_bps = float(
            brentq(
                objective,
                search_lower_bps,
                effective_upper_bps,
                xtol=1e-12,
            )
        )

    return CostBreakEvenResult(
        status=BREAK_EVEN_STATUS_ROOT_FOUND,
        break_even_cost_bps=break_even_cost_bps,
        eligible_observations=len(gross_values),
        gross_cumulative_return=gross_cumulative_return,
        total_turnover=total_turnover,
        search_lower_bps=search_lower_bps,
        search_upper_bps=search_upper_bps,
        effective_upper_bps=effective_upper_bps,
        objective_at_lower=gross_log_wealth,
        objective_at_effective_upper=upper_objective,
    )


@dataclass(frozen=True, slots=True)
class TrendRatioConfigurationRun:
    """One configuration evaluated through the existing strategy engine."""

    configuration: TrendRatioConfiguration
    strategy_bundle: TrendRatioBundle
    filter_lag: FilterLagDiagnostics
    gross_performance: PerformanceMetrics
    net_performance: PerformanceMetrics
    cost_break_even: CostBreakEvenResult


def run_trend_ratio_configuration(
    frame: pd.DataFrame,
    *,
    configuration: TrendRatioConfiguration,
    cost_bps_per_turnover: Real = (
        BASELINE_COST_BPS_PER_TURNOVER
    ),
    annualization_factor: Real = DAY07_ANNUALIZATION_FACTOR,
    price_column: str = "close",
    return_column: str = "close_to_close_simple_return",
) -> TrendRatioConfigurationRun:
    """Evaluate one configuration through the unchanged Day 6 engine.

    This function performs no independent signal or P&L reconstruction.
    Strategy observations are produced exclusively by
    ``build_trend_ratio_strategy``.
    """

    if not isinstance(configuration, TrendRatioConfiguration):
        raise TypeError(
            "configuration must be a TrendRatioConfiguration."
        )

    if (
        isinstance(cost_bps_per_turnover, bool)
        or not isinstance(cost_bps_per_turnover, Real)
    ):
        raise TypeError(
            "cost_bps_per_turnover must be a finite real number."
        )

    normalized_cost = float(cost_bps_per_turnover)

    if not math.isfinite(normalized_cost):
        raise ValueError(
            "cost_bps_per_turnover must be finite."
        )

    if normalized_cost < 0.0:
        raise ValueError(
            "cost_bps_per_turnover must be non-negative."
        )

    parameters = TrendRatioParameters(
        short_window=configuration.short_window,
        long_window=configuration.long_window,
        neutral_band=configuration.neutral_band,
        cost_bps_per_turnover=normalized_cost,
        price_column=price_column,
        return_column=return_column,
    )

    strategy_bundle = build_trend_ratio_strategy(
        frame,
        parameters=parameters,
    )

    observations = strategy_bundle.observations

    gross_performance = calculate_performance_metrics(
        observations["gross_strategy_return"],
        annualization_factor=annualization_factor,
    )
    net_performance = calculate_performance_metrics(
        observations["net_strategy_return"],
        annualization_factor=annualization_factor,
    )

    cost_break_even = calculate_cost_break_even(
        observations["gross_strategy_return"],
        observations["turnover"],
    )

    return TrendRatioConfigurationRun(
        configuration=configuration,
        strategy_bundle=strategy_bundle,
        filter_lag=calculate_filter_lag(configuration),
        gross_performance=gross_performance,
        net_performance=net_performance,
        cost_break_even=cost_break_even,
    )


ANNUAL_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "calendar_year",
    "observations",
    "position_eligible_observations",
    "gross_return",
    "net_return",
    "net_annualized_volatility",
    "net_sharpe_ratio",
    "net_max_drawdown",
    "turnover",
    "long_exposure_pct",
    "short_exposure_pct",
    "neutral_exposure_pct",
)

ANNUAL_CONSISTENCY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "calendar_years",
    "positive_net_years",
    "worst_annual_net_return",
    "median_annual_net_return",
    "standard_deviation_annual_net_return",
    "positive_net_year_proportion",
)


def _validate_annual_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Validate completed strategy observations for annual summaries."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")

    if observations.empty:
        raise ValueError("observations must not be empty.")

    required_columns = (
        "timestamp",
        "symbol",
        "position",
        "position_eligible",
        "turnover",
        "gross_strategy_return",
        "net_strategy_return",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in observations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Annual strategy observations are missing required columns: "
            f"{missing_columns}."
        )

    result = observations.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Annual strategy observations contain malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise ValueError(
            "Annual strategy observations contain missing timestamps."
        )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result["symbol"].isna().any() or result["symbol"].eq("").any():
        raise ValueError(
            "Annual strategy observations require valid symbols."
        )

    for column in (
        "position",
        "turnover",
        "gross_strategy_return",
        "net_strategy_return",
    ):
        if pd.api.types.is_bool_dtype(result[column].dtype):
            raise TypeError(f"{column} must not use a Boolean dtype.")

        try:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{column} must contain numeric values."
            ) from exc

        if not result[column].map(math.isfinite).all():
            raise ValueError(
                f"{column} must contain only finite values."
            )

    if not result["position"].isin((-1.0, 0.0, 1.0)).all():
        raise ValueError(
            "position may contain only -1, 0, or 1."
        )

    if result["turnover"].lt(0.0).any():
        raise ValueError("turnover must be non-negative.")

    if result["gross_strategy_return"].le(-1.0).any():
        raise ValueError(
            "gross_strategy_return must be greater than -1.0."
        )

    if result["net_strategy_return"].le(-1.0).any():
        raise ValueError(
            "net_strategy_return must be greater than -1.0."
        )

    if result["position_eligible"].isna().any():
        raise ValueError(
            "position_eligible must not contain missing values."
        )

    result["position_eligible"] = (
        result["position_eligible"].astype(bool)
    )
    result["calendar_year"] = (
        result["timestamp"].dt.year.astype(int)
    )

    result = result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_mask = result.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Annual strategy observations contain duplicate "
            "symbol-timestamp rows."
        )

    return result


def _annual_exposure_percentage(
    position_sample: pd.DataFrame,
    *,
    position_value: int,
) -> float:
    """Calculate exposure over position-eligible observations only."""

    if position_sample.empty:
        return float("nan")

    return float(
        100.0
        * position_sample["position"].eq(position_value).mean()
    )


def build_annual_strategy_results(
    observations: pd.DataFrame,
    *,
    annualization_factor: Real = DAY07_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Summarize years from completed full-sample strategy observations.

    Signals and positions must already have been generated over the complete
    chronological sample. This function does not rerun the strategy and
    therefore does not reset positions at calendar-year boundaries.
    """

    validated = _validate_annual_observations(observations)
    records: list[dict[str, object]] = []

    for (symbol, calendar_year), group in validated.groupby(
        ["symbol", "calendar_year"],
        observed=True,
        sort=True,
    ):
        if len(group) < 2:
            raise ValueError(
                "Each annual group must contain at least two observations. "
                f"Received {symbol} {calendar_year}: {len(group)}."
            )

        gross_metrics = calculate_performance_metrics(
            group["gross_strategy_return"],
            annualization_factor=annualization_factor,
        )
        net_metrics = calculate_performance_metrics(
            group["net_strategy_return"],
            annualization_factor=annualization_factor,
        )

        position_sample = group.loc[
            group["position_eligible"]
        ]

        records.append(
            {
                "symbol": str(symbol),
                "calendar_year": int(calendar_year),
                "observations": int(len(group)),
                "position_eligible_observations": int(
                    len(position_sample)
                ),
                "gross_return": gross_metrics.cumulative_return,
                "net_return": net_metrics.cumulative_return,
                "net_annualized_volatility": (
                    net_metrics.annualized_volatility
                ),
                "net_sharpe_ratio": net_metrics.sharpe_ratio,
                "net_max_drawdown": net_metrics.max_drawdown,
                "turnover": float(group["turnover"].sum()),
                "long_exposure_pct": _annual_exposure_percentage(
                    position_sample,
                    position_value=1,
                ),
                "short_exposure_pct": _annual_exposure_percentage(
                    position_sample,
                    position_value=-1,
                ),
                "neutral_exposure_pct": _annual_exposure_percentage(
                    position_sample,
                    position_value=0,
                ),
            }
        )

    result = pd.DataFrame.from_records(
        records,
        columns=ANNUAL_RESULT_COLUMNS,
    )

    if int(result["observations"].sum()) != len(validated):
        raise RuntimeError(
            "Annual summaries failed to preserve all observations."
        )

    return result


def build_annual_consistency_summary(
    annual_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate annual net-return consistency by symbol."""

    if not isinstance(annual_results, pd.DataFrame):
        raise TypeError(
            "annual_results must be a pandas DataFrame."
        )

    if annual_results.empty:
        raise ValueError("annual_results must not be empty.")

    required_columns = (
        "symbol",
        "calendar_year",
        "net_return",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in annual_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Annual results are missing required columns: "
            f"{missing_columns}."
        )

    validated = annual_results.copy(deep=True)
    validated["symbol"] = (
        validated["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    try:
        validated["calendar_year"] = pd.to_numeric(
            validated["calendar_year"],
            errors="raise",
        ).astype(int)
        validated["net_return"] = pd.to_numeric(
            validated["net_return"],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "calendar_year and net_return must be numeric."
        ) from exc

    if not validated["net_return"].map(math.isfinite).all():
        raise ValueError(
            "net_return must contain only finite values."
        )

    if validated.duplicated(
        ["symbol", "calendar_year"],
        keep=False,
    ).any():
        raise ValueError(
            "annual_results must contain one row per symbol-year."
        )

    records: list[dict[str, object]] = []

    for symbol, group in validated.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        ordered = group.sort_values(
            "calendar_year",
            kind="stable",
        )
        annual_net_returns = ordered["net_return"]
        calendar_years = int(len(annual_net_returns))
        positive_net_years = int(
            annual_net_returns.gt(0.0).sum()
        )

        records.append(
            {
                "symbol": str(symbol),
                "calendar_years": calendar_years,
                "positive_net_years": positive_net_years,
                "worst_annual_net_return": float(
                    annual_net_returns.min()
                ),
                "median_annual_net_return": float(
                    annual_net_returns.median()
                ),
                "standard_deviation_annual_net_return": float(
                    annual_net_returns.std(ddof=1)
                )
                if calendar_years > 1
                else float("nan"),
                "positive_net_year_proportion": float(
                    positive_net_years / calendar_years
                ),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=ANNUAL_CONSISTENCY_COLUMNS,
    )


REGIME_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "regime",
    "observations",
    "position_eligible_observations",
    "net_return",
    "net_annualized_volatility",
    "net_sharpe_ratio",
    "invested_exposure_pct",
    "long_exposure_pct",
    "short_exposure_pct",
    "neutral_exposure_pct",
    "turnover",
)

EXPECTED_VOLATILITY_REGIMES: Final[frozenset[str]] = frozenset(
    {
        "normal_volatility",
        "high_volatility",
    }
)


def _normalize_session_dates(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Normalize session labels to timezone-naive midnight timestamps."""

    try:
        normalized = (
            pd.to_datetime(
                values.copy(deep=True),
                utc=True,
                errors="raise",
            )
            .dt.tz_convert(None)
            .dt.normalize()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} contain malformed session dates."
        ) from exc

    if normalized.isna().any():
        raise ValueError(
            f"{context} contain missing session dates."
        )

    return normalized


def _validate_regime_observations(
    regime_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the existing Day 5 volatility-regime output schema."""

    if not isinstance(regime_observations, pd.DataFrame):
        raise TypeError(
            "regime_observations must be a pandas DataFrame."
        )

    if regime_observations.empty:
        raise ValueError(
            "regime_observations must not be empty."
        )

    required_columns = (
        "session_date",
        "regime",
        "annualized_total_realized_volatility",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in regime_observations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Regime observations are missing required Day 5 columns: "
            f"{missing_columns}."
        )

    result = regime_observations.loc[
        :,
        required_columns,
    ].copy(deep=True)

    result["session_date"] = _normalize_session_dates(
        result["session_date"],
        context="Regime observations",
    )
    result["regime"] = (
        result["regime"]
        .astype("string")
        .str.strip()
    )

    if result["regime"].isna().any() or result["regime"].eq("").any():
        raise ValueError(
            "Regime observations contain missing regime labels."
        )

    unknown_regimes = sorted(
        set(result["regime"].astype(str))
        - EXPECTED_VOLATILITY_REGIMES
    )

    if unknown_regimes:
        raise ValueError(
            "Regime observations contain incompatible labels: "
            f"{unknown_regimes}."
        )

    try:
        result["annualized_total_realized_volatility"] = (
            pd.to_numeric(
                result[
                    "annualized_total_realized_volatility"
                ],
                errors="raise",
            ).astype(float)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "annualized_total_realized_volatility must be numeric."
        ) from exc

    if not result[
        "annualized_total_realized_volatility"
    ].map(math.isfinite).all():
        raise ValueError(
            "annualized_total_realized_volatility must be finite."
        )

    if result[
        "annualized_total_realized_volatility"
    ].lt(0.0).any():
        raise ValueError(
            "annualized_total_realized_volatility must be "
            "non-negative."
        )

    if result.duplicated(
        "session_date",
        keep=False,
    ).any():
        raise ValueError(
            "Regime observations must contain one row per session_date."
        )

    return result.sort_values(
        "session_date",
        kind="stable",
    ).reset_index(drop=True)


def _validate_strategy_session_dates(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Validate strategy observations and their exchange-session labels."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError(
            "observations must be a pandas DataFrame."
        )

    if "session_date" not in observations.columns:
        raise ValueError(
            "Strategy observations must contain session_date so Day 5 "
            "session regimes can be joined without deriving dates from UTC "
            "timestamps."
        )

    validated = _validate_annual_observations(observations)
    validated["session_date"] = _normalize_session_dates(
        validated["session_date"],
        context="Strategy observations",
    )

    return validated


def _regime_exposure_percentage(
    position_sample: pd.DataFrame,
    *,
    position_value: int,
) -> float:
    """Calculate one position-state exposure percentage."""

    if position_sample.empty:
        return float("nan")

    return float(
        100.0
        * position_sample["position"].eq(position_value).mean()
    )


def build_regime_strategy_results(
    observations: pd.DataFrame,
    regime_observations: pd.DataFrame,
    *,
    annualization_factor: Real = DAY07_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Summarize completed strategy observations by Day 5 regime.

    This function joins session-level regime labels onto already completed
    strategy observations. It does not recalculate signals, positions, returns,
    or volatility regimes.
    """

    validated_observations = _validate_strategy_session_dates(
        observations
    )
    validated_regimes = _validate_regime_observations(
        regime_observations
    )

    regimes_for_join = validated_regimes.rename(
        columns={
            "annualized_total_realized_volatility": (
                "benchmark_annualized_total_realized_volatility"
            )
        }
    )

    merged = validated_observations.merge(
        regimes_for_join,
        on="session_date",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    if len(merged) != len(validated_observations):
        raise RuntimeError(
            "Regime join changed the number of strategy observations."
        )

    unmatched = merged["_merge"].ne("both")

    if unmatched.any():
        missing_dates = (
            merged.loc[unmatched, "session_date"]
            .drop_duplicates()
            .sort_values()
            .dt.strftime("%Y-%m-%d")
            .tolist()[:10]
        )
        raise ValueError(
            "No Day 5 volatility regime exists for one or more strategy "
            f"sessions. Examples: {missing_dates}."
        )

    merged = merged.drop(columns=["_merge"])

    records: list[dict[str, object]] = []

    for (symbol, regime), group in merged.groupby(
        ["symbol", "regime"],
        observed=True,
        sort=True,
    ):
        if len(group) < 2:
            raise ValueError(
                "Each symbol-regime group requires at least two "
                f"observations. Received {symbol}/{regime}: {len(group)}."
            )

        net_metrics = calculate_performance_metrics(
            group["net_strategy_return"],
            annualization_factor=annualization_factor,
        )

        position_sample = group.loc[
            group["position_eligible"]
        ]

        invested_exposure_pct = (
            float(
                100.0
                * position_sample["position"].ne(0).mean()
            )
            if not position_sample.empty
            else float("nan")
        )

        records.append(
            {
                "symbol": str(symbol),
                "regime": str(regime),
                "observations": int(len(group)),
                "position_eligible_observations": int(
                    len(position_sample)
                ),
                "net_return": net_metrics.cumulative_return,
                "net_annualized_volatility": (
                    net_metrics.annualized_volatility
                ),
                "net_sharpe_ratio": net_metrics.sharpe_ratio,
                "invested_exposure_pct": invested_exposure_pct,
                "long_exposure_pct": _regime_exposure_percentage(
                    position_sample,
                    position_value=1,
                ),
                "short_exposure_pct": _regime_exposure_percentage(
                    position_sample,
                    position_value=-1,
                ),
                "neutral_exposure_pct": _regime_exposure_percentage(
                    position_sample,
                    position_value=0,
                ),
                "turnover": float(group["turnover"].sum()),
            }
        )

    result = pd.DataFrame.from_records(
        records,
        columns=REGIME_RESULT_COLUMNS,
    )

    if int(result["observations"].sum()) != len(
        validated_observations
    ):
        raise RuntimeError(
            "Regime summaries failed to preserve all strategy "
            "observations."
        )

    expected_turnover = float(
        validated_observations["turnover"].sum()
    )
    summarized_turnover = float(result["turnover"].sum())

    if not math.isclose(
        summarized_turnover,
        expected_turnover,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Regime summaries failed to preserve total turnover."
        )

    return result.sort_values(
        ["symbol", "regime"],
        kind="stable",
    ).reset_index(drop=True)


def build_volatility_regime_strategy_results(
    observations: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    stress_quantile: float = 0.80,
    annualization_factor: Real = DAY07_ANNUALIZATION_FACTOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Day 7 regime results using the existing Day 5 definition."""

    regime_observations, regime_definition = (
        build_volatility_regimes(
            daily_volatility,
            benchmark_symbol=benchmark_symbol,
            stress_quantile=stress_quantile,
        )
    )

    regime_results = build_regime_strategy_results(
        observations,
        regime_observations,
        annualization_factor=annualization_factor,
    )

    return (
        regime_results,
        regime_definition.copy(deep=True),
    )


SIGNAL_VALIDATION_BUCKET_COUNT: Final[int] = 5

FORWARD_SIGNAL_SAMPLE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timestamp",
    "first_forward_timestamp",
    "forward_end_timestamp",
    "horizon_bars",
    "continuous_signal",
    "forward_return",
)

SIGNAL_VALIDATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "horizon_bars",
    "observations",
    "pearson_information_coefficient",
    "spearman_information_coefficient",
    "requested_signal_buckets",
    "actual_signal_buckets",
    "bucket_mean_spearman_monotonicity",
    "adjacent_increasing_bucket_proportion",
)

SIGNAL_BUCKET_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "horizon_bars",
    "signal_bucket",
    "observations",
    "signal_minimum",
    "signal_maximum",
    "signal_mean",
    "mean_forward_return",
    "median_forward_return",
)


def _validate_signal_horizons(
    horizons: Sequence[int],
) -> tuple[int, ...]:
    """Validate deterministic, positive and strictly increasing horizons."""

    validated: list[int] = []

    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError(
                "Signal-validation horizons must be integers."
            )

        if horizon <= 0:
            raise ValueError(
                "Signal-validation horizons must be positive."
            )

        validated.append(horizon)

    normalized = tuple(validated)

    if not normalized:
        raise ValueError(
            "Signal-validation horizons must not be empty."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Signal-validation horizons must not contain duplicates."
        )

    if any(
        current >= following
        for current, following in zip(
            normalized,
            normalized[1:],
        )
    ):
        raise ValueError(
            "Signal-validation horizons must be strictly increasing."
        )

    return normalized


def _validate_signal_observations(
    observations: pd.DataFrame,
    *,
    price_column: str,
    ratio_column: str,
) -> pd.DataFrame:
    """Validate strategy observations used for forward-signal analysis."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError(
            "observations must be a pandas DataFrame."
        )

    if observations.empty:
        raise ValueError(
            "observations must not be empty."
        )

    required_columns = (
        "timestamp",
        "symbol",
        price_column,
        ratio_column,
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in observations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Signal observations are missing required columns: "
            f"{missing_columns}."
        )

    result = observations.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Signal observations contain malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise ValueError(
            "Signal observations contain missing timestamps."
        )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result["symbol"].isna().any() or result["symbol"].eq("").any():
        raise ValueError(
            "Signal observations require valid symbols."
        )

    if pd.api.types.is_bool_dtype(result[price_column].dtype):
        raise TypeError(
            f"{price_column} must not use a Boolean dtype."
        )

    if pd.api.types.is_bool_dtype(result[ratio_column].dtype):
        raise TypeError(
            f"{ratio_column} must not use a Boolean dtype."
        )

    try:
        result[price_column] = pd.to_numeric(
            result[price_column],
            errors="raise",
        ).astype(float)
        result[ratio_column] = pd.to_numeric(
            result[ratio_column],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{price_column} and {ratio_column} must be numeric."
        ) from exc

    if not result[price_column].map(math.isfinite).all():
        raise ValueError(
            f"{price_column} must contain only finite values."
        )

    if result[price_column].le(0.0).any():
        raise ValueError(
            f"{price_column} must contain only positive values."
        )

    non_missing_ratios = result[ratio_column].dropna()

    if not non_missing_ratios.map(math.isfinite).all():
        raise ValueError(
            f"Non-missing {ratio_column} values must be finite."
        )

    result = result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    if result.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise ValueError(
            "Signal observations contain duplicate symbol-timestamp rows."
        )

    return result


def build_forward_signal_sample(
    observations: pd.DataFrame,
    *,
    horizon_bars: int,
    price_column: str = "close",
    ratio_column: str = "ma_price_ratio",
) -> pd.DataFrame:
    """Build one diagnostic signal/forward-return sample.

    For signal observed at bar t, the forward return is P[t+h] / P[t] - 1.
    Its first constituent one-bar return therefore begins at t+1.
    """

    if isinstance(horizon_bars, bool) or not isinstance(
        horizon_bars,
        int,
    ):
        raise TypeError(
            "horizon_bars must be an integer."
        )

    if horizon_bars <= 0:
        raise ValueError(
            "horizon_bars must be positive."
        )

    validated = _validate_signal_observations(
        observations,
        price_column=price_column,
        ratio_column=ratio_column,
    )

    grouped = validated.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    validated["continuous_signal"] = (
        validated[ratio_column] - 1.0
    )
    validated["first_forward_timestamp"] = (
        grouped["timestamp"].shift(-1)
    )
    validated["forward_end_timestamp"] = (
        grouped["timestamp"].shift(-horizon_bars)
    )
    validated["forward_price"] = (
        grouped[price_column].shift(-horizon_bars)
    )
    validated["forward_return"] = (
        validated["forward_price"]
        / validated[price_column]
        - 1.0
    )
    validated["horizon_bars"] = horizon_bars

    eligible = validated.loc[
        validated["continuous_signal"].notna()
        & validated["first_forward_timestamp"].notna()
        & validated["forward_end_timestamp"].notna()
        & validated["forward_return"].notna()
    ].copy()

    if not eligible.empty:
        if not (
            eligible["first_forward_timestamp"]
            > eligible["timestamp"]
        ).all():
            raise RuntimeError(
                "Forward-return analysis included a return beginning "
                "at or before the signal timestamp."
            )

        if not (
            eligible["forward_end_timestamp"]
            >= eligible["first_forward_timestamp"]
        ).all():
            raise RuntimeError(
                "Forward-return end timestamps precede their first "
                "constituent return timestamps."
            )

    return eligible.loc[
        :,
        FORWARD_SIGNAL_SAMPLE_COLUMNS,
    ].reset_index(drop=True)


def _safe_information_coefficient(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    method: str,
) -> float:
    """Calculate a correlation or return NaN when undefined."""

    if len(signal) < 2:
        return float("nan")

    if signal.nunique(dropna=True) < 2:
        return float("nan")

    if forward_return.nunique(dropna=True) < 2:
        return float("nan")

    value = signal.corr(
        forward_return,
        method=method,
    )

    return (
        float(value)
        if pd.notna(value)
        else float("nan")
    )


def _assign_signal_buckets(
    sample: pd.DataFrame,
    *,
    requested_bucket_count: int,
) -> pd.Series:
    """Assign deterministic equal-frequency signal buckets.

    Stable first-rank ordering resolves tied signal values without random
    assignment. No price or forward-return information enters bucket creation.
    """

    if sample.empty:
        return pd.Series(
            index=sample.index,
            dtype="Int64",
        )

    actual_bucket_count = min(
        requested_bucket_count,
        len(sample),
    )

    stable_rank = sample["continuous_signal"].rank(
        method="first",
        ascending=True,
    )

    bucket_codes = pd.qcut(
        stable_rank,
        q=actual_bucket_count,
        labels=False,
        duplicates="drop",
    )

    return (
        bucket_codes.astype("int64") + 1
    ).astype("Int64")


def build_signal_validation_results(
    observations: pd.DataFrame,
    *,
    horizons: Sequence[int] = FORWARD_RETURN_HORIZONS,
    signal_bucket_count: int = SIGNAL_VALIDATION_BUCKET_COUNT,
    price_column: str = "close",
    ratio_column: str = "ma_price_ratio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate information coefficients and signal-bucket diagnostics."""

    validated_horizons = _validate_signal_horizons(horizons)

    if isinstance(signal_bucket_count, bool) or not isinstance(
        signal_bucket_count,
        int,
    ):
        raise TypeError(
            "signal_bucket_count must be an integer."
        )

    if signal_bucket_count < 2:
        raise ValueError(
            "signal_bucket_count must be at least two."
        )

    summary_records: list[dict[str, object]] = []
    bucket_records: list[dict[str, object]] = []

    for horizon in validated_horizons:
        forward_sample = build_forward_signal_sample(
            observations,
            horizon_bars=horizon,
            price_column=price_column,
            ratio_column=ratio_column,
        )

        for symbol, symbol_sample in forward_sample.groupby(
            "symbol",
            observed=True,
            sort=True,
        ):
            sample = symbol_sample.sort_values(
                "timestamp",
                kind="stable",
            ).copy()

            pearson_ic = _safe_information_coefficient(
                sample["continuous_signal"],
                sample["forward_return"],
                method="pearson",
            )
            spearman_ic = _safe_information_coefficient(
                sample["continuous_signal"],
                sample["forward_return"],
                method="spearman",
            )

            sample["signal_bucket"] = _assign_signal_buckets(
                sample,
                requested_bucket_count=signal_bucket_count,
            )

            grouped_buckets = (
                sample.groupby(
                    "signal_bucket",
                    observed=True,
                    sort=True,
                )
                .agg(
                    observations=(
                        "forward_return",
                        "size",
                    ),
                    signal_minimum=(
                        "continuous_signal",
                        "min",
                    ),
                    signal_maximum=(
                        "continuous_signal",
                        "max",
                    ),
                    signal_mean=(
                        "continuous_signal",
                        "mean",
                    ),
                    mean_forward_return=(
                        "forward_return",
                        "mean",
                    ),
                    median_forward_return=(
                        "forward_return",
                        "median",
                    ),
                )
                .reset_index()
            )

            actual_bucket_count = int(
                len(grouped_buckets)
            )

            if actual_bucket_count >= 2:
                bucket_numbers = grouped_buckets[
                    "signal_bucket"
                ].astype(float)
                bucket_means = grouped_buckets[
                    "mean_forward_return"
                ].astype(float)

                bucket_monotonicity = (
                    _safe_information_coefficient(
                        bucket_numbers,
                        bucket_means,
                        method="spearman",
                    )
                )

                adjacent_increasing_proportion = float(
                    bucket_means.diff().dropna().gt(0.0).mean()
                )
            else:
                bucket_monotonicity = float("nan")
                adjacent_increasing_proportion = float("nan")

            summary_records.append(
                {
                    "symbol": str(symbol),
                    "horizon_bars": int(horizon),
                    "observations": int(len(sample)),
                    "pearson_information_coefficient": pearson_ic,
                    "spearman_information_coefficient": spearman_ic,
                    "requested_signal_buckets": signal_bucket_count,
                    "actual_signal_buckets": actual_bucket_count,
                    "bucket_mean_spearman_monotonicity": (
                        bucket_monotonicity
                    ),
                    "adjacent_increasing_bucket_proportion": (
                        adjacent_increasing_proportion
                    ),
                }
            )

            for row in grouped_buckets.itertuples(
                index=False,
            ):
                bucket_records.append(
                    {
                        "symbol": str(symbol),
                        "horizon_bars": int(horizon),
                        "signal_bucket": int(
                            row.signal_bucket
                        ),
                        "observations": int(
                            row.observations
                        ),
                        "signal_minimum": float(
                            row.signal_minimum
                        ),
                        "signal_maximum": float(
                            row.signal_maximum
                        ),
                        "signal_mean": float(
                            row.signal_mean
                        ),
                        "mean_forward_return": float(
                            row.mean_forward_return
                        ),
                        "median_forward_return": float(
                            row.median_forward_return
                        ),
                    }
                )

    summary = pd.DataFrame.from_records(
        summary_records,
        columns=SIGNAL_VALIDATION_SUMMARY_COLUMNS,
    )
    buckets = pd.DataFrame.from_records(
        bucket_records,
        columns=SIGNAL_BUCKET_RESULT_COLUMNS,
    )

    return (
        summary.sort_values(
            ["symbol", "horizon_bars"],
            kind="stable",
        ).reset_index(drop=True),
        buckets.sort_values(
            ["symbol", "horizon_bars", "signal_bucket"],
            kind="stable",
        ).reset_index(drop=True),
    )


CONFIGURATION_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "configuration_id",
    "short_window",
    "long_window",
    "neutral_band",
    "short_filter_lag_bars",
    "long_filter_lag_bars",
    "lag_spread_bars",
    "observations",
    "position_eligible_observations",
    "gross_cumulative_return",
    "gross_sharpe_ratio",
    "net_cumulative_return",
    "net_sharpe_ratio",
    "net_max_drawdown",
    "total_turnover",
    "position_changing_bars",
    "long_exposure_pct",
    "short_exposure_pct",
    "neutral_exposure_pct",
    "break_even_status",
    "break_even_cost_bps",
    "calendar_years",
    "positive_net_years",
    "worst_annual_net_return",
    "median_annual_net_return",
    "standard_deviation_annual_net_return",
    "positive_net_year_proportion",
)

HOLDING_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "configuration_id",
    "short_window",
    "long_window",
    "neutral_band",
    "eligible_observations",
    "non_zero_episode_count",
    "long_episode_count",
    "short_episode_count",
    "median_holding_duration_bars",
    "mean_holding_duration_bars",
    "holding_duration_25th_percentile_bars",
    "holding_duration_75th_percentile_bars",
    "maximum_holding_duration_bars",
    "overnight_carry_episode_count",
    "session_crossing_episode_proportion",
    "whipsaw_count",
    "whipsaw_rate_per_1000_eligible_observations",
    "whipsaw_episode_proportion",
)

NEIGHBORHOOD_STABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "short_window",
    "long_window",
    "neutral_band",
    "neighbor_count",
    "neighbor_configuration_ids",
    "net_sharpe_ratio",
    "total_turnover",
    "median_neighbor_net_sharpe",
    "minimum_neighbor_net_sharpe",
    "standard_deviation_neighbor_net_sharpe",
    "proportion_neighbors_positive_net_return",
    "proportion_neighbors_positive_break_even",
    "median_neighbor_turnover",
    "mean_absolute_one_step_net_sharpe_difference",
    "mean_absolute_one_step_turnover_difference",
)


@dataclass(frozen=True, slots=True)
class TrendRatioSensitivityTables:
    """Compact in-memory outputs from the frozen Day 7 grid."""

    parameter_results: pd.DataFrame
    annual_results: pd.DataFrame
    annual_consistency: pd.DataFrame
    regime_results: pd.DataFrame
    regime_definition: pd.DataFrame
    holding_diagnostics: pd.DataFrame
    signal_validation: pd.DataFrame
    signal_buckets: pd.DataFrame
    neighborhood_stability: pd.DataFrame


def _validate_frozen_configurations(
    configurations: Sequence[TrendRatioConfiguration],
) -> tuple[TrendRatioConfiguration, ...]:
    """Validate a deterministic subset of the frozen Day 7 grid."""

    normalized = tuple(configurations)

    if not normalized:
        raise ValueError(
            "configurations must not be empty."
        )

    if not all(
        isinstance(configuration, TrendRatioConfiguration)
        for configuration in normalized
    ):
        raise TypeError(
            "Every configuration must be a "
            "TrendRatioConfiguration."
        )

    configuration_ids = tuple(
        configuration.configuration_id
        for configuration in normalized
    )

    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError(
            "configurations must not contain duplicates."
        )

    invalid = [
        configuration.configuration_id
        for configuration in normalized
        if (
            configuration.short_window not in SHORT_WINDOWS
            or configuration.long_window not in LONG_WINDOWS
            or configuration.neutral_band not in NEUTRAL_BANDS
        )
    ]

    if invalid:
        raise ValueError(
            "Every configuration must belong to the frozen Day 7 "
            f"parameter grid. Invalid configurations: {invalid}."
        )

    return normalized


def _extract_single_symbol(frame: pd.DataFrame) -> str:
    """Extract one normalized symbol from a strategy input frame."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if "symbol" not in frame.columns:
        raise ValueError("frame must contain a symbol column.")

    symbols = (
        frame["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    symbols = [
        str(symbol)
        for symbol in symbols
        if str(symbol)
    ]

    if len(symbols) != 1:
        raise ValueError(
            "Day 7 grid orchestration requires exactly one symbol. "
            f"Received: {symbols}."
        )

    return symbols[0]


def _configuration_prefix(
    configuration: TrendRatioConfiguration,
) -> dict[str, object]:
    """Return the repeated compact configuration identifiers."""

    return {
        "configuration_id": configuration.configuration_id,
        "short_window": configuration.short_window,
        "long_window": configuration.long_window,
        "neutral_band": configuration.neutral_band,
    }


def _attach_configuration_columns(
    table: pd.DataFrame,
    configuration: TrendRatioConfiguration,
) -> pd.DataFrame:
    """Attach configuration fields without mutating the supplied table."""

    result = table.copy(deep=True)
    prefix = _configuration_prefix(configuration)

    for column, value in reversed(tuple(prefix.items())):
        result.insert(0, column, value)

    return result


def _eligible_position_exposure(
    observations: pd.DataFrame,
    *,
    position_value: int,
) -> float:
    """Calculate exposure over position-eligible observations."""

    sample = observations.loc[
        observations["position_eligible"].astype(bool)
    ]

    if sample.empty:
        return float("nan")

    return float(
        100.0
        * sample["position"].eq(position_value).mean()
    )


def _empty_configured_table(
    base_columns: Sequence[str],
) -> pd.DataFrame:
    """Create an empty compact table with configuration columns."""

    return pd.DataFrame(
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *tuple(base_columns),
        )
    )


def build_neighborhood_stability(
    parameter_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate component-based one-step neighbourhood diagnostics.

    No composite stability score is created.
    """

    if not isinstance(parameter_results, pd.DataFrame):
        raise TypeError(
            "parameter_results must be a pandas DataFrame."
        )

    if parameter_results.empty:
        raise ValueError(
            "parameter_results must not be empty."
        )

    required_columns = (
        "configuration_id",
        "short_window",
        "long_window",
        "neutral_band",
        "net_sharpe_ratio",
        "net_cumulative_return",
        "total_turnover",
        "break_even_status",
        "break_even_cost_bps",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in parameter_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Parameter results are missing neighbourhood columns: "
            f"{missing_columns}."
        )

    validated = parameter_results.copy(deep=True)

    if validated["configuration_id"].duplicated().any():
        raise ValueError(
            "parameter_results contain duplicate configuration_id values."
        )

    for column in (
        "short_window",
        "long_window",
        "neutral_band",
        "net_sharpe_ratio",
        "net_cumulative_return",
        "total_turnover",
        "break_even_cost_bps",
    ):
        try:
            validated[column] = pd.to_numeric(
                validated[column],
                errors="coerce"
                if column in (
                    "net_sharpe_ratio",
                    "break_even_cost_bps",
                )
                else "raise",
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{column} must contain numeric values."
            ) from exc

    if not validated[
        "net_cumulative_return"
    ].map(math.isfinite).all():
        raise ValueError(
            "net_cumulative_return must contain finite values."
        )

    if not validated["total_turnover"].map(math.isfinite).all():
        raise ValueError(
            "total_turnover must contain finite values."
        )

    if validated["total_turnover"].lt(0.0).any():
        raise ValueError(
            "total_turnover must be non-negative."
        )

    configurations: list[TrendRatioConfiguration] = []

    for row in validated.itertuples(index=False):
        configuration = TrendRatioConfiguration(
            short_window=int(row.short_window),
            long_window=int(row.long_window),
            neutral_band=float(row.neutral_band),
        )

        if configuration.configuration_id != row.configuration_id:
            raise ValueError(
                "configuration_id does not match its parameter values: "
                f"{row.configuration_id}."
            )

        configurations.append(configuration)

    configuration_sequence = _validate_frozen_configurations(
        configurations
    )
    neighborhood_map = build_neighborhood_map(
        configuration_sequence
    )

    indexed = validated.set_index(
        "configuration_id",
        drop=False,
    )
    records: list[dict[str, object]] = []

    for configuration in configuration_sequence:
        configuration_id = configuration.configuration_id
        own_row = indexed.loc[configuration_id]
        neighbor_ids = neighborhood_map[configuration_id]
        neighbors = indexed.loc[list(neighbor_ids)].copy()

        neighbor_sharpe = (
            neighbors["net_sharpe_ratio"]
            .replace([float("inf"), float("-inf")], float("nan"))
            .dropna()
        )
        neighbor_turnover = neighbors["total_turnover"]

        own_sharpe = float(own_row["net_sharpe_ratio"])
        own_turnover = float(own_row["total_turnover"])

        median_neighbor_sharpe = (
            float(neighbor_sharpe.median())
            if not neighbor_sharpe.empty
            else float("nan")
        )
        minimum_neighbor_sharpe = (
            float(neighbor_sharpe.min())
            if not neighbor_sharpe.empty
            else float("nan")
        )
        standard_deviation_neighbor_sharpe = (
            float(neighbor_sharpe.std(ddof=0))
            if not neighbor_sharpe.empty
            else float("nan")
        )
        median_neighbor_turnover = (
            float(neighbor_turnover.median())
            if not neighbor_turnover.empty
            else float("nan")
        )

        positive_net_return_proportion = (
            float(
                neighbors[
                    "net_cumulative_return"
                ].gt(0.0).mean()
            )
            if len(neighbors)
            else float("nan")
        )

        positive_break_even = (
            neighbors["break_even_status"].eq(
                BREAK_EVEN_STATUS_ROOT_FOUND
            )
            & neighbors["break_even_cost_bps"].gt(0.0)
        )
        positive_break_even_proportion = (
            float(positive_break_even.mean())
            if len(neighbors)
            else float("nan")
        )

        if math.isfinite(own_sharpe) and not neighbor_sharpe.empty:
            sharpe_sensitivity = float(
                neighbor_sharpe.sub(own_sharpe).abs().mean()
            )
        else:
            sharpe_sensitivity = float("nan")

        turnover_sensitivity = (
            float(
                neighbor_turnover.sub(own_turnover).abs().mean()
            )
            if len(neighbor_turnover)
            else float("nan")
        )

        records.append(
            {
                "configuration_id": configuration_id,
                "short_window": configuration.short_window,
                "long_window": configuration.long_window,
                "neutral_band": configuration.neutral_band,
                "neighbor_count": len(neighbor_ids),
                "neighbor_configuration_ids": "|".join(
                    neighbor_ids
                ),
                "net_sharpe_ratio": own_sharpe,
                "total_turnover": own_turnover,
                "median_neighbor_net_sharpe": (
                    median_neighbor_sharpe
                ),
                "minimum_neighbor_net_sharpe": (
                    minimum_neighbor_sharpe
                ),
                "standard_deviation_neighbor_net_sharpe": (
                    standard_deviation_neighbor_sharpe
                ),
                "proportion_neighbors_positive_net_return": (
                    positive_net_return_proportion
                ),
                "proportion_neighbors_positive_break_even": (
                    positive_break_even_proportion
                ),
                "median_neighbor_turnover": (
                    median_neighbor_turnover
                ),
                "mean_absolute_one_step_net_sharpe_difference": (
                    sharpe_sensitivity
                ),
                "mean_absolute_one_step_turnover_difference": (
                    turnover_sensitivity
                ),
            }
        )

    return (
        pd.DataFrame.from_records(
            records,
            columns=NEIGHBORHOOD_STABILITY_COLUMNS,
        )
        .sort_values(
            [
                "short_window",
                "long_window",
                "neutral_band",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def run_trend_ratio_sensitivity_grid(
    frame: pd.DataFrame,
    *,
    configurations: Sequence[
        TrendRatioConfiguration
    ] | None = None,
    cost_bps_per_turnover: Real = (
        BASELINE_COST_BPS_PER_TURNOVER
    ),
    annualization_factor: Real = DAY07_ANNUALIZATION_FACTOR,
    price_column: str = "close",
    return_column: str = "close_to_close_simple_return",
    session_column: str = "session_date",
    signal_horizons: Sequence[int] = FORWARD_RETURN_HORIZONS,
    daily_volatility: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    stress_quantile: float = 0.80,
) -> TrendRatioSensitivityTables:
    """Run a deterministic subset or the complete frozen Day 7 grid.

    Only compact summaries are retained. Full bar-level outputs remain local to
    each loop iteration and are not returned or written.
    """

    symbol = _extract_single_symbol(frame)
    configuration_sequence = _validate_frozen_configurations(
        build_parameter_grid()
        if configurations is None
        else configurations
    )

    if configurations is None and len(configuration_sequence) != 36:
        raise RuntimeError(
            "The frozen default Day 7 grid must contain 36 configurations."
        )

    regime_observations: pd.DataFrame | None = None
    regime_definition = pd.DataFrame()

    if daily_volatility is not None:
        regime_observations, regime_definition = (
            build_volatility_regimes(
                daily_volatility,
                benchmark_symbol=benchmark_symbol,
                stress_quantile=stress_quantile,
            )
        )

    parameter_records: list[dict[str, object]] = []
    annual_tables: list[pd.DataFrame] = []
    annual_consistency_tables: list[pd.DataFrame] = []
    regime_tables: list[pd.DataFrame] = []
    holding_records: list[dict[str, object]] = []
    signal_tables: list[pd.DataFrame] = []
    signal_bucket_tables: list[pd.DataFrame] = []

    for configuration in configuration_sequence:
        run = run_trend_ratio_configuration(
            frame,
            configuration=configuration,
            cost_bps_per_turnover=cost_bps_per_turnover,
            annualization_factor=annualization_factor,
            price_column=price_column,
            return_column=return_column,
        )

        observations = run.strategy_bundle.observations

        if session_column not in observations.columns:
            raise ValueError(
                f"Strategy observations must contain {session_column!r} "
                "for holding and regime diagnostics."
            )

        annual = build_annual_strategy_results(
            observations,
            annualization_factor=annualization_factor,
        )
        annual_consistency = (
            build_annual_consistency_summary(annual)
        )

        if len(annual_consistency) != 1:
            raise RuntimeError(
                "Single-symbol grid orchestration produced an unexpected "
                "annual-consistency row count."
            )

        annual_row = annual_consistency.iloc[0]

        holding = calculate_holding_diagnostics(
            observations["position"],
            session_labels=observations[session_column],
        )

        signal_summary, signal_buckets = (
            build_signal_validation_results(
                observations,
                horizons=signal_horizons,
                price_column=price_column,
                ratio_column="ma_price_ratio",
            )
        )

        if regime_observations is not None:
            regime = build_regime_strategy_results(
                observations,
                regime_observations,
                annualization_factor=annualization_factor,
            )
            regime_tables.append(
                _attach_configuration_columns(
                    regime,
                    configuration,
                )
            )

        eligible_positions = observations.loc[
            observations["position_eligible"].astype(bool)
        ]

        parameter_records.append(
            {
                "symbol": symbol,
                **_configuration_prefix(configuration),
                "short_filter_lag_bars": (
                    run.filter_lag.short_filter_lag_bars
                ),
                "long_filter_lag_bars": (
                    run.filter_lag.long_filter_lag_bars
                ),
                "lag_spread_bars": (
                    run.filter_lag.lag_spread_bars
                ),
                "observations": int(len(observations)),
                "position_eligible_observations": int(
                    len(eligible_positions)
                ),
                "gross_cumulative_return": (
                    run.gross_performance.cumulative_return
                ),
                "gross_sharpe_ratio": (
                    run.gross_performance.sharpe_ratio
                ),
                "net_cumulative_return": (
                    run.net_performance.cumulative_return
                ),
                "net_sharpe_ratio": (
                    run.net_performance.sharpe_ratio
                ),
                "net_max_drawdown": (
                    run.net_performance.max_drawdown
                ),
                "total_turnover": float(
                    observations["turnover"].sum()
                ),
                "position_changing_bars": int(
                    observations["turnover"].gt(0.0).sum()
                ),
                "long_exposure_pct": (
                    _eligible_position_exposure(
                        observations,
                        position_value=1,
                    )
                ),
                "short_exposure_pct": (
                    _eligible_position_exposure(
                        observations,
                        position_value=-1,
                    )
                ),
                "neutral_exposure_pct": (
                    _eligible_position_exposure(
                        observations,
                        position_value=0,
                    )
                ),
                "break_even_status": (
                    run.cost_break_even.status
                ),
                "break_even_cost_bps": (
                    run.cost_break_even.break_even_cost_bps
                ),
                "calendar_years": int(
                    annual_row["calendar_years"]
                ),
                "positive_net_years": int(
                    annual_row["positive_net_years"]
                ),
                "worst_annual_net_return": float(
                    annual_row["worst_annual_net_return"]
                ),
                "median_annual_net_return": float(
                    annual_row["median_annual_net_return"]
                ),
                "standard_deviation_annual_net_return": float(
                    annual_row[
                        "standard_deviation_annual_net_return"
                    ]
                ),
                "positive_net_year_proportion": float(
                    annual_row[
                        "positive_net_year_proportion"
                    ]
                ),
            }
        )

        holding_records.append(
            {
                "symbol": symbol,
                **_configuration_prefix(configuration),
                **asdict(holding),
            }
        )

        annual_tables.append(
            _attach_configuration_columns(
                annual,
                configuration,
            )
        )
        annual_consistency_tables.append(
            _attach_configuration_columns(
                annual_consistency,
                configuration,
            )
        )
        signal_tables.append(
            _attach_configuration_columns(
                signal_summary,
                configuration,
            )
        )
        signal_bucket_tables.append(
            _attach_configuration_columns(
                signal_buckets,
                configuration,
            )
        )

    parameter_results = (
        pd.DataFrame.from_records(
            parameter_records,
            columns=CONFIGURATION_RESULT_COLUMNS,
        )
        .sort_values(
            [
                "short_window",
                "long_window",
                "neutral_band",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    holding_diagnostics = (
        pd.DataFrame.from_records(
            holding_records,
            columns=HOLDING_RESULT_COLUMNS,
        )
        .sort_values(
            [
                "short_window",
                "long_window",
                "neutral_band",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    annual_results = pd.concat(
        annual_tables,
        ignore_index=True,
    )
    annual_consistency_results = pd.concat(
        annual_consistency_tables,
        ignore_index=True,
    )
    signal_validation = pd.concat(
        signal_tables,
        ignore_index=True,
    )
    signal_bucket_results = pd.concat(
        signal_bucket_tables,
        ignore_index=True,
    )

    if regime_tables:
        regime_results = pd.concat(
            regime_tables,
            ignore_index=True,
        )
    else:
        regime_results = _empty_configured_table(
            REGIME_RESULT_COLUMNS
        )

    neighborhood_stability = build_neighborhood_stability(
        parameter_results
    )

    return TrendRatioSensitivityTables(
        parameter_results=parameter_results,
        annual_results=annual_results,
        annual_consistency=annual_consistency_results,
        regime_results=regime_results,
        regime_definition=regime_definition.copy(deep=True),
        holding_diagnostics=holding_diagnostics,
        signal_validation=signal_validation,
        signal_buckets=signal_bucket_results,
        neighborhood_stability=neighborhood_stability,
    )
