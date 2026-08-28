"""Deterministic foundations for Day 9 EMA/MACD sensitivity analysis.

This module defines the frozen parameter grid, recursive-filter diagnostics,
and the axis-adjacent neighbourhood relationship.

It does not load market data, execute strategies, optimise parameters, or
write artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
import math
from numbers import Integral, Real
from typing import Final

import pandas as pd

from systematic_alpha.analysis.dependence_diagnostics import (
    build_volatility_regimes,
)
from systematic_alpha.analysis.ema_macd_baseline import (
    DAY08_FORWARD_HORIZONS,
    EMA_MACD_SIGNAL_BUCKET_COLUMNS,
    EMA_MACD_SIGNAL_SUMMARY_COLUMNS,
    build_ema_macd_signal_validation,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    BREAK_EVEN_STATUS_ROOT_FOUND,
    REGIME_RESULT_COLUMNS,
    CostBreakEvenResult,
    HoldingDiagnostics,
    build_annual_consistency_summary,
    build_annual_strategy_results,
    build_regime_strategy_results,
    calculate_cost_break_even,
    calculate_holding_diagnostics,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdBundle,
    EmaMacdParameters,
    build_ema_macd_strategy,
    calculate_ema_alpha,
    calculate_ema_half_life_bars,
)


FAST_WINDOWS: Final[tuple[int, ...]] = (8, 12, 16)
SLOW_WINDOWS: Final[tuple[int, ...]] = (20, 26, 32)
SIGNAL_WINDOWS: Final[tuple[int, ...]] = (6, 9, 12)
NEUTRAL_BANDS: Final[tuple[float, ...]] = (
    0.0,
    0.00025,
    0.00050,
    0.00100,
)

BASELINE_FAST_WINDOW: Final[int] = 12
BASELINE_SLOW_WINDOW: Final[int] = 26
BASELINE_SIGNAL_WINDOW: Final[int] = 9
BASELINE_NEUTRAL_BAND: Final[float] = 0.00050

EXPECTED_CONFIGURATION_COUNT: Final[int] = 108


class EmaMacdSensitivityError(ValueError):
    """Raised when the frozen Day 9 experiment is malformed."""


@dataclass(frozen=True, order=True, slots=True)
class EmaMacdConfiguration:
    """One immutable EMA/MACD sensitivity configuration."""

    fast_window: int
    slow_window: int
    signal_window: int
    neutral_band: float

    def __post_init__(self) -> None:
        for name in (
            "fast_window",
            "slow_window",
            "signal_window",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")

            if value <= 0:
                raise EmaMacdSensitivityError(
                    f"{name} must be strictly positive."
                )

        if self.fast_window >= self.slow_window:
            raise EmaMacdSensitivityError(
                "fast_window must be smaller than slow_window."
            )

        if (
            isinstance(self.neutral_band, bool)
            or not isinstance(self.neutral_band, Real)
        ):
            raise TypeError("neutral_band must be a real number.")

        normalized_band = float(self.neutral_band)

        if not math.isfinite(normalized_band):
            raise EmaMacdSensitivityError(
                "neutral_band must be finite."
            )

        if normalized_band < 0.0:
            raise EmaMacdSensitivityError(
                "neutral_band must be non-negative."
            )

        object.__setattr__(
            self,
            "neutral_band",
            normalized_band,
        )

    @property
    def configuration_id(self) -> str:
        """Return a stable compact identifier."""

        band_token = f"{self.neutral_band:.5f}".replace(
            ".",
            "p",
        )

        return (
            f"f{self.fast_window:03d}"
            f"_s{self.slow_window:03d}"
            f"_m{self.signal_window:03d}"
            f"_b{band_token}"
        )


@dataclass(frozen=True, slots=True)
class EmaMacdFilterDiagnostics:
    """Descriptive recursive-filter diagnostics in observation bars."""

    fast_alpha: float
    slow_alpha: float
    signal_alpha: float
    fast_half_life_bars: float
    slow_half_life_bars: float
    signal_half_life_bars: float
    slow_minus_fast_half_life_bars: float


def _validate_strictly_increasing_axis(
    values: Sequence[int | float],
    *,
    name: str,
) -> tuple[int | float, ...]:
    """Validate one declared ordered parameter axis."""

    normalized = tuple(values)

    if not normalized:
        raise EmaMacdSensitivityError(
            f"{name} must not be empty."
        )

    if len(set(normalized)) != len(normalized):
        raise EmaMacdSensitivityError(
            f"{name} must not contain duplicates."
        )

    if any(
        current >= following
        for current, following in zip(
            normalized,
            normalized[1:],
        )
    ):
        raise EmaMacdSensitivityError(
            f"{name} must be strictly increasing."
        )

    return normalized


def build_ema_macd_parameter_grid(
    *,
    fast_windows: Sequence[int] = FAST_WINDOWS,
    slow_windows: Sequence[int] = SLOW_WINDOWS,
    signal_windows: Sequence[int] = SIGNAL_WINDOWS,
    neutral_bands: Sequence[float] = NEUTRAL_BANDS,
) -> tuple[EmaMacdConfiguration, ...]:
    """Build the deterministic Cartesian Day 9 parameter grid."""

    validated_fast = tuple(
        int(value)
        for value in _validate_strictly_increasing_axis(
            fast_windows,
            name="fast_windows",
        )
    )
    validated_slow = tuple(
        int(value)
        for value in _validate_strictly_increasing_axis(
            slow_windows,
            name="slow_windows",
        )
    )
    validated_signal = tuple(
        int(value)
        for value in _validate_strictly_increasing_axis(
            signal_windows,
            name="signal_windows",
        )
    )
    validated_bands = tuple(
        float(value)
        for value in _validate_strictly_increasing_axis(
            neutral_bands,
            name="neutral_bands",
        )
    )

    if validated_fast[0] <= 0:
        raise EmaMacdSensitivityError(
            "fast_windows must contain positive values."
        )

    if validated_slow[0] <= 0:
        raise EmaMacdSensitivityError(
            "slow_windows must contain positive values."
        )

    if validated_signal[0] <= 0:
        raise EmaMacdSensitivityError(
            "signal_windows must contain positive values."
        )

    if validated_bands[0] < 0.0:
        raise EmaMacdSensitivityError(
            "neutral_bands must contain non-negative values."
        )

    if max(validated_fast) >= min(validated_slow):
        raise EmaMacdSensitivityError(
            "Every declared fast window must be smaller than "
            "every declared slow window."
        )

    configurations = tuple(
        EmaMacdConfiguration(
            fast_window=fast_window,
            slow_window=slow_window,
            signal_window=signal_window,
            neutral_band=neutral_band,
        )
        for (
            fast_window,
            slow_window,
            signal_window,
            neutral_band,
        ) in product(
            validated_fast,
            validated_slow,
            validated_signal,
            validated_bands,
        )
    )

    identifiers = tuple(
        configuration.configuration_id
        for configuration in configurations
    )

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "Parameter-grid construction produced duplicate "
            "configuration identifiers."
        )

    return configurations


def calculate_filter_diagnostics(
    configuration: EmaMacdConfiguration,
) -> EmaMacdFilterDiagnostics:
    """Calculate descriptive EMA alpha and half-life diagnostics."""

    if not isinstance(configuration, EmaMacdConfiguration):
        raise TypeError(
            "configuration must be an EmaMacdConfiguration."
        )

    fast_half_life = calculate_ema_half_life_bars(
        configuration.fast_window
    )
    slow_half_life = calculate_ema_half_life_bars(
        configuration.slow_window
    )

    return EmaMacdFilterDiagnostics(
        fast_alpha=calculate_ema_alpha(
            configuration.fast_window
        ),
        slow_alpha=calculate_ema_alpha(
            configuration.slow_window
        ),
        signal_alpha=calculate_ema_alpha(
            configuration.signal_window
        ),
        fast_half_life_bars=fast_half_life,
        slow_half_life_bars=slow_half_life,
        signal_half_life_bars=calculate_ema_half_life_bars(
            configuration.signal_window
        ),
        slow_minus_fast_half_life_bars=(
            slow_half_life - fast_half_life
        ),
    )


def configurations_are_neighbors(
    left: EmaMacdConfiguration,
    right: EmaMacdConfiguration,
    *,
    fast_windows: Sequence[int] = FAST_WINDOWS,
    slow_windows: Sequence[int] = SLOW_WINDOWS,
    signal_windows: Sequence[int] = SIGNAL_WINDOWS,
    neutral_bands: Sequence[float] = NEUTRAL_BANDS,
) -> bool:
    """Return whether two configurations are one grid step apart.

    Adjacent configurations differ along exactly one parameter axis by one
    declared position. Their other three parameter values are identical.
    """

    axis_indexes = (
        {
            value: index
            for index, value in enumerate(fast_windows)
        },
        {
            value: index
            for index, value in enumerate(slow_windows)
        },
        {
            value: index
            for index, value in enumerate(signal_windows)
        },
        {
            float(value): index
            for index, value in enumerate(neutral_bands)
        },
    )

    try:
        left_coordinates = (
            axis_indexes[0][left.fast_window],
            axis_indexes[1][left.slow_window],
            axis_indexes[2][left.signal_window],
            axis_indexes[3][left.neutral_band],
        )
        right_coordinates = (
            axis_indexes[0][right.fast_window],
            axis_indexes[1][right.slow_window],
            axis_indexes[2][right.signal_window],
            axis_indexes[3][right.neutral_band],
        )
    except KeyError as exc:
        raise EmaMacdSensitivityError(
            "Both configurations must belong to the declared grid."
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
    configurations: Sequence[EmaMacdConfiguration] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build a deterministic symmetric neighbourhood map."""

    parameter_grid = (
        build_ema_macd_parameter_grid()
        if configurations is None
        else tuple(configurations)
    )

    identifiers = tuple(
        configuration.configuration_id
        for configuration in parameter_grid
    )

    if len(set(identifiers)) != len(identifiers):
        raise EmaMacdSensitivityError(
            "configurations must not contain duplicates."
        )

    neighborhoods: dict[str, tuple[str, ...]] = {}

    for configuration in parameter_grid:
        neighbor_ids = sorted(
            candidate.configuration_id
            for candidate in parameter_grid
            if candidate != configuration
            and configurations_are_neighbors(
                configuration,
                candidate,
            )
        )

        neighborhoods[
            configuration.configuration_id
        ] = tuple(neighbor_ids)

    return neighborhoods


DAY09_ANNUALIZATION_FACTOR: Final[int] = 252 * 26
DAY09_COST_BPS_PER_TURNOVER: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class EmaMacdConfigurationRun:
    """One configuration evaluated through the Day 8 strategy engine."""

    configuration: EmaMacdConfiguration
    strategy_bundle: EmaMacdBundle
    filter_diagnostics: EmaMacdFilterDiagnostics
    gross_performance: PerformanceMetrics
    net_performance: PerformanceMetrics
    holding_diagnostics: HoldingDiagnostics
    cost_break_even: CostBreakEvenResult
    annual_results: pd.DataFrame
    annual_consistency: pd.DataFrame


def _validate_non_negative_real(
    value: Real,
    *,
    name: str,
) -> float:
    """Validate one finite non-negative real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a finite real number."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise EmaMacdSensitivityError(
            f"{name} must be finite."
        )

    if normalized < 0.0:
        raise EmaMacdSensitivityError(
            f"{name} must be non-negative."
        )

    return normalized


def configuration_to_parameters(
    configuration: EmaMacdConfiguration,
    *,
    cost_bps_per_turnover: Real = (
        DAY09_COST_BPS_PER_TURNOVER
    ),
    price_column: str = "close",
    return_column: str = "close_to_close_simple_return",
) -> EmaMacdParameters:
    """Translate one grid configuration into Day 8 parameters."""

    if not isinstance(configuration, EmaMacdConfiguration):
        raise TypeError(
            "configuration must be an EmaMacdConfiguration."
        )

    normalized_cost = _validate_non_negative_real(
        cost_bps_per_turnover,
        name="cost_bps_per_turnover",
    )

    return EmaMacdParameters(
        fast_window=configuration.fast_window,
        slow_window=configuration.slow_window,
        signal_window=configuration.signal_window,
        neutral_band=configuration.neutral_band,
        cost_bps_per_turnover=normalized_cost,
        price_column=price_column,
        return_column=return_column,
    )


def run_ema_macd_configuration(
    frame: pd.DataFrame,
    *,
    configuration: EmaMacdConfiguration,
    cost_bps_per_turnover: Real = (
        DAY09_COST_BPS_PER_TURNOVER
    ),
    annualization_factor: Real = (
        DAY09_ANNUALIZATION_FACTOR
    ),
    price_column: str = "close",
    return_column: str = "close_to_close_simple_return",
    session_column: str = "session_date",
) -> EmaMacdConfigurationRun:
    """Evaluate one configuration through the unchanged Day 8 engine.

    This function does not independently reconstruct EMA states, signals,
    positions, turnover, transaction costs, or strategy returns.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if not isinstance(configuration, EmaMacdConfiguration):
        raise TypeError(
            "configuration must be an EmaMacdConfiguration."
        )

    if (
        not isinstance(session_column, str)
        or not session_column.strip()
    ):
        raise TypeError(
            "session_column must be a non-empty string."
        )

    parameters = configuration_to_parameters(
        configuration,
        cost_bps_per_turnover=cost_bps_per_turnover,
        price_column=price_column,
        return_column=return_column,
    )

    strategy_bundle = build_ema_macd_strategy(
        frame,
        parameters=parameters,
    )
    observations = strategy_bundle.observations

    symbols = (
        observations["symbol"]
        .drop_duplicates()
        .tolist()
    )

    if len(symbols) != 1:
        raise EmaMacdSensitivityError(
            "One EMA/MACD sensitivity run requires exactly "
            "one symbol."
        )

    if session_column not in observations.columns:
        raise EmaMacdSensitivityError(
            f"Strategy observations must contain "
            f"{session_column!r} for holding diagnostics."
        )

    gross_performance = calculate_performance_metrics(
        observations["gross_strategy_return"],
        annualization_factor=annualization_factor,
    )
    net_performance = calculate_performance_metrics(
        observations["net_strategy_return"],
        annualization_factor=annualization_factor,
    )

    holding_diagnostics = calculate_holding_diagnostics(
        observations["position"],
        session_labels=observations[session_column],
    )
    cost_break_even = calculate_cost_break_even(
        observations["gross_strategy_return"],
        observations["turnover"],
    )

    annual_results = build_annual_strategy_results(
        observations,
        annualization_factor=annualization_factor,
    )
    annual_consistency = (
        build_annual_consistency_summary(
            annual_results
        )
    )

    if len(annual_consistency) != 1:
        raise RuntimeError(
            "A single-symbol configuration run produced an "
            "unexpected annual-consistency row count."
        )

    return EmaMacdConfigurationRun(
        configuration=configuration,
        strategy_bundle=strategy_bundle,
        filter_diagnostics=calculate_filter_diagnostics(
            configuration
        ),
        gross_performance=gross_performance,
        net_performance=net_performance,
        holding_diagnostics=holding_diagnostics,
        cost_break_even=cost_break_even,
        annual_results=annual_results,
        annual_consistency=annual_consistency,
    )


EMA_MACD_CONFIGURATION_RESULT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "symbol",
    "configuration_id",
    "fast_window",
    "slow_window",
    "signal_window",
    "neutral_band",
    "fast_alpha",
    "slow_alpha",
    "signal_alpha",
    "fast_half_life_bars",
    "slow_half_life_bars",
    "signal_half_life_bars",
    "slow_minus_fast_half_life_bars",
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

EMA_MACD_HOLDING_RESULT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "symbol",
    "configuration_id",
    "fast_window",
    "slow_window",
    "signal_window",
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

EMA_MACD_NEIGHBORHOOD_STABILITY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "configuration_id",
    "fast_window",
    "slow_window",
    "signal_window",
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

CONFIGURATION_PARAMETER_COLUMNS: Final[
    tuple[str, ...]
] = (
    "configuration_id",
    "fast_window",
    "slow_window",
    "signal_window",
    "neutral_band",
)


@dataclass(frozen=True, slots=True)
class EmaMacdSensitivityTables:
    """Compact in-memory outputs from the frozen Day 9 grid."""

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
    configurations: Sequence[EmaMacdConfiguration],
) -> tuple[EmaMacdConfiguration, ...]:
    """Validate a deterministic subset of the frozen grid."""

    normalized = tuple(configurations)

    if not normalized:
        raise EmaMacdSensitivityError(
            "configurations must not be empty."
        )

    if not all(
        isinstance(
            configuration,
            EmaMacdConfiguration,
        )
        for configuration in normalized
    ):
        raise TypeError(
            "Every configuration must be an "
            "EmaMacdConfiguration."
        )

    identifiers = tuple(
        configuration.configuration_id
        for configuration in normalized
    )

    if len(set(identifiers)) != len(identifiers):
        raise EmaMacdSensitivityError(
            "configurations must not contain duplicates."
        )

    invalid = [
        configuration.configuration_id
        for configuration in normalized
        if (
            configuration.fast_window not in FAST_WINDOWS
            or configuration.slow_window not in SLOW_WINDOWS
            or configuration.signal_window not in SIGNAL_WINDOWS
            or configuration.neutral_band
            not in NEUTRAL_BANDS
        )
    ]

    if invalid:
        raise EmaMacdSensitivityError(
            "Every configuration must belong to the frozen "
            "Day 9 parameter grid. Invalid configurations: "
            f"{invalid}."
        )

    return normalized


def _extract_single_symbol(
    frame: pd.DataFrame,
) -> str:
    """Extract exactly one normalized strategy symbol."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if "symbol" not in frame.columns:
        raise EmaMacdSensitivityError(
            "frame must contain a symbol column."
        )

    symbols = (
        frame["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )
    normalized = [
        str(symbol)
        for symbol in symbols
        if str(symbol)
    ]

    if len(normalized) != 1:
        raise EmaMacdSensitivityError(
            "Day 9 grid orchestration requires exactly one "
            f"symbol. Received: {normalized}."
        )

    return normalized[0]


def _configuration_prefix(
    configuration: EmaMacdConfiguration,
) -> dict[str, object]:
    """Return repeated compact configuration identifiers."""

    return {
        "configuration_id": (
            configuration.configuration_id
        ),
        "fast_window": configuration.fast_window,
        "slow_window": configuration.slow_window,
        "signal_window": configuration.signal_window,
        "neutral_band": configuration.neutral_band,
    }


def _attach_configuration_columns(
    table: pd.DataFrame,
    configuration: EmaMacdConfiguration,
) -> pd.DataFrame:
    """Attach configuration fields without mutating a table."""

    result = table.copy(deep=True)
    prefix = _configuration_prefix(configuration)

    for column, value in reversed(
        tuple(prefix.items())
    ):
        result.insert(0, column, value)

    return result


def _eligible_position_exposure(
    observations: pd.DataFrame,
    *,
    position_value: int,
) -> float:
    """Calculate exposure over position-eligible bars."""

    eligible = observations.loc[
        observations["position_eligible"].astype(bool)
    ]

    if eligible.empty:
        return float("nan")

    return float(
        100.0
        * eligible["position"].eq(position_value).mean()
    )


def _empty_configured_table(
    base_columns: Sequence[str],
) -> pd.DataFrame:
    """Build an empty table with configuration columns."""

    return pd.DataFrame(
        columns=(
            *CONFIGURATION_PARAMETER_COLUMNS,
            *tuple(base_columns),
        )
    )


def _configuration_sort_columns() -> list[str]:
    """Return the deterministic Day 9 table sort order."""

    return [
        "fast_window",
        "slow_window",
        "signal_window",
        "neutral_band",
    ]


def build_ema_macd_neighborhood_stability(
    parameter_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build component-based one-step stability diagnostics.

    No composite stability score or post-hoc ranking is created.
    """

    if not isinstance(parameter_results, pd.DataFrame):
        raise TypeError(
            "parameter_results must be a pandas DataFrame."
        )

    if parameter_results.empty:
        raise EmaMacdSensitivityError(
            "parameter_results must not be empty."
        )

    required_columns = (
        *CONFIGURATION_PARAMETER_COLUMNS,
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
        raise EmaMacdSensitivityError(
            "Parameter results are missing neighbourhood "
            f"columns: {missing_columns}."
        )

    validated = parameter_results.copy(deep=True)

    if validated["configuration_id"].duplicated().any():
        raise EmaMacdSensitivityError(
            "parameter_results contain duplicate "
            "configuration_id values."
        )

    numeric_columns = (
        "fast_window",
        "slow_window",
        "signal_window",
        "neutral_band",
        "net_sharpe_ratio",
        "net_cumulative_return",
        "total_turnover",
        "break_even_cost_bps",
    )

    for column in numeric_columns:
        try:
            validated[column] = pd.to_numeric(
                validated[column],
                errors=(
                    "coerce"
                    if column
                    in (
                        "net_sharpe_ratio",
                        "break_even_cost_bps",
                    )
                    else "raise"
                ),
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{column} must contain numeric values."
            ) from exc

    for column in (
        "fast_window",
        "slow_window",
        "signal_window",
    ):
        valid_integer = validated[column].map(
            lambda value: (
                math.isfinite(value)
                and value.is_integer()
            )
        )

        if not valid_integer.all():
            raise EmaMacdSensitivityError(
                f"{column} must contain finite integer values."
            )

    if not validated[
        "neutral_band"
    ].map(math.isfinite).all():
        raise EmaMacdSensitivityError(
            "neutral_band must contain finite values."
        )

    if not validated[
        "net_cumulative_return"
    ].map(math.isfinite).all():
        raise EmaMacdSensitivityError(
            "net_cumulative_return must contain finite values."
        )

    if not validated[
        "total_turnover"
    ].map(math.isfinite).all():
        raise EmaMacdSensitivityError(
            "total_turnover must contain finite values."
        )

    if validated["total_turnover"].lt(0.0).any():
        raise EmaMacdSensitivityError(
            "total_turnover must be non-negative."
        )

    configurations: list[
        EmaMacdConfiguration
    ] = []

    for row in validated.itertuples(index=False):
        configuration = EmaMacdConfiguration(
            fast_window=int(row.fast_window),
            slow_window=int(row.slow_window),
            signal_window=int(row.signal_window),
            neutral_band=float(row.neutral_band),
        )

        if (
            configuration.configuration_id
            != row.configuration_id
        ):
            raise EmaMacdSensitivityError(
                "configuration_id does not match its "
                "parameter values: "
                f"{row.configuration_id}."
            )

        configurations.append(configuration)

    configuration_sequence = (
        _validate_frozen_configurations(
            configurations
        )
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
        configuration_id = (
            configuration.configuration_id
        )
        own_row = indexed.loc[configuration_id]
        neighbor_ids = neighborhood_map[
            configuration_id
        ]

        neighbors = (
            indexed.loc[list(neighbor_ids)].copy()
            if neighbor_ids
            else indexed.iloc[0:0].copy()
        )

        neighbor_sharpe = (
            neighbors["net_sharpe_ratio"]
            .replace(
                [
                    float("inf"),
                    float("-inf"),
                ],
                float("nan"),
            )
            .dropna()
        )
        neighbor_turnover = neighbors[
            "total_turnover"
        ]

        own_sharpe = float(
            own_row["net_sharpe_ratio"]
        )
        own_turnover = float(
            own_row["total_turnover"]
        )

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

        sharpe_sensitivity = (
            float(
                neighbor_sharpe
                .sub(own_sharpe)
                .abs()
                .mean()
            )
            if (
                math.isfinite(own_sharpe)
                and not neighbor_sharpe.empty
            )
            else float("nan")
        )
        turnover_sensitivity = (
            float(
                neighbor_turnover
                .sub(own_turnover)
                .abs()
                .mean()
            )
            if len(neighbor_turnover)
            else float("nan")
        )

        records.append(
            {
                **_configuration_prefix(
                    configuration
                ),
                "neighbor_count": len(
                    neighbor_ids
                ),
                "neighbor_configuration_ids": (
                    "|".join(neighbor_ids)
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
            columns=(
                EMA_MACD_NEIGHBORHOOD_STABILITY_COLUMNS
            ),
        )
        .sort_values(
            _configuration_sort_columns(),
            kind="stable",
        )
        .reset_index(drop=True)
    )


def run_ema_macd_sensitivity_grid(
    frame: pd.DataFrame,
    *,
    configurations: Sequence[
        EmaMacdConfiguration
    ] | None = None,
    cost_bps_per_turnover: Real = (
        DAY09_COST_BPS_PER_TURNOVER
    ),
    annualization_factor: Real = (
        DAY09_ANNUALIZATION_FACTOR
    ),
    price_column: str = "close",
    return_column: str = (
        "close_to_close_simple_return"
    ),
    session_column: str = "session_date",
    signal_horizons: Sequence[int] = (
        DAY08_FORWARD_HORIZONS
    ),
    daily_volatility: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    stress_quantile: float = 0.80,
) -> EmaMacdSensitivityTables:
    """Run a subset or the complete frozen Day 9 grid.

    Full observation-level outputs remain local to each loop
    iteration. Only compact summary tables are retained.
    """

    symbol = _extract_single_symbol(frame)

    configuration_sequence = (
        _validate_frozen_configurations(
            build_ema_macd_parameter_grid()
            if configurations is None
            else configurations
        )
    )

    if (
        configurations is None
        and len(configuration_sequence)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise RuntimeError(
            "The frozen default Day 9 grid must contain "
            f"{EXPECTED_CONFIGURATION_COUNT} configurations."
        )

    regime_observations: pd.DataFrame | None = None
    regime_definition = pd.DataFrame()

    if daily_volatility is not None:
        (
            regime_observations,
            regime_definition,
        ) = build_volatility_regimes(
            daily_volatility,
            benchmark_symbol=benchmark_symbol,
            stress_quantile=stress_quantile,
        )

    parameter_records: list[
        dict[str, object]
    ] = []
    annual_tables: list[pd.DataFrame] = []
    annual_consistency_tables: list[
        pd.DataFrame
    ] = []
    regime_tables: list[pd.DataFrame] = []
    holding_records: list[
        dict[str, object]
    ] = []
    signal_tables: list[pd.DataFrame] = []
    signal_bucket_tables: list[
        pd.DataFrame
    ] = []

    for configuration in configuration_sequence:
        run = run_ema_macd_configuration(
            frame,
            configuration=configuration,
            cost_bps_per_turnover=(
                cost_bps_per_turnover
            ),
            annualization_factor=(
                annualization_factor
            ),
            price_column=price_column,
            return_column=return_column,
            session_column=session_column,
        )

        observations = (
            run.strategy_bundle.observations
        )

        if session_column not in observations.columns:
            raise EmaMacdSensitivityError(
                "Strategy observations must contain "
                f"{session_column!r} for holding and "
                "regime diagnostics."
            )

        annual = run.annual_results.copy(deep=True)
        annual_consistency = (
            run.annual_consistency.copy(deep=True)
        )

        if len(annual_consistency) != 1:
            raise RuntimeError(
                "Single-symbol grid orchestration produced "
                "an unexpected annual-consistency row count."
            )

        annual_row = annual_consistency.iloc[0]
        holding = run.holding_diagnostics

        signal_summary, signal_buckets = (
            build_ema_macd_signal_validation(
                observations,
                horizons=signal_horizons,
                price_column=price_column,
                signal_column=(
                    "normalized_macd_histogram"
                ),
            )
        )

        if regime_observations is not None:
            regime = build_regime_strategy_results(
                observations,
                regime_observations,
                annualization_factor=(
                    annualization_factor
                ),
            )
            regime_tables.append(
                _attach_configuration_columns(
                    regime,
                    configuration,
                )
            )

        eligible_positions = observations.loc[
            observations[
                "position_eligible"
            ].astype(bool)
        ]
        filters = run.filter_diagnostics

        parameter_records.append(
            {
                "symbol": symbol,
                **_configuration_prefix(
                    configuration
                ),
                "fast_alpha": filters.fast_alpha,
                "slow_alpha": filters.slow_alpha,
                "signal_alpha": (
                    filters.signal_alpha
                ),
                "fast_half_life_bars": (
                    filters.fast_half_life_bars
                ),
                "slow_half_life_bars": (
                    filters.slow_half_life_bars
                ),
                "signal_half_life_bars": (
                    filters.signal_half_life_bars
                ),
                "slow_minus_fast_half_life_bars": (
                    filters
                    .slow_minus_fast_half_life_bars
                ),
                "observations": int(
                    len(observations)
                ),
                "position_eligible_observations": (
                    int(len(eligible_positions))
                ),
                "gross_cumulative_return": (
                    run.gross_performance
                    .cumulative_return
                ),
                "gross_sharpe_ratio": (
                    run.gross_performance
                    .sharpe_ratio
                ),
                "net_cumulative_return": (
                    run.net_performance
                    .cumulative_return
                ),
                "net_sharpe_ratio": (
                    run.net_performance
                    .sharpe_ratio
                ),
                "net_max_drawdown": (
                    run.net_performance
                    .max_drawdown
                ),
                "total_turnover": float(
                    observations["turnover"].sum()
                ),
                "position_changing_bars": int(
                    observations[
                        "turnover"
                    ].gt(0.0).sum()
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
                    run.cost_break_even
                    .break_even_cost_bps
                ),
                "calendar_years": int(
                    annual_row["calendar_years"]
                ),
                "positive_net_years": int(
                    annual_row[
                        "positive_net_years"
                    ]
                ),
                "worst_annual_net_return": float(
                    annual_row[
                        "worst_annual_net_return"
                    ]
                ),
                "median_annual_net_return": float(
                    annual_row[
                        "median_annual_net_return"
                    ]
                ),
                "standard_deviation_annual_net_return": (
                    float(
                        annual_row[
                            "standard_deviation_annual_net_return"
                        ]
                    )
                ),
                "positive_net_year_proportion": (
                    float(
                        annual_row[
                            "positive_net_year_proportion"
                        ]
                    )
                ),
            }
        )

        holding_records.append(
            {
                "symbol": symbol,
                **_configuration_prefix(
                    configuration
                ),
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
            columns=(
                EMA_MACD_CONFIGURATION_RESULT_COLUMNS
            ),
        )
        .sort_values(
            _configuration_sort_columns(),
            kind="stable",
        )
        .reset_index(drop=True)
    )

    holding_diagnostics = (
        pd.DataFrame.from_records(
            holding_records,
            columns=(
                EMA_MACD_HOLDING_RESULT_COLUMNS
            ),
        )
        .sort_values(
            _configuration_sort_columns(),
            kind="stable",
        )
        .reset_index(drop=True)
    )

    annual_results = pd.concat(
        annual_tables,
        ignore_index=True,
    ).sort_values(
        [
            *_configuration_sort_columns(),
            "calendar_year",
        ],
        kind="stable",
    ).reset_index(drop=True)

    annual_consistency_results = pd.concat(
        annual_consistency_tables,
        ignore_index=True,
    ).sort_values(
        _configuration_sort_columns(),
        kind="stable",
    ).reset_index(drop=True)

    signal_validation = pd.concat(
        signal_tables,
        ignore_index=True,
    ).sort_values(
        [
            *_configuration_sort_columns(),
            "horizon_bars",
        ],
        kind="stable",
    ).reset_index(drop=True)

    signal_bucket_results = pd.concat(
        signal_bucket_tables,
        ignore_index=True,
    ).sort_values(
        [
            *_configuration_sort_columns(),
            "horizon_bars",
            "signal_bucket",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if regime_tables:
        regime_results = pd.concat(
            regime_tables,
            ignore_index=True,
        ).sort_values(
            [
                *_configuration_sort_columns(),
                "regime",
            ],
            kind="stable",
        ).reset_index(drop=True)
    else:
        regime_results = _empty_configured_table(
            REGIME_RESULT_COLUMNS
        )

    neighborhood_stability = (
        build_ema_macd_neighborhood_stability(
            parameter_results
        )
    )

    return EmaMacdSensitivityTables(
        parameter_results=parameter_results,
        annual_results=annual_results,
        annual_consistency=(
            annual_consistency_results
        ),
        regime_results=regime_results,
        regime_definition=(
            regime_definition.copy(deep=True)
        ),
        holding_diagnostics=holding_diagnostics,
        signal_validation=signal_validation,
        signal_buckets=signal_bucket_results,
        neighborhood_stability=(
            neighborhood_stability
        ),
    )
