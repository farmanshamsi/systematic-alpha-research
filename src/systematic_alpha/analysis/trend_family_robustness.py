"""Development-only frequency robustness for frozen trend baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import pandas as pd

from systematic_alpha.analysis.eda_features import (
    REQUIRED_COLUMNS as CANONICAL_BAR_COLUMNS,
    build_return_features,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdParameters,
    build_ema_macd_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    DEFAULT_COST_BPS_PER_TURNOVER,
    DEFAULT_LONG_WINDOW,
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_SHORT_WINDOW,
    TrendRatioParameters,
    build_trend_ratio_strategy,
)


DEVELOPMENT_DATASET_ID: Final[str] = (
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical"
)

DEVELOPMENT_START: Final[pd.Timestamp] = (
    pd.Timestamp("2020-01-02")
)
DEVELOPMENT_END: Final[pd.Timestamp] = (
    pd.Timestamp("2025-12-31")
)

ROBUSTNESS_STRATEGIES: Final[
    tuple[str, ...]
] = (
    "trend_ratio",
    "ema_macd",
)
ROBUSTNESS_SYMBOLS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
)
ROBUSTNESS_FREQUENCIES: Final[
    tuple[str, ...]
] = (
    "15min",
    "30min",
    "60min",
)

ANNUALIZATION_FACTORS: Final[
    dict[str, int]
] = {
    "15min": 252 * 26,
    "30min": 252 * 13,
    "60min": 252 * 7,
}

EXPECTED_PORTFOLIO_OBSERVATIONS: Final[
    dict[str, int]
] = {
    "15min": 117_192,
    "30min": 58_596,
    "60min": 31_560,
}

TREND_RATIO_CONFIGURATION_ID: Final[str] = (
    "trend_ratio_day06_baseline_v1"
)
EMA_MACD_CONFIGURATION_ID: Final[str] = (
    "ema_macd_day08_baseline_v1"
)

TREND_RATIO_PARAMETERS: Final[
    TrendRatioParameters
] = TrendRatioParameters(
    short_window=DEFAULT_SHORT_WINDOW,
    long_window=DEFAULT_LONG_WINDOW,
    neutral_band=DEFAULT_NEUTRAL_BAND,
    cost_bps_per_turnover=(
        DEFAULT_COST_BPS_PER_TURNOVER
    ),
    price_column="close",
    return_column=(
        "close_to_close_simple_return"
    ),
)
EMA_MACD_PARAMETERS: Final[
    EmaMacdParameters
] = EmaMacdParameters()

CONFIGURATION_IDS: Final[dict[str, str]] = {
    "trend_ratio": (
        TREND_RATIO_CONFIGURATION_ID
    ),
    "ema_macd": EMA_MACD_CONFIGURATION_ID,
}

REQUIRED_RESULT_COLUMNS: Final[
    tuple[str, ...]
] = (
    "strategy",
    "symbol",
    "frequency",
    "dataset_id",
    "configuration_id",
    "start_timestamp",
    "end_timestamp",
    "sessions",
    "observations",
    "annualization_factor",
    "partial_bar_count",
    "warmup_observations",
    "active_observations",
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


class TrendFamilyRobustnessError(ValueError):
    """Raised when Day 10 robustness cannot run safely."""


@dataclass(frozen=True, slots=True)
class RobustnessRunSpec:
    """One immutable strategy-symbol-frequency run."""

    strategy: str
    symbol: str
    frequency: str
    annualization_factor: int

    def __post_init__(self) -> None:
        """Validate membership in the frozen Day 10 matrix."""

        if self.strategy not in ROBUSTNESS_STRATEGIES:
            raise TrendFamilyRobustnessError(
                "strategy is outside the frozen Day 10 "
                "matrix."
            )

        if self.symbol not in ROBUSTNESS_SYMBOLS:
            raise TrendFamilyRobustnessError(
                "symbol is outside the frozen Day 10 "
                "matrix."
            )

        if (
            self.frequency
            not in ROBUSTNESS_FREQUENCIES
        ):
            raise TrendFamilyRobustnessError(
                "frequency is outside the frozen Day 10 "
                "matrix."
            )

        expected_factor = ANNUALIZATION_FACTORS[
            self.frequency
        ]

        if (
            isinstance(
                self.annualization_factor,
                bool,
            )
            or self.annualization_factor
            != expected_factor
        ):
            raise TrendFamilyRobustnessError(
                "annualization_factor does not match the "
                "frozen frequency mapping."
            )


def build_robustness_run_matrix() -> tuple[
    RobustnessRunSpec,
    ...,
]:
    """Construct the deterministic frozen 18-run matrix."""

    matrix = tuple(
        RobustnessRunSpec(
            strategy=strategy,
            symbol=symbol,
            frequency=frequency,
            annualization_factor=(
                ANNUALIZATION_FACTORS[
                    frequency
                ]
            ),
        )
        for strategy in ROBUSTNESS_STRATEGIES
        for symbol in ROBUSTNESS_SYMBOLS
        for frequency in ROBUSTNESS_FREQUENCIES
    )

    keys = {
        (
            specification.strategy,
            specification.symbol,
            specification.frequency,
        )
        for specification in matrix
    }

    if len(matrix) != 18 or len(keys) != 18:
        raise RuntimeError(
            "The frozen Day 10 matrix must contain "
            "18 unique runs."
        )

    return matrix


def _validate_development_bars(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the supplied development-only universe."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    if bars.empty:
        raise TrendFamilyRobustnessError(
            "bars must not be empty."
        )

    missing_columns = sorted(
        set(CANONICAL_BAR_COLUMNS).difference(
            bars.columns
        )
    )

    if missing_columns:
        raise TrendFamilyRobustnessError(
            "Canonical development bars are missing "
            f"required columns: {missing_columns}."
        )

    result = bars.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendFamilyRobustnessError(
            "Development bars contain malformed "
            "timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise TrendFamilyRobustnessError(
            "Development timestamps cannot be missing."
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
    expected_symbols = set(
        ROBUSTNESS_SYMBOLS
    )

    if actual_symbols != expected_symbols:
        raise TrendFamilyRobustnessError(
            "Development bars must contain exactly the "
            "frozen symbols. "
            f"Expected: {sorted(expected_symbols)}; "
            f"actual: {sorted(actual_symbols)}."
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
        raise TrendFamilyRobustnessError(
            "Bars must remain within the development "
            "period from 2020-01-02 through 2025-12-31."
        )

    return result


def _prepare_frequency_data(
    source: pd.DataFrame,
) -> dict[
    str,
    tuple[pd.DataFrame, pd.DataFrame],
]:
    """Aggregate bars and rebuild returns at each frequency."""

    prepared: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    for frequency in ROBUSTNESS_FREQUENCIES:
        try:
            aggregated = aggregate_session_bars(
                source,
                frequency,
            )
        except SessionAggregationError as exc:
            raise TrendFamilyRobustnessError(
                "Frequency aggregation failed for "
                f"{frequency}: {exc}"
            ) from exc

        if (
            len(source)
            == EXPECTED_PORTFOLIO_OBSERVATIONS[
                "15min"
            ]
            and len(aggregated)
            != EXPECTED_PORTFOLIO_OBSERVATIONS[
                frequency
            ]
        ):
            raise RuntimeError(
                f"{frequency} aggregation produced "
                "an unexpected canonical row count."
            )

        feature_bundle = build_return_features(
            aggregated,
            expected_symbols=(
                ROBUSTNESS_SYMBOLS
            ),
        )
        prepared[frequency] = (
            aggregated,
            feature_bundle.bars,
        )

    return prepared


def _execute_frozen_strategy(
    frame: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    """Execute one existing frozen strategy engine."""

    if strategy == "trend_ratio":
        bundle = build_trend_ratio_strategy(
            frame,
            parameters=(
                TREND_RATIO_PARAMETERS
            ),
        )
    elif strategy == "ema_macd":
        bundle = build_ema_macd_strategy(
            frame,
            parameters=EMA_MACD_PARAMETERS,
        )
    else:
        raise RuntimeError(
            f"Unexpected frozen strategy: {strategy}."
        )

    observations = (
        bundle.observations.copy(deep=True)
    )
    _validate_position_delay(observations)

    return observations


def _validate_position_delay(
    observations: pd.DataFrame,
) -> None:
    """Confirm positions remain delayed by one observation."""

    required = {
        "symbol",
        "signal",
        "position",
    }
    missing = sorted(
        required.difference(
            observations.columns
        )
    )

    if missing:
        raise TrendFamilyRobustnessError(
            "Strategy observations are missing delay "
            f"columns: {missing}."
        )

    expected = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["signal"]
        .shift(1, fill_value=0)
    )
    actual = pd.to_numeric(
        observations["position"],
        errors="coerce",
    )

    if (
        actual.isna().any()
        or not actual.eq(expected).all()
    ):
        raise TrendFamilyRobustnessError(
            "Strategy positions must preserve the existing "
            "one-observation delay."
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
    """Calculate existing-style eligible-position exposure."""

    required = {
        "position",
        "position_eligible",
    }
    missing = sorted(
        required.difference(
            observations.columns
        )
    )

    if missing:
        raise TrendFamilyRobustnessError(
            "Strategy observations are missing exposure "
            f"columns: {missing}."
        )

    eligible_mask = observations[
        "position_eligible"
    ].astype(bool)
    eligible = observations.loc[
        eligible_mask,
        "position",
    ]
    active_observations = int(
        eligible_mask.sum()
    )
    warmup_observations = int(
        len(observations)
        - active_observations
    )

    if eligible.empty:
        undefined = float("nan")

        return (
            warmup_observations,
            active_observations,
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
    average_exposure = (
        long_exposure + short_exposure
    )

    return (
        warmup_observations,
        active_observations,
        average_exposure,
        long_exposure,
        short_exposure,
        flat_exposure,
    )


def _build_result_record(
    specification: RobustnessRunSpec,
    *,
    frequency_bars: pd.DataFrame,
    feature_bars: pd.DataFrame,
) -> dict[str, object]:
    """Execute and summarize one frozen robustness run."""

    symbol_bars = (
        frequency_bars.loc[
            frequency_bars[
                "symbol"
            ].eq(specification.symbol)
        ]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    symbol_features = (
        feature_bars.loc[
            feature_bars[
                "symbol"
            ].eq(specification.symbol)
        ]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    if symbol_bars.empty or symbol_features.empty:
        raise RuntimeError(
            "A frozen robustness run has no symbol data."
        )

    observations = _execute_frozen_strategy(
        symbol_features,
        strategy=specification.strategy,
    )

    metrics = calculate_performance_metrics(
        observations[
            "net_strategy_return"
        ],
        annualization_factor=(
            specification.annualization_factor
        ),
    )

    if metrics.observations != len(
        observations
    ):
        raise RuntimeError(
            "Performance metrics changed the observation "
            "count."
        )

    (
        warmup_observations,
        active_observations,
        average_exposure,
        long_exposure,
        short_exposure,
        flat_exposure,
    ) = _calculate_exposures(observations)

    return {
        "strategy": specification.strategy,
        "symbol": specification.symbol,
        "frequency": (
            specification.frequency
        ),
        "dataset_id": DEVELOPMENT_DATASET_ID,
        "configuration_id": (
            CONFIGURATION_IDS[
                specification.strategy
            ]
        ),
        "start_timestamp": (
            observations["timestamp"].min()
        ),
        "end_timestamp": (
            observations["timestamp"].max()
        ),
        "sessions": int(
            symbol_bars[
                "session_date"
            ].nunique()
        ),
        "observations": int(
            len(observations)
        ),
        "annualization_factor": (
            specification.annualization_factor
        ),
        "partial_bar_count": int(
            symbol_bars[
                "is_partial_bar"
            ].sum()
        ),
        "warmup_observations": (
            warmup_observations
        ),
        "active_observations": (
            active_observations
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
        "turnover": float(
            observations["turnover"].sum()
        ),
        "average_exposure": (
            average_exposure
        ),
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "flat_exposure": flat_exposure,
        "trade_count": int(
            observations["turnover"]
            .gt(0.0)
            .sum()
        ),
    }


def _validate_metric_values(
    results: pd.DataFrame,
) -> None:
    """Validate finite metrics wherever they are defined."""

    always_finite = (
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "turnover",
    )

    for column in always_finite:
        if not results[
            column
        ].map(math.isfinite).all():
            raise RuntimeError(
                f"{column} must be finite."
            )

    defined_sharpe = results[
        "sharpe_ratio"
    ].dropna()

    if not defined_sharpe.map(
        math.isfinite
    ).all():
        raise RuntimeError(
            "Defined Sharpe ratios must be finite."
        )

    active = results[
        "active_observations"
    ].gt(0)
    exposure_columns = (
        "average_exposure",
        "long_exposure",
        "short_exposure",
        "flat_exposure",
    )

    for column in exposure_columns:
        if not results.loc[
            active,
            column,
        ].map(math.isfinite).all():
            raise RuntimeError(
                f"Defined {column} values must be finite."
            )


def _validate_complete_results(
    results: pd.DataFrame,
) -> None:
    """Validate all frozen Day 10 result invariants."""

    if tuple(results.columns) != (
        REQUIRED_RESULT_COLUMNS
    ):
        raise RuntimeError(
            "Day 10 result schema is incomplete."
        )

    if len(results) != 18:
        raise RuntimeError(
            "Day 10 robustness must retain all 18 runs."
        )

    key_columns = [
        "strategy",
        "symbol",
        "frequency",
    ]

    if results.duplicated(
        key_columns
    ).any():
        raise RuntimeError(
            "Day 10 result keys must be unique."
        )

    if not results["dataset_id"].eq(
        DEVELOPMENT_DATASET_ID
    ).all():
        raise RuntimeError(
            "Day 10 dataset lineage changed."
        )

    actual_keys = list(
        results[
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
        raise RuntimeError(
            "Day 10 results are not in frozen order."
        )

    expected_ids = {
        strategy: CONFIGURATION_IDS[
            strategy
        ]
        for strategy in ROBUSTNESS_STRATEGIES
    }
    observed_ids = (
        results.groupby(
            "strategy",
            observed=True,
            sort=False,
        )["configuration_id"]
        .agg(
            lambda values: tuple(
                values.unique()
            )
        )
        .to_dict()
    )

    if observed_ids != {
        strategy: (
            configuration_id,
        )
        for (
            strategy,
            configuration_id,
        ) in expected_ids.items()
    }:
        raise RuntimeError(
            "Frozen configuration identifiers changed."
        )

    if len(set(expected_ids.values())) != 2:
        raise RuntimeError(
            "Strategy configuration identifiers must be "
            "distinct."
        )

    expected_factors = results[
        "frequency"
    ].map(ANNUALIZATION_FACTORS)

    if not results[
        "annualization_factor"
    ].eq(expected_factors).all():
        raise RuntimeError(
            "Frequency annualization factors changed."
        )

    expected_partial_bars = (
        results["sessions"].where(
            results["frequency"].eq("60min"),
            0,
        )
    )

    if not results[
        "partial_bar_count"
    ].eq(expected_partial_bars).all():
        raise RuntimeError(
            "Partial-bar counts do not match the frozen "
            "frequency policy."
        )

    starts = pd.to_datetime(
        results["start_timestamp"],
        utc=True,
        errors="raise",
    )
    ends = pd.to_datetime(
        results["end_timestamp"],
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
        raise RuntimeError(
            "A result extends outside the development "
            "period."
        )

    if not results[
        "warmup_observations"
    ].add(
        results["active_observations"]
    ).eq(
        results["observations"]
    ).all():
        raise RuntimeError(
            "Warmup and active observations do not "
            "reconcile."
        )

    _validate_metric_values(results)


def run_trend_family_robustness(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Execute the frozen 18-run development robustness matrix."""

    source = _validate_development_bars(
        bars
    )
    prepared = _prepare_frequency_data(
        source
    )
    records: list[dict[str, object]] = []

    for specification in (
        build_robustness_run_matrix()
    ):
        (
            frequency_bars,
            feature_bars,
        ) = prepared[
            specification.frequency
        ]
        records.append(
            _build_result_record(
                specification,
                frequency_bars=frequency_bars,
                feature_bars=feature_bars,
            )
        )

    results = pd.DataFrame.from_records(
        records,
        columns=REQUIRED_RESULT_COLUMNS,
    )
    _validate_complete_results(results)

    return results
