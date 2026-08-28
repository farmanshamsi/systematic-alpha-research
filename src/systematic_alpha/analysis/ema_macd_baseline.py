"""Development-only analysis for the frozen Day 8 EMA/MACD baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    CostBreakEvenResult,
    HoldingDiagnostics,
    calculate_cost_break_even,
    calculate_holding_diagnostics,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdBundle,
    EmaMacdError,
    EmaMacdParameters,
    build_ema_macd_strategy,
)


DAY08_ANNUALIZATION_FACTOR: Final[int] = 252 * 26
DAY08_FORWARD_HORIZONS: Final[tuple[int, ...]] = (1, 4, 8, 16)
DAY08_SIGNAL_BUCKET_COUNT: Final[int] = 5

PERFORMANCE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "series",
    "observations",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
)

EMA_MACD_FORWARD_SAMPLE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timestamp",
    "first_forward_timestamp",
    "forward_end_timestamp",
    "horizon_bars",
    "continuous_signal",
    "forward_return",
)

EMA_MACD_SIGNAL_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
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

EMA_MACD_SIGNAL_BUCKET_COLUMNS: Final[tuple[str, ...]] = (
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


class EmaMacdBaselineError(ValueError):
    """Raised when the Day 8 baseline cannot be analysed safely."""


@dataclass(frozen=True)
class EmaMacdBaselineAnalysis:
    """Compact development-only evidence for the frozen EMA/MACD baseline."""

    strategy_bundle: EmaMacdBundle
    performance_summary: pd.DataFrame
    holding_diagnostics: pd.DataFrame
    cost_break_even: pd.DataFrame
    signal_validation: pd.DataFrame
    signal_buckets: pd.DataFrame


def _validate_positive_real(
    value: Real,
    *,
    name: str,
) -> float:
    """Validate a finite strictly positive real number."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise EmaMacdBaselineError(
            f"{name} must be a finite real number."
        )

    normalized = float(value)

    if not math.isfinite(normalized) or normalized <= 0.0:
        raise EmaMacdBaselineError(
            f"{name} must be finite and strictly positive."
        )

    return normalized


def _validate_horizons(
    horizons: Sequence[int],
) -> tuple[int, ...]:
    """Validate deterministic strictly increasing forward horizons."""

    normalized = tuple(horizons)

    if not normalized:
        raise EmaMacdBaselineError(
            "Forward horizons must not be empty."
        )

    for horizon in normalized:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise EmaMacdBaselineError(
                "Forward horizons must be integers."
            )

        if horizon <= 0:
            raise EmaMacdBaselineError(
                "Forward horizons must be strictly positive."
            )

    if len(set(normalized)) != len(normalized):
        raise EmaMacdBaselineError(
            "Forward horizons must not contain duplicates."
        )

    if any(
        current >= following
        for current, following in zip(
            normalized,
            normalized[1:],
        )
    ):
        raise EmaMacdBaselineError(
            "Forward horizons must be strictly increasing."
        )

    return normalized


def _validate_signal_frame(
    observations: pd.DataFrame,
    *,
    price_column: str,
    signal_column: str,
) -> pd.DataFrame:
    """Validate observations for EMA/MACD forward-signal analysis."""

    if not isinstance(observations, pd.DataFrame):
        raise EmaMacdBaselineError(
            "observations must be a pandas DataFrame."
        )

    if observations.empty:
        raise EmaMacdBaselineError(
            "observations must not be empty."
        )

    required_columns = (
        "timestamp",
        "symbol",
        price_column,
        signal_column,
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in observations.columns
    ]

    if missing_columns:
        raise EmaMacdBaselineError(
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
        raise EmaMacdBaselineError(
            "Signal observations contain malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise EmaMacdBaselineError(
            "Signal observations contain missing timestamps."
        )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result["symbol"].isna().any() or result["symbol"].eq("").any():
        raise EmaMacdBaselineError(
            "Signal observations contain invalid symbols."
        )

    try:
        result[price_column] = pd.to_numeric(
            result[price_column],
            errors="raise",
        ).astype(float)
        result[signal_column] = pd.to_numeric(
            result[signal_column],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise EmaMacdBaselineError(
            f"{price_column} and {signal_column} must be numeric."
        ) from exc

    prices = result[price_column]

    if prices.isna().any():
        raise EmaMacdBaselineError(
            f"{price_column} must not contain missing values."
        )

    if not prices.map(math.isfinite).all():
        raise EmaMacdBaselineError(
            f"{price_column} must contain finite values."
        )

    if prices.le(0.0).any():
        raise EmaMacdBaselineError(
            f"{price_column} must contain positive values."
        )

    available_signal = result[signal_column].dropna()

    if not available_signal.map(math.isfinite).all():
        raise EmaMacdBaselineError(
            f"Non-missing {signal_column} values must be finite."
        )

    result = result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    if result.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise EmaMacdBaselineError(
            "Signal observations contain duplicate symbol-timestamp rows."
        )

    return result


def build_ema_macd_forward_signal_sample(
    observations: pd.DataFrame,
    *,
    horizon_bars: int,
    price_column: str = "close",
    signal_column: str = "normalized_macd_histogram",
) -> pd.DataFrame:
    """Build one EMA/MACD signal and forward-return sample.

    At signal time t, the h-bar return is P[t+h] / P[t] - 1. Its first
    constituent one-bar return therefore begins at t+1.
    """

    if isinstance(horizon_bars, bool) or not isinstance(
        horizon_bars,
        int,
    ):
        raise EmaMacdBaselineError(
            "horizon_bars must be an integer."
        )

    if horizon_bars <= 0:
        raise EmaMacdBaselineError(
            "horizon_bars must be strictly positive."
        )

    validated = _validate_signal_frame(
        observations,
        price_column=price_column,
        signal_column=signal_column,
    )

    grouped = validated.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    validated["continuous_signal"] = validated[signal_column]
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
                "Forward returns begin at or before the signal timestamp."
            )

        if not (
            eligible["forward_end_timestamp"]
            >= eligible["first_forward_timestamp"]
        ).all():
            raise RuntimeError(
                "Forward-return end precedes its first constituent return."
            )

    return eligible.loc[
        :,
        EMA_MACD_FORWARD_SAMPLE_COLUMNS,
    ].reset_index(drop=True)


def _safe_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    method: str,
) -> float:
    """Calculate one correlation or return NaN when undefined."""

    if len(left) < 2:
        return float("nan")

    if left.nunique(dropna=True) < 2:
        return float("nan")

    if right.nunique(dropna=True) < 2:
        return float("nan")

    value = left.corr(right, method=method)

    return float(value) if pd.notna(value) else float("nan")


def _assign_equal_frequency_buckets(
    sample: pd.DataFrame,
    *,
    requested_bucket_count: int,
) -> pd.Series:
    """Assign deterministic equal-frequency buckets from signal only."""

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


def build_ema_macd_signal_validation(
    observations: pd.DataFrame,
    *,
    horizons: Sequence[int] = DAY08_FORWARD_HORIZONS,
    signal_bucket_count: int = DAY08_SIGNAL_BUCKET_COUNT,
    price_column: str = "close",
    signal_column: str = "normalized_macd_histogram",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build IC and bucket diagnostics for the normalised histogram."""

    validated_horizons = _validate_horizons(horizons)

    if isinstance(signal_bucket_count, bool) or not isinstance(
        signal_bucket_count,
        int,
    ):
        raise EmaMacdBaselineError(
            "signal_bucket_count must be an integer."
        )

    if signal_bucket_count < 2:
        raise EmaMacdBaselineError(
            "signal_bucket_count must be at least two."
        )

    summary_records: list[dict[str, object]] = []
    bucket_records: list[dict[str, object]] = []

    for horizon in validated_horizons:
        forward_sample = build_ema_macd_forward_signal_sample(
            observations,
            horizon_bars=horizon,
            price_column=price_column,
            signal_column=signal_column,
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

            pearson_ic = _safe_correlation(
                sample["continuous_signal"],
                sample["forward_return"],
                method="pearson",
            )
            spearman_ic = _safe_correlation(
                sample["continuous_signal"],
                sample["forward_return"],
                method="spearman",
            )

            sample["signal_bucket"] = (
                _assign_equal_frequency_buckets(
                    sample,
                    requested_bucket_count=signal_bucket_count,
                )
            )

            grouped_buckets = (
                sample.groupby(
                    "signal_bucket",
                    observed=True,
                    sort=True,
                )
                .agg(
                    observations=("forward_return", "size"),
                    signal_minimum=("continuous_signal", "min"),
                    signal_maximum=("continuous_signal", "max"),
                    signal_mean=("continuous_signal", "mean"),
                    mean_forward_return=("forward_return", "mean"),
                    median_forward_return=("forward_return", "median"),
                )
                .reset_index()
            )

            actual_bucket_count = int(len(grouped_buckets))

            if actual_bucket_count >= 2:
                bucket_numbers = grouped_buckets[
                    "signal_bucket"
                ].astype(float)
                bucket_means = grouped_buckets[
                    "mean_forward_return"
                ].astype(float)

                bucket_monotonicity = _safe_correlation(
                    bucket_numbers,
                    bucket_means,
                    method="spearman",
                )
                adjacent_increasing = float(
                    bucket_means.diff().dropna().gt(0.0).mean()
                )
            else:
                bucket_monotonicity = float("nan")
                adjacent_increasing = float("nan")

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
                        adjacent_increasing
                    ),
                }
            )

            for row in grouped_buckets.itertuples(index=False):
                bucket_records.append(
                    {
                        "symbol": str(symbol),
                        "horizon_bars": int(horizon),
                        "signal_bucket": int(row.signal_bucket),
                        "observations": int(row.observations),
                        "signal_minimum": float(row.signal_minimum),
                        "signal_maximum": float(row.signal_maximum),
                        "signal_mean": float(row.signal_mean),
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
        columns=EMA_MACD_SIGNAL_SUMMARY_COLUMNS,
    )
    buckets = pd.DataFrame.from_records(
        bucket_records,
        columns=EMA_MACD_SIGNAL_BUCKET_COLUMNS,
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


def _prepare_performance_returns(
    observations: pd.DataFrame,
    *,
    return_column: str,
) -> pd.DataFrame:
    """Prepare complete return series for the performance engine.

    The first close-to-close buy-and-hold return may be missing because
    no preceding close exists. No other missing return is permitted.
    Strategy gross and net returns must already be complete.
    """

    required_columns = (
        return_column,
        "gross_strategy_return",
        "net_strategy_return",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in observations.columns
    ]

    if missing_columns:
        raise EmaMacdBaselineError(
            "Performance observations are missing required columns: "
            f"{missing_columns}."
        )

    performance = pd.DataFrame(
        {
            "buy_and_hold": observations[
                return_column
            ].copy(),
            "ema_macd_gross": observations[
                "gross_strategy_return"
            ].copy(),
            "ema_macd_net": observations[
                "net_strategy_return"
            ].copy(),
        },
        index=observations.index.copy(),
    )

    buy_and_hold_missing = np.flatnonzero(
        performance["buy_and_hold"]
        .isna()
        .to_numpy()
    )

    if len(buy_and_hold_missing):
        if not np.array_equal(
            buy_and_hold_missing,
            np.array([0]),
        ):
            raise EmaMacdBaselineError(
                "Buy-and-hold returns may be missing only for "
                "the first observation."
            )

        performance.iloc[
            0,
            performance.columns.get_loc("buy_and_hold"),
        ] = 0.0

    strategy_columns = (
        "ema_macd_gross",
        "ema_macd_net",
    )

    if performance.loc[:, strategy_columns].isna().any().any():
        raise EmaMacdBaselineError(
            "EMA/MACD gross and net returns must not contain "
            "missing observations."
        )

    return performance


def _performance_record(
    *,
    symbol: str,
    series_name: str,
    observations: int,
    metrics: PerformanceMetrics,
) -> dict[str, object]:
    """Convert one existing performance object to a compact record."""

    return {
        "symbol": symbol,
        "series": series_name,
        "observations": observations,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
    }


def analyse_ema_macd_baseline(
    frame: pd.DataFrame,
    *,
    parameters: EmaMacdParameters = EmaMacdParameters(),
    annualization_factor: Real = DAY08_ANNUALIZATION_FACTOR,
    session_column: str = "session_date",
    signal_horizons: Sequence[int] = DAY08_FORWARD_HORIZONS,
) -> EmaMacdBaselineAnalysis:
    """Analyse the frozen EMA/MACD baseline on completed observations."""

    normalized_annualization = _validate_positive_real(
        annualization_factor,
        name="annualization_factor",
    )

    if not isinstance(frame, pd.DataFrame):
        raise EmaMacdBaselineError(
            "frame must be a pandas DataFrame."
        )

    if session_column not in frame.columns:
        raise EmaMacdBaselineError(
            f"frame must contain {session_column!r} for "
            "session-crossing holding diagnostics."
        )

    try:
        strategy_bundle = build_ema_macd_strategy(
            frame,
            parameters=parameters,
        )
    except EmaMacdError as exc:
        if (
            "Missing returns are permitted only for the first "
            "observation"
        ) in str(exc):
            raise EmaMacdBaselineError(
                "Buy-and-hold returns may be missing only for "
                "the first observation."
            ) from exc

        raise
    observations = strategy_bundle.observations

    symbols = observations["symbol"].drop_duplicates().tolist()

    if len(symbols) != 1:
        raise EmaMacdBaselineError(
            "Day 8 baseline analysis requires exactly one symbol."
        )

    symbol = str(symbols[0])

    performance_returns = _prepare_performance_returns(
        observations,
        return_column=parameters.return_column,
    )

    buy_and_hold_metrics = calculate_performance_metrics(
        performance_returns["buy_and_hold"],
        annualization_factor=normalized_annualization,
    )
    gross_metrics = calculate_performance_metrics(
        performance_returns["ema_macd_gross"],
        annualization_factor=normalized_annualization,
    )
    net_metrics = calculate_performance_metrics(
        performance_returns["ema_macd_net"],
        annualization_factor=normalized_annualization,
    )

    performance_summary = pd.DataFrame.from_records(
        [
            _performance_record(
                symbol=symbol,
                series_name="buy_and_hold",
                observations=len(observations),
                metrics=buy_and_hold_metrics,
            ),
            _performance_record(
                symbol=symbol,
                series_name="ema_macd_gross",
                observations=len(observations),
                metrics=gross_metrics,
            ),
            _performance_record(
                symbol=symbol,
                series_name="ema_macd_net",
                observations=len(observations),
                metrics=net_metrics,
            ),
        ],
        columns=PERFORMANCE_SUMMARY_COLUMNS,
    )

    holding: HoldingDiagnostics = calculate_holding_diagnostics(
        observations["position"],
        session_labels=observations[session_column],
    )
    holding_diagnostics = pd.DataFrame.from_records(
        [
            {
                "symbol": symbol,
                **asdict(holding),
            }
        ]
    )

    break_even: CostBreakEvenResult = calculate_cost_break_even(
        observations["gross_strategy_return"],
        observations["turnover"],
    )
    cost_break_even = pd.DataFrame.from_records(
        [
            {
                "symbol": symbol,
                **asdict(break_even),
            }
        ]
    )

    signal_validation, signal_buckets = (
        build_ema_macd_signal_validation(
            observations,
            horizons=signal_horizons,
            price_column=parameters.price_column,
            signal_column="normalized_macd_histogram",
        )
    )

    return EmaMacdBaselineAnalysis(
        strategy_bundle=strategy_bundle,
        performance_summary=performance_summary,
        holding_diagnostics=holding_diagnostics,
        cost_break_even=cost_break_even,
        signal_validation=signal_validation,
        signal_buckets=signal_buckets,
    )
