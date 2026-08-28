"""Stylized-fact statistics for Day 05 development data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from systematic_alpha.analysis._validation import (
    make_column_validator,
    make_numeric_cleaner,
)


DEFAULT_QUANTILES: Final[tuple[float, ...]] = (
    0.001,
    0.01,
    0.05,
    0.50,
    0.95,
    0.99,
    0.999,
)

DEFAULT_INTRADAY_LAGS: Final[tuple[int, ...]] = (
    1,
    2,
    4,
    8,
    13,
    25,
)

DEFAULT_DAILY_LAGS: Final[tuple[int, ...]] = (
    5,
    10,
    20,
)

TAIL_THRESHOLDS: Final[tuple[float, ...]] = (
    3.0,
    4.0,
    5.0,
)


class StylizedFactsError(ValueError):
    """Raised when stylized-fact statistics cannot be computed."""


_validate_columns = make_column_validator(
    StylizedFactsError
)

_clean_numeric_series = make_numeric_cleaner(
    StylizedFactsError
)


@dataclass(frozen=True)
class StylizedFactsBundle:
    """Machine-readable stylized-fact output tables."""

    moments: pd.DataFrame
    quantiles: pd.DataFrame
    tail_rates: pd.DataFrame
    intraday_acf: pd.DataFrame
    daily_acf: pd.DataFrame
    daily_ljung_box: pd.DataFrame






def _moment_record(
    values: pd.Series,
    *,
    symbol: str,
    frequency: str,
    return_type: str,
) -> dict[str, float | int | str]:
    """Compute distribution moments and normality evidence."""

    clean = _clean_numeric_series(values)

    sample_mean = float(clean.mean())
    sample_variance = float(clean.var(ddof=1))
    sample_std = float(clean.std(ddof=1))

    if sample_std <= 0.0:
        raise StylizedFactsError(
            f"{symbol} {return_type} has zero variance."
        )

    jarque_bera = stats.jarque_bera(
        clean.to_numpy()
    )

    return {
        "symbol": symbol,
        "frequency": frequency,
        "return_type": return_type,
        "observations": int(len(clean)),
        "mean": sample_mean,
        "median": float(clean.median()),
        "variance": sample_variance,
        "standard_deviation": sample_std,
        "mean_absolute_return": float(
            clean.abs().mean()
        ),
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
        "skewness": float(
            stats.skew(
                clean.to_numpy(),
                bias=False,
            )
        ),
        "excess_kurtosis": float(
            stats.kurtosis(
                clean.to_numpy(),
                fisher=True,
                bias=False,
            )
        ),
        "jarque_bera_statistic": float(
            jarque_bera.statistic
        ),
        "jarque_bera_pvalue": float(
            jarque_bera.pvalue
        ),
    }


def _quantile_records(
    values: pd.Series,
    *,
    symbol: str,
    frequency: str,
    return_type: str,
    probabilities: Sequence[float],
) -> list[dict[str, float | int | str]]:
    """Compute empirical distribution quantiles."""

    clean = _clean_numeric_series(values)

    invalid = [
        probability
        for probability in probabilities
        if not 0.0 <= probability <= 1.0
    ]

    if invalid:
        raise StylizedFactsError(
            "Quantile probabilities must lie in [0, 1]. "
            f"Invalid values: {invalid}"
        )

    quantiles = clean.quantile(
        list(probabilities)
    )

    records = []

    for probability, value in quantiles.items():
        records.append(
            {
                "symbol": symbol,
                "frequency": frequency,
                "return_type": return_type,
                "probability": float(probability),
                "quantile": float(value),
                "observations": int(len(clean)),
            }
        )

    return records


def _tail_records(
    values: pd.Series,
    *,
    symbol: str,
    frequency: str,
    return_type: str,
) -> list[dict[str, float | int | str]]:
    """Compare empirical standardized tails with normal tails."""

    clean = _clean_numeric_series(values)

    sample_mean = float(clean.mean())
    sample_std = float(clean.std(ddof=1))

    if sample_std <= 0.0:
        raise StylizedFactsError(
            f"{symbol} {return_type} has zero variance."
        )

    z_scores = (
        clean - sample_mean
    ) / sample_std

    records = []

    for threshold in TAIL_THRESHOLDS:
        left_count = int(
            z_scores.le(-threshold).sum()
        )

        right_count = int(
            z_scores.ge(threshold).sum()
        )

        two_sided_count = left_count + right_count

        empirical_left = (
            left_count / len(clean)
        )

        empirical_right = (
            right_count / len(clean)
        )

        empirical_two_sided = (
            two_sided_count / len(clean)
        )

        normal_one_sided = float(
            stats.norm.sf(threshold)
        )

        normal_two_sided = (
            2.0 * normal_one_sided
        )

        ratio = (
            empirical_two_sided
            / normal_two_sided
            if normal_two_sided > 0.0
            else np.nan
        )

        records.append(
            {
                "symbol": symbol,
                "frequency": frequency,
                "return_type": return_type,
                "threshold_sigma": threshold,
                "observations": int(len(clean)),
                "left_tail_count": left_count,
                "right_tail_count": right_count,
                "two_sided_count": two_sided_count,
                "empirical_left_rate": empirical_left,
                "empirical_right_rate": empirical_right,
                "empirical_two_sided_rate": (
                    empirical_two_sided
                ),
                "normal_two_sided_rate": (
                    normal_two_sided
                ),
                "empirical_to_normal_ratio": ratio,
            }
        )

    return records


def build_distribution_tables(
    bar_features: pd.DataFrame,
    session_features: pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build moments, quantiles and standardized-tail tables."""

    _validate_columns(
        bar_features,
        (
            "symbol",
            "intraday_log_return",
        ),
        context="Bar features",
    )

    _validate_columns(
        session_features,
        (
            "symbol",
            "overnight_log_return",
            "regular_session_log_return",
            "close_to_close_log_return",
        ),
        context="Session features",
    )

    specifications = (
        (
            bar_features,
            "15min_within_session",
            "intraday_log_return",
        ),
        (
            session_features,
            "daily_session",
            "overnight_log_return",
        ),
        (
            session_features,
            "daily_session",
            "regular_session_log_return",
        ),
        (
            session_features,
            "daily_session",
            "close_to_close_log_return",
        ),
    )

    moment_records = []
    quantile_records = []
    tail_records = []

    for frame, frequency, return_column in specifications:
        for symbol, group in frame.groupby(
            "symbol",
            observed=True,
            sort=True,
        ):
            values = group[return_column]

            moment_records.append(
                _moment_record(
                    values,
                    symbol=str(symbol),
                    frequency=frequency,
                    return_type=return_column,
                )
            )

            quantile_records.extend(
                _quantile_records(
                    values,
                    symbol=str(symbol),
                    frequency=frequency,
                    return_type=return_column,
                    probabilities=quantiles,
                )
            )

            tail_records.extend(
                _tail_records(
                    values,
                    symbol=str(symbol),
                    frequency=frequency,
                    return_type=return_column,
                )
            )

    return (
        pd.DataFrame(moment_records),
        pd.DataFrame(quantile_records),
        pd.DataFrame(tail_records),
    )


def _transform_returns(
    values: pd.Series,
    transformation: str,
) -> pd.Series:
    """Apply a raw, absolute or squared-return transform."""

    if transformation == "raw":
        return values

    if transformation == "absolute":
        return values.abs()

    if transformation == "squared":
        return values.pow(2)

    raise StylizedFactsError(
        "Unsupported return transformation: "
        f"{transformation}"
    )


def build_intraday_acf_table(
    bar_features: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_INTRADAY_LAGS,
) -> pd.DataFrame:
    """Calculate pooled within-session intraday autocorrelations."""

    _validate_columns(
        bar_features,
        (
            "symbol",
            "session_date",
            "timestamp",
            "intraday_log_return",
        ),
        context="Intraday ACF input",
    )

    invalid_lags = [
        lag
        for lag in lags
        if not isinstance(lag, int) or lag <= 0
    ]

    if invalid_lags:
        raise StylizedFactsError(
            "ACF lags must be positive integers. "
            f"Invalid lags: {invalid_lags}"
        )

    frame = bar_features.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="raise",
    )

    frame = frame.sort_values(
        ["symbol", "session_date", "timestamp"],
        kind="stable",
    )

    records = []

    for symbol, symbol_frame in frame.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        for transformation in (
            "raw",
            "absolute",
            "squared",
        ):
            transformed = _transform_returns(
                pd.to_numeric(
                    symbol_frame[
                        "intraday_log_return"
                    ],
                    errors="coerce",
                ),
                transformation,
            )

            working = symbol_frame[
                [
                    "session_date",
                    "timestamp",
                ]
            ].copy()

            working["value"] = transformed

            for lag in lags:
                working["lagged_value"] = (
                    working.groupby(
                        "session_date",
                        observed=True,
                        sort=False,
                    )["value"]
                    .shift(lag)
                )

                valid = working[
                    ["value", "lagged_value"]
                ].replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).dropna()

                correlation = (
                    float(
                        valid["value"].corr(
                            valid["lagged_value"]
                        )
                    )
                    if len(valid) >= 3
                    else np.nan
                )

                records.append(
                    {
                        "symbol": str(symbol),
                        "frequency": (
                            "15min_within_session"
                        ),
                        "return_type": (
                            "intraday_log_return"
                        ),
                        "transformation": transformation,
                        "lag": int(lag),
                        "pair_count": int(len(valid)),
                        "autocorrelation": correlation,
                    }
                )

    return pd.DataFrame(records)


def build_daily_acf_table(
    session_features: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_DAILY_LAGS,
) -> pd.DataFrame:
    """Calculate daily return, absolute-return and squared-return ACFs."""

    _validate_columns(
        session_features,
        (
            "symbol",
            "session_date",
            "close_to_close_log_return",
        ),
        context="Daily ACF input",
    )

    records = []

    frame = session_features.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    )

    for symbol, group in frame.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        base = pd.to_numeric(
            group["close_to_close_log_return"],
            errors="coerce",
        )

        for transformation in (
            "raw",
            "absolute",
            "squared",
        ):
            values = _transform_returns(
                base,
                transformation,
            )

            for lag in lags:
                valid = pd.concat(
                    [
                        values.rename("value"),
                        values.shift(lag).rename(
                            "lagged_value"
                        ),
                    ],
                    axis=1,
                ).replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).dropna()

                correlation = (
                    float(
                        valid["value"].corr(
                            valid["lagged_value"]
                        )
                    )
                    if len(valid) >= 3
                    else np.nan
                )

                records.append(
                    {
                        "symbol": str(symbol),
                        "frequency": "daily_session",
                        "return_type": (
                            "close_to_close_log_return"
                        ),
                        "transformation": transformation,
                        "lag": int(lag),
                        "pair_count": int(len(valid)),
                        "autocorrelation": correlation,
                    }
                )

    return pd.DataFrame(records)


def build_daily_ljung_box_table(
    session_features: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_DAILY_LAGS,
) -> pd.DataFrame:
    """Apply Ljung–Box tests to daily raw, absolute and squared returns."""

    _validate_columns(
        session_features,
        (
            "symbol",
            "session_date",
            "close_to_close_log_return",
        ),
        context="Ljung-Box input",
    )

    records = []

    frame = session_features.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    )

    for symbol, group in frame.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        base = _clean_numeric_series(
            group["close_to_close_log_return"],
            minimum_observations=max(lags) + 5,
        )

        for transformation in (
            "raw",
            "absolute",
            "squared",
        ):
            values = _transform_returns(
                base,
                transformation,
            )

            result = acorr_ljungbox(
                values.to_numpy(),
                lags=list(lags),
                return_df=True,
                model_df=0,
            )

            for lag, row in result.iterrows():
                records.append(
                    {
                        "symbol": str(symbol),
                        "frequency": "daily_session",
                        "return_type": (
                            "close_to_close_log_return"
                        ),
                        "transformation": transformation,
                        "lag": int(lag),
                        "observations": int(
                            len(values)
                        ),
                        "ljung_box_statistic": float(
                            row["lb_stat"]
                        ),
                        "ljung_box_pvalue": float(
                            row["lb_pvalue"]
                        ),
                        "reject_at_5pct": bool(
                            row["lb_pvalue"] < 0.05
                        ),
                    }
                )

    return pd.DataFrame(records)


def build_stylized_facts(
    bar_features: pd.DataFrame,
    session_features: pd.DataFrame,
) -> StylizedFactsBundle:
    """Build all Day 05 stylized-fact evidence tables."""

    moments, quantiles, tail_rates = (
        build_distribution_tables(
            bar_features,
            session_features,
        )
    )

    intraday_acf = build_intraday_acf_table(
        bar_features
    )

    daily_acf = build_daily_acf_table(
        session_features
    )

    daily_ljung_box = (
        build_daily_ljung_box_table(
            session_features
        )
    )

    return StylizedFactsBundle(
        moments=moments,
        quantiles=quantiles,
        tail_rates=tail_rates,
        intraday_acf=intraday_acf,
        daily_acf=daily_acf,
        daily_ljung_box=daily_ljung_box,
    )
