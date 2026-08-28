"""Realized volatility and intraday seasonality for Day 05."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import (
    make_column_validator,
)


TRADING_DAYS_PER_YEAR = 252


class VolatilitySeasonalityError(ValueError):
    """Raised when volatility evidence cannot be constructed."""


_validate_columns = make_column_validator(
    VolatilitySeasonalityError
)


@dataclass(frozen=True)
class VolatilitySeasonalityBundle:
    """Daily volatility and intraday seasonality tables."""

    daily_volatility: pd.DataFrame
    intraday_seasonality: pd.DataFrame




def _prepare_bar_returns(
    bar_features: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one regular-session return piece per bar."""

    _validate_columns(
        bar_features,
        (
            "timestamp",
            "symbol",
            "session_date",
            "local_time",
            "bar_number",
            "bars_in_session",
            "is_session_open_bar",
            "is_session_close_bar",
            "intraday_log_return",
            "bar_open_to_close_log_return",
            "high_low_log_range",
            "volume",
            "trade_count",
        ),
        context="Bar return features",
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
    ).reset_index(drop=True)

    frame["regular_session_return_piece"] = np.where(
        frame["is_session_open_bar"],
        pd.to_numeric(
            frame["bar_open_to_close_log_return"],
            errors="coerce",
        ),
        pd.to_numeric(
            frame["intraday_log_return"],
            errors="coerce",
        ),
    )

    frame["regular_session_return_piece"] = (
        frame["regular_session_return_piece"]
        .replace([np.inf, -np.inf], np.nan)
    )

    missing_returns = int(
        frame["regular_session_return_piece"]
        .isna()
        .sum()
    )

    if missing_returns:
        raise VolatilitySeasonalityError(
            "Regular-session return construction "
            f"produced {missing_returns} missing bars."
        )

    frame["absolute_return_piece"] = (
        frame["regular_session_return_piece"].abs()
    )

    frame["squared_return_piece"] = (
        frame["regular_session_return_piece"].pow(2)
    )

    session_group = frame.groupby(
        ["symbol", "session_date"],
        observed=True,
        sort=False,
    )

    frame["previous_absolute_return"] = (
        session_group[
            "absolute_return_piece"
        ].shift(1)
    )

    frame["bipower_product"] = (
        frame["absolute_return_piece"]
        * frame["previous_absolute_return"]
    )

    return frame


def build_daily_volatility_table(
    bar_features: pd.DataFrame,
    session_features: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (21, 63),
) -> pd.DataFrame:
    """Build daily realized and range-based volatility evidence."""

    _validate_columns(
        session_features,
        (
            "symbol",
            "session_date",
            "session_open",
            "session_high",
            "session_low",
            "session_close",
            "overnight_log_return",
            "regular_session_log_return",
            "close_to_close_log_return",
        ),
        context="Session return features",
    )

    invalid_windows = [
        window
        for window in rolling_windows
        if not isinstance(window, int) or window <= 1
    ]

    if invalid_windows:
        raise VolatilitySeasonalityError(
            "Rolling windows must be integers greater than one. "
            f"Invalid windows: {invalid_windows}"
        )

    bars = _prepare_bar_returns(bar_features)

    grouped = bars.groupby(
        ["symbol", "session_date"],
        observed=True,
        sort=True,
    )

    daily = grouped.agg(
        bars=("timestamp", "size"),
        first_timestamp=("timestamp", "first"),
        last_timestamp=("timestamp", "last"),
        regular_session_log_return_sum=(
            "regular_session_return_piece",
            "sum",
        ),
        realized_absolute_variation=(
            "absolute_return_piece",
            "sum",
        ),
        realized_variance_regular=(
            "squared_return_piece",
            "sum",
        ),
        bipower_product_sum=(
            "bipower_product",
            "sum",
        ),
        maximum_absolute_bar_return=(
            "absolute_return_piece",
            "max",
        ),
        session_volume=("volume", "sum"),
        session_trade_count=("trade_count", "sum"),
    ).reset_index()

    daily["bipower_variation"] = (
        np.pi
        / 2.0
        * daily["bipower_product_sum"]
    )

    daily["jump_variation_proxy"] = (
        daily["realized_variance_regular"]
        - daily["bipower_variation"]
    ).clip(lower=0.0)

    daily["jump_variation_share"] = np.where(
        daily["realized_variance_regular"] > 0.0,
        (
            daily["jump_variation_proxy"]
            / daily["realized_variance_regular"]
        ),
        np.nan,
    )

    sessions = session_features.copy()

    sessions = sessions.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    )

    daily = daily.merge(
        sessions[
            [
                "symbol",
                "session_date",
                "session_open",
                "session_high",
                "session_low",
                "session_close",
                "overnight_log_return",
                "regular_session_log_return",
                "close_to_close_log_return",
            ]
        ],
        on=["symbol", "session_date"],
        how="inner",
        validate="one_to_one",
    )

    daily["regular_return_reconciliation_error"] = (
        daily["regular_session_log_return_sum"]
        - daily["regular_session_log_return"]
    )

    daily["overnight_variance"] = (
        daily["overnight_log_return"].pow(2)
    )

    daily["total_realized_variance"] = (
        daily["realized_variance_regular"]
        + daily["overnight_variance"].fillna(0.0)
    )

    daily["overnight_variance_share"] = np.where(
        daily["total_realized_variance"] > 0.0,
        (
            daily["overnight_variance"]
            / daily["total_realized_variance"]
        ),
        np.nan,
    )

    daily["annualized_regular_realized_volatility"] = (
        np.sqrt(
            TRADING_DAYS_PER_YEAR
            * daily["realized_variance_regular"]
        )
    )

    daily["annualized_total_realized_volatility"] = (
        np.sqrt(
            TRADING_DAYS_PER_YEAR
            * daily["total_realized_variance"]
        )
    )

    log_high_low = np.log(
        daily["session_high"]
        / daily["session_low"]
    )

    daily["parkinson_variance"] = (
        log_high_low.pow(2)
        / (4.0 * np.log(2.0))
    )

    daily["annualized_parkinson_volatility"] = (
        np.sqrt(
            TRADING_DAYS_PER_YEAR
            * daily["parkinson_variance"]
        )
    )

    for window in rolling_windows:
        daily[
            f"rolling_total_realized_volatility_{window}d"
        ] = (
            daily.groupby(
                "symbol",
                observed=True,
                sort=False,
            )["total_realized_variance"]
            .transform(
                lambda values: np.sqrt(
                    TRADING_DAYS_PER_YEAR
                    * values.rolling(
                        window=window,
                        min_periods=window,
                    ).mean()
                )
            )
        )

        daily[
            f"rolling_close_to_close_volatility_{window}d"
        ] = (
            daily.groupby(
                "symbol",
                observed=True,
                sort=False,
            )["close_to_close_log_return"]
            .transform(
                lambda values: (
                    values.rolling(
                        window=window,
                        min_periods=window,
                    ).std(ddof=1)
                    * np.sqrt(
                        TRADING_DAYS_PER_YEAR
                    )
                )
            )
        )

    return daily.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    ).reset_index(drop=True)


def build_intraday_seasonality_table(
    bar_features: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate volatility and activity by intraday bar position."""

    bars = _prepare_bar_returns(bar_features)

    session_group = bars.groupby(
        ["symbol", "session_date"],
        observed=True,
        sort=False,
    )

    bars["session_total_volume"] = (
        session_group["volume"].transform("sum")
    )

    bars["session_total_trade_count"] = (
        session_group[
            "trade_count"
        ].transform("sum")
    )

    bars["session_total_absolute_return"] = (
        session_group[
            "absolute_return_piece"
        ].transform("sum")
    )

    bars["session_total_squared_return"] = (
        session_group[
            "squared_return_piece"
        ].transform("sum")
    )

    bars["volume_share"] = np.where(
        bars["session_total_volume"] > 0.0,
        bars["volume"] / bars["session_total_volume"],
        np.nan,
    )

    bars["trade_count_share"] = np.where(
        bars["session_total_trade_count"] > 0.0,
        (
            bars["trade_count"]
            / bars["session_total_trade_count"]
        ),
        np.nan,
    )

    bars["absolute_return_share"] = np.where(
        bars["session_total_absolute_return"] > 0.0,
        (
            bars["absolute_return_piece"]
            / bars["session_total_absolute_return"]
        ),
        np.nan,
    )

    bars["squared_return_share"] = np.where(
        bars["session_total_squared_return"] > 0.0,
        (
            bars["squared_return_piece"]
            / bars["session_total_squared_return"]
        ),
        np.nan,
    )

    seasonality = (
        bars.groupby(
            [
                "symbol",
                "bar_number",
                "local_time",
            ],
            observed=True,
            sort=True,
        )
        .agg(
            observations=("timestamp", "size"),
            sessions=("session_date", "nunique"),
            mean_return=(
                "regular_session_return_piece",
                "mean",
            ),
            standard_deviation=(
                "regular_session_return_piece",
                "std",
            ),
            mean_absolute_return=(
                "absolute_return_piece",
                "mean",
            ),
            median_absolute_return=(
                "absolute_return_piece",
                "median",
            ),
            mean_squared_return=(
                "squared_return_piece",
                "mean",
            ),
            mean_high_low_log_range=(
                "high_low_log_range",
                "mean",
            ),
            mean_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            mean_trade_count=("trade_count", "mean"),
            mean_volume_share=("volume_share", "mean"),
            mean_trade_count_share=(
                "trade_count_share",
                "mean",
            ),
            mean_absolute_return_share=(
                "absolute_return_share",
                "mean",
            ),
            mean_squared_return_share=(
                "squared_return_share",
                "mean",
            ),
            opening_bar_rate=(
                "is_session_open_bar",
                "mean",
            ),
            closing_bar_rate=(
                "is_session_close_bar",
                "mean",
            ),
            average_session_length=(
                "bars_in_session",
                "mean",
            ),
        )
        .reset_index()
    )

    return seasonality


def build_volatility_seasonality(
    bar_features: pd.DataFrame,
    session_features: pd.DataFrame,
) -> VolatilitySeasonalityBundle:
    """Build all D05-T02C analytical evidence."""

    return VolatilitySeasonalityBundle(
        daily_volatility=build_daily_volatility_table(
            bar_features,
            session_features,
        ),
        intraday_seasonality=(
            build_intraday_seasonality_table(
                bar_features
            )
        ),
    )
