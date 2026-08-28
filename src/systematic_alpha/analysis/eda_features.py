"""Return construction and data-quality features for Day 05 EDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import pandas as pd


LOCAL_TIMEZONE: Final[str] = "America/New_York"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
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
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)

FATAL_QUALITY_ISSUES: Final[frozenset[str]] = frozenset(
    {
        "duplicate_symbol_timestamp",
        "missing_open",
        "missing_high",
        "missing_low",
        "missing_close",
        "nonpositive_open",
        "nonpositive_high",
        "nonpositive_low",
        "nonpositive_close",
        "high_below_ohlc_max",
        "low_above_ohlc_min",
        "missing_volume",
        "negative_volume",
        "missing_trade_count",
        "negative_trade_count",
    }
)


class EdaFeatureError(ValueError):
    """Raised when EDA features cannot be constructed safely."""


@dataclass(frozen=True)
class EdaFeatureBundle:
    """Bar-level and session-level analytical features."""

    bars: pd.DataFrame
    sessions: pd.DataFrame


@dataclass(frozen=True)
class DataQualityBundle:
    """Machine-readable data-quality evidence tables."""

    integrity: pd.DataFrame
    missingness: pd.DataFrame
    symbols: pd.DataFrame
    sessions: pd.DataFrame


def _require_columns(frame: pd.DataFrame) -> None:
    """Require the canonical bar schema."""

    missing = set(REQUIRED_COLUMNS).difference(frame.columns)

    if missing:
        raise EdaFeatureError(
            "Canonical bars are missing required columns: "
            f"{sorted(missing)}"
        )


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical bars for deterministic analysis."""

    _require_columns(frame)

    bars = frame.copy()

    bars["timestamp"] = pd.to_datetime(
        bars["timestamp"],
        utc=True,
        errors="raise",
    )

    bars["symbol"] = (
        bars["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for column in NUMERIC_COLUMNS:
        bars[column] = pd.to_numeric(
            bars[column],
            errors="coerce",
        )

    bars = bars.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    bars["local_timestamp"] = (
        bars["timestamp"]
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    bars["session_date"] = (
        bars["local_timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    bars["local_time"] = (
        bars["local_timestamp"]
        .dt.strftime("%H:%M:%S")
    )

    return bars


def build_data_quality_tables(
    frame: pd.DataFrame,
) -> DataQualityBundle:
    """Build data-quality tables without hiding invalid observations."""

    bars = normalize_bars(frame)

    duplicate_mask = bars.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    )

    ohlc_max = bars[
        ["open", "high", "low", "close"]
    ].max(axis=1)

    ohlc_min = bars[
        ["open", "high", "low", "close"]
    ].min(axis=1)

    tolerance = 1.0e-10

    vwap_outside_range = (
        bars["vwap"].notna()
        & (
            (bars["vwap"] < bars["low"] - tolerance)
            | (
                bars["vwap"]
                > bars["high"] + tolerance
            )
        )
    )

    issue_counts = {
        "duplicate_symbol_timestamp": int(
            duplicate_mask.sum()
        ),
        "missing_open": int(
            bars["open"].isna().sum()
        ),
        "missing_high": int(
            bars["high"].isna().sum()
        ),
        "missing_low": int(
            bars["low"].isna().sum()
        ),
        "missing_close": int(
            bars["close"].isna().sum()
        ),
        "nonpositive_open": int(
            (bars["open"] <= 0).sum()
        ),
        "nonpositive_high": int(
            (bars["high"] <= 0).sum()
        ),
        "nonpositive_low": int(
            (bars["low"] <= 0).sum()
        ),
        "nonpositive_close": int(
            (bars["close"] <= 0).sum()
        ),
        "high_below_ohlc_max": int(
            (bars["high"] < ohlc_max).sum()
        ),
        "low_above_ohlc_min": int(
            (bars["low"] > ohlc_min).sum()
        ),
        "missing_volume": int(
            bars["volume"].isna().sum()
        ),
        "negative_volume": int(
            (bars["volume"] < 0).sum()
        ),
        "zero_volume": int(
            (bars["volume"] == 0).sum()
        ),
        "missing_trade_count": int(
            bars["trade_count"].isna().sum()
        ),
        "negative_trade_count": int(
            (bars["trade_count"] < 0).sum()
        ),
        "zero_trade_count": int(
            (bars["trade_count"] == 0).sum()
        ),
        "missing_vwap": int(
            bars["vwap"].isna().sum()
        ),
        "nonpositive_vwap": int(
            (bars["vwap"] <= 0).sum()
        ),
        "vwap_outside_low_high": int(
            vwap_outside_range.sum()
        ),
    }

    integrity = pd.DataFrame(
        {
            "issue": list(issue_counts),
            "count": list(issue_counts.values()),
        }
    )

    integrity["fatal"] = integrity["issue"].isin(
        FATAL_QUALITY_ISSUES
    )

    integrity["passed"] = integrity["count"].eq(0)

    missingness = (
        bars[list(REQUIRED_COLUMNS)]
        .isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .reset_index(names="column")
    )

    missingness["missing_rate"] = (
        missingness["missing_count"] / len(bars)
    )

    symbol_groups = bars.groupby(
        "symbol",
        observed=True,
        sort=True,
    )

    symbols = symbol_groups.agg(
        rows=("timestamp", "size"),
        sessions=("session_date", "nunique"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        minimum_open=("open", "min"),
        maximum_high=("high", "max"),
        minimum_low=("low", "min"),
        maximum_close=("close", "max"),
        total_volume=("volume", "sum"),
        total_trade_count=("trade_count", "sum"),
        zero_volume_rows=(
            "volume",
            lambda values: int(
                values.eq(0).sum()
            ),
        ),
        zero_trade_count_rows=(
            "trade_count",
            lambda values: int(
                values.eq(0).sum()
            ),
        ),
    ).reset_index()

    session_counts = (
        bars.groupby(
            ["symbol", "session_date"],
            observed=True,
            sort=True,
        )
        .agg(
            bars=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            session_volume=("volume", "sum"),
            session_trade_count=(
                "trade_count",
                "sum",
            ),
        )
        .reset_index()
    )

    sessions = (
        session_counts.groupby(
            "symbol",
            observed=True,
            sort=True,
        )
        .agg(
            sessions=("session_date", "size"),
            minimum_bars=("bars", "min"),
            median_bars=("bars", "median"),
            maximum_bars=("bars", "max"),
            full_26_bar_sessions=(
                "bars",
                lambda values: int(
                    values.eq(26).sum()
                ),
            ),
            early_close_sessions=(
                "bars",
                lambda values: int(
                    values.lt(26).sum()
                ),
            ),
        )
        .reset_index()
    )

    return DataQualityBundle(
        integrity=integrity,
        missingness=missingness,
        symbols=symbols,
        sessions=sessions,
    )


def find_fatal_quality_failures(
    integrity: pd.DataFrame,
) -> pd.DataFrame:
    """Return fatal quality issues with positive counts."""

    required = {
        "issue",
        "count",
        "fatal",
    }

    missing = required.difference(
        integrity.columns
    )

    if missing:
        raise EdaFeatureError(
            "Integrity table is missing columns: "
            f"{sorted(missing)}"
        )

    return integrity.loc[
        integrity["fatal"]
        & integrity["count"].gt(0)
    ].copy()


def assert_analysis_ready(
    frame: pd.DataFrame,
) -> DataQualityBundle:
    """Require the dataset to pass fatal analytical checks."""

    quality = build_data_quality_tables(frame)

    failures = find_fatal_quality_failures(
        quality.integrity
    )

    if not failures.empty:
        details = failures[
            ["issue", "count"]
        ].to_dict("records")

        raise EdaFeatureError(
            "Canonical bars failed fatal quality checks: "
            f"{details}"
        )

    return quality


def build_return_features(
    frame: pd.DataFrame,
    *,
    expected_symbols: Sequence[str] | None = None,
) -> EdaFeatureBundle:
    """Construct bar and session return features."""

    assert_analysis_ready(frame)

    bars = normalize_bars(frame)

    if expected_symbols is not None:
        expected = {
            str(symbol).strip().upper()
            for symbol in expected_symbols
        }

        actual = set(
            bars["symbol"].dropna()
        )

        if actual != expected:
            raise EdaFeatureError(
                "Return-feature universe mismatch. "
                f"Expected: {sorted(expected)}; "
                f"actual: {sorted(actual)}."
            )

    session_group = bars.groupby(
        ["symbol", "session_date"],
        observed=True,
        sort=False,
    )

    bars["bar_number"] = (
        session_group.cumcount() + 1
    )

    bars["bars_in_session"] = (
        session_group["timestamp"]
        .transform("size")
        .astype("int64")
    )

    bars["is_session_open_bar"] = (
        bars["bar_number"].eq(1)
    )

    bars["is_session_close_bar"] = (
        bars["bar_number"]
        .eq(bars["bars_in_session"])
    )

    symbol_group = bars.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    bars["previous_close"] = (
        symbol_group["close"].shift(1)
    )

    bars["previous_session_date"] = (
        symbol_group["session_date"].shift(1)
    )

    same_session = bars["session_date"].eq(
        bars["previous_session_date"]
    )

    bars["close_to_close_simple_return"] = (
        bars["close"] / bars["previous_close"]
        - 1.0
    )

    bars["close_to_close_log_return"] = (
        np.log(bars["close"])
        - np.log(bars["previous_close"])
    )

    bars["intraday_simple_return"] = (
        bars["close_to_close_simple_return"]
        .where(same_session)
    )

    bars["intraday_log_return"] = (
        bars["close_to_close_log_return"]
        .where(same_session)
    )

    overnight_mask = (
        bars["is_session_open_bar"]
        & bars["previous_close"].notna()
    )

    bars["overnight_simple_return"] = (
        (
            bars["open"]
            / bars["previous_close"]
            - 1.0
        )
        .where(overnight_mask)
    )

    bars["overnight_log_return"] = (
        (
            np.log(bars["open"])
            - np.log(bars["previous_close"])
        )
        .where(overnight_mask)
    )

    bars["bar_open_to_close_simple_return"] = (
        bars["close"] / bars["open"] - 1.0
    )

    bars["bar_open_to_close_log_return"] = (
        np.log(bars["close"])
        - np.log(bars["open"])
    )

    bars["high_low_log_range"] = (
        np.log(bars["high"])
        - np.log(bars["low"])
    )

    sessions = (
        bars.groupby(
            ["symbol", "session_date"],
            observed=True,
            sort=True,
        )
        .agg(
            session_open_timestamp=(
                "timestamp",
                "first",
            ),
            session_close_timestamp=(
                "timestamp",
                "last",
            ),
            session_open=("open", "first"),
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
            bars=("timestamp", "size"),
            session_volume=("volume", "sum"),
            session_trade_count=(
                "trade_count",
                "sum",
            ),
        )
        .reset_index()
    )

    session_symbol_group = sessions.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    sessions["previous_session_close"] = (
        session_symbol_group[
            "session_close"
        ].shift(1)
    )

    sessions["overnight_simple_return"] = (
        sessions["session_open"]
        / sessions["previous_session_close"]
        - 1.0
    )

    sessions["overnight_log_return"] = (
        np.log(sessions["session_open"])
        - np.log(
            sessions["previous_session_close"]
        )
    )

    sessions["regular_session_simple_return"] = (
        sessions["session_close"]
        / sessions["session_open"]
        - 1.0
    )

    sessions["regular_session_log_return"] = (
        np.log(sessions["session_close"])
        - np.log(sessions["session_open"])
    )

    sessions["close_to_close_simple_return"] = (
        sessions["session_close"]
        / sessions["previous_session_close"]
        - 1.0
    )

    sessions["close_to_close_log_return"] = (
        np.log(sessions["session_close"])
        - np.log(
            sessions["previous_session_close"]
        )
    )

    sessions["log_return_decomposition_error"] = (
        sessions["close_to_close_log_return"]
        - (
            sessions["overnight_log_return"]
            + sessions[
                "regular_session_log_return"
            ]
        )
    )

    return EdaFeatureBundle(
        bars=bars,
        sessions=sessions,
    )
