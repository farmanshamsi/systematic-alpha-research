"""Daily OHLCV reconciliation between independent data providers."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ExternalReconciliationError(ValueError):
    """Raised when daily data cannot be reconciled safely."""


DAILY_COMPARISON_COLUMNS = [
    "timestamp",
    "session_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "feed",
]


def _validate_daily_frame(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ExternalReconciliationError(
            f"{frame_name} is empty."
        )

    required = set(DAILY_COMPARISON_COLUMNS)
    missing = required.difference(frame.columns)

    if missing:
        raise ExternalReconciliationError(
            f"{frame_name} is missing columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    result["session_date"] = (
        result["session_date"]
        .astype("string")
        .str.strip()
    )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    if (
        result[["open", "high", "low", "close"]]
        <= 0
    ).any().any():
        raise ExternalReconciliationError(
            f"{frame_name} contains nonpositive prices."
        )

    if (result["volume"] < 0).any():
        raise ExternalReconciliationError(
            f"{frame_name} contains negative volume."
        )

    if result.duplicated(
        ["symbol", "session_date"]
    ).any():
        raise ExternalReconciliationError(
            f"{frame_name} contains duplicate symbol/date rows."
        )

    return result.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_intraday_to_daily(
    frame: pd.DataFrame,
    *,
    exchange_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Aggregate canonical intraday bars by local trading session."""

    if frame.empty:
        raise ExternalReconciliationError(
            "Intraday bar dataset is empty."
        )

    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "feed",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ExternalReconciliationError(
            "Intraday frame is missing columns: "
            f"{sorted(missing)}"
        )

    working = frame.copy()

    working["timestamp"] = pd.to_datetime(
        working["timestamp"],
        utc=True,
        errors="raise",
    )

    working = working.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    )

    if working["source"].nunique(dropna=False) != 1:
        raise ExternalReconciliationError(
            "Intraday frame must contain one source."
        )

    if working["feed"].nunique(dropna=False) != 1:
        raise ExternalReconciliationError(
            "Intraday frame must contain one feed."
        )

    source = str(working["source"].iloc[0])
    feed = str(working["feed"].iloc[0])

    working["session_date"] = (
        working["timestamp"]
        .dt.tz_convert(exchange_timezone)
        .dt.strftime("%Y-%m-%d")
    )

    daily = (
        working.groupby(
            ["symbol", "session_date"],
            sort=True,
            as_index=False,
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )

    daily["timestamp"] = pd.to_datetime(
        daily["session_date"],
        utc=True,
    )
    daily["source"] = source
    daily["feed"] = feed

    return _validate_daily_frame(
        daily[DAILY_COMPARISON_COLUMNS],
        frame_name="Aggregated primary daily frame",
    )


def canonical_daily_to_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Convert canonical provider daily bars to comparison form."""

    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "feed",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ExternalReconciliationError(
            "Canonical daily frame is missing columns: "
            f"{sorted(missing)}"
        )

    result = frame[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "feed",
        ]
    ].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    result["session_date"] = (
        result["timestamp"].dt.strftime("%Y-%m-%d")
    )

    return _validate_daily_frame(
        result[DAILY_COMPARISON_COLUMNS],
        frame_name="External daily frame",
    )


def reconcile_daily_ohlcv(
    primary: pd.DataFrame,
    external: pd.DataFrame,
    *,
    price_tolerance_bps: float,
    volume_tolerance_pct: float,
) -> pd.DataFrame:
    """Create a structured outer-join OHLCV discrepancy report."""

    if not np.isfinite(price_tolerance_bps):
        raise ExternalReconciliationError(
            "Price tolerance must be finite."
        )

    if not np.isfinite(volume_tolerance_pct):
        raise ExternalReconciliationError(
            "Volume tolerance must be finite."
        )

    if price_tolerance_bps < 0:
        raise ExternalReconciliationError(
            "Price tolerance cannot be negative."
        )

    if volume_tolerance_pct < 0:
        raise ExternalReconciliationError(
            "Volume tolerance cannot be negative."
        )

    primary_checked = _validate_daily_frame(
        primary,
        frame_name="Primary daily frame",
    )

    external_checked = _validate_daily_frame(
        external,
        frame_name="External daily frame",
    )

    value_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    primary_selected = primary_checked[
        [
            "session_date",
            "symbol",
            *value_columns,
            "source",
            "feed",
        ]
    ].rename(
        columns={
            **{
                column: f"{column}_primary"
                for column in value_columns
            },
            "source": "primary_source",
            "feed": "primary_feed",
        }
    )

    external_selected = external_checked[
        [
            "session_date",
            "symbol",
            *value_columns,
            "source",
            "feed",
        ]
    ].rename(
        columns={
            **{
                column: f"{column}_external"
                for column in value_columns
            },
            "source": "external_source",
            "feed": "external_feed",
        }
    )

    report = primary_selected.merge(
        external_selected,
        on=["session_date", "symbol"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    report["primary_present"] = (
        report["_merge"] != "right_only"
    )
    report["external_present"] = (
        report["_merge"] != "left_only"
    )

    price_pass_columns: list[str] = []

    for field in ["open", "high", "low", "close"]:
        absolute_difference = (
            report[f"{field}_primary"]
            - report[f"{field}_external"]
        ).abs()

        denominator = (
            report[f"{field}_external"]
            .abs()
            .replace(0, np.nan)
        )

        report[f"{field}_abs_diff"] = (
            absolute_difference
        )

        report[f"{field}_pct_diff"] = (
            absolute_difference / denominator * 100.0
        )

        report[f"{field}_abs_bps"] = (
            absolute_difference / denominator * 10000.0
        )

        pass_column = f"{field}_pass"
        price_pass_columns.append(pass_column)

        report[pass_column] = (
            report["primary_present"]
            & report["external_present"]
            & (
                report[f"{field}_abs_bps"]
                <= price_tolerance_bps
            )
        )

    volume_difference = (
        report["volume_primary"]
        - report["volume_external"]
    ).abs()

    volume_denominator = (
        report["volume_external"]
        .abs()
        .replace(0, np.nan)
    )

    report["volume_abs_diff"] = volume_difference
    report["volume_pct_diff"] = (
        volume_difference
        / volume_denominator
        * 100.0
    )

    report["volume_pass"] = (
        report["primary_present"]
        & report["external_present"]
        & (
            report["volume_pct_diff"]
            <= volume_tolerance_pct
        )
    )

    report["price_pass"] = report[
        price_pass_columns
    ].all(axis=1)

    report["overall_pass"] = (
        report["primary_present"]
        & report["external_present"]
        & report["price_pass"]
        & report["volume_pass"]
    )

    report["status"] = np.select(
        [
            ~report["primary_present"],
            ~report["external_present"],
            report["overall_pass"],
            ~report["price_pass"] & ~report["volume_pass"],
            ~report["price_pass"],
            ~report["volume_pass"],
        ],
        [
            "missing_primary",
            "missing_external",
            "pass",
            "price_and_volume_mismatch",
            "price_mismatch",
            "volume_mismatch",
        ],
        default="unclassified",
    )

    report["price_tolerance_bps"] = float(
        price_tolerance_bps
    )
    report["volume_tolerance_pct"] = float(
        volume_tolerance_pct
    )

    report["timestamp"] = pd.to_datetime(
        report["session_date"],
        utc=True,
    )

    report = report.drop(columns="_merge")

    return report.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    ).reset_index(drop=True)
