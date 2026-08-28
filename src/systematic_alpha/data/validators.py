"""Validation utilities for normalized market data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from systematic_alpha.data.schemas import BAR_COLUMNS


class DataValidationError(ValueError):
    """Raised when normalized market data fails validation."""


@dataclass(frozen=True)
class BarValidationReport:
    """Summary of checks performed on a normalized bar dataset."""

    row_count: int
    symbol_count: int
    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp
    duplicate_rows: int
    missing_values: int
    internal_missing_bars: int
    ohlc_violations: int
    negative_volume_rows: int
    timestamps_are_utc: bool
    timestamps_are_sorted: bool

    @property
    def passed(self) -> bool:
        return (
            self.row_count > 0
            and self.duplicate_rows == 0
            and self.missing_values == 0
            and self.internal_missing_bars == 0
            and self.ohlc_violations == 0
            and self.negative_volume_rows == 0
            and self.timestamps_are_utc
            and self.timestamps_are_sorted
        )


def _count_internal_missing_bars(
    frame: pd.DataFrame,
    *,
    expected_minutes: int,
    exchange_timezone: str,
) -> int:
    """Count missing bars between the first and last observation per session."""

    local_timestamp = frame["timestamp"].dt.tz_convert(exchange_timezone)

    working = frame.assign(
        _local_date=local_timestamp.dt.date,
    )

    missing_count = 0

    for (_, _), group in working.groupby(
        ["symbol", "_local_date"],
        sort=False,
    ):
        observed = pd.DatetimeIndex(group["timestamp"].drop_duplicates())

        if len(observed) <= 1:
            continue

        expected = pd.date_range(
            start=observed.min(),
            end=observed.max(),
            freq=f"{expected_minutes}min",
            tz="UTC",
        )

        missing_count += len(expected.difference(observed))

    return int(missing_count)


def validate_bars(
    frame: pd.DataFrame,
    *,
    expected_minutes: int = 1,
    exchange_timezone: str = "America/New_York",
) -> BarValidationReport:
    """Validate normalized OHLCV bars and return a structured report."""

    if frame.empty:
        raise DataValidationError("Bar dataset is empty.")

    missing_columns = set(BAR_COLUMNS).difference(frame.columns)

    if missing_columns:
        raise DataValidationError(
            f"Missing normalized bar columns: {sorted(missing_columns)}"
        )

    timestamps_are_utc = (
        isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)
        and str(frame["timestamp"].dt.tz) == "UTC"
    )

    ordered = frame.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    timestamps_are_sorted = (
        frame.reset_index(drop=True)[["symbol", "timestamp"]]
        .equals(ordered[["symbol", "timestamp"]])
    )

    duplicate_rows = int(
        frame.duplicated(["symbol", "timestamp"]).sum()
    )

    missing_values = int(frame[BAR_COLUMNS].isna().sum().sum())

    high_reference = frame[["open", "close", "low"]].max(axis=1)
    low_reference = frame[["open", "close", "high"]].min(axis=1)

    ohlc_violations = int(
        (
            (frame["high"] < high_reference)
            | (frame["low"] > low_reference)
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()
    )

    negative_volume_rows = int((frame["volume"] < 0).sum())

    internal_missing_bars = (
        _count_internal_missing_bars(
            frame,
            expected_minutes=expected_minutes,
            exchange_timezone=exchange_timezone,
        )
        if timestamps_are_utc
        else -1
    )

    return BarValidationReport(
        row_count=len(frame),
        symbol_count=frame["symbol"].nunique(),
        minimum_timestamp=frame["timestamp"].min(),
        maximum_timestamp=frame["timestamp"].max(),
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        internal_missing_bars=internal_missing_bars,
        ohlc_violations=ohlc_violations,
        negative_volume_rows=negative_volume_rows,
        timestamps_are_utc=timestamps_are_utc,
        timestamps_are_sorted=timestamps_are_sorted,
    )


def assert_valid_bars(
    frame: pd.DataFrame,
    *,
    expected_minutes: int = 1,
    exchange_timezone: str = "America/New_York",
) -> BarValidationReport:
    """Validate bars and raise an exception when any check fails."""

    report = validate_bars(
        frame,
        expected_minutes=expected_minutes,
        exchange_timezone=exchange_timezone,
    )

    if not report.passed:
        raise DataValidationError(
            "Bar validation failed: "
            f"duplicates={report.duplicate_rows}, "
            f"missing_values={report.missing_values}, "
            f"internal_missing_bars={report.internal_missing_bars}, "
            f"ohlc_violations={report.ohlc_violations}, "
            f"negative_volume_rows={report.negative_volume_rows}, "
            f"utc={report.timestamps_are_utc}, "
            f"sorted={report.timestamps_are_sorted}"
        )

    return report
