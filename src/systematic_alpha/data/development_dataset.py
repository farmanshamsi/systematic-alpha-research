"""Development-only market-data chunking and assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import exchange_calendars as xcals
import pandas as pd

from systematic_alpha.data.sample_windows import (
    SampleWindow,
    SampleWindowError,
)


class DevelopmentDatasetError(ValueError):
    """Raised when development data cannot be assembled safely."""


@dataclass(frozen=True)
class DevelopmentChunk:
    """One immutable calendar-year development-data request."""

    label: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp

    def request_bounds(self) -> tuple[datetime, datetime]:
        """Return API-compatible UTC request boundaries."""

        return (
            self.start.to_pydatetime(),
            self.end_exclusive.to_pydatetime(),
        )


def _clean_symbols(symbols: Sequence[str]) -> list[str]:
    """Normalize and validate the development universe."""

    clean = sorted(
        {
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        }
    )

    if not clean:
        raise DevelopmentDatasetError(
            "At least one development symbol is required."
        )

    return clean


def build_yearly_development_chunks(
    window: SampleWindow,
) -> list[DevelopmentChunk]:
    """Split the development sample into calendar-year chunks."""

    start_year = window.development_start.year
    final_included_year = (
        window.development_end_exclusive
        - pd.Timedelta(days=1)
    ).year

    chunks: list[DevelopmentChunk] = []

    for year in range(start_year, final_included_year + 1):
        year_start = pd.Timestamp(
            year=year,
            month=1,
            day=1,
            tz="UTC",
        )

        following_year = pd.Timestamp(
            year=year + 1,
            month=1,
            day=1,
            tz="UTC",
        )

        chunk_start = max(
            window.development_start,
            year_start,
        )

        chunk_end = min(
            window.development_end_exclusive,
            following_year,
        )

        if chunk_start >= chunk_end:
            continue

        chunks.append(
            DevelopmentChunk(
                label=str(year),
                start=chunk_start,
                end_exclusive=chunk_end,
            )
        )

    if not chunks:
        raise DevelopmentDatasetError(
            "Development sample produced no calendar-year chunks."
        )

    return chunks


def combine_canonical_bar_chunks(
    frames: Sequence[pd.DataFrame],
    *,
    window: SampleWindow,
    expected_symbols: Sequence[str],
) -> pd.DataFrame:
    """Combine canonical yearly bar chunks with leakage controls."""

    symbols = _clean_symbols(expected_symbols)

    if not frames:
        raise DevelopmentDatasetError(
            "No canonical bar chunks were supplied."
        )

    checked_frames: list[pd.DataFrame] = []

    required_columns = {
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
    }

    for index, frame in enumerate(frames):
        if frame.empty:
            raise DevelopmentDatasetError(
                f"Canonical chunk {index} is empty."
            )

        missing = required_columns.difference(frame.columns)

        if missing:
            raise DevelopmentDatasetError(
                f"Canonical chunk {index} is missing columns: "
                f"{sorted(missing)}"
            )

        checked_frames.append(frame.copy())

    combined = pd.concat(
        checked_frames,
        ignore_index=True,
        sort=False,
    )

    try:
        combined = window.validate_development_frame(combined)
    except SampleWindowError as exc:
        raise DevelopmentDatasetError(
            "Combined dataset violated the development window."
        ) from exc

    combined["symbol"] = (
        combined["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    actual_symbols = set(
        combined["symbol"].dropna().tolist()
    )

    expected_symbol_set = set(symbols)

    missing_symbols = expected_symbol_set.difference(
        actual_symbols
    )

    unexpected_symbols = actual_symbols.difference(
        expected_symbol_set
    )

    if missing_symbols or unexpected_symbols:
        raise DevelopmentDatasetError(
            "Development-universe mismatch. "
            f"Missing: {sorted(missing_symbols)}; "
            f"unexpected: {sorted(unexpected_symbols)}."
        )

    duplicate_mask = combined.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise DevelopmentDatasetError(
            "Duplicate symbol/timestamp bars detected. "
            f"Affected rows: {duplicate_count}."
        )

    if combined["source"].nunique(dropna=False) != 1:
        raise DevelopmentDatasetError(
            "Combined development bars contain multiple sources."
        )

    if combined["feed"].nunique(dropna=False) != 1:
        raise DevelopmentDatasetError(
            "Combined development bars contain multiple feeds."
        )

    combined = combined.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    for symbol, symbol_frame in combined.groupby(
        "symbol",
        sort=True,
    ):
        if symbol_frame.empty:
            raise DevelopmentDatasetError(
                f"No observations found for {symbol}."
            )

        if not symbol_frame["timestamp"].is_monotonic_increasing:
            raise DevelopmentDatasetError(
                f"Timestamps are not increasing for {symbol}."
            )

    return combined


def build_monthly_development_chunks(
    window: SampleWindow,
) -> list[DevelopmentChunk]:
    """Split the development sample into resumable calendar months."""

    month_start = window.development_start.replace(day=1)
    chunks: list[DevelopmentChunk] = []

    while month_start < window.development_end_exclusive:
        next_month = month_start + pd.offsets.MonthBegin(1)

        chunk_start = max(
            window.development_start,
            month_start,
        )

        chunk_end = min(
            window.development_end_exclusive,
            next_month,
        )

        if chunk_start < chunk_end:
            chunks.append(
                DevelopmentChunk(
                    label=month_start.strftime("%Y-%m"),
                    start=chunk_start,
                    end_exclusive=chunk_end,
                )
            )

        month_start = next_month

    if not chunks:
        raise DevelopmentDatasetError(
            "Development sample produced no monthly chunks."
        )

    return chunks


def filter_regular_session_bars(
    frame: pd.DataFrame,
    *,
    calendar_name: str = "XNYS",
) -> pd.DataFrame:
    """Keep bars inside each official exchange session."""

    if frame.empty:
        raise DevelopmentDatasetError(
            "Cannot filter an empty bar dataset."
        )

    if "timestamp" not in frame.columns:
        raise DevelopmentDatasetError(
            "Exchange-session filtering requires timestamp."
        )

    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    calendar = xcals.get_calendar(calendar_name)

    local_dates = (
        result["timestamp"]
        .dt.tz_convert(calendar.tz)
        .dt.strftime("%Y-%m-%d")
    )

    schedule = calendar.schedule.loc[
        local_dates.min():local_dates.max(),
        ["open", "close"],
    ].copy()

    if schedule.empty:
        raise DevelopmentDatasetError(
            "Exchange calendar returned no sessions "
            "for the supplied observations."
        )

    schedule["_session_date"] = (
        schedule.index.strftime("%Y-%m-%d")
    )

    schedule = schedule.rename(
        columns={
            "open": "_session_open",
            "close": "_session_close",
        }
    )

    result["_session_date"] = local_dates

    result = result.merge(
        schedule[
            [
                "_session_date",
                "_session_open",
                "_session_close",
            ]
        ],
        on="_session_date",
        how="left",
        validate="many_to_one",
    )

    inside_session = (
        result["_session_open"].notna()
        & (
            result["timestamp"]
            >= result["_session_open"]
        )
        & (
            result["timestamp"]
            < result["_session_close"]
        )
    )

    result = result.loc[inside_session].copy()

    result = result.drop(
        columns=[
            "_session_date",
            "_session_open",
            "_session_close",
        ]
    )

    if result.empty:
        raise DevelopmentDatasetError(
            "No official exchange-session observations remained."
        )

    sort_columns = ["timestamp"]

    if "symbol" in result.columns:
        sort_columns = ["symbol", "timestamp"]

    return result.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)


def validate_complete_session_grid(
    frame: pd.DataFrame,
    *,
    chunk: DevelopmentChunk,
    expected_symbols: Sequence[str],
    timeframe_minutes: int = 15,
    calendar_name: str = "XNYS",
) -> None:
    """Require every expected bar in every exchange session."""

    if timeframe_minutes <= 0:
        raise DevelopmentDatasetError(
            "Timeframe minutes must be positive."
        )

    if frame.empty:
        raise DevelopmentDatasetError(
            "Cannot validate an empty session grid."
        )

    required = {"timestamp", "symbol"}
    missing_columns = required.difference(frame.columns)

    if missing_columns:
        raise DevelopmentDatasetError(
            "Session-grid validation is missing columns: "
            f"{sorted(missing_columns)}"
        )

    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result.duplicated(
        ["symbol", "timestamp"]
    ).any():
        raise DevelopmentDatasetError(
            "Session grid contains duplicate "
            "symbol/timestamp observations."
        )

    outside_chunk = (
        (result["timestamp"] < chunk.start)
        | (
            result["timestamp"]
            >= chunk.end_exclusive
        )
    )

    if outside_chunk.any():
        raise DevelopmentDatasetError(
            "Session grid contains observations "
            "outside the requested chunk."
        )

    expected = _clean_symbols(expected_symbols)
    actual_symbols = set(result["symbol"].dropna())

    if actual_symbols != set(expected):
        raise DevelopmentDatasetError(
            "Session-grid universe mismatch. "
            f"Expected: {sorted(expected)}; "
            f"actual: {sorted(actual_symbols)}."
        )

    calendar = xcals.get_calendar(calendar_name)

    start_date = chunk.start.strftime("%Y-%m-%d")

    final_instant = (
        chunk.end_exclusive
        - pd.Timedelta(nanoseconds=1)
    )

    end_date = final_instant.strftime("%Y-%m-%d")

    schedule = calendar.schedule.loc[
        start_date:end_date,
        ["open", "close"],
    ]

    if schedule.empty:
        raise DevelopmentDatasetError(
            "Exchange calendar returned no sessions "
            "for the requested chunk."
        )

    expected_frames: list[pd.DataFrame] = []

    frequency = f"{timeframe_minutes}min"

    for _, session in schedule.iterrows():
        timestamps = pd.date_range(
            start=session["open"],
            end=session["close"],
            freq=frequency,
            inclusive="left",
        )

        for symbol in expected:
            expected_frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "timestamp": timestamps,
                    }
                )
            )

    expected_frame = pd.concat(
        expected_frames,
        ignore_index=True,
    )

    expected_index = pd.MultiIndex.from_frame(
        expected_frame[
            ["symbol", "timestamp"]
        ]
    )

    actual_index = pd.MultiIndex.from_frame(
        result[
            ["symbol", "timestamp"]
        ]
    )

    missing_slots = expected_index.difference(
        actual_index
    )

    extra_slots = actual_index.difference(
        expected_index
    )

    if len(missing_slots) or len(extra_slots):
        raise DevelopmentDatasetError(
            "Incomplete official exchange-session grid. "
            f"Missing slots: {len(missing_slots)} "
            f"{list(missing_slots[:5])}; "
            f"extra slots: {len(extra_slots)} "
            f"{list(extra_slots[:5])}."
        )


def validate_chunk_edge_coverage(
    frame: pd.DataFrame,
    *,
    chunk: DevelopmentChunk,
    expected_symbols: Sequence[str],
    tolerance_days: int = 10,
) -> None:
    """Reject responses missing the beginning or end of a chunk."""

    if tolerance_days <= 0:
        raise DevelopmentDatasetError(
            "Coverage tolerance must be positive."
        )

    if frame.empty:
        raise DevelopmentDatasetError(
            "Cannot validate coverage of an empty dataset."
        )

    required = {"timestamp", "symbol"}

    missing = required.difference(frame.columns)

    if missing:
        raise DevelopmentDatasetError(
            "Coverage validation is missing columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    expected = set(_clean_symbols(expected_symbols))
    actual = set(result["symbol"].dropna())

    if actual != expected:
        raise DevelopmentDatasetError(
            "Coverage universe mismatch. "
            f"Expected: {sorted(expected)}; "
            f"actual: {sorted(actual)}."
        )

    earliest_allowed = (
        chunk.start
        + pd.Timedelta(days=tolerance_days)
    )

    latest_allowed = (
        chunk.end_exclusive
        - pd.Timedelta(days=tolerance_days)
    )

    failures: list[str] = []

    for symbol in sorted(expected):
        symbol_rows = result.loc[
            result["symbol"] == symbol,
            "timestamp",
        ]

        first_timestamp = symbol_rows.min()
        last_timestamp = symbol_rows.max()

        if first_timestamp > earliest_allowed:
            failures.append(
                f"{symbol} begins at "
                f"{first_timestamp.isoformat()}"
            )

        if last_timestamp < latest_allowed:
            failures.append(
                f"{symbol} ends at "
                f"{last_timestamp.isoformat()}"
            )

    if failures:
        raise DevelopmentDatasetError(
            "Chunk failed edge-coverage validation: "
            + "; ".join(failures)
        )
