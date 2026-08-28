"""Frozen operational SPY signal construction for Day 21."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Final

import exchange_calendars as xcals
import pandas as pd

from systematic_alpha.analysis.reversion_inference import CONFIGURATIONS
from systematic_alpha.data.development_dataset import filter_regular_session_bars
from systematic_alpha.strategies.ou_vwap_reversion import (
    OuVwapReversionParameters,
    build_ou_vwap_reversion_strategy,
)


DAY21_SYMBOL: Final[str] = "SPY"
DAY21_CANDIDATE_ID: Final[str] = "ou_vwap_slow"
DAY21_TIMEFRAME_MINUTES: Final[int] = 15
OPERATIONAL_DATA_START: Final[pd.Timestamp] = pd.Timestamp(
    "2026-07-01T00:00:00Z"
)
MAX_SIGNAL_AGE: Final[timedelta] = timedelta(minutes=20)


class Day21SignalError(ValueError):
    """Raised when a live operational signal cannot be built safely."""


@dataclass(frozen=True, slots=True)
class Day21SignalSnapshot:
    candidate_id: str
    symbol: str
    computed_at: datetime
    bar_start: datetime
    bar_end: datetime
    last_close: float
    position: int
    raw_signal: int
    signal_available: bool
    signal_fresh: bool
    signal_age_seconds: float
    regime_eligible: bool
    ou_zscore: float | None
    ou_half_life_bars: float | None
    variance_ratio: float | None
    operational_rows: int
    operational_sessions: int
    data_start: datetime
    data_end: datetime
    locked_research_data_accessed: bool = False


def _utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise Day21SignalError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise Day21SignalError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _slow_parameters() -> OuVwapReversionParameters:
    matches = tuple(
        item
        for item in CONFIGURATIONS
        if item.configuration_id == DAY21_CANDIDATE_ID
    )
    if len(matches) != 1:
        raise RuntimeError("Frozen Day 21 candidate is unavailable or ambiguous.")
    return matches[0]


def _attach_operational_session_fields(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
) -> pd.DataFrame:
    """Validate full-session prefixes and attach causal feature fields."""

    filtered = filter_regular_session_bars(bars, calendar_name="XNYS")
    filtered = filtered.loc[
        filtered["timestamp"].ge(OPERATIONAL_DATA_START)
    ].copy()
    if filtered.empty:
        raise Day21SignalError("No post-lock operational bars are available.")

    bar_delta = pd.Timedelta(minutes=DAY21_TIMEFRAME_MINUTES)
    complete = filtered["timestamp"].add(bar_delta).le(pd.Timestamp(as_of))
    filtered = filtered.loc[complete].copy()
    if filtered.empty:
        raise Day21SignalError("No completed operational bar is available.")

    local_dates = (
        filtered["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[
        local_dates.min():local_dates.max(), ["open", "close"]
    ].copy()
    schedule["session_date_text"] = schedule.index.strftime("%Y-%m-%d")
    schedule = schedule.reset_index(drop=True)
    filtered["session_date_text"] = local_dates
    filtered = filtered.merge(
        schedule,
        on="session_date_text",
        how="left",
        validate="many_to_one",
    )
    if filtered[["open_y", "close_y"]].isna().any().any():
        raise Day21SignalError("An operational bar has no XNYS session.")

    filtered = filtered.rename(columns={"open_x": "open", "close_x": "close"})
    latest_session = filtered["session_date_text"].max()
    for session_text, group in filtered.groupby(
        "session_date_text", observed=True, sort=True
    ):
        session_open = pd.Timestamp(group["open_y"].iloc[0])
        session_close = pd.Timestamp(group["close_y"].iloc[0])
        expected = pd.date_range(
            start=session_open,
            end=session_close,
            freq=f"{DAY21_TIMEFRAME_MINUTES}min",
            inclusive="left",
        )
        observed = pd.DatetimeIndex(group["timestamp"].sort_values())
        expected_prefix = expected[: len(observed)]
        if len(observed) != len(expected_prefix) or not all(
            left == right
            for left, right in zip(observed, expected_prefix, strict=True)
        ):
            raise Day21SignalError(
                f"Operational session {session_text} is not a gap-free prefix."
            )
        if session_text != latest_session and len(observed) != len(expected):
            raise Day21SignalError(
                f"Historical operational session {session_text} is incomplete."
            )

    filtered["session_date"] = pd.to_datetime(
        filtered["session_date_text"], utc=True, errors="raise"
    )
    filtered["is_session_close_bar"] = filtered["timestamp"].eq(
        filtered["close_y"].sub(bar_delta)
    )
    filtered = filtered.sort_values("timestamp", kind="stable").reset_index(
        drop=True
    )
    filtered["close_to_close_simple_return"] = filtered["close"].pct_change()
    return filtered


def build_day21_signal(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
) -> Day21SignalSnapshot:
    """Build the frozen, lagged SPY operational signal from completed bars."""

    observed_at = _utc(as_of, name="as_of")
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise Day21SignalError("Operational bars must be a non-empty DataFrame.")
    required = {
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
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise Day21SignalError(f"Operational bars are missing columns: {missing}.")
    working = bars.copy(deep=True)
    working["timestamp"] = pd.to_datetime(
        working["timestamp"], utc=True, errors="raise"
    )
    working["symbol"] = working["symbol"].astype("string").str.strip().str.upper()
    symbols = tuple(working["symbol"].dropna().unique())
    if symbols != (DAY21_SYMBOL,):
        raise Day21SignalError("Day 21 operational data must contain only SPY.")
    if working["timestamp"].min() < OPERATIONAL_DATA_START:
        raise Day21SignalError("Operational bars overlap the locked 2026 interval.")

    features = _attach_operational_session_fields(working, as_of=observed_at)
    bundle = build_ou_vwap_reversion_strategy(
        features,
        parameters=_slow_parameters(),
        allow_incomplete_final_session=True,
    )
    latest = bundle.observations.iloc[-1]
    bar_start = pd.Timestamp(latest["timestamp"]).to_pydatetime()
    bar_end = bar_start + timedelta(minutes=DAY21_TIMEFRAME_MINUTES)
    age = observed_at - bar_end
    if age < timedelta(0):
        raise Day21SignalError("Latest operational bar has not completed.")
    if not bool(latest["signal_available"]):
        raise Day21SignalError("Day 21 signal warm-up is incomplete.")

    def optional_float(value: object) -> float | None:
        if pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            raise Day21SignalError("Signal diagnostic is non-finite.")
        return number

    return Day21SignalSnapshot(
        candidate_id=DAY21_CANDIDATE_ID,
        symbol=DAY21_SYMBOL,
        computed_at=observed_at,
        bar_start=bar_start,
        bar_end=bar_end,
        last_close=float(latest["close"]),
        position=int(latest["position"]),
        raw_signal=int(latest["signal"]),
        signal_available=bool(latest["signal_available"]),
        signal_fresh=age <= MAX_SIGNAL_AGE,
        signal_age_seconds=float(age.total_seconds()),
        regime_eligible=bool(latest["regime_eligible"]),
        ou_zscore=optional_float(latest["ou_zscore"]),
        ou_half_life_bars=optional_float(latest["ou_half_life_bars"]),
        variance_ratio=optional_float(latest["variance_ratio"]),
        operational_rows=int(len(bundle.observations)),
        operational_sessions=int(bundle.observations["session_date"].nunique()),
        data_start=pd.Timestamp(bundle.observations["timestamp"].min()).to_pydatetime(),
        data_end=pd.Timestamp(bundle.observations["timestamp"].max()).to_pydatetime(),
    )
