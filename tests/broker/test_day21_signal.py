from datetime import timedelta
import math

import exchange_calendars as xcals
import pandas as pd
import pytest

from systematic_alpha.broker.day21_signal import (
    DAY21_CANDIDATE_ID,
    Day21SignalError,
    build_day21_signal,
)


def operational_bars(*, partial_final_bars: int = 12) -> tuple[pd.DataFrame, object]:
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc["2026-07-01":"2026-07-20"].iloc[:12]
    rows: list[dict[str, object]] = []
    index = 0
    for session_number, (_, session) in enumerate(schedule.iterrows()):
        starts = pd.date_range(
            session["open"], session["close"], freq="15min", inclusive="left"
        )
        if session_number == len(schedule) - 1:
            starts = starts[:partial_final_bars]
        for timestamp in starts:
            close = 500.0 + 2.0 * math.sin(index / 5.0) + 0.001 * index
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": "SPY",
                    "open": close - 0.05,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 10_000,
                    "trade_count": 500,
                    "vwap": close,
                    "source": "alpaca",
                    "feed": "iex",
                }
            )
            index += 1
    frame = pd.DataFrame(rows)
    as_of = frame["timestamp"].iloc[-1].to_pydatetime() + timedelta(minutes=20)
    return frame, as_of


def test_build_day21_signal_uses_post_lock_gap_free_completed_bars() -> None:
    bars, as_of = operational_bars()
    snapshot = build_day21_signal(bars, as_of=as_of)
    assert snapshot.candidate_id == DAY21_CANDIDATE_ID
    assert snapshot.symbol == "SPY"
    assert snapshot.signal_available
    assert snapshot.signal_fresh
    assert snapshot.signal_age_seconds == 300.0
    assert snapshot.operational_rows == len(bars)
    assert snapshot.data_start >= pd.Timestamp("2026-07-01", tz="UTC").to_pydatetime()
    assert not snapshot.locked_research_data_accessed


def test_build_day21_signal_marks_weekend_snapshot_stale() -> None:
    bars, as_of = operational_bars()
    snapshot = build_day21_signal(bars, as_of=as_of + timedelta(days=2))
    assert not snapshot.signal_fresh


def test_build_day21_signal_accepts_microsecond_provider_resolution() -> None:
    bars, as_of = operational_bars()
    bars["timestamp"] = bars["timestamp"].astype("datetime64[us, UTC]")
    snapshot = build_day21_signal(bars, as_of=as_of)
    assert snapshot.signal_available


def test_build_day21_signal_rejects_locked_interval_overlap() -> None:
    bars, as_of = operational_bars()
    bars.loc[0, "timestamp"] = pd.Timestamp("2026-06-30T13:30:00Z")
    with pytest.raises(Day21SignalError, match="overlap"):
        build_day21_signal(bars, as_of=as_of)


def test_build_day21_signal_rejects_gap_in_historical_session() -> None:
    bars, as_of = operational_bars()
    bars = bars.drop(index=5).reset_index(drop=True)
    with pytest.raises(Day21SignalError, match="gap-free prefix"):
        build_day21_signal(bars, as_of=as_of)
