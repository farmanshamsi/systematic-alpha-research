import pandas as pd
import pytest

from systematic_alpha.data.resampling import (
    EventBarError,
    build_dollar_bars,
    build_tick_bars,
    build_volume_bars,
)
from systematic_alpha.data.schemas import normalize_trades


def make_trades(
    *,
    symbols: list[str],
    timestamps: list[str],
    prices: list[float],
    sizes: list[float],
    identifiers: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "id": identifiers,
            "price": prices,
            "size": sizes,
            "exchange": ["V"] * len(symbols),
            "conditions": [["@"]] * len(symbols),
            "tape": ["C"] * len(symbols),
        }
    )

    return normalize_trades(
        frame,
        source="test",
        feed="test",
    )


def test_tick_bars_preserve_events_and_final_partial() -> None:
    trades = make_trades(
        symbols=["SPY"] * 5,
        timestamps=[
            "2025-12-15T14:30:00.000001Z",
            "2025-12-15T14:30:00.000002Z",
            "2025-12-15T14:30:00.000003Z",
            "2025-12-15T14:30:00.000004Z",
            "2025-12-15T14:30:00.000005Z",
        ],
        prices=[100, 101, 102, 103, 104],
        sizes=[10, 20, 30, 40, 50],
        identifiers=["1", "2", "3", "4", "5"],
    )

    result = build_tick_bars(
        trades,
        trades_per_bar=2,
    )

    assert len(result) == 3
    assert result["trade_count"].tolist() == [2, 2, 1]
    assert result["is_complete"].tolist() == [
        True,
        True,
        False,
    ]

    assert result.loc[0, "open"] == 100
    assert result.loc[0, "close"] == 101
    assert result.loc[0, "high"] == 101
    assert result.loc[0, "low"] == 100

    assert float(result["volume"].sum()) == 150
    assert int(result["trade_count"].sum()) == 5


def test_volume_bars_use_whole_trade_crossing() -> None:
    trades = make_trades(
        symbols=["SPY"] * 3,
        timestamps=[
            "2025-12-15T14:30:00.000001Z",
            "2025-12-15T14:30:00.000002Z",
            "2025-12-15T14:30:00.000003Z",
        ],
        prices=[100, 101, 102],
        sizes=[60, 50, 40],
        identifiers=["1", "2", "3"],
    )

    result = build_volume_bars(
        trades,
        shares_per_bar=100,
    )

    assert result["volume"].tolist() == [110, 40]
    assert result["trade_count"].tolist() == [2, 1]
    assert result["is_complete"].tolist() == [
        True,
        False,
    ]

    assert float(result["volume"].sum()) == 150


def test_dollar_bars_do_not_mix_symbols() -> None:
    trades = make_trades(
        symbols=["SPY", "QQQ", "SPY", "QQQ"],
        timestamps=[
            "2025-12-15T14:30:00.000001Z",
            "2025-12-15T14:30:00.000001Z",
            "2025-12-15T14:30:00.000002Z",
            "2025-12-15T14:30:00.000002Z",
        ],
        prices=[100, 200, 100, 200],
        sizes=[6, 3, 6, 3],
        identifiers=["s1", "q1", "s2", "q2"],
    )

    result = build_dollar_bars(
        trades,
        dollars_per_bar=1000,
    )

    assert result["symbol"].tolist() == ["QQQ", "SPY"]
    assert result["trade_count"].tolist() == [2, 2]
    assert result["dollar_value"].tolist() == [1200, 1200]
    assert result["is_complete"].tolist() == [True, True]


def test_same_timestamp_can_close_multiple_event_bars() -> None:
    timestamp = "2025-12-15T14:30:00.000001Z"

    trades = make_trades(
        symbols=["SPY"] * 3,
        timestamps=[timestamp] * 3,
        prices=[100, 101, 102],
        sizes=[1, 1, 1],
        identifiers=["1", "2", "3"],
    )

    result = build_tick_bars(
        trades,
        trades_per_bar=1,
    )

    assert len(result) == 3
    assert result["timestamp"].nunique() == 1
    assert result["bar_sequence"].tolist() == [1, 2, 3]


def test_duplicate_trade_identifiers_are_rejected() -> None:
    trades = make_trades(
        symbols=["SPY", "SPY"],
        timestamps=[
            "2025-12-15T14:30:00.000001Z",
            "2025-12-15T14:30:00.000002Z",
        ],
        prices=[100, 101],
        sizes=[10, 20],
        identifiers=["duplicate", "duplicate"],
    )

    with pytest.raises(
        EventBarError,
        match="Duplicate trade identifiers",
    ):
        build_tick_bars(
            trades,
            trades_per_bar=1,
        )
