import pandas as pd

from systematic_alpha.data.resampling import resample_bars
from systematic_alpha.data.schemas import normalize_bars


def test_resample_one_minute_to_fifteen_minutes() -> None:
    timestamps = pd.date_range(
        "2025-12-15T14:30:00Z",
        periods=30,
        freq="1min",
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * 30,
            "open": [100.0 + i for i in range(30)],
            "high": [101.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [100] * 30,
            "trade_count": [10] * 30,
            "vwap": [100.25 + i for i in range(30)],
        }
    )

    normalized = normalize_bars(
        frame,
        source="test",
        feed="test",
    )

    result = resample_bars(
        normalized,
        timeframe_minutes=15,
    )

    assert len(result) == 2

    first = result.iloc[0]

    assert first["timestamp"] == pd.Timestamp(
        "2025-12-15T14:30:00Z"
    )
    assert first["open"] == 100.0
    assert first["high"] == 115.0
    assert first["low"] == 99.0
    assert first["close"] == 114.5
    assert first["volume"] == 1500
    assert first["trade_count"] == 150
