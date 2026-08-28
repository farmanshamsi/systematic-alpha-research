import pandas as pd

from systematic_alpha.data.schemas import normalize_bars
from systematic_alpha.data.validators import validate_bars


def make_frame(timestamps: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * len(timestamps),
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.5] * len(timestamps),
            "volume": [1000] * len(timestamps),
            "trade_count": [50] * len(timestamps),
            "vwap": [100.25] * len(timestamps),
        }
    )

    return normalize_bars(
        frame,
        source="test",
        feed="test",
    )


def test_complete_minute_sequence_passes() -> None:
    frame = make_frame(
        [
            "2025-12-15T14:30:00Z",
            "2025-12-15T14:31:00Z",
            "2025-12-15T14:32:00Z",
        ]
    )

    report = validate_bars(frame)

    assert report.passed
    assert report.internal_missing_bars == 0


def test_missing_minute_is_detected() -> None:
    frame = make_frame(
        [
            "2025-12-15T14:30:00Z",
            "2025-12-15T14:32:00Z",
        ]
    )

    report = validate_bars(frame)

    assert not report.passed
    assert report.internal_missing_bars == 1
