import pandas as pd
import pytest

from systematic_alpha.data.schemas import (
    SchemaError,
    normalize_bars,
    normalize_quotes,
    normalize_trades,
)


def test_normalize_bars() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02 14:30:00"],
            "symbol": ["spy"],
            "open": [590.0],
            "high": [591.0],
            "low": [589.5],
            "close": [590.5],
            "volume": [1000],
        }
    )

    result = normalize_bars(
        frame,
        source="alpaca",
        feed="iex",
    )

    assert result.loc[0, "symbol"] == "SPY"
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_invalid_bar_high_raises() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02 14:30:00"],
            "symbol": ["SPY"],
            "open": [590.0],
            "high": [589.0],
            "low": [588.0],
            "close": [590.5],
            "volume": [1000],
        }
    )

    with pytest.raises(SchemaError):
        normalize_bars(frame, source="alpaca", feed="iex")


def test_crossed_quote_raises() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02 14:30:00"],
            "symbol": ["SPY"],
            "bid_price": [591.0],
            "bid_size": [100],
            "ask_price": [590.0],
            "ask_size": [100],
        }
    )

    with pytest.raises(SchemaError):
        normalize_quotes(frame, source="alpaca", feed="iex")


def test_normalize_trades() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02 14:30:00"],
            "symbol": ["spy"],
            "price": [590.25],
            "size": [20],
        }
    )

    result = normalize_trades(
        frame,
        source="alpaca",
        feed="iex",
    )

    assert result.loc[0, "symbol"] == "SPY"
    assert result.loc[0, "price"] == 590.25


def test_bar_optional_fields_are_numeric() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T14:30:00Z"],
            "symbol": ["SPY"],
            "open": [590.0],
            "high": [591.0],
            "low": [589.5],
            "close": [590.5],
            "volume": [1000],
            "trade_count": ["42"],
            "vwap": ["590.25"],
        }
    )

    result = normalize_bars(
        frame,
        source="alpaca",
        feed="iex",
    )

    assert str(result["trade_count"].dtype) == "Int64"
    assert str(result["vwap"].dtype) == "Float64"
    assert result.loc[0, "trade_count"] == 42
    assert result.loc[0, "vwap"] == 590.25


def test_negative_trade_count_raises() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T14:30:00Z"],
            "symbol": ["SPY"],
            "open": [590.0],
            "high": [591.0],
            "low": [589.5],
            "close": [590.5],
            "volume": [1000],
            "trade_count": [-1],
            "vwap": [590.25],
        }
    )

    with pytest.raises(SchemaError):
        normalize_bars(frame, source="alpaca", feed="iex")


def test_nonpositive_vwap_raises() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T14:30:00Z"],
            "symbol": ["SPY"],
            "open": [590.0],
            "high": [591.0],
            "low": [589.5],
            "close": [590.5],
            "volume": [1000],
            "trade_count": [42],
            "vwap": [0],
        }
    )

    with pytest.raises(SchemaError):
        normalize_bars(frame, source="alpaca", feed="iex")
