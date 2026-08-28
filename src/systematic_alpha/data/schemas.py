"""Normalized pandas schemas for market-data objects."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class SchemaError(ValueError):
    """Raised when market data fails schema validation."""


BAR_COLUMNS = [
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
]

QUOTE_COLUMNS = [
    "timestamp",
    "symbol",
    "bid_price",
    "bid_size",
    "bid_exchange",
    "ask_price",
    "ask_size",
    "ask_exchange",
    "conditions",
    "tape",
    "source",
    "feed",
]

TRADE_COLUMNS = [
    "timestamp",
    "symbol",
    "id",
    "price",
    "size",
    "exchange",
    "conditions",
    "tape",
    "source",
    "feed",
]


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    frame_name: str,
) -> None:
    missing = set(required).difference(frame.columns)

    if missing:
        raise SchemaError(
            f"{frame_name} is missing required columns: {sorted(missing)}"
        )


def _normalize_common(
    frame: pd.DataFrame,
    *,
    source: str,
    feed: str,
) -> pd.DataFrame:
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

    if result["symbol"].isna().any() or (result["symbol"] == "").any():
        raise SchemaError("Symbol values cannot be empty.")

    result["source"] = source
    result["feed"] = feed

    return result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)


def normalize_bars(
    frame: pd.DataFrame,
    *,
    source: str,
    feed: str,
) -> pd.DataFrame:
    """Normalize and validate OHLCV bars."""

    required = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    _require_columns(frame, required, "Bar frame")
    result = _normalize_common(frame, source=source, feed=feed)

    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="raise")

    if "trade_count" not in result:
        result["trade_count"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )
    else:
        result["trade_count"] = pd.to_numeric(
            result["trade_count"],
            errors="raise",
        ).astype("Int64")

    if "vwap" not in result:
        result["vwap"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Float64",
        )
    else:
        result["vwap"] = pd.to_numeric(
            result["vwap"],
            errors="raise",
        ).astype("Float64")

    if (result["trade_count"].dropna() < 0).any():
        raise SchemaError("Trade count cannot be negative.")

    if (result["vwap"].dropna() <= 0).any():
        raise SchemaError("VWAP must be strictly positive when available.")

    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise SchemaError("OHLC prices must be strictly positive.")

    if (result["volume"] < 0).any():
        raise SchemaError("Volume cannot be negative.")

    price_max = result[["open", "close", "low"]].max(axis=1)
    price_min = result[["open", "close", "high"]].min(axis=1)

    if (result["high"] < price_max).any():
        raise SchemaError("High price is inconsistent with OHLC values.")

    if (result["low"] > price_min).any():
        raise SchemaError("Low price is inconsistent with OHLC values.")

    duplicates = result.duplicated(["symbol", "timestamp"])
    if duplicates.any():
        raise SchemaError(
            "Duplicate bar rows detected for symbol and timestamp."
        )

    return result[BAR_COLUMNS]


def normalize_quotes(
    frame: pd.DataFrame,
    *,
    source: str,
    feed: str,
) -> pd.DataFrame:
    """Normalize and validate bid/ask quotes."""

    required = [
        "timestamp",
        "symbol",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
    ]

    _require_columns(frame, required, "Quote frame")
    result = _normalize_common(frame, source=source, feed=feed)

    numeric_columns = [
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")

    if (result[["bid_price", "ask_price"]] <= 0).any().any():
        raise SchemaError("Quote prices must be strictly positive.")

    if (result[["bid_size", "ask_size"]] < 0).any().any():
        raise SchemaError("Quote sizes cannot be negative.")

    if (result["bid_price"] > result["ask_price"]).any():
        raise SchemaError("Crossed quote detected: bid exceeds ask.")

    for optional_column in [
        "bid_exchange",
        "ask_exchange",
        "tape",
    ]:
        if optional_column not in result:
            result[optional_column] = pd.Series(
                pd.NA,
                index=result.index,
                dtype="string",
            )
        else:
            result[optional_column] = (
                result[optional_column].astype("string")
            )

    if "conditions" not in result:
        result["conditions"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="object",
        )

    return result[QUOTE_COLUMNS]


def normalize_trades(
    frame: pd.DataFrame,
    *,
    source: str,
    feed: str,
) -> pd.DataFrame:
    """Normalize and validate individual trades."""

    required = [
        "timestamp",
        "symbol",
        "price",
        "size",
    ]

    _require_columns(frame, required, "Trade frame")
    result = _normalize_common(frame, source=source, feed=feed)

    result["price"] = pd.to_numeric(result["price"], errors="raise")
    result["size"] = pd.to_numeric(result["size"], errors="raise")

    if (result["price"] <= 0).any():
        raise SchemaError("Trade prices must be strictly positive.")

    if (result["size"] <= 0).any():
        raise SchemaError("Trade sizes must be strictly positive.")

    for optional_column in ["id", "exchange", "tape"]:
        if optional_column not in result:
            result[optional_column] = pd.Series(
                pd.NA,
                index=result.index,
                dtype="string",
            )
        else:
            result[optional_column] = (
                result[optional_column].astype("string")
            )

    if "conditions" not in result:
        result["conditions"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="object",
        )

    return result[TRADE_COLUMNS]
