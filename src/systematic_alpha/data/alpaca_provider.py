"""Alpaca historical market-data adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from systematic_alpha.data.config_loader import (
    load_alpaca_credentials,
    load_project_config,
)
from systematic_alpha.data.schemas import normalize_bars


class AlpacaDataError(RuntimeError):
    """Raised when Alpaca data cannot be retrieved or normalized."""


@dataclass(frozen=True)
class BarFetchResult:
    """Provider-shaped and canonical representations of one request."""

    raw: pd.DataFrame
    normalized: pd.DataFrame
    request_metadata: dict[str, Any]


def _as_utc_datetime(value: str | datetime | pd.Timestamp) -> datetime:
    """Convert a datetime-like value to a timezone-aware UTC datetime."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def _resolve_feed(feed_name: str) -> DataFeed:
    """Convert the configured feed name to Alpaca's enum."""

    feeds = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
    }

    try:
        return feeds[feed_name.lower()]
    except KeyError as exc:
        raise AlpacaDataError(
            f"Unsupported Alpaca stock feed: {feed_name!r}. "
            f"Supported feeds: {sorted(feeds)}"
        ) from exc


class AlpacaBarProvider:
    """Provider-independent adapter around Alpaca historical stock bars."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        client: StockHistoricalDataClient | Any | None = None,
    ) -> None:
        self.config = config or load_project_config()

        feed_name = self.config["broker"].get("stock_data_feed", "iex")
        self.feed = _resolve_feed(feed_name)

        if client is None:
            credentials = load_alpaca_credentials(self.config)
            client = StockHistoricalDataClient(
                api_key=credentials.api_key,
                secret_key=credentials.secret_key,
            )

        self.client = client

    def fetch_bars_bundle(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        timeframe_minutes: int = 1,
    ) -> BarFetchResult:
        """Fetch provider-shaped and canonical historical bars."""

        clean_symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            }
        )

        if not clean_symbols:
            raise AlpacaDataError("At least one symbol is required.")

        if not isinstance(timeframe_minutes, int) or timeframe_minutes <= 0:
            raise AlpacaDataError(
                "timeframe_minutes must be a positive integer."
            )

        start_utc = _as_utc_datetime(start)
        end_utc = _as_utc_datetime(end)

        if start_utc >= end_utc:
            raise AlpacaDataError("Start time must precede end time.")

        request = StockBarsRequest(
            symbol_or_symbols=clean_symbols,
            timeframe=TimeFrame(
                timeframe_minutes,
                TimeFrameUnit.Minute,
            ),
            start=start_utc,
            end=end_utc,
            adjustment=Adjustment.ALL,
            feed=self.feed,
        )

        try:
            response = self.client.get_stock_bars(request)
        except Exception as exc:
            raise AlpacaDataError(
                "Alpaca historical-bar request failed."
            ) from exc

        try:
            provider_frame = response.df.copy()
        except AttributeError as exc:
            raise AlpacaDataError(
                "Alpaca response did not expose a DataFrame."
            ) from exc

        if provider_frame.empty:
            raise AlpacaDataError(
                "Alpaca returned no bars for the requested interval."
            )

        # Materialize Alpaca's symbol/timestamp index as columns for Parquet.
        # No canonical sorting, type coercion, validation or field injection
        # is applied to this raw representation.
        raw_frame = provider_frame.reset_index()

        normalized = normalize_bars(
            raw_frame,
            source="alpaca",
            feed=self.feed.value,
        )

        metadata = {
            "provider": "alpaca",
            "feed": self.feed.value,
            "symbols": clean_symbols,
            "timeframe_minutes": timeframe_minutes,
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "adjustment": Adjustment.ALL.value,
        }

        return BarFetchResult(
            raw=raw_frame,
            normalized=normalized,
            request_metadata=metadata,
        )

    def fetch_bars(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        timeframe_minutes: int = 1,
    ) -> pd.DataFrame:
        """Fetch canonical historical equity bars."""

        return self.fetch_bars_bundle(
            symbols=symbols,
            start=start,
            end=end,
            timeframe_minutes=timeframe_minutes,
        ).normalized
