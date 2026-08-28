"""Alpaca historical quote and trade data adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockQuotesRequest,
    StockTradesRequest,
)

from systematic_alpha.data.config_loader import (
    load_alpaca_credentials,
    load_project_config,
)
from systematic_alpha.data.schemas import (
    normalize_quotes,
    normalize_trades,
)


class AlpacaMicrostructureError(RuntimeError):
    """Raised when Alpaca quote or trade data cannot be retrieved."""


@dataclass(frozen=True)
class MicrostructureFetchResult:
    """Raw and canonical representations of one provider request."""

    raw: pd.DataFrame
    normalized: pd.DataFrame
    request_metadata: dict[str, Any]


def _as_utc_datetime(
    value: str | datetime | pd.Timestamp,
) -> datetime:
    """Convert a datetime-like value to timezone-aware UTC."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def _resolve_feed(feed_name: str) -> DataFeed:
    """Convert a configured feed name to Alpaca's enum."""

    feeds = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
    }

    try:
        return feeds[feed_name.lower()]
    except KeyError as exc:
        raise AlpacaMicrostructureError(
            f"Unsupported Alpaca feed: {feed_name!r}. "
            f"Supported feeds: {sorted(feeds)}"
        ) from exc


class AlpacaMicrostructureProvider:
    """Fetch provider-shaped and canonical quotes and trades."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        client: StockHistoricalDataClient | Any | None = None,
    ) -> None:
        self.config = config or load_project_config()

        feed_name = self.config["broker"].get(
            "stock_data_feed",
            "iex",
        )
        self.feed = _resolve_feed(feed_name)

        if client is None:
            credentials = load_alpaca_credentials(self.config)

            client = StockHistoricalDataClient(
                api_key=credentials.api_key,
                secret_key=credentials.secret_key,
            )

        self.client = client

    @staticmethod
    def _clean_symbols(
        symbols: Sequence[str],
    ) -> list[str]:
        clean = sorted(
            {
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            }
        )

        if not clean:
            raise AlpacaMicrostructureError(
                "At least one symbol is required."
            )

        return clean

    @staticmethod
    def _validate_window(
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
    ) -> tuple[datetime, datetime]:
        start_utc = _as_utc_datetime(start)
        end_utc = _as_utc_datetime(end)

        if start_utc >= end_utc:
            raise AlpacaMicrostructureError(
                "Start time must precede end time."
            )

        return start_utc, end_utc

    def fetch_quotes_bundle(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
    ) -> MicrostructureFetchResult:
        """Fetch raw and canonical historical Level-1 quotes."""

        clean_symbols = self._clean_symbols(symbols)
        start_utc, end_utc = self._validate_window(start, end)

        request = StockQuotesRequest(
            symbol_or_symbols=clean_symbols,
            start=start_utc,
            end=end_utc,
            feed=self.feed,
        )

        try:
            response = self.client.get_stock_quotes(request)
        except Exception as exc:
            raise AlpacaMicrostructureError(
                "Alpaca historical quote request failed."
            ) from exc

        try:
            provider_frame = response.df.copy()
        except AttributeError as exc:
            raise AlpacaMicrostructureError(
                "Alpaca quote response did not expose a DataFrame."
            ) from exc

        if provider_frame.empty:
            raise AlpacaMicrostructureError(
                "Alpaca returned no quotes."
            )

        raw_frame = provider_frame.reset_index()

        normalized = normalize_quotes(
            raw_frame,
            source="alpaca",
            feed=self.feed.value,
        )

        return MicrostructureFetchResult(
            raw=raw_frame,
            normalized=normalized,
            request_metadata={
                "provider": "alpaca",
                "data_kind": "quotes",
                "feed": self.feed.value,
                "symbols": clean_symbols,
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
            },
        )

    def fetch_trades_bundle(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
    ) -> MicrostructureFetchResult:
        """Fetch raw and canonical historical trades."""

        clean_symbols = self._clean_symbols(symbols)
        start_utc, end_utc = self._validate_window(start, end)

        request = StockTradesRequest(
            symbol_or_symbols=clean_symbols,
            start=start_utc,
            end=end_utc,
            feed=self.feed,
        )

        try:
            response = self.client.get_stock_trades(request)
        except Exception as exc:
            raise AlpacaMicrostructureError(
                "Alpaca historical trade request failed."
            ) from exc

        try:
            provider_frame = response.df.copy()
        except AttributeError as exc:
            raise AlpacaMicrostructureError(
                "Alpaca trade response did not expose a DataFrame."
            ) from exc

        if provider_frame.empty:
            raise AlpacaMicrostructureError(
                "Alpaca returned no trades."
            )

        raw_frame = provider_frame.reset_index()

        normalized = normalize_trades(
            raw_frame,
            source="alpaca",
            feed=self.feed.value,
        )

        return MicrostructureFetchResult(
            raw=raw_frame,
            normalized=normalized,
            request_metadata={
                "provider": "alpaca",
                "data_kind": "trades",
                "feed": self.feed.value,
                "symbols": clean_symbols,
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
            },
        )

    def fetch_quotes(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch canonical historical Level-1 quotes."""

        return self.fetch_quotes_bundle(
            symbols=symbols,
            start=start,
            end=end,
        ).normalized

    def fetch_trades(
        self,
        *,
        symbols: Sequence[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch canonical historical trades."""

        return self.fetch_trades_bundle(
            symbols=symbols,
            start=start,
            end=end,
        ).normalized
