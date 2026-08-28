"""Yahoo Finance daily market-data adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from systematic_alpha.data.schemas import normalize_bars


class YahooDataError(RuntimeError):
    """Raised when Yahoo daily data cannot be retrieved or normalized."""


@dataclass(frozen=True)
class YahooDailyFetchResult:
    """Provider-shaped and canonical daily Yahoo data."""

    raw: pd.DataFrame
    normalized: pd.DataFrame
    request_metadata: dict[str, Any]


def _clean_symbols(symbols: Sequence[str]) -> list[str]:
    clean = sorted(
        {
            symbol.strip().upper()
            for symbol in symbols
            if symbol.strip()
        }
    )

    if not clean:
        raise YahooDataError("At least one symbol is required.")

    return clean


def _as_date(
    value: str | date | datetime | pd.Timestamp,
) -> date:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert("UTC")
            .tz_localize(None)
        )

    return timestamp.normalize().date()


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            str(column[0])
            for column in result.columns
        ]

    return result


class YahooDailyProvider:
    """Fetch adjusted daily OHLCV data from Yahoo Finance."""

    def __init__(
        self,
        *,
        downloader: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self.downloader = downloader or yf.download

    def fetch_daily_bundle(
        self,
        *,
        symbols: Sequence[str],
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
    ) -> YahooDailyFetchResult:
        """Fetch provider-shaped and canonical daily data.

        The end date is exclusive, matching yfinance semantics.
        """

        clean_symbols = _clean_symbols(symbols)
        start_date = _as_date(start)
        end_date = _as_date(end)

        if start_date >= end_date:
            raise YahooDataError(
                "Start date must precede exclusive end date."
            )

        raw_pieces: list[pd.DataFrame] = []
        normalized_pieces: list[pd.DataFrame] = []

        for symbol in clean_symbols:
            try:
                downloaded = self.downloader(
                    symbol,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    back_adjust=False,
                    actions=False,
                    repair=False,
                    prepost=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                    multi_level_index=False,
                )
            except Exception as exc:
                raise YahooDataError(
                    f"Yahoo download failed for {symbol}."
                ) from exc

            if downloaded is None or downloaded.empty:
                raise YahooDataError(
                    f"Yahoo returned no daily rows for {symbol}."
                )

            downloaded = _flatten_columns(downloaded)

            raw_piece = downloaded.reset_index()

            date_candidates = [
                column
                for column in raw_piece.columns
                if str(column).strip().lower()
                in {"date", "datetime", "index"}
            ]

            if not date_candidates:
                raise YahooDataError(
                    "Yahoo response did not expose a date column."
                )

            date_column = date_candidates[0]

            raw_piece.insert(
                0,
                "requested_symbol",
                symbol,
            )

            normalized_input = raw_piece.rename(
                columns={
                    date_column: "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            ).copy()

            normalized_input["symbol"] = symbol

            required = {
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }

            missing = required.difference(
                normalized_input.columns
            )

            if missing:
                raise YahooDataError(
                    "Yahoo daily response is missing fields: "
                    f"{sorted(missing)}"
                )

            normalized = normalize_bars(
                normalized_input,
                source="yfinance",
                feed="yahoo",
            )

            raw_pieces.append(raw_piece)
            normalized_pieces.append(normalized)

        raw = pd.concat(
            raw_pieces,
            ignore_index=True,
            sort=False,
        )

        normalized = (
            pd.concat(
                normalized_pieces,
                ignore_index=True,
            )
            .sort_values(
                ["symbol", "timestamp"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        return YahooDailyFetchResult(
            raw=raw,
            normalized=normalized,
            request_metadata={
                "provider": "yfinance",
                "feed": "yahoo",
                "symbols": clean_symbols,
                "interval": "1d",
                "start_inclusive": start_date.isoformat(),
                "end_exclusive": end_date.isoformat(),
                "auto_adjust": True,
                "back_adjust": False,
                "repair": False,
                "prepost": False,
            },
        )

    def fetch_daily(
        self,
        *,
        symbols: Sequence[str],
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch canonical Yahoo daily bars."""

        return self.fetch_daily_bundle(
            symbols=symbols,
            start=start,
            end=end,
        ).normalized
