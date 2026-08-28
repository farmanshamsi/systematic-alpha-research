"""Time-bar construction from normalized one-minute OHLCV data."""

from __future__ import annotations

import pandas as pd

from systematic_alpha.data.schemas import normalize_bars
from systematic_alpha.data.validators import assert_valid_bars


class ResamplingError(ValueError):
    """Raised when market bars cannot be resampled safely."""


def resample_bars(
    frame: pd.DataFrame,
    *,
    timeframe_minutes: int,
) -> pd.DataFrame:
    """Aggregate one-minute bars into larger regular time bars."""

    if timeframe_minutes <= 1:
        raise ResamplingError(
            "timeframe_minutes must be greater than one."
        )

    assert_valid_bars(frame, expected_minutes=1)

    if frame["source"].nunique() != 1:
        raise ResamplingError(
            "Input frame must contain exactly one data source."
        )

    if frame["feed"].nunique() != 1:
        raise ResamplingError(
            "Input frame must contain exactly one market-data feed."
        )

    source = str(frame["source"].iloc[0])
    feed = str(frame["feed"].iloc[0])

    pieces: list[pd.DataFrame] = []
    rule = f"{timeframe_minutes}min"

    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        working = (
            symbol_frame
            .sort_values("timestamp")
            .set_index("timestamp")
            .copy()
        )

        working["_vwap_notional"] = (
            working["vwap"].fillna(working["close"])
            * working["volume"]
        )

        resampler = working.resample(
            rule,
            label="left",
            closed="left",
            origin="start_day",
            offset="30min",
        )

        result = resampler.agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trade_count": "sum",
                "_vwap_notional": "sum",
            }
        )

        result = result.dropna(
            subset=["open", "high", "low", "close"]
        )

        result["vwap"] = (
            result["_vwap_notional"]
            / result["volume"].replace(0, pd.NA)
        )

        result["symbol"] = symbol
        result = result.drop(columns="_vwap_notional")
        result = result.reset_index()

        pieces.append(result)

    if not pieces:
        raise ResamplingError("Resampling produced no bars.")

    combined = pd.concat(pieces, ignore_index=True)

    normalized = normalize_bars(
        combined,
        source=source,
        feed=feed,
    )

    assert_valid_bars(
        normalized,
        expected_minutes=timeframe_minutes,
    )

    return normalized



EVENT_BAR_COLUMNS = [
    "timestamp",
    "symbol",
    "bar_sequence",
    "bar_type",
    "threshold",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "dollar_value",
    "start_timestamp",
    "end_timestamp",
    "is_complete",
    "source",
    "feed",
]


class EventBarError(ValueError):
    """Raised when event bars cannot be constructed safely."""


def _prepare_event_trades(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and order canonical trades for event aggregation."""

    if frame.empty:
        raise EventBarError("Trade dataset is empty.")

    required = {
        "timestamp",
        "symbol",
        "price",
        "size",
        "source",
        "feed",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise EventBarError(
            "Trade frame is missing required columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()
    result["_event_order"] = range(len(result))

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

    result["price"] = pd.to_numeric(
        result["price"],
        errors="raise",
    )

    result["size"] = pd.to_numeric(
        result["size"],
        errors="raise",
    )

    if result["symbol"].isna().any():
        raise EventBarError("Trade symbols cannot be missing.")

    if (result["symbol"] == "").any():
        raise EventBarError("Trade symbols cannot be empty.")

    if (result["price"] <= 0).any():
        raise EventBarError(
            "Trade prices must be strictly positive."
        )

    if (result["size"] <= 0).any():
        raise EventBarError(
            "Trade sizes must be strictly positive."
        )

    if result[["source", "feed"]].isna().any().any():
        raise EventBarError(
            "Trade source and feed cannot be missing."
        )

    if result["source"].nunique(dropna=False) != 1:
        raise EventBarError(
            "Input trades must contain exactly one data source."
        )

    if result["feed"].nunique(dropna=False) != 1:
        raise EventBarError(
            "Input trades must contain exactly one data feed."
        )

    if "id" in result.columns:
        identified = result.loc[
            result["id"].notna(),
            ["symbol", "id"],
        ].copy()

        identified["id"] = identified["id"].astype("string")

        if identified.duplicated(["symbol", "id"]).any():
            raise EventBarError(
                "Duplicate trade identifiers detected."
            )

    return result.sort_values(
        ["symbol", "timestamp", "_event_order"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_event_threshold(
    *,
    bar_type: str,
    threshold: int | float,
) -> tuple[str, float]:
    """Validate an event-bar type and its threshold."""

    normalized_type = bar_type.strip().lower()

    if normalized_type not in {"tick", "volume", "dollar"}:
        raise EventBarError(
            "bar_type must be 'tick', 'volume', or 'dollar'."
        )

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise EventBarError(
            "Event-bar threshold must be numeric."
        ) from exc

    if not pd.notna(threshold_value):
        raise EventBarError(
            "Event-bar threshold must be finite."
        )

    if threshold_value == float("inf"):
        raise EventBarError(
            "Event-bar threshold must be finite."
        )

    if threshold_value <= 0:
        raise EventBarError(
            "Event-bar threshold must be strictly positive."
        )

    if (
        normalized_type == "tick"
        and (
            isinstance(threshold, bool)
            or not threshold_value.is_integer()
        )
    ):
        raise EventBarError(
            "Tick-bar threshold must be a positive integer."
        )

    return normalized_type, threshold_value


def _finalize_event_bar(
    *,
    symbol: str,
    sequence: int,
    bar_type: str,
    threshold: float,
    state: dict[str, object],
    is_complete: bool,
    source: str,
    feed: str,
) -> dict[str, object]:
    """Convert one event accumulation state into a bar row."""

    volume = float(state["volume"])
    dollar_value = float(state["dollar_value"])
    end_timestamp = state["end_timestamp"]

    return {
        "timestamp": end_timestamp,
        "symbol": symbol,
        "bar_sequence": sequence,
        "bar_type": bar_type,
        "threshold": threshold,
        "open": float(state["open"]),
        "high": float(state["high"]),
        "low": float(state["low"]),
        "close": float(state["close"]),
        "volume": volume,
        "trade_count": int(state["trade_count"]),
        "vwap": dollar_value / volume,
        "dollar_value": dollar_value,
        "start_timestamp": state["start_timestamp"],
        "end_timestamp": end_timestamp,
        "is_complete": is_complete,
        "source": source,
        "feed": feed,
    }


def build_event_bars(
    frame: pd.DataFrame,
    *,
    bar_type: str,
    threshold: int | float,
) -> pd.DataFrame:
    """Construct tick, volume, or dollar bars from canonical trades.

    Trades remain atomic. When a trade crosses a volume or dollar
    threshold, the whole trade belongs to the closing bar, so completed
    bars may overshoot their threshold. The final residual bar is
    retained and marked ``is_complete=False``.

    The output preserves every input trade, share and dollar of notional.
    Multiple event bars may share the same closing timestamp; use
    ``bar_sequence`` as the within-symbol ordering key.
    """

    normalized_type, threshold_value = (
        _validate_event_threshold(
            bar_type=bar_type,
            threshold=threshold,
        )
    )

    trades = _prepare_event_trades(frame)

    source = str(trades["source"].iloc[0])
    feed = str(trades["feed"].iloc[0])

    rows: list[dict[str, object]] = []

    for symbol, symbol_frame in trades.groupby(
        "symbol",
        sort=True,
    ):
        sequence = 0
        state: dict[str, object] | None = None

        for event in symbol_frame.itertuples(index=False):
            price = float(event.price)
            size = float(event.size)
            notional = price * size

            if state is None:
                state = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": size,
                    "trade_count": 1,
                    "dollar_value": notional,
                    "start_timestamp": event.timestamp,
                    "end_timestamp": event.timestamp,
                }
            else:
                state["high"] = max(
                    float(state["high"]),
                    price,
                )
                state["low"] = min(
                    float(state["low"]),
                    price,
                )
                state["close"] = price
                state["volume"] = (
                    float(state["volume"]) + size
                )
                state["trade_count"] = (
                    int(state["trade_count"]) + 1
                )
                state["dollar_value"] = (
                    float(state["dollar_value"])
                    + notional
                )
                state["end_timestamp"] = event.timestamp

            if normalized_type == "tick":
                metric = float(state["trade_count"])
            elif normalized_type == "volume":
                metric = float(state["volume"])
            else:
                metric = float(state["dollar_value"])

            if metric >= threshold_value:
                sequence += 1

                rows.append(
                    _finalize_event_bar(
                        symbol=str(symbol),
                        sequence=sequence,
                        bar_type=normalized_type,
                        threshold=threshold_value,
                        state=state,
                        is_complete=True,
                        source=source,
                        feed=feed,
                    )
                )

                state = None

        if state is not None:
            sequence += 1

            rows.append(
                _finalize_event_bar(
                    symbol=str(symbol),
                    sequence=sequence,
                    bar_type=normalized_type,
                    threshold=threshold_value,
                    state=state,
                    is_complete=False,
                    source=source,
                    feed=feed,
                )
            )

    if not rows:
        raise EventBarError(
            "Event-bar construction produced no bars."
        )

    result = pd.DataFrame(rows)

    for column in [
        "timestamp",
        "start_timestamp",
        "end_timestamp",
    ]:
        result[column] = pd.to_datetime(
            result[column],
            utc=True,
            errors="raise",
        )

    for column in [
        "symbol",
        "bar_type",
        "source",
        "feed",
    ]:
        result[column] = result[column].astype("string")

    result["bar_sequence"] = (
        pd.to_numeric(
            result["bar_sequence"],
            errors="raise",
        )
        .astype("Int64")
    )

    result["trade_count"] = (
        pd.to_numeric(
            result["trade_count"],
            errors="raise",
        )
        .astype("Int64")
    )

    for column in [
        "threshold",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "dollar_value",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("Float64")

    result["is_complete"] = result["is_complete"].astype(bool)

    result = result.sort_values(
        ["symbol", "bar_sequence"],
        kind="stable",
    ).reset_index(drop=True)

    if (
        result["timestamp"]
        != result["end_timestamp"]
    ).any():
        raise EventBarError(
            "Event-bar timestamp must equal end_timestamp."
        )

    high_reference = result[
        ["open", "close", "low"]
    ].max(axis=1)

    low_reference = result[
        ["open", "close", "high"]
    ].min(axis=1)

    if (result["high"] < high_reference).any():
        raise EventBarError(
            "Event-bar high is inconsistent with OHLC values."
        )

    if (result["low"] > low_reference).any():
        raise EventBarError(
            "Event-bar low is inconsistent with OHLC values."
        )

    for symbol, symbol_trades in trades.groupby(
        "symbol",
        sort=True,
    ):
        symbol_bars = result.loc[
            result["symbol"] == symbol
        ]

        input_volume = float(symbol_trades["size"].sum())
        output_volume = float(symbol_bars["volume"].sum())

        input_notional = float(
            (
                symbol_trades["price"]
                * symbol_trades["size"]
            ).sum()
        )

        output_notional = float(
            symbol_bars["dollar_value"].sum()
        )

        input_events = len(symbol_trades)
        output_events = int(
            symbol_bars["trade_count"].sum()
        )

        if abs(input_volume - output_volume) > 1e-8:
            raise EventBarError(
                f"Volume was not preserved for {symbol}."
            )

        notional_tolerance = max(
            1e-8,
            abs(input_notional) * 1e-12,
        )

        if (
            abs(input_notional - output_notional)
            > notional_tolerance
        ):
            raise EventBarError(
                f"Dollar notional was not preserved for {symbol}."
            )

        if input_events != output_events:
            raise EventBarError(
                f"Trade count was not preserved for {symbol}."
            )

    return result[EVENT_BAR_COLUMNS]


def build_tick_bars(
    frame: pd.DataFrame,
    *,
    trades_per_bar: int,
) -> pd.DataFrame:
    """Construct fixed-trade-count event bars."""

    return build_event_bars(
        frame,
        bar_type="tick",
        threshold=trades_per_bar,
    )


def build_volume_bars(
    frame: pd.DataFrame,
    *,
    shares_per_bar: int | float,
) -> pd.DataFrame:
    """Construct whole-trade volume bars."""

    return build_event_bars(
        frame,
        bar_type="volume",
        threshold=shares_per_bar,
    )


def build_dollar_bars(
    frame: pd.DataFrame,
    *,
    dollars_per_bar: int | float,
) -> pd.DataFrame:
    """Construct whole-trade dollar bars."""

    return build_event_bars(
        frame,
        bar_type="dollar",
        threshold=dollars_per_bar,
    )
