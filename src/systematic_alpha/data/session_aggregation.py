"""Session-contained aggregation of canonical 15-minute bars."""

from __future__ import annotations

import math
from typing import Final

import pandas as pd


SOURCE_FREQUENCY: Final[str] = "15min"
SUPPORTED_TARGET_FREQUENCIES: Final[
    tuple[str, ...]
] = (
    "15min",
    "30min",
    "60min",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "source",
    "feed",
)

ECONOMIC_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
)

ALLOWED_SESSION_BAR_COUNTS: Final[
    frozenset[int]
] = frozenset({14, 26})

SOURCE_BARS_PER_TARGET: Final[
    dict[str, int]
] = {
    "15min": 1,
    "30min": 2,
    "60min": 4,
}

EXPECTED_OUTPUT_BARS: Final[
    dict[str, dict[int, int]]
] = {
    "15min": {
        14: 14,
        26: 26,
    },
    "30min": {
        14: 7,
        26: 13,
    },
    "60min": {
        14: 4,
        26: 7,
    },
}

SESSION_COLUMN: Final[str] = "session_date"
EXCHANGE_TIMEZONE: Final[str] = "America/New_York"


class SessionAggregationError(ValueError):
    """Raised when canonical bars cannot be aggregated safely."""


def _validate_target_frequency(
    target_frequency: str,
) -> str:
    """Validate one frozen Day 10 target frequency."""

    if not isinstance(target_frequency, str):
        raise TypeError(
            "target_frequency must be a string."
        )

    if (
        target_frequency
        not in SUPPORTED_TARGET_FREQUENCIES
    ):
        raise SessionAggregationError(
            "target_frequency must be one of "
            f"{SUPPORTED_TARGET_FREQUENCIES}."
        )

    return target_frequency


def _normalize_session_dates(
    values: pd.Series,
) -> pd.Series:
    """Normalize supplied session labels to ISO dates."""

    try:
        normalized = pd.to_datetime(
            values.copy(deep=True),
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise SessionAggregationError(
            "session_date contains malformed values."
        ) from exc

    if normalized.isna().any():
        raise SessionAggregationError(
            "session_date cannot contain missing values."
        )

    return normalized.dt.strftime("%Y-%m-%d")


def _validate_numeric_columns(
    frame: pd.DataFrame,
) -> None:
    """Validate canonical numeric values without mutating input."""

    numeric_columns = list(ECONOMIC_COLUMNS)

    if "trade_count" in frame.columns:
        numeric_columns.append("trade_count")

    for column in numeric_columns:
        try:
            numeric = pd.to_numeric(
                frame[column],
                errors="raise",
            )
        except (TypeError, ValueError) as exc:
            raise SessionAggregationError(
                f"{column} must contain numeric values."
            ) from exc

        if numeric.isna().any():
            raise SessionAggregationError(
                f"{column} cannot contain missing values."
            )

        if not numeric.map(math.isfinite).all():
            raise SessionAggregationError(
                f"{column} must contain finite values."
            )

    prices = frame[
        ["open", "high", "low", "close"]
    ]

    if prices.le(0.0).any().any():
        raise SessionAggregationError(
            "OHLC prices must be strictly positive."
        )

    if frame["volume"].lt(0.0).any():
        raise SessionAggregationError(
            "volume must be non-negative."
        )

    if frame["vwap"].le(0.0).any():
        raise SessionAggregationError(
            "vwap must be strictly positive."
        )

    if (
        "trade_count" in frame.columns
        and frame["trade_count"].lt(0.0).any()
    ):
        raise SessionAggregationError(
            "trade_count must be non-negative."
        )

    ohlc_max = prices.max(axis=1)
    ohlc_min = prices.min(axis=1)

    if frame["high"].lt(ohlc_max).any():
        raise SessionAggregationError(
            "high is inconsistent with OHLC values."
        )

    if frame["low"].gt(ohlc_min).any():
        raise SessionAggregationError(
            "low is inconsistent with OHLC values."
        )


def _prepare_source_bars(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and copy canonical 15-minute source bars."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "bars must be a pandas DataFrame."
        )

    if bars.empty:
        raise SessionAggregationError(
            "bars must not be empty."
        )

    missing = sorted(
        set(REQUIRED_COLUMNS).difference(
            bars.columns
        )
    )

    if missing:
        raise SessionAggregationError(
            "Canonical bars are missing required columns: "
            f"{missing}."
        )

    result = bars.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise SessionAggregationError(
            "timestamp contains malformed values."
        ) from exc

    if result["timestamp"].isna().any():
        raise SessionAggregationError(
            "timestamp cannot contain missing values."
        )

    normalized_symbols = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        normalized_symbols.isna().any()
        or normalized_symbols.eq("").any()
    ):
        raise SessionAggregationError(
            "symbol cannot contain missing or empty values."
        )

    original_symbols = result[
        "symbol"
    ].astype("string")

    if not original_symbols.eq(
        normalized_symbols
    ).all():
        raise SessionAggregationError(
            "symbol values must already be canonical "
            "uppercase identifiers."
        )

    for column in ("source", "feed"):
        normalized = (
            result[column]
            .astype("string")
            .str.strip()
        )

        if (
            normalized.isna().any()
            or normalized.eq("").any()
        ):
            raise SessionAggregationError(
                f"{column} cannot contain missing or "
                "empty values."
            )

    _validate_numeric_columns(result)

    derived_session = (
        result["timestamp"]
        .dt.tz_convert(EXCHANGE_TIMEZONE)
        .dt.strftime("%Y-%m-%d")
    )

    if SESSION_COLUMN in result.columns:
        supplied_session = (
            _normalize_session_dates(
                result[SESSION_COLUMN]
            )
        )

        if not supplied_session.eq(
            derived_session
        ).all():
            raise SessionAggregationError(
                "session_date does not match the timestamp "
                "trading session."
            )

        result[SESSION_COLUMN] = supplied_session
    else:
        result[SESSION_COLUMN] = derived_session

    group_columns = [
        "symbol",
        SESSION_COLUMN,
    ]

    if result.duplicated(
        [
            *group_columns,
            "timestamp",
        ]
    ).any():
        raise SessionAggregationError(
            "Canonical bars contain duplicate timestamps "
            "within a symbol/session."
        )

    for _, group in result.groupby(
        group_columns,
        observed=True,
        sort=False,
    ):
        if not group[
            "timestamp"
        ].is_monotonic_increasing:
            raise SessionAggregationError(
                "Timestamps must be sorted within every "
                "symbol/session."
            )

        differences = group[
            "timestamp"
        ].diff().dropna()

        if not differences.eq(
            pd.Timedelta(minutes=15)
        ).all():
            raise SessionAggregationError(
                "Source timestamps must be consecutive "
                "15-minute bars within each session."
            )

    session_sizes = result.groupby(
        group_columns,
        observed=True,
        sort=False,
    ).size()

    if not session_sizes.isin(
        ALLOWED_SESSION_BAR_COUNTS
    ).all():
        invalid_sizes = sorted(
            {
                int(value)
                for value in session_sizes
                if value
                not in ALLOWED_SESSION_BAR_COUNTS
            }
        )
        raise SessionAggregationError(
            "Every canonical source session must contain "
            "14 or 26 bars. Invalid sizes: "
            f"{invalid_sizes}."
        )

    return result


def _attach_metadata(
    frame: pd.DataFrame,
    *,
    target_frequency: str,
    source_bar_count: int | pd.Series,
) -> pd.DataFrame:
    """Attach consistent frequency diagnostics."""

    result = frame.copy(deep=True)
    result["bar_frequency"] = target_frequency
    result["source_frequency"] = (
        SOURCE_FREQUENCY
    )
    result["source_bar_count"] = (
        source_bar_count
    )
    result["source_bar_count"] = (
        pd.to_numeric(
            result["source_bar_count"],
            errors="raise",
        ).astype("int64")
    )
    result["is_partial_bar"] = (
        result["source_bar_count"]
        < SOURCE_BARS_PER_TARGET[
            target_frequency
        ]
    )

    return result


def _validate_output_contract(
    source: pd.DataFrame,
    output: pd.DataFrame,
    *,
    target_frequency: str,
) -> None:
    """Validate deterministic session-contained output."""

    group_columns = [
        "symbol",
        SESSION_COLUMN,
    ]
    source_sizes = source.groupby(
        group_columns,
        observed=True,
        sort=False,
    ).size()
    output_sizes = output.groupby(
        group_columns,
        observed=True,
        sort=False,
    ).size()

    expected_sizes = source_sizes.map(
        EXPECTED_OUTPUT_BARS[
            target_frequency
        ]
    )

    if not output_sizes.reindex(
        expected_sizes.index
    ).equals(expected_sizes):
        raise RuntimeError(
            "Session aggregation produced unexpected "
            "output row counts."
        )

    membership = output.groupby(
        group_columns,
        observed=True,
        sort=False,
    ).agg(
        symbol_count=("symbol", "nunique"),
        session_count=(
            SESSION_COLUMN,
            "nunique",
        ),
    )

    if (
        membership["symbol_count"].ne(1).any()
        or membership[
            "session_count"
        ].ne(1).any()
    ):
        raise RuntimeError(
            "An output group contains multiple symbols "
            "or sessions."
        )

    if target_frequency == "30min":
        if (
            output["source_bar_count"].ne(2).any()
            or output["is_partial_bar"].any()
        ):
            raise RuntimeError(
                "30-minute bars must contain exactly two "
                "source bars and cannot be partial."
            )

    if target_frequency == "60min":
        allowed_counts = {2, 4}

        if not set(
            output["source_bar_count"]
        ).issubset(allowed_counts):
            raise RuntimeError(
                "60-minute source-bar counts must be "
                "two or four."
            )

        final_bars = (
            output.groupby(
                group_columns,
                observed=True,
                sort=False,
            )
            .tail(1)
        )

        if (
            final_bars[
                "source_bar_count"
            ].ne(2).any()
            or not final_bars[
                "is_partial_bar"
            ].all()
        ):
            raise RuntimeError(
                "Every 60-minute session must retain one "
                "two-input partial closing bar."
            )

        non_final = output.drop(
            index=final_bars.index
        )

        if (
            non_final[
                "source_bar_count"
            ].ne(4).any()
            or non_final[
                "is_partial_bar"
            ].any()
        ):
            raise RuntimeError(
                "Non-final 60-minute bars must contain "
                "four source bars."
            )


def _aggregate_larger_bars(
    source: pd.DataFrame,
    *,
    target_frequency: str,
) -> pd.DataFrame:
    """Aggregate validated source bars within each session."""

    group_columns = [
        "symbol",
        SESSION_COLUMN,
    ]
    source_bars_per_target = (
        SOURCE_BARS_PER_TARGET[
            target_frequency
        ]
    )
    working = source.copy(deep=True)
    working["_source_position"] = (
        working.groupby(
            group_columns,
            observed=True,
            sort=False,
        ).cumcount()
    )
    working["_output_bucket"] = (
        working["_source_position"]
        // source_bars_per_target
    )
    working["_vwap_notional"] = (
        working["vwap"]
        * working["volume"]
    )

    aggregation: dict[
        str,
        tuple[str, str],
    ] = {
        "timestamp": ("timestamp", "first"),
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "volume": ("volume", "sum"),
        "_vwap_notional": (
            "_vwap_notional",
            "sum",
        ),
        "source_bar_count": (
            "timestamp",
            "size",
        ),
        "source": ("source", "first"),
        "feed": ("feed", "first"),
        "_source_count": (
            "source",
            "nunique",
        ),
        "_feed_count": (
            "feed",
            "nunique",
        ),
    }

    if "trade_count" in working.columns:
        aggregation["trade_count"] = (
            "trade_count",
            "sum",
        )

    keys = [
        *group_columns,
        "_output_bucket",
    ]
    result = (
        working.groupby(
            keys,
            observed=True,
            sort=False,
        )
        .agg(**aggregation)
        .reset_index()
    )

    if (
        result["_source_count"].ne(1).any()
        or result["_feed_count"].ne(1).any()
    ):
        raise SessionAggregationError(
            "An output bar cannot combine multiple "
            "sources or feeds."
        )

    total_volume = result["volume"].where(
        result["volume"].ne(0.0)
    )
    result["vwap"] = (
        result["_vwap_notional"]
        / total_volume
    )
    result = result.drop(
        columns=[
            "_output_bucket",
            "_vwap_notional",
            "_source_count",
            "_feed_count",
        ]
    )

    return _attach_metadata(
        result,
        target_frequency=target_frequency,
        source_bar_count=(
            result["source_bar_count"]
        ),
    )


def aggregate_session_bars(
    bars: pd.DataFrame,
    target_frequency: str,
) -> pd.DataFrame:
    """Aggregate canonical 15-minute bars within trading sessions.

    The supported targets are ``15min``, ``30min`` and ``60min``.
    The 60-minute result retains each final two-input closing bar
    and marks it as partial.
    """

    frequency = _validate_target_frequency(
        target_frequency
    )
    source = _prepare_source_bars(bars)

    if frequency == SOURCE_FREQUENCY:
        result = _attach_metadata(
            source,
            target_frequency=frequency,
            source_bar_count=1,
        )
    else:
        result = _aggregate_larger_bars(
            source,
            target_frequency=frequency,
        )

    result = result.sort_values(
        [
            "symbol",
            SESSION_COLUMN,
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    _validate_output_contract(
        source,
        result,
        target_frequency=frequency,
    )

    return result
