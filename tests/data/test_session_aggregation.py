"""Tests for session-contained canonical 15-minute aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)


ECONOMIC_COLUMNS = [
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


def make_session(
    *,
    symbol: str = "SPY",
    session_date: str = "2025-01-02",
    bar_count: int = 26,
    price_offset: float = 0.0,
) -> pd.DataFrame:
    """Create one deterministic canonical 15-minute session."""

    timestamps = pd.date_range(
        f"{session_date} 14:30:00+00:00",
        periods=bar_count,
        freq="15min",
    )
    sequence = pd.Series(
        range(bar_count),
        dtype=float,
    )
    open_price = 100.0 + price_offset + sequence

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "session_date": session_date,
            "open": open_price,
            "high": open_price + 2.0,
            "low": open_price - 1.0,
            "close": open_price + 1.0,
            "volume": sequence + 1.0,
            "trade_count": (
                sequence.astype(int) + 10
            ),
            "vwap": open_price + 0.25,
            "source": "test",
            "feed": "sip",
        }
    )


def test_regular_session_aggregates_to_thirteen_30min_bars() -> None:
    result = aggregate_session_bars(
        make_session(),
        "30min",
    )

    assert len(result) == 13
    assert result["source_bar_count"].eq(2).all()
    assert not result["is_partial_bar"].any()


def test_early_close_aggregates_to_seven_30min_bars() -> None:
    result = aggregate_session_bars(
        make_session(bar_count=14),
        "30min",
    )

    assert len(result) == 7
    assert result["source_bar_count"].eq(2).all()
    assert not result["is_partial_bar"].any()


def test_regular_session_retains_partial_60min_close() -> None:
    result = aggregate_session_bars(
        make_session(),
        "60min",
    )

    assert len(result) == 7
    assert result["source_bar_count"].tolist() == [
        4,
        4,
        4,
        4,
        4,
        4,
        2,
    ]
    assert result["is_partial_bar"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_early_close_retains_partial_60min_close() -> None:
    result = aggregate_session_bars(
        make_session(bar_count=14),
        "60min",
    )

    assert len(result) == 4
    assert result["source_bar_count"].tolist() == [
        4,
        4,
        4,
        2,
    ]
    assert result["is_partial_bar"].tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_ohlcv_uses_declared_aggregation_rules() -> None:
    result = aggregate_session_bars(
        make_session(),
        "30min",
    )
    first = result.iloc[0]

    assert first["timestamp"] == pd.Timestamp(
        "2025-01-02 14:30:00+00:00"
    )
    assert first["open"] == 100.0
    assert first["high"] == 103.0
    assert first["low"] == 99.0
    assert first["close"] == 102.0
    assert first["volume"] == 3.0


def test_trade_count_is_summed_when_present() -> None:
    result = aggregate_session_bars(
        make_session(),
        "30min",
    )

    assert result.iloc[0]["trade_count"] == 21

    without_trade_count = make_session().drop(
        columns="trade_count"
    )
    result_without = aggregate_session_bars(
        without_trade_count,
        "30min",
    )

    assert (
        "trade_count"
        not in result_without.columns
    )


def test_vwap_is_volume_weighted() -> None:
    result = aggregate_session_bars(
        make_session(),
        "30min",
    )
    expected = (
        (100.25 * 1.0) + (101.25 * 2.0)
    ) / 3.0

    assert result.iloc[0]["vwap"] == pytest.approx(
        expected
    )


def test_symbols_are_aggregated_independently() -> None:
    bars = pd.concat(
        [
            make_session(
                symbol="SPY",
                price_offset=0.0,
            ),
            make_session(
                symbol="QQQ",
                price_offset=100.0,
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_session_bars(
        bars,
        "30min",
    )

    assert result.groupby(
        "symbol",
        observed=True,
    ).size().to_dict() == {
        "QQQ": 13,
        "SPY": 13,
    }
    assert result.loc[
        result["symbol"].eq("QQQ"),
        "open",
    ].min() == 200.0
    assert result.loc[
        result["symbol"].eq("SPY"),
        "open",
    ].min() == 100.0


def test_sessions_are_aggregated_independently() -> None:
    bars = pd.concat(
        [
            make_session(
                session_date="2025-01-02",
            ),
            make_session(
                session_date="2025-01-03",
                price_offset=50.0,
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_session_bars(
        bars,
        "60min",
    )

    assert result.groupby(
        "session_date",
        observed=True,
    ).size().to_dict() == {
        "2025-01-02": 7,
        "2025-01-03": 7,
    }
    assert result.groupby(
        "session_date",
        observed=True,
    )["is_partial_bar"].sum().to_dict() == {
        "2025-01-02": 1,
        "2025-01-03": 1,
    }


def test_output_order_is_symbol_session_timestamp() -> None:
    bars = pd.concat(
        [
            make_session(
                symbol="SPY",
                session_date="2025-01-03",
            ),
            make_session(
                symbol="QQQ",
                session_date="2025-01-03",
            ),
            make_session(
                symbol="SPY",
                session_date="2025-01-02",
            ),
            make_session(
                symbol="QQQ",
                session_date="2025-01-02",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_session_bars(
        bars,
        "30min",
    )
    expected = result.sort_values(
        [
            "symbol",
            "session_date",
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_invalid_target_frequency_is_rejected() -> None:
    with pytest.raises(
        SessionAggregationError,
        match="target_frequency",
    ):
        aggregate_session_bars(
            make_session(),
            "45min",
        )


def test_missing_required_columns_are_rejected() -> None:
    with pytest.raises(
        SessionAggregationError,
        match="missing required columns",
    ):
        aggregate_session_bars(
            make_session().drop(columns="close"),
            "30min",
        )


def test_invalid_source_session_size_is_rejected() -> None:
    with pytest.raises(
        SessionAggregationError,
        match="14 or 26",
    ):
        aggregate_session_bars(
            make_session(bar_count=25),
            "30min",
        )


def test_duplicate_timestamps_are_rejected() -> None:
    bars = make_session()
    bars.loc[1, "timestamp"] = bars.loc[
        0,
        "timestamp",
    ]

    with pytest.raises(
        SessionAggregationError,
        match="duplicate timestamps",
    ):
        aggregate_session_bars(
            bars,
            "30min",
        )


def test_unsorted_session_timestamps_are_rejected() -> None:
    bars = make_session()
    bars.iloc[[0, 1]] = bars.iloc[
        [1, 0]
    ].to_numpy()

    with pytest.raises(
        SessionAggregationError,
        match="sorted",
    ):
        aggregate_session_bars(
            bars,
            "30min",
        )


def test_15min_pass_through_preserves_economic_values() -> None:
    bars = make_session().drop(
        columns="session_date"
    )
    original = bars.copy(deep=True)

    result = aggregate_session_bars(
        bars,
        "15min",
    )

    pd.testing.assert_frame_equal(
        bars,
        original,
    )
    pd.testing.assert_frame_equal(
        result[ECONOMIC_COLUMNS],
        original[ECONOMIC_COLUMNS],
    )

    assert result["session_date"].eq(
        "2025-01-02"
    ).all()
    assert result["bar_frequency"].eq(
        "15min"
    ).all()
    assert result["source_frequency"].eq(
        "15min"
    ).all()
    assert result["source_bar_count"].eq(1).all()
    assert not result["is_partial_bar"].any()
