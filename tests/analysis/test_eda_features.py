"""Tests for Day 05 EDA return and quality features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.eda_features import (
    EdaFeatureError,
    build_data_quality_tables,
    build_return_features,
)


def make_test_bars() -> pd.DataFrame:
    """Create two complete synthetic sessions."""

    rows = []

    specifications = {
        "SPY": {
            "2020-01-02": [
                (100.0, 101.0),
                (101.0, 102.0),
                (102.0, 103.0),
            ],
            "2020-01-03": [
                (104.0, 105.0),
                (105.0, 106.0),
                (106.0, 107.0),
            ],
        },
        "QQQ": {
            "2020-01-02": [
                (200.0, 201.0),
                (201.0, 202.0),
                (202.0, 203.0),
            ],
            "2020-01-03": [
                (204.0, 205.0),
                (205.0, 206.0),
                (206.0, 207.0),
            ],
        },
    }

    session_starts = {
        "2020-01-02": pd.Timestamp(
            "2020-01-02T14:30:00Z"
        ),
        "2020-01-03": pd.Timestamp(
            "2020-01-03T14:30:00Z"
        ),
    }

    for symbol, sessions in specifications.items():
        for session_date, values in sessions.items():
            start = session_starts[session_date]

            for offset, (open_price, close_price) in enumerate(
                values
            ):
                timestamp = (
                    start
                    + pd.Timedelta(
                        minutes=15 * offset
                    )
                )

                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "open": open_price,
                        "high": close_price + 0.5,
                        "low": open_price - 0.5,
                        "close": close_price,
                        "volume": 1_000 + offset,
                        "trade_count": 100 + offset,
                        "vwap": (
                            open_price + close_price
                        ) / 2.0,
                        "source": "test",
                        "feed": "sip",
                    }
                )

    return pd.DataFrame(rows)


def test_build_return_features_separates_overnight() -> None:
    bars = make_test_bars()

    result = build_return_features(
        bars,
        expected_symbols=("SPY", "QQQ"),
    )

    spy = result.bars.loc[
        result.bars["symbol"].eq("SPY")
    ].reset_index(drop=True)

    second_session_open = spy.loc[
        spy["session_date"].eq("2020-01-03")
    ].iloc[0]

    expected_overnight = np.log(104.0 / 103.0)

    assert second_session_open[
        "is_session_open_bar"
    ]

    assert np.isclose(
        second_session_open[
            "overnight_log_return"
        ],
        expected_overnight,
    )

    assert pd.isna(
        second_session_open[
            "intraday_log_return"
        ]
    )


def test_session_log_return_decomposition() -> None:
    result = build_return_features(
        make_test_bars()
    )

    valid = result.sessions[
        "previous_session_close"
    ].notna()

    errors = result.sessions.loc[
        valid,
        "log_return_decomposition_error",
    ]

    assert np.allclose(
        errors.to_numpy(),
        0.0,
        atol=1.0e-12,
    )


def test_bar_numbering_resets_by_session() -> None:
    result = build_return_features(
        make_test_bars()
    )

    counts = (
        result.bars.groupby(
            ["symbol", "session_date"],
            observed=True,
        )
        .agg(
            first_bar=("bar_number", "min"),
            last_bar=("bar_number", "max"),
            bars=("bar_number", "size"),
        )
    )

    assert counts["first_bar"].eq(1).all()
    assert counts["last_bar"].eq(3).all()
    assert counts["bars"].eq(3).all()


def test_quality_table_detects_integrity_failure() -> None:
    bars = make_test_bars()

    bars.loc[0, "high"] = (
        bars.loc[0, "close"] - 1.0
    )

    bars.loc[1, "volume"] = -1.0

    quality = build_data_quality_tables(bars)

    counts = quality.integrity.set_index(
        "issue"
    )["count"]

    assert counts["high_below_ohlc_max"] == 1
    assert counts["negative_volume"] == 1

    with pytest.raises(
        EdaFeatureError,
        match="fatal quality checks",
    ):
        build_return_features(bars)


def test_duplicate_timestamp_is_rejected() -> None:
    bars = make_test_bars()

    duplicate = bars.iloc[[0]].copy()

    bars = pd.concat(
        [bars, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        EdaFeatureError,
        match="duplicate_symbol_timestamp",
    ):
        build_return_features(bars)
