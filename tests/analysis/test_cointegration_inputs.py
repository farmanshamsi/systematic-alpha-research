"""Pair-input contracts for Day 14 cointegration feasibility."""

from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pandas as pd

from systematic_alpha.analysis.cointegration_feasibility import (
    CointegrationInputs,
    build_cointegration_inputs,
)


def make_bars() -> pd.DataFrame:
    """Build two synthetic sessions with one missing QQQ bar."""

    sessions = (
        "2020-01-02",
        "2020-01-03",
    )
    times = (
        "14:30:00+00:00",
        "14:45:00+00:00",
        "15:00:00+00:00",
    )
    bases = {
        "SPY": 300.0,
        "QQQ": 200.0,
        "IWM": 150.0,
    }

    rows: list[dict[str, object]] = []

    for symbol, base in bases.items():
        for session_index, session in enumerate(sessions):
            for bar_index, time in enumerate(times):
                if (
                    symbol == "QQQ"
                    and session == "2020-01-03"
                    and time == "14:45:00+00:00"
                ):
                    continue

                close = (
                    base
                    + 2.0 * session_index
                    + 0.5 * bar_index
                )

                rows.append(
                    {
                        "timestamp": pd.Timestamp(
                            f"{session} {time}"
                        ),
                        "symbol": symbol,
                        "open": close - 0.1,
                        "high": close + 0.2,
                        "low": close - 0.2,
                        "close": close,
                        "volume": 1_000.0,
                        "trade_count": 100,
                        "vwap": close,
                        "source": "synthetic",
                        "feed": "test",
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["symbol", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def test_input_bundle_is_frozen_slotted_and_ordered() -> None:
    inputs = build_cointegration_inputs(
        make_bars()
    )

    assert isinstance(
        inputs,
        CointegrationInputs,
    )
    assert is_dataclass(inputs)
    assert (
        inputs.__dataclass_params__.frozen
    )
    assert not hasattr(inputs, "__dict__")

    assert tuple(
        item.pair_id
        for item in inputs.pair_inputs
    ) == (
        "SPY_QQQ",
        "SPY_IWM",
        "QQQ_IWM",
    )


def test_pair_alignment_is_exact_and_pair_specific() -> None:
    inputs = build_cointegration_inputs(
        make_bars()
    )

    by_id = {
        item.pair_id: item
        for item in inputs.pair_inputs
    }

    assert len(
        by_id["SPY_QQQ"].daily_log_prices
    ) == 2
    assert len(
        by_id["SPY_IWM"].daily_log_prices
    ) == 2
    assert len(
        by_id["QQQ_IWM"].daily_log_prices
    ) == 2

    assert len(
        by_id["SPY_QQQ"].intraday_log_prices
    ) == 5
    assert len(
        by_id["SPY_IWM"].intraday_log_prices
    ) == 6
    assert len(
        by_id["QQQ_IWM"].intraday_log_prices
    ) == 5

    missing_timestamp = pd.Timestamp(
        "2020-01-03 14:45:00+00:00"
    )

    assert missing_timestamp not in set(
        by_id["SPY_QQQ"]
        .intraday_log_prices[
            "timestamp"
        ]
    )
    assert missing_timestamp in set(
        by_id["SPY_IWM"]
        .intraday_log_prices[
            "timestamp"
        ]
    )
    assert missing_timestamp not in set(
        by_id["QQQ_IWM"]
        .intraday_log_prices[
            "timestamp"
        ]
    )


def test_pair_frames_contain_only_finite_log_prices() -> None:
    inputs = build_cointegration_inputs(
        make_bars()
    )

    for item in inputs.pair_inputs:
        assert list(
            item.daily_log_prices.columns
        ) == [
            "session_date",
            "y_log_price",
            "x_log_price",
        ]
        assert list(
            item.intraday_log_prices.columns
        ) == [
            "timestamp",
            "session_date",
            "y_log_price",
            "x_log_price",
        ]

        assert np.isfinite(
            item.daily_log_prices[
                [
                    "y_log_price",
                    "x_log_price",
                ]
            ].to_numpy()
        ).all()

        assert np.isfinite(
            item.intraday_log_prices[
                [
                    "y_log_price",
                    "x_log_price",
                ]
            ].to_numpy()
        ).all()
