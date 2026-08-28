"""Tests for realized volatility and intraday seasonality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.volatility_seasonality import (
    VolatilitySeasonalityError,
    build_daily_volatility_table,
    build_intraday_seasonality_table,
)


def make_features(
    sessions: int = 70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic synthetic bar/session features."""

    bar_rows = []
    session_rows = []

    dates = pd.date_range(
        "2020-01-02",
        periods=sessions,
        freq="B",
    )

    previous_close = 100.0

    for date_index, date in enumerate(dates):
        session_date = date.strftime("%Y-%m-%d")

        session_open = (
            previous_close
            * np.exp(0.001)
        )

        returns = np.array(
            [0.002, -0.001, 0.003],
            dtype=float,
        )

        prices = [session_open]

        for return_value in returns:
            prices.append(
                prices[-1]
                * np.exp(return_value)
            )

        session_close = prices[-1]

        start = pd.Timestamp(
            f"{session_date}T14:30:00Z"
        )

        for index, return_value in enumerate(
            returns
        ):
            bar_open = prices[index]
            bar_close = prices[index + 1]

            bar_rows.append(
                {
                    "timestamp": (
                        start
                        + pd.Timedelta(
                            minutes=15 * index
                        )
                    ),
                    "symbol": "SPY",
                    "session_date": session_date,
                    "local_time": (
                        pd.Timestamp(
                            start
                            + pd.Timedelta(
                                minutes=15 * index
                            )
                        )
                        .tz_convert(
                            "America/New_York"
                        )
                        .strftime("%H:%M:%S")
                    ),
                    "bar_number": index + 1,
                    "bars_in_session": 3,
                    "is_session_open_bar": (
                        index == 0
                    ),
                    "is_session_close_bar": (
                        index == 2
                    ),
                    "intraday_log_return": (
                        np.nan
                        if index == 0
                        else return_value
                    ),
                    "bar_open_to_close_log_return": (
                        return_value
                    ),
                    "high_low_log_range": 0.005,
                    "volume": float(
                        100 * (index + 1)
                    ),
                    "trade_count": float(
                        10 * (index + 1)
                    ),
                }
            )

        overnight_return = np.log(
            session_open / previous_close
        )

        regular_return = np.log(
            session_close / session_open
        )

        session_rows.append(
            {
                "symbol": "SPY",
                "session_date": session_date,
                "session_open": session_open,
                "session_high": max(prices) * 1.001,
                "session_low": min(prices) * 0.999,
                "session_close": session_close,
                "overnight_log_return": (
                    np.nan
                    if date_index == 0
                    else overnight_return
                ),
                "regular_session_log_return": (
                    regular_return
                ),
                "close_to_close_log_return": (
                    np.nan
                    if date_index == 0
                    else overnight_return
                    + regular_return
                ),
            }
        )

        previous_close = session_close

    return (
        pd.DataFrame(bar_rows),
        pd.DataFrame(session_rows),
    )


def test_realized_variance_includes_opening_bar() -> None:
    bars, sessions = make_features()

    result = build_daily_volatility_table(
        bars,
        sessions,
    )

    expected = (
        0.002**2
        + (-0.001) ** 2
        + 0.003**2
    )

    assert np.isclose(
        result.iloc[0][
            "realized_variance_regular"
        ],
        expected,
    )


def test_regular_return_reconciles() -> None:
    bars, sessions = make_features()

    result = build_daily_volatility_table(
        bars,
        sessions,
    )

    assert np.allclose(
        result[
            "regular_return_reconciliation_error"
        ],
        0.0,
        atol=1.0e-12,
    )


def test_seasonality_volume_shares() -> None:
    bars, _ = make_features()

    result = build_intraday_seasonality_table(
        bars
    )

    expected = {
        1: 1.0 / 6.0,
        2: 2.0 / 6.0,
        3: 3.0 / 6.0,
    }

    for bar_number, share in expected.items():
        actual = result.loc[
            result["bar_number"].eq(bar_number),
            "mean_volume_share",
        ].iloc[0]

        assert np.isclose(actual, share)


def test_rolling_volatility_columns_exist() -> None:
    bars, sessions = make_features()

    result = build_daily_volatility_table(
        bars,
        sessions,
        rolling_windows=(21, 63),
    )

    assert (
        result[
            "rolling_total_realized_volatility_21d"
        ].notna().sum()
        == len(result) - 20
    )

    assert (
        result[
            "rolling_total_realized_volatility_63d"
        ].notna().sum()
        == len(result) - 62
    )


def test_missing_column_is_rejected() -> None:
    bars, sessions = make_features()

    bars = bars.drop(
        columns=["bar_open_to_close_log_return"]
    )

    with pytest.raises(
        VolatilitySeasonalityError,
        match="missing required columns",
    ):
        build_daily_volatility_table(
            bars,
            sessions,
        )
