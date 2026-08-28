"""Tests for Day 05 stylized-fact calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.stylized_facts import (
    StylizedFactsError,
    build_daily_ljung_box_table,
    build_distribution_tables,
    build_intraday_acf_table,
)


def make_bar_features() -> pd.DataFrame:
    """Create two sessions with known lag counts."""

    rows = []

    session_values = {
        "2020-01-02": [
            np.nan,
            0.01,
            -0.01,
            0.02,
            -0.02,
        ],
        "2020-01-03": [
            np.nan,
            0.02,
            -0.02,
            0.01,
            -0.01,
        ],
    }

    for session_date, returns in session_values.items():
        start = pd.Timestamp(
            f"{session_date}T14:30:00Z"
        )

        for index, value in enumerate(returns):
            rows.append(
                {
                    "symbol": "SPY",
                    "session_date": session_date,
                    "timestamp": (
                        start
                        + pd.Timedelta(
                            minutes=15 * index
                        )
                    ),
                    "intraday_log_return": value,
                }
            )

    return pd.DataFrame(rows)


def make_session_features(
    observations: int = 500,
) -> pd.DataFrame:
    """Create seeded AR(1) daily returns."""

    rng = np.random.default_rng(42)

    innovations = rng.normal(
        loc=0.0,
        scale=0.01,
        size=observations,
    )

    returns = np.zeros(observations)

    for index in range(1, observations):
        returns[index] = (
            0.75 * returns[index - 1]
            + innovations[index]
        )

    dates = pd.date_range(
        "2020-01-02",
        periods=observations,
        freq="B",
    )

    return pd.DataFrame(
        {
            "symbol": "SPY",
            "session_date": (
                dates.strftime("%Y-%m-%d")
            ),
            "overnight_log_return": (
                0.25 * returns
            ),
            "regular_session_log_return": (
                0.75 * returns
            ),
            "close_to_close_log_return": returns,
        }
    )


def test_intraday_acf_never_bridges_sessions() -> None:
    table = build_intraday_acf_table(
        make_bar_features(),
        lags=(1,),
    )

    raw = table.loc[
        table["transformation"].eq("raw")
    ].iloc[0]

    # Four finite intraday observations per session produce
    # three lag-one pairs per session: 2 * 3 = 6.
    assert raw["pair_count"] == 6


def test_distribution_tables_have_expected_rows() -> None:
    bars = make_bar_features()

    sessions = make_session_features(
        observations=100
    )

    moments, quantiles, tails = (
        build_distribution_tables(
            bars,
            sessions,
        )
    )

    assert len(moments) == 4
    assert len(quantiles) == 28
    assert len(tails) == 12

    assert {
        "skewness",
        "excess_kurtosis",
        "jarque_bera_pvalue",
    }.issubset(moments.columns)


def test_ljung_box_detects_ar1_dependence() -> None:
    table = build_daily_ljung_box_table(
        make_session_features(),
        lags=(5, 10),
    )

    raw = table.loc[
        table["transformation"].eq("raw")
    ]

    assert raw["reject_at_5pct"].all()
    assert raw["ljung_box_pvalue"].max() < 0.001


def test_tail_table_counts_extremes() -> None:
    sessions = make_session_features(
        observations=100
    )

    sessions.loc[
        sessions.index[-1],
        "close_to_close_log_return",
    ] = 1.0

    moments, _, tails = build_distribution_tables(
        make_bar_features(),
        sessions,
    )

    daily = moments.loc[
        moments["return_type"].eq(
            "close_to_close_log_return"
        )
    ].iloc[0]

    daily_tails = tails.loc[
        tails["return_type"].eq(
            "close_to_close_log_return"
        )
    ]

    assert daily["maximum"] == 1.0
    assert daily_tails[
        "two_sided_count"
    ].max() >= 1


def test_missing_required_column_is_rejected() -> None:
    bars = make_bar_features().drop(
        columns=["intraday_log_return"]
    )

    with pytest.raises(
        StylizedFactsError,
        match="missing required columns",
    ):
        build_distribution_tables(
            bars,
            make_session_features(),
        )
