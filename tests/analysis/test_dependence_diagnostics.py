"""Tests for Day 05 dependence diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.dependence_diagnostics import (
    DependenceDiagnosticError,
    build_daily_return_panel,
    build_pairwise_dependence_table,
    build_regime_dependence_tables,
    build_rolling_dependence_table,
    build_tail_dependence_table,
)


def make_session_features(
    observations: int = 500,
) -> pd.DataFrame:
    """Create correlated synthetic daily returns."""

    rng = np.random.default_rng(42)

    common = rng.standard_t(
        df=5,
        size=observations,
    )

    noise_a = rng.normal(
        scale=0.35,
        size=observations,
    )

    noise_b = rng.normal(
        scale=0.45,
        size=observations,
    )

    spy = 0.01 * common
    qqq = 0.01 * (
        0.85 * common + noise_a
    )
    iwm = 0.01 * (
        0.65 * common + noise_b
    )

    dates = pd.date_range(
        "2020-01-02",
        periods=observations,
        freq="B",
    )

    rows = []

    for symbol, values in {
        "SPY": spy,
        "QQQ": qqq,
        "IWM": iwm,
    }.items():
        for date, value in zip(
            dates,
            values,
            strict=True,
        ):
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": (
                        date.strftime("%Y-%m-%d")
                    ),
                    "close_to_close_log_return": (
                        value
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_daily_volatility(
    observations: int = 500,
) -> pd.DataFrame:
    """Create a deterministic SPY volatility regime series."""

    dates = pd.date_range(
        "2020-01-02",
        periods=observations,
        freq="B",
    )

    volatility = np.linspace(
        0.10,
        0.60,
        observations,
    )

    return pd.DataFrame(
        {
            "symbol": "SPY",
            "session_date": (
                dates.strftime("%Y-%m-%d")
            ),
            "annualized_total_realized_volatility": (
                volatility
            ),
        }
    )


def test_daily_panel_is_synchronized() -> None:
    panel = build_daily_return_panel(
        make_session_features()
    )

    assert list(panel.columns) == [
        "IWM",
        "QQQ",
        "SPY",
    ]

    assert panel.shape == (500, 3)


def test_pairwise_dependence_detects_correlation() -> None:
    panel = build_daily_return_panel(
        make_session_features()
    )

    table = build_pairwise_dependence_table(
        panel
    )

    assert len(table) == 3

    assert (
        table["pearson_correlation"] > 0.5
    ).all()

    assert (
        table["spearman_correlation"] > 0.5
    ).all()

    assert (
        table["kendall_tau"] > 0.3
    ).all()


def test_tail_dependence_uses_tail_probability() -> None:
    observations = 1_000

    values = np.arange(
        observations,
        dtype=float,
    )

    panel = pd.DataFrame(
        {
            "SPY": values,
            "QQQ": values,
        }
    )

    table = build_tail_dependence_table(
        panel,
        tail_probabilities=(0.10,),
    )

    result = table.iloc[0]

    assert np.isclose(
        result["lower_coexceedance_ratio"],
        1.0,
        atol=0.02,
    )

    assert np.isclose(
        result["upper_coexceedance_ratio"],
        1.0,
        atol=0.02,
    )


def test_rolling_dependence_has_expected_rows() -> None:
    panel = build_daily_return_panel(
        make_session_features(
            observations=100
        )
    )

    table = build_rolling_dependence_table(
        panel,
        windows=(20,),
    )

    assert len(table) == 3 * (100 - 20 + 1)


def test_regime_tables_contain_both_regimes() -> None:
    panel = build_daily_return_panel(
        make_session_features()
    )

    dependence, tails, definition = (
        build_regime_dependence_tables(
            panel,
            make_daily_volatility(),
            stress_quantile=0.80,
            tail_probabilities=(0.05,),
        )
    )

    assert set(
        dependence["sample"]
    ) == {
        "normal_volatility",
        "high_volatility",
    }

    assert set(
        tails["sample"]
    ) == {
        "normal_volatility",
        "high_volatility",
    }

    assert (
        definition.iloc[0][
            "stress_observations"
        ]
        == 100
    )


def test_missing_return_column_is_rejected() -> None:
    with pytest.raises(
        DependenceDiagnosticError,
        match="missing required columns",
    ):
        build_daily_return_panel(
            make_session_features().drop(
                columns=[
                    "close_to_close_log_return"
                ]
            )
        )
