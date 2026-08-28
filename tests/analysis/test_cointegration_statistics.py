"""Statistical contracts for Day 14 cointegration feasibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_alpha.analysis.cointegration_feasibility import (
    run_cointegration_feasibility,
)


def make_cointegrated_bars() -> pd.DataFrame:
    """Build one strongly cointegrated pair and one independent series."""

    sessions = pd.bdate_range(
        "2020-01-02",
        "2025-12-31",
    )
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(
                f"{session.date()} {time}",
                tz="UTC",
            )
            for session in sessions
            for time in ("14:30:00", "14:45:00")
        ]
    )

    rng = np.random.default_rng(14)
    observations = len(timestamps)

    common = 5.50 + np.cumsum(
        rng.normal(0.0, 0.004, observations)
    )
    independent = 5.00 + np.cumsum(
        rng.normal(0.0, 0.005, observations)
    )

    residual = np.empty(observations)
    residual[0] = 0.0

    for index in range(1, observations):
        residual[index] = (
            0.70 * residual[index - 1]
            + rng.normal(0.0, 0.002)
        )

    log_prices = {
        "QQQ": common,
        "SPY": 0.30 + 1.10 * common + residual,
        "IWM": independent,
    }

    rows: list[dict[str, object]] = []

    for symbol, values in log_prices.items():
        for timestamp, log_price in zip(
            timestamps,
            values,
            strict=True,
        ):
            close = float(np.exp(log_price))

            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close,
                    "high": close * 1.0001,
                    "low": close * 0.9999,
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


def test_statistical_engine_detects_only_predeclared_cointegration() -> None:
    results = run_cointegration_feasibility(
        make_cointegrated_bars()
    )

    assert len(
        results.pair_input_diagnostics
    ) == 3
    assert len(
        results.series_integration_diagnostics
    ) == 6
    assert len(
        results.cointegration_diagnostics
    ) == 3
    assert len(
        results.fold_stability_diagnostics
    ) == 12

    integration = (
        results.series_integration_diagnostics
        .groupby("symbol")["plausibly_i1"]
        .all()
    )
    assert integration.to_dict() == {
        "IWM": True,
        "QQQ": True,
        "SPY": True,
    }

    cointegration = (
        results.cointegration_diagnostics
        .set_index("pair_id")
    )

    assert bool(
        cointegration.loc[
            "SPY_QQQ",
            "holm_reject",
        ]
    )
    assert not bool(
        cointegration.loc[
            "SPY_IWM",
            "holm_reject",
        ]
    )
    assert not bool(
        cointegration.loc[
            "QQQ_IWM",
            "holm_reject",
        ]
    )

    assert 1.05 < float(
        cointegration.loc[
            "SPY_QQQ",
            "beta",
        ]
    ) < 1.15

    spy_qqq_folds = (
        results.fold_stability_diagnostics.loc[
            lambda frame: frame[
                "pair_id"
            ].eq("SPY_QQQ")
        ]
    )

    assert len(spy_qqq_folds) == 4
    assert (
        spy_qqq_folds[
            "beta_sign_stable"
        ].astype(bool)
    ).all()
    assert (
        spy_qqq_folds[
            "beta_relative_deviation"
        ] <= 0.25
    ).all()
    assert (
        spy_qqq_folds[
            "test_residual_stationary"
        ].astype(bool).sum()
        >= 3
    )


def test_ou_and_final_eligibility_are_conditional() -> None:
    results = run_cointegration_feasibility(
        make_cointegrated_bars()
    )

    ou = results.ou_diagnostics.set_index(
        "pair_id"
    )

    assert ou["attempted"].astype(bool).to_dict() == {
        "SPY_QQQ": True,
        "SPY_IWM": False,
        "QQQ_IWM": False,
    }
    assert bool(ou.loc["SPY_QQQ", "ou_pass"])
    assert 0.0 < float(
        ou.loc["SPY_QQQ", "phi"]
    ) < 1.0
    assert 1.0 <= float(
        ou.loc["SPY_QQQ", "half_life_bars"]
    ) <= 130.0

    assert (
        ou.loc[
            ["SPY_IWM", "QQQ_IWM"],
            "rejection_reason",
        ]
        == "prior_statistical_gates_failed"
    ).all()

    eligibility = (
        results.pair_eligibility.set_index(
            "pair_id"
        )
    )

    assert eligibility[
        "eligible"
    ].astype(bool).to_dict() == {
        "SPY_QQQ": True,
        "SPY_IWM": False,
        "QQQ_IWM": False,
    }
    assert (
        eligibility.loc[
            "SPY_QQQ",
            "rejection_reasons",
        ]
        == ""
    )
    for pair_id in ("SPY_IWM", "QQQ_IWM"):
        reasons = str(
            eligibility.loc[
                pair_id,
                "rejection_reasons",
            ]
        )

        assert "holm_cointegration_failed" in reasons
        assert "ou_not_attempted" in reasons
        assert "ou_failed" not in reasons
