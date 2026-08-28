"""Leakage and input-boundary guardrails for Day 14."""

from __future__ import annotations

import pandas as pd
import pytest

from systematic_alpha.analysis.cointegration_feasibility import (
    CointegrationFeasibilityError,
    run_cointegration_feasibility,
)
from tests.analysis.test_cointegration_statistics import (
    make_cointegrated_bars,
)


def test_unmatched_locked_period_bar_is_rejected() -> None:
    bars = make_cointegrated_bars()

    locked_bar = bars.loc[
        bars["symbol"].eq("SPY")
    ].iloc[-1].copy()

    locked_bar["timestamp"] = pd.Timestamp(
        "2026-01-02 14:30:00",
        tz="UTC",
    )

    contaminated = pd.concat(
        [
            bars,
            pd.DataFrame([locked_bar]),
        ],
        ignore_index=True,
    ).sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    with pytest.raises(
        CointegrationFeasibilityError,
        match="locked 2026 period",
    ):
        run_cointegration_feasibility(
            contaminated
        )
