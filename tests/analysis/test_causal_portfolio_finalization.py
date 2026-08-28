"""Tests for the Day 25 causal portfolio dependency correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.causal_portfolio_finalization import (
    CausalPortfolioFinalizationError,
    build_causal_session_returns,
    write_causal_portfolio_artifacts,
)
from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS
from systematic_alpha.analysis.trend_methodology_finalization import FINAL_TIMING


def feature_panel() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol_order, symbol in enumerate(("SPY", "QQQ", "IWM")):
        count = 80
        close = 100.0 + symbol_order * 10.0 + np.sin(np.arange(count) / 4.0) * 2.0
        parts.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-01-02 14:30Z", periods=count, freq="15min"
                    ),
                    "symbol": [symbol] * count,
                    "session_date": [
                        f"2025-01-{2 + index // 20:02d}" for index in range(count)
                    ],
                    "open": close - 0.1,
                    "close": close,
                    "close_to_close_simple_return": pd.Series(close).pct_change(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_final_timing_builds_all_sleeves_on_one_calendar() -> None:
    result = build_causal_session_returns(feature_panel())
    assert tuple(result["sleeve_id"].drop_duplicates()) == SLEEVE_IDS
    assert len(result) == 4 * len(SLEEVE_IDS)
    assert result["timing_convention"].eq(FINAL_TIMING).all()
    assert result["cost_bps_per_turnover"].eq(1.0).all()
    assert result.groupby("sleeve_id")["session_date"].nunique().eq(4).all()
    assert np.isfinite(result["session_return"]).all()


def test_writer_rejects_noncontract_directory_before_writing(tmp_path) -> None:
    with pytest.raises(CausalPortfolioFinalizationError, match="must be named"):
        write_causal_portfolio_artifacts(None, tmp_path / "winner")  # type: ignore[arg-type]
