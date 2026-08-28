"""Focused tests for the frozen Day 26 profitability experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from systematic_alpha.analysis.phase2_profitability import (
    Phase2ProfitabilityError,
    apply_ou_cost_margin_gate,
    apply_ou_next_open_accounting,
    audit_development_data,
    build_persistent_hysteresis_signal,
)


def test_persistent_hysteresis_requires_four_confirmations_and_half_band_exit() -> None:
    ratios = [1.002, 1.002, 1.002, 1.002, 1.0008, 1.0005, 1.002]
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-02 14:30Z", periods=len(ratios), freq="15min"),
            "symbol": ["SPY"] * len(ratios),
            "ma_price_ratio": ratios,
            "signal_available": [True] * len(ratios),
            "signal": [0] * len(ratios),
        }
    )
    result = build_persistent_hysteresis_signal(source)
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert result["confirmation_bars"].tolist() == [1, 2, 3, 4, 4, 0, 1]


def test_ou_cost_margin_gate_blocks_small_proxy_and_allows_large_proxy() -> None:
    timestamps = pd.date_range("2025-01-02 14:30Z", periods=5, freq="15min")
    source = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * 5,
            "session_date": [pd.Timestamp("2025-01-02", tz="UTC")] * 5,
            "is_session_close_bar": [False, False, False, False, True],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 105.0],
            "log_price_residual": [-0.0005, -0.004, -0.003, -0.002, -0.001],
            "ou_equilibrium": [0.0] * 5,
            "ou_phi": [0.5] * 5,
            "ou_zscore": [-3.0, -3.0, -2.0, -1.0, -0.2],
            "regime_eligible": [True] * 5,
            "signal_available": [True] * 5,
            "close_to_close_simple_return": [float("nan"), 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = apply_ou_cost_margin_gate(
        source, execution_reset_timestamps=(timestamps[0],)
    )
    assert not bool(result.iloc[0]["entry_gate"])
    assert bool(result.iloc[1]["entry_gate"])
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["position"].tolist() == [0, 0, 1, 1, 1]

    raw_columns = (
        "log_price_residual",
        "ou_equilibrium",
        "ou_phi",
        "ou_zscore",
        "regime_eligible",
        "signal_available",
        "entry_gate",
        "signal",
        "holding_bars",
        "signal_score",
    )
    timed = apply_ou_next_open_accounting(result)
    pd.testing.assert_frame_equal(
        timed.loc[:, raw_columns],
        result.loc[:, raw_columns],
        check_exact=True,
    )
    assert timed["position"].tolist() == [0, 0, 1, 1, 1]
    assert timed.loc[4, "pnl_proxy_return"] == pytest.approx(
        105.0 / 104.0 - 1.0
    )


def test_ou_next_open_accounting_charges_terminal_liquidation() -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 14:30Z",
                    "2025-01-02 14:45Z",
                    "2025-01-02 15:00Z",
                    "2025-01-03 14:30Z",
                    "2025-01-03 14:45Z",
                    "2025-01-03 15:00Z",
                ],
                utc=True,
            ),
            "symbol": ["SPY"] * 6,
            "session_date": ["2025-01-02"] * 3 + ["2025-01-03"] * 3,
            "open": [100.0, 101.0, 102.0, 110.0, 111.0, 112.0],
            "close": [100.5, 101.5, 103.0, 110.5, 111.5, 113.0],
            "signal": [1, 1, 0, 0, 1, 0],
            "signal_available": [True] * 6,
        }
    )
    result = apply_ou_next_open_accounting(source)
    assert result["open_turnover"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert result["close_turnover"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert result["turnover"].sum() == pytest.approx(4.0)
    assert result.loc[result["is_session_close"], "ending_position"].eq(0).all()
    assert result.loc[2, "pnl_proxy_return"] == pytest.approx(
        103.0 / 102.0 - 1.0
    )
    assert result.loc[5, "pnl_proxy_return"] == pytest.approx(
        113.0 / 112.0 - 1.0
    )


def test_data_audit_rejects_consumed_or_later_rows() -> None:
    source = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-02 14:30Z")],
            "symbol": ["SPY"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
            "trade_count": [100],
            "vwap": [100.2],
            "source": ["alpaca"],
            "feed": ["sip"],
        }
    )
    with pytest.raises(Phase2ProfitabilityError, match="prohibited 2026"):
        audit_development_data(
            source,
            source_dataset_id="locked.parquet",
            source_sha256="0" * 64,
        )
