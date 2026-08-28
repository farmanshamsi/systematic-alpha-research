"""Input and frozen-execution contracts for Day 15."""

from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.strategy_diversification as diversification
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS as DAY10_CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS as DAY10_EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS as DAY10_TREND_RATIO_PARAMETERS,
)


def make_canonical_bars(
    *,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
) -> pd.DataFrame:
    """Build compact, complete 15-minute synthetic source sessions."""

    sessions = pd.bdate_range("2020-01-02", periods=60)
    rows: list[dict[str, object]] = []
    symbol_phase = {"SPY": 0.0, "QQQ": 1.1, "IWM": 2.2}

    for symbol in symbols:
        phase = symbol_phase.get(symbol, 0.5)
        close = 100.0 + 20.0 * phase
        sequence = 0
        for session in sessions:
            timestamps = pd.date_range(
                f"{session.date()} 14:30:00+00:00",
                periods=26,
                freq="15min",
            )
            for timestamp in timestamps:
                simple_return = (
                    0.0035 * np.sin(sequence / 7.0 + phase)
                    + 0.0015 * np.cos(sequence / 19.0 + 2.0 * phase)
                )
                close *= 1.0 + simple_return
                open_price = close * (1.0 - 0.0002)
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "open": open_price,
                        "high": close * 1.0005,
                        "low": open_price * 0.9995,
                        "close": close,
                        "volume": float(1_000 + sequence % 50),
                        "trade_count": 100 + sequence % 20,
                        "vwap": (open_price + close) / 2.0,
                        "source": "synthetic",
                        "feed": "test",
                    }
                )
                sequence += 1

    return (
        pd.DataFrame(rows)
        .sort_values(["symbol", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def test_frozen_sleeves_have_exact_order_and_day10_contracts() -> None:
    sleeves = diversification.build_frozen_sleeves()

    assert len(sleeves) == 6
    assert all(is_dataclass(sleeve) for sleeve in sleeves)
    assert all(sleeve.__dataclass_params__.frozen for sleeve in sleeves)
    assert tuple(sleeve.sleeve_id for sleeve in sleeves) == (
        "trend_ratio_spy",
        "trend_ratio_qqq",
        "trend_ratio_iwm",
        "ema_macd_spy",
        "ema_macd_qqq",
        "ema_macd_iwm",
    )
    assert tuple((sleeve.strategy, sleeve.symbol) for sleeve in sleeves) == (
        ("trend_ratio", "SPY"),
        ("trend_ratio", "QQQ"),
        ("trend_ratio", "IWM"),
        ("ema_macd", "SPY"),
        ("ema_macd", "QQQ"),
        ("ema_macd", "IWM"),
    )
    assert all(sleeve.frequency == "15min" for sleeve in sleeves)
    assert diversification.TREND_RATIO_PARAMETERS is (
        DAY10_TREND_RATIO_PARAMETERS
    )
    assert diversification.EMA_MACD_PARAMETERS is DAY10_EMA_MACD_PARAMETERS
    assert diversification.CONFIGURATION_IDS == DAY10_CONFIGURATION_IDS


def test_input_builder_executes_six_sleeves_and_compounds_sessions() -> None:
    bars = make_canonical_bars()
    inputs = diversification.build_strategy_diversification_inputs(bars)

    assert list(inputs.session_return_panel.columns) == list(
        diversification.SLEEVE_IDS
    )
    assert inputs.session_return_panel.index.name == "session_date"
    assert len(inputs.session_return_panel) == 60
    assert len(inputs.sleeve_session_returns) == 6 * 60
    assert len(inputs.sleeve_input_diagnostics) == 6
    assert inputs.sleeve_input_diagnostics["sessions"].eq(60).all()
    assert inputs.sleeve_input_diagnostics["observations"].eq(
        60 * 26
    ).all()
    assert inputs.sleeve_input_diagnostics[
        "non_degenerate"
    ].astype(bool).all()
    assert inputs.sleeve_input_diagnostics[
        "finite_returns"
    ].astype(bool).all()
    assert inputs.sleeve_input_diagnostics[
        "exact_calendar_aligned"
    ].astype(bool).all()


def test_session_return_uses_simple_return_product() -> None:
    result = diversification.compound_intraday_returns(
        pd.Series([-0.10, 0.20], dtype="float64")
    )
    assert result == pytest.approx((1.0 - 0.10) * (1.0 + 0.20) - 1.0)


@pytest.mark.parametrize(
    "column",
    [
        "timestamp",
        "symbol",
        "close",
        "trade_count",
    ],
)
def test_missing_required_bar_columns_are_rejected(column: str) -> None:
    bars = make_canonical_bars().drop(columns=column)

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="missing required columns",
    ):
        diversification.build_strategy_diversification_inputs(bars)


def test_unexpected_symbol_is_rejected_before_strategy_execution() -> None:
    bars = make_canonical_bars(symbols=("SPY", "QQQ", "DIA"))

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="exactly SPY, QQQ, and IWM",
    ):
        diversification.build_strategy_diversification_inputs(bars)
