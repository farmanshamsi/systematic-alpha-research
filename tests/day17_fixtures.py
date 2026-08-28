"""Shared deterministic fixtures for Day 17 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_day17_development_bars() -> pd.DataFrame:
    """Build compact complete-session bars across all development years."""

    rng = np.random.default_rng(1701)
    rows: list[dict[str, object]] = []
    symbol_offsets = {"SPY": 0.0, "QQQ": 0.15, "IWM": -0.10}
    for symbol, offset in symbol_offsets.items():
        state = 0.0
        level = 4.6 + offset
        for year in range(2020, 2026):
            sessions = pd.bdate_range(f"{year}-02-03", periods=6)
            for session in sessions:
                for bar in range(26):
                    state = 0.45 * state + rng.normal(0.0, 0.0025)
                    level += rng.normal(0.0, 0.00015)
                    close = float(np.exp(level + state))
                    timestamp = pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(
                        hours=14, minutes=30 + 15 * bar
                    )
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "open": close,
                            "high": close * 1.0002,
                            "low": close * 0.9998,
                            "close": close,
                            "volume": float(1_000 + 10 * bar),
                            "trade_count": 100 + bar,
                            "vwap": close * (1.0 - 0.0001 * np.sign(state)),
                            "source": "synthetic",
                            "feed": "test",
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)
