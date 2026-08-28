"""Deterministic synthetic inputs shared by Day 16 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS


def make_day16_panel(*, mean_return: float = 0.0002) -> pd.DataFrame:
    """Build a small non-degenerate panel spanning every frozen fold."""

    sessions = pd.date_range(
        "2020-01-02",
        "2025-12-31",
        freq="14D",
        tz="UTC",
        name="session_date",
    )
    time = np.arange(len(sessions), dtype="float64")
    records: dict[str, np.ndarray] = {}
    for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
        scale = 0.0015 + sleeve_order * 0.00055
        records[sleeve_id] = (
            mean_return
            + scale * np.sin(time / (2.3 + sleeve_order * 0.4))
            + 0.0007 * np.cos(time / (1.7 + sleeve_order * 0.3))
        )
    return pd.DataFrame(records, index=sessions)
