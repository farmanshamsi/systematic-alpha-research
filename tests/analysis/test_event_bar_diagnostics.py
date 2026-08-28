"""Tests for the D05-T03 event-bar engineering audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.event_bar_diagnostics import (
    EventBarDiagnosticError,
    build_event_bar_diagnostics,
    build_time_bars_from_trades,
    calculate_event_thresholds,
    normalize_trade_sample,
)


def make_trades(
    observations: int = 40,
) -> pd.DataFrame:
    """Create deterministic canonical trades."""

    timestamps = pd.date_range(
        "2025-12-15T14:30:00Z",
        periods=observations,
        freq="500ms",
    )

    prices = (
        100.0
        + np.sin(
            np.arange(observations) / 4.0
        )
        * 0.10
    )

    sizes = (
        np.arange(observations) % 5
        + 1
    ).astype(float)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "id": [
                f"trade-{index}"
                for index in range(
                    observations
                )
            ],
            "price": prices,
            "size": sizes,
            "source": "test",
            "feed": "test",
        }
    )


def test_time_bars_preserve_source_quantities() -> None:
    trades = make_trades()

    bars = build_time_bars_from_trades(
        trades,
        rule="2s",
    )

    assert int(
        bars["trade_count"].sum()
    ) == len(trades)

    assert np.isclose(
        float(bars["volume"].sum()),
        float(trades["size"].sum()),
    )

    assert np.isclose(
        float(
            bars["dollar_value"].sum()
        ),
        float(
            (
                trades["price"]
                * trades["size"]
            ).sum()
        ),
    )


def test_thresholds_are_positive() -> None:
    thresholds = calculate_event_thresholds(
        make_trades(),
        target_bar_count=10,
    )

    assert thresholds["tick"] == 4.0
    assert thresholds["volume"] > 0.0
    assert thresholds["dollar"] > 0.0


def test_bundle_contains_four_sampling_methods() -> None:
    bundle = build_event_bar_diagnostics(
        make_trades(),
        target_bar_count=10,
    )

    assert set(
        bundle.comparison[
            "sampling_method"
        ]
    ) == {
        "time_2s",
        "tick",
        "volume",
        "dollar",
    }


def test_all_methods_conserve_trades() -> None:
    bundle = build_event_bar_diagnostics(
        make_trades(),
        target_bar_count=10,
    )

    assert bundle.conservation[
        "trade_count_error"
    ].eq(0).all()

    assert np.allclose(
        bundle.conservation[
            "volume_error"
        ],
        0.0,
    )

    assert np.allclose(
        bundle.conservation[
            "dollar_value_error"
        ],
        0.0,
        atol=1.0e-8,
    )


def test_duplicate_trade_ids_are_rejected() -> None:
    trades = make_trades()

    trades.loc[1, "id"] = trades.loc[
        0,
        "id",
    ]

    with pytest.raises(
        EventBarDiagnosticError,
        match="Duplicate trade identifiers",
    ):
        normalize_trade_sample(trades)
