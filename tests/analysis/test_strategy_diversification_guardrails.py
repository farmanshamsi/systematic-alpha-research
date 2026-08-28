"""Fail-closed guardrails for Day 15 strategy diversification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.strategy_diversification as diversification
from tests.analysis.test_strategy_diversification_inputs import (
    make_canonical_bars,
)
from tests.analysis.test_strategy_diversification_statistics import (
    make_weakly_correlated_panel,
)


def panel_to_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert a synthetic panel to validated long-form inputs."""

    return (
        panel.rename_axis("session_date")
        .reset_index()
        .melt(
            id_vars="session_date",
            var_name="sleeve_id",
            value_name="session_return",
        )
    )


def test_malformed_and_2026_bar_dates_are_rejected() -> None:
    malformed = make_canonical_bars()
    malformed["timestamp"] = malformed["timestamp"].astype("object")
    malformed.loc[0, "timestamp"] = "not-a-date"

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="malformed timestamps",
    ):
        diversification.build_strategy_diversification_inputs(malformed)

    future = make_canonical_bars()
    future.loc[0, "timestamp"] = pd.Timestamp(
        "2026-01-02 14:30:00",
        tz="UTC",
    )

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="2026 data are forbidden",
    ):
        diversification.build_strategy_diversification_inputs(future)


def test_out_of_development_and_lowercase_symbols_are_rejected() -> None:
    old = make_canonical_bars()
    old.loc[0, "timestamp"] = pd.Timestamp(
        "2020-01-01 14:30:00",
        tz="UTC",
    )
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="development dates",
    ):
        diversification.build_strategy_diversification_inputs(old)

    lowercase = make_canonical_bars()
    lowercase["symbol"] = lowercase["symbol"].str.lower()
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="canonical uppercase",
    ):
        diversification.build_strategy_diversification_inputs(lowercase)


def test_duplicate_malformed_missing_and_nonfinite_returns_are_rejected() -> None:
    rows = panel_to_long(make_weakly_correlated_panel())

    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="Duplicate sleeve/session",
    ):
        diversification.build_exact_return_panel(duplicate)

    malformed = rows.copy()
    malformed["session_date"] = malformed["session_date"].astype("object")
    malformed.loc[0, "session_date"] = "not-a-date"
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="malformed dates",
    ):
        diversification.build_exact_return_panel(malformed)

    for invalid in (np.nan, np.inf, -np.inf):
        nonfinite = rows.copy()
        nonfinite.loc[0, "session_return"] = invalid
        with pytest.raises(
            diversification.StrategyDiversificationError,
            match="finite and non-missing",
        ):
            diversification.build_exact_return_panel(nonfinite)

    impossible = rows.copy()
    impossible.loc[0, "session_return"] = -1.0
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="greater than -1",
    ):
        diversification.build_exact_return_panel(impossible)


def test_missing_required_long_form_columns_and_sleeves_are_rejected() -> None:
    rows = panel_to_long(make_weakly_correlated_panel())

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="missing required columns",
    ):
        diversification.build_exact_return_panel(
            rows.drop(columns="session_return")
        )

    missing_sleeve = rows.loc[
        ~rows["sleeve_id"].eq(diversification.SLEEVE_IDS[-1])
    ]
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="exactly the six frozen sleeves",
    ):
        diversification.build_exact_return_panel(missing_sleeve)


def test_calendar_mismatch_is_rejected_without_fill_or_interpolation() -> None:
    rows = panel_to_long(make_weakly_correlated_panel())
    missing_row = rows.drop(index=rows.index[0]).reset_index(drop=True)

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="same session calendar",
    ):
        diversification.build_exact_return_panel(missing_row)


def test_zero_and_near_zero_variance_sleeves_fail_closed() -> None:
    panel = make_weakly_correlated_panel()
    panel[diversification.SLEEVE_IDS[0]] = 0.001

    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="Zero or near-zero variance",
    ):
        diversification.analyze_strategy_diversification_panel(panel)

    near_zero = make_weakly_correlated_panel()
    near_zero[diversification.SLEEVE_IDS[0]] = np.linspace(
        0.0,
        1e-9,
        len(near_zero),
    )
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match="Zero or near-zero variance",
    ):
        diversification.analyze_strategy_diversification_panel(near_zero)


def test_position_delay_validation_fails_closed() -> None:
    valid = pd.DataFrame(
        {
            "signal": [0, 1, -1, 1],
            "position": [0, 0, 1, -1],
        }
    )
    diversification._validate_position_delay(valid)

    invalid = valid.copy()
    invalid.loc[2, "position"] = -1
    with pytest.raises(
        diversification.StrategyDiversificationError,
        match=r"position\[t\] == signal\[t-1\]",
    ):
        diversification._validate_position_delay(invalid)
