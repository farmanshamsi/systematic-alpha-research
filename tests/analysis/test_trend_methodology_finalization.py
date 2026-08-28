"""Tests for the frozen trend timing and positioning finalization."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.causal_bar_execution import (
    apply_causal_next_open_overnight_flat,
)
from systematic_alpha.analysis.trend_methodology_finalization import (
    APPROVED_ARTIFACT_NAMES,
    FINAL_TIMING,
    PARITY_COLUMNS,
    ROBUSTNESS_COLUMNS,
    SENSITIVITY_COLUMNS,
    TIMING_COLUMNS,
    WALK_FORWARD_COLUMNS,
    TrendMethodologyError,
    TrendMethodologyResults,
    apply_next_open_overnight_flat,
    apply_saved_timing,
    build_model_observations,
    sequential_next_open_replay,
    write_trend_methodology_artifacts,
)


def timing_input() -> pd.DataFrame:
    return pd.DataFrame(
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
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "signal": [1, -1, 1, -1, 1, 0],
            "signal_available": [True] * 6,
        }
    )


def strategy_input() -> pd.DataFrame:
    count = 80
    close = 100.0 + np.sin(np.arange(count) / 4.0) * 2.0
    returns = pd.Series(close).pct_change()
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-02 14:30Z", periods=count, freq="15min"),
            "symbol": ["SPY"] * count,
            "session_date": [f"2025-01-{2 + index // 20:02d}" for index in range(count)],
            "open": close - 0.1,
            "close": close,
            "close_to_close_simple_return": returns,
        }
    )


def empty_results() -> TrendMethodologyResults:
    return TrendMethodologyResults(
        timing_comparison=pd.DataFrame(columns=TIMING_COLUMNS),
        walk_forward=pd.DataFrame(columns=WALK_FORWARD_COLUMNS),
        robustness=pd.DataFrame(columns=ROBUSTNESS_COLUMNS),
        long_flat_sensitivity=pd.DataFrame(columns=SENSITIVITY_COLUMNS),
        replay_parity=pd.DataFrame(columns=PARITY_COLUMNS),
    )


def test_next_open_timing_is_causal_and_forces_session_close_flat() -> None:
    result = apply_next_open_overnight_flat(
        timing_input(),
        cost_bps_per_turnover=1.0,
    )
    assert result["position"].tolist() == [0, 1, -1, 1, -1, 1]
    assert result.loc[result["is_session_close"], "ending_position"].eq(0).all()
    assert result["turnover"].tolist() == [0.0, 1.0, 3.0, 1.0, 2.0, 3.0]
    assert result.loc[1, "pnl_proxy_return"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert result.loc[2, "pnl_proxy_return"] == pytest.approx(102.5 / 102.0 - 1.0)
    assert result.loc[2, "transaction_cost"] == pytest.approx(0.0003)


def test_sequential_replay_matches_vectorized_accounting() -> None:
    source = timing_input()
    batch = apply_next_open_overnight_flat(source, cost_bps_per_turnover=2.5)
    replay = sequential_next_open_replay(source, cost_bps_per_turnover=2.5)
    assert np.array_equal(batch["position"].to_numpy(), replay["position"].to_numpy())
    assert np.array_equal(
        batch["ending_position"].to_numpy(), replay["ending_position"].to_numpy()
    )
    for column in (
        "turnover",
        "gross_strategy_return",
        "transaction_cost",
        "net_strategy_return",
    ):
        assert np.allclose(batch[column], replay[column], atol=1.0e-12, rtol=0.0)


def test_trend_wrapper_is_exactly_identical_to_shared_timing() -> None:
    source = timing_input()
    shared = apply_causal_next_open_overnight_flat(
        source, cost_bps_per_turnover=2.5
    )
    wrapped = apply_next_open_overnight_flat(
        source, cost_bps_per_turnover=2.5
    )

    pd.testing.assert_frame_equal(wrapped, shared, check_exact=True)


def test_next_open_timing_rejects_negative_cost() -> None:
    with pytest.raises(TrendMethodologyError, match="Cost"):
        apply_next_open_overnight_flat(timing_input(), cost_bps_per_turnover=-1.0)


def test_long_flat_model_never_creates_short_signal_or_position() -> None:
    observations = build_model_observations(
        strategy_input(),
        model_id="price_ratio_long_flat",
    )
    assert observations["signal"].isin((0, 1)).all()
    assert observations["position"].isin((0, 1)).all()


def test_historical_timing_reprices_cost_without_changing_gross() -> None:
    source = build_model_observations(
        strategy_input(),
        model_id="price_ratio_long_short_neutral",
    )
    gross = source["gross_strategy_return"].copy()
    result = apply_saved_timing(source, cost_bps_per_turnover=5.0)
    assert result["gross_strategy_return"].equals(gross)
    assert np.allclose(result["transaction_cost"], result["turnover"] * 0.0005)


def test_artifact_writer_is_exact_hashed_and_overwrite_protected(tmp_path) -> None:
    output = tmp_path / "methodology"
    paths = write_trend_methodology_artifacts(
        empty_results(),
        output,
        source_dataset_id="development",
        source_sha256="a" * 64,
    )
    assert tuple(path.name for path in paths) == APPROVED_ARTIFACT_NAMES
    assert {path.name for path in output.iterdir()} == set(APPROVED_ARTIFACT_NAMES)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert manifest["locked_period_accessed"] is False
    with pytest.raises(FileExistsError):
        write_trend_methodology_artifacts(
            empty_results(),
            output,
            source_dataset_id="development",
            source_sha256="a" * 64,
        )


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(TrendMethodologyError, match="Unknown frozen model"):
        build_model_observations(strategy_input(), model_id="winner")


def test_final_timing_identifier_is_frozen() -> None:
    assert FINAL_TIMING == "next_bar_open_overnight_flat_v1"
