"""Tests for the fail-closed one-time locked final-test protocol."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.locked_final_test import (
    APPROVED_ARTIFACT_NAMES,
    AUTHORIZATION_CODE,
    FROZEN_MODELS,
    LockedFinalTestError,
    evaluate_locked_final_test,
    require_authorization,
    validate_locked_bars,
    write_locked_final_test_artifacts,
)


def bars_for_sessions(session_dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol_order, symbol in enumerate(("SPY", "QQQ", "IWM")):
        sequence = 0
        for session_date in session_dates:
            start = pd.Timestamp(f"{session_date} 14:30:00Z")
            for bar in range(26):
                price = (
                    100.0
                    + 10.0 * symbol_order
                    + 0.03 * sequence
                    + 0.8 * np.sin(sequence / 7.0 + symbol_order)
                )
                rows.append(
                    {
                        "timestamp": start + pd.Timedelta(minutes=15 * bar),
                        "symbol": symbol,
                        "open": price - 0.02,
                        "high": price + 0.08,
                        "low": price - 0.08,
                        "close": price,
                        "volume": 1000.0 + sequence,
                        "trade_count": 100 + sequence,
                        "vwap": price - 0.01,
                        "source": "synthetic",
                        "feed": "sip",
                    }
                )
                sequence += 1
    return pd.DataFrame.from_records(rows)


def test_authorization_code_is_exact() -> None:
    with pytest.raises(LockedFinalTestError, match="Exact"):
        require_authorization(None)
    with pytest.raises(LockedFinalTestError, match="Exact"):
        require_authorization("yes")
    require_authorization(AUTHORIZATION_CODE)


def test_locked_scope_rejects_any_outside_row() -> None:
    source = bars_for_sessions(["2026-01-02"])
    source.loc[0, "timestamp"] = pd.Timestamp("2025-12-31 20:45Z")
    with pytest.raises(LockedFinalTestError, match="outside"):
        validate_locked_bars(source)


def test_every_frozen_result_is_evaluated_and_initially_flat(tmp_path) -> None:
    development = bars_for_sessions(
        [
            "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18",
            "2025-12-19", "2025-12-22", "2025-12-23", "2025-12-29",
            "2025-12-30",
        ]
    )
    locked = bars_for_sessions(["2026-01-02", "2026-01-05"])
    results = evaluate_locked_final_test(development, locked)
    assert tuple(results.performance["model_id"]) == FROZEN_MODELS
    assert results.performance["initial_position"].eq(0).all()
    assert results.performance["overnight_position_violations"].eq(0).all()
    assert len(results.session_returns) == 2

    output = tmp_path / "day25_final_test"
    paths = write_locked_final_test_artifacts(
        results,
        locked,
        output,
        frozen_hashes={"development": "a" * 64},
        request_metadata={"feed": "sip"},
    )
    assert tuple(path.name for path in paths) == APPROVED_ARTIFACT_NAMES
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["one_time_run_complete"] is True
    for name, expected in manifest["hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError, match="already exists"):
        write_locked_final_test_artifacts(
            results,
            locked,
            output,
            frozen_hashes={"development": "a" * 64},
            request_metadata={"feed": "sip"},
        )
