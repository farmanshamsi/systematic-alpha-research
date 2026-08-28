"""Tests for the representative five-session event-time experiment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.event_time_finalization import (
    APPROVED_ARTIFACT_NAMES,
    SESSION_DATES,
    EventTimeFinalizationError,
    run_event_time_finalization,
    validate_representative_trades,
    write_event_time_artifacts,
)


def representative_trades() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    sequence = 0
    for session_index, session_date in enumerate(SESSION_DATES):
        local = pd.date_range(
            f"{session_date} 09:30",
            periods=390,
            freq="1min",
            tz="America/New_York",
        )
        for minute, timestamp in enumerate(local):
            sequence += 1
            records.append(
                {
                    "timestamp": timestamp.tz_convert("UTC"),
                    "symbol": "SPY",
                    "id": f"trade-{sequence}",
                    "price": 500.0 + session_index + 0.01 * minute + 0.2 * np.sin(minute / 9.0),
                    "size": float(1 + minute % 20),
                    "exchange": "V",
                    "conditions": [],
                    "tape": "B",
                    "source": "alpaca",
                    "feed": "iex",
                }
            )
    return pd.DataFrame.from_records(records)


def test_representative_experiment_has_exact_scope_and_conservation() -> None:
    results = run_event_time_finalization(representative_trades())
    assert len(results.thresholds) == 5
    assert results.thresholds["target_bars"].eq(26).all()
    assert results.sampling_comparison["sampling_method"].tolist() == [
        "time_15min",
        "dollar",
    ]
    assert len(results.session_comparison) == 10
    assert len(results.indicator_comparison) == 2
    assert results.indicator_comparison["signal_available_observations"].gt(0).all()
    assert results.conservation["trade_count_error"].eq(0).all()
    assert results.conservation["volume_error"].abs().le(1.0e-8).all()
    assert results.conservation["dollar_value_error"].abs().le(1.0e-6).all()


def test_missing_predeclared_session_fails_closed() -> None:
    frame = representative_trades()
    local_dates = frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    with pytest.raises(EventTimeFinalizationError, match="Expected predeclared sessions"):
        validate_representative_trades(frame.loc[~local_dates.eq(SESSION_DATES[-1])])


def test_short_session_fails_closed() -> None:
    frame = representative_trades()
    local_dates = frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    target = local_dates.eq(SESSION_DATES[0])
    keep = ~target | frame["timestamp"].dt.tz_convert("America/New_York").dt.time.__ge__(pd.Timestamp("12:00").time())
    with pytest.raises(EventTimeFinalizationError, match="representative regular session"):
        validate_representative_trades(frame.loc[keep])


def test_event_time_writer_hashes_exact_allow_list(tmp_path) -> None:
    results = run_event_time_finalization(representative_trades())
    output = tmp_path / "event_time"
    paths = write_event_time_artifacts(
        results,
        output,
        source_dataset_id="five-session",
        source_sha256="b" * 64,
    )
    assert tuple(path.name for path in paths) == APPROVED_ARTIFACT_NAMES
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert manifest["locked_period_accessed"] is False
    with pytest.raises(FileExistsError):
        write_event_time_artifacts(
            results,
            output,
            source_dataset_id="five-session",
            source_sha256="b" * 64,
        )
