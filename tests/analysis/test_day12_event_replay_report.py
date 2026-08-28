"""Contract tests for deterministic Day 12 replay reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.day12_event_replay_report as report
from systematic_alpha.analysis.day12_event_replay_report import (
    APPROVED_DAY12_ARTIFACT_NAMES,
    DAY12_EVENT_COUNT_COLUMNS,
    DAY12_PARITY_COLUMNS,
    DAY12_PERFORMANCE_COLUMNS,
    DAY12_POSITION_DIAGNOSTIC_COLUMNS,
    DAY12_REPLAY_SUMMARY_COLUMNS,
    Day12DatasetAudit,
    Day12ReportError,
    build_day12_report,
    run_day12_replay_study,
    write_day12_artifacts,
)
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS,
)


FIXED_TIMESTAMP = "2026-07-27T12:00:00Z"
FIXED_COMMIT = "a" * 40


def make_bars(
    *,
    session_dates: tuple[str, ...] = (
        "2024-12-30",
        "2024-12-31",
        "2025-01-02",
    ),
) -> pd.DataFrame:
    """Build compact complete SPY development sessions."""

    records: list[dict[str, object]] = []
    observation = 0

    for session_date in session_dates:
        timestamps = pd.date_range(
            f"{session_date} 14:30:00",
            periods=26,
            freq="15min",
            tz="UTC",
        )

        for timestamp in timestamps:
            close = float(
                100.0
                + observation * 0.05
                + np.sin(observation / 3.0)
            )
            open_price = close - 0.08
            records.append(
                {
                    "timestamp": timestamp,
                    "session_date": session_date,
                    "symbol": "SPY",
                    "open": open_price,
                    "high": max(open_price, close) + 0.2,
                    "low": min(open_price, close) - 0.2,
                    "close": close,
                    "volume": float(1_000 + observation),
                    "trade_count": 20 + observation,
                    "vwap": (open_price + close) / 2.0,
                    "source": "alpaca",
                    "feed": "sip",
                }
            )
            observation += 1

    return pd.DataFrame.from_records(records)


def make_audit(
    bars: pd.DataFrame,
) -> Day12DatasetAudit:
    """Build compact, valid development lineage."""

    return Day12DatasetAudit(
        dataset_id=(
            "spy_qqq_iwm_15min_2020-01-02_"
            "2025-12-31_sip_v3_development_canonical"
        ),
        dataset_path=(
            "data/processed/bars/"
            "spy_qqq_iwm_15min_2020-01-02_"
            "2025-12-31_sip_v3_development_canonical.parquet"
        ),
        manifest_sha256="b" * 64,
        canonical_row_count=117_192,
        spy_row_count=len(bars),
        spy_session_count=int(
            bars["session_date"].nunique()
        ),
        minimum_timestamp=pd.Timestamp(
            "2020-01-02 14:30:00+00:00"
        ),
        maximum_timestamp=pd.Timestamp(
            "2025-12-31 20:45:00+00:00"
        ),
    )


def test_study_uses_public_replay_for_both_frozen_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = make_bars()
    calls: list[str] = []
    public_replay = (
        report.run_trend_family_event_replay
    )

    def recording_replay(
        received: pd.DataFrame,
        *,
        strategy: str,
        frequency: str = "15min",
        evaluation_start: pd.Timestamp | None = None,
        evaluation_end_exclusive: pd.Timestamp | None = None,
    ):
        calls.append(strategy)
        return public_replay(
            received,
            strategy=strategy,
            frequency=frequency,
            evaluation_start=evaluation_start,
            evaluation_end_exclusive=(
                evaluation_end_exclusive
            ),
        )

    monkeypatch.setattr(
        report,
        "run_trend_family_event_replay",
        recording_replay,
    )
    study = run_day12_replay_study(
        bars,
        verify_determinism=False,
    )

    assert calls == [
        "trend_ratio",
        "ema_macd",
    ]
    assert tuple(
        result.strategy
        for result in study.replay_results
    ) == (
        "trend_ratio",
        "ema_macd",
    )


def test_study_tables_have_stable_schemas_and_reconcile() -> None:
    study = run_day12_replay_study(
        make_bars(),
        verify_determinism=False,
    )

    assert tuple(
        study.replay_summary.columns
    ) == DAY12_REPLAY_SUMMARY_COLUMNS
    assert tuple(
        study.performance.columns
    ) == DAY12_PERFORMANCE_COLUMNS
    assert tuple(
        study.event_counts.columns
    ) == DAY12_EVENT_COUNT_COLUMNS
    assert tuple(
        study.position_diagnostics.columns
    ) == DAY12_POSITION_DIAGNOSTIC_COLUMNS
    assert tuple(
        study.vectorized_parity.columns
    ) == DAY12_PARITY_COLUMNS
    assert len(study.replay_summary) == 2
    assert len(study.performance) == 4
    assert len(study.event_counts) == 10
    assert len(study.position_diagnostics) == 2
    assert len(study.vectorized_parity) == 16
    assert study.vectorized_parity[
        "passed"
    ].all()
    assert study.replay_summary[
        "configuration_id"
    ].tolist() == [
        CONFIGURATION_IDS["trend_ratio"],
        CONFIGURATION_IDS["ema_macd"],
    ]

    for result in study.replay_results:
        selected = study.event_counts.loc[
            study.event_counts["strategy"].eq(
                result.strategy
            )
        ]
        assert int(selected["count"].sum()) == len(
            result.events
        )
        summary = study.replay_summary.loc[
            study.replay_summary["strategy"].eq(
                result.strategy
            )
        ].iloc[0]
        assert summary["final_equity"] == pytest.approx(
            result.observations[
                "ending_equity"
            ].iloc[-1]
        )


def test_parity_failure_raises_explicit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = report.build_trend_ratio_strategy

    def mismatched_builder(*args, **kwargs):
        bundle = original(*args, **kwargs)
        observations = bundle.observations.copy(
            deep=True
        )
        observations.loc[
            observations.index[-1],
            "position",
        ] *= -1
        return type(bundle)(
            observations=observations,
            diagnostics=bundle.diagnostics,
            parameters=bundle.parameters,
        )

    monkeypatch.setattr(
        report,
        "build_trend_ratio_strategy",
        mismatched_builder,
    )

    with pytest.raises(
        Day12ReportError,
        match="parity",
    ):
        run_day12_replay_study(
            make_bars(),
            verify_determinism=False,
        )


def test_study_rejects_2026_and_does_not_mutate_input() -> None:
    bars = make_bars()
    original = bars.copy(deep=True)
    first = run_day12_replay_study(
        bars,
        verify_determinism=False,
    )
    second = run_day12_replay_study(
        bars,
        verify_determinism=False,
    )

    pd.testing.assert_frame_equal(
        bars,
        original,
    )
    pd.testing.assert_frame_equal(
        first.replay_summary,
        second.replay_summary,
    )
    pd.testing.assert_frame_equal(
        first.vectorized_parity,
        second.vectorized_parity,
    )

    locked = bars.copy(deep=True)
    locked.loc[
        locked.index[-1],
        "timestamp",
    ] = pd.Timestamp(
        "2026-01-02 14:30:00+00:00"
    )

    with pytest.raises(
        (Day12ReportError, ValueError),
        match="2026|development",
    ):
        run_day12_replay_study(
            locked,
            verify_determinism=False,
        )


def test_report_contains_methodology_and_neutral_conclusion() -> None:
    bars = make_bars()
    text = build_day12_report(
        run_day12_replay_study(
            bars,
            verify_determinism=False,
        ),
        make_audit(bars),
    ).lower()

    for phrase in (
        "objective and scope",
        "frozen constraints",
        "market bar",
        "target-position order",
        "portfolio snapshot",
        "one-observation execution delay",
        "normalized-notional",
        "vectorised versus event-driven parity",
        "no partial fills",
        "no order-book model",
        "no latency model",
        "no concurrency",
        "no live broker",
        "no tuning",
        "no 2026 evaluation",
        "does not select a winner",
    ):
        assert phrase in text

    for token in (
        "winner_strategy",
        "selected_strategy",
        "parameter_rank",
    ):
        assert token not in text


def write_compact(
    tmp_path: Path,
    *,
    name: str = "day12",
):
    """Write one deterministic compact Day 12 package."""

    bars = make_bars()

    return write_day12_artifacts(
        run_day12_replay_study(
            bars,
            verify_determinism=False,
        ),
        make_audit(bars),
        artifact_directory=tmp_path / name,
        generation_timestamp=FIXED_TIMESTAMP,
        source_git_commit=FIXED_COMMIT,
    )


def test_artifact_set_manifest_hashes_and_determinism(
    tmp_path: Path,
) -> None:
    first = write_compact(
        tmp_path,
        name="first",
    )
    second = write_compact(
        tmp_path,
        name="second",
    )

    assert {
        path.name
        for path in first.artifact_paths
    } == set(APPROVED_DAY12_ARTIFACT_NAMES)

    for name in APPROVED_DAY12_ARTIFACT_NAMES:
        assert (
            first.artifact_directory / name
        ).read_bytes() == (
            second.artifact_directory / name
        ).read_bytes()

    manifest = json.loads(
        (
            first.artifact_directory
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["day"] == 12
    assert manifest[
        "locked_2026_period_accessed"
    ] is False
    assert manifest[
        "parameter_selection_performed"
    ] is False
    assert not any(
        "winner" in key.lower()
        for key in manifest
    )
    assert manifest["strategies"] == [
        "trend_ratio",
        "ema_macd",
    ]
    assert manifest["symbol"] == "SPY"
    assert manifest["frequency"] == "15min"

    for name, digest in manifest[
        "artifact_sha256"
    ].items():
        assert digest == hashlib.sha256(
            (
                first.artifact_directory
                / name
            ).read_bytes()
        ).hexdigest()


def test_overwrite_is_safe_and_explicit(
    tmp_path: Path,
) -> None:
    write_compact(tmp_path)

    with pytest.raises(
        Day12ReportError,
        match="--overwrite",
    ):
        write_compact(tmp_path)

    bars = make_bars()
    replaced = write_day12_artifacts(
        run_day12_replay_study(
            bars,
            verify_determinism=False,
        ),
        make_audit(bars),
        artifact_directory=tmp_path / "day12",
        overwrite=True,
        generation_timestamp=FIXED_TIMESTAMP,
        source_git_commit=FIXED_COMMIT,
    )

    assert len(replaced.artifact_paths) == len(
        APPROVED_DAY12_ARTIFACT_NAMES
    )
