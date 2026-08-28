"""Contract tests for the canonical Day 11 report runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day11_trend_walk_forward as runner
from systematic_alpha.analysis.day11_trend_walk_forward_report import (
    APPROVED_DAY11_ARTIFACT_NAMES,
    Day11DatasetAudit,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    AGGREGATE_RESULT_COLUMNS,
    CONFIGURATION_IDS,
    FOLD_RESULT_COLUMNS,
    WALK_FORWARD_STRATEGIES,
    TrendFamilyWalkForwardResults,
)


FIXED_TIMESTAMP = "2026-07-26T12:00:00Z"
FIXED_COMMIT = "a" * 40


def make_results() -> TrendFamilyWalkForwardResults:
    """Create the smallest valid public Day 11 result."""

    fold_records: list[
        dict[str, object]
    ] = []

    for strategy in WALK_FORWARD_STRATEGIES:
        for index, year in enumerate(
            range(2022, 2026)
        ):
            record = {
                column: 0.0
                for column in (
                    FOLD_RESULT_COLUMNS
                )
            }
            record.update(
                {
                    "strategy": strategy,
                    "symbol": "SPY",
                    "frequency": "15min",
                    "fold_id": f"wf_{year}",
                    "configuration_id": (
                        CONFIGURATION_IDS[
                            strategy
                        ]
                    ),
                    "train_start_timestamp": (
                        pd.Timestamp(
                            "2020-01-02 "
                            "14:30:00+00:00"
                        )
                    ),
                    "train_end_timestamp": (
                        pd.Timestamp(
                            f"{year - 1}-12-31 "
                            "20:45:00+00:00"
                        )
                    ),
                    "test_start_timestamp": (
                        pd.Timestamp(
                            f"{year}-01-03 "
                            "14:30:00+00:00"
                        )
                    ),
                    "test_end_timestamp": (
                        pd.Timestamp(
                            f"{year}-12-30 "
                            "20:45:00+00:00"
                        )
                    ),
                    "train_sessions": (
                        500 + 250 * index
                    ),
                    "test_sessions": 1,
                    "train_observations": (
                        13_000 + 6_500 * index
                    ),
                    "test_observations": 26,
                    "annualization_factor": (
                        6_552.0
                    ),
                    "purge_sessions": 0,
                    "embargo_sessions": 0,
                    "indicator_history_observations": (
                        13_000 + 6_500 * index
                    ),
                    "initial_test_position": 0,
                    "initial_test_turnover": 0.0,
                    "warmup_observations": 2,
                    "active_observations": 24,
                    "cumulative_return": (
                        -0.01 + 0.01 * index
                    ),
                    "annualized_return": (
                        -0.01 + 0.01 * index
                    ),
                    "annualized_volatility": 0.1,
                    "sharpe_ratio": (
                        -0.2 + 0.1 * index
                    ),
                    "maximum_drawdown": -0.1,
                    "turnover": 10.0,
                    "average_exposure": 75.0,
                    "long_exposure": 40.0,
                    "short_exposure": 35.0,
                    "flat_exposure": 25.0,
                    "trade_count": 8,
                }
            )
            fold_records.append(record)

    folds = pd.DataFrame.from_records(
        fold_records,
        columns=FOLD_RESULT_COLUMNS,
    )
    aggregate_records: list[
        dict[str, object]
    ] = []

    for strategy in WALK_FORWARD_STRATEGIES:
        record = {
            column: 0.0
            for column in (
                AGGREGATE_RESULT_COLUMNS
            )
        }
        record.update(
            {
                "strategy": strategy,
                "symbol": "SPY",
                "frequency": "15min",
                "configuration_id": (
                    CONFIGURATION_IDS[
                        strategy
                    ]
                ),
                "folds": 4,
                "test_start_timestamp": (
                    pd.Timestamp(
                        "2022-01-03 "
                        "14:30:00+00:00"
                    )
                ),
                "test_end_timestamp": (
                    pd.Timestamp(
                        "2025-12-30 "
                        "20:45:00+00:00"
                    )
                ),
                "test_sessions": 4,
                "test_observations": 104,
                "annualization_factor": (
                    6_552.0
                ),
                "cumulative_return": -0.02,
                "annualized_return": -0.01,
                "annualized_volatility": 0.1,
                "sharpe_ratio": -0.1,
                "maximum_drawdown": -0.15,
                "turnover": 40.0,
                "average_exposure": 75.0,
                "long_exposure": 40.0,
                "short_exposure": 35.0,
                "flat_exposure": 25.0,
                "trade_count": 32,
            }
        )
        aggregate_records.append(record)

    aggregate = pd.DataFrame.from_records(
        aggregate_records,
        columns=AGGREGATE_RESULT_COLUMNS,
    )

    return TrendFamilyWalkForwardResults(
        fold_results=folds,
        aggregate_results=aggregate,
    )


def make_audit() -> Day11DatasetAudit:
    """Create canonical-shaped report lineage."""

    return Day11DatasetAudit(
        dataset_id=runner.DEVELOPMENT_DATASET_ID,
        dataset_path=(
            runner
            .CANONICAL_DATASET_RELATIVE_PATH
            .as_posix()
        ),
        manifest_sha256="b" * 64,
        canonical_row_count=117_192,
        spy_row_count=39_064,
        spy_session_count=1_508,
        minimum_timestamp=pd.Timestamp(
            "2020-01-02 14:30:00+00:00"
        ),
        maximum_timestamp=pd.Timestamp(
            "2025-12-31 20:45:00+00:00"
        ),
    )


def test_cli_has_narrow_output_and_overwrite_options() -> None:
    arguments = runner.parse_args(
        [
            "--artifact-directory",
            "custom/day11",
            "--overwrite",
            "--generation-timestamp",
            FIXED_TIMESTAMP,
        ]
    )

    assert arguments.artifact_directory == Path(
        "custom/day11"
    )
    assert arguments.overwrite is True
    assert (
        arguments.generation_timestamp
        == FIXED_TIMESTAMP
    )
    assert not hasattr(
        arguments,
        "dataset_path",
    )


def test_execute_uses_canonical_loader_and_public_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = pd.DataFrame(
        {
            "sentinel": [1],
        }
    )
    calls: list[object] = []

    monkeypatch.setattr(
        runner,
        "load_canonical_day11_input",
        lambda: runner.CanonicalDay11Input(
            bars=bars,
            audit=make_audit(),
        ),
    )

    def fake_engine(
        received: pd.DataFrame,
    ) -> TrendFamilyWalkForwardResults:
        calls.append(received)
        return make_results()

    monkeypatch.setattr(
        runner,
        "run_trend_family_walk_forward",
        fake_engine,
    )
    result = runner.execute_day11(
        artifact_directory=(
            tmp_path / "artifacts"
        ),
        generation_timestamp=(
            FIXED_TIMESTAMP
        ),
        source_git_commit=FIXED_COMMIT,
    )

    assert calls == [bars]
    assert {
        path.name
        for path in result.artifact_paths
    } == set(
        APPROVED_DAY11_ARTIFACT_NAMES
    )


def make_compact_canonical() -> pd.DataFrame:
    """Create canonical-shaped compact rows for audit validation."""

    frames: list[pd.DataFrame] = []

    for session_date, count in (
        (
            "2020-01-02",
            14,
        ),
        (
            "2025-12-31",
            26,
        ),
    ):
        timestamps = pd.date_range(
            f"{session_date} 14:30:00+00:00",
            periods=count,
            freq="15min",
        )
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": "SPY",
                    "session_date": (
                        session_date
                    ),
                }
            )
        )

    for symbol in (
        "QQQ",
        "IWM",
    ):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": [
                        pd.Timestamp(
                            "2020-01-02 "
                            "14:30:00+00:00"
                        )
                    ],
                    "symbol": [symbol],
                    "session_date": [
                        "2020-01-02"
                    ],
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def test_dataset_audit_rejects_locked_period_and_wrong_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = make_compact_canonical().drop(
        columns="session_date"
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_CANONICAL_ROWS",
        len(bars),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_SPY_ROWS",
        40,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_SPY_SESSIONS",
        2,
    )

    audit = runner._build_dataset_audit(
        bars,
        manifest_sha256="b" * 64,
    )

    assert audit.spy_row_count == 40
    assert audit.spy_session_count == 2

    locked = bars.copy(deep=True)
    locked.loc[
        locked.index[-1],
        "timestamp",
    ] = pd.Timestamp(
        "2026-01-02 14:30:00+00:00"
    )

    with pytest.raises(
        runner.Day11RunnerError,
        match="2026|2025-12-31",
    ):
        runner._build_dataset_audit(
            locked,
            manifest_sha256="b" * 64,
        )

    unexpected = bars.copy(deep=True)
    unexpected.loc[
        unexpected["symbol"].eq(
            "IWM"
        ),
        "symbol",
    ] = "DIA"

    with pytest.raises(
        runner.Day11RunnerError,
        match="symbols",
    ):
        runner._build_dataset_audit(
            unexpected,
            manifest_sha256="b" * 64,
        )


def test_manifest_validation_locks_identity_rows_and_hash() -> None:
    manifest = {
        "dataset_id": (
            runner.DEVELOPMENT_DATASET_ID
        ),
        "dataset_kind": "bars",
        "row_count": (
            runner.EXPECTED_CANONICAL_ROWS
        ),
        "sha256": "A" * 64,
    }

    assert runner._validate_manifest(
        manifest
    ) == "a" * 64

    manifest["row_count"] = 1

    with pytest.raises(
        runner.Day11RunnerError,
        match="row count",
    ):
        runner._validate_manifest(manifest)


def test_runner_source_has_no_strategy_or_sensitivity_math() -> None:
    source = Path(
        runner.__file__
    ).read_text(
        encoding="utf-8"
    ).lower()

    for token in (
        "build_trend_ratio_strategy",
        "build_ema_macd_strategy",
        "ema_macd_sensitivity",
        "trend_ratio_sensitivity",
        "day09",
        "day07",
    ):
        assert token not in source

    assert (
        "run_trend_family_walk_forward"
        in source
    )
