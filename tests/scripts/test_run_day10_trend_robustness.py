"""Tests for the Day 10 execution and reporting runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_day10_trend_robustness as runner
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    CONFIGURATION_IDS,
    DEVELOPMENT_DATASET_ID,
    REQUIRED_RESULT_COLUMNS,
    build_robustness_run_matrix,
)


FIXED_GENERATION_TIMESTAMP = (
    "2026-07-26T12:00:00Z"
)
FIXED_GIT_COMMIT = "a" * 40


def make_canonical_bars(
    *,
    session_date: str = "2025-01-02",
    symbols: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
    ),
    bars_per_session: int = 26,
) -> pd.DataFrame:
    """Create compact canonical multi-symbol bars."""

    rows: list[dict[str, object]] = []
    offsets = {
        "SPY": 0.0,
        "QQQ": 100.0,
        "IWM": 200.0,
        "DIA": 300.0,
    }

    for symbol in symbols:
        timestamps = pd.date_range(
            f"{session_date} 14:30:00+00:00",
            periods=bars_per_session,
            freq="15min",
        )

        for index, timestamp in enumerate(
            timestamps
        ):
            close = (
                100.0
                + offsets[symbol]
                + 0.15 * index
                + 0.50 * np.sin(index / 2.0)
            )
            open_price = close - 0.10

            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": open_price,
                    "high": close + 0.25,
                    "low": open_price - 0.25,
                    "close": close,
                    "volume": float(
                        1_000 + index
                    ),
                    "trade_count": 100 + index,
                    "vwap": (
                        open_price + close
                    ) / 2.0,
                    "source": "test",
                    "feed": "sip",
                }
            )

    return pd.DataFrame(rows)


def make_matrix() -> pd.DataFrame:
    """Create one deterministic 18-row analysis result."""

    records: list[dict[str, object]] = []
    observation_counts = {
        "15min": 26,
        "30min": 13,
        "60min": 7,
    }

    for index, specification in enumerate(
        build_robustness_run_matrix()
    ):
        observations = observation_counts[
            specification.frequency
        ]
        annualized_return = (
            -0.08 + 0.01 * index
        )
        sharpe_ratio = (
            -0.60 + 0.08 * index
        )

        records.append(
            {
                "strategy": (
                    specification.strategy
                ),
                "symbol": specification.symbol,
                "frequency": (
                    specification.frequency
                ),
                "dataset_id": (
                    DEVELOPMENT_DATASET_ID
                ),
                "configuration_id": (
                    CONFIGURATION_IDS[
                        specification.strategy
                    ]
                ),
                "start_timestamp": (
                    pd.Timestamp(
                        "2025-01-02 14:30:00+00:00"
                    )
                ),
                "end_timestamp": (
                    pd.Timestamp(
                        "2025-01-02 20:45:00+00:00"
                    )
                ),
                "sessions": 1,
                "observations": observations,
                "annualization_factor": (
                    specification
                    .annualization_factor
                ),
                "partial_bar_count": (
                    1
                    if specification.frequency
                    == "60min"
                    else 0
                ),
                "warmup_observations": min(
                    2,
                    observations,
                ),
                "active_observations": max(
                    observations - 2,
                    0,
                ),
                "annualized_return": (
                    annualized_return
                ),
                "annualized_volatility": (
                    0.15 + 0.002 * index
                ),
                "sharpe_ratio": sharpe_ratio,
                "maximum_drawdown": (
                    -0.10 - 0.005 * index
                ),
                "turnover": (
                    10.0 + index
                ),
                "average_exposure": 80.0,
                "long_exposure": 45.0,
                "short_exposure": 35.0,
                "flat_exposure": 20.0,
                "trade_count": 5 + index,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=REQUIRED_RESULT_COLUMNS,
    )


@pytest.fixture
def compact_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    """Patch only full-history counts and analysis execution."""

    matrix = make_matrix()
    calls: list[pd.DataFrame] = []

    monkeypatch.setattr(
        runner,
        "EXPECTED_SOURCE_ROW_COUNT",
        78,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_SESSION_SIZE_COUNTS",
        {
            26: 3,
        },
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_FREQUENCY_ROW_COUNTS",
        {
            "15min": 78,
            "30min": 39,
            "60min": 21,
        },
    )

    def fake_analysis(
        bars: pd.DataFrame,
    ) -> pd.DataFrame:
        calls.append(
            bars.copy(deep=True)
        )
        return matrix.copy(deep=True)

    monkeypatch.setattr(
        runner,
        "run_trend_family_robustness",
        fake_analysis,
    )

    return matrix, calls


def write_source(
    directory: Path,
    bars: pd.DataFrame | None = None,
) -> Path:
    """Write a temporary canonical Parquet source."""

    path = directory / "canonical.parquet"
    (
        bars
        if bars is not None
        else make_canonical_bars()
    ).to_parquet(
        path,
        index=False,
    )

    return path


def execute_compact(
    tmp_path: Path,
    *,
    artifact_name: str = "artifacts",
    overwrite: bool = False,
):
    """Run the patched compact artifact workflow."""

    return runner.execute_day10(
        dataset_path=write_source(tmp_path),
        artifact_directory=(
            tmp_path / artifact_name
        ),
        overwrite=overwrite,
        generation_timestamp=(
            FIXED_GENERATION_TIMESTAMP
        ),
        source_git_commit=(
            FIXED_GIT_COMMIT
        ),
    )


def test_input_period_rejection() -> None:
    with pytest.raises(
        runner.Day10RunnerError,
        match="development period",
    ):
        runner.validate_canonical_input(
            make_canonical_bars(
                session_date="2026-01-02"
            )
        )


def test_missing_symbol_rejection() -> None:
    with pytest.raises(
        runner.Day10RunnerError,
        match="missing required symbols",
    ):
        runner.validate_canonical_input(
            make_canonical_bars(
                symbols=("SPY", "QQQ")
            )
        )


def test_unexpected_symbol_rejection() -> None:
    with pytest.raises(
        runner.Day10RunnerError,
        match="unexpected symbols",
    ):
        runner.validate_canonical_input(
            make_canonical_bars(
                symbols=(
                    "SPY",
                    "QQQ",
                    "IWM",
                    "DIA",
                )
            )
        )


def test_runner_creates_complete_ordered_artifacts(
    tmp_path: Path,
    compact_contract,
) -> None:
    matrix, calls = compact_contract

    result = execute_compact(tmp_path)

    assert len(calls) == 1
    assert len(result.matrix) == 18
    assert tuple(
        result.matrix.columns
    ) == REQUIRED_RESULT_COLUMNS
    pd.testing.assert_frame_equal(
        result.matrix,
        matrix,
    )

    assert {
        path.name
        for path in result.artifact_paths
    } == set(
        runner.APPROVED_ARTIFACT_NAMES
    )
    assert all(
        path.exists()
        and path.stat().st_size > 0
        for path in result.artifact_paths
    )

    matrix_csv = pd.read_csv(
        result.artifact_directory
        / "matrix.csv"
    )
    summary_csv = pd.read_csv(
        result.artifact_directory
        / "summary.csv"
    )

    assert len(matrix_csv) == 18
    assert tuple(
        matrix_csv.columns
    ) == REQUIRED_RESULT_COLUMNS
    assert tuple(
        summary_csv.columns
    ) == runner.SUMMARY_COLUMNS
    assert len(summary_csv) == 18


def test_manifest_contains_lineage_counts_and_hashes(
    tmp_path: Path,
    compact_contract,
) -> None:
    execute_compact(tmp_path)
    directory = tmp_path / "artifacts"
    manifest = json.loads(
        (
            directory / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["day"] == 10
    assert manifest["analysis_name"] == (
        "trend_family_robustness"
    )
    assert manifest[
        "generation_timestamp"
    ] == FIXED_GENERATION_TIMESTAMP
    assert manifest[
        "source_dataset_identifier"
    ] == DEVELOPMENT_DATASET_ID
    assert manifest["source_row_count"] == 78
    assert manifest["development_start"] == (
        "2020-01-02"
    )
    assert manifest["development_end"] == (
        "2025-12-31"
    )
    assert manifest["matrix_run_count"] == 18
    assert manifest[
        "annualization_factors"
    ] == {
        key: value
        for key, value in (
            ANNUALIZATION_FACTORS.items()
        )
    }
    assert manifest[
        "strategy_configuration_identifiers"
    ] == CONFIGURATION_IDS
    assert manifest[
        "output_row_counts_by_frequency"
    ] == {
        "15min": 78,
        "30min": 39,
        "60min": 21,
    }
    assert manifest[
        "source_git_commit"
    ] == FIXED_GIT_COMMIT
    assert manifest[
        "locked_2026_period_accessed"
    ] is False

    expected_hashed = {
        name
        for name in (
            runner.APPROVED_ARTIFACT_NAMES
        )
        if name != "manifest.json"
    }
    hashes = manifest["artifact_sha256"]

    assert set(hashes) == expected_hashed

    for name, digest in hashes.items():
        expected_digest = hashlib.sha256(
            (directory / name).read_bytes()
        ).hexdigest()

        assert digest == expected_digest


def test_report_contains_required_research_statements(
    tmp_path: Path,
    compact_contract,
) -> None:
    execute_compact(tmp_path)
    report = (
        tmp_path
        / "artifacts"
        / "report.md"
    ).read_text(
        encoding="utf-8"
    ).lower()

    for phrase in (
        "profitability was not an acceptance criterion",
        "fixed-bar parameters imply longer clock-time "
        "horizons at lower frequencies",
        "60-minute partial closing bars were retained",
        "development-sample robustness evidence, not "
        "locked-test results",
        "2026 locked test period was not accessed",
        "no day 9 best configuration was selected",
        "diagnostic rules, not performance acceptance "
        "criteria",
    ):
        assert phrase in report

    for heading in (
        "1. objective",
        "2. frozen experimental design",
        "3. data and aggregation contract",
        "4. strategy configuration lock",
        "5. cross-asset results",
        "6. cross-frequency results",
        "7. economic-coherence assessment",
        "8. degenerate-behaviour flags",
        "9. limitations",
        "10. day 10 conclusion",
    ):
        assert heading in report


def test_refuses_overwrite_without_permission_and_allows_it(
    tmp_path: Path,
    compact_contract,
) -> None:
    execute_compact(tmp_path)

    with pytest.raises(
        runner.Day10RunnerError,
        match="--overwrite",
    ):
        execute_compact(tmp_path)

    overwritten = execute_compact(
        tmp_path,
        overwrite=True,
    )

    assert len(overwritten.artifact_paths) == 7


def test_artifacts_are_deterministic_with_fixed_provenance(
    tmp_path: Path,
    compact_contract,
) -> None:
    first = execute_compact(
        tmp_path,
        artifact_name="first",
    )
    second = execute_compact(
        tmp_path,
        artifact_name="second",
    )

    for name in runner.APPROVED_ARTIFACT_NAMES:
        assert (
            first.artifact_directory / name
        ).read_bytes() == (
            second.artifact_directory / name
        ).read_bytes()


def test_summary_uses_each_strategy_spy_15min_reference() -> None:
    summary = runner.build_summary(
        make_matrix()
    )

    references = summary.loc[
        summary["symbol"].eq("SPY")
        & summary["frequency"].eq(
            "15min"
        )
    ]

    assert len(references) == 2
    assert references[
        "reference_symbol"
    ].eq("SPY").all()
    assert references[
        "reference_frequency"
    ].eq("15min").all()
    assert references[
        "sharpe_delta"
    ].eq(0.0).all()
    assert references[
        "annualized_return_delta"
    ].eq(0.0).all()
    assert references[
        "maximum_drawdown_delta"
    ].eq(0.0).all()
    assert references[
        "turnover_ratio_to_reference"
    ].eq(1.0).all()


def test_cli_parser_supports_requested_options(
    tmp_path: Path,
) -> None:
    arguments = runner.parse_args(
        [
            "--dataset-path",
            "input.parquet",
            "--artifact-directory",
            "output",
            "--overwrite",
        ]
    )

    assert arguments.dataset_path == Path(
        "input.parquet"
    )
    assert arguments.artifact_directory == Path(
        "output"
    )
    assert arguments.overwrite is True

    assert runner._display_path(
        tmp_path / "inside",
        project_root=tmp_path,
    ) == "inside"

    outside = tmp_path.parent / "outside"

    assert runner._display_path(
        outside,
        project_root=tmp_path,
    ) == outside.as_posix()
