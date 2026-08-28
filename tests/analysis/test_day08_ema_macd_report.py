"""Tests for compact Day 8 EMA/MACD artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.day08_ema_macd_report import (
    APPROVED_ARTIFACT_NAMES,
    Day08ReportError,
    calculate_file_sha256,
    write_ema_macd_baseline_artifacts,
)
from systematic_alpha.analysis.ema_macd_baseline import (
    analyse_ema_macd_baseline,
)


VALID_MANIFEST_SHA256 = "a" * 64


def make_report_frame(
    *,
    observations: int = 100,
) -> pd.DataFrame:
    """Create deterministic development-like synthetic bars."""

    observation_number = np.arange(
        observations,
        dtype=float,
    )
    close = (
        100.0
        + 0.04 * observation_number
        + 1.4 * np.sin(observation_number / 3.2)
        + 0.35 * np.cos(observation_number / 8.0)
    )

    session_count = int(np.ceil(observations / 10))
    session_dates = (
        pd.date_range(
            "2025-01-02",
            periods=session_count,
            freq="B",
        )
        .repeat(10)[:observations]
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=observations,
                freq="15min",
                tz="UTC",
            ),
            "session_date": session_dates,
            "symbol": "SPY",
            "close": close,
        }
    )
    frame["close_to_close_simple_return"] = (
        frame["close"].pct_change(fill_method=None)
    )

    return frame


def build_analysis():
    """Build one synthetic Day 8 analysis object."""

    return analyse_ema_macd_baseline(
        make_report_frame()
    )


def write_report(
    output_directory: Path,
):
    """Write one synthetic compact report."""

    return write_ema_macd_baseline_artifacts(
        build_analysis(),
        output_directory=output_directory,
        dataset_identifier="synthetic-day08-test",
        dataset_manifest_sha256=VALID_MANIFEST_SHA256,
    )


def test_writer_creates_exactly_the_approved_artifacts(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "day08"

    written = write_report(output_directory)

    assert tuple(
        path.name for path in written
    ) == APPROVED_ARTIFACT_NAMES

    assert sorted(
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    ) == sorted(APPROVED_ARTIFACT_NAMES)

    assert all(
        path.exists() and path.stat().st_size > 0
        for path in written
    )


def test_metadata_preserves_locked_period_safeguards(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "day08"

    write_report(output_directory)

    metadata = json.loads(
        (
            output_directory / "metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert metadata["development_sample_start"] == (
        "2020-01-02"
    )
    assert metadata["development_sample_end"] == (
        "2025-12-31"
    )
    assert metadata["locked_period_accessed"] is False
    assert (
        metadata[
            "parameter_selected_using_locked_period"
        ]
        is False
    )

    assert metadata["fast_window"] == 12
    assert metadata["slow_window"] == 26
    assert metadata["signal_window"] == 9
    assert metadata["neutral_band"] == pytest.approx(
        0.0005
    )
    assert metadata["position_timing"] == (
        "signal shifted by one bar"
    )
    assert (
        metadata["full_bar_level_artifacts_written"]
        is False
    )


def test_writer_rejects_non_frozen_development_period(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Day08ReportError,
        match="must remain 2025-12-31",
    ):
        write_ema_macd_baseline_artifacts(
            build_analysis(),
            output_directory=tmp_path / "day08",
            dataset_identifier="synthetic-day08-test",
            dataset_manifest_sha256=VALID_MANIFEST_SHA256,
            development_end="2026-01-02",
        )


def test_writer_rejects_invalid_manifest_digest(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Day08ReportError,
        match="64 lowercase hexadecimal",
    ):
        write_ema_macd_baseline_artifacts(
            build_analysis(),
            output_directory=tmp_path / "day08",
            dataset_identifier="synthetic-day08-test",
            dataset_manifest_sha256="not-a-valid-digest",
        )


def test_writer_rejects_unapproved_existing_files(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "day08"
    output_directory.mkdir()
    (
        output_directory / "full_bar_output.csv"
    ).write_text(
        "forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(
        Day08ReportError,
        match="unapproved files",
    ):
        write_report(output_directory)


def test_no_absolute_local_path_appears_in_text_artifacts(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "day08"

    write_report(output_directory)

    text = "\n".join(
        (
            output_directory / name
        ).read_text(encoding="utf-8")
        for name in APPROVED_ARTIFACT_NAMES
        if name.endswith((".json", ".csv", ".md"))
    )

    assert "/Users/" not in text
    assert "/home/" not in text
    assert "C:\\" not in text

    normalized_text = text.lower()

    for phrase in (
        "development sample only",
        "not parameter optimisation",
        "do not establish alpha",
        "locked final period was not accessed",
    ):
        assert phrase in normalized_text


def test_no_full_bar_level_or_parquet_artifact_is_written(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "day08"

    write_report(output_directory)

    names = {
        path.name
        for path in output_directory.iterdir()
    }

    assert not any(
        name.endswith(
            (
                ".parquet",
                ".feather",
                ".pickle",
                ".pkl",
            )
        )
        for name in names
    )

    assert "observations.csv" not in names
    assert "bar_level_results.csv" not in names
    assert "full_bar_output.csv" not in names


def test_figures_are_nonempty_and_have_stable_digests(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    write_report(first_directory)
    write_report(second_directory)

    for figure_name in (
        "cumulative_wealth.png",
        "position_and_signal.png",
    ):
        first = first_directory / figure_name
        second = second_directory / figure_name

        assert first.stat().st_size > 1_000
        assert second.stat().st_size > 1_000

        assert calculate_file_sha256(first) == (
            calculate_file_sha256(second)
        )


def test_artifact_writing_does_not_mutate_analysis(
    tmp_path: Path,
) -> None:
    analysis = build_analysis()

    original_observations = (
        analysis.strategy_bundle.observations.copy(
            deep=True
        )
    )
    original_performance = (
        analysis.performance_summary.copy(deep=True)
    )

    write_ema_macd_baseline_artifacts(
        analysis,
        output_directory=tmp_path / "day08",
        dataset_identifier="synthetic-day08-test",
        dataset_manifest_sha256=VALID_MANIFEST_SHA256,
    )

    pd.testing.assert_frame_equal(
        analysis.strategy_bundle.observations,
        original_observations,
    )
    pd.testing.assert_frame_equal(
        analysis.performance_summary,
        original_performance,
    )
