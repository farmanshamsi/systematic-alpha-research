"""Artifact-writing contracts for Day 15 reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import systematic_alpha.analysis.day15_strategy_diversification_report as reporting
import systematic_alpha.analysis.strategy_diversification as diversification
from tests.analysis.test_strategy_diversification_statistics import (
    make_weakly_correlated_panel,
)


@pytest.fixture(scope="module")
def day15_report() -> reporting.Day15StrategyDiversificationReport:
    """Build one deterministic synthetic report without market data."""

    results = diversification.analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )
    return reporting.build_day15_strategy_diversification_report(results)


def read_payloads(directory: Path) -> dict[str, bytes]:
    """Read the exact approved payload set."""

    return {
        name: (directory / name).read_bytes()
        for name in reporting.APPROVED_DAY15_ARTIFACT_NAMES
    }


def test_writer_creates_exact_allow_list_schemas_and_row_counts(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    output = tmp_path / "day15"
    paths = reporting.write_day15_strategy_diversification_artifacts(
        day15_report,
        output,
    )

    assert tuple(path.name for path in paths) == (
        reporting.APPROVED_DAY15_ARTIFACT_NAMES
    )
    assert {item.name for item in output.iterdir()} == set(
        reporting.APPROVED_DAY15_ARTIFACT_NAMES
    )
    assert all(item.is_file() for item in output.iterdir())

    contracts = {
        reporting.SLEEVE_INPUT_FILENAME: (
            diversification.SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
            6,
        ),
        reporting.FULL_SAMPLE_PAIRWISE_FILENAME: (
            diversification.FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS,
            15,
        ),
        reporting.FOLD_PAIRWISE_FILENAME: (
            diversification.FOLD_PAIRWISE_CORRELATION_COLUMNS,
            120,
        ),
        reporting.FOLD_COVARIANCE_FILENAME: (
            diversification.FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS,
            8,
        ),
        reporting.ENSEMBLE_FEASIBILITY_FILENAME: (
            diversification.ENSEMBLE_FEASIBILITY_COLUMNS,
            1,
        ),
    }
    for filename, (columns, rows) in contracts.items():
        frame = pd.read_csv(output / filename)
        assert tuple(frame.columns) == columns
        assert len(frame) == rows


def test_every_text_payload_has_stable_single_final_newline(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    output = tmp_path / "day15"
    reporting.write_day15_strategy_diversification_artifacts(
        day15_report,
        output,
    )

    for name in reporting.APPROVED_DAY15_ARTIFACT_NAMES:
        payload = (output / name).read_bytes()
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")


def test_manifest_is_strict_json_and_hashes_every_non_manifest_file(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    output = tmp_path / "day15"
    reporting.write_day15_strategy_diversification_artifacts(
        day15_report,
        output,
    )
    raw = (output / reporting.MANIFEST_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "NaN" not in raw
    assert "Infinity" not in raw
    manifest = json.loads(raw)

    expected_hashes = set(reporting.APPROVED_DAY15_ARTIFACT_NAMES) - {
        reporting.MANIFEST_FILENAME
    }
    assert set(manifest["artifact_sha256"]) == expected_hashes
    assert reporting.MANIFEST_FILENAME not in manifest["artifact_sha256"]

    for name, expected in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == expected


def test_overwrite_protection_and_repeated_bytes_are_deterministic(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    output = tmp_path / "artifacts" / "day15"
    reporting.write_day15_strategy_diversification_artifacts(
        day15_report,
        output,
    )
    first = read_payloads(output)

    with pytest.raises(FileExistsError):
        reporting.write_day15_strategy_diversification_artifacts(
            day15_report,
            output,
        )

    (output / "unapproved.txt").write_text("remove only with target")
    reporting.write_day15_strategy_diversification_artifacts(
        day15_report,
        output,
        overwrite=True,
    )
    assert read_payloads(output) == first
    assert {item.name for item in output.iterdir()} == set(
        reporting.APPROVED_DAY15_ARTIFACT_NAMES
    )


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "artifacts",
        "repository-root",
        "home",
        "Day15",
        "day15-artifacts",
    ),
)
def test_overwrite_rejects_non_day15_directory_names(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
    unsafe_name: str,
) -> None:
    output = tmp_path / unsafe_name
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("retain", encoding="utf-8")

    with pytest.raises(ValueError, match="final name 'day15'"):
        reporting.write_day15_strategy_diversification_artifacts(
            day15_report,
            output,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "retain"


def test_atomic_replacement_rolls_back_and_cleans_staging_on_failure(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "day15"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    real_replace = reporting.os.replace
    calls = 0

    def fail_destination_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("deliberate replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_destination_replace)

    with pytest.raises(OSError, match="deliberate"):
        reporting.write_day15_strategy_diversification_artifacts(
            day15_report,
            output,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert not tuple(tmp_path.glob(".day15-stage-*"))
    assert not tuple(tmp_path.glob(".day15-backup-*"))


@pytest.mark.parametrize("invalid", [None, 7, object()])
def test_invalid_output_directory_types_fail_closed(
    invalid: object,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    with pytest.raises(TypeError, match="output_directory"):
        reporting.write_day15_strategy_diversification_artifacts(
            day15_report,
            invalid,  # type: ignore[arg-type]
        )


def test_overwrite_must_be_boolean(
    tmp_path: Path,
    day15_report: reporting.Day15StrategyDiversificationReport,
) -> None:
    with pytest.raises(TypeError, match="overwrite"):
        reporting.write_day15_strategy_diversification_artifacts(
            day15_report,
            tmp_path / "day15",
            overwrite=1,  # type: ignore[arg-type]
        )
