"""Atomic deterministic writer contracts for Day 16."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import systematic_alpha.analysis.day16_portfolio_validation_report as reporting
import systematic_alpha.analysis.portfolio_allocation_validation as allocation
from tests.day16_fixtures import make_day16_panel


@pytest.fixture(scope="module")
def day16_report() -> reporting.Day16PortfolioValidationReport:
    results = allocation.analyze_portfolio_allocation_panel(make_day16_panel())
    return reporting.build_day16_portfolio_validation_report(results)


def read_payloads(directory: Path) -> dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in reporting.APPROVED_DAY16_ARTIFACT_NAMES
    }


def test_writer_creates_exact_allow_list_schemas_and_row_counts(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    output = tmp_path / "day16"
    paths = reporting.write_day16_portfolio_validation_artifacts(
        day16_report,
        output,
    )
    assert tuple(path.name for path in paths) == (
        reporting.APPROVED_DAY16_ARTIFACT_NAMES
    )
    assert {item.name for item in output.iterdir()} == set(
        reporting.APPROVED_DAY16_ARTIFACT_NAMES
    )
    assert all(item.is_file() for item in output.iterdir())

    contracts = {
        reporting.ALLOCATION_WEIGHTS_FILENAME: (
            allocation.ALLOCATION_WEIGHT_COLUMNS,
            72,
        ),
        reporting.ALLOCATION_DIAGNOSTICS_FILENAME: (
            allocation.ALLOCATION_DIAGNOSTIC_COLUMNS,
            12,
        ),
        reporting.FOLD_PORTFOLIO_PERFORMANCE_FILENAME: (
            allocation.FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
            12,
        ),
        reporting.AGGREGATE_PORTFOLIO_PERFORMANCE_FILENAME: (
            allocation.AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
            3,
        ),
        reporting.PORTFOLIO_RETURN_PANEL_FILENAME: (
            allocation.PORTFOLIO_RETURN_PANEL_COLUMNS,
            len(day16_report.portfolio_return_panel),
        ),
    }
    for filename, (columns, rows) in contracts.items():
        frame = pd.read_csv(output / filename)
        assert tuple(frame.columns) == columns
        assert len(frame) == rows


def test_every_text_payload_has_exactly_one_final_newline(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    output = tmp_path / "day16"
    reporting.write_day16_portfolio_validation_artifacts(day16_report, output)
    for name in reporting.APPROVED_DAY16_ARTIFACT_NAMES:
        payload = (output / name).read_bytes()
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")


def test_manifest_is_strict_json_and_hashes_every_non_manifest_file(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    output = tmp_path / "day16"
    reporting.write_day16_portfolio_validation_artifacts(day16_report, output)
    raw = (output / reporting.MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    manifest = json.loads(raw)
    expected = set(reporting.APPROVED_DAY16_ARTIFACT_NAMES) - {
        reporting.MANIFEST_FILENAME
    }
    assert set(manifest["artifact_sha256"]) == expected
    assert reporting.MANIFEST_FILENAME not in manifest["artifact_sha256"]
    for name, expected_hash in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_overwrite_is_exact_and_repeated_bytes_are_deterministic(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    output = tmp_path / "artifacts" / "day16"
    reporting.write_day16_portfolio_validation_artifacts(day16_report, output)
    first = read_payloads(output)
    with pytest.raises(FileExistsError):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            output,
        )

    (output / "unapproved.txt").write_text("remove only with exact target")
    reporting.write_day16_portfolio_validation_artifacts(
        day16_report,
        output,
        overwrite=True,
    )
    assert read_payloads(output) == first
    assert {item.name for item in output.iterdir()} == set(
        reporting.APPROVED_DAY16_ARTIFACT_NAMES
    )


@pytest.mark.parametrize(
    "unsafe_name",
    ("artifacts", "repository-root", "home", "Day16", "day16-artifacts"),
)
def test_non_day16_overwrite_destination_is_rejected_before_writes(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
    unsafe_name: str,
) -> None:
    output = tmp_path / unsafe_name
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("retain", encoding="utf-8")
    with pytest.raises(ValueError, match="final name 'day16'"):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            output,
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "retain"


def test_non_day16_new_destination_does_not_create_parent_or_files(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    output = tmp_path / "new-parent" / "unsafe"
    with pytest.raises(ValueError, match="final name 'day16'"):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            output,
            overwrite=True,
        )
    assert not output.parent.exists()


def test_atomic_replacement_rolls_back_and_cleans_staging_on_failure(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "day16"
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
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            output,
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "original"
    assert not tuple(tmp_path.glob(".day16-stage-*"))
    assert not tuple(tmp_path.glob(".day16-backup-*"))


def test_payload_failure_leaves_no_partial_destination(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "day16"

    def fail_payloads(*args: object, **kwargs: object) -> dict[str, bytes]:
        raise OSError("deliberate payload failure")

    monkeypatch.setattr(reporting, "_artifact_payloads", fail_payloads)
    with pytest.raises(OSError, match="deliberate payload failure"):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".day16-stage-*"))


@pytest.mark.parametrize("invalid", [None, 7, object(), ""])
def test_invalid_output_directory_values_fail_closed(
    invalid: object,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            invalid,  # type: ignore[arg-type]
        )


def test_overwrite_must_be_boolean(
    tmp_path: Path,
    day16_report: reporting.Day16PortfolioValidationReport,
) -> None:
    with pytest.raises(TypeError, match="overwrite"):
        reporting.write_day16_portfolio_validation_artifacts(
            day16_report,
            tmp_path / "day16",
            overwrite=1,  # type: ignore[arg-type]
        )


def test_writer_rejects_non_report_objects(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Day16PortfolioValidationReport"):
        reporting.write_day16_portfolio_validation_artifacts(
            object(),  # type: ignore[arg-type]
            tmp_path / "day16",
        )
