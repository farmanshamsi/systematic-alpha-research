from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import systematic_alpha.broker.day18_report as report_module
from systematic_alpha.broker.day18_report import (
    APPROVED_DAY18_ARTIFACT_NAMES,
    ASSET_COLUMNS,
    CAPABILITY_COLUMNS,
    build_day18_preflight_report,
    write_day18_preflight_artifacts,
)
from systematic_alpha.broker.paper_boundary import CORE_SYMBOLS
from tests.day18_fixtures import passing_preflight_result


def test_report_has_exact_safe_schemas_and_order() -> None:
    report = build_day18_preflight_report(passing_preflight_result())

    assert report.summary["core_symbols"] == list(CORE_SYMBOLS)
    assert report.summary["order_submission_enabled"] is False
    assert report.summary["order_submission_occurred"] is False
    assert report.summary["credential_values_persisted"] is False
    assert len(report.asset_rows) == 3
    assert tuple(report.asset_rows[0]) == ASSET_COLUMNS
    assert tuple(row["symbol"] for row in report.asset_rows) == CORE_SYMBOLS
    assert len(report.capability_rows) == 11
    assert tuple(report.capability_rows[0]) == CAPABILITY_COLUMNS
    assert not any(
        row["day18_authorized"] for row in report.capability_rows
    )


def test_report_rejects_unsafe_result_flags() -> None:
    result = passing_preflight_result()
    with pytest.raises(ValueError):
        build_day18_preflight_report(
            replace(result, order_submission_occurred=True)
        )
    with pytest.raises(ValueError):
        build_day18_preflight_report(
            replace(result, credential_values_persisted=True)
        )


def test_failed_preflight_is_retained_without_reinterpretation() -> None:
    result = replace(
        passing_preflight_result(),
        account_gate_passed=False,
        preflight_passed=False,
    )
    report = build_day18_preflight_report(result)
    assert report.summary["account_gate_passed"] is False
    assert report.summary["preflight_passed"] is False
    assert "Preflight: **FAIL**" in report.report


def test_writer_emits_exact_allow_list_and_valid_hashes(tmp_path: Path) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    destination = tmp_path / "day18"
    paths = write_day18_preflight_artifacts(report, destination)

    assert tuple(path.name for path in paths) == APPROVED_DAY18_ARTIFACT_NAMES
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY18_ARTIFACT_NAMES)
    )

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["artifact_order"] == list(APPROVED_DAY18_ARTIFACT_NAMES)
    assert manifest["safety"]["paper_only"] is True
    assert manifest["safety"]["order_submission_occurred"] is False
    assert manifest["safety"]["locked_2026_data_accessed"] is False
    for row in manifest["artifacts"]:
        payload = (destination / row["filename"]).read_bytes()
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_writer_fixed_input_replays_byte_for_byte(tmp_path: Path) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_day18_preflight_artifacts(report, first)
    write_day18_preflight_artifacts(report, second)

    for filename in APPROVED_DAY18_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_writer_refuses_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    destination = tmp_path / "day18"
    write_day18_preflight_artifacts(report, destination)
    with pytest.raises(FileExistsError):
        write_day18_preflight_artifacts(report, destination)


def test_overwrite_removes_unapproved_files(tmp_path: Path) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    destination = tmp_path / "day18"
    write_day18_preflight_artifacts(report, destination)
    (destination / "unexpected.txt").write_text("remove on replacement")

    write_day18_preflight_artifacts(report, destination, overwrite=True)
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY18_ARTIFACT_NAMES)
    )


def test_overwrite_rolls_back_when_replacement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    destination = tmp_path / "day18"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("original")

    original_replace = report_module.os.replace
    calls = 0

    def fail_second_replace(source, target) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replacement failure")
        original_replace(source, target)

    monkeypatch.setattr(report_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        write_day18_preflight_artifacts(
            report,
            destination,
            overwrite=True,
        )
    assert sentinel.read_text() == "original"
    assert tuple(path.name for path in destination.iterdir()) == (
        "sentinel.txt",
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        b"ALPACA_API_KEY=value",
        b"ALPACA_SECRET_KEY=value",
        b"account_number=123",
        b"buying_power=1000",
        b"portfolio_value=1000",
    ],
)
def test_redaction_check_rejects_sensitive_fields(forbidden: bytes) -> None:
    with pytest.raises(ValueError, match="forbidden sensitive field"):
        report_module._verify_redaction({"unsafe.txt": forbidden})


def test_artifacts_contain_no_dummy_credential_values(tmp_path: Path) -> None:
    report = build_day18_preflight_report(passing_preflight_result())
    destination = tmp_path / "day18"
    write_day18_preflight_artifacts(report, destination)
    combined = b"\n".join(path.read_bytes() for path in destination.iterdir())
    assert b"visible-key" not in combined
    assert b"visible-secret" not in combined
