"""Tests for the deterministic Day 17 report bundle."""

from __future__ import annotations

import hashlib
import json

import pytest

from systematic_alpha.analysis.day17_reversion_inference_report import (
    APPROVED_DAY17_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
    Day17ReportError,
    build_day17_reversion_inference_report,
    write_day17_reversion_inference_artifacts,
)
from systematic_alpha.analysis.reversion_inference import run_reversion_inference
from tests.day17_fixtures import make_day17_development_bars


@pytest.fixture(scope="module")
def report():
    results = run_reversion_inference(
        make_day17_development_bars(), bootstrap_replications=20
    )
    return build_day17_reversion_inference_report(results)


def test_report_contains_scope_inference_and_honest_boundary(report) -> None:
    assert "# Day 17 OU/VWAP Reversion and Statistical Inference" in report.report
    assert "Locked 2026 data were not accessed" in report.report
    assert "Profitability is not an acceptance condition" in report.report
    assert report.manifest["ranking_performed"] is False
    assert report.manifest["selection_performed"] is False
    assert report.manifest["inference_contract"]["bootstrap_replications"] == 20


def test_reordered_results_fail_closed() -> None:
    results = run_reversion_inference(
        make_day17_development_bars(), bootstrap_replications=5
    )
    results.signal_diagnostics.iloc[[0, 1]] = results.signal_diagnostics.iloc[
        [1, 0]
    ].to_numpy()
    with pytest.raises(Day17ReportError, match="row order"):
        build_day17_reversion_inference_report(results)


def test_writer_is_exact_hashed_and_replayable(report, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = write_day17_reversion_inference_artifacts(report, first)
    write_day17_reversion_inference_artifacts(report, second)
    assert tuple(path.name for path in paths) == APPROVED_DAY17_ARTIFACT_NAMES
    assert tuple(sorted(path.name for path in first.iterdir())) == tuple(
        sorted(APPROVED_DAY17_ARTIFACT_NAMES)
    )
    manifest = json.loads((first / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert MANIFEST_FILENAME not in {
        item["filename"] for item in manifest["artifacts"]
    }
    for item in manifest["artifacts"]:
        payload = (first / item["filename"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["bytes"] == len(payload)
    for name in APPROVED_DAY17_ARTIFACT_NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_writer_protects_existing_directory(report, tmp_path) -> None:
    destination = tmp_path / "day17"
    write_day17_reversion_inference_artifacts(report, destination)
    with pytest.raises(FileExistsError):
        write_day17_reversion_inference_artifacts(report, destination)
