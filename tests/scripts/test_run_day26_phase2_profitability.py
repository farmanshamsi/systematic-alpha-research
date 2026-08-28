"""Tests for the thin Day 26 runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.run_day26_phase2_profitability as runner


def test_parse_args_uses_frozen_defaults() -> None:
    args = runner.parse_args([])
    assert args.dataset_path == runner.DEFAULT_DATASET_PATH
    assert args.artifact_directory == runner.DEFAULT_ARTIFACT_DIRECTORY
    assert args.overwrite is False


def test_execute_day26_hashes_analyzes_and_writes(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "development.csv"
    dataset.write_text("timestamp,symbol\n2025-01-02T14:30:00Z,SPY\n", encoding="utf-8")
    output = tmp_path / "artifacts"
    observed: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(runner, "_read_dataset", lambda path: pd.DataFrame({"x": [1]}))

    def analyze(frame, *, source_dataset_id, source_sha256):
        observed["frame"] = frame
        observed["dataset_id"] = source_dataset_id
        observed["sha256"] = source_sha256
        return sentinel

    def write(results, directory, *, overwrite):
        observed["results"] = results
        observed["directory"] = directory
        observed["overwrite"] = overwrite
        return tuple(directory / name for name in runner.APPROVED_ARTIFACT_NAMES)

    monkeypatch.setattr(runner, "run_phase2_profitability", analyze)
    monkeypatch.setattr(runner, "write_phase2_artifacts", write)
    result = runner.execute_day26(
        dataset_path=dataset,
        artifact_directory=output,
        overwrite=True,
    )
    assert observed["dataset_id"] == dataset.name
    assert len(str(observed["sha256"])) == 64
    assert observed["results"] is sentinel
    assert observed["directory"] == output
    assert observed["overwrite"] is True
    assert result.evaluation_complete is True
