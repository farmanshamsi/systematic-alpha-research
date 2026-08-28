"""Command-line contracts for the Day 17 runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day17_reversion_inference as runner
from systematic_alpha.analysis.day17_reversion_inference_report import (
    APPROVED_DAY17_ARTIFACT_NAMES,
)


def test_default_paths_and_cli_contract() -> None:
    defaults = runner.parse_args([])
    assert defaults.dataset_path == runner.DEFAULT_DATASET_PATH
    assert defaults.artifact_directory == Path("artifacts/day17")
    assert defaults.overwrite is False
    selected = runner.parse_args(
        [
            "--dataset-path",
            "synthetic.csv",
            "--artifact-directory",
            "tmp/day17",
            "--overwrite",
        ]
    )
    assert selected.dataset_path == Path("synthetic.csv")
    assert selected.artifact_directory == Path("tmp/day17")
    assert selected.overwrite is True


def test_reader_rejects_unsafe_paths(tmp_path: Path) -> None:
    text = tmp_path / "bars.txt"
    text.write_text("not bars", encoding="utf-8")
    with pytest.raises(runner.Day17RunnerError, match="Parquet or CSV"):
        runner._read_selected_dataset(text)
    with pytest.raises(FileNotFoundError):
        runner._read_selected_dataset(tmp_path / "missing.csv")
    with pytest.raises(runner.Day17RunnerError, match="must be a file"):
        runner._read_selected_dataset(tmp_path)


def test_execute_calls_pipeline_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    output = tmp_path / "day17"
    loaded = pd.DataFrame({"timestamp": ["2025-01-02"], "symbol": ["SPY"]})
    validated = loaded.assign(validated=True)
    analysis = object()
    report = object()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runner, "_read_selected_dataset", lambda path: loaded.copy())

    def validate(frame):
        calls.append(("validate", frame.copy()))
        return validated.copy()

    def analyze(frame):
        calls.append(("analyze", frame.copy()))
        return analysis

    def build(results):
        calls.append(("report", results))
        return report

    def write(supplied, directory, *, overwrite):
        calls.append(("write", supplied, Path(directory), overwrite))
        return tuple(Path(directory) / name for name in APPROVED_DAY17_ARTIFACT_NAMES)

    monkeypatch.setattr(runner, "validate_canonical_input", validate)
    monkeypatch.setattr(runner, "run_reversion_inference", analyze)
    monkeypatch.setattr(runner, "build_day17_reversion_inference_report", build)
    monkeypatch.setattr(runner, "write_day17_reversion_inference_artifacts", write)
    result = runner.execute_day17(
        dataset_path=dataset, artifact_directory=output, overwrite=True
    )
    assert [item[0] for item in calls] == ["validate", "analyze", "report", "write"]
    assert calls[-1][-1] is True
    assert result.analysis_results is analysis
    assert result.report is report
    assert result.evaluation_complete is True


def test_execute_rejects_incomplete_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_read_selected_dataset", lambda path: object())
    monkeypatch.setattr(runner, "validate_canonical_input", lambda frame: frame)
    monkeypatch.setattr(runner, "run_reversion_inference", lambda frame: object())
    monkeypatch.setattr(
        runner, "build_day17_reversion_inference_report", lambda results: object()
    )
    monkeypatch.setattr(
        runner, "write_day17_reversion_inference_artifacts", lambda *args, **kwargs: ()
    )
    with pytest.raises(RuntimeError, match="writing did not complete"):
        runner.execute_day17(dataset_path=dataset, artifact_directory=tmp_path / "out")


def test_main_resolves_project_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    expected = SimpleNamespace(artifact_paths=(), evaluation_complete=True)
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "execute_day17", execute)
    assert runner.main([]) is expected
    assert captured["dataset_path"] == tmp_path / runner.DEFAULT_DATASET_PATH
    assert captured["artifact_directory"] == tmp_path / "artifacts/day17"
