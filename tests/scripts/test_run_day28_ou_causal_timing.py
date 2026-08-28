"""Tests for the thin development-only Day 28 runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day28_ou_causal_timing as runner


def test_parse_args_uses_frozen_paths_and_has_no_overwrite() -> None:
    args = runner.parse_args([])
    assert args.dataset_path == runner.DEFAULT_DATASET_PATH
    assert args.day17_comparator_directory == Path("artifacts/day17")
    assert args.day26_comparator_directory == Path("artifacts/day26")
    assert args.artifact_directory == Path("artifacts/day28_ou_causal_timing")
    assert not hasattr(args, "overwrite")


def test_reader_rejects_missing_and_unsupported_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner._read_dataset(tmp_path / "missing.parquet")
    text = tmp_path / "bars.txt"
    text.write_text("not bars", encoding="utf-8")
    with pytest.raises(runner.Day28RunnerError, match="Parquet or CSV"):
        runner._read_dataset(text)


def test_execute_hashes_runs_and_writes_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "development.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    day17 = tmp_path / "day17"
    day26 = tmp_path / "day26"
    output = tmp_path / "day28_ou_causal_timing"
    loaded = pd.DataFrame({"sentinel": [1]})
    analysis = object()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(runner, "_read_dataset", lambda path: loaded.copy())

    def analyze(frame, **kwargs):
        calls.append(("analyze", kwargs))
        pd.testing.assert_frame_equal(frame, loaded)
        return analysis

    def write(results, directory):
        calls.append(("write", directory))
        assert results is analysis
        return tuple(Path(directory) / name for name in runner.APPROVED_ARTIFACT_NAMES)

    monkeypatch.setattr(runner, "run_day28_ou_causal_timing", analyze)
    monkeypatch.setattr(runner, "write_day28_artifacts", write)
    result = runner.execute_day28(
        dataset_path=dataset,
        day17_comparator_directory=day17,
        day26_comparator_directory=day26,
        artifact_directory=output,
    )
    assert [name for name, _ in calls] == ["analyze", "write"]
    assert len(result.source_sha256) == 64
    assert result.evaluation_complete is True


def test_main_resolves_all_project_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    expected = SimpleNamespace(
        dataset_path=tmp_path / runner.DEFAULT_DATASET_PATH,
        source_sha256="0" * 64,
        day17_comparator_directory=tmp_path / "artifacts/day17",
        day26_comparator_directory=tmp_path / "artifacts/day26",
        artifact_directory=tmp_path / "artifacts/day28_ou_causal_timing",
        artifact_paths=(),
        evaluation_complete=True,
    )
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "execute_day28", execute)
    assert runner.main([]) is expected
    assert captured == {
        "dataset_path": tmp_path / runner.DEFAULT_DATASET_PATH,
        "day17_comparator_directory": tmp_path / "artifacts/day17",
        "day26_comparator_directory": tmp_path / "artifacts/day26",
        "artifact_directory": tmp_path / "artifacts/day28_ou_causal_timing",
    }
