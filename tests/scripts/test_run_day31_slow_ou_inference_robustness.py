"""Tests for the isolated development-only Day 31 runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day31_slow_ou_inference_robustness as runner


def test_parse_args_uses_frozen_paths_and_has_no_overwrite() -> None:
    args = runner.parse_args([])
    assert args.dataset_path == runner.DEFAULT_DATASET_PATH
    assert args.day28_directory == Path("artifacts/day28_ou_causal_timing")
    assert args.artifact_directory == Path(
        "artifacts/day31_slow_ou_inference_robustness"
    )
    assert not hasattr(args, "overwrite")
    assert runner.EXPECTED_SOURCE_SHA256 == (
        "30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2"
    )


def test_reader_rejects_missing_unsupported_and_non_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner._read_dataset(tmp_path / "missing.parquet")
    unsupported = tmp_path / "bars.txt"
    unsupported.write_text("not bars\n", encoding="utf-8")
    with pytest.raises(runner.Day31RunnerError, match="Parquet or CSV"):
        runner._read_dataset(unsupported)
    with pytest.raises(TypeError, match="pathlib.Path"):
        runner._read_dataset("bars.csv")  # type: ignore[arg-type]


def test_execute_runs_analysis_and_writer_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / runner.DEFAULT_DATASET_PATH.name
    dataset.write_text("placeholder\n", encoding="utf-8")
    day28 = tmp_path / "day28"
    output = tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name
    loaded = pd.DataFrame({"sentinel": [1]})
    analysis = object()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(runner, "_read_dataset", lambda path: loaded.copy())
    monkeypatch.setattr(
        runner, "sha256_file", lambda path: runner.EXPECTED_SOURCE_SHA256
    )

    def analyze(frame, **kwargs):
        calls.append(("analyze", kwargs))
        pd.testing.assert_frame_equal(frame, loaded)
        assert kwargs["generation_timestamp"] == "fixed"
        return analysis

    def write(results, directory):
        calls.append(("write", directory))
        assert results is analysis
        return tuple(Path(directory) / name for name in runner.APPROVED_ARTIFACT_NAMES)

    monkeypatch.setattr(runner, "run_day31_slow_ou_robustness", analyze)
    monkeypatch.setattr(runner, "write_day31_artifacts", write)
    result = runner.execute_day31(
        dataset_path=dataset,
        day28_directory=day28,
        artifact_directory=output,
        generation_timestamp="fixed",
    )
    assert [name for name, _ in calls] == ["analyze", "write"]
    assert result.evaluation_complete is True
    assert result.source_sha256 == runner.EXPECTED_SOURCE_SHA256


def test_existing_output_refuses_before_source_or_comparator_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name
    output.mkdir()
    sentinel = output / "sentinel.csv"
    sentinel.write_text("preserve\n", encoding="utf-8")
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("must not be called")

    monkeypatch.setattr(runner, "sha256_file", forbidden)
    monkeypatch.setattr(runner, "_read_dataset", forbidden)
    monkeypatch.setattr(runner, "run_day31_slow_ou_robustness", forbidden)
    with pytest.raises(FileExistsError, match="sentinel.csv"):
        runner.execute_day31(
            dataset_path=tmp_path / runner.DEFAULT_DATASET_PATH.name,
            day28_directory=tmp_path / "day28",
            artifact_directory=output,
        )
    assert touched is False
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_execute_rejects_dataset_substitution_and_wrong_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(runner.Day31RunnerError, match="exact frozen"):
        runner.execute_day31(
            dataset_path=tmp_path / "different.parquet",
            day28_directory=tmp_path / "day28",
            artifact_directory=tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name,
        )
    dataset = tmp_path / runner.DEFAULT_DATASET_PATH.name
    dataset.write_text("wrong\n", encoding="utf-8")
    monkeypatch.setattr(runner, "sha256_file", lambda path: "0" * 64)
    with pytest.raises(runner.Day31RunnerError, match="SHA-256"):
        runner.execute_day31(
            dataset_path=dataset,
            day28_directory=tmp_path / "day28",
            artifact_directory=tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name,
        )


def test_main_resolves_project_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = SimpleNamespace(
        dataset_path=tmp_path / runner.DEFAULT_DATASET_PATH,
        source_sha256="0" * 64,
        day28_directory=tmp_path / runner.DEFAULT_DAY28_DIRECTORY,
        artifact_directory=tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY,
        artifact_paths=(),
        evaluation_complete=True,
    )
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "execute_day31", execute)
    assert runner.main([]) is expected
    assert captured == {
        "dataset_path": tmp_path / runner.DEFAULT_DATASET_PATH,
        "day28_directory": tmp_path / runner.DEFAULT_DAY28_DIRECTORY,
        "artifact_directory": tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY,
    }
