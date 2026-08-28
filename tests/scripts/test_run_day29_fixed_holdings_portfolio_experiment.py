"""Tests for the thin development-only Day 29 runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day29_fixed_holdings_portfolio_experiment as runner


def test_parse_args_uses_frozen_paths_and_has_no_overwrite() -> None:
    args = runner.parse_args([])
    assert args.dataset_path == runner.DEFAULT_DATASET_PATH
    assert args.day16_comparator_directory == Path("artifacts/day16")
    assert args.day25_comparator_directory == Path(
        "artifacts/day25_causal_portfolio_finalization"
    )
    assert args.artifact_directory == Path(
        "artifacts/day29_fixed_holdings_portfolio_experiment"
    )
    assert not hasattr(args, "overwrite")
    assert runner.EXPECTED_SOURCE_SHA256 == (
        "30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2"
    )


def test_reader_rejects_missing_unsupported_and_non_path_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        runner._read_dataset(tmp_path / "missing.parquet")
    text = tmp_path / "bars.txt"
    text.write_text("not bars", encoding="utf-8")
    with pytest.raises(runner.Day29RunnerError, match="Parquet or CSV"):
        runner._read_dataset(text)
    with pytest.raises(TypeError, match="pathlib.Path"):
        runner._read_dataset("bars.csv")  # type: ignore[arg-type]


def _bar_range(*, end: str = "2025-12-31") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-02 14:30:00+00:00", f"{end} 20:45:00+00:00"],
                utc=True,
            ),
            "symbol": ["SPY", "SPY"],
        }
    )


def test_bar_audit_requires_exact_development_range_and_rejects_2026() -> None:
    audit = runner.audit_canonical_bar_range(_bar_range())
    assert audit["session_min"] == "2020-01-02"
    assert audit["session_max"] == "2025-12-31"
    assert audit["contains_2026_or_later"] is False
    with pytest.raises(runner.Day29RunnerError, match="2026"):
        runner.audit_canonical_bar_range(_bar_range(end="2026-01-02"))
    with pytest.raises(runner.Day29RunnerError, match="exactly"):
        runner.audit_canonical_bar_range(_bar_range(end="2025-12-30"))


def test_execute_runs_each_stage_once_and_preserves_exact_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / runner.DEFAULT_DATASET_PATH.name
    dataset.write_text("placeholder\n", encoding="utf-8")
    day16 = tmp_path / "day16"
    day25 = tmp_path / "day25"
    output = tmp_path / "artifacts" / runner.DEFAULT_ARTIFACT_DIRECTORY.name
    loaded = pd.DataFrame({"loaded": [1]})
    validated = _bar_range()
    panel = pd.DataFrame({"panel": [1]})
    historical = object()
    corrected = object()
    evidence_frame = pd.DataFrame({"sentinel": [1]})
    analysis = SimpleNamespace(
        source_and_method_metadata={"sentinel": True},
        target_and_covariance_invariance=evidence_frame,
        fold_performance_comparison=evidence_frame,
        aggregate_performance_comparison=evidence_frame,
        fold_turnover_comparison=evidence_frame,
        ending_weight_drift=evidence_frame,
        corrected_weight_path=evidence_frame,
        wealth_identity_checks=evidence_frame,
        portfolio_return_comparison=evidence_frame,
        comparator_snapshot={"x": "y"},
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(runner, "_read_dataset", lambda path: loaded.copy())
    monkeypatch.setattr(
        runner, "sha256_file", lambda path: runner.EXPECTED_SOURCE_SHA256
    )

    def validate(frame):
        calls.append(("validate_bars", frame.copy()))
        return validated.copy()

    def audit(frame):
        calls.append(("audit_bars", frame.copy()))
        return {"session_min": "2020-01-02", "session_max": "2025-12-31"}

    def diversify(frame):
        calls.append(("build_panel", frame.copy()))
        return SimpleNamespace(copy_session_return_panel=lambda: panel.copy())

    def historical_analysis(frame, *, require_canonical_counts):
        calls.append(("historical", require_canonical_counts))
        pd.testing.assert_frame_equal(frame, panel)
        return historical

    def corrected_analysis(frame, *, require_canonical_counts):
        calls.append(("corrected", require_canonical_counts))
        pd.testing.assert_frame_equal(frame, panel)
        return corrected

    def comparator_validation(value, *, day16_directory):
        calls.append(("validate_day16", day16_directory))
        assert value is historical

    def build(frame, **kwargs):
        calls.append(("build", kwargs))
        pd.testing.assert_frame_equal(frame, panel)
        assert kwargs["historical_results"] is historical
        assert kwargs["corrected_results"] is corrected
        assert kwargs["require_canonical_counts"] is True
        assert kwargs["require_exact_development_range"] is True
        return analysis

    def make_results(**kwargs):
        calls.append(("decorate_metadata", kwargs["source_and_method_metadata"]))
        return analysis

    def write(value, directory):
        calls.append(("write", directory))
        assert value is analysis
        return tuple(
            Path(directory) / name for name in runner.APPROVED_ARTIFACT_NAMES
        )

    monkeypatch.setattr(runner, "load_comparator_snapshot", lambda **kwargs: {"x": "y"})
    monkeypatch.setattr(runner, "validate_canonical_input", validate)
    monkeypatch.setattr(runner, "audit_canonical_bar_range", audit)
    monkeypatch.setattr(runner, "run_strategy_diversification", diversify)
    monkeypatch.setattr(
        runner, "analyze_portfolio_allocation_panel", historical_analysis
    )
    monkeypatch.setattr(
        runner,
        "analyze_portfolio_allocation_panel_fixed_holdings",
        corrected_analysis,
    )
    monkeypatch.setattr(
        runner, "validate_historical_day16_comparators", comparator_validation
    )
    monkeypatch.setattr(runner, "build_day29_experiment", build)
    monkeypatch.setattr(runner, "Day29FixedHoldingsExperimentResults", make_results)
    monkeypatch.setattr(runner, "write_day29_artifacts", write)
    result = runner.execute_day29(
        dataset_path=dataset,
        day16_comparator_directory=day16,
        day25_comparator_directory=day25,
        artifact_directory=output,
        generation_timestamp="fixed",
    )
    assert [name for name, _ in calls] == [
        "validate_bars",
        "audit_bars",
        "build_panel",
        "historical",
        "validate_day16",
        "corrected",
        "build",
        "decorate_metadata",
        "write",
    ]
    assert result.evaluation_complete is True
    assert result.source_sha256 == runner.EXPECTED_SOURCE_SHA256


def test_existing_output_fails_before_dataset_or_comparator_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / runner.DEFAULT_DATASET_PATH.name
    output = tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name
    output.mkdir()
    (output / "sentinel.csv").write_text("preserve\n", encoding="utf-8")
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("must not be called")

    monkeypatch.setattr(runner, "load_comparator_snapshot", forbidden)
    monkeypatch.setattr(runner, "_read_dataset", forbidden)
    with pytest.raises(FileExistsError, match="sentinel.csv"):
        runner.execute_day29(
            dataset_path=dataset,
            day16_comparator_directory=tmp_path / "day16",
            day25_comparator_directory=tmp_path / "day25",
            artifact_directory=output,
        )
    assert touched is False
    assert (output / "sentinel.csv").read_text(encoding="utf-8") == "preserve\n"


def test_execute_rejects_silent_dataset_substitution(tmp_path: Path) -> None:
    with pytest.raises(runner.Day29RunnerError, match="exact frozen Day 16"):
        runner.execute_day29(
            dataset_path=tmp_path / "different.csv",
            day16_comparator_directory=tmp_path / "day16",
            day25_comparator_directory=tmp_path / "day25",
            artifact_directory=(
                tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name
            ),
        )


def test_execute_rejects_wrong_canonical_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / runner.DEFAULT_DATASET_PATH.name
    dataset.write_text("different bytes\n", encoding="utf-8")
    monkeypatch.setattr(runner, "load_comparator_snapshot", lambda **kwargs: {"x": "y"})
    with pytest.raises(runner.Day29RunnerError, match="SHA-256"):
        runner.execute_day29(
            dataset_path=dataset,
            day16_comparator_directory=tmp_path / "day16",
            day25_comparator_directory=tmp_path / "day25",
            artifact_directory=(
                tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY.name
            ),
        )


def test_main_resolves_project_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = SimpleNamespace(
        dataset_path=tmp_path / runner.DEFAULT_DATASET_PATH,
        source_sha256="0" * 64,
        day16_comparator_directory=tmp_path / runner.DEFAULT_DAY16_COMPARATOR_DIRECTORY,
        day25_comparator_directory=tmp_path / runner.DEFAULT_DAY25_COMPARATOR_DIRECTORY,
        artifact_directory=tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY,
        artifact_paths=(),
        evaluation_complete=True,
    )
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "execute_day29", execute)
    assert runner.main([]) is expected
    assert captured == {
        "dataset_path": tmp_path / runner.DEFAULT_DATASET_PATH,
        "day16_comparator_directory": (
            tmp_path / runner.DEFAULT_DAY16_COMPARATOR_DIRECTORY
        ),
        "day25_comparator_directory": (
            tmp_path / runner.DEFAULT_DAY25_COMPARATOR_DIRECTORY
        ),
        "artifact_directory": tmp_path / runner.DEFAULT_ARTIFACT_DIRECTORY,
    }
