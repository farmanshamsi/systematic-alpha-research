"""Command-line contracts for the Day 15 runner."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day15_strategy_diversification as runner
from systematic_alpha.analysis.day15_strategy_diversification_report import (
    APPROVED_DAY15_ARTIFACT_NAMES,
    build_day15_strategy_diversification_report,
)
from systematic_alpha.analysis.strategy_diversification import (
    analyze_strategy_diversification_panel,
)
from tests.analysis.test_strategy_diversification_statistics import (
    make_weakly_correlated_panel,
)


def make_selected_bars() -> pd.DataFrame:
    """Build a tiny development-only loader result for monkeypatching."""

    return pd.DataFrame(
        {
            "timestamp": ["2025-12-31 14:30:00+00:00"],
            "symbol": ["SPY"],
            "close": [100.0],
        }
    )


def test_default_paths_and_cli_argument_parsing_are_frozen() -> None:
    assert runner.DEFAULT_DATASET_PATH == Path(
        "data/processed/bars/"
        "spy_qqq_iwm_15min_"
        "2020-01-02_2025-12-31_"
        "sip_v3_development_canonical.parquet"
    )
    assert runner.DEFAULT_ARTIFACT_DIRECTORY == Path("artifacts/day15")

    defaults = runner.parse_args([])
    assert defaults.dataset_path == runner.DEFAULT_DATASET_PATH
    assert defaults.artifact_directory == runner.DEFAULT_ARTIFACT_DIRECTORY
    assert defaults.overwrite is False

    selected = runner.parse_args(
        [
            "--dataset-path",
            "synthetic.csv",
            "--artifact-directory",
            "tmp/day15",
            "--overwrite",
        ]
    )
    assert selected.dataset_path == Path("synthetic.csv")
    assert selected.artifact_directory == Path("tmp/day15")
    assert selected.overwrite is True


def test_unsupported_suffix_and_missing_dataset_are_rejected(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "synthetic.txt"
    unsupported.write_text("not market data", encoding="utf-8")

    with pytest.raises(runner.Day15RunnerError, match="Parquet or CSV"):
        runner._read_selected_dataset(unsupported)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        runner._read_selected_dataset(tmp_path / "missing.csv")


def test_execute_calls_development_analysis_and_passes_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "synthetic.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    artifact_directory = tmp_path / "day15"
    loaded = make_selected_bars()
    validated = loaded.assign(validated=True)
    analysis_results = object()
    report = SimpleNamespace(
        ensemble_feasibility=pd.DataFrame(
            {"ensemble_feasible": [False]}
        )
    )
    calls: list[object] = []

    monkeypatch.setattr(
        runner,
        "_read_selected_dataset",
        lambda path: loaded.copy(deep=True),
    )

    def validator_spy(frame: pd.DataFrame) -> pd.DataFrame:
        calls.append(("validate", frame.copy(deep=True)))
        return validated.copy(deep=True)

    def analysis_spy(frame: pd.DataFrame) -> object:
        calls.append(("analyze", frame.copy(deep=True)))
        return analysis_results

    def report_spy(results: object) -> object:
        calls.append(("report", results))
        return report

    def writer_spy(
        supplied: object,
        output: str | Path,
        *,
        overwrite: bool = False,
    ) -> tuple[Path, ...]:
        calls.append(("write", supplied, Path(output), overwrite))
        return tuple(
            Path(output) / name for name in APPROVED_DAY15_ARTIFACT_NAMES
        )

    monkeypatch.setattr(runner, "validate_canonical_input", validator_spy)
    monkeypatch.setattr(runner, "run_strategy_diversification", analysis_spy)
    monkeypatch.setattr(
        runner,
        "build_day15_strategy_diversification_report",
        report_spy,
    )
    monkeypatch.setattr(
        runner,
        "write_day15_strategy_diversification_artifacts",
        writer_spy,
    )

    result = runner.execute_day15(
        dataset_path=dataset,
        artifact_directory=artifact_directory,
        overwrite=True,
    )

    assert [call[0] for call in calls] == [
        "validate",
        "analyze",
        "report",
        "write",
    ]
    assert calls[-1][-1] is True
    assert result.analysis_results is analysis_results
    assert result.ensemble_feasible is False
    assert is_dataclass(result)
    assert result.__dataclass_params__.frozen
    assert not hasattr(result, "__dict__")


def test_execute_with_synthetic_results_writes_only_approved_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "synthetic.csv"
    make_selected_bars().to_csv(dataset, index=False)
    output = tmp_path / "day15"
    synthetic_results = analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )

    monkeypatch.setattr(
        runner,
        "validate_canonical_input",
        lambda frame: frame.copy(deep=True),
    )
    monkeypatch.setattr(
        runner,
        "run_strategy_diversification",
        lambda frame: synthetic_results,
    )

    result = runner.execute_day15(
        dataset_path=dataset,
        artifact_directory=output,
    )

    assert result.ensemble_feasible is True
    assert {item.name for item in output.iterdir()} == set(
        APPROVED_DAY15_ARTIFACT_NAMES
    )
    assert tuple(path.name for path in result.artifact_paths) == (
        APPROVED_DAY15_ARTIFACT_NAMES
    )


def test_main_prints_paths_and_ensemble_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_results = analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )
    report = build_day15_strategy_diversification_report(
        analysis_results
    )
    artifact_directory = tmp_path / "artifacts" / "day15"
    paths = tuple(
        artifact_directory / name for name in APPROVED_DAY15_ARTIFACT_NAMES
    )
    supplied: list[tuple[Path, Path, bool]] = []

    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute_spy(
        *,
        dataset_path: str | Path,
        artifact_directory: str | Path,
        overwrite: bool,
    ) -> runner.Day15RunResult:
        supplied.append(
            (Path(dataset_path), Path(artifact_directory), overwrite)
        )
        return runner.Day15RunResult(
            dataset_path=Path(dataset_path),
            artifact_directory=Path(artifact_directory),
            analysis_results=analysis_results,
            report=report,
            artifact_paths=paths,
            ensemble_feasible=True,
        )

    monkeypatch.setattr(runner, "execute_day15", execute_spy)
    result = runner.main(
        [
            "--dataset-path",
            "synthetic.csv",
            "--artifact-directory",
            "artifacts/day15",
            "--overwrite",
        ]
    )
    output = capsys.readouterr().out

    assert supplied == [
        (
            tmp_path / "synthetic.csv",
            artifact_directory,
            True,
        )
    ]
    assert "synthetic.csv" in output
    assert "artifacts/day15" in output
    assert "ensemble_feasible: True" in output
    assert result.ensemble_feasible is True


def test_runner_source_contains_no_parallel_research_logic() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8").lower()

    assert "validate_canonical_input" in source
    assert "run_strategy_diversification" in source
    assert "build_day15_strategy_diversification_report" in source
    assert "write_day15_strategy_diversification_artifacts" in source
    assert "2026" not in source
    for forbidden in (
        "optimisation",
        "optimization",
        "ranking",
        "allocation",
        "locked_period",
    ):
        assert forbidden not in source
