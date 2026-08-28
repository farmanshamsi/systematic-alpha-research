"""Command-line contracts for the Day 16 runner."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_day16_portfolio_validation as runner
from systematic_alpha.analysis.day16_portfolio_validation_report import (
    APPROVED_DAY16_ARTIFACT_NAMES,
    build_day16_portfolio_validation_report,
)
from systematic_alpha.analysis.portfolio_allocation_validation import (
    analyze_portfolio_allocation_panel,
)
from tests.day16_fixtures import make_day16_panel


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
    assert runner.DEFAULT_ARTIFACT_DIRECTORY == Path("artifacts/day16")
    defaults = runner.parse_args([])
    assert defaults.dataset_path == runner.DEFAULT_DATASET_PATH
    assert defaults.artifact_directory == runner.DEFAULT_ARTIFACT_DIRECTORY
    assert defaults.overwrite is False

    selected = runner.parse_args(
        [
            "--dataset-path",
            "synthetic.csv",
            "--artifact-directory",
            "tmp/day16",
            "--overwrite",
        ]
    )
    assert selected.dataset_path == Path("synthetic.csv")
    assert selected.artifact_directory == Path("tmp/day16")
    assert selected.overwrite is True


def test_unsupported_suffix_missing_file_and_directory_are_rejected(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "synthetic.txt"
    unsupported.write_text("not market data", encoding="utf-8")
    with pytest.raises(runner.Day16RunnerError, match="Parquet or CSV"):
        runner._read_selected_dataset(unsupported)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        runner._read_selected_dataset(tmp_path / "missing.csv")
    with pytest.raises(runner.Day16RunnerError, match="must be a file"):
        runner._read_selected_dataset(tmp_path)


@pytest.mark.parametrize("invalid", [None, "path", 7])
def test_reader_requires_a_path_object(invalid: object) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        runner._read_selected_dataset(invalid)  # type: ignore[arg-type]


def test_execute_calls_validation_analysis_report_and_writer_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "synthetic.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    artifact_directory = tmp_path / "day16"
    loaded = make_selected_bars()
    validated = loaded.assign(validated=True)
    analysis_results = SimpleNamespace(evaluation_complete=True)
    report = object()
    calls: list[object] = []

    monkeypatch.setattr(
        runner,
        "_read_selected_dataset",
        lambda path: loaded.copy(deep=True),
    )

    def validator_spy(frame: pd.DataFrame) -> pd.DataFrame:
        calls.append(("validate", frame.copy(deep=True)))
        return validated.copy(deep=True)

    def analysis_spy(
        frame: pd.DataFrame,
        *,
        require_canonical_counts: bool,
    ) -> object:
        calls.append(
            (
                "analyze",
                frame.copy(deep=True),
                require_canonical_counts,
            )
        )
        return analysis_results

    def report_spy(results: object) -> object:
        calls.append(("report", results))
        return report

    def writer_spy(
        supplied: object,
        output: str | Path,
        *,
        overwrite: bool,
    ) -> tuple[Path, ...]:
        calls.append(("write", supplied, Path(output), overwrite))
        return tuple(
            Path(output) / name for name in APPROVED_DAY16_ARTIFACT_NAMES
        )

    monkeypatch.setattr(runner, "validate_canonical_input", validator_spy)
    monkeypatch.setattr(runner, "run_portfolio_allocation", analysis_spy)
    monkeypatch.setattr(
        runner,
        "build_day16_portfolio_validation_report",
        report_spy,
    )
    monkeypatch.setattr(
        runner,
        "write_day16_portfolio_validation_artifacts",
        writer_spy,
    )
    result = runner.execute_day16(
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
    assert calls[1][2] is True
    assert calls[-1][-1] is True
    assert result.analysis_results is analysis_results
    assert result.evaluation_complete is True
    assert is_dataclass(result)
    assert result.__dataclass_params__.frozen
    assert not hasattr(result, "__dict__")


def test_execute_with_synthetic_results_writes_only_approved_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "synthetic.csv"
    make_selected_bars().to_csv(dataset, index=False)
    output = tmp_path / "day16"
    synthetic_results = analyze_portfolio_allocation_panel(make_day16_panel())
    monkeypatch.setattr(
        runner,
        "validate_canonical_input",
        lambda frame: frame.copy(deep=True),
    )
    monkeypatch.setattr(
        runner,
        "run_portfolio_allocation",
        lambda frame, require_canonical_counts: synthetic_results,
    )
    result = runner.execute_day16(
        dataset_path=dataset,
        artifact_directory=output,
    )
    assert result.evaluation_complete is True
    assert {item.name for item in output.iterdir()} == set(
        APPROVED_DAY16_ARTIFACT_NAMES
    )
    assert tuple(path.name for path in result.artifact_paths) == (
        APPROVED_DAY16_ARTIFACT_NAMES
    )


def test_execute_rejects_incomplete_writes_and_mechanical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "synthetic.csv"
    dataset.write_text("placeholder\n", encoding="utf-8")
    incomplete = SimpleNamespace(evaluation_complete=False)
    monkeypatch.setattr(runner, "_read_selected_dataset", lambda path: object())
    monkeypatch.setattr(runner, "validate_canonical_input", lambda frame: frame)
    monkeypatch.setattr(
        runner,
        "run_portfolio_allocation",
        lambda frame, require_canonical_counts: incomplete,
    )
    monkeypatch.setattr(
        runner,
        "build_day16_portfolio_validation_report",
        lambda results: object(),
    )
    monkeypatch.setattr(
        runner,
        "write_day16_portfolio_validation_artifacts",
        lambda *args, **kwargs: (),
    )
    with pytest.raises(RuntimeError, match="writing did not complete"):
        runner.execute_day16(
            dataset_path=dataset,
            artifact_directory=tmp_path / "day16",
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("dataset_path", None),
        ("artifact_directory", 7),
        ("overwrite", 1),
    ),
)
def test_execute_argument_types_fail_before_loading(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "dataset_path": tmp_path / "synthetic.csv",
        "artifact_directory": tmp_path / "day16",
        "overwrite": False,
    }
    arguments[keyword] = value
    with pytest.raises(TypeError):
        runner.execute_day16(**arguments)  # type: ignore[arg-type]


def test_main_prints_paths_and_only_mechanical_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_results = analyze_portfolio_allocation_panel(make_day16_panel())
    report = build_day16_portfolio_validation_report(analysis_results)
    artifact_directory = tmp_path / "artifacts" / "day16"
    paths = tuple(
        artifact_directory / name for name in APPROVED_DAY16_ARTIFACT_NAMES
    )
    supplied: list[tuple[Path, Path, bool]] = []
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)

    def execute_spy(
        *,
        dataset_path: str | Path,
        artifact_directory: str | Path,
        overwrite: bool,
    ) -> runner.Day16RunResult:
        supplied.append(
            (Path(dataset_path), Path(artifact_directory), overwrite)
        )
        return runner.Day16RunResult(
            dataset_path=Path(dataset_path),
            artifact_directory=Path(artifact_directory),
            analysis_results=analysis_results,
            report=report,
            artifact_paths=paths,
            evaluation_complete=True,
        )

    monkeypatch.setattr(runner, "execute_day16", execute_spy)
    result = runner.main(
        [
            "--dataset-path",
            "synthetic.csv",
            "--artifact-directory",
            "artifacts/day16",
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
    assert "artifacts/day16" in output
    assert "evaluation_complete: True" in output
    assert "sharpe" not in output.lower()
    assert "winner" not in output.lower()
    assert result.evaluation_complete is True


def test_runner_source_is_thin_and_never_reads_day15_csv_artifacts() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8").lower()
    assert "validate_canonical_input" in source
    assert "run_portfolio_allocation" in source
    assert "build_day16_portfolio_validation_report" in source
    assert "write_day16_portfolio_validation_artifacts" in source
    assert "allocation_weights.csv" not in source
    assert "portfolio_return_panel.csv" not in source
    assert "artifacts/day15" not in source
    assert "pd.read_csv" in source
    assert "pd.read_parquet" in source
    for forbidden in ("ranking", "winner", "submit_order", "paper_order"):
        assert forbidden not in source
