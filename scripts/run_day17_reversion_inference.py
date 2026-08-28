"""Run the development-only Day 17 reversion and inference study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day17_reversion_inference_report import (
    APPROVED_DAY17_ARTIFACT_NAMES,
    Day17ReversionInferenceReport,
    build_day17_reversion_inference_report,
    write_day17_reversion_inference_artifacts,
)
from systematic_alpha.analysis.reversion_inference import (
    ReversionInferenceResults,
    run_reversion_inference,
)
from systematic_alpha.data.config_loader import find_project_root

try:
    from scripts.run_day10_trend_robustness import validate_canonical_input
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_day10_trend_robustness import validate_canonical_input


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day17")


class Day17RunnerError(ValueError):
    """Raised when the Day 17 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day17RunResult:
    dataset_path: Path
    artifact_directory: Path
    analysis_results: ReversionInferenceResults
    report: Day17ReversionInferenceReport
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen development-only Day 17 OU/VWAP reversion "
            "and statistical inference study."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_selected_dataset(path: Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        raise TypeError("dataset path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}.")
    if not path.is_file():
        raise Day17RunnerError("Dataset path must be a file.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Day17RunnerError("Dataset must be Parquet or CSV.")


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display(path: Path, *, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def execute_day17(
    *,
    dataset_path: str | Path,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day17RunResult:
    """Load, validate, analyze, report, and write one canonical Day 17 run."""

    if not isinstance(dataset_path, (str, Path)):
        raise TypeError("dataset_path must be a path.")
    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    source_path = Path(dataset_path)
    output_directory = Path(artifact_directory)
    source = _read_selected_dataset(source_path)
    validated = validate_canonical_input(source)
    results = run_reversion_inference(validated)
    report = build_day17_reversion_inference_report(results)
    paths = write_day17_reversion_inference_artifacts(
        report, output_directory, overwrite=overwrite
    )
    if len(paths) != len(APPROVED_DAY17_ARTIFACT_NAMES):
        raise RuntimeError("Day 17 artifact writing did not complete.")
    return Day17RunResult(
        dataset_path=source_path,
        artifact_directory=output_directory,
        analysis_results=results,
        report=report,
        artifact_paths=paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day17RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    dataset_path = _resolve(arguments.dataset_path, project_root=project_root)
    artifact_directory = _resolve(
        arguments.artifact_directory, project_root=project_root
    )
    result = execute_day17(
        dataset_path=dataset_path,
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )
    print("===== DAY 17 REVERSION INFERENCE COMPLETE =====")
    print("Source dataset:", _display(dataset_path, project_root=project_root))
    print(
        "Artifact directory:",
        _display(artifact_directory, project_root=project_root),
    )
    for path in result.artifact_paths:
        print(_display(path, project_root=project_root))
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
