"""Run the development-only Day 15 strategy-diversification study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day15_strategy_diversification_report import (
    APPROVED_DAY15_ARTIFACT_NAMES,
    Day15StrategyDiversificationReport,
    build_day15_strategy_diversification_report,
    write_day15_strategy_diversification_artifacts,
)
from systematic_alpha.analysis.strategy_diversification import (
    StrategyDiversificationResults,
    run_strategy_diversification,
)
from systematic_alpha.data.config_loader import find_project_root
try:
    from scripts.run_day10_trend_robustness import (
        validate_canonical_input,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_day10_trend_robustness import (
        validate_canonical_input,
    )


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day15")


class Day15RunnerError(ValueError):
    """Raised when the Day 15 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day15RunResult:
    """Immutable in-memory and written Day 15 run result."""

    dataset_path: Path
    artifact_directory: Path
    analysis_results: StrategyDiversificationResults
    report: Day15StrategyDiversificationReport
    artifact_paths: tuple[Path, ...]
    ensemble_feasible: bool

    def __post_init__(self) -> None:
        """Freeze path and artifact-path containers."""

        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(
            self,
            "artifact_directory",
            Path(self.artifact_directory),
        )
        object.__setattr__(
            self,
            "artifact_paths",
            tuple(Path(path) for path in self.artifact_paths),
        )
        object.__setattr__(
            self,
            "ensemble_feasible",
            bool(self.ensemble_feasible),
        )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the narrow deterministic Day 15 command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen development-only Day 15 "
            "strategy-diversification study."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args(argv)


def _read_selected_dataset(path: Path) -> pd.DataFrame:
    """Read one explicitly selected Parquet or CSV input."""

    if not isinstance(path, Path):
        raise TypeError("dataset path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}.")
    if not path.is_file():
        raise Day15RunnerError("Dataset path must be a file.")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise Day15RunnerError("Dataset must be Parquet or CSV.")


def _resolve_from_project_root(
    path: Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve one CLI path relative to the repository root."""

    if path.is_absolute():
        return path
    return project_root / path


def _display_path(path: Path, *, project_root: Path) -> str:
    """Display a repository-relative path when possible."""

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def execute_day15(
    *,
    dataset_path: str | Path,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day15RunResult:
    """Load, validate, analyze, report, and write one Day 15 run."""

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
    results = run_strategy_diversification(validated)
    report = build_day15_strategy_diversification_report(results)
    paths = write_day15_strategy_diversification_artifacts(
        report,
        output_directory,
        overwrite=overwrite,
    )

    if len(paths) != len(APPROVED_DAY15_ARTIFACT_NAMES):
        raise RuntimeError("Day 15 artifact writing did not complete.")
    ensemble_feasible = bool(
        report.ensemble_feasibility.iloc[0]["ensemble_feasible"]
    )
    return Day15RunResult(
        dataset_path=source_path,
        artifact_directory=output_directory,
        analysis_results=results,
        report=report,
        artifact_paths=paths,
        ensemble_feasible=ensemble_feasible,
    )


def main(
    argv: Sequence[str] | None = None,
) -> Day15RunResult:
    """Run the complete Day 15 command-line workflow."""

    arguments = parse_args(argv)
    project_root = find_project_root()
    dataset_path = _resolve_from_project_root(
        arguments.dataset_path,
        project_root=project_root,
    )
    artifact_directory = _resolve_from_project_root(
        arguments.artifact_directory,
        project_root=project_root,
    )
    result = execute_day15(
        dataset_path=dataset_path,
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )

    print("===== DAY 15 STRATEGY DIVERSIFICATION COMPLETE =====")
    print(
        "Source dataset:",
        _display_path(dataset_path, project_root=project_root),
    )
    print(
        "Artifact directory:",
        _display_path(artifact_directory, project_root=project_root),
    )
    for path in result.artifact_paths:
        print(_display_path(path, project_root=project_root))
    print("ensemble_feasible:", result.ensemble_feasible)
    return result


if __name__ == "__main__":
    main()
