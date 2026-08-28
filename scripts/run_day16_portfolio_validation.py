"""Run the development-only Day 16 portfolio validation study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day16_portfolio_validation_report import (
    APPROVED_DAY16_ARTIFACT_NAMES,
    Day16PortfolioValidationReport,
    build_day16_portfolio_validation_report,
    write_day16_portfolio_validation_artifacts,
)
from systematic_alpha.analysis.portfolio_allocation_validation import (
    PortfolioAllocationResults,
    run_portfolio_allocation,
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
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day16")


class Day16RunnerError(ValueError):
    """Raised when the Day 16 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day16RunResult:
    """Immutable in-memory and written Day 16 run result."""

    dataset_path: Path
    artifact_directory: Path
    analysis_results: PortfolioAllocationResults
    report: Day16PortfolioValidationReport
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool

    def __post_init__(self) -> None:
        """Freeze paths and normalize the mechanical completion field."""

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
            "evaluation_complete",
            bool(self.evaluation_complete),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the narrow deterministic Day 16 command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen development-only Day 16 portfolio "
            "allocation and economic validation study."
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_selected_dataset(path: Path) -> pd.DataFrame:
    """Read one explicitly selected Parquet or CSV dataset."""

    if not isinstance(path, Path):
        raise TypeError("dataset path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}.")
    if not path.is_file():
        raise Day16RunnerError("Dataset path must be a file.")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise Day16RunnerError("Dataset must be Parquet or CSV.")


def _resolve_from_project_root(path: Path, *, project_root: Path) -> Path:
    """Resolve one command-line path relative to the repository root."""

    if path.is_absolute():
        return path
    return project_root / path


def _display_path(path: Path, *, project_root: Path) -> str:
    """Display a repository-relative path when possible."""

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def execute_day16(
    *,
    dataset_path: str | Path,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day16RunResult:
    """Load, validate, analyze, report, and write one canonical Day 16 run."""

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
    results = run_portfolio_allocation(
        validated,
        require_canonical_counts=True,
    )
    report = build_day16_portfolio_validation_report(results)
    paths = write_day16_portfolio_validation_artifacts(
        report,
        output_directory,
        overwrite=overwrite,
    )
    if len(paths) != len(APPROVED_DAY16_ARTIFACT_NAMES):
        raise RuntimeError("Day 16 artifact writing did not complete.")
    if not results.evaluation_complete:
        raise RuntimeError("Day 16 mechanical evaluation did not complete.")

    return Day16RunResult(
        dataset_path=source_path,
        artifact_directory=output_directory,
        analysis_results=results,
        report=report,
        artifact_paths=paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day16RunResult:
    """Run the complete Day 16 command-line workflow."""

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
    result = execute_day16(
        dataset_path=dataset_path,
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )

    print("===== DAY 16 PORTFOLIO VALIDATION COMPLETE =====")
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
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
