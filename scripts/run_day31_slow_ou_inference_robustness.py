"""Run the development-only Day 31 slow OU robustness experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day31_slow_ou_inference_robustness import (
    APPROVED_ARTIFACT_NAMES,
    Day31SlowOuRobustnessResults,
    describe_existing_bundle,
    run_day31_slow_ou_robustness,
    sha256_file,
    write_day31_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
EXPECTED_SOURCE_SHA256: Final[str] = (
    "30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2"
)
DEFAULT_DAY28_DIRECTORY: Final[Path] = Path("artifacts/day28_ou_causal_timing")
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day31_slow_ou_inference_robustness"
)


class Day31RunnerError(ValueError):
    """Raised when the Day 31 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day31RunResult:
    dataset_path: Path
    source_sha256: str
    day28_directory: Path
    artifact_directory: Path
    analysis_results: Day31SlowOuRobustnessResults
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen slow OU development-only inference sensitivities without "
            "selection or promotion."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--day28-directory", type=Path, default=DEFAULT_DAY28_DIRECTORY
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    return parser.parse_args(argv)


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _read_dataset(path: Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Development dataset does not exist: {path}.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Day31RunnerError("Development dataset must be Parquet or CSV.")


def execute_day31(
    *,
    dataset_path: str | Path,
    day28_directory: str | Path,
    artifact_directory: str | Path,
    generation_timestamp: str | None = None,
) -> Day31RunResult:
    """Authenticate, calculate, and write exactly one isolated Day 31 bundle."""

    for name, value in (
        ("dataset_path", dataset_path),
        ("day28_directory", day28_directory),
        ("artifact_directory", artifact_directory),
    ):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be a path.")
    source_path = Path(dataset_path)
    comparator_directory = Path(day28_directory)
    output_directory = Path(artifact_directory)
    if output_directory.exists():
        raise FileExistsError(
            "Day 31 output directory already exists; refusing overwrite. "
            + describe_existing_bundle(output_directory)
        )
    if source_path.name != DEFAULT_DATASET_PATH.name:
        raise Day31RunnerError("Day 31 requires the exact frozen canonical dataset.")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise Day31RunnerError("Day 31 canonical source SHA-256 mismatch.")
    bars = _read_dataset(source_path)
    timestamp = generation_timestamp or datetime.now(timezone.utc).isoformat()
    results = run_day31_slow_ou_robustness(
        bars,
        source_dataset_path=source_path.resolve().as_posix(),
        source_sha256=source_sha256,
        day28_directory=comparator_directory,
        generation_timestamp=timestamp,
    )
    artifact_paths = write_day31_artifacts(results, output_directory)
    if tuple(path.name for path in artifact_paths) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Day 31 artifact writing did not complete.")
    return Day31RunResult(
        dataset_path=source_path,
        source_sha256=source_sha256,
        day28_directory=comparator_directory,
        artifact_directory=output_directory,
        analysis_results=results,
        artifact_paths=artifact_paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day31RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    result = execute_day31(
        dataset_path=_resolve(arguments.dataset_path, project_root=project_root),
        day28_directory=_resolve(
            arguments.day28_directory, project_root=project_root
        ),
        artifact_directory=_resolve(
            arguments.artifact_directory, project_root=project_root
        ),
    )
    print("===== DAY 31 SLOW OU INFERENCE ROBUSTNESS COMPLETE =====")
    print("Source dataset:", result.dataset_path)
    print("Source SHA-256:", result.source_sha256)
    print("Day 28 comparator:", result.day28_directory)
    print("Artifact directory:", result.artifact_directory)
    for path in result.artifact_paths:
        print(path)
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
