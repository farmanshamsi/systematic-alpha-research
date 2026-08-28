"""Run the development-only Day 28 corrected OU timing experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day28_ou_causal_timing import (
    APPROVED_ARTIFACT_NAMES,
    Day28OuCausalTimingResults,
    run_day28_ou_causal_timing,
    write_day28_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_DAY17_COMPARATOR_DIRECTORY: Final[Path] = Path("artifacts/day17")
DEFAULT_DAY26_COMPARATOR_DIRECTORY: Final[Path] = Path("artifacts/day26")
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day28_ou_causal_timing")


class Day28RunnerError(ValueError):
    """Raised when the Day 28 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day28RunResult:
    dataset_path: Path
    source_sha256: str
    day17_comparator_directory: Path
    day26_comparator_directory: Path
    artifact_directory: Path
    analysis_results: Day28OuCausalTimingResults
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the versioned development-only comparison of historical and "
            "corrected OU/VWAP timing."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--day17-comparator-directory",
        type=Path,
        default=DEFAULT_DAY17_COMPARATOR_DIRECTORY,
    )
    parser.add_argument(
        "--day26-comparator-directory",
        type=Path,
        default=DEFAULT_DAY26_COMPARATOR_DIRECTORY,
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    return parser.parse_args(argv)


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Development dataset does not exist: {path}.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Day28RunnerError("Development dataset must be Parquet or CSV.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_day28(
    *,
    dataset_path: str | Path,
    day17_comparator_directory: str | Path,
    day26_comparator_directory: str | Path,
    artifact_directory: str | Path,
) -> Day28RunResult:
    """Load, authenticate, calculate once, and write one isolated Day 28 bundle."""

    for name, value in (
        ("dataset_path", dataset_path),
        ("day17_comparator_directory", day17_comparator_directory),
        ("day26_comparator_directory", day26_comparator_directory),
        ("artifact_directory", artifact_directory),
    ):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be a path.")
    source_path = Path(dataset_path)
    day17_directory = Path(day17_comparator_directory)
    day26_directory = Path(day26_comparator_directory)
    output_directory = Path(artifact_directory)
    source_sha256 = _sha256(source_path)
    bars = _read_dataset(source_path)
    results = run_day28_ou_causal_timing(
        bars,
        source_dataset_path=source_path.resolve().as_posix(),
        source_sha256=source_sha256,
        day17_comparator_directory=day17_directory,
        day26_comparator_directory=day26_directory,
    )
    artifact_paths = write_day28_artifacts(results, output_directory)
    if len(artifact_paths) != len(APPROVED_ARTIFACT_NAMES):
        raise RuntimeError("Day 28 artifact writing did not complete.")
    return Day28RunResult(
        dataset_path=source_path,
        source_sha256=source_sha256,
        day17_comparator_directory=day17_directory,
        day26_comparator_directory=day26_directory,
        artifact_directory=output_directory,
        analysis_results=results,
        artifact_paths=artifact_paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day28RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    result = execute_day28(
        dataset_path=_resolve(arguments.dataset_path, project_root=project_root),
        day17_comparator_directory=_resolve(
            arguments.day17_comparator_directory, project_root=project_root
        ),
        day26_comparator_directory=_resolve(
            arguments.day26_comparator_directory, project_root=project_root
        ),
        artifact_directory=_resolve(
            arguments.artifact_directory, project_root=project_root
        ),
    )
    print("===== DAY 28 OU CAUSAL TIMING COMPLETE =====")
    print("Source dataset:", result.dataset_path)
    print("Source SHA-256:", result.source_sha256)
    print("Day 17 comparator:", result.day17_comparator_directory)
    print("Day 26 comparator:", result.day26_comparator_directory)
    print("Artifact directory:", result.artifact_directory)
    for path in result.artifact_paths:
        print(path)
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
