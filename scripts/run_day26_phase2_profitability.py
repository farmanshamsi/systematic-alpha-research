"""Run the frozen development-only Day 26 profitability experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.phase2_profitability import (
    APPROVED_ARTIFACT_NAMES,
    Phase2ProfitabilityResults,
    run_phase2_profitability,
    write_phase2_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day26")


class Day26RunnerError(ValueError):
    """Raised when the Day 26 runner cannot execute safely."""


@dataclass(frozen=True, slots=True)
class Day26RunResult:
    dataset_path: Path
    source_sha256: str
    artifact_directory: Path
    analysis_results: Phase2ProfitabilityResults
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared development-only Day 26 Phase II "
            "profitability experiment."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Day26RunnerError("Dataset must be Parquet or CSV.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def execute_day26(
    *,
    dataset_path: str | Path,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day26RunResult:
    """Load, audit, evaluate, and write one deterministic Day 26 bundle."""

    if not isinstance(dataset_path, (str, Path)):
        raise TypeError("dataset_path must be a path.")
    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    source_path = Path(dataset_path)
    output_path = Path(artifact_directory)
    source_sha256 = _sha256(source_path)
    source = _read_dataset(source_path)
    results = run_phase2_profitability(
        source,
        source_dataset_id=source_path.name,
        source_sha256=source_sha256,
    )
    paths = write_phase2_artifacts(results, output_path, overwrite=overwrite)
    if len(paths) != len(APPROVED_ARTIFACT_NAMES):
        raise RuntimeError("Day 26 artifact writing did not complete.")
    return Day26RunResult(
        dataset_path=source_path,
        source_sha256=source_sha256,
        artifact_directory=output_path,
        analysis_results=results,
        artifact_paths=paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day26RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    dataset_path = _resolve(arguments.dataset_path, project_root=project_root)
    artifact_directory = _resolve(
        arguments.artifact_directory, project_root=project_root
    )
    result = execute_day26(
        dataset_path=dataset_path,
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )
    print("===== DAY 26 PHASE II PROFITABILITY COMPLETE =====")
    print("Source dataset:", dataset_path)
    print("Source SHA-256:", result.source_sha256)
    print("Artifact directory:", artifact_directory)
    for path in result.artifact_paths:
        print(path)
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
