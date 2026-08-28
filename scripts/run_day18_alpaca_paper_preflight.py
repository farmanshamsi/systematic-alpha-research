"""Run the frozen read-only Day 18 Alpaca paper preflight."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from systematic_alpha.broker.day18_report import (
    APPROVED_DAY18_ARTIFACT_NAMES,
    Day18PreflightReport,
    build_day18_preflight_report,
    write_day18_preflight_artifacts,
)
from systematic_alpha.broker.paper_boundary import (
    AlpacaPaperBroker,
    PaperBrokerPreflightError,
    PreflightResult,
    ReadOnlyPaperBroker,
    run_paper_preflight,
)
from systematic_alpha.data.config_loader import find_project_root, load_project_config


DEFAULT_CONFIG_PATH: Final[Path] = Path("config/base.yaml")
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day18")


@dataclass(frozen=True, slots=True)
class Day18RunResult:
    """Immutable Day 18 broker preflight and artifact result."""

    artifact_directory: Path
    preflight: PreflightResult
    report: Day18PreflightReport
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the read-only Alpaca paper-broker Day 18 preflight."
    )
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display(path: Path, *, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def execute_day18(
    *,
    config: Mapping[str, object],
    artifact_directory: str | Path,
    overwrite: bool = False,
    broker: ReadOnlyPaperBroker | None = None,
) -> Day18RunResult:
    """Construct, preflight, report, and write exactly once in that order."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")

    adapter = broker or AlpacaPaperBroker(config=config)
    preflight = run_paper_preflight(adapter)
    report = build_day18_preflight_report(preflight)
    output_directory = Path(artifact_directory)
    paths = write_day18_preflight_artifacts(
        report, output_directory, overwrite=overwrite
    )
    if len(paths) != len(APPROVED_DAY18_ARTIFACT_NAMES):
        raise RuntimeError("Day 18 artifact writing did not complete.")
    return Day18RunResult(
        artifact_directory=output_directory,
        preflight=preflight,
        report=report,
        artifact_paths=paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day18RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    config_path = _resolve(arguments.config_path, project_root=project_root)
    artifact_directory = _resolve(
        arguments.artifact_directory, project_root=project_root
    )
    config = load_project_config(config_path)
    result = execute_day18(
        config=config,
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )
    if not result.preflight.preflight_passed:
        raise PaperBrokerPreflightError(
            "Day 18 paper preflight failed; inspect the redacted artifacts."
        )
    print("===== DAY 18 ALPACA PAPER PREFLIGHT COMPLETE =====")
    print(
        "Artifact directory:",
        _display(artifact_directory, project_root=project_root),
    )
    print("preflight_passed:", result.preflight.preflight_passed)
    print("order_submission_occurred: false")
    return result


if __name__ == "__main__":
    main()
