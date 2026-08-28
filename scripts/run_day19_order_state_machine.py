"""Run the frozen synthetic Day 19 order-state machine scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.broker.day19_report import (
    APPROVED_DAY19_ARTIFACT_NAMES,
    Day19OrderStateReport,
    build_day19_order_state_report,
    write_day19_order_state_artifacts,
)
from systematic_alpha.broker.day19_scenarios import (
    Day19ScenarioResults,
    run_day19_scenarios,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day19")


@dataclass(frozen=True, slots=True)
class Day19RunResult:
    artifact_directory: Path
    scenario_results: Day19ScenarioResults
    report: Day19OrderStateReport
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen synthetic Day 19 order-state scenarios."
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


def execute_day19(
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day19RunResult:
    """Run scenarios, build report, and write artifacts exactly once."""

    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    results = run_day19_scenarios()
    report = build_day19_order_state_report(results)
    output_directory = Path(artifact_directory)
    paths = write_day19_order_state_artifacts(
        report,
        output_directory,
        overwrite=overwrite,
    )
    if len(paths) != len(APPROVED_DAY19_ARTIFACT_NAMES):
        raise RuntimeError("Day 19 artifact writing did not complete.")
    return Day19RunResult(
        artifact_directory=output_directory,
        scenario_results=results,
        report=report,
        artifact_paths=paths,
        evaluation_complete=results.evaluation_complete,
    )


def main(argv: Sequence[str] | None = None) -> Day19RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    artifact_directory = _resolve(
        arguments.artifact_directory,
        project_root=project_root,
    )
    result = execute_day19(
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )
    print("===== DAY 19 ORDER STATE MACHINE COMPLETE =====")
    print(
        "Artifact directory:",
        _display(artifact_directory, project_root=project_root),
    )
    print("scenarios:", len(result.scenario_results.scenario_summary))
    print("evaluation_complete:", result.evaluation_complete)
    print("broker_network_accessed: false")
    print("credentials_accessed: false")
    print("orders_submitted: false")
    return result


if __name__ == "__main__":
    main()
