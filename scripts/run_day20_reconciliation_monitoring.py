"""Run frozen synthetic Day 20 reconciliation and monitoring scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.broker.day20_report import (
    APPROVED_DAY20_ARTIFACT_NAMES,
    Day20ReconciliationReport,
    build_day20_reconciliation_report,
    write_day20_reconciliation_artifacts,
)
from systematic_alpha.broker.day20_scenarios import (
    Day20ScenarioResults,
    run_day20_scenarios,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day20")


@dataclass(frozen=True, slots=True)
class Day20RunResult:
    artifact_directory: Path
    scenario_results: Day20ScenarioResults
    report: Day20ReconciliationReport
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen synthetic Day 20 reconciliation scenarios."
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


def execute_day20(
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
) -> Day20RunResult:
    """Run scenarios, build the report, and write artifacts exactly once."""

    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    results = run_day20_scenarios()
    report = build_day20_reconciliation_report(results)
    output_directory = Path(artifact_directory)
    paths = write_day20_reconciliation_artifacts(
        report, output_directory, overwrite=overwrite
    )
    if len(paths) != len(APPROVED_DAY20_ARTIFACT_NAMES):
        raise RuntimeError("Day 20 artifact writing did not complete.")
    return Day20RunResult(
        artifact_directory=output_directory,
        scenario_results=results,
        report=report,
        artifact_paths=paths,
        evaluation_complete=results.evaluation_complete,
    )


def main(argv: Sequence[str] | None = None) -> Day20RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    artifact_directory = _resolve(
        arguments.artifact_directory, project_root=project_root
    )
    result = execute_day20(
        artifact_directory=artifact_directory,
        overwrite=arguments.overwrite,
    )
    print("===== DAY 20 RECONCILIATION AND MONITORING COMPLETE =====")
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

