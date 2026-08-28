"""Run the frozen, offline Day 23 operational validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.operations.day23_report import write_day23_artifacts
from systematic_alpha.operations.runtime_validation import run_operational_validation


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day23")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Day 23 reproducible-operations validation."
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--probe-parent", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--clean-environment-validated", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _output_path(path: str | Path) -> Path:
    output = Path(path)
    return output if output.is_absolute() else PROJECT_ROOT / output


def execute_day23(
    *,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    probe_parent: str | Path | None = None,
    check_only: bool = False,
    clean_environment_validated: bool = False,
    overwrite: bool = False,
):
    """Run health checks and optionally emit the frozen evidence bundle."""

    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if probe_parent is not None and not isinstance(probe_parent, (str, Path)):
        raise TypeError("probe_parent must be a path or None.")
    if any(type(value) is not bool for value in (check_only, clean_environment_validated, overwrite)):
        raise TypeError("Day 23 flags must be booleans.")
    result = run_operational_validation(
        PROJECT_ROOT,
        probe_parent=probe_parent,
        clean_environment_validated=clean_environment_validated,
    )
    if not result.evaluation_complete:
        failed = [check.check_id for check in result.health_checks if not check.passed]
        raise RuntimeError("Day 23 health checks failed: " + ",".join(failed))
    paths: tuple[Path, ...] = ()
    if not check_only:
        paths = write_day23_artifacts(
            result,
            _output_path(artifact_directory),
            overwrite=overwrite,
        )
    return result, paths


def main(argv: Sequence[str] | None = None):
    arguments = parse_args(argv)
    result, paths = execute_day23(
        artifact_directory=arguments.artifact_directory,
        probe_parent=arguments.probe_parent,
        check_only=arguments.check_only,
        clean_environment_validated=arguments.clean_environment_validated,
        overwrite=arguments.overwrite,
    )
    summary = {
        "evaluation_complete": result.evaluation_complete,
        "health_checks_passed": sum(
            check.passed for check in result.health_checks
        ),
        "health_checks_total": len(result.health_checks),
        "dependency_rows": len(result.dependency_audit),
        "schedule_rows": len(result.schedule_entries),
        "clean_environment_validated": result.clean_environment_validated,
        "container_static_validation_passed": (
            result.container_static_validation_passed
        ),
        "container_runtime_available": result.container_runtime_available,
        "artifact_files": len(paths),
        "broker_network_accessed": False,
        "credentials_accessed": False,
        "orders_submitted": False,
        "locked_final_test_data_accessed": False,
    }
    print("===== DAY 23 OFFLINE OPERATIONAL VALIDATION COMPLETE =====")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return result, paths


if __name__ == "__main__":
    main()

