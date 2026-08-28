"""Run the frozen synthetic Day 22 execution and risk validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.analysis.day22_execution_report import write_day22_execution_artifacts
from systematic_alpha.analysis.day22_scenarios import run_day22_synthetic_scenarios
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day22")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Day 22 execution validation."
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def execute_day22(
    *, artifact_directory: str | Path, overwrite: bool = False
):
    if not isinstance(artifact_directory, (str, Path)):
        raise TypeError("artifact_directory must be a path.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    results = run_day22_synthetic_scenarios()
    paths = write_day22_execution_artifacts(
        results, Path(artifact_directory), overwrite=overwrite
    )
    return results, paths


def main(argv: Sequence[str] | None = None):
    arguments = parse_args(argv)
    root = find_project_root()
    output = arguments.artifact_directory
    if not output.is_absolute():
        output = root / output
    results, paths = execute_day22(
        artifact_directory=output, overwrite=arguments.overwrite
    )
    print("===== DAY 22 EXECUTION VALIDATION COMPLETE =====")
    print("execution_legs:", len(results.execution_shortfall))
    print("round_trips:", len(results.round_trip_pnl))
    print("daily_observations:", len(results.daily_performance))
    print("risk_metrics_available:", results.risk_summary[0]["risk_metrics_available"])
    print("artifact_files:", len(paths))
    print("broker_network_accessed: false")
    print("credentials_accessed: false")
    print("orders_submitted: false")
    print("live_campaign_authorized: false")
    return results, paths


if __name__ == "__main__":
    main()

