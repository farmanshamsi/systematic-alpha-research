"""Write the frozen, network-free Day 21 known-answer bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.broker.day21_report import write_day21_artifacts
from systematic_alpha.broker.day21_scenarios import run_day21_synthetic_happy_path
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day21/synthetic")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic Day 21 validation.")
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    arguments = parse_args(argv)
    root = find_project_root()
    output = arguments.artifact_directory
    if not output.is_absolute():
        output = root / output
    result = run_day21_synthetic_happy_path()
    write_day21_artifacts(result, output, overwrite=arguments.overwrite)
    print("===== DAY 21 SYNTHETIC VALIDATION COMPLETE =====")
    print("execution_complete:", result.execution_complete)
    print("shutdown_reconciled:", result.shutdown_reconciled)
    print("realized_round_trip_pnl:", result.realized_round_trip_pnl)
    print("broker_network_accessed: false")
    print("credentials_accessed: false")
    print("orders_submitted: false")
    return result


if __name__ == "__main__":
    main()

