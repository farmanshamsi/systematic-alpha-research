"""Fail-closed dispatcher for the three frozen Day 23 job entry points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from scripts.run_day21_controlled_paper_execution import execute_day21
from scripts.run_day22_live_campaign import (
    DEFAULT_ARTIFACT_DIRECTORY as DAY22_ARTIFACT_DIRECTORY,
    execute_due_slot,
)
from scripts.run_day23_operational_validation import execute_day23


KNOWN_JOBS = (
    "health-smoke",
    "day22-campaign-once",
    "day21-strategy-once",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch one frozen operational job without retry."
    )
    parser.add_argument("job_id", choices=KNOWN_JOBS)
    parser.add_argument("--authorized-paper-campaign", action="store_true")
    parser.add_argument("--authorized-paper-order", action="store_true")
    return parser.parse_args(argv)


def run_scheduled_job(
    job_id: str,
    *,
    authorized_paper_campaign: bool = False,
    authorized_paper_order: bool = False,
) -> dict[str, object]:
    """Run exactly one known job; unknown or unauthorized jobs fail closed."""

    if type(authorized_paper_campaign) is not bool or type(authorized_paper_order) is not bool:
        raise TypeError("Authorization flags must be booleans.")
    if job_id == "health-smoke":
        if authorized_paper_campaign or authorized_paper_order:
            raise PermissionError("Health smoke accepts no order authorization.")
        result, paths = execute_day23(check_only=True)
        return {
            "job_id": job_id,
            "outcome": "passed",
            "health_checks_passed": sum(
                check.passed for check in result.health_checks
            ),
            "artifact_files_written": len(paths),
            "order_submission_occurred": False,
        }
    if job_id == "day22-campaign-once":
        if not authorized_paper_campaign or authorized_paper_order:
            raise PermissionError(
                "Day 22 requires only --authorized-paper-campaign."
            )
        return {
            "job_id": job_id,
            **execute_due_slot(artifact_directory=DAY22_ARTIFACT_DIRECTORY),
        }
    if job_id == "day21-strategy-once":
        if not authorized_paper_order or authorized_paper_campaign:
            raise PermissionError(
                "Day 21 requires only --authorized-paper-order."
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result = execute_day21(
            artifact_directory=Path("artifacts/day21/scheduled") / run_id,
            authorized_paper_order=True,
            overwrite=False,
        )
        return {
            "job_id": job_id,
            "outcome": result.outcome,
            "order_submission_occurred": result.order_submission_occurred,
            "execution_complete": result.execution_complete,
            "manual_recovery_required": result.manual_recovery_required,
        }
    raise ValueError(f"Unknown scheduled job: {job_id}")


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = parse_args(argv)
    result = run_scheduled_job(
        arguments.job_id,
        authorized_paper_campaign=arguments.authorized_paper_campaign,
        authorized_paper_order=arguments.authorized_paper_order,
    )
    print("===== SCHEDULED JOB COMPLETE =====")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()

