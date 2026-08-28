"""Run the frozen Day 21 controlled Alpaca paper-execution protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    DAY21_AUTHORIZATION_SCOPE,
    AlpacaControlledPaperBroker,
    Day21Authorization,
    Day21ExecutionResult,
    run_controlled_paper_execution,
)
from systematic_alpha.broker.day21_report import write_day21_artifacts
from systematic_alpha.broker.day21_signal import OPERATIONAL_DATA_START, build_day21_signal
from systematic_alpha.data.alpaca_provider import AlpacaBarProvider
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day21/live")
DAY20_DIRECTORY: Final[Path] = Path("artifacts/day20")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded Alpaca paper-only Day 21 session."
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--authorized-paper-order", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def verify_day20_prerequisite(project_root: Path) -> bool:
    """Verify the frozen Day 20 hashes and passing reconciled scenario."""

    directory = project_root / DAY20_DIRECTORY
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        if manifest["evaluation_complete"] is not True:
            return False
        artifacts = manifest["artifacts"]
        for artifact in artifacts:
            name = artifact["filename"]
            expected = artifact["sha256"]
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                return False
        with (directory / "scenario_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = tuple(csv.DictReader(stream))
        reconciled = tuple(
            row for row in rows if row["scenario_id"] == "fully_reconciled"
        )
        return (
            len(reconciled) == 1
            and reconciled[0]["observed_operational_gate_passed"] == "true"
            and reconciled[0]["can_submit_orders"] == "false"
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def execute_day21(
    *,
    artifact_directory: str | Path,
    authorized_paper_order: bool,
    overwrite: bool = False,
) -> Day21ExecutionResult:
    """Fetch isolated operational bars, gate, execute if eligible, and persist."""

    if type(authorized_paper_order) is not bool or not authorized_paper_order:
        raise PermissionError(
            "Day 21 requires the explicit --authorized-paper-order flag."
        )
    project_root = find_project_root()
    broker = AlpacaControlledPaperBroker()
    clock = broker.get_market_clock()
    provider = AlpacaBarProvider()
    bars = provider.fetch_bars(
        symbols=["SPY"],
        start=OPERATIONAL_DATA_START,
        end=clock.timestamp,
        timeframe_minutes=15,
    )
    signal = build_day21_signal(bars, as_of=clock.timestamp)
    authorization = Day21Authorization(
        approved=True,
        scope=DAY21_AUTHORIZATION_SCOPE,
        paper_endpoint=ALPACA_PAPER_BASE_URL,
        kill_switch_armed=True,
    )
    result = run_controlled_paper_execution(
        broker,
        signal=signal,
        authorization=authorization,
        day20_gate_passed=verify_day20_prerequisite(project_root),
    )
    output = Path(artifact_directory)
    if not output.is_absolute():
        output = project_root / output
    write_day21_artifacts(result, output, overwrite=overwrite)
    return result


def main(argv: Sequence[str] | None = None) -> Day21ExecutionResult:
    arguments = parse_args(argv)
    result = execute_day21(
        artifact_directory=arguments.artifact_directory,
        authorized_paper_order=arguments.authorized_paper_order,
        overwrite=arguments.overwrite,
    )
    print("===== DAY 21 CONTROLLED ALPACA PAPER SESSION COMPLETE =====")
    print("outcome:", result.outcome)
    print("order_submission_occurred:", result.order_submission_occurred)
    print("entry_filled_quantity:", result.entry_filled_quantity)
    print("flatten_filled_quantity:", result.flatten_filled_quantity)
    print("shutdown_reconciled:", result.shutdown_reconciled)
    print("execution_complete:", result.execution_complete)
    print("manual_recovery_required:", result.manual_recovery_required)
    print("real_money_orders_submitted: false")
    print("credential_values_persisted: false")
    return result


if __name__ == "__main__":
    main()
