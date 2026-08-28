"""Capture a mutation-free live Day 21 gate and signal bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.broker.controlled_paper_execution import day21_client_order_ids
from systematic_alpha.broker.day21_read_only import (
    AlpacaDay21ReadOnlyReader,
    build_day21_read_only_result,
)
from systematic_alpha.broker.day21_report import write_day21_artifacts
from systematic_alpha.broker.day21_signal import OPERATIONAL_DATA_START, build_day21_signal
from systematic_alpha.data.alpaca_provider import AlpacaBarProvider
from systematic_alpha.data.config_loader import find_project_root
try:
    from scripts.run_day21_controlled_paper_execution import verify_day20_prerequisite
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_day21_controlled_paper_execution import verify_day20_prerequisite


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day21/live_read_only")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mutation-free Day 21 live gates.")
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    arguments = parse_args(argv)
    project_root = find_project_root()
    reader = AlpacaDay21ReadOnlyReader()
    preliminary = reader.capture(entry_client_order_id="axiom-day21-spy-e-preliminary")
    bars = AlpacaBarProvider().fetch_bars(
        symbols=["SPY"],
        start=OPERATIONAL_DATA_START,
        end=preliminary.preflight.clock.timestamp,
        timeframe_minutes=15,
    )
    signal = build_day21_signal(bars, as_of=preliminary.preflight.clock.timestamp)
    entry_id, _ = day21_client_order_ids(signal)
    state = reader.capture(entry_client_order_id=entry_id)
    result = build_day21_read_only_result(
        signal=signal,
        state=state,
        day20_gate_passed=verify_day20_prerequisite(project_root),
    )
    output = arguments.artifact_directory
    if not output.is_absolute():
        output = project_root / output
    write_day21_artifacts(result, output, overwrite=arguments.overwrite)
    print("===== DAY 21 READ-ONLY LIVE PROBE COMPLETE =====")
    print("outcome:", result.outcome)
    print("abort_reasons:", "|".join(result.abort_reasons) or "none")
    print("shutdown_reconciled:", result.shutdown_reconciled)
    print("orders_submitted: false")
    print("orders_canceled: false")
    print("positions_mutated: false")
    print("credentials_persisted: false")
    return result


if __name__ == "__main__":
    main()
