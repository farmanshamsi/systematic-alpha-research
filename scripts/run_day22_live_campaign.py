"""Run the separately authorized Day 22 Alpaca paper calibration campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Final, Sequence

from systematic_alpha.broker.day22_calibration_campaign import (
    DAY22_SLOT_WINDOW,
    AlpacaDay22CampaignBroker,
    authorized_day22_campaign,
    frozen_campaign_slots,
    run_day22_calibration_slot,
)
from systematic_alpha.broker.day22_campaign_report import (
    begin_slot_attempt,
    campaign_counts,
    campaign_lock,
    campaign_status_summary,
    finalize_slot_state,
    initialize_live_campaign,
    load_campaign_state,
    mark_missed_slots,
    select_due_slot,
    validate_campaign_state,
    write_slot_artifacts,
)
from systematic_alpha.broker.controlled_paper_execution import DAY21_SYMBOL
try:
    from scripts.run_day21_controlled_paper_execution import (
        verify_day20_prerequisite,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_day21_controlled_paper_execution import (  # type: ignore[no-redef]
        verify_day20_prerequisite,
    )


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day22/live_campaign")
WATCH_POLL_SECONDS: Final[float] = 15.0
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Day 22 Alpaca paper calibration campaign."
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--authorized-paper-campaign", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.watch and arguments.preflight_only:
        parser.error("--watch and --preflight-only cannot be combined")
    return arguments


def _output_path(path: str | Path) -> Path:
    output = Path(path)
    return output if output.is_absolute() else PROJECT_ROOT / output


def initialize_authorized_campaign(directory: str | Path) -> Path:
    output = _output_path(directory)
    initialize_live_campaign(output, authorized_day22_campaign())
    return output


def run_read_only_preflight(directory: str | Path) -> dict[str, object]:
    """Verify current paper state without consuming a campaign slot."""

    output = initialize_authorized_campaign(directory)
    with campaign_lock(output):
        state = validate_campaign_state(output)
        broker = AlpacaDay22CampaignBroker()
        preflight = broker.run_preflight()
        clock = broker.get_market_clock()
        position = broker.get_position(DAY21_SYMBOL)
        open_orders = broker.get_open_orders(DAY21_SYMBOL)
        summary = {
            "paper_preflight_passed": preflight.preflight_passed,
            "paper_endpoint_verified": True,
            "market_is_open": clock.is_open,
            "clock_timestamp_utc": clock.timestamp.isoformat(),
            "spy_position_is_flat": position is None or position.quantity == 0,
            "open_spy_orders": len(open_orders),
            "manual_recovery_required": bool(state["manual_recovery_required"]),
            "day20_prerequisite_passed": verify_day20_prerequisite(
                PROJECT_ROOT
            ),
            "orders_submitted": False,
            "orders_canceled": False,
            "campaign_slot_consumed": False,
            "credentials_persisted": False,
            "real_money_endpoint_accessed": False,
        }
        return summary


def execute_due_slot(
    *,
    artifact_directory: str | Path,
    observed_at: datetime | None = None,
    broker: object | None = None,
    now: object | None = None,
    sleep: object | None = None,
) -> dict[str, object]:
    """Consume at most one due slot; do nothing when no exact slot is due."""

    output = initialize_authorized_campaign(artifact_directory)
    current = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with campaign_lock(output):
        missed = mark_missed_slots(output, observed_at=current)
        state = validate_campaign_state(output)
        if state["manual_recovery_required"]:
            return {
                "outcome": "manual_recovery_required",
                "missed_slots_recorded": list(missed),
                "order_submission_occurred": False,
                **campaign_status_summary(output),
            }
        slot = select_due_slot(output, observed_at=current)
        if slot is None:
            return {
                "outcome": "no_campaign_slot_due",
                "missed_slots_recorded": list(missed),
                "order_submission_occurred": False,
                **campaign_status_summary(output),
            }
        total, session, latched, consumed = campaign_counts(state, slot)
        live_broker = broker or AlpacaDay22CampaignBroker()
        now_callable = now if callable(now) else lambda: datetime.now(timezone.utc)
        sleep_callable = sleep if callable(sleep) else time.sleep
        result = run_day22_calibration_slot(
            live_broker,  # type: ignore[arg-type]
            slot=slot,
            authorization=authorized_day22_campaign(),
            day20_gate_passed=verify_day20_prerequisite(PROJECT_ROOT),
            prior_entry_attempts_total=total,
            prior_entry_attempts_session=session,
            manual_recovery_latched=latched,
            slot_already_consumed=consumed,
            now=now_callable,
            sleep=sleep_callable,
            before_entry_submit=lambda: begin_slot_attempt(
                output,
                slot,
                observed_at=now_callable(),
            ),
        )
        evidence_directory = write_slot_artifacts(result, output)
        finalize_slot_state(
            output,
            result,
            evidence_directory,
            observed_at=now_callable(),
        )
        return {
            "outcome": result.outcome,
            "schedule_order": slot.schedule_order,
            "missed_slots_recorded": list(missed),
            "order_submission_occurred": result.entry_submission_occurred,
            "flatten_submission_occurred": result.flatten_submission_occurred,
            "entry_filled_quantity": str(result.entry_filled_quantity),
            "flatten_filled_quantity": str(result.flatten_filled_quantity),
            "shutdown_reconciled": result.shutdown_reconciled,
            "manual_recovery_required": result.manual_recovery_required,
            "artifact_directory": str(evidence_directory),
            **campaign_status_summary(output),
        }


def _campaign_finished(at: datetime) -> bool:
    last = frozen_campaign_slots()[-1]
    return at >= last.scheduled_at + DAY22_SLOT_WINDOW


def watch_campaign(directory: str | Path) -> dict[str, object]:
    """Poll through the frozen campaign horizon without moving any slot."""

    output = initialize_authorized_campaign(directory)
    while True:
        current = datetime.now(timezone.utc)
        summary = execute_due_slot(
            artifact_directory=output,
            observed_at=current,
        )
        if summary["manual_recovery_required"] or _campaign_finished(current):
            return summary
        time.sleep(WATCH_POLL_SECONDS)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    os.chdir(PROJECT_ROOT)
    arguments = parse_args(argv)
    if not arguments.authorized_paper_campaign:
        raise PermissionError(
            "Day 22 requires the explicit --authorized-paper-campaign flag."
        )
    if arguments.preflight_only:
        result = run_read_only_preflight(arguments.artifact_directory)
    elif arguments.watch:
        result = watch_campaign(arguments.artifact_directory)
    else:
        result = execute_due_slot(artifact_directory=arguments.artifact_directory)
    print("===== DAY 22 ALPACA PAPER CALIBRATION STATUS =====")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print("real_money_orders_submitted: false")
    print("credential_values_persisted: false")
    return result


if __name__ == "__main__":
    main()
