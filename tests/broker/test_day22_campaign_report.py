from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from systematic_alpha.broker.controlled_paper_execution import GateResult, PositionCashSnapshot
from systematic_alpha.broker.day22_calibration_campaign import (
    DAY22_GATE_ORDER,
    DAY22_LIVE_SCHEMA_VERSION,
    Day22SlotResult,
    authorized_day22_campaign,
    day22_client_order_ids,
    frozen_campaign_slots,
)
from systematic_alpha.broker.day22_campaign_report import (
    ACTIVATION_FILENAME,
    SLOT_APPROVED_FILENAMES,
    audit_live_campaign,
    begin_slot_attempt,
    campaign_counts,
    campaign_status_summary,
    finalize_slot_state,
    initialize_live_campaign,
    load_reconciled_execution_records,
    load_campaign_state,
    mark_missed_slots,
    select_due_slot,
    validate_campaign_state,
    write_slot_artifacts,
)


NOW = datetime(2026, 8, 3, 14, 15, 10, tzinfo=timezone.utc)


def skipped_result() -> Day22SlotResult:
    slot = frozen_campaign_slots()[0]
    entry_id, flatten_id = day22_client_order_ids(slot)
    return Day22SlotResult(
        schema_version=DAY22_LIVE_SCHEMA_VERSION,
        slot=slot,
        gates=tuple(
            GateResult(gate_id, gate_id != "fresh_valid_quote", "safe")
            for gate_id in DAY22_GATE_ORDER
        ),
        entry_client_order_id=entry_id,
        flatten_client_order_id=flatten_id,
        legs=(),
        position_cash=(
            PositionCashSnapshot(
                phase="startup",
                observed_at=NOW,
                spy_quantity=Decimal("0"),
                cash=Decimal("100000"),
            ),
        ),
        entry_submission_occurred=False,
        flatten_submission_occurred=False,
        entry_filled_quantity=Decimal("0"),
        flatten_filled_quantity=Decimal("0"),
        realized_round_trip_pnl=None,
        execution_complete=False,
        shutdown_reconciled=True,
        manual_recovery_required=False,
        outcome="skipped_before_submission",
        abort_reasons=("fresh_valid_quote",),
    )


def test_initialization_is_deterministic_and_frozen(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    first = initialize_live_campaign(output, authorized_day22_campaign())
    before = tuple(path.read_bytes() for path in first)
    second = initialize_live_campaign(output, authorized_day22_campaign())
    assert before == tuple(path.read_bytes() for path in second)
    state = validate_campaign_state(output)
    assert len(state["slots"]) == 10
    assert all(row["status"] == "authorized_scheduled" for row in state["slots"])
    manifest = json.loads((output / ACTIVATION_FILENAME).read_text("utf-8"))
    assert manifest["authorization"]["approved"]
    assert not manifest["authorization"]["real_money_trading_authorized"]
    assert not manifest["alpha_eligible"]


def test_activation_tamper_is_detected(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    initialize_live_campaign(output, authorized_day22_campaign())
    manifest_path = output / ACTIVATION_FILENAME
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["activation_date"] = "2026-08-04"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="activation hash"):
        validate_campaign_state(output)


def test_missed_slot_is_consumed_and_never_rescheduled(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    initialize_live_campaign(output, authorized_day22_campaign())
    slot = frozen_campaign_slots()[0]
    missed = mark_missed_slots(
        output,
        observed_at=slot.scheduled_at + timedelta(seconds=60),
    )
    assert missed == (1,)
    state = load_campaign_state(output)
    assert state["slots"][0]["status"] == "skipped_missed_window"
    assert select_due_slot(output, observed_at=NOW) is None


def test_intent_latches_state_before_broker_mutation(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    initialize_live_campaign(output, authorized_day22_campaign())
    slot = frozen_campaign_slots()[0]
    begin_slot_attempt(output, slot, observed_at=NOW)
    state = validate_campaign_state(output)
    total, session, latched, consumed = campaign_counts(state, slot)
    assert (total, session) == (0, 0)
    assert latched
    assert consumed
    assert state["slots"][0]["status"] == "in_progress"


def test_slot_artifacts_are_exact_hashed_and_finalize_state(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    initialize_live_campaign(output, authorized_day22_campaign())
    result = skipped_result()
    artifact = write_slot_artifacts(result, output)
    assert {path.name for path in artifact.iterdir()} == set(
        SLOT_APPROVED_FILENAMES
    )
    manifest = json.loads((artifact / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["hashes"].items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == expected
    finalize_slot_state(output, result, artifact, observed_at=NOW)
    state = validate_campaign_state(output)
    assert state["slots"][0]["status"] == "skipped_or_incomplete_reconciled"
    assert not state["manual_recovery_required"]
    summary = campaign_status_summary(output)
    assert summary["remaining_slots"] == 9
    assert summary["entry_submissions"] == 0
    audit = audit_live_campaign(output)
    assert audit["verified_slot_directories"] == 1
    assert audit["hash_audit_passed"]
    assert audit["credential_scan_passed"]
    assert load_reconciled_execution_records(output) == ()


def test_slot_writer_rejects_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "live_campaign"
    initialize_live_campaign(output, authorized_day22_campaign())
    result = skipped_result()
    write_slot_artifacts(result, output)
    with pytest.raises(FileExistsError):
        write_slot_artifacts(result, output)
