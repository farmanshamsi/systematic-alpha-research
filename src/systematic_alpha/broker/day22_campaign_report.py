"""Atomic, credential-free state and evidence for the Day 22 live campaign."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, Iterator, Mapping

from systematic_alpha.analysis.execution_performance_validation import ExecutionRecord
from systematic_alpha.broker.day22_calibration_campaign import (
    DAY22_AUTHORIZATION_SCOPE,
    DAY22_CAMPAIGN_ID,
    DAY22_GATE_ORDER,
    DAY22_LIVE_SCHEMA_VERSION,
    DAY22_MAX_ROUND_TRIPS,
    DAY22_MAX_SESSION_ROUND_TRIPS,
    DAY22_SLOT_WINDOW,
    CampaignSlot,
    Day22CampaignAuthorization,
    Day22SlotResult,
    frozen_campaign_slots,
)


ACTIVATION_FILENAME: Final[str] = "activation_manifest.json"
STATE_FILENAME: Final[str] = "campaign_state.json"
LOCK_FILENAME: Final[str] = ".campaign.lock"
SLOT_APPROVED_FILENAMES: Final[tuple[str, ...]] = (
    "gate_results.csv",
    "quote_snapshots.csv",
    "execution_records.csv",
    "order_events.csv",
    "position_cash_snapshots.csv",
    "result.json",
    "manifest.json",
)

GATE_COLUMNS: Final[tuple[str, ...]] = (
    "gate_order",
    "gate_id",
    "passed",
    "safe_detail",
)
QUOTE_COLUMNS: Final[tuple[str, ...]] = (
    "leg",
    "quote_at_utc",
    "bid_price",
    "ask_price",
    "arrival_mid",
    "local_submitted_at_utc",
)
EXECUTION_COLUMNS: Final[tuple[str, ...]] = (
    "execution_id",
    "round_trip_id",
    "purpose",
    "leg",
    "symbol",
    "side",
    "quantity",
    "decision_at_utc",
    "decision_price",
    "quote_at_utc",
    "bid_price",
    "ask_price",
    "local_submitted_at_utc",
    "broker_submitted_at_utc",
    "filled_at_utc",
    "fill_price",
    "commission",
)
ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "event_sequence",
    "leg",
    "broker_order_id",
    "client_order_id",
    "symbol",
    "side",
    "order_type",
    "time_in_force",
    "requested_quantity",
    "filled_quantity",
    "filled_average_price",
    "status",
    "broker_submitted_at_utc",
    "filled_at_utc",
)
POSITION_COLUMNS: Final[tuple[str, ...]] = (
    "phase",
    "observed_at_utc",
    "spy_quantity",
    "cash",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: tuple[Mapping[str, object], ...], columns: tuple[str, ...]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 22 live artifact schema changed.")
        writer.writerow(
            {
                key: (
                    "true"
                    if value is True
                    else "false" if value is False else "" if value is None else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def _iso(value: datetime | None) -> str:
    return "" if value is None else value.astimezone(timezone.utc).isoformat()


def _text(value: object | None) -> str:
    return "" if value is None else str(value)


def _slot_row(slot: CampaignSlot) -> dict[str, object]:
    return {
        "schedule_order": slot.schedule_order,
        "campaign_id": slot.campaign_id,
        "session_date": slot.session_date.isoformat(),
        "scheduled_at_utc": slot.scheduled_at.isoformat(),
        "scheduled_at_new_york": slot.scheduled_at_new_york,
        "entry_side": slot.entry_side,
        "quantity": str(slot.quantity),
        "maximum_notional_usd": str(slot.maximum_notional_usd),
        "purpose": slot.purpose,
    }


def _schedule_payload(slots: tuple[CampaignSlot, ...]) -> bytes:
    return _json_bytes([_slot_row(slot) for slot in slots])


def activation_manifest(
    authorization: Day22CampaignAuthorization,
) -> dict[str, object]:
    """Build the deterministic authorization and schedule freeze."""

    if not isinstance(authorization, Day22CampaignAuthorization):
        raise TypeError("authorization must be Day22CampaignAuthorization.")
    slots = frozen_campaign_slots()
    return {
        "schema_version": DAY22_LIVE_SCHEMA_VERSION,
        "campaign_id": DAY22_CAMPAIGN_ID,
        "authorization": {
            "approved": True,
            "scope": DAY22_AUTHORIZATION_SCOPE,
            "source": "explicit_user_approval",
            "recorded_on": "2026-08-03",
            "paper_endpoint": authorization.paper_endpoint,
            "real_money_trading_authorized": False,
            "maximum_entries": authorization.maximum_entries,
            "maximum_flattens": authorization.maximum_flattens,
            "maximum_round_trips_per_session": (
                authorization.maximum_round_trips_per_session
            ),
            "uncertain_fill_and_manual_recovery_acknowledged": True,
        },
        "activation_date": authorization.activation_date.isoformat(),
        "slot_window_seconds": int(DAY22_SLOT_WINDOW.total_seconds()),
        "schedule_sha256": hashlib.sha256(_schedule_payload(slots)).hexdigest(),
        "schedule": [_slot_row(slot) for slot in slots],
        "evidence_purpose": "calibration_probe",
        "alpha_eligible": False,
        "missed_slots_rescheduled": False,
    }


def initial_campaign_state(
    authorization: Day22CampaignAuthorization,
) -> dict[str, object]:
    manifest = activation_manifest(authorization)
    return {
        "schema_version": DAY22_LIVE_SCHEMA_VERSION,
        "campaign_id": DAY22_CAMPAIGN_ID,
        "activation_manifest_sha256": hashlib.sha256(
            _json_bytes(manifest)
        ).hexdigest(),
        "manual_recovery_required": False,
        "state_revision": 0,
        "slots": [
            {
                **row,
                "status": "authorized_scheduled",
                "outcome": None,
                "entry_submission_occurred": False,
                "flatten_submission_occurred": False,
                "entry_filled_quantity": "0",
                "flatten_filled_quantity": "0",
                "artifact_directory": None,
                "artifact_manifest_sha256": None,
                "updated_at_utc": None,
            }
            for row in manifest["schedule"]  # type: ignore[index]
        ],
    }


def _assert_no_credentials(payloads: Mapping[str, bytes]) -> None:
    forbidden = (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"APCA-API-KEY-ID",
        b"APCA-API-SECRET-KEY",
    )
    for name, payload in payloads.items():
        if any(marker in payload for marker in forbidden):
            raise ValueError(f"Credential-like content detected in {name}.")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def initialize_live_campaign(
    directory: str | Path,
    authorization: Day22CampaignAuthorization,
) -> tuple[Path, Path]:
    """Create or verify the exact activation freeze and initial state."""

    output = Path(directory)
    manifest_payload = _json_bytes(activation_manifest(authorization))
    state_payload = _json_bytes(initial_campaign_state(authorization))
    _assert_no_credentials(
        {ACTIVATION_FILENAME: manifest_payload, STATE_FILENAME: state_payload}
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / ACTIVATION_FILENAME
    state_path = output / STATE_FILENAME
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_payload:
            raise ValueError("Existing Day 22 activation freeze does not match.")
    else:
        _atomic_write(manifest_path, manifest_payload)
    if not state_path.exists():
        _atomic_write(state_path, state_payload)
    validate_campaign_state(output)
    return manifest_path, state_path


def load_campaign_state(directory: str | Path) -> dict[str, object]:
    try:
        value = json.loads((Path(directory) / STATE_FILENAME).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Day 22 campaign state cannot be read.") from exc
    if not isinstance(value, dict):
        raise ValueError("Day 22 campaign state is malformed.")
    return value


def validate_campaign_state(directory: str | Path) -> dict[str, object]:
    output = Path(directory)
    try:
        manifest_payload = (output / ACTIVATION_FILENAME).read_bytes()
        manifest = json.loads(manifest_payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Day 22 activation manifest cannot be read.") from exc
    state = load_campaign_state(output)
    if (
        manifest.get("schema_version") != DAY22_LIVE_SCHEMA_VERSION
        or manifest.get("campaign_id") != DAY22_CAMPAIGN_ID
        or state.get("schema_version") != DAY22_LIVE_SCHEMA_VERSION
        or state.get("campaign_id") != DAY22_CAMPAIGN_ID
        or state.get("activation_manifest_sha256")
        != hashlib.sha256(manifest_payload).hexdigest()
    ):
        raise ValueError("Day 22 campaign identity or activation hash changed.")
    slots = state.get("slots")
    if not isinstance(slots, list) or len(slots) != DAY22_MAX_ROUND_TRIPS:
        raise ValueError("Day 22 campaign state must contain ten slots.")
    orders = [row.get("schedule_order") for row in slots if isinstance(row, dict)]
    if orders != list(range(1, DAY22_MAX_ROUND_TRIPS + 1)):
        raise ValueError("Day 22 campaign state schedule order changed.")
    if any(
        row.get("purpose") != "calibration_probe"
        or row.get("quantity") != "0.01"
        for row in slots
    ):
        raise ValueError("Day 22 campaign scope changed in state.")
    in_progress = [row for row in slots if row.get("status") == "in_progress"]
    if in_progress and not state.get("manual_recovery_required"):
        raise ValueError("Interrupted campaign attempt is not recovery-latched.")
    return state


@contextmanager
def campaign_lock(directory: str | Path) -> Iterator[None]:
    """Prevent concurrent schedulers from consuming one slot twice."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / LOCK_FILENAME).open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Day 22 campaign process holds the lock.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_state(directory: Path, state: Mapping[str, object]) -> None:
    payload = _json_bytes(state)
    _assert_no_credentials({STATE_FILENAME: payload})
    _atomic_write(directory / STATE_FILENAME, payload)


def mark_missed_slots(
    directory: str | Path,
    *,
    observed_at: datetime,
) -> tuple[int, ...]:
    """Consume every expired scheduled slot without rescheduling it."""

    output = Path(directory)
    state = validate_campaign_state(output)
    now = observed_at.astimezone(timezone.utc)
    changed: list[int] = []
    slots = state["slots"]
    for row in slots:  # type: ignore[assignment]
        if row["status"] != "authorized_scheduled":
            continue
        scheduled = datetime.fromisoformat(row["scheduled_at_utc"])
        if now >= scheduled + DAY22_SLOT_WINDOW:
            row["status"] = "skipped_missed_window"
            row["outcome"] = "missed_not_rescheduled"
            row["updated_at_utc"] = now.isoformat()
            changed.append(int(row["schedule_order"]))
    if changed:
        state["state_revision"] = int(state["state_revision"]) + 1
        _write_state(output, state)
    return tuple(changed)


def select_due_slot(
    directory: str | Path,
    *,
    observed_at: datetime,
) -> CampaignSlot | None:
    state = validate_campaign_state(directory)
    now = observed_at.astimezone(timezone.utc)
    by_order = {slot.schedule_order: slot for slot in frozen_campaign_slots()}
    due = []
    for row in state["slots"]:  # type: ignore[assignment]
        scheduled = datetime.fromisoformat(row["scheduled_at_utc"])
        if (
            row["status"] == "authorized_scheduled"
            and scheduled <= now < scheduled + DAY22_SLOT_WINDOW
        ):
            due.append(by_order[int(row["schedule_order"])])
    if len(due) > 1:
        raise RuntimeError("More than one Day 22 slot is due.")
    return None if not due else due[0]


def campaign_counts(
    state: Mapping[str, object], slot: CampaignSlot
) -> tuple[int, int, bool, bool]:
    rows = state["slots"]
    attempted = [row for row in rows if row["entry_submission_occurred"]]  # type: ignore[index]
    session_attempted = [
        row
        for row in attempted
        if row["session_date"] == slot.session_date.isoformat()
    ]
    matching = [row for row in rows if row["schedule_order"] == slot.schedule_order]  # type: ignore[index]
    if len(matching) != 1:
        raise ValueError("Campaign slot is absent from state.")
    consumed = matching[0]["status"] != "authorized_scheduled"
    return (
        len(attempted),
        len(session_attempted),
        bool(state["manual_recovery_required"]),
        consumed,
    )


def begin_slot_attempt(
    directory: str | Path,
    slot: CampaignSlot,
    *,
    observed_at: datetime,
) -> None:
    """Persist an intent before the first broker mutation."""

    output = Path(directory)
    state = validate_campaign_state(output)
    matches = [
        row for row in state["slots"] if row["schedule_order"] == slot.schedule_order  # type: ignore[index]
    ]
    if len(matches) != 1 or matches[0]["status"] != "authorized_scheduled":
        raise ValueError("Day 22 slot is already consumed.")
    matches[0]["status"] = "in_progress"
    matches[0]["outcome"] = "broker_submission_may_have_started"
    matches[0]["updated_at_utc"] = observed_at.astimezone(timezone.utc).isoformat()
    state["manual_recovery_required"] = True
    state["state_revision"] = int(state["state_revision"]) + 1
    _write_state(output, state)


def _slot_payloads(result: Day22SlotResult) -> dict[str, bytes]:
    if tuple(item.gate_id for item in result.gates) != DAY22_GATE_ORDER:
        raise ValueError("Day 22 live gate order changed.")
    gates = tuple(
        {
            "gate_order": index,
            "gate_id": item.gate_id,
            "passed": item.passed,
            "safe_detail": item.safe_detail,
        }
        for index, item in enumerate(result.gates, start=1)
    )
    quotes = tuple(
        {
            "leg": leg.leg,
            "quote_at_utc": "" if leg.quote is None else leg.quote.quote_at.isoformat(),
            "bid_price": "" if leg.quote is None else leg.quote.bid_price,
            "ask_price": "" if leg.quote is None else leg.quote.ask_price,
            "arrival_mid": "" if leg.quote is None else leg.quote.mid_price,
            "local_submitted_at_utc": leg.local_submitted_at.isoformat(),
        }
        for leg in result.legs
    )
    execution_rows: list[dict[str, object]] = []
    for leg in result.legs:
        final = leg.order_events[-1]
        if (
            leg.quote is None
            or final.filled_quantity <= 0
            or final.filled_average_price is None
            or final.filled_at is None
        ):
            continue
        execution_rows.append(
            {
                "execution_id": (
                    f"day22-c{result.slot.schedule_order:02d}-{leg.leg}"
                ),
                "round_trip_id": f"day22-c{result.slot.schedule_order:02d}",
                "purpose": result.slot.purpose,
                "leg": leg.leg,
                "symbol": final.symbol,
                "side": final.side,
                "quantity": final.filled_quantity,
                "decision_at_utc": leg.quote.quote_at.isoformat(),
                "decision_price": leg.quote.mid_price,
                "quote_at_utc": leg.quote.quote_at.isoformat(),
                "bid_price": leg.quote.bid_price,
                "ask_price": leg.quote.ask_price,
                "local_submitted_at_utc": leg.local_submitted_at.isoformat(),
                "broker_submitted_at_utc": final.submitted_at.isoformat(),
                "filled_at_utc": final.filled_at.isoformat(),
                "fill_price": final.filled_average_price,
                "commission": Decimal("0"),
            }
        )
    order_rows: list[dict[str, object]] = []
    sequence = 0
    for leg in result.legs:
        for item in leg.order_events:
            sequence += 1
            order_rows.append(
                {
                    "event_sequence": sequence,
                    "leg": leg.leg,
                    "broker_order_id": item.broker_order_id,
                    "client_order_id": item.client_order_id,
                    "symbol": item.symbol,
                    "side": item.side,
                    "order_type": item.order_type,
                    "time_in_force": item.time_in_force,
                    "requested_quantity": item.requested_quantity,
                    "filled_quantity": item.filled_quantity,
                    "filled_average_price": _text(item.filled_average_price),
                    "status": item.status,
                    "broker_submitted_at_utc": item.submitted_at.isoformat(),
                    "filled_at_utc": _iso(item.filled_at),
                }
            )
    positions = tuple(
        {
            "phase": item.phase,
            "observed_at_utc": item.observed_at.isoformat(),
            "spy_quantity": item.spy_quantity,
            "cash": item.cash,
        }
        for item in result.position_cash
    )
    summary = {
        "schema_version": result.schema_version,
        "campaign_id": result.slot.campaign_id,
        "schedule_order": result.slot.schedule_order,
        "session_date": result.slot.session_date.isoformat(),
        "scheduled_at_utc": result.slot.scheduled_at.isoformat(),
        "purpose": result.slot.purpose,
        "alpha_eligible": False,
        "entry_client_order_id": result.entry_client_order_id,
        "flatten_client_order_id": result.flatten_client_order_id,
        "entry_submission_occurred": result.entry_submission_occurred,
        "flatten_submission_occurred": result.flatten_submission_occurred,
        "entry_filled_quantity": str(result.entry_filled_quantity),
        "flatten_filled_quantity": str(result.flatten_filled_quantity),
        "realized_round_trip_pnl": _text(result.realized_round_trip_pnl),
        "execution_complete": result.execution_complete,
        "shutdown_reconciled": result.shutdown_reconciled,
        "manual_recovery_required": result.manual_recovery_required,
        "outcome": result.outcome,
        "abort_reasons": list(result.abort_reasons),
        "paper_endpoint_only": True,
        "real_money_trading_authorized": False,
        "credentials_persisted": False,
    }
    return {
        "gate_results.csv": _csv_bytes(gates, GATE_COLUMNS),
        "quote_snapshots.csv": _csv_bytes(quotes, QUOTE_COLUMNS),
        "execution_records.csv": _csv_bytes(
            tuple(execution_rows), EXECUTION_COLUMNS
        ),
        "order_events.csv": _csv_bytes(tuple(order_rows), ORDER_COLUMNS),
        "position_cash_snapshots.csv": _csv_bytes(positions, POSITION_COLUMNS),
        "result.json": _json_bytes(summary),
    }


def write_slot_artifacts(
    result: Day22SlotResult,
    directory: str | Path,
) -> Path:
    """Atomically persist one immutable slot evidence directory."""

    if not isinstance(result, Day22SlotResult):
        raise TypeError("result must be Day22SlotResult.")
    root = Path(directory)
    output = root / f"slot_{result.slot.schedule_order:02d}"
    payloads = _slot_payloads(result)
    manifest = {
        "schema_version": DAY22_LIVE_SCHEMA_VERSION,
        "campaign_id": DAY22_CAMPAIGN_ID,
        "schedule_order": result.slot.schedule_order,
        "artifact_order": list(SLOT_APPROVED_FILENAMES),
        "hash_algorithm": "sha256",
        "hashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
        "credential_values_persisted": False,
        "calibration_pnl_alpha_eligible": False,
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    _assert_no_credentials(payloads)
    if output.exists():
        raise FileExistsError(f"Day 22 slot artifact already exists: {output}")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=root))
    try:
        for name in SLOT_APPROVED_FILENAMES:
            (stage / name).write_bytes(payloads[name])
        os.replace(stage, output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    if {path.name for path in output.iterdir()} != set(SLOT_APPROVED_FILENAMES):
        raise RuntimeError("Day 22 slot artifact allow-list changed.")
    return output


def finalize_slot_state(
    directory: str | Path,
    result: Day22SlotResult,
    artifact_directory: Path,
    *,
    observed_at: datetime,
) -> None:
    output = Path(directory)
    state = load_campaign_state(output)
    matches = [
        row
        for row in state["slots"]  # type: ignore[index]
        if row["schedule_order"] == result.slot.schedule_order
    ]
    if len(matches) != 1 or matches[0]["status"] not in {
        "authorized_scheduled",
        "in_progress",
    }:
        raise ValueError("Day 22 slot state cannot be finalized.")
    row = matches[0]
    row["status"] = (
        "completed_reconciled"
        if result.execution_complete
        else (
            "manual_recovery_required"
            if result.manual_recovery_required
            else "skipped_or_incomplete_reconciled"
        )
    )
    row["outcome"] = result.outcome
    row["entry_submission_occurred"] = result.entry_submission_occurred
    row["flatten_submission_occurred"] = result.flatten_submission_occurred
    row["entry_filled_quantity"] = str(result.entry_filled_quantity)
    row["flatten_filled_quantity"] = str(result.flatten_filled_quantity)
    row["artifact_directory"] = artifact_directory.name
    row["artifact_manifest_sha256"] = hashlib.sha256(
        (artifact_directory / "manifest.json").read_bytes()
    ).hexdigest()
    row["updated_at_utc"] = observed_at.astimezone(timezone.utc).isoformat()
    state["manual_recovery_required"] = result.manual_recovery_required
    state["state_revision"] = int(state["state_revision"]) + 1
    _write_state(output, state)
    validate_campaign_state(output)


def campaign_status_summary(directory: str | Path) -> dict[str, object]:
    state = validate_campaign_state(directory)
    rows = state["slots"]
    return {
        "campaign_id": DAY22_CAMPAIGN_ID,
        "authorized_slots": DAY22_MAX_ROUND_TRIPS,
        "entry_submissions": sum(
            bool(row["entry_submission_occurred"]) for row in rows  # type: ignore[index]
        ),
        "flatten_submissions": sum(
            bool(row["flatten_submission_occurred"]) for row in rows  # type: ignore[index]
        ),
        "completed_round_trips": sum(
            row["status"] == "completed_reconciled" for row in rows  # type: ignore[index]
        ),
        "missed_slots": sum(
            row["status"] == "skipped_missed_window" for row in rows  # type: ignore[index]
        ),
        "remaining_slots": sum(
            row["status"] == "authorized_scheduled" for row in rows  # type: ignore[index]
        ),
        "manual_recovery_required": bool(state["manual_recovery_required"]),
        "state_revision": int(state["state_revision"]),
    }


def audit_live_campaign(directory: str | Path) -> dict[str, object]:
    """Verify state references, exact slot allow-lists, and every stored hash."""

    output = Path(directory)
    state = validate_campaign_state(output)
    referenced: set[str] = set()
    verified_files = 0
    for row in state["slots"]:  # type: ignore[assignment]
        artifact_name = row["artifact_directory"]
        if artifact_name is None:
            continue
        if artifact_name != f"slot_{int(row['schedule_order']):02d}":
            raise ValueError("Day 22 slot artifact reference changed.")
        referenced.add(artifact_name)
        slot_directory = output / artifact_name
        if {path.name for path in slot_directory.iterdir()} != set(
            SLOT_APPROVED_FILENAMES
        ):
            raise ValueError("Day 22 slot artifact allow-list changed.")
        manifest_payload = (slot_directory / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_payload).hexdigest() != row[
            "artifact_manifest_sha256"
        ]:
            raise ValueError("Day 22 state-to-slot manifest hash mismatch.")
        manifest = json.loads(manifest_payload)
        if (
            manifest.get("schema_version") != DAY22_LIVE_SCHEMA_VERSION
            or manifest.get("campaign_id") != DAY22_CAMPAIGN_ID
            or manifest.get("schedule_order") != row["schedule_order"]
            or manifest.get("artifact_order") != list(SLOT_APPROVED_FILENAMES)
        ):
            raise ValueError("Day 22 slot manifest identity changed.")
        payloads: dict[str, bytes] = {}
        for name, expected in manifest["hashes"].items():
            payload = (slot_directory / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Day 22 slot artifact hash mismatch.")
            payloads[name] = payload
            verified_files += 1
        _assert_no_credentials(payloads)
    actual = {
        path.name
        for path in output.iterdir()
        if path.is_dir() and path.name.startswith("slot_")
    }
    if actual != referenced:
        raise ValueError("Unreferenced Day 22 slot artifact directory detected.")
    return {
        **campaign_status_summary(output),
        "verified_slot_directories": len(referenced),
        "verified_non_manifest_files": verified_files,
        "hash_audit_passed": True,
        "credential_scan_passed": True,
    }


def load_reconciled_execution_records(
    directory: str | Path,
) -> tuple[ExecutionRecord, ...]:
    """Load only complete reconciled live calibration round trips."""

    output = Path(directory)
    audit_live_campaign(output)
    state = load_campaign_state(output)
    records: list[ExecutionRecord] = []
    for row in state["slots"]:  # type: ignore[assignment]
        if row["status"] != "completed_reconciled":
            continue
        artifact = output / row["artifact_directory"] / "execution_records.csv"
        with artifact.open("r", encoding="utf-8", newline="") as stream:
            execution_rows = tuple(csv.DictReader(stream))
        if len(execution_rows) != 2 or [
            item["leg"] for item in execution_rows
        ] != ["entry", "exit"]:
            raise ValueError(
                "Reconciled Day 22 slot lacks two complete execution records."
            )
        for item in execution_rows:
            records.append(
                ExecutionRecord(
                    execution_id=item["execution_id"],
                    round_trip_id=item["round_trip_id"],
                    purpose=item["purpose"],
                    leg=item["leg"],
                    symbol=item["symbol"],
                    side=item["side"],
                    quantity=Decimal(item["quantity"]),
                    decision_at=datetime.fromisoformat(item["decision_at_utc"]),
                    decision_price=Decimal(item["decision_price"]),
                    quote_at=datetime.fromisoformat(item["quote_at_utc"]),
                    bid_price=Decimal(item["bid_price"]),
                    ask_price=Decimal(item["ask_price"]),
                    submitted_at=datetime.fromisoformat(
                        item["local_submitted_at_utc"]
                    ),
                    broker_submitted_at=datetime.fromisoformat(
                        item["broker_submitted_at_utc"]
                    ),
                    filled_at=datetime.fromisoformat(item["filled_at_utc"]),
                    fill_price=Decimal(item["fill_price"]),
                    commission=Decimal(item["commission"]),
                )
            )
    return tuple(records)
