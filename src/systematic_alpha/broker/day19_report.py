"""Deterministic artifact bundle for the Day 19 order-state machine."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from systematic_alpha.broker.day19_scenarios import (
    SCENARIO_ORDER,
    Day19ScenarioResults,
)
from systematic_alpha.broker.order_state import REASON_CODES, STATUS_ORDER


SCENARIO_SUMMARY_FILENAME: Final[str] = "scenario_summary.csv"
FINAL_STATES_FILENAME: Final[str] = "final_states.csv"
TRANSITION_LOG_FILENAME: Final[str] = "transition_log.csv"
REJECTION_DIAGNOSTICS_FILENAME: Final[str] = "rejection_diagnostics.csv"
TIMEOUT_DIAGNOSTICS_FILENAME: Final[str] = "timeout_diagnostics.csv"
STATE_TRANSITION_MATRIX_FILENAME: Final[str] = "state_transition_matrix.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY19_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SCENARIO_SUMMARY_FILENAME,
    FINAL_STATES_FILENAME,
    TRANSITION_LOG_FILENAME,
    REJECTION_DIAGNOSTICS_FILENAME,
    TIMEOUT_DIAGNOSTICS_FILENAME,
    STATE_TRANSITION_MATRIX_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)
SCENARIO_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "expected_final_status",
    "observed_final_status",
    "expected_reason_code",
    "observed_reason_code",
    "accepted_events",
    "duplicate_events",
    "rejected_events",
    "timeout_events",
    "recovery_required",
    "global_recovery_required",
    "scenario_passed",
)
FINAL_STATES_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "client_order_id",
    "broker_order_id",
    "symbol",
    "side",
    "order_type",
    "time_in_force",
    "requested_quantity",
    "status",
    "filled_quantity",
    "filled_average_price",
    "last_provider_sequence",
    "last_event_at",
    "last_received_at",
    "replacement_order_id",
    "terminal",
    "recovery_required",
)
TRANSITION_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "audit_sequence",
    "client_order_id",
    "broker_order_id",
    "event_id",
    "provider_sequence",
    "previous_status",
    "incoming_status",
    "resulting_status",
    "action",
    "incremental_fill",
    "cumulative_filled_quantity",
    "reason_code",
    "event_at",
    "received_at",
    "recovery_required",
)
TIMEOUT_DIAGNOSTICS_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "client_order_id",
    "reason_code",
    "as_of",
    "reference_time",
    "elapsed_seconds",
    "last_provider_sequence",
    "recovery_required",
)
STATE_TRANSITION_MATRIX_COLUMNS: Final[tuple[str, ...]] = (
    "from_status",
    "to_status",
    "allowed",
    "terminal_from_status",
)


@dataclass(frozen=True, slots=True)
class Day19OrderStateReport:
    """Frozen report and manifest built from synthetic scenario results."""

    results: Day19ScenarioResults
    report: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.results, Day19ScenarioResults):
            raise TypeError("results must be Day19ScenarioResults.")
        object.__setattr__(
            self, "manifest", MappingProxyType(dict(self.manifest))
        )


def _markdown(results: Day19ScenarioResults) -> str:
    rejection_lines = "\n".join(
        f"- {row['scenario_id']}: `{row['reason_code']}`"
        for row in results.rejection_diagnostics
    )
    timeout_lines = "\n".join(
        f"- {row['scenario_id']}: `{row['reason_code']}` after "
        f"{row['elapsed_seconds']} seconds"
        for row in results.timeout_diagnostics
    )
    final_lines = "\n".join(
        f"- {row['scenario_id']}: status=`{row['status']}`, "
        f"filled={row['filled_quantity']}, "
        f"recovery_required={str(row['recovery_required']).lower()}"
        for row in results.final_states
    )
    return f"""# Day 19 Synthetic Order-State Machine

## Outcome

- Evaluation complete: `true`
- Frozen scenarios: `{len(results.scenario_summary)}`
- Scenario gates passed: `{
        sum(row['scenario_passed'] for row in results.scenario_summary)
    }`
- Final local states: `{len(results.final_states)}`
- Append-only audit rows: `{len(results.transition_log)}`
- Expected rejected messages: `{len(results.rejection_diagnostics)}`
- Timeout diagnostics: `{len(results.timeout_diagnostics)}`
- Transition-matrix rows: `{len(results.state_transition_matrix)}`
- Broker network accessed: `false`
- Credentials accessed: `false`
- Orders submitted: `false`
- Locked 2026 data accessed: `false`

## Final scenario states

{final_lines}

The unknown-order scenario intentionally has no final local state and instead
sets the global recovery flag.

## Expected rejected messages

{rejection_lines}

These rejected messages did not change status, cumulative fill, average fill
price, or the last accepted provider sequence. The affected known orders were
marked for recovery and later broker reconciliation.

## Timeout evidence

{timeout_lines}

A timeout does not invent a broker state or cancel an order. It preserves the
last known state and requires reconciliation.

## Interpretation

The state machine now provides deterministic client-order idempotency,
event-delivery idempotency, legal transition enforcement, cumulative-to-
incremental fill accounting, terminal immutability, explicit failure reasons,
and timeout recovery flags. These are operational controls, not evidence of
execution quality or profitability.

Day 20 should connect this local state to independent broker snapshots and
stream-health monitoring. Day 19 does not authorize any paper-order submission.
"""


def _require_schema(
    rows: tuple[Mapping[str, object], ...],
    columns: tuple[str, ...],
    *,
    name: str,
) -> None:
    for row in rows:
        if tuple(row) != columns:
            raise ValueError(f"{name} schema or column order changed.")


def build_day19_order_state_report(
    results: Day19ScenarioResults,
) -> Day19OrderStateReport:
    """Validate and build the frozen Day 19 report contract."""

    if not isinstance(results, Day19ScenarioResults):
        raise TypeError("results must be Day19ScenarioResults.")
    if not results.evaluation_complete:
        raise ValueError("Day 19 scenario evaluation is incomplete.")
    if tuple(row["scenario_id"] for row in results.scenario_summary) != (
        SCENARIO_ORDER
    ):
        raise ValueError("Day 19 scenario order changed.")
    expected_counts = (9, 8, 32, 3, 1, len(STATUS_ORDER) ** 2)
    observed_counts = (
        len(results.scenario_summary),
        len(results.final_states),
        len(results.transition_log),
        len(results.rejection_diagnostics),
        len(results.timeout_diagnostics),
        len(results.state_transition_matrix),
    )
    if observed_counts != expected_counts:
        raise ValueError("Day 19 result row counts changed.")
    if not all(row["scenario_passed"] for row in results.scenario_summary):
        raise ValueError("A Day 19 scenario gate failed.")
    if any(
        row["reason_code"] not in REASON_CODES
        for row in (
            *results.rejection_diagnostics,
            *results.timeout_diagnostics,
        )
    ):
        raise ValueError("Day 19 reason-code vocabulary changed.")

    _require_schema(
        results.scenario_summary,
        SCENARIO_SUMMARY_COLUMNS,
        name="scenario_summary",
    )
    _require_schema(
        results.final_states,
        FINAL_STATES_COLUMNS,
        name="final_states",
    )
    _require_schema(
        results.transition_log,
        TRANSITION_LOG_COLUMNS,
        name="transition_log",
    )
    _require_schema(
        results.rejection_diagnostics,
        TRANSITION_LOG_COLUMNS,
        name="rejection_diagnostics",
    )
    _require_schema(
        results.timeout_diagnostics,
        TIMEOUT_DIAGNOSTICS_COLUMNS,
        name="timeout_diagnostics",
    )
    _require_schema(
        results.state_transition_matrix,
        STATE_TRANSITION_MATRIX_COLUMNS,
        name="state_transition_matrix",
    )

    manifest = {
        "schema_version": "day19_order_state_artifacts_v1",
        "artifact_order": list(APPROVED_DAY19_ARTIFACT_NAMES),
        "scenario_order": list(SCENARIO_ORDER),
        "status_order": [status.value for status in STATUS_ORDER],
        "reason_codes": list(REASON_CODES),
        "row_counts": {
            "scenario_summary": observed_counts[0],
            "final_states": observed_counts[1],
            "transition_log": observed_counts[2],
            "rejection_diagnostics": observed_counts[3],
            "timeout_diagnostics": observed_counts[4],
            "state_transition_matrix": observed_counts[5],
        },
        "evaluation_complete": True,
        "safety": {
            "synthetic_only": True,
            "broker_network_accessed": False,
            "credentials_accessed": False,
            "order_submission_enabled": False,
            "order_submission_occurred": False,
            "account_mutation_occurred": False,
            "canonical_market_data_accessed": False,
            "locked_2026_data_accessed": False,
        },
    }
    return Day19OrderStateReport(
        results=results,
        report=_markdown(results),
        manifest=manifest,
    )


def _csv_bytes(
    rows: tuple[Mapping[str, object], ...],
    columns: tuple[str, ...],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 19 CSV schema or column order changed.")
        writer.writerow(
            {
                key: str(value).lower() if type(value) is bool else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payloads(report: Day19OrderStateReport) -> dict[str, bytes]:
    results = report.results
    return {
        SCENARIO_SUMMARY_FILENAME: _csv_bytes(
            results.scenario_summary, SCENARIO_SUMMARY_COLUMNS
        ),
        FINAL_STATES_FILENAME: _csv_bytes(
            results.final_states, FINAL_STATES_COLUMNS
        ),
        TRANSITION_LOG_FILENAME: _csv_bytes(
            results.transition_log, TRANSITION_LOG_COLUMNS
        ),
        REJECTION_DIAGNOSTICS_FILENAME: _csv_bytes(
            results.rejection_diagnostics, TRANSITION_LOG_COLUMNS
        ),
        TIMEOUT_DIAGNOSTICS_FILENAME: _csv_bytes(
            results.timeout_diagnostics, TIMEOUT_DIAGNOSTICS_COLUMNS
        ),
        STATE_TRANSITION_MATRIX_FILENAME: _csv_bytes(
            results.state_transition_matrix,
            STATE_TRANSITION_MATRIX_COLUMNS,
        ),
        REPORT_FILENAME: report.report.encode("utf-8"),
    }


def _manifest_bytes(
    report: Day19OrderStateReport,
    payloads: Mapping[str, bytes],
) -> bytes:
    manifest = dict(report.manifest)
    manifest["artifacts"] = [
        {
            "filename": filename,
            "bytes": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in APPROVED_DAY19_ARTIFACT_NAMES
        if filename != MANIFEST_FILENAME
    ]
    return _json_bytes(manifest)


def write_day19_order_state_artifacts(
    report: Day19OrderStateReport,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly eight artifacts with atomic replacement and rollback."""

    if not isinstance(report, Day19OrderStateReport):
        raise TypeError("report must be Day19OrderStateReport.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    destination = Path(output_directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        payloads = _payloads(report)
        payloads[MANIFEST_FILENAME] = _manifest_bytes(report, payloads)
        if tuple(payloads) != APPROVED_DAY19_ARTIFACT_NAMES:
            raise RuntimeError("Day 19 payload allow-list mismatch.")
        for filename, payload in payloads.items():
            (staged / filename).write_bytes(payload)
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup-",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(staged, destination)
        staged = Path()
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
            backup = None
        raise
    finally:
        if staged and staged.exists() and staged != Path():
            shutil.rmtree(staged)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)

    names = tuple(sorted(path.name for path in destination.iterdir()))
    if names != tuple(sorted(APPROVED_DAY19_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 19 artifact allow-list verification failed.")
    return tuple(destination / name for name in APPROVED_DAY19_ARTIFACT_NAMES)
