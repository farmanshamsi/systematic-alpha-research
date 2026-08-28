"""Deterministic artifact bundle for Day 20 reconciliation and monitoring."""

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

from systematic_alpha.broker.day20_scenarios import (
    OPERATIONAL_DECISION_COLUMNS,
    POSITION_CASH_COLUMNS,
    RECONCILIATION_DIAGNOSTIC_COLUMNS,
    RECONCILIATION_SUMMARY_COLUMNS,
    SCENARIO_ORDER,
    SCENARIO_SUMMARY_COLUMNS,
    STREAM_LOG_COLUMNS,
    Day20ScenarioResults,
)
from systematic_alpha.broker.monitoring import MONITOR_REASON_CODES
from systematic_alpha.broker.reconciliation import RECONCILIATION_REASON_CODES


SCENARIO_SUMMARY_FILENAME: Final[str] = "scenario_summary.csv"
RECONCILIATION_SUMMARY_FILENAME: Final[str] = "reconciliation_summary.csv"
RECONCILIATION_DIAGNOSTICS_FILENAME: Final[str] = (
    "reconciliation_diagnostics.csv"
)
POSITION_CASH_FILENAME: Final[str] = "position_cash_reconciliation.csv"
STREAM_LOG_FILENAME: Final[str] = "stream_transition_log.csv"
OPERATIONAL_DECISIONS_FILENAME: Final[str] = "operational_decisions.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY20_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SCENARIO_SUMMARY_FILENAME,
    RECONCILIATION_SUMMARY_FILENAME,
    RECONCILIATION_DIAGNOSTICS_FILENAME,
    POSITION_CASH_FILENAME,
    STREAM_LOG_FILENAME,
    OPERATIONAL_DECISIONS_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)
DAY20_REASON_CODES: Final[tuple[str, ...]] = (
    *RECONCILIATION_REASON_CODES,
    *MONITOR_REASON_CODES,
)
EXPECTED_ROW_COUNTS: Final[tuple[int, ...]] = (12, 12, 10, 24, 23, 12)


@dataclass(frozen=True, slots=True)
class Day20ReconciliationReport:
    results: Day20ScenarioResults
    report: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.results, Day20ScenarioResults):
            raise TypeError("results must be Day20ScenarioResults.")
        object.__setattr__(
            self, "manifest", MappingProxyType(dict(self.manifest))
        )


def _markdown(results: Day20ScenarioResults) -> str:
    scenario_lines = "\n".join(
        f"- {row['scenario_id']}: gate="
        f"`{str(row['observed_operational_gate_passed']).lower()}`, "
        f"stream=`{row['stream_final_state']}`, reasons="
        f"`{row['observed_reason_codes'] or 'none'}`"
        for row in results.scenario_summary
    )
    diagnostic_lines = "\n".join(
        f"- {row['scenario_id']}: `{row['reason_code']}` "
        f"({row['local_value']} vs {row['broker_value']})"
        for row in results.reconciliation_diagnostics
    )
    return f"""# Day 20 Synthetic Reconciliation and Monitoring

## Outcome

- Evaluation complete: `true`
- Frozen scenarios: `{len(results.scenario_summary)}`
- Scenario gates passed: `{
        sum(row['scenario_passed'] for row in results.scenario_summary)
    }`
- Reconciliation diagnostics: `{len(results.reconciliation_diagnostics)}`
- Position/cash comparisons: `{len(results.position_cash_reconciliation)}`
- Stream audit rows: `{len(results.stream_transition_log)}`
- Operational decisions: `{len(results.operational_decisions)}`
- Broker network accessed: `false`
- Credentials accessed: `false`
- Orders submitted: `false`
- Orders canceled or replaced: `false`
- Account or position mutation: `false`
- Canonical market data accessed: `false`
- Locked 2026 data accessed: `false`

## Scenario decisions

{scenario_lines}

## Reconciliation diagnostics

{diagnostic_lines}

The fully reconciled, partial-fill, and recovered-stream scenarios pass the
synthetic operational gate. Passing does not authorize order submission. Every
mismatch, limit breach, exhausted reconnect sequence, and kill-switch state
blocks the gate without changing broker or local economic state.

## Interpretation

Day 20 closes the synthetic reconciliation and monitoring gap between local
order state and independent broker order, fill, position, and cash snapshots.
It also demonstrates deterministic stale-stream detection, bounded `1/2/4`
backoff, terminal circuit breaking, exposure limits, and a latched kill switch.

These controls improve operational safety. They are not evidence of execution
quality, profitability, or correct behavior against a live broker stream. Day
21 requires a separately frozen controlled-paper protocol and explicit user
authorization before any paper order can be submitted.
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


def build_day20_reconciliation_report(
    results: Day20ScenarioResults,
) -> Day20ReconciliationReport:
    """Validate and build the frozen Day 20 report contract."""

    if not isinstance(results, Day20ScenarioResults):
        raise TypeError("results must be Day20ScenarioResults.")
    if not results.evaluation_complete:
        raise ValueError("Day 20 scenario evaluation is incomplete.")
    if tuple(row["scenario_id"] for row in results.scenario_summary) != (
        SCENARIO_ORDER
    ):
        raise ValueError("Day 20 scenario order changed.")
    observed_counts = (
        len(results.scenario_summary),
        len(results.reconciliation_summary),
        len(results.reconciliation_diagnostics),
        len(results.position_cash_reconciliation),
        len(results.stream_transition_log),
        len(results.operational_decisions),
    )
    if observed_counts != EXPECTED_ROW_COUNTS:
        raise ValueError("Day 20 result row counts changed.")
    if not all(row["scenario_passed"] for row in results.scenario_summary):
        raise ValueError("A Day 20 scenario gate failed.")
    if any(
        row["day20_order_submission_authorized"]
        or row["can_submit_orders"]
        for row in (
            *results.scenario_summary,
            *results.operational_decisions,
        )
    ):
        raise ValueError("Day 20 order authorization changed.")
    if any(
        row["reason_code"] not in DAY20_REASON_CODES
        for row in results.reconciliation_diagnostics
    ):
        raise ValueError("Day 20 diagnostic vocabulary changed.")
    _require_schema(
        results.scenario_summary,
        SCENARIO_SUMMARY_COLUMNS,
        name="scenario_summary",
    )
    _require_schema(
        results.reconciliation_summary,
        RECONCILIATION_SUMMARY_COLUMNS,
        name="reconciliation_summary",
    )
    _require_schema(
        results.reconciliation_diagnostics,
        RECONCILIATION_DIAGNOSTIC_COLUMNS,
        name="reconciliation_diagnostics",
    )
    _require_schema(
        results.position_cash_reconciliation,
        POSITION_CASH_COLUMNS,
        name="position_cash_reconciliation",
    )
    _require_schema(
        results.stream_transition_log,
        STREAM_LOG_COLUMNS,
        name="stream_transition_log",
    )
    _require_schema(
        results.operational_decisions,
        OPERATIONAL_DECISION_COLUMNS,
        name="operational_decisions",
    )

    manifest = {
        "schema_version": "day20_reconciliation_monitoring_artifacts_v1",
        "artifact_order": list(APPROVED_DAY20_ARTIFACT_NAMES),
        "scenario_order": list(SCENARIO_ORDER),
        "reason_codes": list(DAY20_REASON_CODES),
        "row_counts": {
            "scenario_summary": observed_counts[0],
            "reconciliation_summary": observed_counts[1],
            "reconciliation_diagnostics": observed_counts[2],
            "position_cash_reconciliation": observed_counts[3],
            "stream_transition_log": observed_counts[4],
            "operational_decisions": observed_counts[5],
        },
        "evaluation_complete": True,
        "safety": {
            "synthetic_only": True,
            "broker_network_accessed": False,
            "credentials_accessed": False,
            "order_submission_enabled": False,
            "order_submission_occurred": False,
            "order_cancel_or_replace_occurred": False,
            "account_or_position_mutation_occurred": False,
            "canonical_market_data_accessed": False,
            "locked_2026_data_accessed": False,
        },
    }
    return Day20ReconciliationReport(
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
            raise ValueError("Day 20 CSV schema or column order changed.")
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
            dict(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _payloads(report: Day20ReconciliationReport) -> dict[str, bytes]:
    results = report.results
    return {
        SCENARIO_SUMMARY_FILENAME: _csv_bytes(
            results.scenario_summary, SCENARIO_SUMMARY_COLUMNS
        ),
        RECONCILIATION_SUMMARY_FILENAME: _csv_bytes(
            results.reconciliation_summary,
            RECONCILIATION_SUMMARY_COLUMNS,
        ),
        RECONCILIATION_DIAGNOSTICS_FILENAME: _csv_bytes(
            results.reconciliation_diagnostics,
            RECONCILIATION_DIAGNOSTIC_COLUMNS,
        ),
        POSITION_CASH_FILENAME: _csv_bytes(
            results.position_cash_reconciliation, POSITION_CASH_COLUMNS
        ),
        STREAM_LOG_FILENAME: _csv_bytes(
            results.stream_transition_log, STREAM_LOG_COLUMNS
        ),
        OPERATIONAL_DECISIONS_FILENAME: _csv_bytes(
            results.operational_decisions, OPERATIONAL_DECISION_COLUMNS
        ),
        REPORT_FILENAME: report.report.encode("utf-8"),
    }


def _manifest_bytes(
    report: Day20ReconciliationReport,
    payloads: Mapping[str, bytes],
) -> bytes:
    manifest = dict(report.manifest)
    manifest["artifacts"] = [
        {
            "filename": filename,
            "bytes": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in APPROVED_DAY20_ARTIFACT_NAMES
        if filename != MANIFEST_FILENAME
    ]
    return _json_bytes(manifest)


def write_day20_reconciliation_artifacts(
    report: Day20ReconciliationReport,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly eight artifacts with atomic replacement and rollback."""

    if not isinstance(report, Day20ReconciliationReport):
        raise TypeError("report must be Day20ReconciliationReport.")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    destination = Path(output_directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.stage-", dir=destination.parent
        )
    )
    backup: Path | None = None
    try:
        payloads = _payloads(report)
        payloads[MANIFEST_FILENAME] = _manifest_bytes(report, payloads)
        if tuple(payloads) != APPROVED_DAY20_ARTIFACT_NAMES:
            raise RuntimeError("Day 20 payload allow-list mismatch.")
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
    if names != tuple(sorted(APPROVED_DAY20_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 20 artifact allow-list verification failed.")
    return tuple(
        destination / name for name in APPROVED_DAY20_ARTIFACT_NAMES
    )

