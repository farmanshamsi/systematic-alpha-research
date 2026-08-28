"""Deterministic report bundle for Day 22 execution validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, Mapping

from systematic_alpha.analysis.execution_performance_validation import (
    CAMPAIGN_SCHEDULE_COLUMNS,
    CAMPAIGN_SUMMARY_COLUMNS,
    DAILY_PERFORMANCE_COLUMNS,
    EXECUTION_SHORTFALL_COLUMNS,
    RISK_SUMMARY_COLUMNS,
    ROUND_TRIP_COLUMNS,
    Day22AnalysisResults,
)


EXECUTION_FILENAME: Final[str] = "execution_shortfall.csv"
ROUND_TRIP_FILENAME: Final[str] = "round_trip_pnl.csv"
DAILY_FILENAME: Final[str] = "daily_performance.csv"
RISK_FILENAME: Final[str] = "risk_summary.csv"
SCHEDULE_FILENAME: Final[str] = "campaign_schedule.csv"
CAMPAIGN_FILENAME: Final[str] = "campaign_summary.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY22_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    EXECUTION_FILENAME,
    ROUND_TRIP_FILENAME,
    DAILY_FILENAME,
    RISK_FILENAME,
    SCHEDULE_FILENAME,
    CAMPAIGN_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)
EXPECTED_ROW_COUNTS: Final[tuple[int, ...]] = (8, 4, 25, 1, 10, 2)


def _csv_bytes(
    rows: tuple[Mapping[str, object], ...], columns: tuple[str, ...]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 22 artifact schema changed.")
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


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _payloads(results: Day22AnalysisResults) -> dict[str, bytes]:
    risk = results.risk_summary[0]
    strategy = results.campaign_summary[0]
    calibration = results.campaign_summary[1]
    report = f"""# Day 22 Execution and Live-Performance Validation

## Outcome

- Evaluation complete: `true`
- Evidence type: `deterministic synthetic known answer`
- Execution legs: `{len(results.execution_shortfall)}`
- Round trips: `{len(results.round_trip_pnl)}`
- Daily observations: `{len(results.daily_performance)}`
- Risk metrics available: `{str(risk['risk_metrics_available']).lower()}`
- Strategy net P&L: `{strategy['net_pnl']}`
- Calibration net P&L: `{calibration['net_pnl']}`
- Calibration P&L counted as alpha: `false`
- Campaign slots: `{len(results.campaign_schedule)}`
- Campaign live authorization granted: `false`

## Execution methodology

Every execution decomposes decision-to-fill shortfall into delay, quoted
half-spread, and residual fill-versus-touch components using one common
decision-price denominator. All eight synthetic legs reconcile to the exact
additive identity. Long and short round trips use observed fill prices once;
shortfall is attribution and is not double-counted as a second P&L deduction.

## Profitability and risk interpretation

Only `strategy_signal` records feed the strategy P&L and risk summary.
`calibration_probe` records are deliberately excluded from alpha claims. The
synthetic strategy P&L of `{strategy['net_pnl']}` validates arithmetic only; it
is not realized paper performance. The 25-day fixture exercises rolling
volatility, historical VaR/ES, drawdown, exposure, turnover, and Beta-to-SPY.

## Prospective campaign

Ten calibration slots are predeclared at 10:15 and 14:15 New York time across
five XNYS sessions with alternating buy/sell entry sides and 0.01-share size.
Every slot remains `planned_not_authorized`. Day 21 authorization cannot be
reused. A separate exact Day 22 authorization and live activation-date freeze
are required before any campaign order.

## Safety and limitations

- Broker network accessed: `false`
- Credentials accessed or persisted: `false`
- Orders submitted, canceled, or replaced: `false`
- Canonical market data accessed: `false`
- Locked 2026 research data accessed: `false`
- Realized execution or profitability claimed: `false`

Paper fills may not reproduce live-market queue position, market impact, or
adverse selection. Even after a live campaign, ten probes are a small execution
sample and must not be presented as proof of profitable alpha.
"""
    return {
        EXECUTION_FILENAME: _csv_bytes(
            results.execution_shortfall, EXECUTION_SHORTFALL_COLUMNS
        ),
        ROUND_TRIP_FILENAME: _csv_bytes(results.round_trip_pnl, ROUND_TRIP_COLUMNS),
        DAILY_FILENAME: _csv_bytes(results.daily_performance, DAILY_PERFORMANCE_COLUMNS),
        RISK_FILENAME: _csv_bytes(results.risk_summary, RISK_SUMMARY_COLUMNS),
        SCHEDULE_FILENAME: _csv_bytes(
            results.campaign_schedule, CAMPAIGN_SCHEDULE_COLUMNS
        ),
        CAMPAIGN_FILENAME: _csv_bytes(
            results.campaign_summary, CAMPAIGN_SUMMARY_COLUMNS
        ),
        REPORT_FILENAME: report.encode("utf-8"),
    }


def write_day22_execution_artifacts(
    results: Day22AnalysisResults,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Atomically write the exact Day 22 deterministic evidence bundle."""

    if not isinstance(results, Day22AnalysisResults):
        raise TypeError("results must be Day22AnalysisResults.")
    if not results.evaluation_complete:
        raise ValueError("Day 22 evaluation is incomplete.")
    counts = (
        len(results.execution_shortfall),
        len(results.round_trip_pnl),
        len(results.daily_performance),
        len(results.risk_summary),
        len(results.campaign_schedule),
        len(results.campaign_summary),
    )
    if counts != EXPECTED_ROW_COUNTS:
        raise ValueError("Day 22 synthetic row counts changed.")
    if any(row["authorization_granted"] for row in results.campaign_schedule):
        raise ValueError("Day 22 synthetic campaign unexpectedly has authorization.")
    if results.campaign_summary[0]["purpose"] != "strategy_signal" or not results.campaign_summary[0]["alpha_eligible"]:
        raise ValueError("Strategy evidence label changed.")
    if results.campaign_summary[1]["purpose"] != "calibration_probe" or results.campaign_summary[1]["alpha_eligible"]:
        raise ValueError("Calibration evidence label changed.")

    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    payloads = _payloads(results)
    manifest = {
        "schema_version": "day22_execution_live_performance_artifacts_v1",
        "artifact_order": list(APPROVED_DAY22_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {
            name: hashlib.sha256(payloads[name]).hexdigest()
            for name in APPROVED_DAY22_ARTIFACT_NAMES
            if name != MANIFEST_FILENAME
        },
        "row_counts": {
            EXECUTION_FILENAME: counts[0],
            ROUND_TRIP_FILENAME: counts[1],
            DAILY_FILENAME: counts[2],
            RISK_FILENAME: counts[3],
            SCHEDULE_FILENAME: counts[4],
            CAMPAIGN_FILENAME: counts[5],
        },
        "evidence_separation": {
            "strategy_signal_alpha_eligible": True,
            "calibration_probe_alpha_eligible": False,
        },
        "safety": {
            "synthetic_only": True,
            "broker_network_accessed": False,
            "credentials_accessed": False,
            "orders_submitted": False,
            "orders_canceled_or_replaced": False,
            "canonical_market_data_accessed": False,
            "locked_2026_data_accessed": False,
            "live_campaign_authorized": False,
        },
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    forbidden = (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"APCA-API-KEY-ID",
    )
    for name, payload in payloads.items():
        if any(marker in payload for marker in forbidden):
            raise ValueError(f"Credential-like content detected in {name}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        for name in APPROVED_DAY22_ARTIFACT_NAMES:
            (stage / name).write_bytes(payloads[name])
        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent)
            )
            backup.rmdir()
            os.replace(output, backup)
        os.replace(stage, output)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if {path.name for path in output.iterdir()} != set(APPROVED_DAY22_ARTIFACT_NAMES):
        raise RuntimeError("Day 22 final artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_DAY22_ARTIFACT_NAMES)

