"""Credential-free evidence bundle for Day 21 controlled paper execution."""

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

from systematic_alpha.broker.controlled_paper_execution import (
    DAY21_NOTIONAL_CAP,
    DAY21_QUANTITY,
    DAY21_SCHEMA_VERSION,
    GATE_ORDER,
    Day21ExecutionResult,
)


PROTOCOL_FILENAME: Final[str] = "protocol.json"
GATES_FILENAME: Final[str] = "gate_results.csv"
SIGNAL_FILENAME: Final[str] = "signal_snapshot.csv"
ORDERS_FILENAME: Final[str] = "order_events.csv"
FILLS_FILENAME: Final[str] = "fill_summary.csv"
POSITIONS_FILENAME: Final[str] = "position_cash_snapshots.csv"
RECONCILIATION_FILENAME: Final[str] = "reconciliation.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY21_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    PROTOCOL_FILENAME,
    GATES_FILENAME,
    SIGNAL_FILENAME,
    ORDERS_FILENAME,
    FILLS_FILENAME,
    POSITIONS_FILENAME,
    RECONCILIATION_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)

GATE_COLUMNS: Final[tuple[str, ...]] = ("gate_order", "gate_id", "passed", "safe_detail")
SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_id",
    "symbol",
    "computed_at_utc",
    "bar_start_utc",
    "bar_end_utc",
    "last_close",
    "position",
    "raw_signal",
    "signal_available",
    "signal_fresh",
    "signal_age_seconds",
    "regime_eligible",
    "ou_zscore",
    "ou_half_life_bars",
    "variance_ratio",
    "operational_rows",
    "operational_sessions",
    "data_start_utc",
    "data_end_utc",
    "locked_research_data_accessed",
)
ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "event_sequence",
    "phase",
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
    "submitted_at_utc",
    "filled_at_utc",
)
FILL_COLUMNS: Final[tuple[str, ...]] = (
    "phase",
    "client_order_id",
    "broker_order_id",
    "side",
    "filled_quantity",
    "filled_average_price",
    "terminal_status",
)
POSITION_COLUMNS: Final[tuple[str, ...]] = (
    "phase",
    "observed_at_utc",
    "spy_quantity",
    "cash",
)


def _iso(value: object | None) -> str:
    return "" if value is None else value.isoformat()  # type: ignore[union-attr]


def _text(value: object | None) -> str:
    return "" if value is None else str(value)


def _csv_bytes(rows: tuple[Mapping[str, object], ...], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 21 artifact schema changed.")
        writer.writerow(
            {
                key: (
                    "true" if value is True else "false" if value is False else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _tables(result: Day21ExecutionResult) -> dict[str, bytes]:
    if tuple(item.gate_id for item in result.gates) != GATE_ORDER:
        raise ValueError("Day 21 gate order changed.")
    signal = result.signal
    gates = tuple(
        {
            "gate_order": index,
            "gate_id": item.gate_id,
            "passed": item.passed,
            "safe_detail": item.safe_detail,
        }
        for index, item in enumerate(result.gates, start=1)
    )
    signal_rows = (
        {
            "candidate_id": signal.candidate_id,
            "symbol": signal.symbol,
            "computed_at_utc": _iso(signal.computed_at),
            "bar_start_utc": _iso(signal.bar_start),
            "bar_end_utc": _iso(signal.bar_end),
            "last_close": signal.last_close,
            "position": signal.position,
            "raw_signal": signal.raw_signal,
            "signal_available": signal.signal_available,
            "signal_fresh": signal.signal_fresh,
            "signal_age_seconds": signal.signal_age_seconds,
            "regime_eligible": signal.regime_eligible,
            "ou_zscore": _text(signal.ou_zscore),
            "ou_half_life_bars": _text(signal.ou_half_life_bars),
            "variance_ratio": _text(signal.variance_ratio),
            "operational_rows": signal.operational_rows,
            "operational_sessions": signal.operational_sessions,
            "data_start_utc": _iso(signal.data_start),
            "data_end_utc": _iso(signal.data_end),
            "locked_research_data_accessed": signal.locked_research_data_accessed,
        },
    )
    orders = tuple(
        {
            "event_sequence": index,
            "phase": (
                "entry"
                if item.client_order_id == result.entry_client_order_id
                else "flatten"
            ),
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
            "submitted_at_utc": _iso(item.submitted_at),
            "filled_at_utc": _iso(item.filled_at),
        }
        for index, item in enumerate(result.order_events, start=1)
    )
    fills_list: list[dict[str, object]] = []
    for phase, client_id in (
        ("entry", result.entry_client_order_id),
        ("flatten", result.flatten_client_order_id),
    ):
        phase_orders = tuple(
            item for item in result.order_events if item.client_order_id == client_id
        )
        if phase_orders:
            item = phase_orders[-1]
            fills_list.append(
                {
                    "phase": phase,
                    "client_order_id": item.client_order_id,
                    "broker_order_id": item.broker_order_id,
                    "side": item.side,
                    "filled_quantity": item.filled_quantity,
                    "filled_average_price": _text(item.filled_average_price),
                    "terminal_status": item.status,
                }
            )
    fills = tuple(fills_list)
    positions = tuple(
        {
            "phase": item.phase,
            "observed_at_utc": _iso(item.observed_at),
            "spy_quantity": item.spy_quantity,
            "cash": item.cash,
        }
        for item in result.position_cash
    )
    protocol = {
        "schema_version": DAY21_SCHEMA_VERSION,
        "candidate_id": "ou_vwap_slow",
        "candidate_role": "operational_probe_candidate",
        "symbol": "SPY",
        "paper_endpoint": "https://paper-api.alpaca.markets",
        "quantity": str(DAY21_QUANTITY),
        "notional_cap_usd": str(DAY21_NOTIONAL_CAP),
        "order_type": "market",
        "time_in_force": "day",
        "extended_hours": False,
        "same_run_flatten_required": True,
        "real_money_trading_authorized": False,
        "profitability_promotion_claimed": False,
    }
    reconciliation = {
        "schema_version": DAY21_SCHEMA_VERSION,
        "outcome": result.outcome,
        "order_submission_occurred": result.order_submission_occurred,
        "entry_filled_quantity": str(result.entry_filled_quantity),
        "flatten_filled_quantity": str(result.flatten_filled_quantity),
        "realized_round_trip_pnl": _text(result.realized_round_trip_pnl),
        "execution_complete": result.execution_complete,
        "shutdown_reconciled": result.shutdown_reconciled,
        "manual_recovery_required": result.manual_recovery_required,
        "abort_reasons": list(result.abort_reasons),
        "credentials_persisted": False,
        "locked_research_data_accessed": False,
        "real_money_endpoint_accessed": False,
    }
    report = f"""# Day 21 Controlled Alpaca Paper Execution

## Outcome

- Outcome: `{result.outcome}`
- Order submission occurred: `{str(result.order_submission_occurred).lower()}`
- Entry filled quantity: `{result.entry_filled_quantity}`
- Flatten filled quantity: `{result.flatten_filled_quantity}`
- Shutdown reconciled: `{str(result.shutdown_reconciled).lower()}`
- Manual recovery required: `{str(result.manual_recovery_required).lower()}`
- Execution complete: `{str(result.execution_complete).lower()}`
- Abort reasons: `{'|'.join(result.abort_reasons) or 'none'}`

## Candidate interpretation

`ou_vwap_slow` is an operational probe candidate. Day 17's positive development
result was statistically inconclusive. This paper run tests order handling and
implementation shortfall; it does not promote the strategy or prove
profitability.

## Safety

- Alpaca paper endpoint only: `true`
- Exact entry size cap: `0.01 SPY share`
- Same-run flatten required: `true`
- Extended-hours queuing: `false`
- Credential values persisted: `false`
- Locked January-June 2026 research data accessed: `false`
- Real-money trading authorized: `false`
"""
    return {
        PROTOCOL_FILENAME: _json_bytes(protocol),
        GATES_FILENAME: _csv_bytes(gates, GATE_COLUMNS),
        SIGNAL_FILENAME: _csv_bytes(signal_rows, SIGNAL_COLUMNS),
        ORDERS_FILENAME: _csv_bytes(orders, ORDER_COLUMNS),
        FILLS_FILENAME: _csv_bytes(fills, FILL_COLUMNS),
        POSITIONS_FILENAME: _csv_bytes(positions, POSITION_COLUMNS),
        RECONCILIATION_FILENAME: _json_bytes(reconciliation),
        REPORT_FILENAME: report.encode("utf-8"),
    }


def write_day21_artifacts(
    result: Day21ExecutionResult,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Atomically write and hash the exact credential-free Day 21 bundle."""

    if not isinstance(result, Day21ExecutionResult):
        raise TypeError("result must be a Day21ExecutionResult.")
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    payloads = _tables(result)
    manifest = {
        "schema_version": "day21_controlled_paper_artifacts_v1",
        "artifact_order": list(APPROVED_DAY21_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {
            name: hashlib.sha256(payloads[name]).hexdigest()
            for name in APPROVED_DAY21_ARTIFACT_NAMES
            if name != MANIFEST_FILENAME
        },
        "row_counts": {
            GATES_FILENAME: len(result.gates),
            SIGNAL_FILENAME: 1,
            ORDERS_FILENAME: len(result.order_events),
            FILLS_FILENAME: len(
                {
                    item.client_order_id for item in result.order_events
                }
            ),
            POSITIONS_FILENAME: len(result.position_cash),
        },
        "safety": {
            "credentials_persisted": False,
            "locked_research_data_accessed": False,
            "real_money_endpoint_accessed": False,
            "real_money_orders_submitted": False,
        },
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    forbidden = (b"ALPACA_API_KEY=", b"ALPACA_SECRET_KEY=", b"APCA-API-KEY-ID")
    for name, payload in payloads.items():
        if any(marker in payload for marker in forbidden):
            raise ValueError(f"Credential-like content detected in {name}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        for name in APPROVED_DAY21_ARTIFACT_NAMES:
            (stage / name).write_bytes(payloads[name])
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
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
    observed = tuple(path.name for path in sorted(output.iterdir(), key=lambda item: APPROVED_DAY21_ARTIFACT_NAMES.index(item.name)))
    if observed != APPROVED_DAY21_ARTIFACT_NAMES:
        raise RuntimeError("Day 21 artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_DAY21_ARTIFACT_NAMES)
