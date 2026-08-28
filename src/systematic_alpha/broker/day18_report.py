"""Deterministic redacted artifact bundle for Day 18 paper preflight."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from systematic_alpha.broker.paper_boundary import (
    ALPACA_PAPER_BASE_URL,
    CORE_SYMBOLS,
    PreflightResult,
)


PREFLIGHT_SUMMARY_FILENAME: Final[str] = "preflight_summary.json"
ASSET_ELIGIBILITY_FILENAME: Final[str] = "asset_eligibility.csv"
CAPABILITY_MATRIX_FILENAME: Final[str] = "capability_matrix.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY18_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    PREFLIGHT_SUMMARY_FILENAME,
    ASSET_ELIGIBILITY_FILENAME,
    CAPABILITY_MATRIX_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)
ASSET_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "asset_class",
    "status",
    "tradable",
    "shortable",
    "easy_to_borrow",
    "fractionable",
    "long_eligible",
    "short_eligible",
    "fractional_eligible",
    "asset_gate_passed",
)
CAPABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "capability_kind",
    "capability_value",
    "provider_supported",
    "day18_authorized",
    "constraint",
    "evidence_source",
)
FORBIDDEN_OUTPUT_TEXT: Final[tuple[str, ...]] = (
    "alpaca_api_key",
    "alpaca_secret_key",
    "apca-api-key-id",
    "apca-api-secret-key",
    "account_number",
    "buying_power",
    "portfolio_value",
)


@dataclass(frozen=True, slots=True)
class Day18PreflightReport:
    """Frozen in-memory representation of the five-file Day 18 bundle."""

    result: PreflightResult
    summary: Mapping[str, object]
    asset_rows: tuple[Mapping[str, object], ...]
    capability_rows: tuple[Mapping[str, object], ...]
    report: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.result, PreflightResult):
            raise TypeError("result must be a PreflightResult.")
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))
        object.__setattr__(
            self,
            "asset_rows",
            tuple(MappingProxyType(dict(row)) for row in self.asset_rows),
        )
        object.__setattr__(
            self,
            "capability_rows",
            tuple(MappingProxyType(dict(row)) for row in self.capability_rows),
        )
        object.__setattr__(
            self, "manifest", MappingProxyType(dict(self.manifest))
        )


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Day 18 timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _summary(result: PreflightResult) -> dict[str, object]:
    return {
        "schema_version": result.schema_version,
        "run_kind": "read_only_alpaca_paper_preflight",
        "paper_endpoint": result.paper_endpoint,
        "sdk": {
            "name": result.sdk_name,
            "version": result.sdk_version,
        },
        "credentials_loaded": result.credentials_loaded,
        "credential_values_persisted": result.credential_values_persisted,
        "order_submission_enabled": result.order_submission_enabled,
        "order_submission_occurred": result.order_submission_occurred,
        "core_symbols": list(result.core_symbols),
        "call_order": list(result.call_order),
        "account_gate_passed": result.account_gate_passed,
        "clock_gate_passed": result.clock_gate_passed,
        "asset_gate_passed": result.asset_gate_passed,
        "preflight_passed": result.preflight_passed,
        "account": {
            "status": result.account.status,
            "trading_blocked": result.account.trading_blocked,
            "account_blocked": result.account.account_blocked,
            "trade_suspended_by_user": (
                result.account.trade_suspended_by_user
            ),
            "shorting_enabled": result.account.shorting_enabled,
        },
        "clock": {
            "timestamp": _iso_utc(result.clock.timestamp),
            "is_open": result.clock.is_open,
            "next_open": _iso_utc(result.clock.next_open),
            "next_close": _iso_utc(result.clock.next_close),
        },
    }


def _asset_rows(result: PreflightResult) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "symbol": asset.symbol,
            "asset_class": asset.asset_class,
            "status": asset.status,
            "tradable": asset.tradable,
            "shortable": asset.shortable,
            "easy_to_borrow": asset.easy_to_borrow,
            "fractionable": asset.fractionable,
            "long_eligible": asset.long_eligible,
            "short_eligible": asset.short_eligible,
            "fractional_eligible": asset.fractional_eligible,
            "asset_gate_passed": asset.gate_passed,
        }
        for asset in result.assets
    )


def _capability_rows(
    result: PreflightResult,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "capability_kind": row.capability_kind,
            "capability_value": row.capability_value,
            "provider_supported": row.provider_supported,
            "day18_authorized": row.day18_authorized,
            "constraint": row.constraint,
            "evidence_source": row.evidence_source,
        }
        for row in result.capabilities
    )


def _markdown(result: PreflightResult) -> str:
    outcome = "PASS" if result.preflight_passed else "FAIL"
    asset_lines = "\n".join(
        (
            f"- {asset.symbol}: long={_bool(asset.long_eligible)}, "
            f"short={_bool(asset.short_eligible)}, "
            f"fractional={_bool(asset.fractional_eligible)}"
        )
        for asset in result.assets
    )
    return f"""# Day 18 Alpaca Paper-Broker Preflight

## Outcome

- Preflight: **{outcome}**
- Endpoint: `{result.paper_endpoint}`
- Broker SDK: `{result.sdk_name} {result.sdk_version}`
- Broker timestamp: `{_iso_utc(result.clock.timestamp)}`
- Market open at snapshot: `{_bool(result.clock.is_open)}`
- Credentials loaded: `{_bool(result.credentials_loaded)}`
- Credential values persisted: `false`
- Order submission enabled: `false`
- Order submission occurred: `false`

## Mechanical gates

- Account gate: `{_bool(result.account_gate_passed)}`
- Clock gate: `{_bool(result.clock_gate_passed)}`
- Asset gate: `{_bool(result.asset_gate_passed)}`

## Core-symbol eligibility

{asset_lines}

## Interpretation

This snapshot verifies a read-only connection to the frozen Alpaca paper
endpoint and checks the current account, clock, and core-symbol state. It does
not authorize an order, validate fill handling, or demonstrate profitable
paper execution. All provider order capabilities remain unauthorized on Day
18. Day 19 is limited to a synthetic order-state machine unless a later frozen
contract explicitly changes that boundary.

## Redaction and limitations

No credential value, account identifier, account number, balance, cash, buying
power, equity, or portfolio value is included. Broker state and asset
eligibility can change after this snapshot and must be rechecked before any
future paper-order session.
"""


def build_day18_preflight_report(
    result: PreflightResult,
) -> Day18PreflightReport:
    """Build one redacted report from a normalized preflight result."""

    if not isinstance(result, PreflightResult):
        raise TypeError("result must be a PreflightResult.")
    if result.paper_endpoint != ALPACA_PAPER_BASE_URL:
        raise ValueError("Day 18 result is not bound to the paper endpoint.")
    if result.core_symbols != CORE_SYMBOLS:
        raise ValueError("Day 18 core-symbol order changed.")
    if result.credential_values_persisted:
        raise ValueError("Credential values cannot be persisted.")
    if result.order_submission_enabled or result.order_submission_occurred:
        raise ValueError("Order submission is prohibited on Day 18.")
    if tuple(asset.symbol for asset in result.assets) != CORE_SYMBOLS:
        raise ValueError("Day 18 asset rows changed order.")
    if any(row.day18_authorized for row in result.capabilities):
        raise ValueError("Day 18 cannot authorize an order capability.")

    summary = _summary(result)
    assets = _asset_rows(result)
    capabilities = _capability_rows(result)
    report = _markdown(result)
    manifest = {
        "schema_version": "day18_alpaca_paper_artifacts_v1",
        "artifact_order": list(APPROVED_DAY18_ARTIFACT_NAMES),
        "source": "read_only_alpaca_paper_preflight",
        "paper_endpoint": result.paper_endpoint,
        "core_symbols": list(result.core_symbols),
        "preflight_passed": result.preflight_passed,
        "safety": {
            "paper_only": True,
            "credential_values_persisted": False,
            "account_identifiers_persisted": False,
            "financial_balances_persisted": False,
            "order_submission_enabled": False,
            "order_submission_occurred": False,
            "locked_2026_data_accessed": False,
        },
    }
    return Day18PreflightReport(
        result=result,
        summary=summary,
        asset_rows=assets,
        capability_rows=capabilities,
        report=report,
        manifest=manifest,
    )


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
            raise ValueError("Day 18 CSV schema or column order changed.")
        writer.writerow(
            {
                key: _bool(value) if type(value) is bool else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue().encode("utf-8")


def _payloads(report: Day18PreflightReport) -> dict[str, bytes]:
    return {
        PREFLIGHT_SUMMARY_FILENAME: _json_bytes(report.summary),
        ASSET_ELIGIBILITY_FILENAME: _csv_bytes(
            report.asset_rows, ASSET_COLUMNS
        ),
        CAPABILITY_MATRIX_FILENAME: _csv_bytes(
            report.capability_rows, CAPABILITY_COLUMNS
        ),
        REPORT_FILENAME: report.report.encode("utf-8"),
    }


def _verify_redaction(payloads: Mapping[str, bytes]) -> None:
    combined = b"\n".join(payloads.values()).decode("utf-8").lower()
    for forbidden in FORBIDDEN_OUTPUT_TEXT:
        if forbidden in combined:
            raise ValueError(
                "Day 18 output contains a forbidden sensitive field."
            )


def _manifest_bytes(
    report: Day18PreflightReport,
    payloads: Mapping[str, bytes],
) -> bytes:
    manifest = dict(report.manifest)
    manifest["artifacts"] = [
        {
            "filename": filename,
            "bytes": len(payloads[filename]),
            "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in APPROVED_DAY18_ARTIFACT_NAMES
        if filename != MANIFEST_FILENAME
    ]
    return _json_bytes(manifest)


def write_day18_preflight_artifacts(
    report: Day18PreflightReport,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly five artifacts using sibling staging and replacement."""

    if not isinstance(report, Day18PreflightReport):
        raise TypeError("report must be a Day18PreflightReport.")
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
        _verify_redaction(payloads)
        payloads[MANIFEST_FILENAME] = _manifest_bytes(report, payloads)
        if tuple(payloads) != APPROVED_DAY18_ARTIFACT_NAMES:
            raise RuntimeError("Day 18 payload allow-list mismatch.")
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
    if names != tuple(sorted(APPROVED_DAY18_ARTIFACT_NAMES)):
        raise RuntimeError("Final Day 18 artifact allow-list verification failed.")
    return tuple(destination / name for name in APPROVED_DAY18_ARTIFACT_NAMES)
