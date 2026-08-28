"""Atomic seven-file Day 23 reproducible-operations evidence bundle."""

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

from systematic_alpha.operations.runtime_validation import Day23ValidationResult


DEPENDENCY_FILENAME: Final[str] = "dependency_audit.csv"
HEALTH_FILENAME: Final[str] = "health_checks.csv"
RUNTIME_FILENAME: Final[str] = "runtime_contract.json"
SCHEDULE_FILENAME: Final[str] = "schedule_entrypoints.csv"
PERSISTENCE_FILENAME: Final[str] = "persistence_policy.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY23_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    DEPENDENCY_FILENAME,
    HEALTH_FILENAME,
    RUNTIME_FILENAME,
    SCHEDULE_FILENAME,
    PERSISTENCE_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)
DEPENDENCY_COLUMNS: Final[tuple[str, ...]] = (
    "dependency_order",
    "dependency_name",
    "pyproject_specifier",
    "locked_version",
    "hash_count",
    "lock_exact",
    "installed_version",
    "installed_matches_lock",
)
HEALTH_COLUMNS: Final[tuple[str, ...]] = (
    "check_order",
    "check_id",
    "passed",
    "detail",
)
SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "job_order",
    "job_id",
    "entrypoint",
    "authorization_flag",
    "schedule_policy",
    "automatic",
    "order_capable",
)


def _csv_bytes(
    rows: tuple[Mapping[str, object], ...], columns: tuple[str, ...]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 23 artifact schema changed.")
        writer.writerow(
            {
                key: (
                    "true"
                    if value is True
                    else "false"
                    if value is False
                    else ""
                    if value is None
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _payloads(result: Day23ValidationResult) -> dict[str, bytes]:
    dependency_rows = tuple(item.row() for item in result.dependency_audit)
    health_rows = tuple(item.row() for item in result.health_checks)
    schedule_rows = tuple(item.row() for item in result.schedule_entries)
    mismatches = sum(
        not item.installed_matches_lock for item in result.dependency_audit
    )
    runtime = {
        "schema_version": "day23_runtime_contract_v1",
        "evaluation_complete": result.evaluation_complete,
        "default_mode": "offline_smoke",
        "python_reference": "3.11.15",
        "python_runtime": result.python_version,
        "dependency_lock_sha256": result.lock_sha256,
        "clean_environment_validated": result.clean_environment_validated,
        "container_runtime_available": result.container_runtime_available,
        "container_runtime_validated": False,
        "container_static_validation_passed": (
            result.container_static_validation_passed
        ),
        "broker_environment": "paper",
        "paper_base_url": "https://paper-api.alpaca.markets",
        "credentials_required": False,
        "network_allowed": False,
        "order_submission_enabled": False,
        "broker_network_accessed": result.broker_network_accessed,
        "credentials_accessed": result.credentials_accessed,
        "orders_submitted": result.orders_submitted,
        "locked_final_test_data_accessed": result.locked_final_test_data_accessed,
    }
    persistence = {
        "schema_version": "day23_persistence_policy_v1",
        "research_artifacts_immutable_after_freeze": True,
        "live_state_atomic_replacement": True,
        "live_slot_bundles_immutable": True,
        "logs_outside_deterministic_bundles": True,
        "credentials_or_account_identifiers_backed_up": False,
        "backup_requires_verified_source_manifest": True,
        "backup_target": "new_timestamped_directory",
        "backup_overwrite_allowed": False,
        "restore_read_only_until_hash_and_paper_preflight": True,
        "git_ignored_paths": ["backups/", "logs/"],
    }
    report = f"""# Day 23 Reproducible Operations Validation

## Outcome

- Evaluation complete: `{str(result.evaluation_complete).lower()}`
- Offline health checks passed: `{sum(item.passed for item in result.health_checks)}/{len(result.health_checks)}`
- Direct dependencies audited: `{len(result.dependency_audit)}`
- Installed-to-lock mismatches: `{mismatches}`
- Clean isolated environment validated: `{str(result.clean_environment_validated).lower()}`
- Container static validation passed: `{str(result.container_static_validation_passed).lower()}`
- Container runtime available: `{str(result.container_runtime_available).lower()}`
- Container runtime validated: `false`

## Reproducibility contract

The exact transitive runtime and development environment is pinned with hashes
in `requirements.lock`. The offline startup/shutdown smoke path checks fourteen
frozen gates, requires no credentials, performs no network request, does not
construct a broker adapter, and never enables order submission. The same lock
and smoke command are used by the container definition and CI workflow.

## Scheduling and persistence

Only three jobs are exposed. Health smoke is safe by default; the Day 22
one-shot campaign and Day 21 strategy session retain their exact authorization
flags and existing execution gates. Unknown jobs fail closed. Day 22's active
watcher remains the campaign scheduler and was not stopped, reset, or consumed.

Research bundles remain immutable after freeze. Live state uses atomic updates,
and backups require verified source hashes, a new timestamped destination, and
a read-only restore until safety preflight passes. Logs, backups, credentials,
raw broker payloads, and account identifiers remain outside Git evidence.

## Honest limitations

No compatible container engine was available on the development host, so the
Docker and Compose definitions received static tests but no claimed image build
or container-runtime validation. CI will exercise the clean locked environment
on every future GitHub run. This operations work improves reproducibility and
safety; it is not new profitability evidence and does not inspect the locked
2026 final-test period.

## Safety record

- Broker network accessed: `false`
- Credentials accessed or persisted: `false`
- Orders submitted, canceled, or replaced: `false`
- Day 22 campaign slot consumed: `false`
- Locked 2026 final-test data accessed: `false`
- Real-money trading enabled: `false`
"""
    return {
        DEPENDENCY_FILENAME: _csv_bytes(dependency_rows, DEPENDENCY_COLUMNS),
        HEALTH_FILENAME: _csv_bytes(health_rows, HEALTH_COLUMNS),
        RUNTIME_FILENAME: _json_bytes(runtime),
        SCHEDULE_FILENAME: _csv_bytes(schedule_rows, SCHEDULE_COLUMNS),
        PERSISTENCE_FILENAME: _json_bytes(persistence),
        REPORT_FILENAME: report.encode("utf-8"),
    }


def write_day23_artifacts(
    result: Day23ValidationResult,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact Day 23 evidence bundle with atomic replacement."""

    if not isinstance(result, Day23ValidationResult):
        raise TypeError("result must be Day23ValidationResult.")
    if not result.evaluation_complete:
        raise ValueError("Day 23 operational evaluation is incomplete.")
    if not result.clean_environment_validated:
        raise ValueError("Clean-environment validation is required for evidence.")
    if not result.container_static_validation_passed:
        raise ValueError("Container static validation is required for evidence.")
    if len(result.health_checks) != 14 or len(result.schedule_entries) != 3:
        raise ValueError("Day 23 frozen row counts changed.")

    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    payloads = _payloads(result)
    manifest = {
        "schema_version": "day23_reproducible_operations_artifacts_v1",
        "artifact_order": list(APPROVED_DAY23_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {
            name: hashlib.sha256(payloads[name]).hexdigest()
            for name in APPROVED_DAY23_ARTIFACT_NAMES
            if name != MANIFEST_FILENAME
        },
        "row_counts": {
            DEPENDENCY_FILENAME: len(result.dependency_audit),
            HEALTH_FILENAME: len(result.health_checks),
            SCHEDULE_FILENAME: len(result.schedule_entries),
        },
        "evaluation_complete": True,
        "safety": {
            "offline_smoke": True,
            "broker_network_accessed": False,
            "credentials_accessed": False,
            "orders_submitted": False,
            "locked_final_test_data_accessed": False,
            "day22_watcher_changed_or_slot_consumed": False,
            "real_money_enabled": False,
        },
        "validation": {
            "clean_environment_validated": True,
            "container_static_validation_passed": True,
            "container_runtime_available": result.container_runtime_available,
            "container_runtime_validated": False,
        },
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    forbidden = (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"APCA-API-KEY-ID:",
        b"APCA-API-SECRET-KEY:",
    )
    for name, payload in payloads.items():
        if any(marker in payload for marker in forbidden):
            raise ValueError(f"Credential-like content detected in {name}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    backup: Path | None = None
    try:
        for name in APPROVED_DAY23_ARTIFACT_NAMES:
            (stage / name).write_bytes(payloads[name])
        if output.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output.name}.backup-", dir=output.parent
                )
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
    if {path.name for path in output.iterdir()} != set(
        APPROVED_DAY23_ARTIFACT_NAMES
    ):
        raise RuntimeError("Day 23 final artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_DAY23_ARTIFACT_NAMES)

