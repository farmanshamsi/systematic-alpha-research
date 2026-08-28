from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import systematic_alpha.broker.day20_report as report_module
from systematic_alpha.broker.day20_report import (
    APPROVED_DAY20_ARTIFACT_NAMES,
    DAY20_REASON_CODES,
    EXPECTED_ROW_COUNTS,
    Day20ReconciliationReport,
    build_day20_reconciliation_report,
    write_day20_reconciliation_artifacts,
)
from systematic_alpha.broker.day20_scenarios import (
    OPERATIONAL_DECISION_COLUMNS,
    POSITION_CASH_COLUMNS,
    RECONCILIATION_DIAGNOSTIC_COLUMNS,
    RECONCILIATION_SUMMARY_COLUMNS,
    SCENARIO_ORDER,
    SCENARIO_SUMMARY_COLUMNS,
    STREAM_LOG_COLUMNS,
    run_day20_scenarios,
)


def _report() -> Day20ReconciliationReport:
    return build_day20_reconciliation_report(run_day20_scenarios())


def test_scenarios_and_report_have_exact_frozen_contract() -> None:
    report = _report()
    results = report.results
    assert results.evaluation_complete is True
    assert tuple(row["scenario_id"] for row in results.scenario_summary) == (
        SCENARIO_ORDER
    )
    assert (
        len(results.scenario_summary),
        len(results.reconciliation_summary),
        len(results.reconciliation_diagnostics),
        len(results.position_cash_reconciliation),
        len(results.stream_transition_log),
        len(results.operational_decisions),
    ) == EXPECTED_ROW_COUNTS
    assert tuple(results.scenario_summary[0]) == SCENARIO_SUMMARY_COLUMNS
    assert tuple(results.reconciliation_summary[0]) == (
        RECONCILIATION_SUMMARY_COLUMNS
    )
    assert tuple(results.reconciliation_diagnostics[0]) == (
        RECONCILIATION_DIAGNOSTIC_COLUMNS
    )
    assert tuple(results.position_cash_reconciliation[0]) == (
        POSITION_CASH_COLUMNS
    )
    assert tuple(results.stream_transition_log[0]) == STREAM_LOG_COLUMNS
    assert tuple(results.operational_decisions[0]) == (
        OPERATIONAL_DECISION_COLUMNS
    )
    assert all(row["scenario_passed"] for row in results.scenario_summary)


def test_manifest_safety_and_order_authorization_are_frozen() -> None:
    report = _report()
    assert report.manifest["schema_version"] == (
        "day20_reconciliation_monitoring_artifacts_v1"
    )
    assert report.manifest["artifact_order"] == list(
        APPROVED_DAY20_ARTIFACT_NAMES
    )
    assert report.manifest["scenario_order"] == list(SCENARIO_ORDER)
    assert report.manifest["reason_codes"] == list(DAY20_REASON_CODES)
    safety = report.manifest["safety"]
    assert isinstance(safety, dict)
    assert safety["synthetic_only"] is True
    assert not any(
        safety[key]
        for key in (
            "broker_network_accessed",
            "credentials_accessed",
            "order_submission_enabled",
            "order_submission_occurred",
            "order_cancel_or_replace_occurred",
            "account_or_position_mutation_occurred",
            "canonical_market_data_accessed",
            "locked_2026_data_accessed",
        )
    )
    assert not any(
        row["day20_order_submission_authorized"] or row["can_submit_orders"]
        for row in report.results.operational_decisions
    )
    assert "not evidence of execution" in report.report
    assert "profitability" in report.report


@pytest.mark.parametrize(
    "mutation",
    (
        {"evaluation_complete": False},
        {"scenario_summary": ()},
        {"reconciliation_summary": ()},
        {"reconciliation_diagnostics": ()},
        {"position_cash_reconciliation": ()},
        {"stream_transition_log": ()},
        {"operational_decisions": ()},
    ),
)
def test_report_rejects_incomplete_or_wrong_count_results(
    mutation: dict[str, object],
) -> None:
    results = run_day20_scenarios()
    with pytest.raises(ValueError):
        build_day20_reconciliation_report(replace(results, **mutation))


def test_report_rejects_schema_reorder() -> None:
    results = run_day20_scenarios()
    first = dict(results.scenario_summary[0])
    changed = {"scenario_id": first.pop("scenario_id"), **first}
    with pytest.raises(ValueError, match="schema or column order"):
        build_day20_reconciliation_report(
            replace(
                results,
                scenario_summary=(changed, *results.scenario_summary[1:]),
            )
        )


def test_writer_emits_exact_allow_list_and_valid_hashes(tmp_path: Path) -> None:
    destination = tmp_path / "day20"
    paths = write_day20_reconciliation_artifacts(_report(), destination)
    assert tuple(path.name for path in paths) == APPROVED_DAY20_ARTIFACT_NAMES
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY20_ARTIFACT_NAMES)
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["artifact_order"] == list(APPROVED_DAY20_ARTIFACT_NAMES)
    assert manifest["row_counts"] == {
        "scenario_summary": 12,
        "reconciliation_summary": 12,
        "reconciliation_diagnostics": 10,
        "position_cash_reconciliation": 24,
        "stream_transition_log": 23,
        "operational_decisions": 12,
    }
    assert len(manifest["artifacts"]) == 7
    for row in manifest["artifacts"]:
        payload = (destination / row["filename"]).read_bytes()
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_writer_fixed_input_replays_byte_for_byte(tmp_path: Path) -> None:
    report = _report()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_day20_reconciliation_artifacts(report, first)
    write_day20_reconciliation_artifacts(report, second)
    for filename in APPROVED_DAY20_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_writer_refuses_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    report = _report()
    destination = tmp_path / "day20"
    write_day20_reconciliation_artifacts(report, destination)
    with pytest.raises(FileExistsError):
        write_day20_reconciliation_artifacts(report, destination)


def test_overwrite_removes_unapproved_files(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "day20"
    write_day20_reconciliation_artifacts(report, destination)
    (destination / "unexpected.txt").write_text("remove on replacement")
    write_day20_reconciliation_artifacts(report, destination, overwrite=True)
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY20_ARTIFACT_NAMES)
    )


def test_overwrite_rolls_back_when_replacement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "day20"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("original")
    original_replace = report_module.os.replace
    calls = 0

    def fail_second_replace(source, target) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replacement failure")
        original_replace(source, target)

    monkeypatch.setattr(report_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        write_day20_reconciliation_artifacts(
            _report(), destination, overwrite=True
        )
    assert sentinel.read_text() == "original"
    assert tuple(path.name for path in destination.iterdir()) == (
        "sentinel.txt",
    )


def test_report_and_writer_public_types_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_day20_reconciliation_report(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        write_day20_reconciliation_artifacts(
            object(), tmp_path / "bad"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        write_day20_reconciliation_artifacts(
            _report(), tmp_path / "bad", overwrite=1  # type: ignore[arg-type]
        )


def test_bundle_contains_no_credentials_or_live_order_endpoint(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "day20"
    write_day20_reconciliation_artifacts(_report(), destination)
    combined = b"\n".join(path.read_bytes() for path in destination.iterdir())
    for forbidden in (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"api.alpaca.markets/v2/orders",
    ):
        assert forbidden not in combined

