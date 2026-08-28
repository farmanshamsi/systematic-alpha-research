from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import systematic_alpha.broker.day19_report as report_module
from systematic_alpha.broker.day19_report import (
    APPROVED_DAY19_ARTIFACT_NAMES,
    FINAL_STATES_COLUMNS,
    SCENARIO_SUMMARY_COLUMNS,
    STATE_TRANSITION_MATRIX_COLUMNS,
    TIMEOUT_DIAGNOSTICS_COLUMNS,
    TRANSITION_LOG_COLUMNS,
    Day19OrderStateReport,
    build_day19_order_state_report,
    write_day19_order_state_artifacts,
)
from systematic_alpha.broker.day19_scenarios import (
    SCENARIO_ORDER,
    run_day19_scenarios,
)
from systematic_alpha.broker.order_state import REASON_CODES, STATUS_ORDER


def _report() -> Day19OrderStateReport:
    return build_day19_order_state_report(run_day19_scenarios())


def test_scenarios_and_report_have_exact_frozen_contract() -> None:
    report = _report()
    results = report.results
    assert results.evaluation_complete is True
    assert tuple(row["scenario_id"] for row in results.scenario_summary) == (
        SCENARIO_ORDER
    )
    assert (
        len(results.scenario_summary),
        len(results.final_states),
        len(results.transition_log),
        len(results.rejection_diagnostics),
        len(results.timeout_diagnostics),
        len(results.state_transition_matrix),
    ) == (9, 8, 32, 3, 1, 361)
    assert tuple(results.scenario_summary[0]) == SCENARIO_SUMMARY_COLUMNS
    assert tuple(results.final_states[0]) == FINAL_STATES_COLUMNS
    assert tuple(results.transition_log[0]) == TRANSITION_LOG_COLUMNS
    assert tuple(results.rejection_diagnostics[0]) == TRANSITION_LOG_COLUMNS
    assert tuple(results.timeout_diagnostics[0]) == TIMEOUT_DIAGNOSTICS_COLUMNS
    assert tuple(results.state_transition_matrix[0]) == (
        STATE_TRANSITION_MATRIX_COLUMNS
    )
    assert all(row["scenario_passed"] for row in results.scenario_summary)


def test_report_manifest_is_synthetic_and_execution_disabled() -> None:
    report = _report()
    assert report.manifest["schema_version"] == (
        "day19_order_state_artifacts_v1"
    )
    assert report.manifest["artifact_order"] == list(
        APPROVED_DAY19_ARTIFACT_NAMES
    )
    assert report.manifest["status_order"] == [
        status.value for status in STATUS_ORDER
    ]
    assert report.manifest["reason_codes"] == list(REASON_CODES)
    safety = report.manifest["safety"]
    assert isinstance(safety, dict)
    assert safety == {
        "synthetic_only": True,
        "broker_network_accessed": False,
        "credentials_accessed": False,
        "order_submission_enabled": False,
        "order_submission_occurred": False,
        "account_mutation_occurred": False,
        "canonical_market_data_accessed": False,
        "locked_2026_data_accessed": False,
    }
    assert "These are operational controls, not evidence of" in report.report
    assert "profitability" in report.report


@pytest.mark.parametrize(
    "mutation",
    (
        {"evaluation_complete": False},
        {"scenario_summary": ()},
        {"final_states": ()},
        {"transition_log": ()},
        {"rejection_diagnostics": ()},
        {"timeout_diagnostics": ()},
        {"state_transition_matrix": ()},
    ),
)
def test_report_rejects_incomplete_or_wrong_count_results(
    mutation: dict[str, object],
) -> None:
    results = run_day19_scenarios()
    with pytest.raises(ValueError):
        build_day19_order_state_report(replace(results, **mutation))


def test_report_rejects_changed_schema_order() -> None:
    results = run_day19_scenarios()
    first = dict(results.scenario_summary[0])
    changed = {"scenario_id": first.pop("scenario_id"), **first}
    rows = (changed, *results.scenario_summary[1:])
    with pytest.raises(ValueError, match="schema or column order"):
        build_day19_order_state_report(
            replace(results, scenario_summary=rows)
        )


def test_writer_emits_exact_allow_list_and_valid_hashes(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "day19"
    paths = write_day19_order_state_artifacts(report, destination)
    assert tuple(path.name for path in paths) == APPROVED_DAY19_ARTIFACT_NAMES
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY19_ARTIFACT_NAMES)
    )

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["artifact_order"] == list(APPROVED_DAY19_ARTIFACT_NAMES)
    assert manifest["row_counts"] == {
        "scenario_summary": 9,
        "final_states": 8,
        "transition_log": 32,
        "rejection_diagnostics": 3,
        "timeout_diagnostics": 1,
        "state_transition_matrix": 361,
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
    write_day19_order_state_artifacts(report, first)
    write_day19_order_state_artifacts(report, second)
    for filename in APPROVED_DAY19_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_writer_refuses_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "day19"
    report = _report()
    write_day19_order_state_artifacts(report, destination)
    with pytest.raises(FileExistsError):
        write_day19_order_state_artifacts(report, destination)


def test_overwrite_removes_unapproved_files(tmp_path: Path) -> None:
    destination = tmp_path / "day19"
    report = _report()
    write_day19_order_state_artifacts(report, destination)
    (destination / "unexpected.txt").write_text("remove on replacement")
    write_day19_order_state_artifacts(report, destination, overwrite=True)
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(APPROVED_DAY19_ARTIFACT_NAMES)
    )


def test_overwrite_rolls_back_when_replacement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "day19"
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
        write_day19_order_state_artifacts(
            _report(), destination, overwrite=True
        )
    assert sentinel.read_text() == "original"
    assert tuple(path.name for path in destination.iterdir()) == (
        "sentinel.txt",
    )


def test_writer_public_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_day19_order_state_artifacts(
            object(), tmp_path / "bad"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        write_day19_order_state_artifacts(
            _report(), tmp_path / "bad", overwrite=1  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        build_day19_order_state_report(object())  # type: ignore[arg-type]


def test_bundle_contains_no_credential_or_live_endpoint_markers(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "day19"
    write_day19_order_state_artifacts(_report(), destination)
    combined = b"\n".join(path.read_bytes() for path in destination.iterdir())
    for forbidden in (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"api.alpaca.markets/v2/orders",
    ):
        assert forbidden not in combined
