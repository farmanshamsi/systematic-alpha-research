from datetime import timedelta
from pathlib import Path

import pytest

import scripts.run_day22_live_campaign as runner
from systematic_alpha.broker.day22_calibration_campaign import frozen_campaign_slots
from systematic_alpha.broker.day22_campaign_report import (
    load_reconciled_execution_records,
    validate_campaign_state,
)
from tests.broker.test_day22_calibration_campaign import FakeCampaignBroker, NOW


def test_runner_refuses_absent_exact_authorization_flag(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="authorized-paper-campaign"):
        runner.main(["--artifact-directory", str(tmp_path / "campaign")])


def test_no_due_slot_initializes_without_broker_access(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    first_slot = frozen_campaign_slots()[0]
    result = runner.execute_due_slot(
        artifact_directory=output,
        observed_at=first_slot.scheduled_at - timedelta(minutes=1),
    )
    assert result["outcome"] == "no_campaign_slot_due"
    assert not result["order_submission_occurred"]
    assert result["remaining_slots"] == 10


def test_due_slot_runs_once_and_persists_reconciled_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "campaign"
    broker = FakeCampaignBroker()
    monkeypatch.setattr(runner, "verify_day20_prerequisite", lambda _: True)
    result = runner.execute_due_slot(
        artifact_directory=output,
        observed_at=NOW,
        broker=broker,
        now=lambda: NOW,
        sleep=lambda _: None,
    )
    assert result["outcome"] == "paper_calibration_round_trip_reconciled"
    assert result["order_submission_occurred"]
    assert result["flatten_submission_occurred"]
    assert result["shutdown_reconciled"]
    assert len(broker.submissions) == 2
    state = validate_campaign_state(output)
    assert state["slots"][0]["status"] == "completed_reconciled"
    assert not state["manual_recovery_required"]
    records = load_reconciled_execution_records(output)
    assert len(records) == 2
    assert [record.leg for record in records] == ["entry", "exit"]
    assert all(record.purpose == "calibration_probe" for record in records)

    second = runner.execute_due_slot(
        artifact_directory=output,
        observed_at=NOW,
        broker=broker,
        now=lambda: NOW,
        sleep=lambda _: None,
    )
    assert second["outcome"] == "no_campaign_slot_due"
    assert len(broker.submissions) == 2


def test_preflight_is_read_only(tmp_path: Path, monkeypatch) -> None:
    broker = FakeCampaignBroker()
    monkeypatch.setattr(runner, "AlpacaDay22CampaignBroker", lambda: broker)
    monkeypatch.setattr(runner, "verify_day20_prerequisite", lambda _: True)
    result = runner.run_read_only_preflight(tmp_path / "campaign")
    assert result["paper_preflight_passed"]
    assert result["spy_position_is_flat"]
    assert result["open_spy_orders"] == 0
    assert result["day20_prerequisite_passed"]
    assert not result["orders_submitted"]
    assert not result["campaign_slot_consumed"]
    assert not broker.submissions


def test_ambiguous_submit_failure_leaves_recovery_latch(
    tmp_path: Path, monkeypatch
) -> None:
    class SubmitFailureBroker(FakeCampaignBroker):
        def submit_market_order(self, **kwargs):
            raise RuntimeError("simulated ambiguous submit failure")

    output = tmp_path / "campaign"
    monkeypatch.setattr(runner, "verify_day20_prerequisite", lambda _: True)
    with pytest.raises(RuntimeError, match="ambiguous"):
        runner.execute_due_slot(
            artifact_directory=output,
            observed_at=NOW,
            broker=SubmitFailureBroker(),
            now=lambda: NOW,
            sleep=lambda _: None,
        )
    state = validate_campaign_state(output)
    assert state["manual_recovery_required"]
    assert state["slots"][0]["status"] == "in_progress"
