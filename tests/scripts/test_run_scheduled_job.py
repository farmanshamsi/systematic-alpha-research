from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_scheduled_job as runner


def test_health_smoke_is_non_order_capable(monkeypatch) -> None:
    health = tuple(SimpleNamespace(passed=True) for _ in range(14))
    monkeypatch.setattr(
        runner,
        "execute_day23",
        lambda **kwargs: (SimpleNamespace(health_checks=health), ()),
    )
    result = runner.run_scheduled_job("health-smoke")
    assert result == {
        "job_id": "health-smoke",
        "outcome": "passed",
        "health_checks_passed": 14,
        "artifact_files_written": 0,
        "order_submission_occurred": False,
    }


@pytest.mark.parametrize(
    ("job_id", "campaign", "order"),
    (
        ("day22-campaign-once", False, False),
        ("day22-campaign-once", True, True),
        ("day21-strategy-once", False, False),
        ("day21-strategy-once", True, True),
        ("health-smoke", True, False),
    ),
)
def test_scheduled_jobs_fail_closed_without_exact_authorization(
    job_id: str, campaign: bool, order: bool
) -> None:
    with pytest.raises(PermissionError):
        runner.run_scheduled_job(
            job_id,
            authorized_paper_campaign=campaign,
            authorized_paper_order=order,
        )


def test_day22_dispatches_one_existing_due_slot_check(monkeypatch) -> None:
    observed: list[Path] = []

    def fake_execute_due_slot(*, artifact_directory):
        observed.append(Path(artifact_directory))
        return {
            "outcome": "no_campaign_slot_due",
            "order_submission_occurred": False,
        }

    monkeypatch.setattr(runner, "execute_due_slot", fake_execute_due_slot)
    result = runner.run_scheduled_job(
        "day22-campaign-once", authorized_paper_campaign=True
    )
    assert result["outcome"] == "no_campaign_slot_due"
    assert observed == [runner.DAY22_ARTIFACT_DIRECTORY]


def test_day21_dispatches_existing_gated_execution(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_execute_day21(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            outcome="signal_gate_blocked",
            order_submission_occurred=False,
            execution_complete=True,
            manual_recovery_required=False,
        )

    monkeypatch.setattr(runner, "execute_day21", fake_execute_day21)
    result = runner.run_scheduled_job(
        "day21-strategy-once", authorized_paper_order=True
    )
    assert result["order_submission_occurred"] is False
    assert calls[0]["authorized_paper_order"] is True
    assert calls[0]["overwrite"] is False
    assert str(calls[0]["artifact_directory"]).startswith(
        "artifacts/day21/scheduled/"
    )


def test_unknown_job_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown scheduled job"):
        runner.run_scheduled_job("not-a-job")

