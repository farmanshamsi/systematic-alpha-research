from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import scripts.run_day18_alpaca_paper_preflight as runner
from systematic_alpha.broker.day18_report import APPROVED_DAY18_ARTIFACT_NAMES
from systematic_alpha.broker.paper_boundary import (
    AlpacaPaperBroker,
    PaperBrokerPreflightError,
)
from tests.day18_fixtures import (
    FakeTradingClient,
    safe_day18_config,
)


def _broker() -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(),
    )


def test_execute_day18_writes_complete_bundle(tmp_path: Path) -> None:
    result = runner.execute_day18(
        config=safe_day18_config(),
        artifact_directory=tmp_path / "day18",
        broker=_broker(),
    )
    assert result.evaluation_complete is True
    assert result.preflight.preflight_passed is True
    assert tuple(path.name for path in result.artifact_paths) == (
        APPROVED_DAY18_ARTIFACT_NAMES
    )


def test_execute_day18_call_order_is_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    counts = {"preflight": 0, "report": 0, "writer": 0}
    original_preflight = runner.run_paper_preflight
    original_report = runner.build_day18_preflight_report
    original_writer = runner.write_day18_preflight_artifacts

    def counted_preflight(broker):
        counts["preflight"] += 1
        return original_preflight(broker)

    def counted_report(result):
        counts["report"] += 1
        return original_report(result)

    def counted_writer(report, destination, *, overwrite=False):
        counts["writer"] += 1
        return original_writer(report, destination, overwrite=overwrite)

    monkeypatch.setattr(runner, "run_paper_preflight", counted_preflight)
    monkeypatch.setattr(runner, "build_day18_preflight_report", counted_report)
    monkeypatch.setattr(
        runner, "write_day18_preflight_artifacts", counted_writer
    )

    runner.execute_day18(
        config=safe_day18_config(),
        artifact_directory=tmp_path / "day18",
        broker=_broker(),
    )
    assert counts == {"preflight": 1, "report": 1, "writer": 1}


def test_failed_gate_is_written_but_not_marked_incomplete(tmp_path: Path) -> None:
    client = FakeTradingClient()
    client.account.trading_blocked = True
    result = runner.execute_day18(
        config=safe_day18_config(),
        artifact_directory=tmp_path / "day18",
        broker=AlpacaPaperBroker(
            config=safe_day18_config(),
            client=client,
        ),
    )
    assert result.evaluation_complete is True
    assert result.preflight.preflight_passed is False
    assert (tmp_path / "day18" / "report.md").exists()


def test_main_raises_after_failed_preflight_artifact_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    passing = runner.execute_day18(
        config=safe_day18_config(),
        artifact_directory=tmp_path / "source",
        broker=_broker(),
    )
    failed = replace(
        passing,
        preflight=replace(
            passing.preflight,
            account_gate_passed=False,
            preflight_passed=False,
        ),
    )
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner, "load_project_config", lambda path: safe_day18_config()
    )
    monkeypatch.setattr(runner, "execute_day18", lambda **kwargs: failed)

    with pytest.raises(PaperBrokerPreflightError):
        runner.main(
            [
                "--config-path",
                "config.yaml",
                "--artifact-directory",
                "artifacts/day18",
            ]
        )
