from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_day19_order_state_machine as runner
from systematic_alpha.broker.day19_report import APPROVED_DAY19_ARTIFACT_NAMES


def test_execute_day19_writes_complete_bundle(tmp_path: Path) -> None:
    result = runner.execute_day19(
        artifact_directory=tmp_path / "day19"
    )
    assert result.evaluation_complete is True
    assert len(result.scenario_results.scenario_summary) == 9
    assert tuple(path.name for path in result.artifact_paths) == (
        APPROVED_DAY19_ARTIFACT_NAMES
    )


def test_execute_day19_calls_each_stage_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    counts = {"scenarios": 0, "report": 0, "writer": 0}
    original_scenarios = runner.run_day19_scenarios
    original_report = runner.build_day19_order_state_report
    original_writer = runner.write_day19_order_state_artifacts

    def counted_scenarios():
        counts["scenarios"] += 1
        return original_scenarios()

    def counted_report(results):
        counts["report"] += 1
        return original_report(results)

    def counted_writer(report, destination, *, overwrite=False):
        counts["writer"] += 1
        return original_writer(report, destination, overwrite=overwrite)

    monkeypatch.setattr(runner, "run_day19_scenarios", counted_scenarios)
    monkeypatch.setattr(
        runner, "build_day19_order_state_report", counted_report
    )
    monkeypatch.setattr(
        runner, "write_day19_order_state_artifacts", counted_writer
    )
    runner.execute_day19(artifact_directory=tmp_path / "day19")
    assert counts == {"scenarios": 1, "report": 1, "writer": 1}


@pytest.mark.parametrize("value", (None, 1, object()))
def test_execute_day19_rejects_non_path_destination(value: object) -> None:
    with pytest.raises(TypeError):
        runner.execute_day19(artifact_directory=value)  # type: ignore[arg-type]


def test_execute_day19_rejects_non_boolean_overwrite(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        runner.execute_day19(
            artifact_directory=tmp_path / "day19",
            overwrite=1,  # type: ignore[arg-type]
        )


def test_main_resolves_relative_path_and_prints_neutral_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)
    result = runner.main(["--artifact-directory", "artifacts/day19"])
    assert result.artifact_directory == tmp_path / "artifacts/day19"
    output = capsys.readouterr().out
    assert "evaluation_complete: True" in output
    assert "broker_network_accessed: false" in output
    assert "credentials_accessed: false" in output
    assert "orders_submitted: false" in output

