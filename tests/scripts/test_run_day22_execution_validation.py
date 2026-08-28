from pathlib import Path

import pytest

import scripts.run_day22_execution_validation as runner
from systematic_alpha.analysis.day22_execution_report import APPROVED_DAY22_ARTIFACT_NAMES


def test_execute_day22_writes_exact_bundle(tmp_path: Path) -> None:
    results, paths = runner.execute_day22(artifact_directory=tmp_path / "day22")
    assert results.evaluation_complete
    assert tuple(path.name for path in paths) == APPROVED_DAY22_ARTIFACT_NAMES


@pytest.mark.parametrize("value", (None, 1, object()))
def test_execute_day22_rejects_non_path(value: object) -> None:
    with pytest.raises(TypeError):
        runner.execute_day22(artifact_directory=value)  # type: ignore[arg-type]


def test_main_prints_neutral_safety_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)
    results, paths = runner.main(["--artifact-directory", "artifacts/day22"])
    assert results.risk_summary[0]["risk_metrics_available"]
    assert len(paths) == 8
    output = capsys.readouterr().out
    assert "broker_network_accessed: false" in output
    assert "orders_submitted: false" in output
    assert "live_campaign_authorized: false" in output

