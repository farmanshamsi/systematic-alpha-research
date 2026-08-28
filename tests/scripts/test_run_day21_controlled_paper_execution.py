from pathlib import Path

import pytest

import scripts.run_day21_controlled_paper_execution as live_runner
import scripts.run_day21_synthetic_validation as synthetic_runner
from systematic_alpha.broker.day21_report import APPROVED_DAY21_ARTIFACT_NAMES


def test_live_runner_refuses_absent_explicit_flag(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="authorized-paper-order"):
        live_runner.execute_day21(
            artifact_directory=tmp_path / "day21",
            authorized_paper_order=False,
        )


def test_synthetic_runner_writes_exact_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(synthetic_runner, "find_project_root", lambda: tmp_path)
    result = synthetic_runner.main(
        ["--artifact-directory", "artifacts/day21/synthetic"]
    )
    output = tmp_path / "artifacts/day21/synthetic"
    assert result.execution_complete
    assert {path.name for path in output.iterdir()} == set(
        APPROVED_DAY21_ARTIFACT_NAMES
    )
    printed = capsys.readouterr().out
    assert "broker_network_accessed: false" in printed
    assert "orders_submitted: false" in printed


def test_day20_prerequisite_verifies_current_bundle() -> None:
    assert live_runner.verify_day20_prerequisite(Path.cwd())
