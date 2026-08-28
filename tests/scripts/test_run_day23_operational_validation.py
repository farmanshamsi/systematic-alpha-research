from pathlib import Path

import pytest

from scripts.run_day23_operational_validation import execute_day23, parse_args


def test_day23_check_only_writes_no_bundle(tmp_path: Path) -> None:
    output = tmp_path / "day23"
    result, paths = execute_day23(
        artifact_directory=output,
        probe_parent=tmp_path,
        check_only=True,
    )
    assert result.evaluation_complete is True
    assert len(result.health_checks) == 14
    assert paths == ()
    assert not output.exists()


def test_day23_artifact_run_requires_clean_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Clean-environment validation"):
        execute_day23(
            artifact_directory=tmp_path / "day23",
            probe_parent=tmp_path,
        )


def test_day23_parser_rejects_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--authorized-paper-campaign"])

