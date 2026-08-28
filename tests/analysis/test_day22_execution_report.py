import hashlib
import json
from pathlib import Path

import pytest

from systematic_alpha.analysis.day22_execution_report import (
    APPROVED_DAY22_ARTIFACT_NAMES,
    write_day22_execution_artifacts,
)
from systematic_alpha.analysis.day22_scenarios import run_day22_synthetic_scenarios


def test_day22_bundle_hashes_replays_and_excludes_credentials(tmp_path: Path) -> None:
    results = run_day22_synthetic_scenarios()
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = write_day22_execution_artifacts(results, first)
    write_day22_execution_artifacts(results, second)
    assert tuple(path.name for path in paths) == APPROVED_DAY22_ARTIFACT_NAMES
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in APPROVED_DAY22_ARTIFACT_NAMES
    )
    manifest = json.loads((first / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["hashes"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == expected
    payload = b"".join(path.read_bytes() for path in paths)
    assert b"ALPACA_API_KEY=" not in payload
    assert b"ALPACA_SECRET_KEY=" not in payload
    assert manifest["evidence_separation"] == {
        "calibration_probe_alpha_eligible": False,
        "strategy_signal_alpha_eligible": True,
    }
    assert manifest["safety"]["live_campaign_authorized"] is False


def test_day22_writer_preserves_existing_bundle_on_staging_failure(
    tmp_path: Path, monkeypatch
) -> None:
    results = run_day22_synthetic_scenarios()
    output = tmp_path / "day22"
    write_day22_execution_artifacts(results, output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    real_write = Path.write_bytes

    def fail_daily(path: Path, data: bytes):
        if path.name == "daily_performance.csv" and ".stage-" in path.parent.name:
            raise OSError("synthetic staging failure")
        return real_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_daily)
    with pytest.raises(OSError, match="synthetic staging failure"):
        write_day22_execution_artifacts(results, output, overwrite=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original

