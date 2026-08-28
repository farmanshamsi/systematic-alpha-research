import hashlib
import json
from pathlib import Path

import pytest

from systematic_alpha.operations.day23_report import (
    APPROVED_DAY23_ARTIFACT_NAMES,
    write_day23_artifacts,
)
from systematic_alpha.operations.runtime_validation import run_operational_validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validated_result(tmp_path: Path):
    return run_operational_validation(
        PROJECT_ROOT,
        probe_parent=tmp_path,
        clean_environment_validated=True,
    )


def test_day23_bundle_replays_hashes_and_excludes_credentials(
    tmp_path: Path,
) -> None:
    result = _validated_result(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = write_day23_artifacts(result, first)
    write_day23_artifacts(result, second)
    assert tuple(path.name for path in paths) == APPROVED_DAY23_ARTIFACT_NAMES
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in APPROVED_DAY23_ARTIFACT_NAMES
    )
    manifest = json.loads((first / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["hashes"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == expected
    assert manifest["row_counts"] == {
        "dependency_audit.csv": 18,
        "health_checks.csv": 14,
        "schedule_entrypoints.csv": 3,
    }
    payload = b"".join(path.read_bytes() for path in paths)
    assert b"ALPACA_API_KEY=" not in payload
    assert b"ALPACA_SECRET_KEY=" not in payload
    assert manifest["safety"]["orders_submitted"] is False
    assert manifest["validation"]["container_runtime_validated"] is False


def test_day23_writer_requires_clean_environment(tmp_path: Path) -> None:
    result = run_operational_validation(PROJECT_ROOT, probe_parent=tmp_path)
    with pytest.raises(ValueError, match="Clean-environment validation"):
        write_day23_artifacts(result, tmp_path / "day23")


def test_day23_writer_preserves_bundle_on_staging_failure(
    tmp_path: Path, monkeypatch
) -> None:
    result = _validated_result(tmp_path)
    output = tmp_path / "day23"
    write_day23_artifacts(result, output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    real_write = Path.write_bytes

    def fail_runtime(path: Path, data: bytes):
        if path.name == "runtime_contract.json" and ".stage-" in path.parent.name:
            raise OSError("synthetic staging failure")
        return real_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_runtime)
    with pytest.raises(OSError, match="synthetic staging failure"):
        write_day23_artifacts(result, output, overwrite=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original

