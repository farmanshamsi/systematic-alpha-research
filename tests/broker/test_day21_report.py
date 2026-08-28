import hashlib
import json

from systematic_alpha.broker.controlled_paper_execution import run_controlled_paper_execution
from systematic_alpha.broker.day21_report import (
    APPROVED_DAY21_ARTIFACT_NAMES,
    write_day21_artifacts,
)
from tests.broker.test_controlled_paper_execution import (
    FakeBroker,
    authorization,
    signal,
)


def test_day21_artifact_bundle_hashes_and_replays(tmp_path) -> None:
    result = run_controlled_paper_execution(
        FakeBroker(),
        signal=signal(),
        authorization=authorization(),
        day20_gate_passed=True,
        sleep=lambda _: None,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = write_day21_artifacts(result, first)
    write_day21_artifacts(result, second)
    assert tuple(path.name for path in paths) == APPROVED_DAY21_ARTIFACT_NAMES
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in APPROVED_DAY21_ARTIFACT_NAMES)
    manifest = json.loads((first / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["hashes"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == expected
    payload = b"".join(path.read_bytes() for path in paths)
    assert b"ALPACA_API_KEY=" not in payload
    assert b"ALPACA_SECRET_KEY=" not in payload
    assert manifest["safety"]["real_money_orders_submitted"] is False

