"""Tests for the Day 24 portable-report runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_day24_technical_report import (
    APPROVED_DAY24_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
    enhance_typeset_math,
    parse_args,
    write_day24_artifacts,
)

from systematic_alpha.analysis.day24_technical_report import (
    REPORT_MATH_GROUPS,
    report_math_placeholder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _renderer(input_path: Path, output_path: Path) -> dict[str, object]:
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    assert artifact["surface"] == "report"
    output_path.write_text(
        '<!doctype html><div id="data-analytics-portable-reader-root">Day 24</div>',
        encoding="utf-8",
    )
    return {
        "ok": True,
        "stages": {
            "validation": "passed",
            "package": "passed",
            "verification": "structural_only",
        },
        "browserWarning": {"code": "browser_unavailable"},
        "viewports": [],
    }


def test_writer_is_exact_hashed_replayable_and_copied(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output_copy = tmp_path / "outputs/report.html"
    paths = write_day24_artifacts(
        PROJECT_ROOT,
        first,
        output_report=output_copy,
        renderer=_renderer,
    )
    write_day24_artifacts(PROJECT_ROOT, second, renderer=_renderer)
    assert tuple(path.name for path in paths) == APPROVED_DAY24_ARTIFACT_NAMES
    assert {path.name for path in first.iterdir()} == set(APPROVED_DAY24_ARTIFACT_NAMES)
    manifest = json.loads((first / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["row_counts"] == {
        "chart_map.csv": 6,
        "charts": 6,
        "claim_inventory.csv": 14,
    }
    assert manifest["report_snapshot_status"] == "partial"
    assert manifest["delivery_validation"]["verification"] == "structural_only"
    assert manifest["safety"]["locked_2026_final_test_data_accessed"] is True
    for name, expected in manifest["hashes"].items():
        observed = hashlib.sha256((first / name).read_bytes()).hexdigest()
        assert observed == expected
    for name in APPROVED_DAY24_ARTIFACT_NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert output_copy.read_bytes() == (first / "report.html").read_bytes()


def test_writer_protects_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "day24"
    write_day24_artifacts(PROJECT_ROOT, destination, renderer=_renderer)
    with pytest.raises(FileExistsError):
        write_day24_artifacts(PROJECT_ROOT, destination, renderer=_renderer)


def test_parser_rejects_execution_authorization_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--authorized-paper-campaign"])


def test_typeset_math_enhancement_is_offline_complete_and_deterministic() -> None:
    paragraphs = "".join(
        f"<p>{report_math_placeholder(group_id)}</p>"
        for group_id, _columns, _equations in REPORT_MATH_GROUPS
    )
    source = (
        "<!doctype html><html><head></head><body>"
        '<section class="portable-notice" aria-labelledby="portable-access-issues">'
        '<h2 id="portable-access-issues">Data access issues</h2>'
        "<p>Empirical Day 22 paper fills are not available.</p></section>"
        f"{paragraphs}</body></html>"
    )
    first, receipt = enhance_typeset_math(source)
    second, replay_receipt = enhance_typeset_math(source)
    assert first == second
    assert (
        receipt
        == replay_receipt
        == {
            "equation_groups": 11,
            "equations": 40,
            "static_replacements": 11,
            "global_notices_removed": 1,
        }
    )
    assert 'data-axiom-typeset-math="true"' in first
    assert 'data-axiom-research-theme="notebook-v1"' in first
    assert "color-scheme:light!important" in first
    assert "border-radius:2px" in first
    assert 'data-axiom-math-group="performance_metrics"' in first
    assert 'role="img" aria-label="LaTeX equation:' in first
    assert "currentColor" in first
    assert "<metadata>" not in first
    assert "#101828" not in first
    assert "Execution evidence limitation" not in first
    assert ">Data access issues</h2>" not in first
    assert '<section class="portable-notice"' not in first
    assert '.portable-markdown p,.portable-markdown li' in first
    assert 'color:#243b53!important' in first
    assert '.access-issue-strip{display:none!important}' in first
    assert "let hydrating = false;" in first
    assert "if (hydrating) return;" in first
    assert "evidenceMarkers.some" in first
    assert "notice.remove();" in first
    assert all(
        f"<p>{report_math_placeholder(group_id)}</p>" not in first
        for group_id, _columns, _equations in REPORT_MATH_GROUPS
    )
