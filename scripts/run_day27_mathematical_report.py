"""Build the versioned Day 27 mathematical-revision report bundle."""

from __future__ import annotations

import argparse
from io import BytesIO
import html
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Final, Mapping, Sequence

from systematic_alpha.analysis.day27_mathematical_report import (
    ARTIFACT_VERSION,
    CHART_COLUMNS,
    CLAIM_COLUMNS,
    REPORT_MATH_GROUPS,
    build_report_artifact,
    chart_inventory,
    claim_inventory,
    csv_bytes,
    json_bytes,
    report_math_placeholder,
    source_notes,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path("artifacts/day27_mathematical_revision")
DEFAULT_OUTPUT_REPORT: Final[Path] = Path(
    "outputs/AXIOM_DAY27_MATHEMATICAL_REPORT_DRAFT.html"
)
NODE_EXECUTABLE: Final[Path] = Path(
    "/Users/majazelahi/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/node/bin/node"
)
DELIVERY_SCRIPT: Final[Path] = Path(
    "/Users/majazelahi/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/"
    "deliver_portable_artifact.mjs"
)

REPORT_ARTIFACT_FILENAME: Final[str] = "report_artifact.json"
REPORT_FILENAME: Final[str] = "report.html"
CLAIMS_FILENAME: Final[str] = "claim_inventory.csv"
CHART_MAP_FILENAME: Final[str] = "chart_map.csv"
SOURCE_NOTES_FILENAME: Final[str] = "source_notes.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_DAY24_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    REPORT_ARTIFACT_FILENAME,
    REPORT_FILENAME,
    CLAIMS_FILENAME,
    CHART_MAP_FILENAME,
    SOURCE_NOTES_FILENAME,
    MANIFEST_FILENAME,
)

Renderer = Callable[[Path, Path], Mapping[str, object]]

MATH_FOREGROUND: Final[str] = "#101828"
MATH_STYLE_MARKER: Final[str] = "data-axiom-typeset-math"


def _latex_svg(latex: str) -> str:
    """Render one supported LaTeX expression as deterministic inline SVG."""

    cache = Path(tempfile.gettempdir()) / "axiom-day24-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    stream = BytesIO()
    with rc_context(
        {
            "mathtext.fontset": "stix",
            "svg.fonttype": "path",
            "svg.hashsalt": "axiom-day24-latex-v1",
        }
    ):
        math_to_image(
            f"${latex}$",
            stream,
            color=MATH_FOREGROUND,
            dpi=144,
            format="svg",
            prop=FontProperties(size=16),
        )
    rendered = stream.getvalue().decode("utf-8")
    match = re.search(r"<svg\b.*?</svg>", rendered, flags=re.DOTALL)
    if match is None:
        raise RuntimeError("Matplotlib did not return an SVG equation.")
    svg = match.group(0)
    svg = re.sub(r"\s*<metadata>.*?</metadata>", "", svg, flags=re.DOTALL)
    svg = re.sub(r'\s*<g id="patch_1">.*?</g>', "", svg, count=1, flags=re.DOTALL)
    svg = svg.replace(MATH_FOREGROUND, "currentColor")
    label = html.escape(f"LaTeX equation: {latex}", quote=True)
    return svg.replace(
        "<svg ",
        f'<svg role="img" aria-label="{label}" focusable="false" ',
        1,
    )


def _math_group_html(group_id: str, columns: int, equations: tuple[str, ...]) -> str:
    items = "".join(
        f'<div class="axiom-equation-line">{_latex_svg(equation)}</div>'
        for equation in equations
    )
    return (
        f'<div class="axiom-equation-group axiom-equation-columns-{columns}" '
        f'data-axiom-math-group="{html.escape(group_id, quote=True)}" '
        'aria-label="Typeset mathematical equations" role="group">'
        f"{items}</div>"
    )


def enhance_typeset_math(report_html: str) -> tuple[str, dict[str, int]]:
    """Replace static markers and hydrate enhanced-reader markers offline."""

    if not isinstance(report_html, str):
        raise TypeError("report_html must be a string.")
    group_html = {
        group_id: _math_group_html(group_id, columns, equations)
        for group_id, columns, equations in REPORT_MATH_GROUPS
    }
    enhanced = report_html
    enhanced, notice_sections_removed = re.subn(
        r'<section class="portable-notice"\s+'
        r'aria-labelledby="portable-access-issues">.*?</section>',
        "",
        enhanced,
        count=1,
        flags=re.DOTALL,
    )
    enhanced, notice_headings_removed = re.subn(
        r'<h2 id="portable-access-issues">'
        r'(?:Data access issues|Execution evidence limitation)</h2>',
        "",
        enhanced,
        count=1,
    )
    global_notices_removed = notice_sections_removed + notice_headings_removed
    static_replacements = 0
    for group_id, group in group_html.items():
        marker = report_math_placeholder(group_id)
        pattern = re.compile(rf"<p>\s*{re.escape(marker)}\s*</p>")
        enhanced, count = pattern.subn(lambda _match: group, enhanced, count=1)
        static_replacements += count
    if static_replacements != len(REPORT_MATH_GROUPS):
        raise ValueError(
            "Portable fallback did not expose every LaTeX equation marker."
        )

    styles = f"""
<style {MATH_STYLE_MARKER}="true" data-axiom-research-theme="notebook-v1">
:root{{color-scheme:light!important;--portable-canvas:#f4f5f3!important;--portable-surface:#ffffff!important;--portable-surface-subtle:#f7f8f6!important;--portable-ink:#1f2933!important;--portable-muted:#52606d!important;--portable-tertiary:#7b8794!important;--portable-table-text:#334e68!important;--portable-border:#d9dee3!important;--portable-accent:#235789!important;--portable-positive:#1f5f3f!important;--portable-positive-bg:#edf5ef!important;--portable-negative:#8f2d2d!important;--portable-negative-bg:#f8eeee!important;--portable-warning-bg:#fff9e8!important;--portable-warning-border:#c9a227!important;--portable-radius:3px!important}}
html,body{{background:#f4f5f3!important;color:#1f2933!important}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif!important;font-size:15px!important;line-height:1.68!important}}
.portable-fallback{{width:min(1040px,100%)!important;max-width:1040px!important;margin:0 auto!important;padding:34px 48px 72px!important;background:#fff!important;border-right:1px solid #d9dee3;border-left:1px solid #d9dee3}}
.portable-page-header{{position:static!important;display:grid!important;grid-template-columns:minmax(0,1fr) auto!important;width:auto!important;height:auto!important;min-height:0!important;margin:0 0 34px!important;padding:0 0 18px!important;border:0!important;border-bottom:2px solid #334e68!important;background:#fff!important}}
.portable-page-header h1{{margin:0!important;overflow:visible!important;color:#102a43!important;font-family:Georgia,"Times New Roman",serif!important;font-size:26px!important;font-weight:600!important;line-height:1.2!important;letter-spacing:0!important;white-space:normal!important}}
.portable-page-meta{{align-items:flex-end!important;gap:4px!important;color:#627d98!important;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:11px!important;font-weight:500!important;line-height:1.4!important}}
.portable-status{{border-radius:2px!important}}
.portable-block-stack{{display:grid!important;grid-template-columns:minmax(0,1fr)!important;gap:26px!important;margin-top:0!important}}
.portable-layout-half,.portable-layout-full{{grid-column:1!important}}
.portable-markdown{{max-width:900px!important;color:#243b53!important}}
.portable-markdown p,.portable-markdown li,.portable-markdown dt,.portable-markdown dd{{color:#243b53!important}}
.portable-markdown p{{margin:0 0 14px!important}}
.portable-markdown strong,.portable-markdown em{{color:inherit!important}}
.portable-markdown a{{color:#235789!important}}
.portable-markdown h1{{color:#102a43!important;font-family:Georgia,"Times New Roman",serif!important;font-size:32px!important;font-weight:600!important;line-height:1.16!important}}
.portable-markdown h2{{margin:16px 0 14px!important;padding-bottom:7px!important;border-bottom:1px solid #9fb3c8!important;color:#102a43!important;font-family:Georgia,"Times New Roman",serif!important;font-size:23px!important;font-weight:600!important;line-height:1.25!important}}
.portable-markdown h3{{margin:12px 0 10px!important;color:#243b53!important;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:15px!important;font-weight:650!important;line-height:1.4!important}}
.portable-markdown code,.portable-markdown pre,.portable-table-scroll th,.portable-table-number{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important}}
.portable-markdown code{{padding:1px 4px!important;border:1px solid #d9dee3!important;border-radius:2px!important;background:#f6f8fa!important;color:#102a43!important}}
.portable-markdown blockquote{{margin:16px 0!important;padding:10px 14px!important;border-left:3px solid #486581!important;background:#f6f8fa!important;color:#334e68!important}}
.portable-content-card{{padding:0!important;border:0!important;border-radius:0!important;background:transparent!important}}
.portable-metric-grid{{grid-template-columns:repeat(auto-fit,minmax(168px,1fr))!important;gap:10px!important}}
.portable-metric-card{{padding:14px 15px!important;border:1px solid #d9dee3!important;border-top:3px solid #486581!important;border-radius:2px!important;background:#fff!important;box-shadow:none!important}}
.portable-metric-label{{color:#52606d!important;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:11px!important;line-height:1.45!important;text-transform:uppercase}}
.portable-metric-value{{color:#102a43!important;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:22px!important;font-weight:650!important}}
.portable-visual-header{{margin:0 0 10px!important;padding-bottom:8px!important;border-bottom:1px solid #bcccdc!important}}
.portable-visual-header>strong,.portable-visual-header h1,.portable-visual-header h2,.portable-visual-header h3{{color:#102a43!important;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:14px!important;font-weight:650!important;line-height:1.45!important}}
.portable-visual-header>span,.portable-visual-header>p,.portable-visual-header p,.portable-visual-header li{{color:#52606d!important;font-size:12px!important;line-height:1.5!important}}
.portable-static-chart{{margin:0!important;border:0!important;border-radius:0!important;background:#fff!important}}
.portable-static-chart-light{{display:block!important}}
.portable-static-chart-dark{{display:none!important}}
.portable-static-chart-legend-wrap,.portable-static-chart-legend{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:11px!important}}
.portable-table-scroll{{overflow:auto!important;border:1px solid #d9dee3!important;border-radius:2px!important;background:#fff!important}}
.portable-table-scroll table{{width:max-content!important;min-width:100%!important}}
.portable-table-scroll th{{padding:8px 10px!important;background:#f0f4f8!important;color:#334e68!important;font-size:10px!important;font-weight:700!important;line-height:1.35!important;text-transform:uppercase!important}}
.portable-table-scroll td{{padding:8px 10px!important;border-bottom:1px solid #e6e9ed!important;color:#334e68!important;font-size:12px!important;line-height:1.45!important}}
.portable-table-scroll tbody tr:nth-child(even) td{{background:#fafbfc!important}}
.portable-table-note{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;font-size:10px!important}}
.portable-notice[aria-labelledby="portable-access-issues"],.access-issue-strip{{display:none!important}}
.portable-sources{{border-top:2px solid #334e68!important}}
.axiom-equation-group{{display:grid;grid-template-columns:minmax(0,1fr);gap:8px 22px;margin:14px 0 20px;padding:14px 16px;border:1px solid #d9dee3;border-left:3px solid #486581;border-radius:2px;background:#fbfcfd;color:#102a43;overflow-x:auto}}
.axiom-equation-columns-2{{grid-template-columns:repeat(2,minmax(0,1fr))}}
.axiom-equation-line{{display:flex;min-height:30px;align-items:center;justify-content:flex-start;color:inherit}}
.axiom-equation-line svg{{display:block;width:auto;max-width:100%;height:auto;max-height:46px;overflow:visible;color:inherit}}
@media(prefers-color-scheme:dark){{:root{{color-scheme:light!important;--portable-canvas:#f4f5f3!important;--portable-surface:#fff!important;--portable-surface-subtle:#f7f8f6!important;--portable-ink:#1f2933!important;--portable-muted:#52606d!important;--portable-tertiary:#7b8794!important;--portable-table-text:#334e68!important;--portable-border:#d9dee3!important;--portable-accent:#235789!important;--portable-positive:#1f5f3f!important;--portable-positive-bg:#edf5ef!important;--portable-negative:#8f2d2d!important;--portable-negative-bg:#f8eeee!important;--portable-warning-bg:#fff9e8!important;--portable-warning-border:#c9a227!important}}}}
@media(max-width:760px){{.portable-fallback{{padding:22px 18px 48px!important;border:0!important}}.portable-page-header{{grid-template-columns:minmax(0,1fr)!important;gap:10px!important}}.portable-page-meta{{align-items:flex-start!important}}.axiom-equation-columns-2{{grid-template-columns:minmax(0,1fr)}}.axiom-equation-group{{padding:12px 10px}}}}
@media print{{html,body,.portable-fallback{{background:#fff!important}}.portable-fallback{{padding:0!important;border:0!important}}.portable-page-header{{position:static!important}}.portable-block-stack{{gap:18px!important}}.portable-metric-card,.portable-table-scroll,.axiom-equation-group{{break-inside:avoid;box-shadow:none!important}}}}
</style>
""".strip()
    if "</head>" not in enhanced:
        raise ValueError("Portable report is missing the document head.")
    enhanced = enhanced.replace("</head>", f"{styles}\n</head>", 1)

    hydration_payload = json.dumps(
        {
            report_math_placeholder(group_id): group
            for group_id, group in group_html.items()
        },
        sort_keys=True,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    script = f"""
<script {MATH_STYLE_MARKER}="true">
(() => {{
  const groups = {hydration_payload};
  const evidenceMarkers = [
    "Empirical Day 22 paper fills",
    "Paper-fill and slippage observations are not yet available"
  ];
  let hydrating = false;
  const hydrate = (root = document) => {{
    if (hydrating) return;
    hydrating = true;
    try {{
      for (const paragraph of root.querySelectorAll("p")) {{
        const key = paragraph.textContent.trim();
        const markup = groups[key];
        if (!markup) continue;
        const template = document.createElement("template");
        template.innerHTML = markup;
        paragraph.replaceWith(template.content);
      }}
      for (const notice of root.querySelectorAll(
        '.portable-notice[aria-labelledby="portable-access-issues"], '
        + '.access-issue-strip, #portable-access-issues'
      )) {{
        const text = notice.textContent || "";
        if (
          notice.id === "portable-access-issues"
          || evidenceMarkers.some((marker) => text.includes(marker))
        ) {{
          notice.remove();
        }}
      }}
    }} finally {{
      hydrating = false;
    }}
  }};
  hydrate();
  const observer = new MutationObserver(() => hydrate());
  observer.observe(document.documentElement, {{ childList: true, subtree: true }});
}})();
</script>
""".strip()
    if "</body>" not in enhanced:
        raise ValueError("Portable report is missing the document body.")
    enhanced = enhanced.replace("</body>", f"{script}\n</body>", 1)
    return enhanced, {
        "equation_groups": len(REPORT_MATH_GROUPS),
        "equations": sum(len(item[2]) for item in REPORT_MATH_GROUPS),
        "static_replacements": static_replacements,
        "global_notices_removed": global_notices_removed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Day 24 portable CQF technical report."
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable_renderer(input_path: Path, output_path: Path) -> Mapping[str, object]:
    if not NODE_EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"Bundled Node runtime is unavailable: {NODE_EXECUTABLE}"
        )
    if not DELIVERY_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Portable delivery script is unavailable: {DELIVERY_SCRIPT}"
        )
    completed = subprocess.run(
        [
            str(NODE_EXECUTABLE),
            str(DELIVERY_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ready-timeout-ms",
            "5000",
            "--action-timeout-ms",
            "2500",
            "--timeout-ms",
            "15000",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Portable report delivery failed: {detail}")
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Portable report delivery returned invalid JSON.") from exc
    if receipt.get("ok") is not True or not output_path.is_file():
        raise RuntimeError(
            "Portable report delivery did not publish a verified HTML file."
        )
    enhanced, math_receipt = enhance_typeset_math(
        output_path.read_text(encoding="utf-8")
    )
    output_path.write_text(enhanced, encoding="utf-8")
    receipt["mathRendering"] = math_receipt
    return receipt


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_report_artifact(artifact: Mapping[str, object]) -> None:
    if artifact.get("surface") != "report":
        raise ValueError("Day 24 artifact must use the report surface.")
    manifest = artifact.get("manifest")
    snapshot = artifact.get("snapshot")
    if not isinstance(manifest, dict) or not isinstance(snapshot, dict):
        raise ValueError("Day 24 report manifest and snapshot are required.")
    charts = manifest.get("charts")
    if not isinstance(charts, list) or len(charts) != 6:
        raise ValueError("Day 24 report must contain exactly six charts.")
    expected_chart_ids = (
        "trend_walk_forward_returns",
        "cross_market_annualized_returns",
        "reversion_cost_sensitivity",
        "annual_trend_vs_spy",
        "annual_ou_vs_equal_weight",
        "annual_short_sleeve_effect",
    )
    if tuple(chart.get("id") for chart in charts) != expected_chart_ids:
        raise ValueError("Day 24 chart contract changed.")
    if snapshot.get("status") != "partial":
        raise ValueError("Day 24 live-evidence limitation must remain visible.")
    issues = snapshot.get("accessIssues")
    issue_ids = (
        [issue.get("id") for issue in issues] if isinstance(issues, list) else []
    )
    if issue_ids != ["live_campaign_pending"]:
        raise ValueError("Day 24 must expose the remaining empirical-live issue.")
    sources = artifact.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Day 24 report sources are required.")
    source_ids = {source.get("id") for source in sources}
    if {
        "day25_locked_performance",
        "day25_locked_methodology",
        "day26_comparison",
        "day26_aggregate",
        "day26_inference",
        "day26_methodology",
        "annual_regime_comparison",
        "annual_short_effect",
        "annual_ou_concentration",
        "annual_regime_methodology",
    }.difference(source_ids):
        raise ValueError(
            "Day 24 report must include locked-test and Phase II evidence."
        )
    for source in sources:
        path = Path(str(source.get("path", "")))
        if path.is_absolute() or ".." in path.parts or not str(path):
            raise ValueError("Day 24 source metadata must be repository-relative.")
    serialized = json_bytes(artifact)
    forbidden = (
        b"ALPACA_API_KEY=",
        b"ALPACA_SECRET_KEY=",
        b"APCA-API-KEY-ID:",
        b"APCA-API-SECRET-KEY:",
    )
    if any(marker in serialized for marker in forbidden):
        raise ValueError("Credential-like content detected in report input.")


def write_day24_artifacts(
    project_root: str | Path,
    directory: str | Path,
    *,
    output_report: str | Path | None = None,
    overwrite: bool = False,
    renderer: Renderer = _portable_renderer,
) -> tuple[Path, ...]:
    """Build the exact six-file bundle with atomic replacement and output copy."""

    root = Path(project_root)
    output = Path(directory)
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    artifact = build_report_artifact(root)
    _validate_report_artifact(artifact)
    claims = claim_inventory(root)
    chart_map = chart_inventory(root)
    if len(claims) != 14 or len(chart_map) != 6:
        raise ValueError("Day 24 claim or chart inventory row count changed.")

    payloads: dict[str, bytes] = {
        REPORT_ARTIFACT_FILENAME: json_bytes(artifact),
        CLAIMS_FILENAME: csv_bytes(claims, CLAIM_COLUMNS),
        CHART_MAP_FILENAME: csv_bytes(chart_map, CHART_COLUMNS),
        SOURCE_NOTES_FILENAME: source_notes(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        for name, payload in payloads.items():
            (stage / name).write_bytes(payload)
        receipt = dict(
            renderer(stage / REPORT_ARTIFACT_FILENAME, stage / REPORT_FILENAME)
        )
        report_html = (stage / REPORT_FILENAME).read_bytes()
        if b"data-analytics-portable-reader" not in report_html:
            raise ValueError("Rendered Day 24 HTML is not a portable report artifact.")
        payloads[REPORT_FILENAME] = report_html
        verification_stage = (
            receipt.get("stages", {}).get("verification")
            if isinstance(receipt.get("stages"), dict)
            else None
        )
        if verification_stage not in {"passed", "structural_only"}:
            raise ValueError("Portable report verification status is invalid.")
        manifest = {
            "schema_version": "day24_technical_report_artifacts_v2",
            "artifact_version": ARTIFACT_VERSION,
            "artifact_order": list(APPROVED_DAY24_ARTIFACT_NAMES),
            "hash_algorithm": "sha256",
            "hashes": {
                name: _sha256(payloads[name])
                for name in APPROVED_DAY24_ARTIFACT_NAMES
                if name != MANIFEST_FILENAME
            },
            "row_counts": {
                CLAIMS_FILENAME: len(claims),
                CHART_MAP_FILENAME: len(chart_map),
                "charts": 6,
            },
            "report_snapshot_status": "partial",
            "delivery_validation": {
                "validation": receipt.get("stages", {}).get("validation"),
                "package": receipt.get("stages", {}).get("package"),
                "verification": verification_stage,
                "browser_warning_code": (
                    receipt.get("browserWarning", {}).get("code")
                    if isinstance(receipt.get("browserWarning"), dict)
                    else None
                ),
                "viewports_verified": len(receipt.get("viewports", [])),
            },
            "safety": {
                "broker_network_accessed": False,
                "credentials_accessed": False,
                "orders_submitted_canceled_or_replaced": False,
                "locked_2026_final_test_data_accessed": True,
                "strategy_retuned_or_promoted": False,
                "day22_watcher_changed_or_slot_consumed": False,
                "git_commit_or_push_performed": False,
            },
        }
        payloads[MANIFEST_FILENAME] = json_bytes(manifest)
        (stage / MANIFEST_FILENAME).write_bytes(payloads[MANIFEST_FILENAME])

        forbidden = (
            b"ALPACA_API_KEY=",
            b"ALPACA_SECRET_KEY=",
            b"APCA-API-KEY-ID:",
            b"APCA-API-SECRET-KEY:",
        )
        for name in APPROVED_DAY24_ARTIFACT_NAMES:
            content = (stage / name).read_bytes()
            if any(marker in content for marker in forbidden):
                raise ValueError(f"Credential-like content detected in {name}.")
        if {path.name for path in stage.iterdir()} != set(
            APPROVED_DAY24_ARTIFACT_NAMES
        ):
            raise RuntimeError("Day 24 staged artifact allow-list changed.")

        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent)
            )
            backup.rmdir()
            os.replace(output, backup)
        os.replace(stage, output)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise

    final_paths = tuple(output / name for name in APPROVED_DAY24_ARTIFACT_NAMES)
    if output_report is not None:
        _atomic_copy(output / REPORT_FILENAME, Path(output_report))
        if _sha256((output / REPORT_FILENAME).read_bytes()) != _sha256(
            Path(output_report).read_bytes()
        ):
            raise RuntimeError("Day 24 output report copy did not verify.")
    return final_paths


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = parse_args(argv)
    paths = write_day24_artifacts(
        PROJECT_ROOT,
        _project_path(arguments.artifact_directory),
        output_report=_project_path(arguments.output_report),
        overwrite=arguments.overwrite,
    )
    manifest = json.loads(
        (_project_path(arguments.artifact_directory) / MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_files": len(paths),
        "headline_claims": manifest["row_counts"][CLAIMS_FILENAME],
        "charts": manifest["row_counts"]["charts"],
        "report_snapshot_status": manifest["report_snapshot_status"],
        "portable_verification": manifest["delivery_validation"]["verification"],
        "broker_network_accessed": False,
        "orders_submitted_canceled_or_replaced": False,
        "locked_2026_final_test_data_accessed": True,
        "day22_watcher_changed_or_slot_consumed": False,
        "git_commit_or_push_performed": False,
    }
    print("===== DAY 27 MATHEMATICAL REPORT COMPLETE =====")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return paths


if __name__ == "__main__":
    main()
