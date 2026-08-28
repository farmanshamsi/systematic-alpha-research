"""Build the versioned Day 27 mathematical-revision evidence payload."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Final, Iterable, Mapping


GENERATED_AT: Final[str] = "2026-08-15T00:00:00Z"
ARTIFACT_VERSION: Final[str] = "day27_mathematical_revision_v0_2_ou_derivation"
OU_DERIVATION_MATH_GROUP_ID: Final[str] = "ou_exact_derivation"

REPORT_MATH_GROUPS: Final[tuple[tuple[str, int, tuple[str, ...]], ...]] = (
    (
        "performance_metrics",
        2,
        (
            r"W_T=\prod_{t=1}^{T}(1+r_t)",
            r"R_{\mathrm{cum}}=W_T-1",
            r"R_{\mathrm{ann}}=W_T^{A/T}-1",
            r"\sigma_{\mathrm{ann}}=s(r_t)\sqrt{A}",
            r"\mathrm{Sharpe}=\frac{\bar r}{s(r_t)}\sqrt{A}",
            r"\mathrm{DD}_t=\frac{W_t}{\max_{s\leq t}W_s}-1",
            r"\mathrm{MDD}=\min_t\mathrm{DD}_t",
        ),
    ),
    (
        "causal_trend_accounting",
        2,
        (
            r"q_t=z_{t-1}",
            r"u_t=\left|q_t-q_{t-1,\mathrm{end}}\right|+"
            r"\mathbf{1}_{\{\mathrm{session\ close}\}}\left|q_t\right|",
            r"r_t^{\mathrm{gross}}=q_t r_t^{\mathrm{proxy}}",
            r"\mathrm{cost}_t=u_t\frac{c_{\mathrm{bps}}}{10{,}000}",
            r"r_t^{\mathrm{net}}=r_t^{\mathrm{gross}}-\mathrm{cost}_t",
            r"q_{t,\mathrm{end}}=0",
        ),
    ),
    (
        "price_ratio_averages",
        1,
        (
            r"\mathrm{SMA}_s(t)=\frac{1}{8}\sum_{i=0}^{7}P_{t-i}",
            r"\mathrm{SMA}_\ell(t)=\frac{1}{32}\sum_{i=0}^{31}P_{t-i}",
            r"\rho_t=\frac{\mathrm{SMA}_s(t)}{\mathrm{SMA}_\ell(t)}",
        ),
    ),
    (
        "price_ratio_signal",
        1,
        (
            r"z_t=+1\quad\mathrm{if}\quad\rho_t>1+\delta",
            r"z_t=-1\quad\mathrm{if}\quad\rho_t<1-\delta",
            r"z_t=0\quad\mathrm{otherwise}",
        ),
    ),
    (
        "ema_definition",
        1,
        (
            r"\alpha_n=\frac{2}{n+1}",
            r"\mathrm{EMA}_n(t)=\alpha_nP_t+(1-\alpha_n)"
            r"\mathrm{EMA}_n(t-1)",
        ),
    ),
    (
        "macd_definition",
        1,
        (
            r"\mathrm{MACD}_t=\mathrm{EMA}_{12}(t)-\mathrm{EMA}_{26}(t)",
            r"\mathrm{Signal}_t=\mathrm{EMA}_{9}(\mathrm{MACD}_t)",
            r"\mathrm{Histogram}_t=\mathrm{MACD}_t-\mathrm{Signal}_t",
            r"H_t=\frac{\mathrm{Histogram}_t}{P_t}",
        ),
    ),
    (
        "ou_reference",
        1,
        (
            r"R_t=\frac{\sum_{i=t-n_R+1}^{t}V_i\,\mathrm{VWAP}_i}"
            r"{\sum_{i=t-n_R+1}^{t}V_i}",
            r"x_t=\log\left(\frac{C_t}{R_t}\right)",
        ),
    ),
    (
        OU_DERIVATION_MATH_GROUP_ID,
        1,
        (
            r"\mathrm{(OU1)}\quad \mathrm{d}x_t=\kappa(\theta-x_t)\,"
            r"\mathrm{d}t+\sigma\,\mathrm{d}W_t,\quad \kappa>0",
            r"\mathrm{(OU2)}\quad \mathrm{d}(e^{\kappa s}x_s)="
            r"\kappa\theta e^{\kappa s}\,\mathrm{d}s+"
            r"\sigma e^{\kappa s}\,\mathrm{d}W_s",
            r"\mathrm{(OU3)}\quad e^{\kappa(t+\Delta)}x_{t+\Delta}-"
            r"e^{\kappa t}x_t=\theta\left[e^{\kappa(t+\Delta)}-"
            r"e^{\kappa t}\right]+\sigma\int_t^{t+\Delta}"
            r"e^{\kappa s}\,\mathrm{d}W_s",
            r"\mathrm{(OU4)}\quad x_{t+\Delta}=\theta+(x_t-\theta)"
            r"e^{-\kappa\Delta}+\sigma\int_t^{t+\Delta}"
            r"e^{-\kappa(t+\Delta-s)}\,\mathrm{d}W_s",
            r"\mathrm{(OU5)}\quad \mathbb{E}[x_{t+\Delta}\mid"
            r"\mathcal{F}_t]=\theta+(x_t-\theta)e^{-\kappa\Delta}",
            r"\mathrm{(OU6)}\quad \mathrm{Var}(x_{t+\Delta}\mid"
            r"\mathcal{F}_t)=\sigma^2\int_t^{t+\Delta}"
            r"e^{-2\kappa(t+\Delta-s)}\,\mathrm{d}s="
            r"\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})",
            r"\mathrm{(OU7)}\quad \mathbb{E}[x_{t+h}-\theta\mid"
            r"\mathcal{F}_t]=e^{-\kappa h}(x_t-\theta)",
            r"\mathrm{(OU8)}\quad x_j=a+\phi x_{j-1}+\varepsilon_j",
            r"\mathrm{(OU9)}\quad \phi=e^{-\kappa\Delta},\qquad"
            r"a=\theta(1-\phi)",
            r"\mathrm{(OU10)}\quad \kappa=-\frac{\log\phi}{\Delta},"
            r"\qquad\theta=\frac{a}{1-\phi}",
            r"\mathrm{(OU11)}\quad \mathbb{E}[\varepsilon_j\mid"
            r"\mathcal{F}_{j-1}]=0,\qquad\sigma_\varepsilon^2="
            r"\frac{\sigma^2}{2\kappa}(1-\phi^2)",
            r"\mathrm{(OU12)}\quad \sigma=\sigma_\varepsilon"
            r"\sqrt{\frac{2\kappa}{1-\phi^2}}",
            r"\mathrm{(OU13)}\quad \bar{x}_{-,t}=\frac{1}{n_{\mathrm{OU}}}"
            r"\sum_{j=t-n_{\mathrm{OU}}+1}^{t}x_{j-1},\qquad"
            r"\bar{x}_{+,t}=\frac{1}{n_{\mathrm{OU}}}"
            r"\sum_{j=t-n_{\mathrm{OU}}+1}^{t}x_j",
            r"\mathrm{(OU14)}\quad \widehat{\phi}_t="
            r"\frac{\sum_j(x_{j-1}-\bar{x}_{-,t})"
            r"(x_j-\bar{x}_{+,t})}{\sum_j(x_{j-1}-\bar{x}_{-,t})^2}",
            r"\mathrm{(OU15)}\quad \widehat{a}_t=\bar{x}_{+,t}-"
            r"\widehat{\phi}_t\bar{x}_{-,t}",
            r"\mathrm{(OU16)}\quad \widehat{\varepsilon}_j=x_j-"
            r"\widehat{a}_t-\widehat{\phi}_t x_{j-1},\qquad"
            r"\widehat{\sigma}_{\varepsilon,t}^2="
            r"\frac{\sum_j\widehat{\varepsilon}_j^2}{n_{\mathrm{OU}}-2}",
            r"\mathrm{(OU17)}\quad \widehat{\sigma}_{x,t}^2="
            r"\frac{\widehat{\sigma}_{\varepsilon,t}^2}"
            r"{1-\widehat{\phi}_t^2},\qquad"
            r"\widehat{\sigma}_{x,t}=\frac{\widehat{\sigma}_{\varepsilon,t}}"
            r"{\sqrt{1-\widehat{\phi}_t^2}}",
            r"\mathrm{(OU18)}\quad h_{1/2}=\frac{\log 2}{\kappa}="
            r"-\frac{\Delta\log 2}{\log\phi},\qquad"
            r"h_{1/2}^{\mathrm{bars}}=-\frac{\log 2}{\log\phi}",
            r"\mathrm{(OU19)}\quad z_t=\frac{x_t-\widehat{\theta}_t}"
            r"{\widehat{\sigma}_{x,t}},\qquad"
            r"\widehat{\theta}_t=\frac{\widehat{a}_t}"
            r"{1-\widehat{\phi}_t}",
            r"\mathrm{(OU20)}\quad \mathrm{VR}_t(q)="
            r"\frac{\widehat{\mathrm{Var}}_t(x_j-x_{j-q})}"
            r"{q\,\widehat{\mathrm{Var}}_t(x_j-x_{j-1})}",
        ),
    ),
    (
        "diversification_ratio",
        1,
        (
            r"\mathrm{DR}(w)=\frac{w^{\mathsf{T}}\sigma}"
            r"{\sqrt{w^{\mathsf{T}}\Sigma w}}",
        ),
    ),
    (
        "execution_shortfall",
        1,
        (
            r"s=+1\ \mathrm{for\ buys},\quad s=-1\ \mathrm{for\ sells}",
            r"\mathrm{delay}_{\mathrm{bps}}=s"
            r"\frac{p_{\mathrm{arrival}}-p_{\mathrm{decision}}}"
            r"{p_{\mathrm{decision}}}\,10{,}000",
            r"\mathrm{execution}_{\mathrm{bps}}=s"
            r"\frac{p_{\mathrm{fill}}-p_{\mathrm{arrival}}}"
            r"{p_{\mathrm{arrival}}}\,10{,}000",
            r"\mathrm{total}_{\mathrm{bps}}=\mathrm{delay}_{\mathrm{bps}}+"
            r"\mathrm{execution}_{\mathrm{bps}}+\mathrm{explicit\ cost}_{\mathrm{bps}}",
            r"\mathrm{completion}=\frac{Q_{\mathrm{filled}}}{Q_{\mathrm{requested}}}",
            r"\mathrm{latency}_{\mathrm{ms}}=t_{\mathrm{fill}}-t_{\mathrm{submission}}",
        ),
    ),
)


def report_math_placeholder(group_id: str) -> str:
    """Return the exact marker replaced by the offline report renderer."""

    known_ids = {item[0] for item in REPORT_MATH_GROUPS}
    if group_id not in known_ids:
        raise ValueError(f"Unknown report math group: {group_id}.")
    return f"[[AXIOM_LATEX_{group_id.upper()}]]"


SOURCE_FILES: Final[dict[str, str]] = {
    "day10_robustness": "artifacts/day10/summary.csv",
    "day11_folds": "artifacts/day11/fold_results.csv",
    "day11_aggregate": "artifacts/day11/aggregate_results.csv",
    "day14_pairs": "artifacts/day14/pair_eligibility.csv",
    "day15_diversification": (
        "artifacts/day25_causal_portfolio_finalization/ensemble_feasibility.csv"
    ),
    "day16_allocation": (
        "artifacts/day25_causal_portfolio_finalization/"
        "aggregate_portfolio_performance.csv"
    ),
    "day17_reversion": "artifacts/day17/aggregate_performance.csv",
    "day17_costs": "artifacts/day17/cost_sensitivity.csv",
    "day17_inference": "artifacts/day17/inference_results.csv",
    "day18_preflight": "artifacts/day18/preflight_summary.json",
    "day19_order_state": "artifacts/day19/scenario_summary.csv",
    "day20_reconciliation": "artifacts/day20/scenario_summary.csv",
    "day21_read_only": "artifacts/day21/live_read_only/gate_results.csv",
    "day22_synthetic": "artifacts/day22/campaign_summary.csv",
    "day23_operations": "artifacts/day23/health_checks.csv",
    "day25_trend_timing": (
        "artifacts/day25_methodological_finalization/timing_comparison.csv"
    ),
    "day25_trend_walk_forward": (
        "artifacts/day25_methodological_finalization/walk_forward.csv"
    ),
    "day25_trend_robustness": (
        "artifacts/day25_methodological_finalization/robustness.csv"
    ),
    "day25_trend_sensitivity": (
        "artifacts/day25_methodological_finalization/long_flat_sensitivity.csv"
    ),
    "day25_trend_parity": (
        "artifacts/day25_methodological_finalization/replay_parity.csv"
    ),
    "day25_event_sampling": (
        "artifacts/day25_event_time_finalization/sampling_comparison.csv"
    ),
    "day25_event_indicator": (
        "artifacts/day25_event_time_finalization/indicator_comparison.csv"
    ),
    "day25_event_conservation": (
        "artifacts/day25_event_time_finalization/conservation.csv"
    ),
    "day25_locked_performance": "artifacts/day25_final_test/performance.csv",
    "day25_locked_methodology": "artifacts/day25_final_test/methodology.json",
    "day26_comparison": "artifacts/day26/comparison.csv",
    "day26_aggregate": "artifacts/day26/aggregate_performance.csv",
    "day26_inference": "artifacts/day26/inference.csv",
    "day26_methodology": "artifacts/day26/methodology.json",
    "annual_regime_comparison": (
        "artifacts/day24_annual_regime/annual_regime_comparison.csv"
    ),
    "annual_short_effect": ("artifacts/day24_annual_regime/short_sleeve_effect.csv"),
    "annual_ou_concentration": ("artifacts/day24_annual_regime/ou_concentration.csv"),
    "annual_regime_methodology": ("artifacts/day24_annual_regime/methodology.json"),
    "base_config": "config/base.yaml",
    "trend_ratio_code": "src/systematic_alpha/strategies/trend_ratio.py",
    "ema_macd_code": "src/systematic_alpha/strategies/ema_macd.py",
    "ou_vwap_code": "src/systematic_alpha/strategies/ou_vwap_reversion.py",
    "reversion_inference_code": "src/systematic_alpha/analysis/reversion_inference.py",
    "trend_finalization_code": (
        "src/systematic_alpha/analysis/trend_methodology_finalization.py"
    ),
    "locked_final_test_code": "src/systematic_alpha/analysis/locked_final_test.py",
    "day25_methodology_specification": (
        "docs/DAY25_METHODOLOGICAL_FINALIZATION_SPECIFICATION.md"
    ),
    "day24_specification": "docs/DAY24_TECHNICAL_REPORT_SPECIFICATION.md",
}

SOURCE_LABELS: Final[dict[str, str]] = {
    "day10_robustness": "Day 10 cross-market and frequency robustness",
    "day11_folds": "Day 11 chronological walk-forward folds",
    "day11_aggregate": "Day 11 aggregate walk-forward performance",
    "day14_pairs": "Day 14 pair-feasibility gates",
    "day15_diversification": "Day 25 causal six-sleeve diversification rebuild",
    "day16_allocation": "Day 25 causal portfolio allocation rebuild",
    "day17_reversion": "Day 17 OU/VWAP performance",
    "day17_costs": "Day 17 OU/VWAP cost sensitivity",
    "day17_inference": "Day 17 statistical inference",
    "day18_preflight": "Day 18 live read-only paper preflight",
    "day19_order_state": "Day 19 order-state scenarios",
    "day20_reconciliation": "Day 20 reconciliation scenarios",
    "day21_read_only": "Day 21 live read-only safety gates",
    "day22_synthetic": "Day 22 synthetic execution benchmark",
    "day23_operations": "Day 23 reproducible operations checks",
    "day25_trend_timing": "Day 25 saved-versus-causal trend timing audit",
    "day25_trend_walk_forward": "Day 25 causal trend walk-forward results",
    "day25_trend_robustness": "Day 25 causal trend robustness matrix",
    "day25_trend_sensitivity": "Day 25 long-flat sensitivity matrix",
    "day25_trend_parity": "Day 25 causal batch/sequential replay parity",
    "day25_event_sampling": "Day 25 five-session event-time sampling comparison",
    "day25_event_indicator": "Day 25 five-session event-time indicator comparison",
    "day25_event_conservation": "Day 25 five-session event-time conservation audit",
    "day25_locked_performance": "One-time locked 2026 final-test performance",
    "day25_locked_methodology": "One-time locked 2026 final-test methodology",
    "day26_comparison": "Day 26 predeclared Phase II development comparison",
    "day26_aggregate": "Day 26 aggregate Phase II development performance",
    "day26_inference": "Day 26 Phase II development inference",
    "day26_methodology": "Day 26 frozen Phase II methodology",
    "annual_regime_comparison": (
        "Post-hoc 2022-2025 annual strategy and price-only benchmark diagnostic"
    ),
    "annual_short_effect": "Post-hoc annual price-ratio short-sleeve diagnostic",
    "annual_ou_concentration": "Post-hoc annual slow OU/VWAP concentration diagnostic",
    "annual_regime_methodology": "Annual regime diagnostic methodology and evidence boundary",
    "base_config": "Frozen project configuration and sample boundaries",
    "trend_ratio_code": "Price-ratio trend implementation",
    "ema_macd_code": "EMA/MACD trend implementation",
    "ou_vwap_code": "OU/VWAP reversion implementation",
    "reversion_inference_code": "Reversion inference implementation",
    "trend_finalization_code": "Causal trend timing implementation",
    "locked_final_test_code": "One-time locked final-test implementation",
    "day25_methodology_specification": "Frozen methodological-finalization specification",
    "day24_specification": "Day 24 frozen reporting specification",
}

MANIFEST_DIRECTORIES: Final[tuple[str, ...]] = (
    "artifacts/day10",
    "artifacts/day11",
    "artifacts/day14",
    "artifacts/day15",
    "artifacts/day16",
    "artifacts/day17",
    "artifacts/day18",
    "artifacts/day19",
    "artifacts/day20",
    "artifacts/day21/live_read_only",
    "artifacts/day22",
    "artifacts/day23",
    "artifacts/day25_methodological_finalization",
    "artifacts/day25_event_time_finalization",
    "artifacts/day25_causal_portfolio_finalization",
    "artifacts/day25_final_test",
    "artifacts/day26",
    "artifacts/day24_annual_regime",
)

CLAIM_COLUMNS: Final[tuple[str, ...]] = (
    "claim_order",
    "claim_id",
    "headline_claim",
    "evidence_class",
    "source_file",
    "status",
    "value_context",
)

CHART_COLUMNS: Final[tuple[str, ...]] = (
    "chart_order",
    "chart_id",
    "question",
    "source_file",
    "dataset",
    "row_count",
    "x_field",
    "y_field",
    "color_field",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _manifest_hashes(manifest: Mapping[str, object]) -> dict[str, str]:
    if isinstance(manifest.get("artifact_sha256"), dict):
        return {
            str(name): str(digest)
            for name, digest in manifest["artifact_sha256"].items()  # type: ignore[union-attr]
        }
    if isinstance(manifest.get("hashes"), dict):
        return {
            str(name): str(digest)
            for name, digest in manifest["hashes"].items()  # type: ignore[union-attr]
        }
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        return {
            str(item["filename"]): str(item["sha256"])
            for item in artifacts
            if isinstance(item, dict)
            and isinstance(item.get("filename"), str)
            and isinstance(item.get("sha256"), str)
        }
    raise ValueError("Unsupported evidence-manifest hash schema.")


def verify_source_manifests(project_root: str | Path) -> tuple[dict[str, object], ...]:
    """Verify every saved evidence manifest used by the Day 24 report."""

    root = Path(project_root)
    results: list[dict[str, object]] = []
    for relative_directory in MANIFEST_DIRECTORIES:
        directory = root / relative_directory
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = _manifest_hashes(manifest)
        if not hashes:
            raise ValueError(f"Evidence manifest contains no hashes: {manifest_path}")
        for filename, expected in hashes.items():
            evidence_path = directory / filename
            observed = _sha256(evidence_path)
            if observed != expected:
                raise ValueError(
                    f"Evidence hash mismatch for {relative_directory}/{filename}."
                )
        results.append(
            {
                "manifest": f"{relative_directory}/manifest.json",
                "verified_files": len(hashes),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    return tuple(results)


def _source(project_root: Path, source_id: str) -> dict[str, object]:
    relative_path = SOURCE_FILES[source_id]
    return {
        "id": source_id,
        "label": SOURCE_LABELS[source_id],
        "path": relative_path,
        "sha256": _sha256(project_root / relative_path),
        "query": {
            "engine": "deterministic-file-snapshot",
            "sql": f"SELECT * FROM reviewed_file('{relative_path}')",
            "description": (
                "Manifest-verified repository file loaded by the deterministic "
                "Day 24 Python report builder; the SQL-form expression records "
                "the complete unfiltered file read used to create the bounded snapshot."
            ),
            "executed_at": GENERATED_AT,
            "tables_used": [relative_path],
        },
    }


def _select_one(rows: Iterable[dict[str, str]], **conditions: str) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one evidence row for {conditions}; found {len(selected)}."
        )
    return selected[0]


def _table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: list[dict[str, object]],
    *,
    default_sort: tuple[str, str] | None = None,
) -> dict[str, object]:
    table: dict[str, object] = {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "dense",
        "layout": "full",
        "columns": columns,
    }
    if default_sort is not None:
        table["defaultSort"] = {
            "field": default_sort[0],
            "direction": default_sort[1],
        }
    return table


def _comparison_chart(
    *,
    chart_id: str,
    title: str,
    subtitle: str,
    question: str,
    rationale: str,
    dataset: str,
    source_id: str,
    x_field: str,
    x_label: str,
    color_field: str,
    color_label: str,
    y_field: str,
    y_label: str,
) -> dict[str, object]:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "headerMarkdown": (
            "Bars start at zero; signed labels and the neutral zero line separate "
            "gains from losses. Hover for the supporting Sharpe, drawdown, sample, "
            "turnover, and cost context retained in each row."
        ),
        "type": "bar",
        "intent": "comparison",
        "question": question,
        "rationale": rationale,
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": {
            "x": {"field": x_field, "type": "ordinal", "label": x_label},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "label": y_label,
                "format": "percent",
            },
            "color": {
                "field": color_field,
                "type": "nominal",
                "label": color_label,
            },
        },
        "yAxisTitle": y_label,
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "categorical", "name": "comparison"},
        "legend": {
            "interactive": True,
            "position": "bottom",
            "sort": "spec",
            "title": color_label,
        },
        "labels": {"values": "auto"},
        "referenceLines": [
            {
                "value": 0,
                "axis": "y",
                "label": "zero return",
                "color": "neutral",
                "lineStyle": "solid",
            }
        ],
    }


def _load_evidence(project_root: Path) -> dict[str, object]:
    day10 = _read_csv(project_root / SOURCE_FILES["day10_robustness"])
    day11_folds = _read_csv(project_root / SOURCE_FILES["day11_folds"])
    day11_aggregate = _read_csv(project_root / SOURCE_FILES["day11_aggregate"])
    day14 = _read_csv(project_root / SOURCE_FILES["day14_pairs"])
    day15 = _read_csv(project_root / SOURCE_FILES["day15_diversification"])
    day16 = _read_csv(project_root / SOURCE_FILES["day16_allocation"])
    day17 = _read_csv(project_root / SOURCE_FILES["day17_reversion"])
    day17_costs = _read_csv(project_root / SOURCE_FILES["day17_costs"])
    day17_inference = _read_csv(project_root / SOURCE_FILES["day17_inference"])
    day18 = json.loads(
        (project_root / SOURCE_FILES["day18_preflight"]).read_text(encoding="utf-8")
    )
    day19 = _read_csv(project_root / SOURCE_FILES["day19_order_state"])
    day20 = _read_csv(project_root / SOURCE_FILES["day20_reconciliation"])
    day21 = _read_csv(project_root / SOURCE_FILES["day21_read_only"])
    day22 = _read_csv(project_root / SOURCE_FILES["day22_synthetic"])
    day23 = _read_csv(project_root / SOURCE_FILES["day23_operations"])
    day25_timing = _read_csv(project_root / SOURCE_FILES["day25_trend_timing"])
    day25_walk_forward = _read_csv(
        project_root / SOURCE_FILES["day25_trend_walk_forward"]
    )
    day25_robustness = _read_csv(project_root / SOURCE_FILES["day25_trend_robustness"])
    day25_sensitivity = _read_csv(
        project_root / SOURCE_FILES["day25_trend_sensitivity"]
    )
    day25_parity = _read_csv(project_root / SOURCE_FILES["day25_trend_parity"])
    day25_event_sampling = _read_csv(
        project_root / SOURCE_FILES["day25_event_sampling"]
    )
    day25_event_indicator = _read_csv(
        project_root / SOURCE_FILES["day25_event_indicator"]
    )
    day25_event_conservation = _read_csv(
        project_root / SOURCE_FILES["day25_event_conservation"]
    )
    day25_locked_performance = _read_csv(
        project_root / SOURCE_FILES["day25_locked_performance"]
    )
    day25_locked_methodology = json.loads(
        (project_root / SOURCE_FILES["day25_locked_methodology"]).read_text(
            encoding="utf-8"
        )
    )
    day26_comparison = _read_csv(project_root / SOURCE_FILES["day26_comparison"])
    day26_aggregate = _read_csv(project_root / SOURCE_FILES["day26_aggregate"])
    day26_inference = _read_csv(project_root / SOURCE_FILES["day26_inference"])
    day26_methodology = json.loads(
        (project_root / SOURCE_FILES["day26_methodology"]).read_text(encoding="utf-8")
    )
    annual_regime = _read_csv(project_root / SOURCE_FILES["annual_regime_comparison"])
    annual_short_effect = _read_csv(project_root / SOURCE_FILES["annual_short_effect"])
    annual_ou_concentration = _read_csv(
        project_root / SOURCE_FILES["annual_ou_concentration"]
    )
    annual_regime_methodology = json.loads(
        (project_root / SOURCE_FILES["annual_regime_methodology"]).read_text(
            encoding="utf-8"
        )
    )

    expected_locked_models = (
        "price_ratio_long_short_neutral",
        "ema_macd_long_short_neutral",
        "ou_vwap_slow_equal_weight",
    )
    if (
        tuple(row["model_id"] for row in day25_locked_performance)
        != expected_locked_models
    ):
        raise ValueError("Locked final-test model order or coverage changed.")
    if (
        day25_locked_methodology.get("one_time_run") is not True
        or day25_locked_methodology.get("authorization_code_validated") is not True
        or day25_locked_methodology.get("all_results_reported") is not True
        or day25_locked_methodology.get("ranking_or_retuning_performed") is not False
        or tuple(day25_locked_methodology.get("models", ())) != expected_locked_models
        or float(day25_locked_methodology.get("cost_bps_per_turnover", -1.0)) != 1.0
        or day25_locked_methodology.get("locked_start") != "2026-01-02T00:00:00+00:00"
        or day25_locked_methodology.get("locked_end_exclusive")
        != "2026-07-01T00:00:00+00:00"
        or day25_locked_methodology.get("development_history_used_for_warmup_only")
        is not True
        or day25_locked_methodology.get("execution_state_reset_at_test_start")
        is not True
    ):
        raise ValueError(
            "Locked final-test methodology does not satisfy the frozen protocol."
        )
    if any(
        row["test_start"] != "2026-01-02"
        or row["test_end"] != "2026-06-30"
        or int(row["sessions"]) != 123
        or _float(row, "cost_bps_per_turnover") != 1.0
        or int(row["initial_position"]) != 0
        or _float(row, "initial_turnover") != 0.0
        or int(row["overnight_position_violations"]) != 0
        for row in day25_locked_performance
    ):
        raise ValueError("Locked final-test performance boundary changed.")

    expected_phase2_keys = (
        ("price_ratio_long_flat", "SPY", 1.0),
        ("price_ratio_long_flat", "SPY", 5.0),
        ("ou_vwap_slow", "equal_weight", 1.0),
        ("ou_vwap_slow", "equal_weight", 5.0),
    )
    observed_phase2_keys = tuple(
        (
            row["strategy_family"],
            row["series"],
            _float(row, "cost_bps_per_turnover"),
        )
        for row in day26_comparison
    )
    if observed_phase2_keys != expected_phase2_keys:
        raise ValueError("Day 26 comparison order or coverage changed.")
    if (
        day26_methodology.get("schema_version") != "day26_phase2_profitability_v1"
        or int(day26_methodology.get("declared_phase2_trials", -1)) != 2
        or day26_methodology.get("locked_2026_interval_accessed") is not False
        or day26_methodology.get("untouched_future_holdout_available") is not False
        or day26_methodology.get("winner_selection_performed") is not False
        or day26_methodology.get("negative_results_suppressed") is not False
        or day26_methodology.get("leverage_used") is not False
    ):
        raise ValueError("Day 26 methodology violates the frozen Phase II boundary.")
    if len(day26_inference) != 10 or any(
        _float(row, "cost_bps_per_turnover") != 1.0
        or int(row["bootstrap_replications"]) != 2_000
        or int(row["hac_lags"]) != 5
        or int(row["declared_phase2_trials"]) != 2
        for row in day26_inference
    ):
        raise ValueError("Day 26 inference coverage or method changed.")
    if len(day26_aggregate) != 10 or any(
        _float(row, "cost_bps_per_turnover") != 1.0
        or int(row["test_sessions"]) != 1_003
        or int(row["overnight_position_violations"]) != 0
        for row in day26_aggregate
    ):
        raise ValueError("Day 26 aggregate performance boundary changed.")

    expected_years = (2022, 2023, 2024, 2025)
    if (
        len(annual_regime) != 24
        or tuple(sorted({int(row["year"]) for row in annual_regime})) != expected_years
        or len(annual_short_effect) != 4
        or tuple(int(row["year"]) for row in annual_short_effect) != expected_years
        or len(annual_ou_concentration) != 4
        or tuple(int(row["year"]) for row in annual_ou_concentration) != expected_years
    ):
        raise ValueError("Annual regime diagnostic coverage changed.")
    if (
        annual_regime_methodology.get("schema_version")
        != "day24_annual_regime_diagnostic_v1"
        or annual_regime_methodology.get("evidence_class")
        != "post_hoc_development_diagnostic"
        or annual_regime_methodology.get("benchmark_dividends_included") is not False
        or annual_regime_methodology.get("idle_cash_yield_included") is not False
        or annual_regime_methodology.get("locked_2026_interval_accessed") is not False
        or annual_regime_methodology.get("selection_or_retuning_performed") is not False
        or tuple(annual_regime_methodology.get("folds", ()))
        != tuple(f"wf_{year}" for year in expected_years)
    ):
        raise ValueError("Annual regime diagnostic methodology boundary changed.")
    annual_strategy_rows = [
        row for row in annual_regime if row["series_type"] == "strategy"
    ]
    if (
        any(_float(row, "cost_bps_per_turnover") != 1.0 for row in annual_strategy_rows)
        or any(
            _float(row, "excess_return") <= 0.0
            for row in annual_strategy_rows
            if int(row["year"]) == 2022
        )
        or any(
            _float(row, "excess_return") >= 0.0
            for row in annual_strategy_rows
            if int(row["year"]) != 2022
        )
        or _float(annual_short_effect[0], "net_return_effect") <= 0.0
        or any(
            _float(row, "net_return_effect") >= 0.0 for row in annual_short_effect[1:]
        )
        or int(annual_ou_concentration[0]["nonzero_portfolio_days"]) != 9
    ):
        raise ValueError("Annual regime diagnostic conclusion changed.")

    annual_rows = [
        {
            "comparison_group": row["comparison_group"],
            "year": int(row["year"]),
            "series": row["series"],
            "series_type": row["series_type"],
            "benchmark_basis": row["benchmark_basis"],
            "gross_return": _float(row, "gross_return"),
            "net_return": _float(row, "net_return"),
            "benchmark_return": _float(row, "benchmark_return"),
            "excess_return": _float(row, "excess_return"),
            "turnover": _float(row, "turnover"),
            "trade_count": _float(row, "trade_count"),
            "long_exposure_pct": _float(row, "long_exposure_pct"),
            "short_exposure_pct": _float(row, "short_exposure_pct"),
            "flat_exposure_pct": _float(row, "flat_exposure_pct"),
            "cost_bps_per_turnover": _float(row, "cost_bps_per_turnover"),
        }
        for row in annual_regime
    ]
    short_effect_rows = [
        {
            "year": int(row["year"]),
            "gross_return_effect": _float(row, "gross_return_effect"),
            "net_return_effect": _float(row, "net_return_effect"),
            "extra_turnover": _float(row, "extra_turnover"),
            "interpretation": row["interpretation"],
        }
        for row in annual_short_effect
    ]
    short_effect_chart_rows = [
        {
            "year": row["year"],
            "effect_type": effect_type,
            "return_effect": row[field],
            "extra_turnover": row["extra_turnover"],
            "interpretation": row["interpretation"],
        }
        for row in short_effect_rows
        for effect_type, field in (
            ("Gross return effect", "gross_return_effect"),
            ("Net return effect", "net_return_effect"),
        )
    ]
    ou_concentration_rows = [
        {
            "year": int(row["year"]),
            "net_cumulative_return": _float(row, "net_cumulative_return"),
            "nonzero_portfolio_days": int(row["nonzero_portfolio_days"]),
            "positive_days": int(row["positive_days"]),
            "negative_days": int(row["negative_days"]),
            "arithmetic_net_return_sum": _float(row, "arithmetic_net_return_sum"),
            "top_three_day_sum": _float(row, "top_three_day_sum"),
            "worst_three_day_sum": _float(row, "worst_three_day_sum"),
            "flat_exposure_pct": _float(row, "flat_exposure_pct"),
        }
        for row in annual_ou_concentration
    ]

    phase2_rows = [
        {
            "strategy_family": row["strategy_family"],
            "series": row["series"],
            "cost_bps": _float(row, "cost_bps_per_turnover"),
            "baseline_configuration_id": row["baseline_configuration_id"],
            "phase2_configuration_id": row["phase2_configuration_id"],
            "baseline_cumulative_return": _float(row, "baseline_cumulative_return"),
            "phase2_cumulative_return": _float(row, "phase2_cumulative_return"),
            "cumulative_return_change": _float(row, "cumulative_return_change"),
            "baseline_turnover": _float(row, "baseline_turnover"),
            "phase2_turnover": _float(row, "phase2_turnover"),
            "turnover_change_pct": _float(row, "turnover_change_pct"),
            "baseline_positive_folds": int(row["baseline_positive_folds"]),
            "phase2_positive_folds": int(row["phase2_positive_folds"]),
            "development_net_return_improved": _truth(
                row["development_net_return_improved"]
            ),
        }
        for row in day26_comparison
    ]
    phase2_inference_rows = [
        {
            "configuration_id": row["configuration_id"],
            "series": row["series"],
            "hac_t": _float(row, "hac_t_statistic"),
            "bootstrap_mean_ci_lower": _float(row, "bootstrap_mean_ci_lower"),
            "bootstrap_mean_ci_upper": _float(row, "bootstrap_mean_ci_upper"),
        }
        for row in day26_inference
    ]
    phase2_aggregate_rows = [
        {
            "configuration_id": row["configuration_id"],
            "series": row["series"],
            "cumulative_return": _float(row, "cumulative_return"),
            "gross_cumulative_return": _float(row, "gross_cumulative_return"),
            "turnover": _float(row, "turnover"),
            "trade_count": _float(row, "trade_count"),
            "break_even_cost_bps": _float(
                row, "approximate_break_even_cost_bps_per_turnover"
            ),
            "positive_folds": int(row["positive_folds"]),
        }
        for row in day26_aggregate
    ]

    model_labels = {
        "price_ratio_long_short_neutral": "Price-ratio long-short-neutral",
        "price_ratio_long_flat": "Price-ratio long-flat comparison",
        "ema_macd_long_short_neutral": "EMA/MACD long-short-neutral",
    }

    trend_rows: list[dict[str, object]] = []
    for row in day25_walk_forward:
        if (
            row["fold_id"] == "aggregate_2022_2025"
            or float(row["cost_bps_per_turnover"]) != 1.0
        ):
            continue
        trend_rows.append(
            {
                "fold": row["fold_id"].replace("wf_", ""),
                "strategy": model_labels[row["model_id"]],
                "strategy_id": row["model_id"],
                "annualized_return": _float(row, "annualized_return"),
                "cumulative_return": _float(row, "cumulative_return"),
                "sharpe_ratio": _float(row, "sharpe_ratio"),
                "maximum_drawdown": _float(row, "maximum_drawdown"),
                "turnover": _float(row, "turnover"),
                "trade_count": int(row["trade_count"]),
                "test_sessions": int(row["test_sessions"]),
                "test_start": row["test_start_timestamp"],
                "test_end": row["test_end_timestamp"],
                "timing_convention": "next_bar_open_overnight_flat_v1",
                "cost_bps_per_turnover": _float(row, "cost_bps_per_turnover"),
            }
        )

    cross_market_rows: list[dict[str, object]] = []
    for row in day25_robustness:
        cross_market_rows.append(
            {
                "case": f"{row['symbol']} {row['frequency']}",
                "symbol": row["symbol"],
                "frequency": row["frequency"],
                "strategy": model_labels[row["model_id"]],
                "strategy_id": row["model_id"],
                "annualized_return": _float(row, "annualized_return"),
                "sharpe_ratio": _float(row, "sharpe_ratio"),
                "maximum_drawdown": _float(row, "maximum_drawdown"),
                "turnover": _float(row, "turnover"),
                "average_exposure_pct": (
                    _float(row, "long_exposure") + _float(row, "short_exposure")
                ),
                "reference_case": "SPY 15min",
                "timing_convention": "next_bar_open_overnight_flat_v1",
            }
        )

    cost_rows: list[dict[str, object]] = []
    for row in day17_costs:
        if row["series"] != "equal_weight":
            continue
        cost_rows.append(
            {
                "cost": f"{int(float(row['cost_bps_per_turnover']))} bps",
                "cost_bps": _float(row, "cost_bps_per_turnover"),
                "configuration": row["configuration_id"]
                .replace("ou_vwap_", "")
                .title(),
                "configuration_id": row["configuration_id"],
                "cumulative_return": _float(row, "cumulative_return"),
                "annualized_return": _float(row, "annualized_return"),
                "sharpe_ratio": _float(row, "sharpe_ratio"),
                "maximum_drawdown": _float(row, "maximum_drawdown"),
                "test_sessions": int(row["test_sessions"]),
            }
        )

    final_aggregate = {
        row["model_id"]: row
        for row in day25_walk_forward
        if row["fold_id"] == "aggregate_2022_2025"
        and float(row["cost_bps_per_turnover"]) == 1.0
    }
    slow_aggregate = _select_one(
        day17, configuration_id="ou_vwap_slow", series="equal_weight"
    )
    slow_inference = _select_one(
        day17_inference, configuration_id="ou_vwap_slow", series="equal_weight"
    )
    diversification = day15[0]

    locked_by_model = {row["model_id"]: row for row in day25_locked_performance}
    locked_outcome_rows = [
        {
            "model_id": "ou_vwap_slow_equal_weight_locked",
            "model": "Slow OU/VWAP residual (locked final)",
            "evidence_class": "locked_final_test",
            "window": "2026-01-02 to 2026-06-30; 123 sessions; 1 bp",
            "cumulative_return": _float(
                locked_by_model["ou_vwap_slow_equal_weight"], "cumulative_return"
            ),
            "sharpe_ratio": _float(
                locked_by_model["ou_vwap_slow_equal_weight"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                locked_by_model["ou_vwap_slow_equal_weight"], "maximum_drawdown"
            ),
            "decision": "positive but small; does not establish deployable profitability",
            "source_path": SOURCE_FILES["day25_locked_performance"],
        },
        {
            "model_id": "price_ratio_long_short_neutral_locked",
            "model": "Price-ratio long-short-neutral (locked final)",
            "evidence_class": "locked_final_test",
            "window": "2026-01-02 to 2026-06-30; 3,198 bars; 1 bp",
            "cumulative_return": _float(
                locked_by_model["price_ratio_long_short_neutral"],
                "cumulative_return",
            ),
            "sharpe_ratio": _float(
                locked_by_model["price_ratio_long_short_neutral"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                locked_by_model["price_ratio_long_short_neutral"],
                "maximum_drawdown",
            ),
            "decision": "failed the one-time locked final test",
            "source_path": SOURCE_FILES["day25_locked_performance"],
        },
        {
            "model_id": "ema_macd_long_short_neutral_locked",
            "model": "EMA/MACD long-short-neutral (locked final)",
            "evidence_class": "locked_final_test",
            "window": "2026-01-02 to 2026-06-30; 3,198 bars; 1 bp",
            "cumulative_return": _float(
                locked_by_model["ema_macd_long_short_neutral"], "cumulative_return"
            ),
            "sharpe_ratio": _float(
                locked_by_model["ema_macd_long_short_neutral"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                locked_by_model["ema_macd_long_short_neutral"], "maximum_drawdown"
            ),
            "decision": "failed the one-time locked final test",
            "source_path": SOURCE_FILES["day25_locked_performance"],
        },
    ]

    development_outcome_rows = [
        {
            "model_id": "price_ratio_long_short_neutral_development",
            "model": "Price-ratio trend (long-short-neutral)",
            "evidence_class": "chronological_out_of_sample",
            "window": "2022-2025; 1,003 sessions; 1 bp; causal next-open",
            "cumulative_return": _float(
                final_aggregate["price_ratio_long_short_neutral"], "cumulative_return"
            ),
            "sharpe_ratio": _float(
                final_aggregate["price_ratio_long_short_neutral"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                final_aggregate["price_ratio_long_short_neutral"], "maximum_drawdown"
            ),
            "decision": "failed under final causal convention",
            "source_path": SOURCE_FILES["day25_trend_walk_forward"],
        },
        {
            "model_id": "price_ratio_long_flat_development",
            "model": "Price-ratio trend (long-flat comparison)",
            "evidence_class": "chronological_out_of_sample",
            "window": "2022-2025; 1,003 sessions; 1 bp; causal next-open",
            "cumulative_return": _float(
                final_aggregate["price_ratio_long_flat"], "cumulative_return"
            ),
            "sharpe_ratio": _float(
                final_aggregate["price_ratio_long_flat"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                final_aggregate["price_ratio_long_flat"], "maximum_drawdown"
            ),
            "decision": "near flat but negative; not a profitable rescue",
            "source_path": SOURCE_FILES["day25_trend_walk_forward"],
        },
        {
            "model_id": "ema_macd_long_short_neutral_development",
            "model": "EMA/MACD trend",
            "evidence_class": "chronological_out_of_sample",
            "window": "2022-2025; 1,003 sessions; 1 bp; causal next-open",
            "cumulative_return": _float(
                final_aggregate["ema_macd_long_short_neutral"], "cumulative_return"
            ),
            "sharpe_ratio": _float(
                final_aggregate["ema_macd_long_short_neutral"], "sharpe_ratio"
            ),
            "maximum_drawdown": _float(
                final_aggregate["ema_macd_long_short_neutral"], "maximum_drawdown"
            ),
            "decision": "failed under final causal convention",
            "source_path": SOURCE_FILES["day25_trend_walk_forward"],
        },
        {
            "model_id": "cointegrated_etf_pairs_development",
            "model": "Cointegrated ETF pairs",
            "evidence_class": "historical_development",
            "window": "3 candidate pairs",
            "cumulative_return": None,
            "sharpe_ratio": None,
            "maximum_drawdown": None,
            "decision": "0 eligible; no backtest promoted",
            "source_path": SOURCE_FILES["day14_pairs"],
        },
        {
            "model_id": "ou_vwap_slow_equal_weight_development",
            "model": "Slow OU/VWAP residual",
            "evidence_class": "chronological_out_of_sample",
            "window": "2022-2025; 1,003 sessions; 1 bps",
            "cumulative_return": _float(slow_aggregate, "cumulative_return"),
            "sharpe_ratio": _float(slow_aggregate, "sharpe_ratio"),
            "maximum_drawdown": _float(slow_aggregate, "maximum_drawdown"),
            "decision": "positive but statistically inconclusive; unpromoted",
            "source_path": SOURCE_FILES["day17_reversion"],
        },
    ]
    outcome_rows = locked_outcome_rows + development_outcome_rows

    pair_rows = [
        {
            "pair": row["pair_id"].replace("_", "/"),
            "holm_cointegration_pass": _truth(row["holm_cointegration_pass"]),
            "stable_beta_pass": _truth(row["fold_beta_stability_pass"]),
            "stationary_fold_count": int(row["stationary_fold_count"]),
            "eligible": _truth(row["eligible"]),
            "rejection_reasons": row["rejection_reasons"].replace("|", "; "),
        }
        for row in day14
    ]

    allocation_rows = [
        {
            "allocation_rule": row["allocation_rule"].replace("_", " ").title(),
            "observations": int(row["observations"]),
            "cumulative_return": _float(row, "cumulative_return"),
            "sharpe_ratio": _float(row, "sharpe_ratio"),
            "maximum_drawdown": _float(row, "maximum_drawdown"),
            "historical_var_95": _float(row, "historical_var_95"),
            "historical_es_95": _float(row, "historical_es_95"),
        }
        for row in day16
    ]

    inference_rows = [
        {
            "model": "Slow OU/VWAP equal weight",
            "observations": int(slow_inference["observations"]),
            "mean_session_return": _float(slow_inference, "mean_session_return"),
            "naive_t": _float(slow_inference, "naive_t_statistic"),
            "hac_lags": int(slow_inference["hac_lags"]),
            "hac_t": _float(slow_inference, "hac_t_statistic"),
            "bootstrap_mean_ci_lower": _float(
                slow_inference, "bootstrap_mean_ci_lower"
            ),
            "bootstrap_mean_ci_upper": _float(
                slow_inference, "bootstrap_mean_ci_upper"
            ),
            "bootstrap_sharpe_ci_lower": _float(
                slow_inference, "bootstrap_sharpe_ci_lower"
            ),
            "bootstrap_sharpe_ci_upper": _float(
                slow_inference, "bootstrap_sharpe_ci_upper"
            ),
            "information_coefficient": _float(
                slow_inference, "information_coefficient"
            ),
            "probabilistic_sharpe_probability": _float(
                slow_inference, "probabilistic_sharpe_probability"
            ),
            "deflated_sharpe_probability": _float(
                slow_inference, "deflated_sharpe_probability"
            ),
            "declared_trials": int(slow_inference["declared_trials"]),
        }
    ]

    timing_rows = [
        {
            "model": model_labels[row["model_id"]],
            "timing_convention": row["timing_convention"],
            "cumulative_return": _float(row, "cumulative_return"),
            "sharpe_ratio": _float(row, "sharpe_ratio"),
            "maximum_drawdown": _float(row, "maximum_drawdown"),
            "turnover": _float(row, "turnover"),
            "overnight_positions_held": int(row["overnight_positions_held"]),
        }
        for row in day25_timing
    ]
    event_indicator_by_method = {
        row["sampling_method"]: row for row in day25_event_indicator
    }
    event_rows = [
        {
            "sampling_method": row["sampling_method"],
            "sessions": int(row["sessions"]),
            "bars": int(row["bars"]),
            "duration_cv": _float(row, "duration_cv"),
            "trade_count_cv": _float(row, "trade_count_cv"),
            "volume_cv": _float(row, "volume_cv"),
            "dollar_value_cv": _float(row, "dollar_value_cv"),
            "signal_observations": int(
                event_indicator_by_method[row["sampling_method"]][
                    "signal_available_observations"
                ]
            ),
            "pearson_forward_association": _float(
                event_indicator_by_method[row["sampling_method"]],
                "pearson_signal_forward_return",
            ),
            "spearman_forward_association": _float(
                event_indicator_by_method[row["sampling_method"]],
                "spearman_signal_forward_return",
            ),
        }
        for row in day25_event_sampling
    ]
    if any(
        int(row["trade_count_error"]) != 0
        or abs(_float(row, "volume_error")) > 1.0e-8
        or abs(_float(row, "dollar_value_error"))
        > max(1.0e-8, abs(_float(row, "input_dollar_value")) * 1.0e-12)
        for row in day25_event_conservation
    ):
        raise ValueError("Day 25 event-time conservation evidence does not pass.")
    if not all(_truth(row["parity_passed"]) for row in day25_parity):
        raise ValueError("Day 25 final trend replay parity does not pass.")

    method_rows = [
        {
            "method": "Next-bar-open forced-flat execution",
            "implementation": "project",
            "purpose": "causal trend return attribution",
            "limitation": "bar open/close remain price proxies, not guaranteed fills",
        },
        {
            "method": "Expanding walk-forward folds",
            "implementation": "project",
            "purpose": "chronological out-of-sample testing",
            "limitation": "four annual folds only",
        },
        {
            "method": "Batch/sequential replay parity",
            "implementation": "project",
            "purpose": "position, forced-exit, turnover, cost, and return equivalence",
            "limitation": "broker queue and spread dynamics remain abstracted",
        },
        {
            "method": "Time versus dollar-bar experiment",
            "implementation": "project",
            "purpose": "compare sampling on 77,053 genuine trades",
            "limitation": "five sessions and 55 available indicator observations per method",
        },
        {
            "method": "Engle-Granger plus Holm control",
            "implementation": "statsmodels + project adjustment",
            "purpose": "pair feasibility with family-wise control",
            "limitation": "low power and regime sensitivity",
        },
        {
            "method": "AR(1) OU half-life",
            "implementation": "project",
            "purpose": "causal reversion-speed filter",
            "limitation": "discrete approximation and rolling estimation noise",
        },
        {
            "method": "Variance-ratio filter",
            "implementation": "project",
            "purpose": "reject trend-like transformed residuals",
            "limitation": "threshold is model-dependent",
        },
        {
            "method": "Sample covariance (ddof=1)",
            "implementation": "NumPy/project",
            "purpose": "six-sleeve dependence estimation",
            "limitation": "unstable in short samples; no shrinkage",
        },
        {
            "method": "Constrained minimum variance",
            "implementation": "CVXPY",
            "purpose": "long-only allocation comparison",
            "limitation": "no expected returns; estimation error persists",
        },
        {
            "method": "Historical VaR and ES",
            "implementation": "project",
            "purpose": "tail-loss summaries",
            "limitation": "non-parametric history may miss new regimes",
        },
        {
            "method": "Newey-West/HAC t statistic",
            "implementation": "statsmodels",
            "purpose": "serial-correlation-aware mean test",
            "limitation": "lag 5 is predeclared, not universally optimal",
        },
        {
            "method": "Circular block bootstrap",
            "implementation": "project",
            "purpose": "mean and Sharpe uncertainty",
            "limitation": "2,000 resamples; block length 5",
        },
        {
            "method": "Probabilistic/deflated Sharpe",
            "implementation": "project formulas",
            "purpose": "non-normality and selection correction",
            "limitation": "depends on declared trial count",
        },
        {
            "method": "One-time locked final test",
            "implementation": "project fail-closed runner",
            "purpose": "evaluate three frozen configurations on January-June 2026 without retuning",
            "limitation": "one 123-session interval cannot establish regime-general profitability",
        },
        {
            "method": "Predeclared Phase II development test",
            "implementation": "project deterministic runner",
            "purpose": "test one trend turnover control and one OU cost-margin gate against retained baselines",
            "limitation": "the consumed 2026 interval is prohibited and no untouched later holdout is locally available",
        },
        {
            "method": "Idempotent order state machine",
            "implementation": "project",
            "purpose": "partial, duplicate, stale, and rejected events",
            "limitation": "synthetic scenarios do not prove provider uptime",
        },
        {
            "method": "Broker/local reconciliation",
            "implementation": "project",
            "purpose": "orders, fills, positions, cash, stream health",
            "limitation": "live mutation evidence remains sparse",
        },
        {
            "method": "SHA-256 artifact manifests",
            "implementation": "project",
            "purpose": "deterministic provenance and replay",
            "limitation": "integrity is not independent replication",
        },
        {
            "method": "Locked environment and CI",
            "implementation": "pip-tools/Docker/GitHub Actions",
            "purpose": "reproducible startup and testing",
            "limitation": "local container runtime unavailable on Day 23",
        },
    ]

    passed_day19 = sum(_truth(row["scenario_passed"]) for row in day19)
    passed_day20 = sum(_truth(row["scenario_passed"]) for row in day20)
    passed_day21 = sum(_truth(row["passed"]) for row in day21)
    passed_day23 = sum(_truth(row["passed"]) for row in day23)
    operational_rows = [
        {
            "control": "Alpaca paper preflight",
            "evidence_class": "live_read_only",
            "result": "passed",
            "denominator": "account, clock, and 3 assets",
            "mutation": "none",
            "source_path": SOURCE_FILES["day18_preflight"],
        },
        {
            "control": "Order-state scenarios",
            "evidence_class": "synthetic_known_answer",
            "result": f"{passed_day19}/{len(day19)} passed",
            "denominator": f"{len(day19)} scenarios",
            "mutation": "none",
            "source_path": SOURCE_FILES["day19_order_state"],
        },
        {
            "control": "Reconciliation/stream scenarios",
            "evidence_class": "synthetic_known_answer",
            "result": f"{passed_day20}/{len(day20)} passed",
            "denominator": f"{len(day20)} scenarios",
            "mutation": "none",
            "source_path": SOURCE_FILES["day20_reconciliation"],
        },
        {
            "control": "Controlled-paper live gate probe",
            "evidence_class": "live_read_only",
            "result": f"{passed_day21}/{len(day21)} gates passed; aborted safely",
            "denominator": f"{len(day21)} gates",
            "mutation": "no order",
            "source_path": SOURCE_FILES["day21_read_only"],
        },
        {
            "control": "Offline operations health",
            "evidence_class": "operational_reproducibility",
            "result": f"{passed_day23}/{len(day23)} passed",
            "denominator": f"{len(day23)} checks",
            "mutation": "none",
            "source_path": SOURCE_FILES["day23_operations"],
        },
    ]

    execution_rows = [
        {
            "purpose": row["purpose"].replace("_", " ").title(),
            "evidence_class": "synthetic_known_answer",
            "alpha_eligible": _truth(row["alpha_eligible"]),
            "executions": int(row["executions"]),
            "round_trips": int(row["round_trips"]),
            "filled_quantity": _float(row, "filled_quantity"),
            "mean_shortfall_bps": _float(row, "mean_total_shortfall_bps"),
            "median_fill_latency_ms": _float(row, "median_fill_latency_ms"),
            "net_pnl": _float(row, "net_pnl"),
            "interpretation": "arithmetic fixture; not empirical profitability",
        }
        for row in day22
    ]

    summary = [
        {
            "locked_slow_reversion_return": locked_outcome_rows[0]["cumulative_return"],
            "locked_trend_ratio_return": locked_outcome_rows[1]["cumulative_return"],
            "locked_ema_macd_return": locked_outcome_rows[2]["cumulative_return"],
            "long_flat_return": development_outcome_rows[1]["cumulative_return"],
            "slow_reversion_return": development_outcome_rows[4]["cumulative_return"],
            "eligible_pairs": sum(row["eligible"] for row in pair_rows),
            "ensemble_diversification_ratio": _float(
                diversification, "median_test_equal_weight_diversification_ratio"
            ),
            "paper_preflight_passed": bool(day18["preflight_passed"]),
        }
    ]

    return {
        "summary": summary,
        "trend_walk_forward": trend_rows,
        "cross_market": cross_market_rows,
        "reversion_costs": cost_rows,
        "model_outcomes": outcome_rows,
        "pair_feasibility": pair_rows,
        "allocation": allocation_rows,
        "inference": inference_rows,
        "trend_timing": timing_rows,
        "event_time": event_rows,
        "methods": method_rows,
        "operations": operational_rows,
        "execution": execution_rows,
        "phase2_comparison": phase2_rows,
        "phase2_inference": phase2_inference_rows,
        "phase2_aggregate": phase2_aggregate_rows,
        "annual_trend_comparison": [
            row for row in annual_rows if row["comparison_group"] == "trend"
        ],
        "annual_reversion_comparison": [
            row for row in annual_rows if row["comparison_group"] == "reversion"
        ],
        "annual_short_effect": short_effect_rows,
        "annual_short_effect_chart": short_effect_chart_rows,
        "annual_ou_concentration": ou_concentration_rows,
    }


def build_report_artifact(project_root: str | Path) -> dict[str, object]:
    """Return the validated canonical portable-report artifact."""

    root = Path(project_root)
    verify_source_manifests(root)
    datasets = _load_evidence(root)
    sources = [_source(root, source_id) for source_id in SOURCE_FILES]

    cards = [
        {
            "id": "slow_reversion_headline",
            "description": "Frozen slow equal-weight OU/VWAP result in the one-time 2026 locked final test: 123 sessions at one basis point per turnover.",
            "dataset": "summary",
            "sourceId": "day25_locked_performance",
            "metrics": [
                {
                    "label": "Locked slow OU/VWAP cumulative return",
                    "field": "locked_slow_reversion_return",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "trend_ratio_headline",
            "description": "Frozen long-short-neutral SPY 15-minute price-ratio trend in the one-time 2026 locked final test under causal next-open, overnight-flat accounting and one-basis-point turnover cost.",
            "dataset": "summary",
            "sourceId": "day25_locked_performance",
            "metrics": [
                {
                    "label": "Locked price-ratio cumulative return",
                    "field": "locked_trend_ratio_return",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "long_flat_headline",
            "description": "Originally intended long-flat price-ratio comparison under the same final causal convention and cost; reported without replacing the historical lineage.",
            "dataset": "summary",
            "sourceId": "day25_trend_walk_forward",
            "metrics": [
                {
                    "label": "Long-flat comparison cumulative return",
                    "field": "long_flat_return",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "ema_macd_headline",
            "description": "Frozen SPY 15-minute EMA/MACD trend in the one-time 2026 locked final test under causal next-open, overnight-flat accounting and one-basis-point turnover cost.",
            "dataset": "summary",
            "sourceId": "day25_locked_performance",
            "metrics": [
                {
                    "label": "Locked EMA/MACD cumulative return",
                    "field": "locked_ema_macd_return",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "eligible_pairs_headline",
            "description": "Predeclared ETF pairs that passed every Day 14 feasibility gate.",
            "dataset": "summary",
            "sourceId": "day14_pairs",
            "metrics": [
                {
                    "label": "Eligible cointegrated pairs",
                    "field": "eligible_pairs",
                    "format": "number",
                }
            ],
        },
    ]

    charts = [
        _comparison_chart(
            chart_id="trend_walk_forward_returns",
            title="Annual trend walk-forward returns",
            subtitle="SPY 15-minute annualized net return under causal next-open, overnight-flat accounting; one-basis-point cost and annual tests, 2022-2025.",
            question="Did either trend family or the required long-flat comparison deliver consistently positive annual out-of-sample returns?",
            rationale="The twelve predeclared fold results expose sign instability and keep the less-negative long-flat comparison from being mistaken for a validated winner.",
            dataset="trend_walk_forward",
            source_id="day25_trend_walk_forward",
            x_field="fold",
            x_label="Test year",
            color_field="strategy",
            color_label="Frozen trend model",
            y_field="annualized_return",
            y_label="Annualized net return",
        ),
        _comparison_chart(
            chart_id="cross_market_annualized_returns",
            title="Cross-market and bar-frequency trend returns",
            subtitle="Annualized net return for three fixed trend configurations across SPY, QQQ, and IWM at 15-, 30-, and 60-minute bars; causal 2020-2025 development accounting.",
            question="Were the trend conclusions robust across markets and bar frequencies?",
            rationale="The twenty-seven-case matrix shows whether conclusions survive positioning, market, and sampling changes rather than one reference backtest.",
            dataset="cross_market",
            source_id="day25_trend_robustness",
            x_field="case",
            x_label="Symbol and bar frequency",
            color_field="strategy",
            color_label="Frozen trend model",
            y_field="annualized_return",
            y_label="Annualized net return",
        ),
        _comparison_chart(
            chart_id="reversion_cost_sensitivity",
            title="OU/VWAP cost sensitivity",
            subtitle="Equal-weight SPY/QQQ/IWM cumulative return over 1,003 walk-forward sessions, 2022-2025.",
            question="How sensitive were the three reversion calibrations to turnover costs?",
            rationale="Cost-by-configuration bars distinguish a low-turnover positive result from fast and base variants whose losses worsen mechanically with costs.",
            dataset="reversion_costs",
            source_id="day17_costs",
            x_field="cost",
            x_label="One-way cost per turnover",
            color_field="configuration",
            color_label="OU/VWAP calibration",
            y_field="cumulative_return",
            y_label="Cumulative net return",
        ),
        _comparison_chart(
            chart_id="annual_trend_vs_spy",
            title="Annual trend returns versus SPY price-only buy-and-hold",
            subtitle="Complete calendar-year folds, 2022-2025; model returns are net of one basis point per unit turnover and the benchmark omits dividends.",
            question="Was relative trend performance broad across years or concentrated in one regime?",
            rationale="The grouped bars compare like-for-like calendar-year endpoints and show whether apparent outperformance repeats beyond 2022.",
            dataset="annual_trend_comparison",
            source_id="annual_regime_comparison",
            x_field="year",
            x_label="Calendar year",
            color_field="series",
            color_label="Model or benchmark",
            y_field="net_return",
            y_label="Calendar-year return",
        ),
        _comparison_chart(
            chart_id="annual_ou_vs_equal_weight",
            title="Annual slow OU/VWAP versus equal-weight price-only buy-and-hold",
            subtitle="Equal starting capital in SPY, QQQ, and IWM; complete calendar-year folds, 2022-2025; benchmark dividends and idle-cash yield are omitted.",
            question="Did the slow OU/VWAP advantage persist across rising and falling benchmark years?",
            rationale="The annual comparison separates the sparse reversion return from the market direction of the matching three-ETF basket.",
            dataset="annual_reversion_comparison",
            source_id="annual_regime_comparison",
            x_field="year",
            x_label="Calendar year",
            color_field="series",
            color_label="Model or benchmark",
            y_field="net_return",
            y_label="Calendar-year return",
        ),
        _comparison_chart(
            chart_id="annual_short_sleeve_effect",
            title="Incremental annual return effect of price-ratio short exposure",
            subtitle="Long-short-neutral minus the matching long-flat result, 2022-2025; positive values mean that the short sleeve helped.",
            question="Did allowing short positions improve the price-ratio rule consistently?",
            rationale="Gross and net differences isolate the direction and cost burden of the short sleeve without relabeling long-flat as a new selected strategy.",
            dataset="annual_short_effect_chart",
            source_id="annual_short_effect",
            x_field="year",
            x_label="Calendar year",
            color_field="effect_type",
            color_label="Return effect",
            y_field="return_effect",
            y_label="Incremental return",
        ),
    ]

    tables = [
        _table(
            "model_outcomes",
            "Development and locked model outcomes",
            "The one-time locked results appear first; every negative, rejected, and inconclusive development result remains visible and no model is promoted.",
            "model_outcomes",
            "day24_specification",
            [
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "evidence_class", "label": "Evidence class", "type": "text"},
                {"field": "window", "label": "Window / denominator", "type": "text"},
                {
                    "field": "cumulative_return",
                    "label": "Cumulative return",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "sharpe_ratio",
                    "label": "Sharpe",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "maximum_drawdown",
                    "label": "Max drawdown",
                    "format": "percent",
                    "movement": True,
                },
                {"field": "decision", "label": "Frozen decision", "type": "text"},
                {"field": "source_path", "label": "Evidence", "type": "text"},
            ],
        ),
        _table(
            "trend_timing",
            "Saved versus causal trend accounting",
            "The historical convention is preserved beside the next-open, overnight-flat final development convention; this is a methodological sensitivity, not a winner search.",
            "trend_timing",
            "day25_trend_timing",
            [
                {"field": "model", "label": "Model", "type": "text"},
                {
                    "field": "timing_convention",
                    "label": "Timing convention",
                    "type": "text",
                },
                {
                    "field": "cumulative_return",
                    "label": "Cumulative return",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "sharpe_ratio",
                    "label": "Sharpe",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "maximum_drawdown",
                    "label": "Max drawdown",
                    "format": "percent",
                    "movement": True,
                },
                {"field": "turnover", "label": "Turnover", "format": "number"},
                {
                    "field": "overnight_positions_held",
                    "label": "Overnight positions",
                    "format": "number",
                },
            ],
        ),
        _table(
            "event_time",
            "Representative time-bar versus dollar-bar evidence",
            "77,053 SPY IEX trades across five predeclared complete 2025 sessions; all trade count, volume, and notional reconcile.",
            "event_time",
            "day25_event_sampling",
            [
                {
                    "field": "sampling_method",
                    "label": "Sampling method",
                    "type": "text",
                },
                {"field": "sessions", "label": "Sessions", "format": "number"},
                {"field": "bars", "label": "Bars", "format": "number"},
                {"field": "duration_cv", "label": "Duration CV", "format": "number"},
                {
                    "field": "trade_count_cv",
                    "label": "Trade-count CV",
                    "format": "number",
                },
                {"field": "volume_cv", "label": "Volume CV", "format": "number"},
                {
                    "field": "dollar_value_cv",
                    "label": "Dollar-value CV",
                    "format": "number",
                },
                {
                    "field": "signal_observations",
                    "label": "Indicator observations",
                    "format": "number",
                },
                {
                    "field": "pearson_forward_association",
                    "label": "Pearson next-event association",
                    "format": "number",
                },
                {
                    "field": "spearman_forward_association",
                    "label": "Spearman next-event association",
                    "format": "number",
                },
            ],
        ),
        _table(
            "pair_feasibility",
            "Cointegration feasibility was rejected before strategy backtesting",
            "All three predeclared ETF pairs failed the Holm-controlled cointegration gate.",
            "pair_feasibility",
            "day14_pairs",
            [
                {"field": "pair", "label": "Pair", "type": "text"},
                {
                    "field": "holm_cointegration_pass",
                    "label": "Holm coint. pass",
                    "type": "boolean",
                },
                {
                    "field": "stable_beta_pass",
                    "label": "Stable beta",
                    "type": "boolean",
                },
                {
                    "field": "stationary_fold_count",
                    "label": "Stationary folds",
                    "format": "number",
                },
                {"field": "eligible", "label": "Eligible", "type": "boolean"},
                {"field": "rejection_reasons", "label": "Reasons", "type": "text"},
            ],
        ),
        _table(
            "slow_inference",
            "Slow OU/VWAP statistical diagnostics",
            "The confidence intervals cross zero and the deflated Sharpe probability is not strong enough to promote the result.",
            "inference",
            "day17_inference",
            [
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "observations", "label": "Sessions", "format": "number"},
                {
                    "field": "mean_session_return",
                    "label": "Mean session return",
                    "format": "percent",
                    "movement": True,
                },
                {"field": "naive_t", "label": "Naive t", "format": "number"},
                {"field": "hac_t", "label": "HAC t (lag 5)", "format": "number"},
                {
                    "field": "bootstrap_mean_ci_lower",
                    "label": "Mean CI low",
                    "format": "percent",
                },
                {
                    "field": "bootstrap_mean_ci_upper",
                    "label": "Mean CI high",
                    "format": "percent",
                },
                {"field": "information_coefficient", "label": "IC", "format": "number"},
                {
                    "field": "probabilistic_sharpe_probability",
                    "label": "PSR probability",
                    "format": "percent",
                },
                {
                    "field": "deflated_sharpe_probability",
                    "label": "DSR probability",
                    "format": "percent",
                },
            ],
        ),
        _table(
            "annual_ou_concentration",
            "Annual slow OU/VWAP return concentration",
            "Post-hoc development diagnostic; non-zero portfolio days and top/worst three-day arithmetic contributions are shown to prevent a sparse annual gain from being mistaken for a broad daily edge.",
            "annual_ou_concentration",
            "annual_ou_concentration",
            [
                {"field": "year", "label": "Year", "format": "number"},
                {
                    "field": "net_cumulative_return",
                    "label": "Net return",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "nonzero_portfolio_days",
                    "label": "Non-zero days",
                    "format": "number",
                },
                {"field": "positive_days", "label": "Positive", "format": "number"},
                {"field": "negative_days", "label": "Negative", "format": "number"},
                {
                    "field": "top_three_day_sum",
                    "label": "Top 3 sum",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "worst_three_day_sum",
                    "label": "Worst 3 sum",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "flat_exposure_pct",
                    "label": "Flat exposure",
                    "format": "number",
                },
            ],
            default_sort=("year", "asc"),
        ),
        _table(
            "allocation",
            "Portfolio allocation did not rescue the six-sleeve research set",
            "All three long-only allocation rules lost money over the same 1,003 walk-forward sessions.",
            "allocation",
            "day16_allocation",
            [
                {"field": "allocation_rule", "label": "Rule", "type": "text"},
                {"field": "observations", "label": "Sessions", "format": "number"},
                {
                    "field": "cumulative_return",
                    "label": "Cumulative return",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "sharpe_ratio",
                    "label": "Sharpe",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "maximum_drawdown",
                    "label": "Max drawdown",
                    "format": "percent",
                    "movement": True,
                },
                {"field": "historical_var_95", "label": "95% VaR", "format": "percent"},
                {"field": "historical_es_95", "label": "95% ES", "format": "percent"},
            ],
        ),
        _table(
            "methods",
            "Numerical and statistical methods audit",
            "The implementation boundary and a material limitation are stated for every major technique.",
            "methods",
            "day24_specification",
            [
                {"field": "method", "label": "Method", "type": "text"},
                {
                    "field": "implementation",
                    "label": "Implemented / adjusted / library",
                    "type": "text",
                },
                {"field": "purpose", "label": "Purpose", "type": "text"},
                {
                    "field": "limitation",
                    "label": "Numerical limitation",
                    "type": "text",
                },
            ],
        ),
        _table(
            "operations",
            "Operational readiness evidence",
            "Live read-only evidence is separated from synthetic failure-handling validation.",
            "operations",
            "day24_specification",
            [
                {"field": "control", "label": "Control", "type": "text"},
                {"field": "evidence_class", "label": "Evidence class", "type": "text"},
                {"field": "result", "label": "Result", "type": "text"},
                {"field": "denominator", "label": "Denominator", "type": "text"},
                {"field": "mutation", "label": "Broker mutation", "type": "text"},
                {"field": "source_path", "label": "Evidence", "type": "text"},
            ],
        ),
        _table(
            "execution",
            "Execution benchmarking fixture",
            "These rows validate calculations and evidence separation; they are not empirical live fills.",
            "execution",
            "day22_synthetic",
            [
                {"field": "purpose", "label": "Purpose", "type": "text"},
                {"field": "evidence_class", "label": "Evidence class", "type": "text"},
                {
                    "field": "alpha_eligible",
                    "label": "Alpha eligible",
                    "type": "boolean",
                },
                {"field": "executions", "label": "Executions", "format": "number"},
                {"field": "round_trips", "label": "Round trips", "format": "number"},
                {
                    "field": "mean_shortfall_bps",
                    "label": "Mean shortfall (bps)",
                    "format": "number",
                },
                {
                    "field": "median_fill_latency_ms",
                    "label": "Median latency (ms)",
                    "format": "number",
                },
                {
                    "field": "net_pnl",
                    "label": "Net P&L (USD)",
                    "format": "currency",
                    "movement": True,
                },
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        ),
    ]

    phase2_comparison_by_key = {
        (row["strategy_family"], row["cost_bps"]): row
        for row in datasets["phase2_comparison"]
    }
    phase2_aggregate_by_id = {
        (row["configuration_id"], row["series"]): row
        for row in datasets["phase2_aggregate"]
    }
    phase2_inference_by_id = {
        (row["configuration_id"], row["series"]): row
        for row in datasets["phase2_inference"]
    }
    phase2_trend_1bp = phase2_comparison_by_key[("price_ratio_long_flat", 1.0)]
    phase2_trend_5bp = phase2_comparison_by_key[("price_ratio_long_flat", 5.0)]
    phase2_ou_1bp = phase2_comparison_by_key[("ou_vwap_slow", 1.0)]
    phase2_trend_baseline = phase2_aggregate_by_id[
        ("price_ratio_long_flat_baseline", "SPY")
    ]
    phase2_trend_candidate = phase2_aggregate_by_id[
        ("price_ratio_persistent_hysteresis_phase2", "SPY")
    ]
    phase2_ou_candidate_inference = phase2_inference_by_id[
        ("ou_vwap_slow_cost_margin_phase2", "equal_weight")
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Axiom: Trend-Following, Reversion, and Fail-Closed Paper Execution\n\n**Final CQF Algorithmic Trading technical report.**\n\nI developed and tested the system on 2020-2025 data. I then ran the separately authorized 2026-01-02 through 2026-06-30 final test once under the frozen protocol and report every result without retuning.",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": f"## 2. Technical summary\n\nI built Axiom as a reproducible quantitative research and paper-execution platform. I separated development from the one-time final test, enforced causal timing and chronological folds, charged turnover costs, measured statistical uncertainty, produced deterministic evidence bundles, and designed broker controls to fail closed. In the one-time 2026 locked final test at one basis point per turnover, I found that slow OU/VWAP returned **+0.43%** with Sharpe **0.572** and maximum drawdown **-0.70%**; price-ratio and EMA/MACD trend returned **-1.90%** and **-4.94%**. I regard the positive reversion result as economically small, not evidence of deployable profitability. In development, slow OU/VWAP returned **+6.03%**, but its bootstrap interval crossed zero and its deflated Sharpe probability was only **62.58%**; the required long-flat comparison returned **-0.45%**, and all three ETF pairs failed the predeclared cointegration gate. I also ran a separately frozen Phase II development test. Persistent hysteresis changed long-flat return from **{100.0 * phase2_trend_1bp['baseline_cumulative_return']:+.2f}%** to **{100.0 * phase2_trend_1bp['phase2_cumulative_return']:+.2f}%**, while the OU cost-margin gate was non-binding and left return at **{100.0 * phase2_ou_1bp['phase2_cumulative_return']:+.2f}%**. I retained every negative and rejected result because my main contribution is the auditable evidence chain, not a manufactured winner. The report remains partial only because empirical paper-fill evidence is absent.",
        },
        {
            "id": "report_contribution",
            "type": "markdown",
            "sourceId": "day24_specification",
            "body": """### 2.1 Purpose, contribution, and interpretation

I do not present this project as a profitable trading product. I present it as a production-style research process that can formulate, implement, falsify, and operationally constrain short-horizon trading hypotheses. That distinction matters because leakage, favorable accounting, parameter search, omitted costs, or selective reporting can manufacture a positive backtest. I therefore made the research protocol observable: I separated development from the one-time locked interval, lagged signals, reset execution state at evaluation boundaries, charged transaction costs on absolute turnover, used chronological annual folds, retained competing model families, reported statistical uncertainty, and made broker-facing actions fail closed.

I addressed the Algorithmic Trading brief through two trend families and one non-trivial reversion family. I related the price-ratio model to moving-average trend rules studied in the technical-trading literature [2], used EMA/MACD as an independent recursive-filter formulation, and treated evidence of time-series persistence in other markets and horizons as motivation rather than proof [3]. For reversion, I first tested cointegrated-pair feasibility and then implemented an independently tradeable transformed residual with rolling OU diagnostics and a variance-ratio regime filter [4]-[7]. I therefore cover both the model mathematics and the engineering required to stop that mathematics from being interpreted too generously.

I reached an intentionally asymmetric conclusion. Slow OU/VWAP is the strongest economic observation because it was positive in development and remained positive in the locked test. I still do not call it validated: the development bootstrap intervals cross zero, DSR is only 62.58%, the locked gain is 0.43%, the locked interval contains 123 sessions, and empirical fill/slippage evidence is absent. Both trend strategies failed in the locked interval. I preserve those facts because they define what my evidence can and cannot support.
""".strip(),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "slow_reversion_headline",
                "trend_ratio_headline",
                "long_flat_headline",
                "ema_macd_headline",
                "eligible_pairs_headline",
            ],
        },
        {
            "id": "key_results_heading",
            "type": "markdown",
            "body": "## 3. Key economic results\n\nI present the immutable one-time locked results first, with the positive-but-small reversion observation before both failed trend tests. I then show the development results without deletion or relabeling. I found only four positive cases in the 27-case causal robustness matrix and no stable pattern across model, symbol, or frequency. Because the markets and dates overlap, I treat that matrix as robustness evidence rather than additional independent out-of-sample proof.",
        },
        {
            "id": "model_outcomes_block",
            "type": "table",
            "tableId": "model_outcomes",
            "layout": "full",
        },
        {
            "id": "locked_result_analysis",
            "type": "markdown",
            "sourceId": "day25_locked_performance",
            "body": """### 3.1 Reading the final-test evidence

I treated the locked test as a single confirmation exercise, not a fifth walk-forward fold or a new calibration sample. Before I requested any locked row, I froze the model set, parameters, universe, cost, timing convention, aggregation rule, and reporting order. I used development history only to warm recursive and rolling states, reset positions and holding state at the first locked observation, and evaluated all three models. The runner wrote an immutable evidence bundle and exposed no ranking or promotion pathway.

The slow OU/VWAP equal-weight series compounded to **+0.4311%**. Its annualized return was **+0.8852%**, annualized volatility **1.5624%**, Sharpe **0.5718**, and maximum drawdown **-0.7034%**. Mean turnover across the three symbol sleeves was **10.6667** units and the strategy was flat for **97.27%** of eligible observations. These values describe a small, low-activity result. They do not imply that the strategy clears realistic capacity, spread, latency, or statistical-significance gates.

The price-ratio long-short-neutral trend compounded to **-1.9003%**, with annualized return **-3.8545%**, Sharpe **-0.3472**, maximum drawdown **-8.0919%**, and turnover **376**. EMA/MACD compounded to **-4.9355%**, with annualized return **-9.8502%**, Sharpe **-1.4061**, maximum drawdown **-5.4307%**, and turnover **314**. Both ended every session flat and recorded zero overnight-position violations. The loss profile differs: the price-ratio strategy suffered the deeper drawdown, while EMA/MACD produced the more persistently negative risk-adjusted return. Neither result was followed by a parameter change.

I therefore do not interpret the output as “one winner and two losers.” I conclude that it contains two rejected locked trend hypotheses and one positive but weak reversion observation. I report the economic signs exactly once and judge their strength using the full chain of development, inference, locked confirmation, and execution limitations.
""".strip(),
        },
        {
            "id": "cross_market_block",
            "type": "chart",
            "chartId": "cross_market_annualized_returns",
            "layout": "full",
        },
        {
            "id": "cross_market_interpretation",
            "type": "markdown",
            "sourceId": "day25_trend_robustness",
            "body": "Across **27 predeclared cases**, I found that changing positioning, ETF, or bar frequency did not create robust profitability. Long-flat was less damaging in several cases, but it was positive in only three of nine cases and the reference walk-forward aggregate remained negative after cost. I promoted no case and used none to retune parameters.",
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": "## 4. Research scope, data, timing, and metric definitions\n\nI use SPY, QQQ, and IWM as the canonical universe. Under my final trend convention, a signal known at a completed bar close becomes the target position at the next regular-session bar open. Within a session, I use open-to-next-open as the gross-return proxy; for the final bar I use open-to-close and then force the position flat. I charge turnover on opening changes, reversals, and every closing exit. I typeset the complete position, return, cost, wealth, Sharpe, and drawdown equations in Section 4.2. Because bar opens and closes are observable proxies rather than guaranteed fills, I report 0, 1, 2.5, and 5 basis-point development stresses. I preserve the earlier close-to-close one-row-lag convention as historical evidence instead of relabeling it. Expanding development history warms indicators, but execution resets flat at every annual or final-test boundary. I ran the locked test on exactly 123 NYSE sessions from 2 January through 30 June 2026 at the frozen one-basis-point cost and did not reuse it as a tuning sample.",
        },
        {
            "id": "data_sample_design",
            "type": "markdown",
            "sourceId": "base_config",
            "body": """### 4.1 Universe, samples, bar construction, and provenance

I use three highly liquid US equity exchange-traded funds: SPY as the primary trend instrument and SPY, QQQ, and IWM as robustness and reversion instruments. I deliberately accept limited cross-sectional breadth in exchange for reproducible high-frequency handling and related but non-identical exposures. This is not a diversified global trend universe, which matters when I compare my results with long-horizon multi-asset time-series momentum evidence [3].

The canonical development interval is **2 January 2020 through 31 December 2025**, inclusive. Its primary bar frequency is **15 minutes**, with 30- and 60-minute resampling used only for the declared robustness matrix. The four chronological test folds are calendar years 2022, 2023, 2024, and 2025. Earlier observations remain available to warm rolling indicators, but fold performance is computed only from the designated test rows and execution begins flat. The locked interval is **2 January through 30 June 2026**, inclusive, and contains **123 NYSE sessions**. No development runner may read a row on or after the locked boundary.

Development and locked 15-minute price bars came from Alpaca's consolidated SIP feed. Alpaca distinguishes SIP, which consolidates US exchange activity, from IEX, which represents a single venue [13]. The representative event-time experiment intentionally uses IEX trades because it is a bounded sampling comparison, not because IEX is assumed to be the complete market. This feed distinction is preserved in the evidence and prevents an IEX activity sample from being mislabeled as consolidated-volume evidence.

The locked panel passed the following deterministic integrity conditions before evaluation:

- exact symbols SPY, QQQ, and IWM;
- exactly 3,198 regular-session bars per symbol and 9,594 rows in total;
- exactly 26 fifteen-minute bars for each symbol-session;
- no duplicate symbol-timestamps, missing grid cells, null required fields, non-positive prices, or inconsistent OHLC relationships;
- timestamps inside the authorized interval only; and
- a canonical in-memory CSV SHA-256 equal to the value recorded in the immutable methodology bundle.

Raw price observations are never embedded in the report. The report reads bounded aggregate and fold artifacts after their manifests have been verified. This keeps the report reproducible without distributing credentials, account identifiers, or a mutable broker state.
""".strip(),
        },
        {
            "id": "evidence_classes_and_metrics",
            "type": "markdown",
            "sourceId": "trend_finalization_code",
            "body": """### 4.2 Evidence classes and performance equations

I separate seven evidence classes because combining them would overstate confidence: historical development research, chronological out-of-sample folds, post-hoc development diagnostics, the one-time locked final test, synthetic known-answer validation, live read-only broker evidence, and operational reproducibility. I use post-hoc diagnostics to explain observed patterns, never to select or retune a model; synthetic fill fixtures to validate arithmetic, not realized slippage; read-only broker calls to validate connectivity, not execution quality; and the locked test to observe generalization, never as a new tuning sample.

        Let `r_t` be a validated simple net return and `A` the observations-per-year factor. The reusable metric implementation computes:

        [[AXIOM_LATEX_PERFORMANCE_METRICS]]

The Sharpe ratio follows the conventional excess-return-per-unit-risk concept [11], with a zero risk-free rate because the strategies are intraday and the analysis concerns short test windows. This is a simplifying assumption, not a claim that cash has no opportunity cost. Annualization matches the evaluation grain: trend bar returns use the observed bars-per-session multiplied by 252, while the equal-weight reversion portfolio uses 252 session observations per year.

For the final trend convention, a signal observed at the close of bar `t` is converted into the position at the next bar open. If the bar is not the last bar in the session, the P&L proxy is open-to-next-open. On the last bar it is open-to-close, followed by a compulsory exit. Total turnover is the absolute opening position change plus the closing liquidation:

        [[AXIOM_LATEX_CAUSAL_TREND_ACCOUNTING]]

This convention is causal but still idealized. Opening and closing prints are benchmark prices, not promises of execution at those prices. The cost grid therefore tests sensitivity to a simple linear cost model; it does not model spread variation, queue priority, temporary impact, or latency.
""".strip(),
        },
        {
            "id": "price_ratio",
            "type": "markdown",
            "body": "## 5. Price-ratio trend model and evidence\n\nI found that the historical lineage could hold long, neutral, or short positions even though the original decision described long or flat positions only. I therefore ran both under exactly the same causal timing and cost protocol and allowed neither to replace the other. In the 2022-2025 aggregate at one basis point, long-short-neutral returned **-27.67%**, Sharpe **-0.577**, and maximum drawdown **-42.47%**. Long-flat returned **-0.45%**, Sharpe **0.027**, and maximum drawdown **-19.09%**. At zero cost long-flat was +18.16%, but turnover erased the edge by one basis point and losses deepened at higher stresses. I treat the nineteen positive rows in the 36-case full-development grid as sensitivity evidence, not permission to select one. Annual fold signs remained unstable. The frozen long-short-neutral strategy then returned **-1.90%** in the 2026 locked test, with Sharpe **-0.347** and maximum drawdown **-8.09%**; I made no parameter change after that failure.",
        },
        {
            "id": "price_ratio_mathematics",
            "type": "markdown",
            "sourceId": "trend_ratio_code",
            "body": """### 5.1 Mathematical specification and temporal implementation

For close price `P_t`, the frozen configuration uses a short simple moving average over eight 15-minute bars and a long simple moving average over 32 bars:

        [[AXIOM_LATEX_PRICE_RATIO_AVERAGES]]

With neutral band `delta = 0.001`, the long-short-neutral target is

        [[AXIOM_LATEX_PRICE_RATIO_SIGNAL]]

The long-flat comparator changes only the lower branch: an observation below the lower ratio band maps to a neutral position rather than a short position. That isolates the disputed positioning choice while holding windows, band, bar data, causal timing, costs, and evaluation folds constant. The comparison is therefore methodological; it is not a retroactive replacement of the implemented Day 6 lineage.

Rolling averages require 32 observations before the signal is available. The vectorized signal at completed bar `t` is shifted one observation before it becomes a tradable position. The final timing layer then assigns the resulting target to the next regular-session opening proxy and charges both opening changes and compulsory session-close exits. Direct reversal from +1 to -1 has turnover two. The next session begins from zero, so no overnight return is attributed.

The rule belongs to the broad moving-average class studied by Brock, Lakonishok, and LeBaron [2], but this implementation differs materially in sampling frequency, universe, cost structure, session handling, and evaluation period. It would therefore be invalid to treat prior literature as a profitability guarantee. The model hypothesis tested here is narrower: whether the short/long price ratio contains sufficiently persistent intraday direction in SPY, QQQ, or IWM to survive a neutral band, one-bar latency, forced flattening, and linear turnover costs.

The main coding risk was not the arithmetic of two averages. It was alignment. The implementation validates timestamps and symbols, rejects duplicates and malformed returns, prevents a signal from using the return it is supposed to predict, identifies warm-up rows explicitly, and recalculates turnover from positions rather than inferring costs from trade counts. Independent sequential replay reconstructs positions, gross return, forced exits, cost, and net return bar by bar. Exact replay parity is the evidence that the vectorized calculation did not hide a shift or boundary error.
""".strip(),
        },
        {
            "id": "price_ratio_diagnosis",
            "type": "markdown",
            "sourceId": "day25_trend_walk_forward",
            "body": """### 5.2 Why the long-flat comparison improves methodology but not the conclusion

My original project decision described a long-flat baseline, whereas my saved Day 6 implementation was long-short-neutral. Silently relabeling the code would have been a serious report defect. I therefore retained the historical lineage and added the intended long-flat rule as a separately named comparator under identical timing and costs.

Long-flat materially reduced the aggregate 2022-2025 loss from **-27.67%** to **-0.45%** at one basis point. That difference shows that the short branch was economically damaging in this sample. It does not show that long-flat is profitable. At zero cost the comparator returned **+18.16%**, but the sign changed by one basis point, demonstrating that turnover rather than raw directional accuracy dominated the practical outcome. Higher cost stresses worsened the loss. Nineteen of 36 full-development parameter-grid cases were positive, but those cases reuse the same development history and cannot be counted as independent confirmation. Choosing the best row after inspection would convert a sensitivity analysis into optimization.

Annual fold behavior also remained unstable. I do not consider a trend rule credible if it depends on one favorable year or one ETF-frequency combination. I expose all twelve annual results for the three fixed trend configurations. Their mixed signs and large dispersion show that the aggregate is not hiding a stable low-volatility edge, and the 2026 locked loss of **-1.90%** is consistent with that instability. I therefore conclude both that the audit resolved a real specification inconsistency and that the corrected comparison still failed to establish a cost-robust trend strategy.
""".strip(),
        },
        {
            "id": "trend_timing_block",
            "type": "table",
            "tableId": "trend_timing",
            "layout": "full",
        },
        {
            "id": "walk_forward_chart",
            "type": "chart",
            "chartId": "trend_walk_forward_returns",
            "layout": "full",
        },
        {
            "id": "annual_trend_diagnostic",
            "type": "markdown",
            "sourceId": "annual_regime_methodology",
            "body": """### 5.3 Post-hoc annual benchmark and regime diagnostic

After completing the frozen strategy tests, I added a separate calendar-year diagnostic to answer a narrower question: did any model beat a matching one-year buy-and-hold benchmark consistently, or only in a particular market environment? I compared the complete 2022-2025 folds with price-only benchmarks built from the first 15-minute open to the final close of each year. I excluded the partial January-June 2026 lockbox because it is not a complete calendar year and cannot become a tuning sample.

I found that **2022 was the only year in which every model/comparator row beat its relevant price-only benchmark**. SPY fell **18.41%**, while price-ratio long-short-neutral returned **+7.42%** net and long-flat returned **+5.50%**. EMA/MACD still lost **16.60%** and beat SPY by only **1.81 percentage points**; because the benchmark omits dividends, I do not treat that small relative difference as a robust economic success. In 2023-2025, none of the fixed trend models beat SPY.

This result supports a regime-dependence diagnosis, not a new causal hypothesis. The annual comparison was designed after the strategy outcomes were known, so I do not use it to select a model or claim that a bear-market filter would have worked out of sample. It identifies a testable next hypothesis: any proposed regime-conditioned rule must be predeclared, frozen, and evaluated on new untouched data.""".strip(),
        },
        {
            "id": "annual_trend_chart",
            "type": "chart",
            "chartId": "annual_trend_vs_spy",
            "layout": "full",
        },
        {
            "id": "annual_trend_chart_interpretation",
            "type": "markdown",
            "sourceId": "annual_regime_comparison",
            "body": "The chart shows a single defensive year rather than repeatable benchmark outperformance. I therefore conclude that the current trend rules are not general annual alternatives to SPY buy-and-hold. Their strongest relative result occurred when the benchmark declined; the result disappeared in all three rising benchmark years.",
        },
        {
            "id": "annual_short_effect_chart",
            "type": "chart",
            "chartId": "annual_short_sleeve_effect",
            "layout": "full",
        },
        {
            "id": "annual_short_effect_interpretation",
            "type": "markdown",
            "sourceId": "annual_short_effect",
            "body": "I also isolated the incremental effect of allowing price-ratio short positions. The short sleeve added **1.92 percentage points net** in 2022, then subtracted **9.18**, **6.16**, and **15.11 percentage points** in 2023, 2024, and 2025. It also added 348-458 turnover units each year. I conclude that the original long-short-neutral lineage depended on an unstable short-side payoff and paid a persistent turnover penalty; this explains why long-flat was materially less negative without making long-flat a validated winner.",
        },
        {
            "id": "ema_macd",
            "type": "markdown",
            "body": "## 6. EMA/MACD trend model and evidence\n\nFor my second trend family, I use fast and slow recursive EMAs plus a smoothed MACD-histogram confirmation. Under the final causal convention, I measured aggregate 2022-2025 performance at one basis point of **-34.23%**, Sharpe **-0.976**, and maximum drawdown **-41.28%**. The zero-cost aggregate was already **-13.62%**, so I conclude that transaction cost was material but not the sole explanation. My batch and independent sequential implementations matched positions, forced exits, turnover, costs, and returns exactly for all three trend configurations. The frozen EMA/MACD strategy then returned **-4.94%** in the 2026 locked test, with Sharpe **-1.406** and maximum drawdown **-5.43%**; I did not retune it after failure.",
        },
        {
            "id": "ema_macd_mathematics",
            "type": "markdown",
            "sourceId": "ema_macd_code",
            "body": """### 6.1 Recursive-filter specification

The second trend family is intentionally not another simple moving-average ratio. For span `n`, the recursive exponential moving average (EMA) uses

        [[AXIOM_LATEX_EMA_DEFINITION]]

The frozen spans are 12 and 26 bars. After their warm-up:

        [[AXIOM_LATEX_MACD_DEFINITION]]

Normalizing the histogram by price makes the neutral threshold comparable through time and across instruments. The target is +1 when `H_t > 0.0005`, -1 when `H_t < -0.0005`, and zero otherwise. The strategy does not use histogram acceleration as a hidden filter; those diagnostics are retained for analysis only. EMA state may continue across sessions because it summarizes historical prices, while tradable execution state is forced flat at every session boundary under the final timing convention.

The implementation uses an adjust-false recursive EMA seeded from the first valid observation and requires the declared minimum number of periods. It rejects missing values after initialization, malformed timestamps, duplicate symbol-timestamps, non-positive prices, and impossible simple returns. Signal availability is recorded explicitly. As with the price-ratio model, the completed-bar signal is lagged and independent replay verifies every downstream position and return.

The model hypothesis is that a normalized positive MACD histogram identifies persistent upward movement and a negative histogram identifies persistent downward movement. Time-series momentum has been documented in diversified futures at longer horizons [3], but that evidence does not establish short-horizon equity-ETF predictability. The present test is deliberately severe: one instrument for the primary fold analysis, intraday bars, a fixed neutral band, no volatility scaling, no regime selection, compulsory daily flattening, and turnover cost. These differences make the experiment a direct test of the implemented rule rather than an attempted replication of the literature.
""".strip(),
        },
        {
            "id": "ema_macd_diagnosis",
            "type": "markdown",
            "sourceId": "day25_trend_walk_forward",
            "body": """### 6.2 Economic diagnosis

EMA/MACD was negative even before costs: the aggregate zero-cost return was **-13.62%**. At one basis point it declined to **-34.23%**, so cost amplification was important but cannot explain the original sign. The model was frequently neutral, especially in the locked interval, yet its active observations did not deliver sufficient directional accuracy to offset adverse moves and turnover.

The locked result reinforces this conclusion. Over 123 sessions, EMA/MACD compounded to **-4.94%** with annualized volatility **7.19%**, Sharpe **-1.406**, maximum drawdown **-5.43%**, and 314 turnover units. It spent approximately **68.95%** of eligible observations flat. A high flat share can reduce volatility, but it cannot rescue a signal whose conditional active return is negative. Because the strategy already failed gross of cost in development and remained negative when frozen, further neutral-band or span search would be a new research program, not a justified final-report correction.

I regard this as a useful falsification result because it distinguishes failure caused mainly by trading intensity from failure caused by the underlying signal. The price-ratio long-flat comparison showed a gross edge erased near one basis point; EMA/MACD did not. I therefore conclude that the two trend families fail for related but non-identical reasons, which is more informative than reporting one aggregate negative number.
""".strip(),
        },
        {
            "id": "reversion_design",
            "type": "markdown",
            "body": "## 7. Pair-feasibility rejection and OU/VWAP reversion design\n\nI tested a conventional ETF-pair route first and rejected it before backtesting because no SPY/QQQ/IWM pair passed Holm-controlled cointegration plus fold-stability requirements. I then implemented a reversion alternative that subtracts a rolling VWAP-like reference, estimates an AR(1)-style OU half-life on the transformed residual, applies a variance-ratio gate, uses asymmetric entry and exit state, limits holding time, delays execution by one bar, forces overnight flatness, and charges turnover costs. I froze fast, base, and slow calibrations as a sensitivity family instead of ranking them into a winner.",
        },
        {
            "id": "pair_feasibility_method",
            "type": "markdown",
            "sourceId": "day14_pairs",
            "body": """### 7.1 Why cointegration was a feasibility gate rather than a pair-ranking exercise

The candidate pairs were fixed in advance: SPY/QQQ, SPY/IWM, and QQQ/IWM. Cointegration is stronger than high correlation. It asks whether a linear combination of non-stationary levels is stationary, which would support an error-correction interpretation [5]. The project therefore did not choose the visually closest or historically best-spread pair. It required each candidate to pass a family of statistical and stability gates.

Because three pair hypotheses were tested, raw p-values were controlled with Holm's sequential procedure [6]. The gate also required a stable hedge-ratio relationship and sufficient stationary behavior across chronological folds. A pair was eligible only if every predeclared condition passed. This intersection rule is conservative, but it protects the project from proceeding with an attractive backtest after an ambiguous stationarity result.

No pair was eligible. That prevented the Johansen/VECM branch from being reached and prevented a pair backtest from being manufactured from the least-bad candidate. The rejection is a substantive result: these related ETFs did not provide sufficiently stable cointegration evidence under the declared test. It also motivated a different reversion object—each instrument's own log deviation from a rolling volume-weighted reference—without claiming that the failed pair evidence validated the alternative.
""".strip(),
        },
        {
            "id": "ou_vwap_mathematics",
            "type": "markdown",
            "sourceId": "ou_vwap_code",
            "body": """### 7.2 OU/VWAP transformed-residual model

The CQF brief warns that applying an Ornstein-Uhlenbeck (OU) process directly to price can become a “straitjacket.” The implementation therefore models a transformed, locally referenced quantity rather than the price level. For close `C_t`, bar volume `V_t`, and provider bar VWAP `VWAP_t`, the rolling reference over `n_R` bars is

        [[AXIOM_LATEX_OU_REFERENCE]]

The code stores these quantities as `volume_weighted_reference` and `log_price_residual`. The denominator is formed as `rolling_volume`; it must be strictly positive or the reference and all downstream diagnostics are masked. The provider's bar VWAP, close, and volume are treated as completed information at the bar close. The current bar is allowed in the reference, but `position = signal.shift(1)` prevents that completed-bar information from trading the return already observed. The log ratio is dimensionless and centers the close relative to recent traded activity; it is not stationary by construction.

To derive the diagnostic rather than assert it, let the locally referenced residual have the continuous-time OU representation in equation OU1 over a window where `kappa`, `theta`, and `sigma` are locally constant. Moving `kappa x_s ds` to the left and multiplying by the integrating factor `e^(kappa s)` gives OU2. Integrating OU2 from `t` to `t + Delta` gives OU3; division by `e^(kappa(t+Delta))` gives the exact solution OU4. The stochastic integral has conditional mean zero. Ito isometry turns its squared kernel into the integral in OU6, while OU5 and OU7 show that the conditional mean and deviation from equilibrium decay exponentially rather than linearly.

Sampling OU4 at equally spaced bar intervals gives OU8-OU12. The implementation takes `Delta = 1` in bar units: `phi` is `ou_phi`, `a` is `ou_intercept`, and `theta` is `ou_equilibrium`. It does not estimate or use continuous-time diffusion volatility `sigma`; equation OU12 is included to distinguish that diffusion scale from the per-bar innovation scale. Over exactly `n_OU = ou_window` transitions, OU13-OU16 derive the rolling OLS quantities actually computed in `_rolling_ou_statistics`. Two fitted coefficients—intercept and slope—explain the `n_OU - 2` residual degrees of freedom. OU17-OU20 then derive the stationary residual scale, half-life, standardized residual, and variance-ratio diagnostic used by the signal gate.

        [[AXIOM_LATEX_OU_EXACT_DERIVATION]]

The following map makes the code connection, signal role, and admissibility condition explicit for every displayed equation in this group:

| Equations | Exact implementation connection | Role in the signal | Admissibility condition or limitation |
|---|---|---|---|
| OU1 | Conceptual parent of `_rolling_ou_statistics`; `kappa`, `theta`, and diffusion `sigma` are not directly fitted fields | Motivates local mean-reversion diagnostics only | Requires locally constant parameters and `kappa > 0`; the strategy neither simulates nor validates a global diffusion |
| OU2-OU4 | Exact continuous-to-discrete derivation behind `ou_phi`, `ou_intercept`, and `ou_equilibrium` | Justifies mapping a completed-bar residual window to an AR(1) gate | Assumes equally spaced bar time `Delta` and an OU law within the local window |
| OU5-OU7 | `ou_equilibrium = intercept / (1 - safe_phi)` and raw `ou_phi = phi` | Supplies equilibrium and exponential decay used to interpret reversion | Conditional Brownian increments must have mean zero and finite variance; decay is admissible only for `0 < phi < 1` |
| OU8-OU12 | `_rolling_ou_statistics`: `intercept`, `phi`, `innovation_std`; `ou_equilibrium` stores the mapped `theta` | Converts the continuous model into estimable per-bar diagnostics | `ou_innovation_std` is discrete innovation volatility; continuous diffusion `sigma` is not a signal field and requires a specified `Delta` |
| OU13-OU16 | `sx`, `sy`, `sxx`, `syy`, `sxy`, `centered_xx`, `centered_xy`, `phi`, `intercept`, `sse`, and `innovation_std` | Estimates the rolling state and its one-step uncertainty before eligibility is assessed | Requires positive lag variance, conditional mean-zero finite-variance innovations, and enough observations for `n_OU - 2 > 0`; OLS does not remove dependence or heteroskedasticity |
| OU17 | `ou_stationary_std = innovation_std / sqrt(1 - safe_phi^2)` | Denominator of `ou_zscore`, hence entry and exit magnitude | This is stationary residual-level volatility, distinct from diffusion and innovation volatility; it is masked unless `0 < phi < 1` and innovation scale is positive |
| OU18 | `ou_half_life_bars = -log(2) / log(safe_phi)` | Must lie between `minimum_half_life` and `maximum_half_life` | Bar-unit identity uses `Delta = 1`; estimates are numerically unstable as `phi` approaches one |
| OU19 | `ou_zscore = (log_price_residual - ou_equilibrium) / ou_stationary_std` | Drives long entry below the negative threshold, short entry above the positive threshold, and mean-proximity exits | Requires finite equilibrium and positive stationary scale; it is descriptive standardization, not a calibrated tail probability |
| OU20 | `variance_ratio`, built in `build_ou_vwap_reversion_strategy` from rolling sample variances of `diff(q)` and `diff(1)` | Requires `variance_ratio < variance_ratio_threshold` before trading | Requires positive one-period change variance; `VR < 1` indicates sublinear variance growth under this estimator, not certain or profitable reversion |

The three volatility concepts therefore have different units and uses. Continuous-time diffusion `sigma` scales Brownian noise per square-root unit of time and is not estimated by the strategy. `ou_innovation_std` is the fitted one-bar AR(1) shock standard deviation. `ou_stationary_std` is the implied standard deviation of the residual level and is the only one used to standardize the live diagnostic. Conflating them would change the entry scale.

The rolling construction adds important failure conditions. It assumes local parameter constancy even though adjacent estimates use heavily overlapping observations and are therefore dependent. It assumes equally spaced bar time; missing or irregular bars would make a bar-count half-life different from elapsed-clock-time decay. It assumes conditional mean-zero, finite-variance innovations for the OLS interpretation, a completed-information VWAP reference, and positive rolling volume. The code retains raw `ou_phi` for audit but masks equilibrium, stationary scale, and half-life unless `0 < phi < 1`; near `phi = 1`, division by `1 - phi` and the logarithmic half-life are numerically unstable. Passing these local compatibility checks is not proof of global stationarity, correct Gaussian diffusion dynamics, forecast stability, or profitability.

The variance ratio follows the idea of comparing multi-period with one-period variance [4]. A value below one is consistent with negative serial dependence, but even the original variance-ratio literature cautions that rejecting a random walk does not prove a profitable mean-reversion strategy.

Trading is permitted only when all diagnostics are available, `0 < phi < 1`, half-life is inside the configuration interval, and `VR_t(4) < 0.95`. From flat, the strategy enters long when `z_t` is at or below the negative entry threshold and short when it is at or above the positive threshold. An active position exits when the residual approaches equilibrium, the regime gate fails, the maximum holding period is reached, or the session ends. The completed-bar state is shifted one bar before execution. Session-close state is zero, so the next session opens flat.

The three configurations were frozen as a sensitivity family:

| Configuration | Reference bars | OU/VR window | Entry | Exit | Half-life range | Max hold |
|---|---:|---:|---:|---:|---:|---:|
| Fast | 26 | 104 | 1.75 | 0.25 | 1-20 bars | 20 bars |
| Base | 32 | 130 | 2.00 | 0.25 | 1-26 bars | 26 bars |
| Slow | 52 | 208 | 2.25 | 0.25 | 1-39 bars | 39 bars |

All use variance-ratio lag four, threshold 0.95, a one-basis-point base cost, and the same SPY/QQQ/IWM universe. These are not three independent discoveries. They share a model family, sample, instruments, and neighboring design choices. The report therefore adjusts inferential interpretation for three declared trials and never calls the best row an optimized winner.
""".strip(),
        },
        {
            "id": "pair_table_block",
            "type": "table",
            "tableId": "pair_feasibility",
            "layout": "full",
        },
        {
            "id": "reversion_results",
            "type": "markdown",
            "body": "## 8. Reversion profitability, cost sensitivity, and inference\n\nAt one basis point, I measured equal-weight fast and base development returns of **-10.51%** and **-4.36%**, while the slow variant returned **+6.03%** with Sharpe **0.639** and maximum drawdown **-1.95%** over **1,003 sessions**. The slow variant remained positive from zero through five basis points (**+6.80% to +3.02%**), consistent with its lower turnover, but all three calibrations share data and design ancestry and are not independent trials. I therefore keep the predeclared development inference inconclusive. In the one-time 2026 locked test, the frozen slow variant returned **+0.43%**, Sharpe **0.572**, and maximum drawdown **-0.70%** over 123 sessions. I view this as directionally consistent but too small and too short to convert the inconclusive development evidence into a profitability claim.",
        },
        {
            "id": "cost_chart",
            "type": "chart",
            "chartId": "reversion_cost_sensitivity",
            "layout": "full",
        },
        {
            "id": "inference_framework",
            "type": "markdown",
            "sourceId": "reversion_inference_code",
            "body": """### 8.1 Statistical inference under dependence and multiple testing

A positive compound return is not sufficient evidence of a positive expected return. Intraday and session returns can be serially dependent, non-Gaussian, and selected from several related trials. The reversion analysis therefore reports complementary diagnostics rather than relying on one conventional t-test.

First, the naive t-statistic uses the sample mean divided by its independent-observation standard error. Second, a Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) estimator with five lags adjusts the variance estimate for short-range serial dependence [8]. The HAC value is not automatically conservative in every finite sample, but disagreement between naive and HAC statistics reveals sensitivity to dependence assumptions.

Third, a deterministic moving-block bootstrap resamples five-session blocks rather than individual sessions. This preserves local dependence more faithfully than an independent bootstrap, following the general block-resampling logic for stationary observations [9]. The implementation uses 2,000 replications and seed 1701. It reports 95% percentile intervals for both the mean session return and the annualized Sharpe ratio. The block length is predeclared; it is not optimized to tighten the interval.

Fourth, the Probabilistic Sharpe Ratio (PSR) expresses the probability that the observed Sharpe exceeds a zero benchmark after accounting for sample length, skewness, and kurtosis. The Deflated Sharpe Ratio (DSR) further penalizes performance inflation from multiple trials and non-normality [10]. The project declares three related OU/VWAP calibration trials and uses their observed Sharpe dispersion. DSR is therefore the more relevant promotion statistic, although it too depends on modeling assumptions.

Finally, an information coefficient measures association between the causal signal score at bar `t` and the next intraday return. It is reported as a predictive diagnostic, not converted into a return claim. A low information coefficient alongside a positive compound return would suggest that a small number of trades, path dependence, or state filters may be driving the result.

For the slow equal-weight series, the naive t-statistic is **1.274** and HAC(5) is **1.421**. The block-bootstrap 95% interval for mean session return is **-0.0015% to +0.0149%**; its Sharpe interval also crosses zero. PSR is **92.28%**, but DSR falls to **62.58%**. These diagnostics agree on the decision even though their point values differ: the observed development return is promising enough to retain, but not strong enough to promote.
""".strip(),
        },
        {
            "id": "inference_block",
            "type": "table",
            "tableId": "slow_inference",
            "layout": "full",
        },
        {
            "id": "inference_interpretation",
            "type": "markdown",
            "sourceId": "day17_inference",
            "body": "For the slow equal-weight series, the naive t statistic was **1.274** and HAC(5) t statistic **1.421**. The 2,000-replication block-bootstrap mean interval was **-0.0015% to +0.0149% per session** and the Sharpe interval also crossed zero. PSR was **92.28%**, but DSR fell to **62.58%** after the declared three-trial adjustment. These diagnostics do not support promotion.",
        },
        {
            "id": "annual_ou_diagnostic",
            "type": "markdown",
            "sourceId": "annual_regime_methodology",
            "body": """### 8.2 Post-hoc annual benchmark and return-concentration diagnostic

I applied the same calendar-year diagnostic to slow OU/VWAP using an equal-starting-capital price-only basket of SPY, QQQ, and IWM. In 2022 the benchmark lost **23.99%** while slow OU/VWAP returned **+4.83%** net. In 2023-2025 the benchmark gained **31.61%**, **21.51%**, and **16.25%**, while slow OU/VWAP returned **-1.27%**, **+0.51%**, and **+1.92%**. The model therefore beat buy-and-hold only in 2022.

I do not interpret this as evidence that the model predicts bear markets. OU/VWAP was flat for more than **97%** of eligible observations in every year, and its 2022 gain came from only **nine non-zero portfolio days**. The three best arithmetic day contributions summed to **5.26%**, more than the full **4.78%** arithmetic annual sum before offsetting losses. I conclude that the positive year is both regime-specific and concentrated, which raises uncertainty rather than reducing it.""".strip(),
        },
        {
            "id": "annual_ou_chart",
            "type": "chart",
            "chartId": "annual_ou_vs_equal_weight",
            "layout": "full",
        },
        {
            "id": "annual_ou_chart_interpretation",
            "type": "markdown",
            "sourceId": "annual_regime_comparison",
            "body": "The benchmark chart shows why I retain the phrase **positive but inconclusive**. Slow OU/VWAP provided a defensive comparison in 2022 and small absolute gains in two later years, but it substantially lagged the rising ETF basket and did not produce broad active-year evidence.",
        },
        {
            "id": "annual_ou_concentration_block",
            "type": "table",
            "tableId": "annual_ou_concentration",
            "layout": "full",
        },
        {
            "id": "portfolio",
            "type": "markdown",
            "body": "## 9. Diversification and allocation evidence\n\nBecause my trend timing changed, I reran the six-sleeve dependency instead of relying on a caveat. Under final causal timing, I measured maximum training absolute correlation of **0.818**, median effective rank of **3.929**, and median equal-weight test diversification ratio of **1.431**. Equal weight, inverse volatility, and constrained minimum variance returned **-35.72%**, **-34.93%**, and **-33.25%** over the common 1,003-session panel. I preserve the historical Day 15-16 evidence, but the table below is the causal rebuild. I found that diversification did not rescue weak sleeves, and I selected no allocation rule.",
        },
        {
            "id": "portfolio_methodology",
            "type": "markdown",
            "sourceId": "day16_allocation",
            "body": """### 9.1 Dependence diagnostics and allocation rules

The portfolio exercise asks whether combining six trend sleeves can reduce risk without inventing expected returns. It does not assume diversification creates alpha. The sleeves arise from two trend families across three related ETFs, so material dependence is expected. Training data are used to estimate covariance and volatility; test-period returns remain chronological.

Three dependency diagnostics are reported. Maximum absolute correlation identifies the strongest pairwise linkage. Effective rank summarizes the number of materially distinct covariance directions rather than treating six nominal sleeves as six independent bets. The diversification ratio compares the weighted average constituent volatility with portfolio volatility:

        [[AXIOM_LATEX_DIVERSIFICATION_RATIO]]

A value above one indicates volatility diversification under the estimated covariance matrix. It does not imply a positive expected return.

The allocation rules are equal weight, inverse volatility, and constrained minimum variance. Minimum variance is solved with long-only, fully invested weights; it does not use a return forecast. All three rules are evaluated on the same 1,003-session test panel after the final trend timing was rebuilt. The results remain negative: **-35.72%**, **-34.93%**, and **-33.25%** cumulative return. The less-negative minimum-variance result is not promoted because all rules inherit weak sleeves and share the same sample. The exercise demonstrates a core portfolio lesson: covariance management can alter the path and drawdown, but it cannot reliably transform negative conditional means into a profitable strategy.

Historical 95% Value at Risk and Expected Shortfall are reported as positive loss magnitudes from the lower tail. These are empirical summaries, not parametric guarantees. With only four annual folds and related sleeves, estimation error is material. The project therefore reports the allocation comparison as a robustness diagnostic and keeps the conclusion at the sleeve level.
""".strip(),
        },
        {
            "id": "allocation_block",
            "type": "table",
            "tableId": "allocation",
            "layout": "full",
        },
        {
            "id": "methods_heading",
            "type": "markdown",
            "body": "## 10. Event-driven parity and numerical/statistical methods\n\nI verified the final trend calculations with exact batch-versus-sequential replay. Separately, I processed **77,053 genuine SPY IEX trades** across five predeclared complete 2025 sessions into 130 time bars and 130 dollar bars with exact trade-count, volume, and notional conservation. I found that dollar bars reduced dollar-value CV from **0.8534** to **0.2077** and volume CV from **0.8204** to **0.1394**, while duration dispersion increased. A session-reset 4/16 indicator had only 55 available observations per method, so I kept its associations descriptive and did not change the primary 15-minute sampling rule. I preserve the first 8/32 cross-gap engineering attempt as invalid rather than hiding it.",
        },
        {
            "id": "event_time_methodology",
            "type": "markdown",
            "sourceId": "day25_event_sampling",
            "body": """### 10.1 Why event-time bars were tested

Clock-time bars contain unequal market activity: a quiet 15-minute interval and an active 15-minute interval receive the same observation weight. Dollar bars instead close after a target notional amount has traded, so their duration varies while economic activity is more even. The experiment tests whether that construction changes sampling quality and simple signal behavior; it is not a profitability contest.

Five complete SPY regular sessions were chosen by calendar rule before download: the fifteenth calendar day of January, April, July, October, and December 2025, moved only if necessary to the next NYSE session. The resulting sample contains **77,053 genuine IEX trades**. For each session, the dollar threshold equals session notional divided by the number of 15-minute benchmark bars. Whole trades remain atomic; no trade is split to force an exact threshold. The last partial dollar bar is retained. Conservation checks require the output bars to reproduce total trade count, volume, and dollar value exactly.

Both sampling methods produced 130 bars. Time bars had dollar-value coefficient of variation **0.8534** and volume coefficient of variation **0.8204**. Dollar bars reduced those values to **0.2077** and **0.1394**, confirming that event-time sampling equalized activity. The trade-off is temporal irregularity: dollar-bar duration dispersion increased because busy periods close bars faster.

The price-ratio diagnostic uses 4/16 windows and a 0.001 band, reset independently inside each session. This correction is methodological. The first engineering version carried 8/32 state across five sessions separated by months. Because a normal session has only 26 fifteen-minute bars, the 32-bar window could not warm within a session; carrying it across a multi-month gap created a false continuous history. That invalid bundle is preserved and excluded from final claims. The corrected windows are the smallest previously declared four-to-one pair that can warm within one session and are applied identically to both bar types.

Only 55 signal observations were available for each method after warm-up. Pearson and Spearman next-event associations are therefore descriptive and not a basis for changing the canonical 15-minute data rule. Dollar bars succeeded at activity equalization; they did not establish a better trading signal.
""".strip(),
        },
        {
            "id": "numerical_implementation_boundary",
            "type": "markdown",
            "sourceId": "day24_specification",
            "body": """### 10.2 What was coded and what was delegated to libraries

The numerical-method table distinguishes project logic from trusted library primitives. Core strategy state, temporal alignment, turnover, cost accounting, fold assembly, OU/AR(1) rolling sums, variance-ratio construction, block-bootstrap resampling, event-bar aggregation, order-state transitions, reconciliation, and deterministic artifact hashing are project implementations or material adjustments. DataFrame operations, timestamps, linear algebra, optimization, descriptive statistics, and exchange calendars use established Python libraries.

This boundary is intentional. Reimplementing a general optimizer or calendar would increase code volume without necessarily increasing validity. The CQF-relevant judgment lies in specifying inputs and constraints, preventing leakage, choosing denominators, handling state boundaries, validating outputs, and exposing numerical limitations. For example, ordinary least squares is a standard calculation; deciding that `phi` outside (0,1) invalidates the OU mapping, that stationary standard deviation must be finite and positive, and that state must reset at the test boundary is project logic.

Exact sequential replay is used as a strong test for vectorized backtests. Synthetic known-answer series test sign, lag, reversal turnover, forced flattening, and cost identities. Deterministic seeds make bootstrap intervals reproducible. SHA-256 manifests make artifact tampering observable. These controls do not prove the economic model is true; they increase confidence that the reported result is the result produced by the declared model.
""".strip(),
        },
        {
            "id": "event_time_block",
            "type": "table",
            "tableId": "event_time",
            "layout": "full",
        },
        {
            "id": "methods_block",
            "type": "table",
            "tableId": "methods",
            "layout": "full",
        },
        {
            "id": "broker",
            "type": "markdown",
            "body": "## 11. Broker architecture, order state, reconciliation, and safety\n\nI designed the Alpaca adapter to be paper-only and fail closed, with credentials kept in the environment. My order state machine handles acknowledgements, partial fills, replacement, cancellation, rejection, duplicates, stale or out-of-order events, and timeouts. I reconcile broker and local orders, fills, positions, and cash; the stream monitor can reconnect or open a circuit. My Day 18 live read-only preflight passed. The Day 21 probe then aborted safely because the market-window and causal-signal gates were not satisfied—an operational success, not a failed trade.",
        },
        {
            "id": "broker_architecture_detail",
            "type": "markdown",
            "sourceId": "day24_specification",
            "body": """### 11.1 Fail-closed order lifecycle

The broker layer is separated from research code so that a backtest cannot accidentally authorize an order. Configuration requires paper mode, disables live trading, disables order submission by default, requires manual confirmation, and exposes a kill switch. Credentials are loaded only from named environment variables; they are not stored in configuration, artifacts, logs, or the report. Alpaca documents separate paper and live domains and credentials [15]; the adapter validates the paper boundary before any mutation path can become eligible.

An order is not treated as a binary submitted/filled object. The state machine represents acknowledgement, open status, partial fills, pending replacement, replacement, pending cancel, cancellation, completion, rejection, expiration, and timeout. Events carry monotonic sequence and timestamp expectations. Duplicate messages are idempotent; stale or out-of-order messages cannot reverse a terminal state. Filled quantity is monotonic and cannot exceed requested quantity. A replacement preserves lineage rather than appearing as an unrelated order.

Streaming events are useful because Alpaca order updates can include fills, partial fills, cancellations, and rejections [14]. They are still not a single source of truth. A stale stream, disconnect, lost message, or inconsistent broker response opens a reconciliation path. The monitor compares local orders, fills, positions, and cash with broker snapshots. Material mismatches can block new actions and open a circuit until state is understood.

The operational tests deliberately inject difficult paths: duplicate events, a fill arriving after a cancel request, replacement races, partial fills, rejected orders, timeouts, stream staleness, position mismatches, cash mismatches, and recovery after reconciliation. Passing synthetic scenarios means the transition and comparison logic handled known answers. It does not mean an unknown production failure cannot occur.

### 11.2 Read-only evidence and the Day 21 abort

Day 18 used a live read-only paper preflight to validate endpoint, account, asset, and market-state access without submitting, cancelling, or replacing an order. Day 21 then evaluated a controlled paper-execution gate. The market-window and causal-signal conditions were not simultaneously satisfied, so the runner aborted without constructing an eligible order. This is an operational success because the intended behavior was to refuse mutation when any gate failed. It is not a failed trade and must not be counted as a live observation.

The distinction between read-only access, a safely aborted order path, and an executed paper order is maintained throughout the report. The repository demonstrates broker safety architecture, but it has no empirical fill sample that can validate realized shortfall or latency.
""".strip(),
        },
        {
            "id": "operations_block",
            "type": "table",
            "tableId": "operations",
            "layout": "full",
        },
        {
            "id": "execution_heading",
            "type": "markdown",
            "body": "## 12. Execution benchmarking and performance reporting\n\nI decompose execution shortfall into decision-to-arrival and arrival-to-fill movement and report fill latency, completion, quantity, commissions, round-trip gross/net P&L, drawdown, VaR, ES, and beta-to-SPY by evidence class. My saved Day 22 rows are deterministic fixtures only. The scheduled campaign produced no immutable fills; I keep that absence visible and do not replace it with synthetic or historical alpha evidence.",
        },
        {
            "id": "execution_benchmark_detail",
            "type": "markdown",
            "sourceId": "day22_synthetic",
            "body": """### 12.1 Shortfall decomposition and evidence boundary

Implementation shortfall measures the difference between a decision benchmark and realized execution. The project decomposes a buy order into decision-to-arrival movement and arrival-to-fill movement, with signs reversed consistently for sells:

        [[AXIOM_LATEX_EXECUTION_SHORTFALL]]

This decomposition is related to the broader execution-cost framework in which transaction costs and volatility risk must be considered jointly [12]. The implementation also reports commissions, partial-fill completion, round-trip gross and net P&L, equity drawdown, historical VaR and Expected Shortfall, and beta to SPY when a valid return series exists.

The Day 22 rows shown below are deterministic known-answer fixtures. Their purpose is to prove arithmetic, sign conventions, aggregation, and evidence labeling. They are explicitly marked ineligible for alpha claims. A fixture can show that a 10-basis-point adverse buy move is recorded as a positive cost; it cannot show what Alpaca would have filled in a live market.

The authorized campaign expired without immutable fills. Two scheduled slots were missed and eight remained stale; no slot was reset or replaced with a synthetic observation. Consequently, realized spread capture, fill probability, latency distribution, cancellation behavior, and shortfall cannot be estimated empirically. The report remains partial for this reason. Strong historical alpha would not cure this gap, and the small locked reversion gain makes the gap even more consequential.
""".strip(),
        },
        {
            "id": "execution_block",
            "type": "table",
            "tableId": "execution",
            "layout": "full",
        },
        {
            "id": "reproducibility",
            "type": "markdown",
            "body": "## 13. Reproducibility, CI, scheduling, and container limitations\n\nI build frozen evidence bundles with exact allow-lists, non-self SHA-256 manifests, sibling staging, atomic replacement, and overwrite protection. I hash-lock the transitive Python environment and use an offline health path that checks 14 gates without credentials, network access, broker construction, or order permission. CI recreates that environment. For the one-time final test, my runner required the exact authorization code, verified frozen development and model hashes before data access, rejected rows outside the interval, required a complete regular-session grid, reset execution flat, reported all three models, and wrote a six-file immutable bundle that refuses overwrite. The saved 9,594-bar panel contains three symbols, 123 complete sessions, zero duplicate symbol-timestamps, zero missing cells, and matching manifest hashes. I performed no ranking or retuning after the run. I also statically validated Docker and Compose definitions, but I make no local image-build or runtime claim because no compatible container runtime was available on the Day 23 host.",
        },
        {
            "id": "reproducibility_protocol",
            "type": "markdown",
            "sourceId": "locked_final_test_code",
            "body": """### 13.1 Deterministic research artifacts

Each research day writes a bounded artifact directory with an exact filename allow-list. CSV and JSON serialization is deterministic. The manifest records a SHA-256 for every sibling file but not for itself, avoiding a recursive self-hash. Output is assembled in a sibling staging directory and atomically replaces the destination only after validation. Existing directories are protected unless an explicit overwrite path is used. These controls make accidental partial output and silent tampering detectable.

The report itself follows the same pattern. Before loading a quantitative table, the builder replays the relevant evidence manifest. It then constructs one canonical artifact containing ordered narrative blocks, cards, charts, tables, bounded datasets, and source metadata. The portable builder embeds that same payload into a self-contained HTML reader and also creates a semantic no-script/print representation. The HTML does not require a CDN, remote chart library, local server, or sidecar data file.

### 13.2 One-time locked-test controls

The final-test runner requires an exact authorization code and refuses to overwrite an existing locked bundle. Before data access, it verifies hashes for the development dataset, trend-finalization implementation, reversion implementation, and Day 25 staging manifest. After data retrieval it rejects out-of-window rows, incomplete symbol-session grids, duplicate timestamps, invalid OHLC, missing required values, or inconsistent feed metadata. Development history may warm model state, but evaluation rows are restricted to the locked interval and execution resets flat at its start.

The output order is fixed: price-ratio long-short-neutral, EMA/MACD long-short-neutral, and slow OU/VWAP equal weight. All three results are written, whether positive or negative. The methodology record states that no ranking or retuning was performed. This is stronger than simply promising not to cherry-pick because the runner has no result-dependent branch.

### 13.3 Tests, environment, and remaining reproducibility limits

The project has focused unit and integration tests for data validation, strategy state, timing, metrics, inference, artifacts, broker state, reconciliation, operations, and submission packaging. A hash-locked dependency file supports clean-environment installation. The offline health path checks 14 gates without reading credentials, creating a broker client, accessing the network, or permitting orders. Continuous integration recreates the environment and runs the test suite.

Docker and Compose definitions make the intended runtime explicit and expose safe scheduled entrypoints. However, no compatible container runtime was available on the validation host. Static configuration checks passed, but the report does not claim that an image was built or executed locally. Portable browser verification was also structural-only because no compatible browser was available to the packaged builder. Payload equality, runtime roots, semantic fallback, and structural validation passed; visual interaction at desktop and narrow widths remains a disclosed QA limitation.
""".strip(),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## 14. Limitations and threats to validity\n\nI limit the strategy study to three correlated US ETFs, one broad development era, four annual folds, and one 123-session locked interval. My next-bar opens and session closes are causal price proxies, not guaranteed fills; fixed basis-point costs omit spread dynamics, queue position, market impact, and intrabar path. I treat the 36-case long-flat grid as full-development sensitivity, not independent out-of-sample selection evidence. The annual buy-and-hold diagnostic is post-hoc and uses price-only benchmarks that omit dividends and idle-cash yield; I use it to diagnose regime dependence, not to claim causal timing skill or select a filter. The event-time experiment covers five spaced sessions and only 55 indicator observations per method. Portfolio weights use only six related trend sleeves and no expected-return model. The development confidence intervals for slow OU/VWAP cross zero, its locked gain is only **0.43%**, and I invented no new post-test inferential gate. Synthetic failure tests and read-only broker checks are not live execution proof, and the expired campaign produced no fill/slippage sample. I therefore conclude that the observed post-2025 evidence is still insufficient for deployment.",
        },
        {
            "id": "validity_analysis",
            "type": "markdown",
            "body": f"""### 14.1 Internal, statistical, external, and operational validity

**Internal validity.** The final trend convention prevents same-row leakage and removes overnight attribution, but bar open and close remain proxies. A marketable order may cross the spread, fill partially, or miss the displayed price. Fixed cost does not condition on volatility, volume, order size, or side. The reversion strategy uses provider bar VWAP rather than reconstructing the full order book. These choices can bias simulated performance upward or downward, and the direction is not identified from the current evidence.

**Statistical validity.** Four annual folds provide chronological separation but limited independent regimes. Returns can be autocorrelated and heteroskedastic; HAC and block bootstrap address only declared dependence structures. The slow configuration was one of three related calibrations, and DSR attempts to penalize that search family, but unrecorded researcher degrees of freedom from earlier project design cannot be reduced to one exact trial count. Confidence intervals crossing zero remain the decisive limitation.

**External validity.** SPY, QQQ, and IWM are correlated US equity ETFs. They do not test currencies, commodities, rates, single stocks, non-US hours, or crisis-only execution. The 2020-2026 period includes unusual regimes but remains one historical path. The success or failure of time-series momentum in diversified futures [3] is not directly transferable to this intraday ETF implementation.

**Construct validity.** The price ratio and normalized MACD histogram are proxies for trend, not exhaustive definitions. The OU-compatible rolling AR(1) is a local diagnostic, not proof of a continuous-time OU law. A variance ratio below one indicates negative serial dependence under its estimator, not guaranteed economic mean reversion [4]. The equal-weight OU/VWAP portfolio averages three contemporaneous session returns and may mask symbol-level heterogeneity.

**Operational validity.** State-machine and reconciliation scenarios cover declared failures but are synthetic. The live evidence consists of read-only access and a safe abort. There is no empirical distribution of fills, shortfall, latency, rejection, cancellation, or recovery. Container and browser checks are partially structural because the required runtimes were unavailable.

### 14.2 The predeclared Phase II development test did not improve the base case

The Day 26 protocol was frozen before its new returns were calculated. It declared exactly two trials, preserved all prior baselines, prohibited the consumed January-June 2026 interval, held exposure and leverage limits fixed, reported four annual folds and four cost stresses, and recorded 2,000-replication moving-block-bootstrap intervals. The repository contained no untouched post-lock sample, so these results are development diagnostics rather than a second final test.

The price-ratio trial asked whether one-hour signal confirmation plus a half-band hysteresis exit could reduce the long-flat baseline's cost burden. It reduced turnover from **{phase2_trend_1bp["baseline_turnover"]:.0f}** to **{phase2_trend_1bp["phase2_turnover"]:.0f}**, a **{phase2_trend_1bp["turnover_change_pct"]:.2f}%** change. That saving was not free: zero-cost gross cumulative return fell from **{100.0 * phase2_trend_baseline["gross_cumulative_return"]:+.2f}%** to **{100.0 * phase2_trend_candidate["gross_cumulative_return"]:+.2f}%**. At the one-basis-point base cost, net return therefore moved from **{100.0 * phase2_trend_1bp["baseline_cumulative_return"]:+.2f}%** to **{100.0 * phase2_trend_1bp["phase2_cumulative_return"]:+.2f}%**, a deterioration of **{100.0 * phase2_trend_1bp["cumulative_return_change"]:+.2f} percentage points**. Both rules had two positive annual folds. At five basis points, the lower-turnover rule was less negative (**{100.0 * phase2_trend_5bp["phase2_cumulative_return"]:+.2f}%** versus **{100.0 * phase2_trend_5bp["baseline_cumulative_return"]:+.2f}%**), but neither case was remotely profitable. The approximate break-even cost stayed below one basis point for both rules. The turnover mechanism worked mechanically; the profitability hypothesis failed at the declared base cost because confirmation discarded too much gross edge.

The OU/VWAP trial required a new entry's model-implied residual convergence before session close to cover a ten-basis-point round-trip margin. It was non-binding: the Phase II rule produced the same **{phase2_ou_1bp["phase2_turnover"]:.0f}** units of equal-weight turnover and the same **{100.0 * phase2_ou_1bp["phase2_cumulative_return"]:+.2f}%** one-basis-point return as the retained slow baseline. Its HAC(5) t statistic remained **{phase2_ou_candidate_inference["hac_t"]:.3f}**, and its bootstrap mean interval remained **{100.0 * phase2_ou_candidate_inference["bootstrap_mean_ci_lower"]:+.4f}% to {100.0 * phase2_ou_candidate_inference["bootstrap_mean_ci_upper"]:+.4f}%** per session, crossing zero. The identical positions show that every baseline entry already cleared the proposed cost-margin proxy; the gate supplied a useful diagnostic but no economic improvement.

This negative Phase II outcome is retained. It prevents the report from claiming that profitability was improved merely because plausible steps were implemented. It also sharpens the next question: further development work must introduce genuinely new information or economically distinct opportunities, not another nearby threshold. **Improved profitability may be claimed only if** a separately frozen design outperforms the retained baseline on the same untouched future holdout, at the same risk and exposure limits, net of empirically grounded execution cost, with all trials disclosed. Until then, improved profitability has not been demonstrated.
""".strip(),
        },
        {
            "id": "discussion",
            "type": "markdown",
            "body": """### 14.3 Discussion: what the project establishes

I established that my research and execution framework can reject its own hypotheses. I regard that as a stronger scientific outcome than a positive curve produced by an opaque notebook, but I do not equate it with commercial readiness.

The long-flat audit is my clearest example. Correcting the intended positioning materially improved the trend result, yet honest transaction-cost accounting removed the apparent gross edge. I made the correction because it resolved a real inconsistency; I did not rewrite the project around the less-negative number. EMA/MACD failed even before costs, showing that my weak strategies do not all share the same failure mechanism.

My subsequent predeclared Phase II experiment strengthened that diagnosis rather than rescuing it. Persistent hysteresis cut long-flat turnover by 10.27% but reduced gross return enough to make the one-basis-point result slightly worse. The slow OU/VWAP cost-margin gate did not remove a single baseline entry. By reporting both outcomes, I reject two plausible but unsupported stories: that modest turnover smoothing was sufficient for the trend rule, or that I could improve the reversion result by excluding entries that did not cover the declared stress margin.

I believe slow OU/VWAP deserves continued attention because it survived the declared cost range in development and remained positive in the one-time locked interval. Its low turnover and small drawdown are economically coherent with a slower, selective state machine. Nevertheless, I found weak inferential evidence, a small locked gain, concentrated annual contributions, and no empirical execution evidence. “Promising but inconclusive” is therefore my evidence-based conclusion, not presentation language.

Operationally, I separated research, broker read-only access, mutation authorization, state reconciliation, and artifact production. The Day 21 abort and the expired no-fill campaign show that my system can refuse to manufacture activity. I consider the remaining gap empirical, not cosmetic. I would require observations from the execution path under a new, immutable prospective protocol before making any production claim.
""".strip(),
        },
        {
            "id": "conclusion",
            "type": "markdown",
            "body": "## 15. Conclusion and recommended next steps\n\nI built a defensible end-to-end research evidence chain: I resolved the positioning mismatch, made trend timing causal and replay-verifiable, used genuine but bounded event-time evidence, kept rejected hypotheses visible, and evaluated the frozen models once on the locked 2026 interval without retuning. I found that both trend models failed and that slow OU/VWAP remained positive by only **0.43%**. I then tested two separately frozen development mechanisms without reopening the lockbox. Persistent hysteresis reduced turnover but slightly worsened the one-basis-point trend result, and the OU cost-margin gate was non-binding. My post-hoc annual diagnostic showed that relative benchmark outperformance occurred only in 2022 and that the slow OU/VWAP gain was sparse and concentrated. **I have not established deployable profitability.** I did not improve profitability at the declared base cost. I retain every model and result as a permanent baseline. My next research step would require genuinely new information or economically distinct opportunities, empirical implementation-shortfall evidence, a predeclared regime hypothesis if used, and a separately frozen untouched future holdout—not another nearby threshold.",
        },
        {
            "id": "references",
            "type": "markdown",
            "body": """## 16. References

1. Fitch Learning. *Certificate in Quantitative Finance Final Project Brief*, January 2026 Cohort, version 4, especially pp. 3, 21-23, and 29-30. Candidate portal document. The brief governs the required mathematical description, two trend strategies, one non-trivial reversion strategy, broker/API handling, execution benchmarking, reproducibility, and report format.

2. Brock, W., Lakonishok, J., and LeBaron, B. (1992). “Simple Technical Trading Rules and the Stochastic Properties of Stock Returns.” *Journal of Finance*, 47(5), 1731-1764. [https://doi.org/10.1111/j.1540-6261.1992.tb04681.x](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x)

3. Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012). “Time Series Momentum.” *Journal of Financial Economics*, 104(2), 228-250. [https://doi.org/10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003)

4. Lo, A. W., and MacKinlay, A. C. (1988). “Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test.” *Review of Financial Studies*, 1(1), 41-66. [https://doi.org/10.1093/rfs/1.1.41](https://doi.org/10.1093/rfs/1.1.41)

5. Engle, R. F., and Granger, C. W. J. (1987). “Co-Integration and Error Correction: Representation, Estimation, and Testing.” *Econometrica*, 55(2), 251-276. [https://doi.org/10.2307/1913236](https://doi.org/10.2307/1913236)

6. Holm, S. (1979). “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, 6(2), 65-70. [https://doi.org/10.2307/4615733](https://doi.org/10.2307/4615733)

7. Uhlenbeck, G. E., and Ornstein, L. S. (1930). “On the Theory of the Brownian Motion.” *Physical Review*, 36, 823-841. [https://doi.org/10.1103/PhysRev.36.823](https://doi.org/10.1103/PhysRev.36.823)

8. Newey, W. K., and West, K. D. (1987). “A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.” *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)

9. Künsch, H. R. (1989). “The Jackknife and the Bootstrap for General Stationary Observations.” *Annals of Statistics*, 17(3), 1217-1241. [https://doi.org/10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)

10. Bailey, D. H., and López de Prado, M. (2014). “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.” *Journal of Portfolio Management*, 40(5), 94-107. [https://doi.org/10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)

11. Sharpe, W. F. (1994). “The Sharpe Ratio.” *Journal of Portfolio Management*, 21(1), 49-58. [https://doi.org/10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501)

12. Almgren, R., and Chriss, N. (2001). “Optimal Execution of Portfolio Transactions.” *Journal of Risk*, 3(2), 5-39. [https://doi.org/10.21314/JOR.2001.041](https://doi.org/10.21314/JOR.2001.041)

13. Alpaca Markets. “Market Data FAQ: IEX and SIP Data.” Official documentation. [https://docs.alpaca.markets/us/docs/market-data-faq](https://docs.alpaca.markets/us/docs/market-data-faq)

14. Alpaca Markets. “Websocket Streaming: Trade Updates.” Official documentation. [https://docs.alpaca.markets/us/v1.4.2/docs/websocket-streaming](https://docs.alpaca.markets/us/v1.4.2/docs/websocket-streaming)

15. Alpaca Markets. “Authentication: Paper Trading.” Official documentation. [https://docs.alpaca.markets/us/v1.1/docs/authentication-1](https://docs.alpaca.markets/us/v1.1/docs/authentication-1)

All external references were checked against publisher, DOI, journal, or official vendor pages. References motivate model and validation choices; they are not used as substitutes for the repository's own empirical evidence.
""".strip(),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Axiom: Trend-Following, Reversion, and Fail-Closed Paper Execution",
            "description": "Detailed CQF algorithmic-trading technical report with model equations, chronological evidence, statistical inference, verified references, and honest execution limitations.",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "partial",
            "datasets": datasets,
            "accessIssues": [
                {
                    "id": "live_campaign_pending",
                    "dataset": "execution",
                    "message": "Empirical Day 22 paper fills and slippage are not yet available; displayed execution rows are synthetic known-answer fixtures only.",
                }
            ],
        },
        "sources": sources,
        "package_info": {
            "root": "axiom-day27-mathematical-revision",
            "manifestPath": "report_artifact.json",
            "snapshotPath": "report_artifact.json",
        },
    }


def claim_inventory(project_root: str | Path) -> tuple[dict[str, object], ...]:
    """Return the exact Day 24 headline-claim inventory."""

    artifact = build_report_artifact(project_root)
    datasets = artifact["snapshot"]["datasets"]  # type: ignore[index]
    outcomes = datasets["model_outcomes"]  # type: ignore[index]
    outcome_by_id = {row["model_id"]: row for row in outcomes}
    inference = datasets["inference"][0]  # type: ignore[index]
    operations = datasets["operations"]  # type: ignore[index]
    event_time = datasets["event_time"]  # type: ignore[index]
    claims = (
        (
            "trend_ratio_failed",
            "Long-short-neutral price-ratio trend lost 1.90% in the one-time locked 2026 final test.",
            "locked_final_test",
            SOURCE_FILES["day25_locked_performance"],
            "complete",
            f"cumulative_return={outcome_by_id['price_ratio_long_short_neutral_locked']['cumulative_return']:.12g}; cost_bps=1",
        ),
        (
            "long_flat_not_rescue",
            "The required long-flat price-ratio development comparison lost 0.45% after one-basis-point turnover cost.",
            "chronological_out_of_sample",
            SOURCE_FILES["day25_trend_walk_forward"],
            "complete",
            f"cumulative_return={outcome_by_id['price_ratio_long_flat_development']['cumulative_return']:.12g}; zero_cost_positive=true",
        ),
        (
            "ema_macd_failed",
            "EMA/MACD trend lost 4.94% in the one-time locked 2026 final test.",
            "locked_final_test",
            SOURCE_FILES["day25_locked_performance"],
            "complete",
            f"cumulative_return={outcome_by_id['ema_macd_long_short_neutral_locked']['cumulative_return']:.12g}; cost_bps=1",
        ),
        (
            "trend_timing_complete",
            "All three final trend configurations use next-bar-open, overnight-flat accounting and match sequential replay.",
            "operational_reproducibility",
            SOURCE_FILES["day25_trend_parity"],
            "complete",
            "parity_passed=3/3; overnight_violations=0",
        ),
        (
            "event_time_complete",
            "Representative event-time evidence uses 77,053 genuine trades over five predeclared complete sessions with exact conservation.",
            "historical_development",
            SOURCE_FILES["day25_event_conservation"],
            "complete",
            f"methods={len(event_time)}; sessions=5; trade_count_volume_notional_pass=true",
        ),
        (
            "pairs_rejected",
            "No predeclared ETF pair passed every cointegration feasibility gate.",
            "historical_development",
            SOURCE_FILES["day14_pairs"],
            "complete",
            "eligible_pairs=0/3",
        ),
        (
            "slow_reversion_positive",
            "Slow equal-weight OU/VWAP returned 0.43% in the one-time locked 2026 final test.",
            "locked_final_test",
            SOURCE_FILES["day25_locked_performance"],
            "complete",
            f"cumulative_return={outcome_by_id['ou_vwap_slow_equal_weight_locked']['cumulative_return']:.12g}",
        ),
        (
            "slow_reversion_inconclusive",
            "Slow OU/VWAP remains statistically inconclusive and unpromoted.",
            "chronological_out_of_sample",
            SOURCE_FILES["day17_inference"],
            "complete",
            f"hac_t={inference['hac_t']:.12g}; dsr={inference['deflated_sharpe_probability']:.12g}",
        ),
        (
            "allocation_not_rescue",
            "All three causal-timing allocation rules lost money over the common walk-forward panel.",
            "chronological_out_of_sample",
            SOURCE_FILES["day16_allocation"],
            "complete",
            "negative_rules=3/3; trend_timing=next_bar_open_overnight_flat_v1",
        ),
        (
            "operational_controls",
            "Synthetic/offline order, reconciliation, and operations controls passed their frozen known-answer checks.",
            "operational_reproducibility",
            SOURCE_FILES["day23_operations"],
            "complete",
            f"day19={operations[1]['result']}; day20={operations[2]['result']}; day23={operations[4]['result']}",
        ),
        (
            "live_read_only_abort",
            "The Day 21 live paper probe aborted without an order when safety gates did not all pass.",
            "live_read_only",
            SOURCE_FILES["day21_read_only"],
            "complete",
            str(operations[3]["result"]),
        ),
        (
            "live_execution_pending",
            "Empirical Day 22 paper fill and slippage results remain pending.",
            "prospective_live_calibration",
            "artifacts/day22/live_campaign/activation_manifest.json",
            "provisional",
            "no empirical slot evidence included",
        ),
        (
            "annual_regime_diagnostic",
            "All four model/comparator rows beat their relevant price-only annual benchmark only in 2022; this is a post-hoc diagnostic, not a selection result.",
            "post_hoc_development_diagnostic",
            SOURCE_FILES["annual_regime_comparison"],
            "complete",
            "complete_years=2022-2025; benchmark_dividends=false; locked_2026=false; selection=false",
        ),
        (
            "profitability_not_established",
            "Deployable profitability has not been established.",
            "locked_final_test",
            SOURCE_FILES["day25_locked_performance"],
            "complete",
            "locked slow reversion gain=0.43%; both locked trend results negative; empirical execution absent",
        ),
    )
    return tuple(
        {
            "claim_order": index,
            "claim_id": values[0],
            "headline_claim": values[1],
            "evidence_class": values[2],
            "source_file": values[3],
            "status": values[4],
            "value_context": values[5],
        }
        for index, values in enumerate(claims, start=1)
    )


def chart_inventory(project_root: str | Path) -> tuple[dict[str, object], ...]:
    """Return an auditable map of the exact three visible report charts."""

    artifact = build_report_artifact(project_root)
    charts = artifact["manifest"]["charts"]  # type: ignore[index]
    datasets = artifact["snapshot"]["datasets"]  # type: ignore[index]
    source_by_id = {source["id"]: source["path"] for source in artifact["sources"]}  # type: ignore[index]
    return tuple(
        {
            "chart_order": index,
            "chart_id": chart["id"],
            "question": chart["question"],
            "source_file": source_by_id[chart["sourceId"]],
            "dataset": chart["dataset"],
            "row_count": len(datasets[chart["dataset"]]),
            "x_field": chart["encodings"]["x"]["field"],
            "y_field": chart["encodings"]["y"]["field"],
            "color_field": chart["encodings"]["color"]["field"],
        }
        for index, chart in enumerate(charts, start=1)
    )


def csv_bytes(rows: Iterable[Mapping[str, object]], columns: tuple[str, ...]) -> bytes:
    """Serialize a frozen CSV schema deterministically."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != columns:
            raise ValueError("Day 24 CSV artifact schema changed.")
        writer.writerow(
            {
                key: (
                    "true"
                    if value is True
                    else "false"
                    if value is False
                    else ""
                    if value is None
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def json_bytes(value: object) -> bytes:
    """Serialize a canonical JSON artifact deterministically."""

    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def source_notes(project_root: str | Path) -> bytes:
    """Return deterministic source-verification notes for the bundle."""

    root = Path(project_root)
    manifests = verify_source_manifests(root)
    lines = [
        "# Day 24 Source Notes",
        "",
        "All quantitative report claims are replayed from frozen repository artifacts.",
        "The report reads aggregate locked-test performance and methodology after the",
        "separately authorized one-time run. Raw locked bars are not embedded in the",
        "report snapshot. No credential, account identifier, or mutable live-campaign",
        "slot result was read into the report.",
        "",
        "## Verified source manifests",
        "",
        "| Manifest | Files verified | Manifest SHA-256 |",
        "|---|---:|---|",
    ]
    for item in manifests:
        lines.append(
            f"| `{item['manifest']}` | {item['verified_files']} | `{item['manifest_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Direct report sources",
            "",
            "| Source | Repository-relative path | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for source_id in SOURCE_FILES:
        relative = SOURCE_FILES[source_id]
        lines.append(
            f"| {SOURCE_LABELS[source_id]} | `{relative}` | `{_sha256(root / relative)}` |"
        )
    lines.extend(
        [
            "",
            "## Visual evidence map note",
            "",
            "The three locked outcomes are shown as exact table rows rather than a new",
            "chart because three endpoint values are too sparse to establish a trend or",
            "distribution. Six native charts cover annual walk-forward stability,",
            "cross-market robustness, reversion cost sensitivity, annual price-only",
            "benchmark comparisons, and the incremental short-sleeve effect. The three",
            "annual diagnostic charts are explicitly post-hoc, exclude the partial 2026",
            "lockbox, and have adjacent interpretation and caveat prose.",
            "",
            "## Governing external reference",
            "",
            "The January 2026 `CQF Final Project Brief - Jan 26 v.4.pdf` was reviewed",
            "for the Algorithmic Trading topic, report structure, numerical-method",
            "documentation, execution, containerization, and submission requirements.",
            "The machine-local source path is intentionally excluded from report metadata.",
            "The visible bibliography contains the governing brief, eleven primary journal",
            "references with DOI links, and three official Alpaca documentation pages.",
            "Publisher, DOI, journal, and vendor pages were checked on 2026-08-10.",
            "",
            "## Evidence boundary",
            "",
            "Day 22 `campaign_summary.csv` is a synthetic known-answer fixture. The",
            "expired campaign produced no fills. The report snapshot is marked `partial`",
            "until empirical fill and slippage evidence is available. The one-time 2026",
            "locked final test is complete, immutable, and reported without retuning.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")
