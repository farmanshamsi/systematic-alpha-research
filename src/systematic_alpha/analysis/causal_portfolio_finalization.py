"""Rebuild the six-sleeve portfolio evidence under final causal trend timing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, Mapping

import numpy as np
import pandas as pd

from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.portfolio_allocation_validation import (
    PortfolioAllocationResults,
    analyze_portfolio_allocation_panel,
)
from systematic_alpha.analysis.strategy_diversification import (
    SLEEVE_IDS,
    StrategyDiversificationResults,
    analyze_strategy_diversification_panel,
    build_exact_return_panel,
    compound_intraday_returns,
)
from systematic_alpha.analysis.trend_methodology_finalization import (
    FINAL_TIMING,
    apply_next_open_overnight_flat,
    build_model_observations,
    prepare_development_bars,
)
from systematic_alpha.analysis.trend_family_walk_forward import REQUIRED_INPUT_SYMBOLS


SPECIFICATION_VERSION: Final[str] = "day25_causal_portfolio_finalization_v1"
COST_BPS_PER_TURNOVER: Final[float] = 1.0
MODEL_BY_STRATEGY: Final[dict[str, str]] = {
    "trend_ratio": "price_ratio_long_short_neutral",
    "ema_macd": "ema_macd_long_short_neutral",
}

SLEEVE_RETURNS_FILENAME: Final[str] = "sleeve_session_returns.csv"
DIVERSIFICATION_FILENAME: Final[str] = "diversification_summary.csv"
FEASIBILITY_FILENAME: Final[str] = "ensemble_feasibility.csv"
WEIGHTS_FILENAME: Final[str] = "allocation_weights.csv"
ALLOCATION_DIAGNOSTICS_FILENAME: Final[str] = "allocation_diagnostics.csv"
FOLD_PERFORMANCE_FILENAME: Final[str] = "fold_portfolio_performance.csv"
AGGREGATE_PERFORMANCE_FILENAME: Final[str] = "aggregate_portfolio_performance.csv"
PORTFOLIO_RETURNS_FILENAME: Final[str] = "portfolio_return_panel.csv"
METHODOLOGY_FILENAME: Final[str] = "methodology.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SLEEVE_RETURNS_FILENAME,
    DIVERSIFICATION_FILENAME,
    FEASIBILITY_FILENAME,
    WEIGHTS_FILENAME,
    ALLOCATION_DIAGNOSTICS_FILENAME,
    FOLD_PERFORMANCE_FILENAME,
    AGGREGATE_PERFORMANCE_FILENAME,
    PORTFOLIO_RETURNS_FILENAME,
    METHODOLOGY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)


class CausalPortfolioFinalizationError(ValueError):
    """Raised when causal portfolio finalization cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class CausalPortfolioFinalizationResults:
    """Final-timing sleeve, diversification, and allocation evidence."""

    sleeve_session_returns: pd.DataFrame
    diversification: StrategyDiversificationResults
    allocation: PortfolioAllocationResults


def _model_and_symbol(sleeve_id: str) -> tuple[str, str, str]:
    for strategy, model_id in MODEL_BY_STRATEGY.items():
        prefix = f"{strategy}_"
        if sleeve_id.startswith(prefix):
            symbol = sleeve_id.removeprefix(prefix).upper()
            if symbol not in REQUIRED_INPUT_SYMBOLS:
                break
            return strategy, model_id, symbol
    raise CausalPortfolioFinalizationError(f"Unknown frozen sleeve: {sleeve_id!r}.")


def build_causal_session_returns(features: pd.DataFrame) -> pd.DataFrame:
    """Build six exact daily sleeve returns from final-timing observations."""

    rows: list[dict[str, object]] = []
    for sleeve_id in SLEEVE_IDS:
        strategy, model_id, symbol = _model_and_symbol(sleeve_id)
        symbol_features = (
            features.loc[features["symbol"].eq(symbol)]
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        if symbol_features.empty:
            raise CausalPortfolioFinalizationError(
                f"No feature observations exist for {symbol}."
            )
        observations = build_model_observations(
            symbol_features,
            model_id=model_id,
        )
        timed = apply_next_open_overnight_flat(
            observations,
            cost_bps_per_turnover=COST_BPS_PER_TURNOVER,
        )
        if timed.loc[timed["is_session_close"], "ending_position"].ne(0).any():
            raise RuntimeError("A final-timing sleeve retained overnight exposure.")
        for session_date, group in timed.groupby(
            "session_date", observed=True, sort=True
        ):
            rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "strategy": strategy,
                    "model_id": model_id,
                    "symbol": symbol,
                    "timing_convention": FINAL_TIMING,
                    "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
                    "session_date": session_date,
                    "session_return": compound_intraday_returns(
                        group["net_strategy_return"]
                    ),
                    "observations": int(len(group)),
                    "turnover": float(group["turnover"].sum()),
                }
            )
    result = pd.DataFrame.from_records(rows)
    expected_rows = result["session_date"].nunique() * len(SLEEVE_IDS)
    if len(result) != expected_rows:
        raise CausalPortfolioFinalizationError(
            "Every sleeve must have exactly the same session calendar."
        )
    if not np.isfinite(result["session_return"].to_numpy(dtype="float64")).all():
        raise CausalPortfolioFinalizationError("Sleeve returns must be finite.")
    return result


def run_causal_portfolio_finalization(
    bars: pd.DataFrame,
) -> CausalPortfolioFinalizationResults:
    """Run final-timing diversification and the frozen three allocation rules."""

    prepared = prepare_development_bars(bars)
    features = build_return_features(
        prepared,
        expected_symbols=REQUIRED_INPUT_SYMBOLS,
    ).bars
    sleeve_returns = build_causal_session_returns(features)
    panel = build_exact_return_panel(
        sleeve_returns[["sleeve_id", "session_date", "session_return"]]
    )
    diversification = analyze_strategy_diversification_panel(panel)
    allocation = analyze_portfolio_allocation_panel(
        panel,
        require_canonical_counts=True,
    )
    return CausalPortfolioFinalizationResults(
        sleeve_session_returns=sleeve_returns,
        diversification=diversification,
        allocation=allocation,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%d",
        na_rep="",
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).rstrip()
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _format(value: object, *, percent: bool = False) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100.0:+.2f}%" if percent else f"{number:.4f}"


def _report(results: CausalPortfolioFinalizationResults) -> str:
    full = results.diversification.full_sample_covariance_diagnostics.iloc[0]
    feasibility = results.diversification.ensemble_feasibility.iloc[0]
    aggregate = results.allocation.aggregate_portfolio_performance
    lines = [
        "# Day 25 Causal Portfolio Finalization",
        "",
        "The historical Day 15-16 allocation study is preserved. This addendum "
        "rebuilds its six trend sleeves under the final next-bar-open, "
        "overnight-flat convention at one basis point. It is a dependency "
        "correction, not a new model search, and no locked 2026 row was used.",
        "",
        "## Diversification diagnostics",
        "",
        f"- Maximum absolute full-sample correlation: {_format(full['maximum_absolute_correlation'])}",
        f"- Effective rank: {_format(full['effective_rank'])}",
        f"- Equal-weight diversification ratio: {_format(full['equal_weight_diversification_ratio'])}",
        f"- Frozen mechanical feasibility gate: {bool(feasibility['ensemble_feasible'])}",
        "",
        "## Aggregate 2022-2025 allocation results",
        "",
        "| Allocation rule | Cumulative return | Sharpe | Maximum drawdown |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.allocation_rule} | "
            f"{_format(row.cumulative_return, percent=True)} | "
            f"{_format(row.sharpe_ratio)} | "
            f"{_format(row.maximum_drawdown, percent=True)} |"
        )
    lines.extend(
        [
            "",
            "All three predeclared rules are reported. Diversification is a risk-"
            "combination property, not evidence that weak sleeves become profitable.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _payloads(
    results: CausalPortfolioFinalizationResults,
) -> dict[str, bytes]:
    allocation = results.allocation
    diversification = results.diversification
    methodology = {
        "schema_version": SPECIFICATION_VERSION,
        "timing_convention": FINAL_TIMING,
        "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
        "sleeves": list(SLEEVE_IDS),
        "allocation_rules": list(allocation.aggregate_portfolio_performance["allocation_rule"]),
        "development_end_exclusive": "2026-01-01",
        "locked_2026_data_accessed": False,
        "winner_selected": False,
        "historical_day15_day16_preserved": True,
    }
    return {
        SLEEVE_RETURNS_FILENAME: _csv_bytes(results.sleeve_session_returns),
        DIVERSIFICATION_FILENAME: _csv_bytes(
            diversification.full_sample_covariance_diagnostics
        ),
        FEASIBILITY_FILENAME: _csv_bytes(diversification.ensemble_feasibility),
        WEIGHTS_FILENAME: _csv_bytes(allocation.allocation_weights),
        ALLOCATION_DIAGNOSTICS_FILENAME: _csv_bytes(
            allocation.allocation_diagnostics
        ),
        FOLD_PERFORMANCE_FILENAME: _csv_bytes(
            allocation.fold_portfolio_performance
        ),
        AGGREGATE_PERFORMANCE_FILENAME: _csv_bytes(
            allocation.aggregate_portfolio_performance
        ),
        PORTFOLIO_RETURNS_FILENAME: _csv_bytes(allocation.portfolio_return_panel),
        METHODOLOGY_FILENAME: _json_bytes(methodology),
        REPORT_FILENAME: _report(results).encode("utf-8"),
    }


def write_causal_portfolio_artifacts(
    results: CausalPortfolioFinalizationResults,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact deterministic finalization bundle atomically."""

    directory = Path(output_directory)
    if directory.name != "day25_causal_portfolio_finalization":
        raise CausalPortfolioFinalizationError(
            "Output directory must be named day25_causal_portfolio_finalization."
        )
    if directory.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {directory}.")
    if directory.is_symlink():
        raise CausalPortfolioFinalizationError("Output cannot be a symlink.")
    payloads = _payloads(results)
    expected = set(APPROVED_ARTIFACT_NAMES) - {MANIFEST_FILENAME}
    if set(payloads) != expected:
        raise RuntimeError("Causal portfolio artifact payload is incomplete.")
    manifest = {
        "schema_version": SPECIFICATION_VERSION,
        "artifact_sha256": {
            name: _sha256(payloads[name]) for name in sorted(payloads)
        },
        "row_counts": {
            SLEEVE_RETURNS_FILENAME: len(results.sleeve_session_returns),
            DIVERSIFICATION_FILENAME: 1,
            FEASIBILITY_FILENAME: 1,
            WEIGHTS_FILENAME: len(results.allocation.allocation_weights),
            ALLOCATION_DIAGNOSTICS_FILENAME: len(
                results.allocation.allocation_diagnostics
            ),
            FOLD_PERFORMANCE_FILENAME: len(
                results.allocation.fold_portfolio_performance
            ),
            AGGREGATE_PERFORMANCE_FILENAME: len(
                results.allocation.aggregate_portfolio_performance
            ),
            PORTFOLIO_RETURNS_FILENAME: len(
                results.allocation.portfolio_return_panel
            ),
        },
        "locked_2026_data_accessed": False,
    }
    complete = {**payloads, MANIFEST_FILENAME: _json_bytes(manifest)}
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".day25-portfolio-", dir=directory.parent))
    backup: Path | None = None
    try:
        for name in APPROVED_ARTIFACT_NAMES:
            (staged / name).write_bytes(complete[name])
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix=".day25-portfolio-backup-", dir=directory.parent))
            backup.rmdir()
            os.replace(directory, backup)
        os.replace(staged, directory)
    except Exception:
        if backup is not None and backup.exists() and not directory.exists():
            os.replace(backup, directory)
        if staged.exists():
            shutil.rmtree(staged)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    return tuple(directory / name for name in APPROVED_ARTIFACT_NAMES)
