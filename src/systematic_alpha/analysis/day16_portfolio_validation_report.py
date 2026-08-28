"""Deterministic reporting for Day 16 portfolio allocation validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
import pandas as pd

from systematic_alpha.analysis.portfolio_allocation_validation import (
    AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
    ALLOCATION_COST_RATE,
    ALLOCATION_DIAGNOSTIC_COLUMNS,
    ALLOCATION_RULES,
    ALLOCATION_WEIGHT_COLUMNS,
    EXPECTED_CANONICAL_FOLD_SESSIONS,
    FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
    HISTORICAL_VAR_QUANTILE,
    MAXIMUM_WEIGHT,
    MINIMUM_VARIANCE_COVARIANCE_COLUMNS,
    PORTFOLIO_RETURN_PANEL_COLUMNS,
    SLSQP_FTOL,
    SLSQP_MAXITER,
    SOLVER_CONSTRAINT_TOLERANCE,
    TRADING_SESSIONS_PER_YEAR,
    WEIGHT_TOLERANCE,
    PortfolioAllocationResults,
    calculate_portfolio_metrics,
)
from systematic_alpha.analysis.strategy_diversification import (
    CONFIGURATION_IDS,
    FREQUENCY,
    FROZEN_SLEEVES,
    SLEEVE_IDS,
    VARIANCE_TOLERANCE,
)
from systematic_alpha.analysis.trend_family_robustness import (
    DEVELOPMENT_DATASET_ID,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    build_walk_forward_folds,
)


DAY16_ARTIFACT_VERSION: Final[str] = "day16_portfolio_validation_v1"

ALLOCATION_WEIGHTS_FILENAME: Final[str] = "allocation_weights.csv"
ALLOCATION_DIAGNOSTICS_FILENAME: Final[str] = "allocation_diagnostics.csv"
FOLD_PORTFOLIO_PERFORMANCE_FILENAME: Final[str] = (
    "fold_portfolio_performance.csv"
)
AGGREGATE_PORTFOLIO_PERFORMANCE_FILENAME: Final[str] = (
    "aggregate_portfolio_performance.csv"
)
PORTFOLIO_RETURN_PANEL_FILENAME: Final[str] = "portfolio_return_panel.csv"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"

APPROVED_DAY16_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    ALLOCATION_WEIGHTS_FILENAME,
    ALLOCATION_DIAGNOSTICS_FILENAME,
    FOLD_PORTFOLIO_PERFORMANCE_FILENAME,
    AGGREGATE_PORTFOLIO_PERFORMANCE_FILENAME,
    PORTFOLIO_RETURN_PANEL_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)

FIXED_ROW_COUNTS: Final[dict[str, int]] = {
    "allocation_weights": 72,
    "allocation_diagnostics": 12,
    "fold_portfolio_performance": 12,
    "aggregate_portfolio_performance": 3,
}


class Day16PortfolioValidationReportError(ValueError):
    """Raised when Day 16 reporting cannot proceed safely."""


def _copy_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return one defensive zero-based result-table copy."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame.copy(deep=True).reset_index(drop=True)


def _validated_table(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
    rows: int | None,
) -> pd.DataFrame:
    """Validate one exact table schema and optional row count."""

    retained = _copy_frame(frame, name=name)
    if tuple(retained.columns) != columns:
        raise Day16PortfolioValidationReportError(
            f"{name} has an unexpected schema."
        )
    if rows is not None and len(retained) != rows:
        raise Day16PortfolioValidationReportError(
            f"{name} must contain exactly {rows} rows."
        )
    return retained


def _freeze_manifest(value: object) -> object:
    """Recursively freeze mutable manifest containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_manifest(item)
                for key, item in deepcopy(dict(value)).items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_manifest(item) for item in deepcopy(value))
    return deepcopy(value)


def _copy_manifest(value: object) -> object:
    """Return mutable copies of recursively frozen manifest values."""

    if isinstance(value, Mapping):
        return {
            str(key): _copy_manifest(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_copy_manifest(item) for item in value]
    return deepcopy(value)


def _iso_date(value: object) -> str:
    """Render one session-like value as a deterministic ISO date."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise Day16PortfolioValidationReportError(
            "Report session values must be valid timestamps."
        ) from exc
    if pd.isna(timestamp):
        raise Day16PortfolioValidationReportError(
            "Report session values cannot be missing."
        )
    return timestamp.strftime("%Y-%m-%d")


def _expected_fold_rule_keys() -> tuple[tuple[str, str], ...]:
    """Return exact fold-major, rule-minor row keys."""

    return tuple(
        (fold.fold_id, rule)
        for fold in build_walk_forward_folds()
        for rule in ALLOCATION_RULES
    )


def _validate_row_order_and_reconciliation(
    *,
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fold_performance: pd.DataFrame,
    aggregate_performance: pd.DataFrame,
    return_panel: pd.DataFrame,
    covariances: pd.DataFrame,
) -> None:
    """Validate all deterministic order and cross-table relationships."""

    expected_fold_rule = _expected_fold_rule_keys()
    expected_weight_keys = tuple(
        (fold_id, rule, sleeve_id, sleeve_order)
        for fold_id, rule in expected_fold_rule
        for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1)
    )
    actual_weight_keys = tuple(
        weights[
            ["fold_id", "allocation_rule", "sleeve_id", "sleeve_order"]
        ].itertuples(index=False, name=None)
    )
    if actual_weight_keys != expected_weight_keys:
        raise Day16PortfolioValidationReportError(
            "allocation_weights does not follow frozen fold/rule/sleeve order."
        )

    for name, table in (
        ("allocation_diagnostics", diagnostics),
        ("fold_portfolio_performance", fold_performance),
    ):
        actual = tuple(
            table[["fold_id", "allocation_rule"]].itertuples(
                index=False,
                name=None,
            )
        )
        if actual != expected_fold_rule:
            raise Day16PortfolioValidationReportError(
                f"{name} does not follow frozen fold/rule order."
            )
    if tuple(aggregate_performance["allocation_rule"]) != ALLOCATION_RULES:
        raise Day16PortfolioValidationReportError(
            "aggregate_portfolio_performance does not follow rule order."
        )

    expected_covariance_keys = tuple(
        (fold.fold_id, sleeve_id, sleeve_order)
        for fold in build_walk_forward_folds()
        for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1)
    )
    actual_covariance_keys = tuple(
        covariances[["fold_id", "sleeve_id", "sleeve_order"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual_covariance_keys != expected_covariance_keys:
        raise Day16PortfolioValidationReportError(
            "Minimum-variance covariance matrices do not follow frozen order."
        )

    if return_panel.empty:
        raise Day16PortfolioValidationReportError(
            "portfolio_return_panel must not be empty."
        )
    sessions = pd.to_datetime(
        return_panel["session_date"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    if sessions.isna().any() or not sessions.is_monotonic_increasing:
        raise Day16PortfolioValidationReportError(
            "Portfolio return sessions must be complete and chronological."
        )
    if sessions.duplicated().any():
        raise Day16PortfolioValidationReportError(
            "Portfolio return sessions must be unique."
        )
    if sessions.max() >= pd.Timestamp("2026-01-01", tz="UTC"):
        raise Day16PortfolioValidationReportError(
            "Locked or later sessions are forbidden in Day 16 reports."
        )

    expected_fold_ids: list[str] = []
    for fold in build_walk_forward_folds():
        mask = (sessions >= fold.test_start) & (sessions < fold.test_end_exclusive)
        if not mask.any():
            raise Day16PortfolioValidationReportError(
                f"{fold.fold_id} has no portfolio return rows."
            )
        expected_fold_ids.extend([fold.fold_id] * int(mask.sum()))
    if tuple(return_panel["fold_id"]) != tuple(expected_fold_ids):
        raise Day16PortfolioValidationReportError(
            "portfolio_return_panel fold identifiers are not exact."
        )

    for column in ALLOCATION_RULES:
        values = pd.to_numeric(return_panel[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        if not np.isfinite(values).all() or np.less_equal(values, -1.0).any():
            raise Day16PortfolioValidationReportError(
                "Portfolio returns must be finite and strictly greater than -1."
            )

    if not diagnostics["constraint_valid"].astype(bool).all():
        raise Day16PortfolioValidationReportError(
            "Every allocation diagnostic must pass mechanical constraints."
        )
    if not weights["constraint_valid"].astype(bool).all():
        raise Day16PortfolioValidationReportError(
            "Every allocation weight row must pass mechanical constraints."
        )

    metric_columns = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "historical_var_95",
        "historical_es_95",
    )
    prior_weights = {
        rule: np.zeros(len(SLEEVE_IDS), dtype="float64")
        for rule in ALLOCATION_RULES
    }
    for fold_id, rule in expected_fold_rule:
        weight_rows = weights.loc[
            weights["fold_id"].eq(fold_id)
            & weights["allocation_rule"].eq(rule)
        ]
        diagnostic = diagnostics.loc[
            diagnostics["fold_id"].eq(fold_id)
            & diagnostics["allocation_rule"].eq(rule)
        ].iloc[0]
        performance = fold_performance.loc[
            fold_performance["fold_id"].eq(fold_id)
            & fold_performance["allocation_rule"].eq(rule)
        ].iloc[0]
        fold_returns = return_panel.loc[return_panel["fold_id"].eq(fold_id)]
        if len(weight_rows) != len(SLEEVE_IDS):
            raise Day16PortfolioValidationReportError(
                "Every fold/rule must retain exactly six sleeve weights."
            )
        if int(diagnostic["test_sessions"]) != len(fold_returns):
            raise Day16PortfolioValidationReportError(
                "Diagnostic test sessions do not match return-panel rows."
            )
        if int(performance["observations"]) != len(fold_returns):
            raise Day16PortfolioValidationReportError(
                "Fold performance observations do not match return-panel rows."
            )
        vector = pd.to_numeric(
            weight_rows["weight"],
            errors="coerce",
        ).to_numpy(dtype="float64")
        if not np.isfinite(vector).all():
            raise Day16PortfolioValidationReportError(
                "Every allocation weight must be finite."
            )
        weight_sum = float(vector.sum())
        gross_weight = float(np.abs(vector).sum())
        minimum_weight = float(vector.min())
        maximum_weight = float(vector.max())
        herfindahl = float(vector @ vector)
        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_TOLERANCE,
        ):
            raise Day16PortfolioValidationReportError(
                "Every fold/rule weight vector must sum to one."
            )
        if minimum_weight < -WEIGHT_TOLERANCE:
            raise Day16PortfolioValidationReportError(
                "Short allocation weights are forbidden."
            )
        if maximum_weight > MAXIMUM_WEIGHT + WEIGHT_TOLERANCE:
            raise Day16PortfolioValidationReportError(
                "Allocation weights exceed the frozen cap."
            )
        if not math.isclose(
            gross_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_TOLERANCE,
        ):
            raise Day16PortfolioValidationReportError(
                "Gross allocation must equal one."
            )
        if not all(
            isinstance(value, (bool, np.bool_)) and bool(value)
            for value in weight_rows["constraint_valid"]
        ) or not isinstance(
            diagnostic["constraint_valid"],
            (bool, np.bool_),
        ) or not bool(diagnostic["constraint_valid"]):
            raise Day16PortfolioValidationReportError(
                "Constraint status fields must be explicit true booleans."
            )

        reported_statistics = {
            "weight_sum": weight_sum,
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "gross_weight": gross_weight,
            "herfindahl_concentration": herfindahl,
            "effective_sleeve_count": 1.0 / herfindahl,
        }
        for column, expected_value in reported_statistics.items():
            if not math.isclose(
                float(diagnostic[column]),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise Day16PortfolioValidationReportError(
                    f"Allocation diagnostic {column} does not reconcile."
                )
        for column in ("weight_sum", "maximum_weight", "gross_weight"):
            row_values = pd.to_numeric(
                weight_rows[column],
                errors="coerce",
            ).to_numpy(dtype="float64")
            if not np.isfinite(row_values).all() or not np.allclose(
                row_values,
                reported_statistics[column],
                rtol=1e-12,
                atol=1e-15,
            ):
                raise Day16PortfolioValidationReportError(
                    f"Allocation weight field {column} does not reconcile."
                )
        training_sessions = pd.to_numeric(
            weight_rows["training_sessions"],
            errors="coerce",
        ).to_numpy(dtype="float64")
        if not np.isfinite(training_sessions).all() or not np.equal(
            training_sessions,
            float(diagnostic["training_sessions"]),
        ).all():
            raise Day16PortfolioValidationReportError(
                "Training-session counts do not reconcile."
            )

        expected_turnover = float(np.abs(vector - prior_weights[rule]).sum())
        expected_cost = expected_turnover * ALLOCATION_COST_RATE
        if not math.isclose(
            float(diagnostic["allocation_turnover"]),
            expected_turnover,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(diagnostic["allocation_cost"]),
            expected_cost,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Day16PortfolioValidationReportError(
                "Allocation turnover or cost does not reconcile to weights."
            )
        prior_weights[rule] = vector.copy()

        applicable = rule == "constrained_minimum_variance"
        if applicable:
            if diagnostic["covariance_estimator"] != (
                "LedoitWolf(assume_centered=False)"
            ) or diagnostic["solver_status"] != "success":
                raise Day16PortfolioValidationReportError(
                    "Minimum-variance estimator and solver status are invalid."
                )
            shrinkage = float(diagnostic["shrinkage_coefficient"])
            if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
                raise Day16PortfolioValidationReportError(
                    "Minimum-variance shrinkage must be finite within [0, 1]."
                )
        elif (
            diagnostic["covariance_estimator"] != "not_applicable"
            or diagnostic["shrinkage_coefficient"] != ""
            or diagnostic["solver_status"] != "not_applicable"
        ):
            raise Day16PortfolioValidationReportError(
                "Non-applicable estimator fields must use neutral values."
            )

        if not math.isclose(
            float(diagnostic["allocation_turnover"]),
            float(performance["allocation_turnover"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(diagnostic["allocation_cost"]),
            float(performance["allocation_cost"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise Day16PortfolioValidationReportError(
                "Fold allocation costs do not reconcile across result tables."
            )

        expected_metrics = calculate_portfolio_metrics(
            pd.Series(
                fold_returns[rule].to_numpy(dtype="float64"),
                index=sessions.loc[fold_returns.index],
                name=rule,
            )
        )
        if _iso_date(performance["start_session"]) != _iso_date(
            sessions.loc[fold_returns.index].min()
        ) or _iso_date(performance["end_session"]) != _iso_date(
            sessions.loc[fold_returns.index].max()
        ):
            raise Day16PortfolioValidationReportError(
                "Fold performance date boundaries do not reconcile."
            )
        for column in metric_columns:
            if not math.isclose(
                float(performance[column]),
                float(expected_metrics[column]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise Day16PortfolioValidationReportError(
                    f"Fold performance metric {column} does not reconcile."
                )

    for row in aggregate_performance.itertuples(index=False):
        if int(row.observations) != len(return_panel):
            raise Day16PortfolioValidationReportError(
                "Aggregate observations must equal the concatenated test rows."
            )
        expected_metrics = calculate_portfolio_metrics(
            return_panel[row.allocation_rule]
        )
        if _iso_date(row.start_session) != _iso_date(sessions.min()) or (
            _iso_date(row.end_session) != _iso_date(sessions.max())
        ):
            raise Day16PortfolioValidationReportError(
                "Aggregate performance date boundaries do not reconcile."
            )
        for column in metric_columns:
            if not math.isclose(
                float(getattr(row, column)),
                float(expected_metrics[column]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise Day16PortfolioValidationReportError(
                    f"Aggregate performance metric {column} does not reconcile."
                )
        selected = diagnostics.loc[
            diagnostics["allocation_rule"].eq(row.allocation_rule)
        ]
        if not math.isclose(
            float(row.total_allocation_turnover),
            float(selected["allocation_turnover"].sum()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(row.total_allocation_cost),
            float(selected["allocation_cost"].sum()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Day16PortfolioValidationReportError(
                "Aggregate allocation turnover or cost does not reconcile."
            )

    for fold in build_walk_forward_folds():
        matrix_rows = covariances.loc[covariances["fold_id"].eq(fold.fold_id)]
        matrix = matrix_rows.loc[:, list(SLEEVE_IDS)].to_numpy(dtype="float64")
        if (
            matrix.shape != (len(SLEEVE_IDS), len(SLEEVE_IDS))
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15)
        ):
            raise Day16PortfolioValidationReportError(
                "Every Ledoit-Wolf covariance matrix must be finite and symmetric."
            )


def _format_number(value: object, *, digits: int = 6) -> str:
    """Render one required finite result deterministically."""

    number = float(value)
    if not math.isfinite(number):
        raise Day16PortfolioValidationReportError(
            "Report metrics must be finite."
        )
    return f"{number:.{digits}f}"


def _allocation_lines(
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    """Render every fold/rule vector and concentration diagnostic."""

    lines = [
        "| Fold | Rule | Weights in frozen sleeve order | Turnover | Cost | "
        "Max weight | HHI | Effective sleeves |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for fold_id, rule in _expected_fold_rule_keys():
        vector = weights.loc[
            weights["fold_id"].eq(fold_id)
            & weights["allocation_rule"].eq(rule),
            "weight",
        ]
        row = diagnostics.loc[
            diagnostics["fold_id"].eq(fold_id)
            & diagnostics["allocation_rule"].eq(rule)
        ].iloc[0]
        rendered_weights = ", ".join(_format_number(value) for value in vector)
        lines.append(
            f"| {fold_id} | {rule} | {rendered_weights} | "
            f"{_format_number(row['allocation_turnover'])} | "
            f"{_format_number(row['allocation_cost'], digits=10)} | "
            f"{_format_number(row['maximum_weight'])} | "
            f"{_format_number(row['herfindahl_concentration'])} | "
            f"{_format_number(row['effective_sleeve_count'])} |"
        )
    return "\n".join(lines)


def _fold_performance_lines(frame: pd.DataFrame) -> str:
    """Render all twelve fold/rule performance rows."""

    lines = [
        "| Fold | Rule | N | Start | End | Cumulative | Annualized | "
        "Ann. vol. | Sharpe | Max drawdown | VaR 95 | ES 95 | Turnover | Cost |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.allocation_rule} | {int(row.observations)} | "
            f"{_iso_date(row.start_session)} | {_iso_date(row.end_session)} | "
            f"{_format_number(row.cumulative_return)} | "
            f"{_format_number(row.annualized_return)} | "
            f"{_format_number(row.annualized_volatility)} | "
            f"{_format_number(row.sharpe_ratio)} | "
            f"{_format_number(row.maximum_drawdown)} | "
            f"{_format_number(row.historical_var_95)} | "
            f"{_format_number(row.historical_es_95)} | "
            f"{_format_number(row.allocation_turnover)} | "
            f"{_format_number(row.allocation_cost, digits=10)} |"
        )
    return "\n".join(lines)


def _aggregate_performance_lines(frame: pd.DataFrame) -> str:
    """Render all three concatenated out-of-sample rows."""

    lines = [
        "| Rule | N | Start | End | Cumulative | Annualized | Ann. vol. | "
        "Sharpe | Max drawdown | VaR 95 | ES 95 | Total turnover | Total cost |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.allocation_rule} | {int(row.observations)} | "
            f"{_iso_date(row.start_session)} | {_iso_date(row.end_session)} | "
            f"{_format_number(row.cumulative_return)} | "
            f"{_format_number(row.annualized_return)} | "
            f"{_format_number(row.annualized_volatility)} | "
            f"{_format_number(row.sharpe_ratio)} | "
            f"{_format_number(row.maximum_drawdown)} | "
            f"{_format_number(row.historical_var_95)} | "
            f"{_format_number(row.historical_es_95)} | "
            f"{_format_number(row.total_allocation_turnover)} | "
            f"{_format_number(row.total_allocation_cost, digits=10)} |"
        )
    return "\n".join(lines)


def _render_report(
    *,
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fold_performance: pd.DataFrame,
    aggregate_performance: pd.DataFrame,
) -> str:
    """Render the neutral Day 16 interpretation contract."""

    markdown = f"""# Day 16 — Portfolio Allocation and Economic Validation

## 1. Objective and boundary

This development-only evaluation distinguishes statistical diversification
from economic performance. It measures mechanically valid portfolios; it does
not select a preferred rule, claim profitability, or authorize paper or live
orders. Negative returns, Sharpe ratios, and other negative results are valid
outcomes and are retained without reinterpretation.

The Day 14 zero-pair cointegration result is retained, so no mean-reversion
sleeve was added. The locked January–June 2026 period was not accessed.

## 2. Frozen inputs and rule order

The six Day 15 sleeves remain in this exact order:
{', '.join(SLEEVE_IDS)}.

The three allocation rules were predeclared and remain in this exact order:
{', '.join(ALLOCATION_RULES)}.

Every fold estimates weights from training rows only, freezes the vector before
test-return use, and holds those weights fixed throughout that fold. There is
no monthly, quarterly, intrafold, or result-triggered rebalancing.

## 3. Allocation methods and costs

Equal weight assigns exactly 1/6 to every sleeve. Inverse volatility uses
ordinary training-sample standard deviations with ddof=1 and deterministic
water filling at the 0.35 cap. Constrained minimum variance is an optimization rule
using LedoitWolf(assume_centered=False) covariance and the frozen SLSQP settings.
No rule was selected using realized performance.

Sleeve returns already contain the frozen one-basis-point strategy cost.
Allocation turnover is charged separately at one basis point on the first test
session of each fold only; strategy costs are not counted twice.

## 4. Fold allocations and concentration

Weights below are listed in the frozen sleeve order.

{_allocation_lines(weights, diagnostics)}

## 5. Fold economic performance

{_fold_performance_lines(fold_performance)}

All reported portfolio returns and metrics are net of the specified allocation
cost. Historical VaR and Expected Shortfall are non-negative loss measures.

## 6. Concatenated 2022–2025 out-of-sample performance

{_aggregate_performance_lines(aggregate_performance)}

## 7. Mechanical evaluation

The frozen schema, development-only scope, chronological train/test separation,
training-only weight estimation, weight constraints, finite covariance inputs,
successful minimum-variance solver status, finite portfolio returns, complete
row counts, and deterministic artifact contract are satisfied.

The final evaluation_complete field is true. It depends only on mechanical
completeness and does not depend on return, Sharpe ratio, drawdown, VaR,
Expected Shortfall, profitability, or relative performance.

## 8. Interpretation and limitations

These results describe allocation behaviour under three fixed rules. Statistical
diversification does not imply positive economic performance. The constrained
minimum-variance calculation uses optimization, but realized results were not
used to rank or select a rule. Leverage, short weights, borrowing, cost-aware
optimization, expected-return inputs, strategy removal, and parameter tuning
remain outside this evaluation.
"""
    lowered = markdown.lower()
    for forbidden in ("best", "winner", "optimal strategy", "deployment"):
        if forbidden in lowered:
            raise RuntimeError(f"Forbidden report wording detected: {forbidden}.")
    return markdown.rstrip() + "\n"


def _fold_definitions() -> list[dict[str, object]]:
    """Build the exact inherited fold definitions."""

    return [
        {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.strftime("%Y-%m-%d"),
            "train_end_exclusive": fold.train_end_exclusive.strftime("%Y-%m-%d"),
            "test_start": fold.test_start.strftime("%Y-%m-%d"),
            "test_end_exclusive": fold.test_end_exclusive.strftime("%Y-%m-%d"),
            "canonical_training_sessions": EXPECTED_CANONICAL_FOLD_SESSIONS[
                fold.fold_id
            ][0],
            "canonical_test_sessions": EXPECTED_CANONICAL_FOLD_SESSIONS[
                fold.fold_id
            ][1],
        }
        for fold in build_walk_forward_folds()
    ]


def _sleeve_universe() -> list[dict[str, object]]:
    """Build exact ordered sleeve provenance."""

    return [
        {
            "sleeve_id": sleeve.sleeve_id,
            "sleeve_order": order,
            "strategy": sleeve.strategy,
            "symbol": sleeve.symbol,
            "frequency": sleeve.frequency,
            "configuration_id": sleeve.configuration_id,
        }
        for order, sleeve in enumerate(FROZEN_SLEEVES, start=1)
    ]


def _covariance_manifest_records(
    covariances: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> list[dict[str, object]]:
    """Record every complete ordered Ledoit-Wolf covariance matrix."""

    records: list[dict[str, object]] = []
    for fold in build_walk_forward_folds():
        rows = covariances.loc[covariances["fold_id"].eq(fold.fold_id)]
        matrix = rows.loc[:, list(SLEEVE_IDS)].to_numpy(dtype="float64")
        diagnostic = diagnostics.loc[
            diagnostics["fold_id"].eq(fold.fold_id)
            & diagnostics["allocation_rule"].eq(
                "constrained_minimum_variance"
            )
        ].iloc[0]
        records.append(
            {
                "fold_id": fold.fold_id,
                "sleeve_order": list(SLEEVE_IDS),
                "shrinkage_coefficient": float(
                    diagnostic["shrinkage_coefficient"]
                ),
                "covariance_matrix": matrix.tolist(),
            }
        )
    return records


def _build_manifest(
    *,
    diagnostics: pd.DataFrame,
    return_panel_rows: int,
    covariances: pd.DataFrame,
) -> dict[str, object]:
    """Build deterministic Day 16 provenance and mechanical status."""

    row_counts = {
        **FIXED_ROW_COUNTS,
        "portfolio_return_panel": int(return_panel_rows),
    }
    mechanical_gates = {
        "exact_input_schema_and_order": True,
        "development_only_dates": True,
        "finite_input_returns": True,
        "train_test_separation": True,
        "training_only_weight_estimation": True,
        "fixed_weights_within_test_fold": True,
        "finite_weights_and_covariance_inputs": True,
        "weight_constraints": True,
        "minimum_variance_solver_success": True,
        "finite_portfolio_returns": True,
        "complete_result_rows": True,
        "deterministic_artifact_contract": True,
    }
    manifest: dict[str, object] = {
        "report_id": "day16_portfolio_validation",
        "artifact_version": DAY16_ARTIFACT_VERSION,
        "schema_version": 1,
        "artifact_filenames": list(APPROVED_DAY16_ARTIFACT_NAMES),
        "development_only": True,
        "dataset_id": DEVELOPMENT_DATASET_ID,
        "frequency": FREQUENCY,
        "development_start": "2020-01-02",
        "development_end": "2025-12-31",
        "locked_period_start": "2026-01-02",
        "locked_period_end": "2026-06-30",
        "locked_period_accessed": False,
        "day14_eligible_cointegration_pairs": 0,
        "sleeve_universe": _sleeve_universe(),
        "sleeve_order": list(SLEEVE_IDS),
        "strategy_configuration_ids": dict(CONFIGURATION_IDS),
        "allocation_rule_order": list(ALLOCATION_RULES),
        "fold_definitions": _fold_definitions(),
        "canonical_expected_portfolio_return_rows": 1003,
        "panel_rebuild_method": (
            "rebuild through Day 15 analysis from selected canonical bars"
        ),
        "alignment_method": "exact common session dates without filling",
        "strategy_cost_convention": (
            "one basis point already embedded in sleeve net_strategy_return"
        ),
        "allocation_cost_rate": ALLOCATION_COST_RATE,
        "allocation_cost_timing": "first test session of each fold only",
        "annualization_sessions": TRADING_SESSIONS_PER_YEAR,
        "risk_free_rate": 0.0,
        "historical_var_return_quantile": HISTORICAL_VAR_QUANTILE,
        "variance_tolerance": VARIANCE_TOLERANCE,
        "weight_tolerance": WEIGHT_TOLERANCE,
        "maximum_weight": MAXIMUM_WEIGHT,
        "minimum_variance_solver": {
            "method": "SLSQP",
            "initial_weights": [1.0 / len(SLEEVE_IDS)] * len(SLEEVE_IDS),
            "analytical_gradient": "2 * covariance * weights",
            "ftol": SLSQP_FTOL,
            "maxiter": SLSQP_MAXITER,
            "constraint_tolerance": SOLVER_CONSTRAINT_TOLERANCE,
            "random_initialization": False,
        },
        "minimum_variance_covariance_estimates": (
            _covariance_manifest_records(covariances, diagnostics)
        ),
        "mechanical_gates": mechanical_gates,
        "row_counts": row_counts,
        "evaluation_complete": bool(all(mechanical_gates.values())),
        "artifact_sha256": {},
        "forward_fill_used": False,
        "backward_fill_used": False,
        "interpolation_used": False,
        "strategy_cost_counted_twice": False,
        "expected_return_inputs_used": False,
        "cost_aware_optimization_used": False,
        "intrafold_rebalancing_used": False,
        "ranking_performed": False,
        "winner_selection_performed": False,
        "sleeve_removal_performed": False,
        "leverage_used": False,
        "short_allocation_used": False,
        "borrowing_used": False,
        "profitability_gate_used": False,
        "paper_or_live_orders_submitted": False,
    }
    _manifest_bytes(manifest)
    return manifest


@dataclass(frozen=True, slots=True)
class Day16PortfolioValidationReport:
    """Immutable Day 16 report and exact artifact evidence."""

    allocation_weights: pd.DataFrame
    allocation_diagnostics: pd.DataFrame
    fold_portfolio_performance: pd.DataFrame
    aggregate_portfolio_performance: pd.DataFrame
    portfolio_return_panel: pd.DataFrame
    manifest: Mapping[str, object]
    report: str

    def __post_init__(self) -> None:
        """Defensively retain every table, manifest field, and text byte."""

        for name in (
            "allocation_weights",
            "allocation_diagnostics",
            "fold_portfolio_performance",
            "aggregate_portfolio_performance",
            "portfolio_return_panel",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(getattr(self, name), name=name),
            )
        object.__setattr__(self, "manifest", _freeze_manifest(self.manifest))
        object.__setattr__(self, "report", str(self.report).rstrip() + "\n")

    def copy_allocation_weights(self) -> pd.DataFrame:
        return self.allocation_weights.copy(deep=True)

    def copy_allocation_diagnostics(self) -> pd.DataFrame:
        return self.allocation_diagnostics.copy(deep=True)

    def copy_fold_portfolio_performance(self) -> pd.DataFrame:
        return self.fold_portfolio_performance.copy(deep=True)

    def copy_aggregate_portfolio_performance(self) -> pd.DataFrame:
        return self.aggregate_portfolio_performance.copy(deep=True)

    def copy_portfolio_return_panel(self) -> pd.DataFrame:
        return self.portfolio_return_panel.copy(deep=True)

    def copy_manifest(self) -> dict[str, object]:
        copied = _copy_manifest(self.manifest)
        if not isinstance(copied, dict):
            raise TypeError("Copied manifest must be a dictionary.")
        return copied


def build_day16_portfolio_validation_report(
    results: PortfolioAllocationResults,
) -> Day16PortfolioValidationReport:
    """Validate analysis evidence and build the neutral Day 16 report."""

    if not isinstance(results, PortfolioAllocationResults):
        raise TypeError("results must be a PortfolioAllocationResults object.")
    if not results.evaluation_complete:
        raise Day16PortfolioValidationReportError(
            "Day 16 analysis must be mechanically complete before reporting."
        )

    weights = _validated_table(
        results.allocation_weights,
        name="allocation_weights",
        columns=ALLOCATION_WEIGHT_COLUMNS,
        rows=FIXED_ROW_COUNTS["allocation_weights"],
    )
    diagnostics = _validated_table(
        results.allocation_diagnostics,
        name="allocation_diagnostics",
        columns=ALLOCATION_DIAGNOSTIC_COLUMNS,
        rows=FIXED_ROW_COUNTS["allocation_diagnostics"],
    )
    fold_performance = _validated_table(
        results.fold_portfolio_performance,
        name="fold_portfolio_performance",
        columns=FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
        rows=FIXED_ROW_COUNTS["fold_portfolio_performance"],
    )
    aggregate_performance = _validated_table(
        results.aggregate_portfolio_performance,
        name="aggregate_portfolio_performance",
        columns=AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
        rows=FIXED_ROW_COUNTS["aggregate_portfolio_performance"],
    )
    return_panel = _validated_table(
        results.portfolio_return_panel,
        name="portfolio_return_panel",
        columns=PORTFOLIO_RETURN_PANEL_COLUMNS,
        rows=None,
    )
    covariances = _validated_table(
        results.minimum_variance_covariances,
        name="minimum_variance_covariances",
        columns=MINIMUM_VARIANCE_COVARIANCE_COLUMNS,
        rows=4 * len(SLEEVE_IDS),
    )
    _validate_row_order_and_reconciliation(
        weights=weights,
        diagnostics=diagnostics,
        fold_performance=fold_performance,
        aggregate_performance=aggregate_performance,
        return_panel=return_panel,
        covariances=covariances,
    )
    manifest = _build_manifest(
        diagnostics=diagnostics,
        return_panel_rows=len(return_panel),
        covariances=covariances,
    )
    markdown = _render_report(
        weights=weights,
        diagnostics=diagnostics,
        fold_performance=fold_performance,
        aggregate_performance=aggregate_performance,
    )
    return Day16PortfolioValidationReport(
        allocation_weights=weights,
        allocation_diagnostics=diagnostics,
        fold_portfolio_performance=fold_performance,
        aggregate_portfolio_performance=aggregate_performance,
        portfolio_return_panel=return_panel,
        manifest=manifest,
        report=markdown,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize one exact evidence table deterministically."""

    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%d",
        na_rep="",
    ).encode("utf-8")


def _report_bytes(report: str) -> bytes:
    """Serialize Markdown with exactly one final newline."""

    return (report.rstrip() + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    """Return one SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def _artifact_payloads(
    report: Day16PortfolioValidationReport,
) -> dict[str, bytes]:
    """Build all approved non-manifest bytes in fixed order."""

    return {
        ALLOCATION_WEIGHTS_FILENAME: _csv_bytes(report.allocation_weights),
        ALLOCATION_DIAGNOSTICS_FILENAME: _csv_bytes(
            report.allocation_diagnostics
        ),
        FOLD_PORTFOLIO_PERFORMANCE_FILENAME: _csv_bytes(
            report.fold_portfolio_performance
        ),
        AGGREGATE_PORTFOLIO_PERFORMANCE_FILENAME: _csv_bytes(
            report.aggregate_portfolio_performance
        ),
        PORTFOLIO_RETURN_PANEL_FILENAME: _csv_bytes(
            report.portfolio_return_panel
        ),
        REPORT_FILENAME: _report_bytes(report.report),
    }


def _manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Serialize strict deterministic JSON with one final newline."""

    try:
        text = json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Day16PortfolioValidationReportError(
            "Manifest must contain finite JSON-compatible values."
        ) from exc
    return (text.rstrip() + "\n").encode("utf-8")


def _validate_output_directory(
    output_directory: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Validate the exact Day 16 destination before any write."""

    if not isinstance(output_directory, (str, Path)):
        raise TypeError("output_directory must be a path.")
    if isinstance(output_directory, str) and not output_directory.strip():
        raise ValueError("output_directory cannot be empty.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    directory = Path(output_directory)
    if directory.name != "day16":
        raise ValueError("output_directory must have the final name 'day16'.")
    if directory.is_symlink():
        raise ValueError("output_directory cannot be a symbolic link.")
    if directory.exists():
        if not directory.is_dir():
            raise ValueError("Day 16 output exists but is not a directory.")
        if not overwrite:
            raise FileExistsError(f"Day 16 output already exists: {directory}.")
    return directory


def _replace_directory(*, staged: Path, destination: Path) -> None:
    """Atomically replace the destination with rollback protection."""

    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=".day16-backup-", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def write_day16_portfolio_validation_artifacts(
    report: Day16PortfolioValidationReport,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly seven deterministic Day 16 artifacts atomically."""

    if not isinstance(report, Day16PortfolioValidationReport):
        raise TypeError("report must be a Day16PortfolioValidationReport.")
    directory = _validate_output_directory(
        output_directory,
        overwrite=overwrite,
    )
    directory.parent.mkdir(parents=True, exist_ok=True)

    payloads = _artifact_payloads(report)
    expected_non_manifest = set(APPROVED_DAY16_ARTIFACT_NAMES) - {
        MANIFEST_FILENAME
    }
    if set(payloads) != expected_non_manifest:
        raise RuntimeError("Day 16 artifact payload set is incomplete.")
    artifact_hashes = {
        name: _sha256_bytes(payload)
        for name, payload in payloads.items()
    }
    manifest = report.copy_manifest()
    manifest["artifact_sha256"] = {
        name: artifact_hashes[name]
        for name in sorted(artifact_hashes)
    }
    expected_payloads = {
        **payloads,
        MANIFEST_FILENAME: _manifest_bytes(manifest),
    }

    with tempfile.TemporaryDirectory(
        prefix=".day16-stage-",
        dir=directory.parent,
    ) as temporary:
        staged = Path(temporary) / "day16"
        staged.mkdir()
        for name in APPROVED_DAY16_ARTIFACT_NAMES:
            (staged / name).write_bytes(expected_payloads[name])

        staged_entries = tuple(item.name for item in staged.iterdir())
        if set(staged_entries) != set(APPROVED_DAY16_ARTIFACT_NAMES):
            raise RuntimeError("Staged Day 16 artifact set is incomplete.")
        if any(not (staged / name).is_file() for name in staged_entries):
            raise RuntimeError("Staged Day 16 entries must all be files.")
        for name, expected in expected_payloads.items():
            if (staged / name).read_bytes() != expected:
                raise RuntimeError(f"Staged Day 16 bytes differ for {name}.")
        for name, digest in artifact_hashes.items():
            if _sha256_bytes((staged / name).read_bytes()) != digest:
                raise RuntimeError(f"Staged Day 16 hash mismatch for {name}.")

        _replace_directory(staged=staged, destination=directory)

    final_entries = tuple(item.name for item in directory.iterdir())
    if set(final_entries) != set(APPROVED_DAY16_ARTIFACT_NAMES):
        raise RuntimeError("Final Day 16 artifact set is not approved.")
    for name, expected in expected_payloads.items():
        path = directory / name
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"Final Day 16 bytes differ for {name}.")

    return tuple(
        directory / name for name in APPROVED_DAY16_ARTIFACT_NAMES
    )
