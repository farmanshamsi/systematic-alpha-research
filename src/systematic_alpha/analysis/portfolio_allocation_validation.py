"""Development-only Day 16 portfolio allocation and validation.

The three allocation rules are predeclared.  This module measures their
mechanical and economic behaviour without ranking rules, selecting a winner,
or authorizing paper or live execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from systematic_alpha.analysis.strategy_diversification import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    SLEEVE_IDS,
    VARIANCE_TOLERANCE,
    StrategyDiversificationResults,
    run_strategy_diversification,
    validate_session_return_panel,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    build_walk_forward_folds,
)


ALLOCATION_RULES: Final[tuple[str, ...]] = (
    "equal_weight",
    "inverse_volatility",
    "constrained_minimum_variance",
)
MAXIMUM_WEIGHT: Final[float] = 0.35
WEIGHT_TOLERANCE: Final[float] = 1e-12
SOLVER_CONSTRAINT_TOLERANCE: Final[float] = 1e-10
SLSQP_FTOL: Final[float] = 1e-12
SLSQP_MAXITER: Final[int] = 10_000
ALLOCATION_COST_RATE: Final[float] = 1.0 / 10_000.0
TRADING_SESSIONS_PER_YEAR: Final[float] = 252.0
HISTORICAL_VAR_QUANTILE: Final[float] = 0.05
FIXED_HOLDINGS_ACCOUNTING_VERSION: Final[str] = (
    "fixed_holdings_fold_rebalance_v1"
)
FIXED_HOLDINGS_WEIGHT_TOLERANCE: Final[float] = 1e-12

EXPECTED_CANONICAL_FOLD_SESSIONS: Final[
    dict[str, tuple[int, int]]
] = {
    "wf_2022": (505, 251),
    "wf_2023": (756, 250),
    "wf_2024": (1006, 252),
    "wf_2025": (1258, 250),
}

ALLOCATION_WEIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "sleeve_order",
    "training_sessions",
    "weight",
    "weight_sum",
    "maximum_weight",
    "gross_weight",
    "constraint_valid",
)

ALLOCATION_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "training_sessions",
    "test_sessions",
    "covariance_estimator",
    "shrinkage_coefficient",
    "solver_status",
    "allocation_turnover",
    "allocation_cost",
    "weight_sum",
    "minimum_weight",
    "maximum_weight",
    "gross_weight",
    "herfindahl_concentration",
    "effective_sleeve_count",
    "constraint_valid",
)

FOLD_PORTFOLIO_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "observations",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "historical_var_95",
    "historical_es_95",
    "allocation_turnover",
    "allocation_cost",
)

AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "allocation_rule",
    "observations",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "historical_var_95",
    "historical_es_95",
    "total_allocation_turnover",
    "total_allocation_cost",
)

PORTFOLIO_RETURN_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "fold_id",
    *ALLOCATION_RULES,
)

MINIMUM_VARIANCE_COVARIANCE_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "sleeve_id",
    "sleeve_order",
    *SLEEVE_IDS,
)

FIXED_HOLDINGS_WEIGHT_PATH_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "sleeve_order",
    "pre_return_weight",
    "post_return_weight",
)

ENDING_FOLD_WEIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "allocation_rule",
    "sleeve_id",
    "sleeve_order",
    "ending_weight",
)


class PortfolioAllocationValidationError(ValueError):
    """Raised when Day 16 analysis cannot proceed safely."""


def _copy_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return a defensive, zero-based copy of one result table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class PortfolioAllocationResults:
    """Defensively retained Day 16 analysis results."""

    allocation_weights: pd.DataFrame
    allocation_diagnostics: pd.DataFrame
    fold_portfolio_performance: pd.DataFrame
    aggregate_portfolio_performance: pd.DataFrame
    portfolio_return_panel: pd.DataFrame
    minimum_variance_covariances: pd.DataFrame
    evaluation_complete: bool

    def __post_init__(self) -> None:
        """Retain independent copies so caller mutation cannot leak in."""

        for name in (
            "allocation_weights",
            "allocation_diagnostics",
            "fold_portfolio_performance",
            "aggregate_portfolio_performance",
            "portfolio_return_panel",
            "minimum_variance_covariances",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "evaluation_complete",
            bool(self.evaluation_complete),
        )

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

    def copy_minimum_variance_covariances(self) -> pd.DataFrame:
        return self.minimum_variance_covariances.copy(deep=True)


@dataclass(frozen=True, slots=True)
class FixedHoldingsPortfolioPath:
    """Defensively retained recursive return and weight paths."""

    gross_portfolio_returns: pd.Series
    pre_return_weights: pd.DataFrame
    post_return_weights: pd.DataFrame
    ending_weights: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.gross_portfolio_returns, pd.Series):
            raise TypeError("gross_portfolio_returns must be a pandas Series.")
        if not isinstance(self.pre_return_weights, pd.DataFrame):
            raise TypeError("pre_return_weights must be a pandas DataFrame.")
        if not isinstance(self.post_return_weights, pd.DataFrame):
            raise TypeError("post_return_weights must be a pandas DataFrame.")
        if not isinstance(self.ending_weights, pd.Series):
            raise TypeError("ending_weights must be a pandas Series.")
        object.__setattr__(
            self,
            "gross_portfolio_returns",
            self.gross_portfolio_returns.copy(deep=True),
        )
        object.__setattr__(
            self,
            "pre_return_weights",
            self.pre_return_weights.copy(deep=True),
        )
        object.__setattr__(
            self,
            "post_return_weights",
            self.post_return_weights.copy(deep=True),
        )
        object.__setattr__(
            self,
            "ending_weights",
            self.ending_weights.copy(deep=True),
        )

    def copy_gross_portfolio_returns(self) -> pd.Series:
        return self.gross_portfolio_returns.copy(deep=True)

    def copy_pre_return_weights(self) -> pd.DataFrame:
        return self.pre_return_weights.copy(deep=True)

    def copy_post_return_weights(self) -> pd.DataFrame:
        return self.post_return_weights.copy(deep=True)

    def copy_ending_weights(self) -> pd.Series:
        return self.ending_weights.copy(deep=True)


@dataclass(frozen=True, slots=True)
class FixedHoldingsPortfolioAllocationResults:
    """Corrected accounting plus the historical-compatible result tables."""

    portfolio_results: PortfolioAllocationResults
    fixed_holdings_weight_path: pd.DataFrame
    ending_fold_weights: pd.DataFrame
    accounting_version: str = FIXED_HOLDINGS_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_results, PortfolioAllocationResults):
            raise TypeError("portfolio_results must be PortfolioAllocationResults.")
        if not isinstance(self.fixed_holdings_weight_path, pd.DataFrame):
            raise TypeError("fixed_holdings_weight_path must be a pandas DataFrame.")
        if not isinstance(self.ending_fold_weights, pd.DataFrame):
            raise TypeError("ending_fold_weights must be a pandas DataFrame.")
        if tuple(self.fixed_holdings_weight_path.columns) != (
            FIXED_HOLDINGS_WEIGHT_PATH_COLUMNS
        ):
            raise PortfolioAllocationValidationError(
                "Fixed-holdings weight-path schema changed."
            )
        if tuple(self.ending_fold_weights.columns) != ENDING_FOLD_WEIGHT_COLUMNS:
            raise PortfolioAllocationValidationError(
                "Ending fold-weight schema changed."
            )
        if self.accounting_version != FIXED_HOLDINGS_ACCOUNTING_VERSION:
            raise PortfolioAllocationValidationError(
                "Fixed-holdings accounting version changed."
            )
        object.__setattr__(
            self,
            "fixed_holdings_weight_path",
            self.fixed_holdings_weight_path.copy(deep=True).reset_index(drop=True),
        )
        object.__setattr__(
            self,
            "ending_fold_weights",
            self.ending_fold_weights.copy(deep=True).reset_index(drop=True),
        )

    @property
    def allocation_weights(self) -> pd.DataFrame:
        return self.portfolio_results.allocation_weights

    @property
    def allocation_diagnostics(self) -> pd.DataFrame:
        return self.portfolio_results.allocation_diagnostics

    @property
    def fold_portfolio_performance(self) -> pd.DataFrame:
        return self.portfolio_results.fold_portfolio_performance

    @property
    def aggregate_portfolio_performance(self) -> pd.DataFrame:
        return self.portfolio_results.aggregate_portfolio_performance

    @property
    def portfolio_return_panel(self) -> pd.DataFrame:
        return self.portfolio_results.portfolio_return_panel

    @property
    def minimum_variance_covariances(self) -> pd.DataFrame:
        return self.portfolio_results.minimum_variance_covariances

    @property
    def evaluation_complete(self) -> bool:
        return self.portfolio_results.evaluation_complete

    def copy_fixed_holdings_weight_path(self) -> pd.DataFrame:
        return self.fixed_holdings_weight_path.copy(deep=True)

    def copy_ending_fold_weights(self) -> pd.DataFrame:
        return self.ending_fold_weights.copy(deep=True)


def _validate_exact_input_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Require the exact Day 15 panel schema, order, calendar, and scope."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")
    if tuple(panel.columns) != SLEEVE_IDS:
        raise PortfolioAllocationValidationError(
            "Day 16 input columns must exactly match the frozen sleeve order."
        )
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise PortfolioAllocationValidationError(
            "Day 16 input requires a UTC DatetimeIndex named session_date."
        )
    if panel.index.name != "session_date":
        raise PortfolioAllocationValidationError(
            "Day 16 input index must be named session_date."
        )
    if panel.index.tz is None or str(panel.index.tz) != "UTC":
        raise PortfolioAllocationValidationError(
            "Day 16 session_date index must use UTC."
        )
    if not panel.index.equals(panel.index.normalize()):
        raise PortfolioAllocationValidationError(
            "Day 16 session_date values must be normalized whole dates."
        )
    if not panel.index.is_monotonic_increasing:
        raise PortfolioAllocationValidationError(
            "Day 16 input sessions must be chronological."
        )
    if panel.index.has_duplicates:
        raise PortfolioAllocationValidationError(
            "Day 16 input sessions must be unique."
        )

    try:
        validated = validate_session_return_panel(panel)
    except (TypeError, ValueError) as exc:
        raise PortfolioAllocationValidationError(
            f"Day 15 session-return validation failed: {exc}"
        ) from exc

    if (
        validated.index.min() < DEVELOPMENT_START
        or validated.index.max() >= DEVELOPMENT_END_EXCLUSIVE
    ):
        raise PortfolioAllocationValidationError(
            "Day 16 input must remain within development-only dates."
        )
    return validated


def _validate_weights(
    weights: np.ndarray,
    *,
    tolerance: float = WEIGHT_TOLERANCE,
) -> np.ndarray:
    """Validate a frozen six-element long-only allocation vector."""

    values = np.asarray(weights, dtype="float64").copy()
    if values.shape != (len(SLEEVE_IDS),):
        raise PortfolioAllocationValidationError(
            "Allocation weights must contain exactly six elements."
        )
    if not np.isfinite(values).all():
        raise PortfolioAllocationValidationError(
            "Allocation weights must be finite."
        )

    weight_sum = float(values.sum())
    gross_weight = float(np.abs(values).sum())
    if abs(weight_sum - 1.0) > tolerance:
        raise PortfolioAllocationValidationError(
            "Allocation weights must sum to one."
        )
    if np.less(values, -tolerance).any():
        raise PortfolioAllocationValidationError(
            "Short allocation weights are forbidden."
        )
    if np.greater(values, MAXIMUM_WEIGHT + tolerance).any():
        raise PortfolioAllocationValidationError(
            "Allocation weights may not exceed the 0.35 cap."
        )
    if abs(gross_weight - 1.0) > tolerance:
        raise PortfolioAllocationValidationError(
            "Gross allocation must equal one; leverage is forbidden."
        )
    return values


def calculate_fixed_holdings_portfolio_path(
    sleeve_returns: pd.DataFrame,
    initial_weights: pd.Series,
    *,
    sleeve_order: tuple[str, ...],
) -> FixedHoldingsPortfolioPath:
    """Recursively account for fixed holdings without intraperiod rebalancing."""

    if not isinstance(sleeve_returns, pd.DataFrame):
        raise TypeError("sleeve_returns must be a pandas DataFrame.")
    if not isinstance(initial_weights, pd.Series):
        raise TypeError("initial_weights must be a pandas Series.")
    if (
        not isinstance(sleeve_order, tuple)
        or not sleeve_order
        or any(not isinstance(value, str) or not value for value in sleeve_order)
        or len(set(sleeve_order)) != len(sleeve_order)
    ):
        raise PortfolioAllocationValidationError(
            "sleeve_order must be a non-empty tuple of unique identifiers."
        )
    if tuple(sleeve_returns.columns) != sleeve_order:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings returns must exactly match the declared sleeve order."
        )
    if tuple(initial_weights.index) != sleeve_order:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings weights must exactly match the declared sleeve order."
        )
    if sleeve_returns.empty:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings returns cannot be empty."
        )
    if sleeve_returns.index.has_duplicates:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings return observations must be unique."
        )
    if not sleeve_returns.index.is_monotonic_increasing:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings return observations must be deterministic and ordered."
        )

    try:
        returns = sleeve_returns.copy(deep=True).apply(
            pd.to_numeric, errors="raise"
        ).astype("float64")
        weights = pd.to_numeric(
            initial_weights.copy(deep=True), errors="raise"
        ).to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings returns and weights must be numeric."
        ) from exc
    return_values = returns.to_numpy(dtype="float64", copy=True)
    if not np.isfinite(return_values).all() or np.less_equal(
        return_values, -1.0
    ).any():
        raise PortfolioAllocationValidationError(
            "Fixed-holdings returns must be finite and strictly greater than -1."
        )
    if not np.isfinite(weights).all():
        raise PortfolioAllocationValidationError(
            "Fixed-holdings initial weights must be finite."
        )
    if np.less(weights, 0.0).any():
        raise PortfolioAllocationValidationError(
            "Fixed-holdings initial weights must be nonnegative."
        )
    if abs(float(weights.sum()) - 1.0) > FIXED_HOLDINGS_WEIGHT_TOLERANCE:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings initial weights must be fully invested."
        )

    row_count = len(returns)
    sleeve_count = len(sleeve_order)
    pre_return = np.empty((row_count, sleeve_count), dtype="float64")
    post_return = np.empty((row_count, sleeve_count), dtype="float64")
    portfolio_returns = np.empty(row_count, dtype="float64")
    current = weights.copy()
    portfolio_wealth = 1.0

    for row_number, sleeve_return in enumerate(return_values):
        pre_return[row_number] = current
        portfolio_return = float(current @ sleeve_return)
        growth = 1.0 + portfolio_return
        if not math.isfinite(growth) or growth <= 0.0:
            raise PortfolioAllocationValidationError(
                "Fixed-holdings portfolio wealth must remain strictly positive."
            )
        updated = current * (1.0 + sleeve_return) / growth
        if (
            not np.isfinite(updated).all()
            or np.less(updated, 0.0).any()
            or abs(float(updated.sum()) - 1.0)
            > FIXED_HOLDINGS_WEIGHT_TOLERANCE
        ):
            raise PortfolioAllocationValidationError(
                "Fixed-holdings post-return weights must be finite, nonnegative, "
                "and sum to one."
            )
        portfolio_wealth *= growth
        if not math.isfinite(portfolio_wealth) or portfolio_wealth <= 0.0:
            raise PortfolioAllocationValidationError(
                "Fixed-holdings portfolio wealth must remain strictly positive."
            )
        portfolio_returns[row_number] = portfolio_return
        post_return[row_number] = updated
        current = updated

    sleeve_terminal_wealth = np.prod(1.0 + return_values, axis=0)
    holdings_terminal_wealth = float(weights @ sleeve_terminal_wealth)
    recursive_terminal_wealth = float(np.prod(1.0 + portfolio_returns))
    if (
        not math.isfinite(holdings_terminal_wealth)
        or holdings_terminal_wealth <= 0.0
        or not math.isfinite(recursive_terminal_wealth)
        or recursive_terminal_wealth <= 0.0
        or not math.isclose(
            holdings_terminal_wealth,
            recursive_terminal_wealth,
            rel_tol=FIXED_HOLDINGS_WEIGHT_TOLERANCE,
            abs_tol=FIXED_HOLDINGS_WEIGHT_TOLERANCE,
        )
    ):
        raise PortfolioAllocationValidationError(
            "Fixed-holdings terminal wealth identity failed."
        )

    return FixedHoldingsPortfolioPath(
        gross_portfolio_returns=pd.Series(
            portfolio_returns,
            index=returns.index.copy(),
            name="gross_portfolio_return",
        ),
        pre_return_weights=pd.DataFrame(
            pre_return,
            index=returns.index.copy(),
            columns=sleeve_order,
        ),
        post_return_weights=pd.DataFrame(
            post_return,
            index=returns.index.copy(),
            columns=sleeve_order,
        ),
        ending_weights=pd.Series(
            current,
            index=pd.Index(sleeve_order, name="sleeve_id"),
            name="ending_weight",
        ),
    )


def calculate_equal_weights() -> np.ndarray:
    """Return the exact predeclared six-sleeve equal-weight vector."""

    return _validate_weights(
        np.full(len(SLEEVE_IDS), 1.0 / len(SLEEVE_IDS), dtype="float64")
    )


def calculate_inverse_volatility_weights(
    training_panel: pd.DataFrame,
) -> np.ndarray:
    """Calculate capped inverse-volatility weights by water filling."""

    validated = _validate_exact_input_panel(training_panel)
    if len(validated) < 2:
        raise PortfolioAllocationValidationError(
            "Inverse volatility requires at least two training sessions."
        )

    volatility = validated.std(axis=0, ddof=1).to_numpy(dtype="float64")
    if (
        not np.isfinite(volatility).all()
        or np.less_equal(volatility, math.sqrt(VARIANCE_TOLERANCE)).any()
    ):
        raise PortfolioAllocationValidationError(
            "Training volatility must be finite and above the inherited "
            "near-zero tolerance."
        )

    raw_scores = np.reciprocal(volatility)
    if not np.isfinite(raw_scores).all() or np.less_equal(raw_scores, 0.0).any():
        raise PortfolioAllocationValidationError(
            "Inverse-volatility raw scores must be finite and positive."
        )

    weights = np.zeros(len(SLEEVE_IDS), dtype="float64")
    uncapped = np.ones(len(SLEEVE_IDS), dtype=bool)
    remaining_mass = 1.0

    for _ in range(len(SLEEVE_IDS) + 1):
        if not uncapped.any():
            break
        active_scores = raw_scores[uncapped]
        candidates = remaining_mass * active_scores / float(active_scores.sum())
        violating = candidates > MAXIMUM_WEIGHT
        active_indices = np.flatnonzero(uncapped)
        if not violating.any():
            weights[active_indices] = candidates
            remaining_mass = 0.0
            break

        capped_indices = active_indices[violating]
        weights[capped_indices] = MAXIMUM_WEIGHT
        uncapped[capped_indices] = False
        remaining_mass = 1.0 - float(weights.sum())
    else:  # pragma: no cover - defensive loop guard
        raise RuntimeError("Inverse-volatility water filling did not converge.")

    if abs(remaining_mass) > WEIGHT_TOLERANCE:
        raise PortfolioAllocationValidationError(
            "Inverse-volatility water filling left unallocated mass."
        )
    return _validate_weights(weights)


def calculate_constrained_minimum_variance_weights(
    training_panel: pd.DataFrame,
) -> tuple[np.ndarray, float, np.ndarray, str]:
    """Fit Ledoit-Wolf covariance and solve the frozen SLSQP problem."""

    validated = _validate_exact_input_panel(training_panel)
    if len(validated) < 2:
        raise PortfolioAllocationValidationError(
            "Minimum variance requires at least two training sessions."
        )
    training_values = validated.to_numpy(dtype="float64", copy=True)

    try:
        estimator = LedoitWolf(assume_centered=False).fit(training_values)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise PortfolioAllocationValidationError(
            "Ledoit-Wolf covariance estimation failed."
        ) from exc

    covariance = np.asarray(estimator.covariance_, dtype="float64").copy()
    shrinkage = float(estimator.shrinkage_)
    if covariance.shape != (len(SLEEVE_IDS), len(SLEEVE_IDS)):
        raise PortfolioAllocationValidationError(
            "Ledoit-Wolf covariance has an unexpected shape."
        )
    if not np.isfinite(covariance).all() or not math.isfinite(shrinkage):
        raise PortfolioAllocationValidationError(
            "Ledoit-Wolf covariance inputs must be finite."
        )
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-15):
        raise PortfolioAllocationValidationError(
            "Ledoit-Wolf covariance must be symmetric."
        )
    if shrinkage < 0.0 or shrinkage > 1.0:
        raise PortfolioAllocationValidationError(
            "Ledoit-Wolf shrinkage must lie within [0, 1]."
        )

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * covariance @ weights

    equality_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda weights: np.ones(len(SLEEVE_IDS), dtype="float64"),
    }
    try:
        solution = minimize(
            objective,
            calculate_equal_weights(),
            jac=gradient,
            method="SLSQP",
            bounds=tuple((0.0, MAXIMUM_WEIGHT) for _ in SLEEVE_IDS),
            constraints=(equality_constraint,),
            options={
                "ftol": SLSQP_FTOL,
                "maxiter": SLSQP_MAXITER,
                "disp": False,
            },
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise PortfolioAllocationValidationError(
            "Constrained minimum-variance optimization failed."
        ) from exc

    solution_values = np.asarray(solution.x, dtype="float64")
    objective_value = float(solution.fun)
    if (
        not bool(solution.success)
        or not np.isfinite(solution_values).all()
        or not math.isfinite(objective_value)
    ):
        raise PortfolioAllocationValidationError(
            "Constrained minimum-variance solver was not successful."
        )
    _validate_weights(
        solution_values,
        tolerance=SOLVER_CONSTRAINT_TOLERANCE,
    )
    weights = _validate_weights(solution_values)
    return weights, shrinkage, covariance, "success"


def calculate_portfolio_metrics(returns: pd.Series) -> dict[str, object]:
    """Calculate the complete frozen Day 16 economic metric set."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if len(returns) < 2:
        raise PortfolioAllocationValidationError(
            "Portfolio metrics require at least two observations."
        )
    try:
        clean = pd.to_numeric(returns.copy(deep=True), errors="raise").astype(
            "float64"
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioAllocationValidationError(
            "Portfolio returns must be numeric."
        ) from exc
    values = clean.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or np.less_equal(values, -1.0).any():
        raise PortfolioAllocationValidationError(
            "Portfolio returns must be finite and strictly greater than -1."
        )

    wealth = np.cumprod(1.0 + values)
    ending_wealth = float(wealth[-1])
    sample_volatility = float(np.std(values, ddof=1))
    if (
        not math.isfinite(ending_wealth)
        or ending_wealth <= 0.0
        or not math.isfinite(sample_volatility)
        or sample_volatility <= 0.0
    ):
        raise PortfolioAllocationValidationError(
            "Portfolio metrics must be finite with positive sample volatility."
        )

    wealth_with_origin = np.concatenate((np.array([1.0]), wealth))
    drawdowns = wealth_with_origin / np.maximum.accumulate(wealth_with_origin) - 1.0
    quantile = float(
        np.quantile(values, HISTORICAL_VAR_QUANTILE, method="linear")
    )
    tail = values[values <= quantile]
    if tail.size == 0:  # pragma: no cover - quantile must select an observation
        raise RuntimeError("Historical Expected Shortfall tail is empty.")

    metrics: dict[str, object] = {
        "observations": int(len(values)),
        "cumulative_return": ending_wealth - 1.0,
        "annualized_return": (
            ending_wealth ** (TRADING_SESSIONS_PER_YEAR / len(values)) - 1.0
        ),
        "annualized_volatility": (
            sample_volatility * math.sqrt(TRADING_SESSIONS_PER_YEAR)
        ),
        "sharpe_ratio": (
            float(np.mean(values))
            / sample_volatility
            * math.sqrt(TRADING_SESSIONS_PER_YEAR)
        ),
        "maximum_drawdown": float(np.min(drawdowns)),
        "historical_var_95": max(0.0, -quantile),
        "historical_es_95": max(0.0, -float(np.mean(tail))),
    }
    numeric_metrics = tuple(
        float(value)
        for key, value in metrics.items()
        if key != "observations"
    )
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise PortfolioAllocationValidationError(
            "All Day 16 performance metrics must be finite."
        )
    return metrics


def _fold_partitions(
    panel: pd.DataFrame,
    *,
    require_canonical_counts: bool,
) -> tuple[tuple[object, pd.DataFrame, pd.DataFrame], ...]:
    """Build the four exact expanding train/test partitions."""

    if not isinstance(require_canonical_counts, bool):
        raise TypeError("require_canonical_counts must be a boolean.")
    partitions: list[tuple[object, pd.DataFrame, pd.DataFrame]] = []
    prior_test_end: pd.Timestamp | None = None
    for fold in build_walk_forward_folds():
        train = panel.loc[
            (panel.index >= fold.train_start)
            & (panel.index < fold.train_end_exclusive)
        ].copy(deep=True)
        test = panel.loc[
            (panel.index >= fold.test_start)
            & (panel.index < fold.test_end_exclusive)
        ].copy(deep=True)
        if len(train) < 2 or len(test) < 2:
            raise PortfolioAllocationValidationError(
                f"{fold.fold_id} requires at least two train and test sessions."
            )
        if train.index.max() >= test.index.min():
            raise PortfolioAllocationValidationError(
                f"{fold.fold_id} training and test sessions overlap."
            )
        if prior_test_end is not None and test.index.min() <= prior_test_end:
            raise PortfolioAllocationValidationError(
                "Day 16 test folds must be chronological and non-overlapping."
            )
        prior_test_end = test.index.max()
        if require_canonical_counts:
            expected_train, expected_test = EXPECTED_CANONICAL_FOLD_SESSIONS[
                fold.fold_id
            ]
            if len(train) != expected_train or len(test) != expected_test:
                raise PortfolioAllocationValidationError(
                    f"{fold.fold_id} canonical session counts must be "
                    f"{expected_train}/{expected_test}; received "
                    f"{len(train)}/{len(test)}."
                )
        partitions.append((fold, train, test))
    return tuple(partitions)


def _constraint_statistics(weights: np.ndarray) -> dict[str, object]:
    """Return constraint and concentration diagnostics for one vector."""

    values = _validate_weights(weights)
    herfindahl = float(values @ values)
    return {
        "weight_sum": float(values.sum()),
        "minimum_weight": float(values.min()),
        "maximum_weight": float(values.max()),
        "gross_weight": float(np.abs(values).sum()),
        "herfindahl_concentration": herfindahl,
        "effective_sleeve_count": 1.0 / herfindahl,
        "constraint_valid": True,
    }


def analyze_portfolio_allocation_panel(
    panel: pd.DataFrame,
    *,
    require_canonical_counts: bool = False,
) -> PortfolioAllocationResults:
    """Evaluate all predeclared rules on the frozen chronological folds."""

    validated = _validate_exact_input_panel(panel)
    partitions = _fold_partitions(
        validated,
        require_canonical_counts=require_canonical_counts,
    )

    weight_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    fold_performance_records: list[dict[str, object]] = []
    covariance_records: list[dict[str, object]] = []
    return_frames: list[pd.DataFrame] = []
    previous_weights = {
        rule: np.zeros(len(SLEEVE_IDS), dtype="float64")
        for rule in ALLOCATION_RULES
    }

    for fold, training_panel, test_panel in partitions:
        minimum_variance = calculate_constrained_minimum_variance_weights(
            training_panel
        )
        weights_by_rule = {
            "equal_weight": calculate_equal_weights(),
            "inverse_volatility": calculate_inverse_volatility_weights(
                training_panel
            ),
            "constrained_minimum_variance": minimum_variance[0],
        }
        shrinkage = minimum_variance[1]
        covariance = minimum_variance[2]
        solver_status = minimum_variance[3]
        for weights in weights_by_rule.values():
            weights.setflags(write=False)

        for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
            covariance_records.append(
                {
                    "fold_id": fold.fold_id,
                    "sleeve_id": sleeve_id,
                    "sleeve_order": sleeve_order,
                    **{
                        column: float(covariance[sleeve_order - 1, column_order])
                        for column_order, column in enumerate(SLEEVE_IDS)
                    },
                }
            )

        test_values = test_panel.to_numpy(dtype="float64", copy=True)
        fold_returns = pd.DataFrame(
            {
                "session_date": test_panel.index.copy(),
                "fold_id": fold.fold_id,
            }
        )
        for rule in ALLOCATION_RULES:
            weights = weights_by_rule[rule]
            statistics = _constraint_statistics(weights)
            turnover = float(np.abs(weights - previous_weights[rule]).sum())
            cost = turnover * ALLOCATION_COST_RATE
            pre_cost_returns = test_values @ weights
            net_returns = np.asarray(pre_cost_returns, dtype="float64").copy()
            net_returns[0] -= cost
            if (
                not np.isfinite(net_returns).all()
                or np.less_equal(net_returns, -1.0).any()
            ):
                raise PortfolioAllocationValidationError(
                    "Post-allocation-cost portfolio returns must be finite "
                    "and strictly greater than -1."
                )
            fold_returns[rule] = net_returns

            for sleeve_order, (sleeve_id, weight) in enumerate(
                zip(SLEEVE_IDS, weights, strict=True),
                start=1,
            ):
                weight_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "allocation_rule": rule,
                        "sleeve_id": sleeve_id,
                        "sleeve_order": sleeve_order,
                        "training_sessions": int(len(training_panel)),
                        "weight": float(weight),
                        "weight_sum": statistics["weight_sum"],
                        "maximum_weight": statistics["maximum_weight"],
                        "gross_weight": statistics["gross_weight"],
                        "constraint_valid": statistics["constraint_valid"],
                    }
                )

            applicable = rule == "constrained_minimum_variance"
            diagnostic_records.append(
                {
                    "fold_id": fold.fold_id,
                    "allocation_rule": rule,
                    "training_sessions": int(len(training_panel)),
                    "test_sessions": int(len(test_panel)),
                    "covariance_estimator": (
                        "LedoitWolf(assume_centered=False)"
                        if applicable
                        else "not_applicable"
                    ),
                    "shrinkage_coefficient": shrinkage if applicable else "",
                    "solver_status": solver_status if applicable else "not_applicable",
                    "allocation_turnover": turnover,
                    "allocation_cost": cost,
                    **statistics,
                }
            )

            metrics = calculate_portfolio_metrics(
                pd.Series(net_returns, index=test_panel.index, name=rule)
            )
            fold_performance_records.append(
                {
                    "fold_id": fold.fold_id,
                    "allocation_rule": rule,
                    **metrics,
                    "start_session": test_panel.index.min(),
                    "end_session": test_panel.index.max(),
                    "allocation_turnover": turnover,
                    "allocation_cost": cost,
                }
            )
            previous_weights[rule] = np.asarray(weights).copy()
        return_frames.append(fold_returns)

    allocation_weights = pd.DataFrame.from_records(
        weight_records,
        columns=ALLOCATION_WEIGHT_COLUMNS,
    )
    allocation_diagnostics = pd.DataFrame.from_records(
        diagnostic_records,
        columns=ALLOCATION_DIAGNOSTIC_COLUMNS,
    )
    fold_performance = pd.DataFrame.from_records(
        fold_performance_records,
        columns=FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
    )
    return_panel = pd.concat(return_frames, ignore_index=True).loc[
        :, list(PORTFOLIO_RETURN_PANEL_COLUMNS)
    ]
    covariance_table = pd.DataFrame.from_records(
        covariance_records,
        columns=MINIMUM_VARIANCE_COVARIANCE_COLUMNS,
    )

    aggregate_records: list[dict[str, object]] = []
    for rule in ALLOCATION_RULES:
        metrics = calculate_portfolio_metrics(return_panel[rule])
        rule_diagnostics = allocation_diagnostics.loc[
            allocation_diagnostics["allocation_rule"].eq(rule)
        ]
        aggregate_records.append(
            {
                "allocation_rule": rule,
                **metrics,
                "start_session": return_panel["session_date"].min(),
                "end_session": return_panel["session_date"].max(),
                "total_allocation_turnover": float(
                    rule_diagnostics["allocation_turnover"].sum()
                ),
                "total_allocation_cost": float(
                    rule_diagnostics["allocation_cost"].sum()
                ),
            }
        )
    aggregate_performance = pd.DataFrame.from_records(
        aggregate_records,
        columns=AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
    )

    expected_counts = (72, 12, 12, 3)
    actual_counts = (
        len(allocation_weights),
        len(allocation_diagnostics),
        len(fold_performance),
        len(aggregate_performance),
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Day 16 result row counts are incomplete: "
            f"{actual_counts} != {expected_counts}."
        )
    if not return_panel["session_date"].is_monotonic_increasing:
        raise PortfolioAllocationValidationError(
            "Out-of-sample portfolio sessions must remain chronological."
        )
    if return_panel["session_date"].duplicated().any():
        raise PortfolioAllocationValidationError(
            "Out-of-sample portfolio sessions must be unique."
        )

    return PortfolioAllocationResults(
        allocation_weights=allocation_weights,
        allocation_diagnostics=allocation_diagnostics,
        fold_portfolio_performance=fold_performance,
        aggregate_portfolio_performance=aggregate_performance,
        portfolio_return_panel=return_panel,
        minimum_variance_covariances=covariance_table,
        evaluation_complete=True,
    )


def analyze_portfolio_allocation_panel_fixed_holdings(
    panel: pd.DataFrame,
    *,
    require_canonical_counts: bool = False,
) -> FixedHoldingsPortfolioAllocationResults:
    """Evaluate frozen targets with one fold-entry rebalance and drifting weights."""

    validated = _validate_exact_input_panel(panel)
    partitions = _fold_partitions(
        validated,
        require_canonical_counts=require_canonical_counts,
    )

    # The frozen historical engine remains the single target-estimation source.
    # Its constant-mix return path is not used by this corrected accounting path.
    target_evidence = analyze_portfolio_allocation_panel(
        validated,
        require_canonical_counts=require_canonical_counts,
    )
    diagnostic_records: list[dict[str, object]] = []
    fold_performance_records: list[dict[str, object]] = []
    return_frames: list[pd.DataFrame] = []
    weight_path_records: list[dict[str, object]] = []
    ending_weight_records: list[dict[str, object]] = []
    previous_ending_weights = {
        rule: np.zeros(len(SLEEVE_IDS), dtype="float64")
        for rule in ALLOCATION_RULES
    }

    for fold, _, test_panel in partitions:
        fold_returns = pd.DataFrame(
            {
                "session_date": test_panel.index.copy(),
                "fold_id": fold.fold_id,
            }
        )
        for rule in ALLOCATION_RULES:
            weight_rows = target_evidence.allocation_weights.loc[
                target_evidence.allocation_weights["fold_id"].eq(fold.fold_id)
                & target_evidence.allocation_weights["allocation_rule"].eq(rule)
            ].sort_values("sleeve_order", kind="stable")
            if tuple(weight_rows["sleeve_id"]) != SLEEVE_IDS:
                raise RuntimeError("Fixed-holdings target sleeve order changed.")
            target = _validate_weights(
                weight_rows["weight"].to_numpy(dtype="float64", copy=True)
            )
            path = calculate_fixed_holdings_portfolio_path(
                test_panel,
                pd.Series(target, index=pd.Index(SLEEVE_IDS, name="sleeve_id")),
                sleeve_order=SLEEVE_IDS,
            )
            turnover = float(
                np.abs(target - previous_ending_weights[rule]).sum()
            )
            cost = turnover * ALLOCATION_COST_RATE
            net_returns = path.gross_portfolio_returns.to_numpy(
                dtype="float64", copy=True
            )
            net_returns[0] -= cost
            if (
                not np.isfinite(net_returns).all()
                or np.less_equal(net_returns, -1.0).any()
            ):
                raise PortfolioAllocationValidationError(
                    "Post-allocation-cost fixed-holdings returns must be finite "
                    "and strictly greater than -1."
                )
            fold_returns[rule] = net_returns

            historical_diagnostic = target_evidence.allocation_diagnostics.loc[
                target_evidence.allocation_diagnostics["fold_id"].eq(fold.fold_id)
                & target_evidence.allocation_diagnostics["allocation_rule"].eq(rule)
            ]
            if len(historical_diagnostic) != 1:
                raise RuntimeError("Fixed-holdings target diagnostic is incomplete.")
            diagnostic = historical_diagnostic.iloc[0].to_dict()
            diagnostic["allocation_turnover"] = turnover
            diagnostic["allocation_cost"] = cost
            diagnostic_records.append(diagnostic)

            metrics = calculate_portfolio_metrics(
                pd.Series(net_returns, index=test_panel.index, name=rule)
            )
            fold_performance_records.append(
                {
                    "fold_id": fold.fold_id,
                    "allocation_rule": rule,
                    **metrics,
                    "start_session": test_panel.index.min(),
                    "end_session": test_panel.index.max(),
                    "allocation_turnover": turnover,
                    "allocation_cost": cost,
                }
            )

            for session_date in test_panel.index:
                pre = path.pre_return_weights.loc[session_date]
                post = path.post_return_weights.loc[session_date]
                for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
                    weight_path_records.append(
                        {
                            "session_date": session_date,
                            "fold_id": fold.fold_id,
                            "allocation_rule": rule,
                            "sleeve_id": sleeve_id,
                            "sleeve_order": sleeve_order,
                            "pre_return_weight": float(pre[sleeve_id]),
                            "post_return_weight": float(post[sleeve_id]),
                        }
                    )
            ending = path.ending_weights.to_numpy(dtype="float64", copy=True)
            for sleeve_order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
                ending_weight_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "allocation_rule": rule,
                        "sleeve_id": sleeve_id,
                        "sleeve_order": sleeve_order,
                        "ending_weight": float(ending[sleeve_order - 1]),
                    }
                )
            previous_ending_weights[rule] = ending
        return_frames.append(fold_returns)

    allocation_diagnostics = pd.DataFrame.from_records(
        diagnostic_records,
        columns=ALLOCATION_DIAGNOSTIC_COLUMNS,
    )
    fold_performance = pd.DataFrame.from_records(
        fold_performance_records,
        columns=FOLD_PORTFOLIO_PERFORMANCE_COLUMNS,
    )
    return_panel = pd.concat(return_frames, ignore_index=True).loc[
        :, list(PORTFOLIO_RETURN_PANEL_COLUMNS)
    ]

    aggregate_records: list[dict[str, object]] = []
    for rule in ALLOCATION_RULES:
        metrics = calculate_portfolio_metrics(return_panel[rule])
        rule_diagnostics = allocation_diagnostics.loc[
            allocation_diagnostics["allocation_rule"].eq(rule)
        ]
        aggregate_records.append(
            {
                "allocation_rule": rule,
                **metrics,
                "start_session": return_panel["session_date"].min(),
                "end_session": return_panel["session_date"].max(),
                "total_allocation_turnover": float(
                    rule_diagnostics["allocation_turnover"].sum()
                ),
                "total_allocation_cost": float(
                    rule_diagnostics["allocation_cost"].sum()
                ),
            }
        )
    aggregate_performance = pd.DataFrame.from_records(
        aggregate_records,
        columns=AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS,
    )
    weight_path = pd.DataFrame.from_records(
        weight_path_records,
        columns=FIXED_HOLDINGS_WEIGHT_PATH_COLUMNS,
    )
    ending_weights = pd.DataFrame.from_records(
        ending_weight_records,
        columns=ENDING_FOLD_WEIGHT_COLUMNS,
    )

    expected_counts = (12, 12, 3, 72)
    actual_counts = (
        len(allocation_diagnostics),
        len(fold_performance),
        len(aggregate_performance),
        len(ending_weights),
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Fixed-holdings result row counts are incomplete: "
            f"{actual_counts} != {expected_counts}."
        )
    if not return_panel["session_date"].is_monotonic_increasing:
        raise PortfolioAllocationValidationError(
            "Fixed-holdings portfolio sessions must remain chronological."
        )
    if return_panel["session_date"].duplicated().any():
        raise PortfolioAllocationValidationError(
            "Fixed-holdings portfolio sessions must be unique."
        )

    compatible_results = PortfolioAllocationResults(
        allocation_weights=target_evidence.copy_allocation_weights(),
        allocation_diagnostics=allocation_diagnostics,
        fold_portfolio_performance=fold_performance,
        aggregate_portfolio_performance=aggregate_performance,
        portfolio_return_panel=return_panel,
        minimum_variance_covariances=(
            target_evidence.copy_minimum_variance_covariances()
        ),
        evaluation_complete=True,
    )
    return FixedHoldingsPortfolioAllocationResults(
        portfolio_results=compatible_results,
        fixed_holdings_weight_path=weight_path,
        ending_fold_weights=ending_weights,
    )


def run_portfolio_allocation(
    bars: pd.DataFrame,
    *,
    require_canonical_counts: bool = True,
) -> PortfolioAllocationResults:
    """Rebuild the Day 15 panel and run the Day 16 allocation study."""

    if not isinstance(require_canonical_counts, bool):
        raise TypeError("require_canonical_counts must be a boolean.")
    day15_results: StrategyDiversificationResults = (
        run_strategy_diversification(bars)
    )
    return analyze_portfolio_allocation_panel(
        day15_results.copy_session_return_panel(),
        require_canonical_counts=require_canonical_counts,
    )
