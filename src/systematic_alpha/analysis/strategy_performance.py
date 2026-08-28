"""Reusable performance metrics for validated simple-return series."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Final, Sequence

import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import make_column_validator


DEFAULT_ANNUALIZATION_FACTOR: Final[float] = 252.0

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "series",
    "observations",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
)


class StrategyPerformanceError(ValueError):
    """Raised when strategy performance cannot be calculated safely."""


_require_columns = make_column_validator(
    StrategyPerformanceError
)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Compact performance statistics for one simple-return series."""

    observations: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def _validate_annualization_factor(
    annualization_factor: Real,
) -> float:
    """Validate and normalize the observations-per-year assumption."""
    if (
        isinstance(annualization_factor, bool)
        or not isinstance(annualization_factor, Real)
    ):
        raise StrategyPerformanceError(
            "annualization_factor must be a finite, "
            "strictly positive real number."
        )

    normalized = float(annualization_factor)

    if not np.isfinite(normalized) or normalized <= 0.0:
        raise StrategyPerformanceError(
            "annualization_factor must be finite and "
            "strictly positive."
        )

    return normalized


def _select_return_series(
    returns: pd.Series | pd.DataFrame,
    *,
    return_column: str | None,
) -> pd.Series:
    """Select one return series without mutating the caller's object."""
    if isinstance(returns, pd.Series):
        if return_column is not None:
            raise StrategyPerformanceError(
                "return_column may be provided only with "
                "a pandas DataFrame."
            )

        return returns.copy()

    if isinstance(returns, pd.DataFrame):
        if not isinstance(return_column, str) or not return_column:
            raise StrategyPerformanceError(
                "A non-empty return_column is required for "
                "pandas DataFrame input."
            )

        _require_columns(
            returns,
            (return_column,),
            context="Strategy performance input",
        )

        return returns[return_column].copy()

    raise StrategyPerformanceError(
        "Returns must be a pandas Series or a selected "
        "pandas DataFrame column."
    )


def _normalize_returns(
    returns: pd.Series | pd.DataFrame,
    *,
    return_column: str | None,
) -> pd.Series:
    """Validate a simple-return series without dropping observations."""
    raw = _select_return_series(
        returns,
        return_column=return_column,
    )

    if raw.empty:
        raise StrategyPerformanceError(
            "Return input cannot be empty."
        )

    if len(raw) < 2:
        raise StrategyPerformanceError(
            "At least two return observations are required."
        )

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    nonnumeric = raw.notna() & numeric.isna()

    if nonnumeric.any():
        bad_rows = nonnumeric[nonnumeric].index.tolist()[:5]
        raise StrategyPerformanceError(
            "Return input contains non-numeric values. "
            f"Example row indices: {bad_rows}."
        )

    if numeric.isna().any():
        raise StrategyPerformanceError(
            "Return input contains missing observations."
        )

    try:
        clean = numeric.astype("float64")
    except (TypeError, ValueError) as exc:
        raise StrategyPerformanceError(
            "Return input must contain real numeric values."
        ) from exc

    if not np.isfinite(
        clean.to_numpy(dtype="float64")
    ).all():
        raise StrategyPerformanceError(
            "Return input must contain only finite values."
        )

    if clean.le(-1.0).any():
        raise StrategyPerformanceError(
            "Simple returns must be strictly greater than -1.0."
        )

    return clean


def _build_wealth_from_returns(
    returns: pd.Series,
) -> pd.Series:
    """Compound validated returns into a wealth index."""
    return (
        returns.add(1.0)
        .cumprod()
        .rename("wealth_index")
    )


def build_wealth_index(
    returns: pd.Series | pd.DataFrame,
    *,
    return_column: str | None = None,
) -> pd.Series:
    """Build compounded wealth from a simple-return series."""
    clean = _normalize_returns(
        returns,
        return_column=return_column,
    )

    return _build_wealth_from_returns(clean)


def calculate_performance_metrics(
    returns: pd.Series | pd.DataFrame,
    *,
    return_column: str | None = None,
    annualization_factor: Real = DEFAULT_ANNUALIZATION_FACTOR,
) -> PerformanceMetrics:
    """Calculate performance statistics for one simple-return series."""
    factor = _validate_annualization_factor(
        annualization_factor
    )

    clean = _normalize_returns(
        returns,
        return_column=return_column,
    )

    wealth = _build_wealth_from_returns(clean)
    observations = len(clean)
    ending_wealth = float(wealth.iloc[-1])
    sample_standard_deviation = float(
        clean.std(ddof=1)
    )

    if clean.eq(clean.iloc[0]).all():
        sample_standard_deviation = 0.0

    annualized_volatility = (
        sample_standard_deviation
        * np.sqrt(factor)
    )

    if sample_standard_deviation == 0.0:
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = float(
            clean.mean()
            / sample_standard_deviation
            * np.sqrt(factor)
        )

    drawdown = (
        wealth
        .div(wealth.cummax())
        .sub(1.0)
    )

    return PerformanceMetrics(
        observations=observations,
        cumulative_return=ending_wealth - 1.0,
        annualized_return=(
            ending_wealth
            ** (factor / observations)
            - 1.0
        ),
        annualized_volatility=float(
            annualized_volatility
        ),
        sharpe_ratio=sharpe_ratio,
        max_drawdown=float(drawdown.min()),
    )


def build_performance_summary(
    frame: pd.DataFrame,
    return_columns: Sequence[str],
    *,
    annualization_factor: Real = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Build an ordered one-row-per-series performance summary."""
    if not isinstance(frame, pd.DataFrame):
        raise StrategyPerformanceError(
            "Performance summary input must be a pandas DataFrame."
        )

    if isinstance(return_columns, (str, bytes)):
        raise StrategyPerformanceError(
            "return_columns must be an ordered collection "
            "of column names."
        )

    try:
        requested = tuple(return_columns)
    except TypeError as exc:
        raise StrategyPerformanceError(
            "return_columns must be an ordered collection "
            "of column names."
        ) from exc

    if any(
        not isinstance(column, str) or not column
        for column in requested
    ):
        raise StrategyPerformanceError(
            "Return-column names must be non-empty strings."
        )

    _require_columns(
        frame,
        requested,
        context="Strategy performance summary input",
    )

    factor = _validate_annualization_factor(
        annualization_factor
    )

    records: list[dict[str, float | int | str]] = []

    for column in requested:
        metrics = calculate_performance_metrics(
            frame,
            return_column=column,
            annualization_factor=factor,
        )

        records.append(
            {
                "series": column,
                "observations": metrics.observations,
                "cumulative_return": (
                    metrics.cumulative_return
                ),
                "annualized_return": (
                    metrics.annualized_return
                ),
                "annualized_volatility": (
                    metrics.annualized_volatility
                ),
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=SUMMARY_COLUMNS,
    )
