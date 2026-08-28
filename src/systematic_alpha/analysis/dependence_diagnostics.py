"""Dependence and finite-tail diagnostics for Day 05."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from systematic_alpha.analysis._validation import (
    make_column_validator,
)


DEFAULT_TAIL_PROBABILITIES = (0.01, 0.05, 0.10)
DEFAULT_ROLLING_WINDOWS = (63, 126)


class DependenceDiagnosticError(ValueError):
    """Raised when dependence diagnostics cannot be calculated."""


_require_columns = make_column_validator(
    DependenceDiagnosticError
)


@dataclass(frozen=True)
class DependenceDiagnosticBundle:
    """Dependence-analysis output tables."""

    pairwise_dependence: pd.DataFrame
    tail_dependence: pd.DataFrame
    rolling_dependence: pd.DataFrame
    regime_dependence: pd.DataFrame
    regime_tail_dependence: pd.DataFrame
    regime_definition: pd.DataFrame




def build_daily_return_panel(
    session_features: pd.DataFrame,
    *,
    return_column: str = "close_to_close_log_return",
) -> pd.DataFrame:
    """Create a synchronized date-by-symbol return panel."""

    _require_columns(
        session_features,
        (
            "symbol",
            "session_date",
            return_column,
        ),
        context="Session return features",
    )

    frame = session_features[
        [
            "symbol",
            "session_date",
            return_column,
        ]
    ].copy()

    frame["symbol"] = (
        frame["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    frame["session_date"] = pd.to_datetime(
        frame["session_date"],
        errors="raise",
    )

    frame[return_column] = pd.to_numeric(
        frame[return_column],
        errors="coerce",
    )

    if frame.duplicated(
        ["symbol", "session_date"]
    ).any():
        raise DependenceDiagnosticError(
            "Session returns contain duplicate "
            "symbol/date observations."
        )

    panel = frame.pivot(
        index="session_date",
        columns="symbol",
        values=return_column,
    ).sort_index()

    panel.columns.name = None

    if panel.shape[1] < 2:
        raise DependenceDiagnosticError(
            "Dependence analysis requires at least two symbols."
        )

    return panel


def _aligned_pair(
    panel: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    *,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Return finite synchronized observations for one pair."""

    pair = panel[
        [symbol_a, symbol_b]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(pair) < minimum_observations:
        raise DependenceDiagnosticError(
            f"Insufficient observations for {symbol_a}/{symbol_b}. "
            f"Required {minimum_observations}; received {len(pair)}."
        )

    return pair


def _pair_dependence_record(
    pair: pd.DataFrame,
    *,
    symbol_a: str,
    symbol_b: str,
    sample_label: str,
) -> dict[str, float | int | str]:
    """Calculate linear and rank dependence."""

    values_a = pair[symbol_a].to_numpy()
    values_b = pair[symbol_b].to_numpy()

    pearson = stats.pearsonr(
        values_a,
        values_b,
    )

    spearman = stats.spearmanr(
        values_a,
        values_b,
    )

    kendall = stats.kendalltau(
        values_a,
        values_b,
    )

    return {
        "sample": sample_label,
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "observations": int(len(pair)),
        "pearson_correlation": float(
            pearson.statistic
        ),
        "pearson_pvalue": float(
            pearson.pvalue
        ),
        "spearman_correlation": float(
            spearman.statistic
        ),
        "spearman_pvalue": float(
            spearman.pvalue
        ),
        "kendall_tau": float(
            kendall.statistic
        ),
        "kendall_pvalue": float(
            kendall.pvalue
        ),
    }


def build_pairwise_dependence_table(
    panel: pd.DataFrame,
    *,
    sample_label: str = "full_development",
) -> pd.DataFrame:
    """Calculate pairwise linear and rank dependence."""

    records = []

    for symbol_a, symbol_b in combinations(
        sorted(panel.columns),
        2,
    ):
        pair = _aligned_pair(
            panel,
            symbol_a,
            symbol_b,
        )

        records.append(
            _pair_dependence_record(
                pair,
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                sample_label=sample_label,
            )
        )

    return pd.DataFrame(records)


def _pseudo_observations(
    values: pd.Series,
) -> pd.Series:
    """Convert continuous observations into rank uniforms."""

    clean = pd.to_numeric(
        values,
        errors="raise",
    )

    n_observations = len(clean)

    if n_observations < 2:
        raise DependenceDiagnosticError(
            "Pseudo-observations require at least two values."
        )

    return (
        clean.rank(
            method="average",
        )
        / (n_observations + 1.0)
    )


def build_tail_dependence_table(
    panel: pd.DataFrame,
    *,
    tail_probabilities: Sequence[float] = (
        DEFAULT_TAIL_PROBABILITIES
    ),
    sample_label: str = "full_development",
) -> pd.DataFrame:
    """Calculate finite-threshold joint-tail co-exceedance."""

    invalid = [
        probability
        for probability in tail_probabilities
        if not 0.0 < probability < 0.5
    ]

    if invalid:
        raise DependenceDiagnosticError(
            "Tail probabilities must lie strictly between "
            f"zero and 0.5. Invalid values: {invalid}"
        )

    records = []

    for symbol_a, symbol_b in combinations(
        sorted(panel.columns),
        2,
    ):
        pair = _aligned_pair(
            panel,
            symbol_a,
            symbol_b,
        )

        uniform_a = _pseudo_observations(
            pair[symbol_a]
        )

        uniform_b = _pseudo_observations(
            pair[symbol_b]
        )

        for probability in tail_probabilities:
            lower_a = uniform_a.le(probability)
            lower_b = uniform_b.le(probability)

            upper_a = uniform_a.ge(
                1.0 - probability
            )

            upper_b = uniform_b.ge(
                1.0 - probability
            )

            lower_joint_rate = float(
                (lower_a & lower_b).mean()
            )

            upper_joint_rate = float(
                (upper_a & upper_b).mean()
            )

            lower_conditional = float(
                lower_joint_rate / probability
            )

            upper_conditional = float(
                upper_joint_rate / probability
            )

            independence_joint_rate = (
                probability**2
            )

            records.append(
                {
                    "sample": sample_label,
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "observations": int(len(pair)),
                    "tail_probability": float(
                        probability
                    ),
                    "lower_joint_rate": (
                        lower_joint_rate
                    ),
                    "upper_joint_rate": (
                        upper_joint_rate
                    ),
                    "independence_joint_rate": (
                        independence_joint_rate
                    ),
                    "lower_coexceedance_ratio": (
                        lower_conditional
                    ),
                    "upper_coexceedance_ratio": (
                        upper_conditional
                    ),
                    "lower_to_independence_multiple": (
                        lower_joint_rate
                        / independence_joint_rate
                    ),
                    "upper_to_independence_multiple": (
                        upper_joint_rate
                        / independence_joint_rate
                    ),
                    "lower_minus_upper_ratio": (
                        lower_conditional
                        - upper_conditional
                    ),
                }
            )

    return pd.DataFrame(records)


def build_rolling_dependence_table(
    panel: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Calculate rolling Pearson dependence."""

    invalid_windows = [
        window
        for window in windows
        if not isinstance(window, int) or window < 10
    ]

    if invalid_windows:
        raise DependenceDiagnosticError(
            "Rolling windows must be integers of at least 10. "
            f"Invalid values: {invalid_windows}"
        )

    records = []

    for symbol_a, symbol_b in combinations(
        sorted(panel.columns),
        2,
    ):
        pair = _aligned_pair(
            panel,
            symbol_a,
            symbol_b,
        )

        for window in windows:
            correlations = (
                pair[symbol_a]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .corr(pair[symbol_b])
            )

            valid = correlations.dropna()

            for session_date, value in valid.items():
                records.append(
                    {
                        "session_date": (
                            pd.Timestamp(
                                session_date
                            )
                        ),
                        "symbol_a": symbol_a,
                        "symbol_b": symbol_b,
                        "window": int(window),
                        "pearson_correlation": float(
                            value
                        ),
                    }
                )

    return pd.DataFrame(records)


def build_volatility_regimes(
    daily_volatility: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    stress_quantile: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define descriptive normal and stress volatility regimes."""

    _require_columns(
        daily_volatility,
        (
            "symbol",
            "session_date",
            "annualized_total_realized_volatility",
        ),
        context="Daily volatility",
    )

    if not 0.5 < stress_quantile < 1.0:
        raise DependenceDiagnosticError(
            "Stress quantile must lie between 0.5 and 1.0."
        )

    benchmark = daily_volatility.loc[
        (
            daily_volatility["symbol"]
            .astype("string")
            .str.upper()
            .eq(benchmark_symbol.upper())
        ),
        [
            "session_date",
            "annualized_total_realized_volatility",
        ],
    ].copy()

    benchmark["session_date"] = pd.to_datetime(
        benchmark["session_date"],
        errors="raise",
    )

    benchmark[
        "annualized_total_realized_volatility"
    ] = pd.to_numeric(
        benchmark[
            "annualized_total_realized_volatility"
        ],
        errors="coerce",
    )

    benchmark = benchmark.dropna()

    if benchmark.empty:
        raise DependenceDiagnosticError(
            f"No volatility observations found for "
            f"{benchmark_symbol}."
        )

    threshold = float(
        benchmark[
            "annualized_total_realized_volatility"
        ].quantile(stress_quantile)
    )

    benchmark["regime"] = np.where(
        benchmark[
            "annualized_total_realized_volatility"
        ].ge(threshold),
        "high_volatility",
        "normal_volatility",
    )

    definition = pd.DataFrame(
        [
            {
                "benchmark_symbol": (
                    benchmark_symbol.upper()
                ),
                "stress_quantile": float(
                    stress_quantile
                ),
                "volatility_threshold": threshold,
                "normal_observations": int(
                    benchmark["regime"]
                    .eq("normal_volatility")
                    .sum()
                ),
                "stress_observations": int(
                    benchmark["regime"]
                    .eq("high_volatility")
                    .sum()
                ),
            }
        ]
    )

    return (
        benchmark[
            [
                "session_date",
                "regime",
                "annualized_total_realized_volatility",
            ]
        ],
        definition,
    )


def build_regime_dependence_tables(
    panel: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    stress_quantile: float = 0.80,
    tail_probabilities: Sequence[float] = (
        DEFAULT_TAIL_PROBABILITIES
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate dependence separately by volatility regime."""

    regimes, definition = build_volatility_regimes(
        daily_volatility,
        benchmark_symbol=benchmark_symbol,
        stress_quantile=stress_quantile,
    )

    working = (
        panel.reset_index()
        .merge(
            regimes[
                [
                    "session_date",
                    "regime",
                ]
            ],
            on="session_date",
            how="inner",
            validate="one_to_one",
        )
        .set_index("session_date")
    )

    dependence_tables = []
    tail_tables = []

    for regime, subset in working.groupby(
        "regime",
        observed=True,
        sort=True,
    ):
        regime_panel = subset.drop(
            columns=["regime"]
        )

        dependence_tables.append(
            build_pairwise_dependence_table(
                regime_panel,
                sample_label=str(regime),
            )
        )

        tail_tables.append(
            build_tail_dependence_table(
                regime_panel,
                tail_probabilities=tail_probabilities,
                sample_label=str(regime),
            )
        )

    return (
        pd.concat(
            dependence_tables,
            ignore_index=True,
        ),
        pd.concat(
            tail_tables,
            ignore_index=True,
        ),
        definition,
    )


def build_dependence_diagnostics(
    session_features: pd.DataFrame,
    daily_volatility: pd.DataFrame,
) -> DependenceDiagnosticBundle:
    """Build all D05-T02D dependence evidence."""

    panel = build_daily_return_panel(
        session_features
    )

    (
        regime_dependence,
        regime_tail_dependence,
        regime_definition,
    ) = build_regime_dependence_tables(
        panel,
        daily_volatility,
    )

    return DependenceDiagnosticBundle(
        pairwise_dependence=(
            build_pairwise_dependence_table(panel)
        ),
        tail_dependence=(
            build_tail_dependence_table(panel)
        ),
        rolling_dependence=(
            build_rolling_dependence_table(panel)
        ),
        regime_dependence=regime_dependence,
        regime_tail_dependence=(
            regime_tail_dependence
        ),
        regime_definition=regime_definition,
    )
