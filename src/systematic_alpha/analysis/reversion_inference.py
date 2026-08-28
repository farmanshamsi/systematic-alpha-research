"""Development-only walk-forward evidence for OU/VWAP reversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from systematic_alpha.analysis.causal_bar_execution import (
    CausalBarExecutionError,
    apply_causal_next_open_overnight_flat,
)
from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)
from systematic_alpha.strategies.ou_vwap_reversion import (
    OuVwapReversionParameters,
    build_ou_vwap_reversion_strategy,
)


DEVELOPMENT_START: Final[pd.Timestamp] = pd.Timestamp("2020-01-02", tz="UTC")
DEVELOPMENT_END_EXCLUSIVE: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-01", tz="UTC"
)
REQUIRED_SYMBOLS: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM")
REPORTED_SERIES: Final[tuple[str, ...]] = (*REQUIRED_SYMBOLS, "equal_weight")
EXPECTED_YEARS: Final[frozenset[int]] = frozenset(range(2020, 2026))
ANNUALIZATION_FACTOR: Final[float] = 252.0
HAC_LAGS: Final[int] = 5
BOOTSTRAP_REPLICATIONS: Final[int] = 2_000
BOOTSTRAP_BLOCK_LENGTH: Final[int] = 5
BOOTSTRAP_SEED: Final[int] = 1_701
COST_STRESS_BPS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 5.0)
EULER_MASCHERONI: Final[float] = 0.5772156649015329

CONFIGURATIONS: Final[tuple[OuVwapReversionParameters, ...]] = (
    OuVwapReversionParameters(
        configuration_id="ou_vwap_fast",
        reference_window=26,
        ou_window=104,
        variance_ratio_lag=4,
        variance_ratio_threshold=0.95,
        entry_threshold=1.75,
        exit_threshold=0.25,
        minimum_half_life=1.0,
        maximum_half_life=20.0,
        maximum_holding_bars=20,
        cost_bps_per_turnover=1.0,
    ),
    OuVwapReversionParameters(
        configuration_id="ou_vwap_base",
        reference_window=32,
        ou_window=130,
        variance_ratio_lag=4,
        variance_ratio_threshold=0.95,
        entry_threshold=2.0,
        exit_threshold=0.25,
        minimum_half_life=1.0,
        maximum_half_life=26.0,
        maximum_holding_bars=26,
        cost_bps_per_turnover=1.0,
    ),
    OuVwapReversionParameters(
        configuration_id="ou_vwap_slow",
        reference_window=52,
        ou_window=208,
        variance_ratio_lag=4,
        variance_ratio_threshold=0.95,
        entry_threshold=2.25,
        exit_threshold=0.25,
        minimum_half_life=1.0,
        maximum_half_life=39.0,
        maximum_holding_bars=39,
        cost_bps_per_turnover=1.0,
    ),
)

CONFIGURATION_IDS: Final[tuple[str, ...]] = tuple(
    item.configuration_id for item in CONFIGURATIONS
)

SIGNAL_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "fold_id",
    "symbol",
    "test_observations",
    "signal_available_observations",
    "regime_eligible_observations",
    "entries",
    "long_exposure_pct",
    "short_exposure_pct",
    "flat_exposure_pct",
    "turnover",
    "initial_position",
    "initial_turnover",
    "overnight_position_violations",
)

FOLD_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "fold_id",
    "series",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
)

AGGREGATE_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "series",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
)

INFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "series",
    "observations",
    "mean_session_return",
    "naive_t_statistic",
    "hac_lags",
    "hac_t_statistic",
    "annualized_sharpe_ratio",
    "bootstrap_replications",
    "bootstrap_block_length",
    "bootstrap_mean_ci_lower",
    "bootstrap_mean_ci_upper",
    "bootstrap_sharpe_ci_lower",
    "bootstrap_sharpe_ci_upper",
    "information_coefficient",
    "information_coefficient_observations",
    "sample_skewness",
    "sample_kurtosis",
    "probabilistic_sharpe_probability",
    "deflated_sharpe_benchmark",
    "deflated_sharpe_probability",
    "declared_trials",
)

COST_SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "series",
    "cost_bps_per_turnover",
    "test_sessions",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
)

RETURN_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "fold_id",
    "session_date",
    *REPORTED_SERIES,
)


class ReversionInferenceError(ValueError):
    """Raised when Day 17 evidence cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class ReversionInferenceResults:
    """Defensively retained Day 17 evidence tables."""

    signal_diagnostics: pd.DataFrame
    fold_performance: pd.DataFrame
    aggregate_performance: pd.DataFrame
    inference_results: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    session_return_panel: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "signal_diagnostics",
            "fold_performance",
            "aggregate_performance",
            "inference_results",
            "cost_sensitivity",
            "session_return_panel",
        ):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            object.__setattr__(self, name, value.copy(deep=True).reset_index(drop=True))


def _validate_and_prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Require canonical development scope and construct return features."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")
    if bars.empty:
        raise ReversionInferenceError("Canonical development bars cannot be empty.")
    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "source",
        "feed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ReversionInferenceError(f"Canonical bars are missing columns: {missing}.")

    scoped = bars.copy(deep=True)
    try:
        scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ReversionInferenceError("Canonical timestamps are malformed.") from exc
    scoped["symbol"] = scoped["symbol"].astype("string").str.strip().str.upper()
    symbols = tuple(sorted(scoped["symbol"].dropna().unique().tolist()))
    if symbols != tuple(sorted(REQUIRED_SYMBOLS)):
        raise ReversionInferenceError(
            f"Day 17 requires exactly {REQUIRED_SYMBOLS}; received {symbols}."
        )
    if scoped["timestamp"].min() < DEVELOPMENT_START or scoped[
        "timestamp"
    ].max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise ReversionInferenceError("Day 17 input must remain development-only.")
    years = frozenset(scoped["timestamp"].dt.year.unique().tolist())
    if years != EXPECTED_YEARS:
        raise ReversionInferenceError(
            "Day 17 requires observations in every development year 2020-2025."
        )
    try:
        aggregated = aggregate_session_bars(scoped, "15min")
    except SessionAggregationError as exc:
        raise ReversionInferenceError(f"Whole-session validation failed: {exc}") from exc
    return build_return_features(
        aggregated, expected_symbols=REQUIRED_SYMBOLS
    ).bars


def _session_panel(observations: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Compound intraday returns to an exact four-series session panel."""

    net_return = observations["gross_strategy_return"] - (
        observations["turnover"] * float(cost_bps) / 10_000.0
    )
    working = observations[["symbol", "session_date"]].copy(deep=True)
    working["net_return"] = net_return
    session = (
        working.groupby(["session_date", "symbol"], observed=True, sort=True)[
            "net_return"
        ]
        .agg(lambda values: float(np.prod(1.0 + values.to_numpy(dtype="float64")) - 1.0))
        .unstack("symbol")
    )
    if tuple(session.columns) != tuple(sorted(REQUIRED_SYMBOLS)):
        session = session.reindex(columns=REQUIRED_SYMBOLS)
    else:
        session = session.reindex(columns=REQUIRED_SYMBOLS)
    if session.isna().any().any():
        raise ReversionInferenceError("Session return panel is not complete across symbols.")
    session["equal_weight"] = session[list(REQUIRED_SYMBOLS)].mean(axis=1)
    session.index = pd.to_datetime(session.index, utc=True).normalize()
    session.index.name = "session_date"
    return session


def _apply_ou_performance_timing(
    observations: pd.DataFrame,
    *,
    fold_id: str,
) -> pd.DataFrame:
    """Apply causal OU timing without changing model-specific columns."""

    try:
        return apply_causal_next_open_overnight_flat(
            observations,
            cost_bps_per_turnover=0.0,
        )
    except CausalBarExecutionError as exc:
        raise ReversionInferenceError(
            f"{fold_id} causal OU timing failed: {exc}"
        ) from exc


def _performance_record(
    returns: pd.Series,
    *,
    configuration_id: str,
    series: str,
    turnover: float,
    fold_id: str | None,
) -> dict[str, object]:
    metrics = calculate_performance_metrics(
        returns.reset_index(drop=True), annualization_factor=ANNUALIZATION_FACTOR
    )
    record: dict[str, object] = {
        "configuration_id": configuration_id,
        "series": series,
        "test_sessions": int(len(returns)),
        "start_session": pd.Timestamp(returns.index.min()).strftime("%Y-%m-%d"),
        "end_session": pd.Timestamp(returns.index.max()).strftime("%Y-%m-%d"),
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": metrics.max_drawdown,
        "turnover": float(turnover),
    }
    if fold_id is not None:
        record["fold_id"] = fold_id
    return record


def _signal_diagnostics(
    observations: pd.DataFrame,
    *,
    configuration_id: str,
    fold_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for symbol in REQUIRED_SYMBOLS:
        group = observations.loc[observations["symbol"].eq(symbol)].copy()
        prior = group["position"].shift(1, fill_value=0)
        entries = int((group["position"].ne(0) & prior.eq(0)).sum())
        count = len(group)

        def pct(value: int) -> float:
            return 100.0 * float(group["position"].eq(value).sum()) / float(count)

        session_open = ~group["session_date"].eq(group["session_date"].shift(1))
        records.append(
            {
                "configuration_id": configuration_id,
                "fold_id": fold_id,
                "symbol": symbol,
                "test_observations": int(count),
                "signal_available_observations": int(group["signal_available"].sum()),
                "regime_eligible_observations": int(group["regime_eligible"].sum()),
                "entries": entries,
                "long_exposure_pct": pct(1),
                "short_exposure_pct": pct(-1),
                "flat_exposure_pct": pct(0),
                "turnover": float(group["turnover"].sum()),
                "initial_position": int(group["position"].iloc[0]),
                "initial_turnover": float(group["turnover"].iloc[0]),
                "overnight_position_violations": int(
                    group.loc[session_open, "position"].ne(0).sum()
                ),
            }
        )
    return records


def _information_pairs(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build causal score/next-return pairs for symbols and equal weight."""

    pairs: dict[str, pd.DataFrame] = {}
    symbol_frames: list[pd.DataFrame] = []
    for symbol in REQUIRED_SYMBOLS:
        group = observations.loc[observations["symbol"].eq(symbol)].copy()
        group["forward_return"] = group["close_to_close_simple_return"].shift(-1)
        valid = group[["signal_score", "forward_return"]].notna().all(axis=1)
        pair = group.loc[valid, ["timestamp", "signal_score", "forward_return"]]
        pairs[symbol] = pair[["signal_score", "forward_return"]].reset_index(drop=True)
        tagged = pair.copy(deep=True)
        tagged["symbol"] = symbol
        symbol_frames.append(tagged)

    combined = pd.concat(symbol_frames, ignore_index=True)
    score = combined.pivot(
        index="timestamp", columns="symbol", values="signal_score"
    ).reindex(columns=REQUIRED_SYMBOLS)
    forward = combined.pivot(
        index="timestamp", columns="symbol", values="forward_return"
    ).reindex(columns=REQUIRED_SYMBOLS)
    complete = score.notna().all(axis=1) & forward.notna().all(axis=1)
    pairs["equal_weight"] = pd.DataFrame(
        {
            "signal_score": score.loc[complete, list(REQUIRED_SYMBOLS)].mean(axis=1),
            "forward_return": forward.loc[complete, list(REQUIRED_SYMBOLS)].mean(axis=1),
        }
    ).reset_index(drop=True)
    return pairs


def _t_statistics(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    standard_deviation = float(np.std(values, ddof=1))
    naive = (
        float(np.mean(values) / (standard_deviation / math.sqrt(n)))
        if standard_deviation > 0.0
        else float("nan")
    )
    demeaned = values - float(np.mean(values))
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, min(HAC_LAGS, n - 1) + 1):
        weight = 1.0 - lag / float(HAC_LAGS + 1)
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * covariance
    hac = (
        float(np.mean(values) / math.sqrt(long_run_variance / n))
        if long_run_variance > 0.0
        else float("nan")
    )
    return naive, hac


def _annualized_sharpe(values: np.ndarray) -> float:
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        return float("nan")
    return float(np.mean(values) / standard_deviation * math.sqrt(ANNUALIZATION_FACTOR))


def _bootstrap_intervals(
    values: np.ndarray,
    *,
    replications: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if replications <= 0:
        raise ReversionInferenceError("bootstrap replications must be positive.")
    n = len(values)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / BOOTSTRAP_BLOCK_LENGTH))
    means = np.empty(replications, dtype="float64")
    sharpes = np.empty(replications, dtype="float64")
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH)
    for replication in range(replications):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        sample = values[indices]
        means[replication] = float(np.mean(sample))
        sharpes[replication] = _annualized_sharpe(sample)
    finite_sharpes = sharpes[np.isfinite(sharpes)]
    if len(finite_sharpes) == 0:
        sharpe_bounds = (float("nan"), float("nan"))
    else:
        sharpe_bounds = tuple(np.quantile(finite_sharpes, [0.025, 0.975]))
    mean_bounds = tuple(np.quantile(means, [0.025, 0.975]))
    return (
        float(mean_bounds[0]),
        float(mean_bounds[1]),
        float(sharpe_bounds[0]),
        float(sharpe_bounds[1]),
    )


def _sharpe_probability(
    *,
    per_period_sharpe: float,
    benchmark: float,
    observations: int,
    sample_skewness: float,
    sample_kurtosis: float,
) -> float:
    denominator_squared = (
        1.0
        - sample_skewness * per_period_sharpe
        + (sample_kurtosis - 1.0) * per_period_sharpe**2 / 4.0
    )
    if denominator_squared <= 0.0:
        return float("nan")
    z_value = (
        (per_period_sharpe - benchmark)
        * math.sqrt(observations - 1.0)
        / math.sqrt(denominator_squared)
    )
    return float(norm.cdf(z_value))


def _deflated_benchmark(trial_sharpes: np.ndarray) -> float:
    finite = trial_sharpes[np.isfinite(trial_sharpes)]
    if len(finite) < 2:
        return float("nan")
    dispersion = float(np.std(finite, ddof=1))
    trials = float(len(CONFIGURATIONS))
    expected_maximum = (
        (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / trials)
        + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (trials * math.e))
    )
    return float(dispersion * expected_maximum)


def run_reversion_inference(
    bars: pd.DataFrame,
    *,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
) -> ReversionInferenceResults:
    """Run the frozen Day 17 walk-forward and inference contract."""

    features = _validate_and_prepare_bars(bars)
    folds = build_walk_forward_folds()
    diagnostic_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    return_panels: list[pd.DataFrame] = []
    aggregate_returns: dict[tuple[str, str], list[pd.Series]] = {}
    aggregate_turnover: dict[tuple[str, str], float] = {}
    cost_returns: dict[tuple[str, str, float], list[pd.Series]] = {}
    information_pairs: dict[tuple[str, str], list[pd.DataFrame]] = {}

    feature_sessions = pd.to_datetime(features["session_date"], utc=True).dt.normalize()
    for configuration_index, parameters in enumerate(CONFIGURATIONS):
        for fold in folds:
            history_mask = feature_sessions.lt(fold.test_end_exclusive)
            test_mask = feature_sessions.ge(fold.test_start) & feature_sessions.lt(
                fold.test_end_exclusive
            )
            history = features.loc[history_mask].copy(deep=True).reset_index(drop=True)
            test_source = features.loc[test_mask]
            if test_source.empty:
                raise ReversionInferenceError(f"{fold.fold_id} has no test observations.")
            reset_timestamps = tuple(
                pd.Timestamp(value)
                for value in test_source.groupby("symbol", observed=True)["timestamp"].min()
            )
            bundle = build_ou_vwap_reversion_strategy(
                history,
                parameters=parameters,
                execution_reset_timestamps=reset_timestamps,
            )
            result_sessions = pd.to_datetime(
                bundle.observations["session_date"], utc=True
            ).dt.normalize()
            test = bundle.observations.loc[
                result_sessions.ge(fold.test_start)
                & result_sessions.lt(fold.test_end_exclusive)
            ].copy(deep=True)
            test = _apply_ou_performance_timing(
                test,
                fold_id=fold.fold_id,
            )
            diagnostic_records.extend(
                _signal_diagnostics(
                    test,
                    configuration_id=parameters.configuration_id,
                    fold_id=fold.fold_id,
                )
            )

            base_panel = _session_panel(test, cost_bps=parameters.cost_bps_per_turnover)
            stored_panel = base_panel.reset_index()
            stored_panel.insert(0, "fold_id", fold.fold_id)
            stored_panel.insert(0, "configuration_id", parameters.configuration_id)
            return_panels.append(stored_panel)

            symbol_turnover = test.groupby("symbol", observed=True)["turnover"].sum()
            for series in REPORTED_SERIES:
                turnover = (
                    float(symbol_turnover[series])
                    if series in REQUIRED_SYMBOLS
                    else float(symbol_turnover.mean())
                )
                fold_records.append(
                    _performance_record(
                        base_panel[series],
                        configuration_id=parameters.configuration_id,
                        series=series,
                        turnover=turnover,
                        fold_id=fold.fold_id,
                    )
                )
                key = (parameters.configuration_id, series)
                aggregate_returns.setdefault(key, []).append(base_panel[series])
                aggregate_turnover[key] = aggregate_turnover.get(key, 0.0) + turnover

            pairs = _information_pairs(test)
            for series, pair in pairs.items():
                information_pairs.setdefault(
                    (parameters.configuration_id, series), []
                ).append(pair)

            for cost in COST_STRESS_BPS:
                panel = _session_panel(test, cost_bps=cost)
                for series in REPORTED_SERIES:
                    cost_returns.setdefault(
                        (parameters.configuration_id, series, cost), []
                    ).append(panel[series])

    fold_performance = pd.DataFrame.from_records(fold_records).loc[
        :, FOLD_PERFORMANCE_COLUMNS
    ]
    aggregate_records: list[dict[str, object]] = []
    concatenated: dict[tuple[str, str], pd.Series] = {}
    for configuration_id in CONFIGURATION_IDS:
        for series in REPORTED_SERIES:
            key = (configuration_id, series)
            values = pd.concat(aggregate_returns[key]).sort_index()
            if values.index.has_duplicates:
                raise ReversionInferenceError("Aggregate test sessions must be unique.")
            concatenated[key] = values
            aggregate_records.append(
                _performance_record(
                    values,
                    configuration_id=configuration_id,
                    series=series,
                    turnover=aggregate_turnover[key],
                    fold_id=None,
                )
            )
    aggregate_performance = pd.DataFrame.from_records(aggregate_records).loc[
        :, AGGREGATE_PERFORMANCE_COLUMNS
    ]

    preliminary: dict[tuple[str, str], dict[str, object]] = {}
    for configuration_index, configuration_id in enumerate(CONFIGURATION_IDS):
        for series_index, series in enumerate(REPORTED_SERIES):
            values = concatenated[(configuration_id, series)].to_numpy(dtype="float64")
            naive_t, hac_t = _t_statistics(values)
            annualized_sharpe = _annualized_sharpe(values)
            per_period_sharpe = annualized_sharpe / math.sqrt(ANNUALIZATION_FACTOR)
            sample_skewness = float(skew(values, bias=False))
            sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
            mean_low, mean_high, sharpe_low, sharpe_high = _bootstrap_intervals(
                values,
                replications=bootstrap_replications,
                seed=(
                    BOOTSTRAP_SEED
                    + configuration_index * len(REPORTED_SERIES)
                    + series_index
                ),
            )
            pair = pd.concat(
                information_pairs[(configuration_id, series)], ignore_index=True
            )
            ic = (
                float(pair["signal_score"].corr(pair["forward_return"]))
                if len(pair) >= 2
                else float("nan")
            )
            preliminary[(configuration_id, series)] = {
                "configuration_id": configuration_id,
                "series": series,
                "observations": int(len(values)),
                "mean_session_return": float(np.mean(values)),
                "naive_t_statistic": naive_t,
                "hac_lags": HAC_LAGS,
                "hac_t_statistic": hac_t,
                "annualized_sharpe_ratio": annualized_sharpe,
                "bootstrap_replications": int(bootstrap_replications),
                "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
                "bootstrap_mean_ci_lower": mean_low,
                "bootstrap_mean_ci_upper": mean_high,
                "bootstrap_sharpe_ci_lower": sharpe_low,
                "bootstrap_sharpe_ci_upper": sharpe_high,
                "information_coefficient": ic,
                "information_coefficient_observations": int(len(pair)),
                "sample_skewness": sample_skewness,
                "sample_kurtosis": sample_kurtosis,
                "probabilistic_sharpe_probability": _sharpe_probability(
                    per_period_sharpe=per_period_sharpe,
                    benchmark=0.0,
                    observations=len(values),
                    sample_skewness=sample_skewness,
                    sample_kurtosis=sample_kurtosis,
                ),
                "per_period_sharpe": per_period_sharpe,
            }

    deflated_benchmarks: dict[str, float] = {}
    for series in REPORTED_SERIES:
        trial_sharpes = np.array(
            [
                preliminary[(configuration_id, series)]["per_period_sharpe"]
                for configuration_id in CONFIGURATION_IDS
            ],
            dtype="float64",
        )
        deflated_benchmarks[series] = _deflated_benchmark(trial_sharpes)

    inference_records: list[dict[str, object]] = []
    for configuration_id in CONFIGURATION_IDS:
        for series in REPORTED_SERIES:
            benchmark = deflated_benchmarks[series]
            record = preliminary[(configuration_id, series)].copy()
            record["deflated_sharpe_benchmark"] = benchmark
            record["deflated_sharpe_probability"] = _sharpe_probability(
                per_period_sharpe=float(record.pop("per_period_sharpe")),
                benchmark=benchmark,
                observations=int(record["observations"]),
                sample_skewness=float(record["sample_skewness"]),
                sample_kurtosis=float(record["sample_kurtosis"]),
            )
            record["declared_trials"] = len(CONFIGURATIONS)
            inference_records.append(record)
    inference_results = pd.DataFrame.from_records(inference_records).loc[
        :, INFERENCE_COLUMNS
    ]

    cost_records: list[dict[str, object]] = []
    for configuration_id in CONFIGURATION_IDS:
        for series in REPORTED_SERIES:
            for cost in COST_STRESS_BPS:
                values = pd.concat(
                    cost_returns[(configuration_id, series, cost)]
                ).sort_index()
                metrics = calculate_performance_metrics(
                    values.reset_index(drop=True),
                    annualization_factor=ANNUALIZATION_FACTOR,
                )
                cost_records.append(
                    {
                        "configuration_id": configuration_id,
                        "series": series,
                        "cost_bps_per_turnover": cost,
                        "test_sessions": int(len(values)),
                        "cumulative_return": metrics.cumulative_return,
                        "annualized_return": metrics.annualized_return,
                        "annualized_volatility": metrics.annualized_volatility,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "maximum_drawdown": metrics.max_drawdown,
                    }
                )
    cost_sensitivity = pd.DataFrame.from_records(cost_records).loc[
        :, COST_SENSITIVITY_COLUMNS
    ]

    signal_diagnostics = pd.DataFrame.from_records(diagnostic_records).loc[
        :, SIGNAL_DIAGNOSTIC_COLUMNS
    ]
    session_return_panel = pd.concat(return_panels, ignore_index=True).loc[
        :, RETURN_PANEL_COLUMNS
    ]
    return ReversionInferenceResults(
        signal_diagnostics=signal_diagnostics,
        fold_performance=fold_performance,
        aggregate_performance=aggregate_performance,
        inference_results=inference_results,
        cost_sensitivity=cost_sensitivity,
        session_return_panel=session_return_panel,
    )
