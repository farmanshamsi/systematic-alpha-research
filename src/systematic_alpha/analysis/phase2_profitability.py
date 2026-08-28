"""Predeclared Phase II development-only profitability experiment."""

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
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis.causal_bar_execution import (
    CausalBarExecutionError,
    apply_causal_next_open_overnight_flat,
)
from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics
from systematic_alpha.analysis.trend_family_walk_forward import (
    REQUIRED_INPUT_SYMBOLS,
    _validate_complete_coverage,
    _validate_date_and_symbol_scope,
    build_walk_forward_folds,
)
from systematic_alpha.analysis.trend_methodology_finalization import (
    apply_next_open_overnight_flat,
)
from systematic_alpha.data.session_aggregation import aggregate_session_bars
from systematic_alpha.strategies.ou_vwap_reversion import (
    OuVwapReversionParameters,
    build_ou_vwap_reversion_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    TrendRatioParameters,
    build_trend_ratio_strategy,
)


SPECIFICATION_VERSION: Final[str] = "day26_phase2_profitability_v1"
DEVELOPMENT_END_EXCLUSIVE: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-01", tz="UTC"
)
CONFIGURATION_IDS: Final[tuple[str, ...]] = (
    "price_ratio_long_flat_baseline",
    "price_ratio_persistent_hysteresis_phase2",
    "ou_vwap_slow_baseline",
    "ou_vwap_slow_cost_margin_phase2",
)
TREND_CONFIGURATION_IDS: Final[tuple[str, ...]] = CONFIGURATION_IDS[:2]
OU_CONFIGURATION_IDS: Final[tuple[str, ...]] = CONFIGURATION_IDS[2:]
OU_SERIES: Final[tuple[str, ...]] = (*REQUIRED_INPUT_SYMBOLS, "equal_weight")
COST_STRESS_BPS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 5.0)
BASE_COST_BPS: Final[float] = 1.0
HAC_LAGS: Final[int] = 5
BOOTSTRAP_REPLICATIONS: Final[int] = 2_000
BOOTSTRAP_BLOCK_LENGTH: Final[int] = 5
BOOTSTRAP_SEED: Final[int] = 2_601
ANNUALIZATION_FACTOR: Final[float] = 252.0

TREND_SHORT_WINDOW: Final[int] = 8
TREND_LONG_WINDOW: Final[int] = 32
TREND_ENTRY_BAND: Final[float] = 0.001
TREND_EXIT_BAND: Final[float] = 0.0005
TREND_CONFIRMATION_BARS: Final[int] = 4

OU_EXPECTED_CONVERGENCE_THRESHOLD: Final[float] = 0.001
OU_SLOW_PARAMETERS: Final[OuVwapReversionParameters] = OuVwapReversionParameters(
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
    cost_bps_per_turnover=0.0,
)

AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "strategy_family",
    "phase",
    "series",
    "cost_bps_per_turnover",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "gross_cumulative_return",
    "cost_drag",
    "turnover",
    "trade_count",
    "long_exposure_pct",
    "short_exposure_pct",
    "flat_exposure_pct",
    "approximate_break_even_cost_bps_per_turnover",
    "positive_folds",
    "overnight_position_violations",
)

FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "strategy_family",
    "phase",
    "fold_id",
    "series",
    "cost_bps_per_turnover",
    "test_sessions",
    "start_session",
    "end_session",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "trade_count",
    "long_exposure_pct",
    "short_exposure_pct",
    "flat_exposure_pct",
    "initial_position",
    "initial_turnover",
    "overnight_position_violations",
)

COST_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in AGGREGATE_COLUMNS if column != "positive_folds"
)

COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_family",
    "series",
    "cost_bps_per_turnover",
    "baseline_configuration_id",
    "phase2_configuration_id",
    "baseline_cumulative_return",
    "phase2_cumulative_return",
    "cumulative_return_change",
    "baseline_annualized_volatility",
    "phase2_annualized_volatility",
    "baseline_sharpe_ratio",
    "phase2_sharpe_ratio",
    "baseline_maximum_drawdown",
    "phase2_maximum_drawdown",
    "baseline_turnover",
    "phase2_turnover",
    "turnover_change_pct",
    "baseline_trade_count",
    "phase2_trade_count",
    "baseline_positive_folds",
    "phase2_positive_folds",
    "development_net_return_improved",
)

INFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "strategy_family",
    "phase",
    "series",
    "cost_bps_per_turnover",
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
    "declared_phase2_trials",
)

SESSION_RETURN_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "fold_id",
    "series",
    "session_date",
    "gross_return",
    "net_return_1bp",
    "turnover",
)

DATA_QUALITY_FILENAME: Final[str] = "data_quality.json"
AGGREGATE_FILENAME: Final[str] = "aggregate_performance.csv"
FOLD_FILENAME: Final[str] = "fold_performance.csv"
COST_FILENAME: Final[str] = "cost_sensitivity.csv"
COMPARISON_FILENAME: Final[str] = "comparison.csv"
INFERENCE_FILENAME: Final[str] = "inference.csv"
SESSION_RETURNS_FILENAME: Final[str] = "session_returns.csv"
METHODOLOGY_FILENAME: Final[str] = "methodology.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    DATA_QUALITY_FILENAME,
    AGGREGATE_FILENAME,
    FOLD_FILENAME,
    COST_FILENAME,
    COMPARISON_FILENAME,
    INFERENCE_FILENAME,
    SESSION_RETURNS_FILENAME,
    METHODOLOGY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)


class Phase2ProfitabilityError(ValueError):
    """Raised when the frozen Phase II experiment cannot run safely."""


@dataclass(frozen=True, slots=True)
class Phase2ProfitabilityResults:
    """All deterministic Phase II evidence tables and source audit."""

    data_quality: dict[str, object]
    aggregate_performance: pd.DataFrame
    fold_performance: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    comparison: pd.DataFrame
    inference: pd.DataFrame
    session_returns: pd.DataFrame


def _configuration_metadata(configuration_id: str) -> tuple[str, str]:
    if configuration_id in TREND_CONFIGURATION_IDS:
        return "price_ratio_long_flat", (
            "baseline" if configuration_id.endswith("baseline") else "phase2"
        )
    if configuration_id in OU_CONFIGURATION_IDS:
        return "ou_vwap_slow", (
            "baseline" if configuration_id.endswith("baseline") else "phase2"
        )
    raise Phase2ProfitabilityError(f"Unknown configuration: {configuration_id!r}.")


def audit_development_data(
    bars: pd.DataFrame,
    *,
    source_dataset_id: str,
    source_sha256: str,
) -> dict[str, object]:
    """Run the compact, high-value source-data quality contract."""

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise Phase2ProfitabilityError("Development bars must be a non-empty DataFrame.")
    required = (
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
    )
    missing = sorted(set(required).difference(bars.columns))
    if missing:
        raise Phase2ProfitabilityError(f"Development bars are missing: {missing}.")
    if not isinstance(source_dataset_id, str) or not source_dataset_id.strip():
        raise Phase2ProfitabilityError("source_dataset_id must be non-empty.")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise Phase2ProfitabilityError("source_sha256 must be lowercase SHA-256 hex.")

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    symbols = bars["symbol"].astype("string").str.strip().str.upper()
    if timestamps.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise Phase2ProfitabilityError("Phase II input contains prohibited 2026 rows.")
    expected_symbols = tuple(sorted(REQUIRED_INPUT_SYMBOLS))
    observed_symbols = tuple(sorted(symbols.dropna().unique().tolist()))
    if observed_symbols != expected_symbols:
        raise Phase2ProfitabilityError(
            f"Phase II requires exactly {expected_symbols}; received {observed_symbols}."
        )

    numeric = bars.loc[:, ["open", "high", "low", "close", "vwap", "volume", "trade_count"]]
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    null_count = int(bars.loc[:, required].isna().sum().sum())
    malformed_numeric = int(numeric.isna().sum().sum())
    duplicate_count = int(
        pd.DataFrame({"symbol": symbols, "timestamp": timestamps})
        .duplicated(["symbol", "timestamp"])
        .sum()
    )
    nonpositive_price_rows = int(
        numeric.loc[:, ["open", "high", "low", "close", "vwap"]]
        .le(0.0)
        .any(axis=1)
        .sum()
    )
    negative_volume_rows = int(numeric["volume"].lt(0.0).sum())
    negative_trade_count_rows = int(numeric["trade_count"].lt(0.0).sum())
    if any(
        (
            null_count,
            malformed_numeric,
            duplicate_count,
            nonpositive_price_rows,
            negative_volume_rows,
            negative_trade_count_rows,
        )
    ):
        raise Phase2ProfitabilityError("Canonical development data failed quality gates.")

    row_counts = (
        pd.DataFrame({"symbol": symbols})
        .groupby("symbol", observed=True)
        .size()
        .sort_index()
    )
    if row_counts.nunique() != 1:
        raise Phase2ProfitabilityError("Symbol row coverage is not balanced.")
    return {
        "source_dataset_id": source_dataset_id,
        "source_sha256": source_sha256,
        "rows": int(len(bars)),
        "columns": int(len(bars.columns)),
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "symbols": list(observed_symbols),
        "rows_by_symbol": {str(key): int(value) for key, value in row_counts.items()},
        "duplicate_symbol_timestamp_rows": duplicate_count,
        "null_values": null_count,
        "malformed_numeric_values": malformed_numeric,
        "nonpositive_price_rows": nonpositive_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "negative_trade_count_rows": negative_trade_count_rows,
        "source_values": sorted(bars["source"].astype(str).unique().tolist()),
        "feed_values": sorted(bars["feed"].astype(str).unique().tolist()),
        "quality_gate_passed": True,
        "locked_period_accessed": False,
    }


def _prepare_features(bars: pd.DataFrame) -> pd.DataFrame:
    scoped = _validate_date_and_symbol_scope(bars)
    fifteen = aggregate_session_bars(scoped, "15min")
    _validate_complete_coverage(fifteen)
    sessions = pd.to_datetime(fifteen["session_date"], utc=True).dt.normalize()
    if sessions.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise Phase2ProfitabilityError("Locked or later rows are prohibited.")
    return build_return_features(
        fifteen, expected_symbols=REQUIRED_INPUT_SYMBOLS
    ).bars


def build_persistent_hysteresis_signal(observations: pd.DataFrame) -> pd.DataFrame:
    """Replace the baseline trend target with the one frozen Phase II target."""

    required = {"symbol", "timestamp", "ma_price_ratio", "signal_available"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise Phase2ProfitabilityError(f"Trend observations are missing: {missing}.")
    result = observations.sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).copy(deep=True)
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("symbol", observed=True, sort=False):
        part = group.copy(deep=True)
        ratio = pd.to_numeric(part["ma_price_ratio"], errors="coerce").to_numpy()
        available = part["signal_available"].astype(bool).to_numpy()
        targets = np.zeros(len(part), dtype="int8")
        confirmations = np.zeros(len(part), dtype="int64")
        current = 0
        consecutive = 0
        for index, (value, is_available) in enumerate(
            zip(ratio, available, strict=True)
        ):
            if not is_available or not math.isfinite(float(value)):
                current = 0
                consecutive = 0
            elif current == 0:
                if value > 1.0 + TREND_ENTRY_BAND:
                    consecutive += 1
                    if consecutive >= TREND_CONFIRMATION_BARS:
                        current = 1
                else:
                    consecutive = 0
            elif value <= 1.0 + TREND_EXIT_BAND:
                current = 0
                consecutive = 0
            targets[index] = current
            confirmations[index] = consecutive
        part["signal"] = targets
        part["confirmation_bars"] = confirmations
        pieces.append(part)
    complete = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if not complete["signal"].isin((0, 1)).all():
        raise RuntimeError("Phase II trend target violated long-flat positioning.")
    return complete


def apply_ou_cost_margin_gate(
    observations: pd.DataFrame,
    *,
    execution_reset_timestamps: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    """Rebuild slow OU/VWAP state with the frozen expected-convergence gate."""

    required = {
        "timestamp",
        "symbol",
        "session_date",
        "is_session_close_bar",
        "log_price_residual",
        "ou_equilibrium",
        "ou_phi",
        "ou_zscore",
        "regime_eligible",
        "signal_available",
        OU_SLOW_PARAMETERS.return_column,
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise Phase2ProfitabilityError(f"OU observations are missing: {missing}.")
    normalized_resets = {value.tz_convert("UTC") for value in execution_reset_timestamps}
    result = observations.sort_values(["symbol", "timestamp"], kind="stable").copy()
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("symbol", observed=True, sort=False):
        part = group.copy(deep=True).reset_index(drop=True)
        session_position = part.groupby("session_date", observed=True).cumcount()
        session_size = part.groupby("session_date", observed=True)["timestamp"].transform("size")
        investable = (
            session_size.sub(session_position).sub(1).clip(lower=0)
            .clip(upper=OU_SLOW_PARAMETERS.maximum_holding_bars)
            .astype("int64")
        )
        distance = part["log_price_residual"].sub(part["ou_equilibrium"]).abs()
        phi = pd.to_numeric(part["ou_phi"], errors="coerce")
        expected = distance.mul(1.0 - phi.pow(investable))
        gate = (
            part["regime_eligible"].astype(bool)
            & investable.ge(1)
            & expected.ge(OU_EXPECTED_CONVERGENCE_THRESHOLD)
        )
        reset = part["timestamp"].isin(normalized_resets).to_numpy(dtype="bool")
        close = part["is_session_close_bar"].astype(bool).to_numpy()
        regime = part["regime_eligible"].astype(bool).to_numpy()
        allowed_entry = gate.to_numpy(dtype="bool")
        zscore = pd.to_numeric(part["ou_zscore"], errors="coerce").to_numpy()
        target = np.zeros(len(part), dtype="int8")
        holding = np.zeros(len(part), dtype="int64")
        current = 0
        age = 0
        for index, (z_value, eligible, entry_allowed, at_close, at_reset) in enumerate(
            zip(zscore, regime, allowed_entry, close, reset, strict=True)
        ):
            if at_reset:
                current = 0
                age = 0
            if at_close:
                current = 0
                age = 0
            elif current == 0:
                age = 0
                if entry_allowed and z_value <= -OU_SLOW_PARAMETERS.entry_threshold:
                    current = 1
                elif entry_allowed and z_value >= OU_SLOW_PARAMETERS.entry_threshold:
                    current = -1
            else:
                age += 1
                exit_for_mean = (
                    current == 1 and z_value >= -OU_SLOW_PARAMETERS.exit_threshold
                ) or (
                    current == -1 and z_value <= OU_SLOW_PARAMETERS.exit_threshold
                )
                if (
                    not eligible
                    or exit_for_mean
                    or age >= OU_SLOW_PARAMETERS.maximum_holding_bars
                ):
                    current = 0
                    age = 0
            target[index] = current
            holding[index] = age if current != 0 else 0
        part["investable_bars_remaining"] = investable
        part["expected_residual_convergence"] = expected.astype("float64")
        part["entry_gate"] = gate.astype(bool)
        part["signal"] = target
        part["holding_bars"] = holding
        part["signal_score"] = (-part["ou_zscore"]).where(
            gate & ~part["is_session_close_bar"].astype(bool)
        )
        part["position"] = part["signal"].shift(1, fill_value=0).astype("int8")
        part.loc[reset, "position"] = 0
        part["position_eligible"] = part["signal_available"].shift(
            1, fill_value=False
        ).astype(bool)
        market_return = part[OU_SLOW_PARAMETERS.return_column].fillna(0.0)
        part["gross_strategy_return"] = part["position"].astype(float).mul(market_return)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def apply_ou_next_open_accounting(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the shared causal timing convention before Phase II repricing."""

    try:
        return apply_causal_next_open_overnight_flat(
            observations,
            cost_bps_per_turnover=0.0,
        )
    except CausalBarExecutionError as exc:
        raise Phase2ProfitabilityError(
            f"Causal OU timing failed: {exc}"
        ) from exc


def _session_panel(observations: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    working = observations[["symbol", "session_date"]].copy(deep=True)
    working["gross_return"] = observations["gross_strategy_return"].astype("float64")
    working["net_return"] = working["gross_return"] - (
        observations["turnover"].astype("float64") * float(cost_bps) / 10_000.0
    )
    working["turnover"] = observations["turnover"].astype("float64")
    grouped = working.groupby(["session_date", "symbol"], observed=True, sort=True)
    gross = grouped["gross_return"].agg(
        lambda values: float(np.prod(1.0 + values.to_numpy(dtype="float64")) - 1.0)
    ).unstack("symbol")
    net = grouped["net_return"].agg(
        lambda values: float(np.prod(1.0 + values.to_numpy(dtype="float64")) - 1.0)
    ).unstack("symbol")
    turnover = grouped["turnover"].sum().unstack("symbol")
    for frame in (gross, net, turnover):
        frame.index = pd.to_datetime(frame.index, utc=True).normalize()
        frame.index.name = "session_date"
    expected_symbols = tuple(sorted(observations["symbol"].unique().tolist()))
    gross = gross.reindex(columns=expected_symbols)
    net = net.reindex(columns=expected_symbols)
    turnover = turnover.reindex(columns=expected_symbols)
    if gross.isna().any().any() or net.isna().any().any() or turnover.isna().any().any():
        raise Phase2ProfitabilityError("Session panel is incomplete.")
    if len(expected_symbols) > 1:
        gross["equal_weight"] = gross.mean(axis=1)
        net["equal_weight"] = net.mean(axis=1)
        turnover["equal_weight"] = turnover.mean(axis=1)
    pieces: list[pd.DataFrame] = []
    for series in net.columns:
        pieces.append(
            pd.DataFrame(
                {
                    "series": str(series),
                    "session_date": net.index,
                    "gross_return": gross[series].to_numpy(dtype="float64"),
                    "net_return": net[series].to_numpy(dtype="float64"),
                    "turnover": turnover[series].to_numpy(dtype="float64"),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def _execution_statistics(observations: pd.DataFrame, *, series: str) -> dict[str, float | int]:
    if series == "equal_weight":
        sample = observations.copy(deep=False)
    else:
        sample = observations.loc[observations["symbol"].eq(series)].copy(deep=False)
    eligible = sample.loc[sample["position_eligible"].astype(bool), "position"]
    if eligible.empty:
        long = short = flat = float("nan")
    else:
        long, short, flat = (
            float(100.0 * eligible.eq(value).mean()) for value in (1, -1, 0)
        )
    turnover_by_symbol = sample.groupby("symbol", observed=True)["turnover"].sum()
    turnover = float(turnover_by_symbol.mean()) if series == "equal_weight" else float(turnover_by_symbol.iloc[0])
    entries: list[int] = []
    violations = 0
    initial_positions: list[int] = []
    initial_turnovers: list[float] = []
    for _, group in sample.groupby("symbol", observed=True, sort=True):
        session = pd.to_datetime(group["session_date"], utc=True).dt.normalize()
        session_open = session.ne(session.shift(1)).fillna(True)
        prior = group["position"].shift(1, fill_value=0).mask(session_open, 0)
        entries.append(int((group["position"].ne(0) & prior.eq(0)).sum()))
        initial_positions.append(int(group.iloc[0]["position"]))
        initial_turnovers.append(float(group.iloc[0]["turnover"]))
        if "ending_position" in group:
            close = group["is_session_close"].astype(bool)
            violations += int(group.loc[close, "ending_position"].ne(0).sum())
        else:
            violations += int(group.loc[session_open, "position"].ne(0).sum())
    trade_count = float(np.mean(entries)) if series == "equal_weight" else float(entries[0])
    return {
        "turnover": turnover,
        "trade_count": trade_count,
        "eligible_observations": int(len(eligible)),
        "long_exposure_pct": long,
        "short_exposure_pct": short,
        "flat_exposure_pct": flat,
        "initial_position": int(max(map(abs, initial_positions))),
        "initial_turnover": float(max(initial_turnovers)),
        "overnight_position_violations": int(violations),
    }


def _performance_record(
    panel: pd.DataFrame,
    *,
    configuration_id: str,
    fold_id: str | None,
    cost_bps: float,
    execution: dict[str, float | int],
) -> dict[str, object]:
    returns = panel["net_return"].reset_index(drop=True)
    metrics = calculate_performance_metrics(
        returns, annualization_factor=ANNUALIZATION_FACTOR
    )
    strategy_family, phase = _configuration_metadata(configuration_id)
    record: dict[str, object] = {
        "configuration_id": configuration_id,
        "strategy_family": strategy_family,
        "phase": phase,
        "series": str(panel.iloc[0]["series"]),
        "cost_bps_per_turnover": float(cost_bps),
        "test_sessions": int(len(panel)),
        "start_session": pd.Timestamp(panel["session_date"].min()).strftime("%Y-%m-%d"),
        "end_session": pd.Timestamp(panel["session_date"].max()).strftime("%Y-%m-%d"),
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": metrics.max_drawdown,
        **execution,
    }
    if fold_id is not None:
        record["fold_id"] = fold_id
    return record


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
    return (
        float(np.mean(values) / standard_deviation * math.sqrt(ANNUALIZATION_FACTOR))
        if standard_deviation > 0.0
        else float("nan")
    )


def _bootstrap_intervals(
    values: np.ndarray, *, replications: int, seed: int
) -> tuple[float, float, float, float]:
    if replications <= 0:
        raise Phase2ProfitabilityError("bootstrap replications must be positive.")
    n = len(values)
    rng = np.random.default_rng(seed)
    block_count = int(math.ceil(n / BOOTSTRAP_BLOCK_LENGTH))
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH)
    means = np.empty(replications, dtype="float64")
    sharpes = np.empty(replications, dtype="float64")
    for replication in range(replications):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        sample = values[indices]
        means[replication] = float(np.mean(sample))
        sharpes[replication] = _annualized_sharpe(sample)
    finite_sharpes = sharpes[np.isfinite(sharpes)]
    mean_bounds = np.quantile(means, [0.025, 0.975])
    sharpe_bounds = (
        np.quantile(finite_sharpes, [0.025, 0.975])
        if len(finite_sharpes)
        else np.array([float("nan"), float("nan")])
    )
    return tuple(float(value) for value in (*mean_bounds, *sharpe_bounds))


def _break_even_cost_bps(gross_return: float, turnover: float) -> float:
    if turnover <= 0.0 or gross_return <= -1.0:
        return float("nan")
    return float(10_000.0 * math.log1p(gross_return) / turnover)


def run_phase2_profitability(
    bars: pd.DataFrame,
    *,
    source_dataset_id: str,
    source_sha256: str,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
) -> Phase2ProfitabilityResults:
    """Run the complete frozen Phase II development experiment."""

    data_quality = audit_development_data(
        bars,
        source_dataset_id=source_dataset_id,
        source_sha256=source_sha256,
    )
    features = _prepare_features(bars)
    sessions = pd.to_datetime(features["session_date"], utc=True).dt.normalize()
    spy = features.loc[features["symbol"].eq("SPY")].reset_index(drop=True)
    trend_baseline = build_trend_ratio_strategy(
        spy,
        parameters=TrendRatioParameters(
            short_window=TREND_SHORT_WINDOW,
            long_window=TREND_LONG_WINDOW,
            neutral_band=TREND_ENTRY_BAND,
            cost_bps_per_turnover=0.0,
            positioning="long_flat",
        ),
    ).observations
    trend_phase2 = build_persistent_hysteresis_signal(trend_baseline)

    fold_records: list[dict[str, object]] = []
    session_records: list[pd.DataFrame] = []
    aggregate_panels: dict[tuple[str, str, float], list[pd.DataFrame]] = {}
    aggregate_execution: dict[tuple[str, str], list[dict[str, float | int]]] = {}
    for fold in build_walk_forward_folds():
        fold_mask_spy = pd.to_datetime(
            trend_baseline["session_date"], utc=True
        ).dt.normalize().ge(fold.test_start) & pd.to_datetime(
            trend_baseline["session_date"], utc=True
        ).dt.normalize().lt(fold.test_end_exclusive)
        for configuration_id, source in zip(
            TREND_CONFIGURATION_IDS, (trend_baseline, trend_phase2), strict=True
        ):
            test_source = source.loc[fold_mask_spy].copy(deep=True).reset_index(drop=True)
            timed = apply_next_open_overnight_flat(
                test_source, cost_bps_per_turnover=0.0
            )
            execution = _execution_statistics(timed, series="SPY")
            aggregate_execution.setdefault((configuration_id, "SPY"), []).append(execution)
            for cost in COST_STRESS_BPS:
                panel = _session_panel(timed, cost_bps=cost)
                panel = panel.loc[panel["series"].eq("SPY")].reset_index(drop=True)
                fold_records.append(
                    _performance_record(
                        panel,
                        configuration_id=configuration_id,
                        fold_id=fold.fold_id,
                        cost_bps=cost,
                        execution=execution,
                    )
                )
                aggregate_panels.setdefault((configuration_id, "SPY", cost), []).append(panel)
            stored = _session_panel(timed, cost_bps=BASE_COST_BPS)
            stored = stored.loc[stored["series"].eq("SPY")].copy()
            stored.insert(0, "fold_id", fold.fold_id)
            stored.insert(0, "configuration_id", configuration_id)
            stored = stored.rename(columns={"net_return": "net_return_1bp"})
            session_records.append(stored.loc[:, SESSION_RETURN_COLUMNS])

        history_mask = sessions.lt(fold.test_end_exclusive)
        test_mask = sessions.ge(fold.test_start) & sessions.lt(fold.test_end_exclusive)
        history = features.loc[history_mask].copy(deep=True).reset_index(drop=True)
        test_source = features.loc[test_mask]
        reset_timestamps = tuple(
            pd.Timestamp(value)
            for value in test_source.groupby("symbol", observed=True)["timestamp"].min()
        )
        baseline_bundle = build_ou_vwap_reversion_strategy(
            history,
            parameters=OU_SLOW_PARAMETERS,
            execution_reset_timestamps=reset_timestamps,
        )
        result_sessions = pd.to_datetime(
            baseline_bundle.observations["session_date"], utc=True
        ).dt.normalize()
        baseline_test = baseline_bundle.observations.loc[
            result_sessions.ge(fold.test_start)
            & result_sessions.lt(fold.test_end_exclusive)
        ].copy(deep=True)
        baseline_test = apply_ou_next_open_accounting(baseline_test)
        phase2_history = apply_ou_cost_margin_gate(
            baseline_bundle.observations,
            execution_reset_timestamps=reset_timestamps,
        )
        phase2_sessions = pd.to_datetime(
            phase2_history["session_date"], utc=True
        ).dt.normalize()
        phase2_test = phase2_history.loc[
            phase2_sessions.ge(fold.test_start)
            & phase2_sessions.lt(fold.test_end_exclusive)
        ].copy(deep=True)
        phase2_test = apply_ou_next_open_accounting(phase2_test)
        for configuration_id, test in zip(
            OU_CONFIGURATION_IDS, (baseline_test, phase2_test), strict=True
        ):
            for series in OU_SERIES:
                execution = _execution_statistics(test, series=series)
                aggregate_execution.setdefault((configuration_id, series), []).append(execution)
                for cost in COST_STRESS_BPS:
                    panel = _session_panel(test, cost_bps=cost)
                    panel = panel.loc[panel["series"].eq(series)].reset_index(drop=True)
                    fold_records.append(
                        _performance_record(
                            panel,
                            configuration_id=configuration_id,
                            fold_id=fold.fold_id,
                            cost_bps=cost,
                            execution=execution,
                        )
                    )
                    aggregate_panels.setdefault((configuration_id, series, cost), []).append(panel)
                stored = _session_panel(test, cost_bps=BASE_COST_BPS)
                stored = stored.loc[stored["series"].eq(series)].copy()
                stored.insert(0, "fold_id", fold.fold_id)
                stored.insert(0, "configuration_id", configuration_id)
                stored = stored.rename(columns={"net_return": "net_return_1bp"})
                session_records.append(stored.loc[:, SESSION_RETURN_COLUMNS])

    fold_performance = pd.DataFrame.from_records(fold_records).loc[:, FOLD_COLUMNS]
    aggregate_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    concatenated_1bp: dict[tuple[str, str], pd.DataFrame] = {}
    for configuration_id in CONFIGURATION_IDS:
        series_order = ("SPY",) if configuration_id in TREND_CONFIGURATION_IDS else OU_SERIES
        for series in series_order:
            execution_parts = aggregate_execution[(configuration_id, series)]
            eligible_total = int(
                sum(int(row["eligible_observations"]) for row in execution_parts)
            )
            if eligible_total <= 0:
                raise Phase2ProfitabilityError("Aggregate exposure has no eligible rows.")
            combined_execution: dict[str, float | int] = {
                "turnover": float(sum(float(row["turnover"]) for row in execution_parts)),
                "trade_count": float(sum(float(row["trade_count"]) for row in execution_parts)),
                "eligible_observations": eligible_total,
                "long_exposure_pct": float(
                    sum(
                        float(row["long_exposure_pct"])
                        * int(row["eligible_observations"])
                        for row in execution_parts
                    )
                    / eligible_total
                ),
                "short_exposure_pct": float(
                    sum(
                        float(row["short_exposure_pct"])
                        * int(row["eligible_observations"])
                        for row in execution_parts
                    )
                    / eligible_total
                ),
                "flat_exposure_pct": float(
                    sum(
                        float(row["flat_exposure_pct"])
                        * int(row["eligible_observations"])
                        for row in execution_parts
                    )
                    / eligible_total
                ),
                "initial_position": int(max(int(row["initial_position"]) for row in execution_parts)),
                "initial_turnover": float(max(float(row["initial_turnover"]) for row in execution_parts)),
                "overnight_position_violations": int(sum(int(row["overnight_position_violations"]) for row in execution_parts)),
            }
            for cost in COST_STRESS_BPS:
                panel = pd.concat(
                    aggregate_panels[(configuration_id, series, cost)], ignore_index=True
                ).sort_values("session_date", kind="stable").reset_index(drop=True)
                if panel["session_date"].duplicated().any():
                    raise Phase2ProfitabilityError("Aggregate test sessions overlap.")
                record = _performance_record(
                    panel,
                    configuration_id=configuration_id,
                    fold_id=None,
                    cost_bps=cost,
                    execution=combined_execution,
                )
                gross_metrics = calculate_performance_metrics(
                    panel["gross_return"], annualization_factor=ANNUALIZATION_FACTOR
                )
                record["gross_cumulative_return"] = gross_metrics.cumulative_return
                record["cost_drag"] = gross_metrics.cumulative_return - float(record["cumulative_return"])
                record["approximate_break_even_cost_bps_per_turnover"] = _break_even_cost_bps(
                    gross_metrics.cumulative_return,
                    float(combined_execution["turnover"]),
                )
                if cost == BASE_COST_BPS:
                    matching = fold_performance.loc[
                        fold_performance["configuration_id"].eq(configuration_id)
                        & fold_performance["series"].eq(series)
                        & fold_performance["cost_bps_per_turnover"].eq(cost)
                    ]
                    record["positive_folds"] = int(matching["cumulative_return"].gt(0.0).sum())
                    aggregate_records.append(record)
                    concatenated_1bp[(configuration_id, series)] = panel
                cost_records.append(record)

    aggregate_performance = pd.DataFrame.from_records(aggregate_records).loc[
        :, AGGREGATE_COLUMNS
    ]
    cost_sensitivity = pd.DataFrame.from_records(cost_records).loc[:, COST_COLUMNS]

    comparison_records: list[dict[str, object]] = []
    for family, series, baseline_id, phase2_id in (
        (
            "price_ratio_long_flat",
            "SPY",
            TREND_CONFIGURATION_IDS[0],
            TREND_CONFIGURATION_IDS[1],
        ),
        (
            "ou_vwap_slow",
            "equal_weight",
            OU_CONFIGURATION_IDS[0],
            OU_CONFIGURATION_IDS[1],
        ),
    ):
        for cost in (1.0, 5.0):
            source = aggregate_performance if cost == 1.0 else cost_sensitivity
            baseline = source.loc[
                source["configuration_id"].eq(baseline_id)
                & source["series"].eq(series)
                & source["cost_bps_per_turnover"].eq(cost)
            ].iloc[0]
            phase2 = source.loc[
                source["configuration_id"].eq(phase2_id)
                & source["series"].eq(series)
                & source["cost_bps_per_turnover"].eq(cost)
            ].iloc[0]
            baseline_positive = int(
                fold_performance.loc[
                    fold_performance["configuration_id"].eq(baseline_id)
                    & fold_performance["series"].eq(series)
                    & fold_performance["cost_bps_per_turnover"].eq(cost),
                    "cumulative_return",
                ].gt(0.0).sum()
            )
            phase2_positive = int(
                fold_performance.loc[
                    fold_performance["configuration_id"].eq(phase2_id)
                    & fold_performance["series"].eq(series)
                    & fold_performance["cost_bps_per_turnover"].eq(cost),
                    "cumulative_return",
                ].gt(0.0).sum()
            )
            turnover_change = (
                100.0 * (float(phase2["turnover"]) / float(baseline["turnover"]) - 1.0)
                if float(baseline["turnover"]) > 0.0
                else float("nan")
            )
            comparison_records.append(
                {
                    "strategy_family": family,
                    "series": series,
                    "cost_bps_per_turnover": cost,
                    "baseline_configuration_id": baseline_id,
                    "phase2_configuration_id": phase2_id,
                    "baseline_cumulative_return": baseline["cumulative_return"],
                    "phase2_cumulative_return": phase2["cumulative_return"],
                    "cumulative_return_change": float(phase2["cumulative_return"]) - float(baseline["cumulative_return"]),
                    "baseline_annualized_volatility": baseline["annualized_volatility"],
                    "phase2_annualized_volatility": phase2["annualized_volatility"],
                    "baseline_sharpe_ratio": baseline["sharpe_ratio"],
                    "phase2_sharpe_ratio": phase2["sharpe_ratio"],
                    "baseline_maximum_drawdown": baseline["maximum_drawdown"],
                    "phase2_maximum_drawdown": phase2["maximum_drawdown"],
                    "baseline_turnover": baseline["turnover"],
                    "phase2_turnover": phase2["turnover"],
                    "turnover_change_pct": turnover_change,
                    "baseline_trade_count": baseline["trade_count"],
                    "phase2_trade_count": phase2["trade_count"],
                    "baseline_positive_folds": baseline_positive,
                    "phase2_positive_folds": phase2_positive,
                    "development_net_return_improved": bool(
                        float(phase2["cumulative_return"]) > float(baseline["cumulative_return"])
                    ),
                }
            )
    comparison = pd.DataFrame.from_records(comparison_records).loc[:, COMPARISON_COLUMNS]

    inference_records: list[dict[str, object]] = []
    for configuration_index, configuration_id in enumerate(CONFIGURATION_IDS):
        series_order = ("SPY",) if configuration_id in TREND_CONFIGURATION_IDS else OU_SERIES
        strategy_family, phase = _configuration_metadata(configuration_id)
        for series_index, series in enumerate(series_order):
            values = concatenated_1bp[(configuration_id, series)]["net_return"].to_numpy(
                dtype="float64"
            )
            naive, hac = _t_statistics(values)
            low_mean, high_mean, low_sharpe, high_sharpe = _bootstrap_intervals(
                values,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + configuration_index * len(OU_SERIES) + series_index,
            )
            inference_records.append(
                {
                    "configuration_id": configuration_id,
                    "strategy_family": strategy_family,
                    "phase": phase,
                    "series": series,
                    "cost_bps_per_turnover": BASE_COST_BPS,
                    "observations": int(len(values)),
                    "mean_session_return": float(np.mean(values)),
                    "naive_t_statistic": naive,
                    "hac_lags": HAC_LAGS,
                    "hac_t_statistic": hac,
                    "annualized_sharpe_ratio": _annualized_sharpe(values),
                    "bootstrap_replications": int(bootstrap_replications),
                    "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
                    "bootstrap_mean_ci_lower": low_mean,
                    "bootstrap_mean_ci_upper": high_mean,
                    "bootstrap_sharpe_ci_lower": low_sharpe,
                    "bootstrap_sharpe_ci_upper": high_sharpe,
                    "declared_phase2_trials": 2,
                }
            )
    inference = pd.DataFrame.from_records(inference_records).loc[:, INFERENCE_COLUMNS]
    session_returns = pd.concat(session_records, ignore_index=True).loc[
        :, SESSION_RETURN_COLUMNS
    ]

    if tuple(aggregate_performance["configuration_id"].drop_duplicates()) != CONFIGURATION_IDS:
        raise RuntimeError("Aggregate configuration order changed.")
    if len(fold_performance) != 160 or len(cost_sensitivity) != 40 or len(comparison) != 4:
        raise RuntimeError("Phase II result row counts changed.")
    if int(fold_performance["overnight_position_violations"].sum()) != 0:
        raise RuntimeError("Phase II evaluation contains overnight positions.")
    return Phase2ProfitabilityResults(
        data_quality=data_quality,
        aggregate_performance=aggregate_performance,
        fold_performance=fold_performance,
        cost_sensitivity=cost_sensitivity,
        comparison=comparison,
        inference=inference,
        session_returns=session_returns,
    )


def _format(value: object, *, percent: bool = False) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100.0:+.2f}%" if percent else f"{number:.3f}"


def render_phase2_report(results: Phase2ProfitabilityResults) -> str:
    """Render an answer-first technical record without a promotion claim."""

    comparisons = results.comparison.loc[
        results.comparison["cost_bps_per_turnover"].eq(1.0)
    ]
    lines = [
        "# Day 26 Phase II Profitability Test",
        "",
        "## Technical summary",
        "",
        "This is a predeclared development-period comparison, not a new final test. "
        "The consumed January-June 2026 interval was not read or reused, and the local "
        "repository contains no untouched later holdout. Both declared Phase II trials "
        "are therefore reported as diagnostic evidence only.",
        "",
        "## Matching-baseline comparison at 1 bp per unit turnover",
        "",
        "| Family | Baseline return | Phase II return | Change | Baseline turnover | Phase II turnover | Positive folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.strategy_family} | {_format(row.baseline_cumulative_return, percent=True)} | "
            f"{_format(row.phase2_cumulative_return, percent=True)} | "
            f"{_format(row.cumulative_return_change, percent=True)} | "
            f"{_format(row.baseline_turnover)} | {_format(row.phase2_turnover)} | "
            f"{row.phase2_positive_folds}/4 |"
        )
    lines.extend(
        [
            "",
            "## Cost stress and uncertainty",
            "",
            "| Configuration | Series | Return at 1 bp | Return at 5 bp | HAC t-stat | Bootstrap mean 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for configuration_id in CONFIGURATION_IDS:
        series = "SPY" if configuration_id in TREND_CONFIGURATION_IDS else "equal_weight"
        one = results.cost_sensitivity.loc[
            results.cost_sensitivity["configuration_id"].eq(configuration_id)
            & results.cost_sensitivity["series"].eq(series)
            & results.cost_sensitivity["cost_bps_per_turnover"].eq(1.0)
        ].iloc[0]
        five = results.cost_sensitivity.loc[
            results.cost_sensitivity["configuration_id"].eq(configuration_id)
            & results.cost_sensitivity["series"].eq(series)
            & results.cost_sensitivity["cost_bps_per_turnover"].eq(5.0)
        ].iloc[0]
        inference = results.inference.loc[
            results.inference["configuration_id"].eq(configuration_id)
            & results.inference["series"].eq(series)
        ].iloc[0]
        interval = (
            f"[{float(inference['bootstrap_mean_ci_lower']):+.6f}, "
            f"{float(inference['bootstrap_mean_ci_upper']):+.6f}]"
        )
        lines.append(
            f"| {configuration_id} | {series} | "
            f"{_format(one['cumulative_return'], percent=True)} | "
            f"{_format(five['cumulative_return'], percent=True)} | "
            f"{_format(inference['hac_t_statistic'])} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A higher development-period net return is recorded only as a matching-"
            "baseline development result. It does not establish deployable profitability, "
            "does not replace the frozen baselines, and does not authorize promotion. A "
            "new untouched future holdout and empirical execution-cost evidence remain "
            "necessary for a profitability-improvement claim.",
            "The slow OU/VWAP expected-convergence gate was non-binding in this "
            "sample: it rejected no baseline entry and therefore left positions, "
            "turnover, and realized returns unchanged.",
            "",
            "## Source and reproducibility",
            "",
            f"- Canonical rows: {results.data_quality['rows']:,}; symbols: "
            f"{', '.join(results.data_quality['symbols'])}.",
            f"- Data end: {results.data_quality['timestamp_max']}; locked rows accessed: no.",
            "- Four expanding annual test folds, four fixed cost stresses, two declared "
            "Phase II trials, and all results reported.",
            "- OU forced exits are charged on the actual session-close row so fold-end "
            "liquidations cannot disappear from cost accounting.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n", float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_phase2_artifacts(
    results: Phase2ProfitabilityResults,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact deterministic ten-file Phase II evidence bundle."""

    if not isinstance(results, Phase2ProfitabilityResults):
        raise TypeError("results must be Phase2ProfitabilityResults.")
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    methodology = {
        "schema_version": SPECIFICATION_VERSION,
        "configurations": list(CONFIGURATION_IDS),
        "declared_phase2_trials": 2,
        "cost_stress_bps": list(COST_STRESS_BPS),
        "folds": [fold.fold_id for fold in build_walk_forward_folds()],
        "development_start": "2020-01-02",
        "development_end": "2025-12-31",
        "locked_2026_interval_accessed": False,
        "untouched_future_holdout_available": False,
        "winner_selection_performed": False,
        "negative_results_suppressed": False,
        "leverage_used": False,
        "broker_or_campaign_mutation": False,
        "trend": {
            "short_window": TREND_SHORT_WINDOW,
            "long_window": TREND_LONG_WINDOW,
            "entry_band": TREND_ENTRY_BAND,
            "exit_band": TREND_EXIT_BAND,
            "confirmation_bars": TREND_CONFIRMATION_BARS,
            "timing": "next_bar_open_overnight_flat_v1",
        },
        "ou": {
            "baseline_configuration": OU_SLOW_PARAMETERS.configuration_id,
            "expected_convergence_threshold": OU_EXPECTED_CONVERGENCE_THRESHOLD,
            "forced_close_cost_charged_on_session_close": True,
        },
        "inference": {
            "hac_lags": HAC_LAGS,
            "bootstrap_replications": int(
                results.inference["bootstrap_replications"].iloc[0]
            ),
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "claim_boundary": "development_evidence_only",
        "break_even_cost_method": (
            "log_linear_approximation_10000_times_log1p_gross_return_divided_by_turnover"
        ),
    }
    payloads: dict[str, bytes] = {
        DATA_QUALITY_FILENAME: _json_bytes(results.data_quality),
        AGGREGATE_FILENAME: _csv_bytes(results.aggregate_performance),
        FOLD_FILENAME: _csv_bytes(results.fold_performance),
        COST_FILENAME: _csv_bytes(results.cost_sensitivity),
        COMPARISON_FILENAME: _csv_bytes(results.comparison),
        INFERENCE_FILENAME: _csv_bytes(results.inference),
        SESSION_RETURNS_FILENAME: _csv_bytes(results.session_returns),
        METHODOLOGY_FILENAME: _json_bytes(methodology),
        REPORT_FILENAME: render_phase2_report(results).encode("utf-8"),
    }
    manifest = {
        "schema_version": "day26_phase2_profitability_artifacts_v1",
        "artifact_order": list(APPROVED_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
        "row_counts": {
            AGGREGATE_FILENAME: len(results.aggregate_performance),
            FOLD_FILENAME: len(results.fold_performance),
            COST_FILENAME: len(results.cost_sensitivity),
            COMPARISON_FILENAME: len(results.comparison),
            INFERENCE_FILENAME: len(results.inference),
            SESSION_RETURNS_FILENAME: len(results.session_returns),
        },
        "locked_period_accessed": False,
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    if tuple(payloads) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Phase II artifact allow-list changed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        for name in APPROVED_ARTIFACT_NAMES:
            (stage / name).write_bytes(payloads[name])
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
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
    observed_names = tuple(sorted(path.name for path in output.iterdir()))
    if observed_names != tuple(sorted(APPROVED_ARTIFACT_NAMES)):
        raise RuntimeError("Final Phase II artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_ARTIFACT_NAMES)
