"""Frozen development-only audit of trend positioning and execution timing."""

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
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    REQUIRED_INPUT_SYMBOLS,
    _validate_complete_coverage,
    _validate_date_and_symbol_scope,
    build_walk_forward_folds,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import build_parameter_grid
from systematic_alpha.data.session_aggregation import aggregate_session_bars
from systematic_alpha.strategies.ema_macd import EmaMacdParameters, build_ema_macd_strategy
from systematic_alpha.strategies.trend_ratio import (
    TrendRatioParameters,
    build_trend_ratio_strategy,
)


SPECIFICATION_VERSION: Final[str] = "day25_methodological_finalization_v1"
SAVED_TIMING: Final[str] = "close_to_close_one_row_lag_overnight_carry"
FINAL_TIMING: Final[str] = "next_bar_open_overnight_flat_v1"
MODEL_IDS: Final[tuple[str, ...]] = (
    "price_ratio_long_short_neutral",
    "price_ratio_long_flat",
    "ema_macd_long_short_neutral",
)
COST_STRESS_BPS: Final[tuple[float, ...]] = (0.0, 1.0, 2.5, 5.0)
ROBUSTNESS_SYMBOLS: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM")
ROBUSTNESS_FREQUENCIES: Final[tuple[str, ...]] = ("15min", "30min", "60min")

TREND_SHORT_WINDOW: Final[int] = 8
TREND_LONG_WINDOW: Final[int] = 32
TREND_NEUTRAL_BAND: Final[float] = 0.001
EMA_FAST_WINDOW: Final[int] = 12
EMA_SLOW_WINDOW: Final[int] = 26
EMA_SIGNAL_WINDOW: Final[int] = 9
EMA_NEUTRAL_BAND: Final[float] = 0.0005

TIMING_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "strategy",
    "positioning",
    "timing_convention",
    "cost_bps_per_turnover",
    "sample_start_timestamp",
    "sample_end_timestamp",
    "sessions",
    "observations",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "trade_count",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "session_close_nonflat_targets",
    "overnight_positions_held",
)

WALK_FORWARD_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "strategy",
    "positioning",
    "fold_id",
    "cost_bps_per_turnover",
    "test_start_timestamp",
    "test_end_timestamp",
    "test_sessions",
    "test_observations",
    "annualization_factor",
    "initial_position",
    "initial_turnover",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "trade_count",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "overnight_position_violations",
)

ROBUSTNESS_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "strategy",
    "positioning",
    "symbol",
    "frequency",
    "cost_bps_per_turnover",
    "sessions",
    "observations",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "trade_count",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "overnight_position_violations",
)

SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "configuration_id",
    "short_window",
    "long_window",
    "neutral_band",
    "positioning",
    "timing_convention",
    "cost_bps_per_turnover",
    "is_predeclared_baseline",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "trade_count",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
)

PARITY_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "observations",
    "position_exact",
    "ending_position_exact",
    "max_abs_turnover_difference",
    "max_abs_gross_return_difference",
    "max_abs_transaction_cost_difference",
    "max_abs_net_return_difference",
    "parity_passed",
)

TIMING_FILENAME: Final[str] = "timing_comparison.csv"
WALK_FORWARD_FILENAME: Final[str] = "walk_forward.csv"
ROBUSTNESS_FILENAME: Final[str] = "robustness.csv"
SENSITIVITY_FILENAME: Final[str] = "long_flat_sensitivity.csv"
PARITY_FILENAME: Final[str] = "replay_parity.csv"
METHODOLOGY_FILENAME: Final[str] = "methodology.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    TIMING_FILENAME,
    WALK_FORWARD_FILENAME,
    ROBUSTNESS_FILENAME,
    SENSITIVITY_FILENAME,
    PARITY_FILENAME,
    METHODOLOGY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)


class TrendMethodologyError(ValueError):
    """Raised when the frozen methodological audit cannot run safely."""


@dataclass(frozen=True, slots=True)
class TrendMethodologyResults:
    """Compact tables from the frozen development-only audit."""

    timing_comparison: pd.DataFrame
    walk_forward: pd.DataFrame
    robustness: pd.DataFrame
    long_flat_sensitivity: pd.DataFrame
    replay_parity: pd.DataFrame


def prepare_development_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact canonical development scope without locked rows."""

    scoped = _validate_date_and_symbol_scope(bars)
    fifteen = aggregate_session_bars(scoped, "15min")
    _validate_complete_coverage(fifteen)
    if pd.to_datetime(fifteen["session_date"]).max() >= pd.Timestamp("2026-01-01"):
        raise TrendMethodologyError("Locked-period rows are prohibited.")
    return fifteen


def _model_descriptor(model_id: str) -> tuple[str, str]:
    if model_id == "price_ratio_long_short_neutral":
        return "price_ratio", "long_short_neutral"
    if model_id == "price_ratio_long_flat":
        return "price_ratio", "long_flat"
    if model_id == "ema_macd_long_short_neutral":
        return "ema_macd", "long_short_neutral"
    raise TrendMethodologyError(f"Unknown frozen model: {model_id!r}.")


def build_model_observations(
    frame: pd.DataFrame,
    *,
    model_id: str,
    trend_parameters: TrendRatioParameters | None = None,
) -> pd.DataFrame:
    """Build signals for one frozen model without relying on reported PnL."""

    strategy, positioning = _model_descriptor(model_id)
    if strategy == "price_ratio":
        parameters = trend_parameters or TrendRatioParameters(
            short_window=TREND_SHORT_WINDOW,
            long_window=TREND_LONG_WINDOW,
            neutral_band=TREND_NEUTRAL_BAND,
            cost_bps_per_turnover=0.0,
            positioning=positioning,
        )
        if parameters.positioning != positioning:
            raise TrendMethodologyError("Trend positioning changed from model id.")
        observations = build_trend_ratio_strategy(
            frame,
            parameters=parameters,
        ).observations
    else:
        if trend_parameters is not None:
            raise TrendMethodologyError("trend_parameters apply only to price ratio.")
        observations = build_ema_macd_strategy(
            frame,
            parameters=EmaMacdParameters(
                fast_window=EMA_FAST_WINDOW,
                slow_window=EMA_SLOW_WINDOW,
                signal_window=EMA_SIGNAL_WINDOW,
                neutral_band=EMA_NEUTRAL_BAND,
                cost_bps_per_turnover=0.0,
            ),
        ).observations
    return observations.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def apply_saved_timing(
    observations: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Recalculate the labeled historical close-to-close convention."""

    result = observations.copy(deep=True)
    result["transaction_cost"] = (
        result["turnover"].astype("float64")
        * float(cost_bps_per_turnover)
        / 10_000.0
    )
    result["net_strategy_return"] = (
        result["gross_strategy_return"].astype("float64")
        - result["transaction_cost"]
    )
    sessions = result["session_date"].astype("string")
    result["is_session_close"] = (
        sessions.ne(sessions.shift(-1)).fillna(True).astype(bool)
    )
    result["ending_position"] = result["position"].astype("int8")
    return result


def apply_next_open_overnight_flat(
    observations: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Apply the frozen causal next-open and forced-flat return convention."""

    try:
        return apply_causal_next_open_overnight_flat(
            observations,
            cost_bps_per_turnover=cost_bps_per_turnover,
        )
    except CausalBarExecutionError as exc:
        raise TrendMethodologyError(str(exc)) from exc


def _annualization_factor(frame: pd.DataFrame) -> float:
    sessions = int(frame["session_date"].nunique())
    if sessions <= 0:
        raise TrendMethodologyError("Performance requires at least one session.")
    return float(252.0 * len(frame) / sessions)


def _exposures(frame: pd.DataFrame) -> tuple[float, float, float]:
    eligible = frame.loc[frame["position_eligible"].astype(bool), "position"]
    if eligible.empty:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(100.0 * eligible.eq(value).mean()) for value in (1, -1, 0))


def _metrics(frame: pd.DataFrame) -> tuple[PerformanceMetrics, float]:
    factor = _annualization_factor(frame)
    return (
        calculate_performance_metrics(
            frame["net_strategy_return"],
            annualization_factor=factor,
        ),
        factor,
    )


def _common_performance(frame: pd.DataFrame) -> dict[str, object]:
    metrics, factor = _metrics(frame)
    long, short, flat = _exposures(frame)
    return {
        "annualization_factor": factor,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": metrics.max_drawdown,
        "turnover": float(frame["turnover"].sum()),
        "trade_count": int(frame["turnover"].gt(0.0).sum()),
        "long_exposure": long,
        "short_exposure": short,
        "flat_exposure": flat,
    }


def _timing_results(spy: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        strategy, positioning = _model_descriptor(model_id)
        signals = build_model_observations(spy, model_id=model_id)
        for convention, result in (
            (SAVED_TIMING, apply_saved_timing(signals, cost_bps_per_turnover=1.0)),
            (FINAL_TIMING, apply_next_open_overnight_flat(signals, cost_bps_per_turnover=1.0)),
        ):
            closes = result["is_session_close"].astype(bool)
            records.append(
                {
                    "model_id": model_id,
                    "strategy": strategy,
                    "positioning": positioning,
                    "timing_convention": convention,
                    "cost_bps_per_turnover": 1.0,
                    "sample_start_timestamp": result["timestamp"].min(),
                    "sample_end_timestamp": result["timestamp"].max(),
                    "sessions": int(result["session_date"].nunique()),
                    "observations": int(len(result)),
                    **_common_performance(result),
                    "session_close_nonflat_targets": int(result.loc[closes, "position"].ne(0).sum()),
                    "overnight_positions_held": (
                        int(result.loc[closes, "ending_position"].ne(0).sum())
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=TIMING_COLUMNS)


def _walk_forward_results(spy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aggregate_inputs: dict[tuple[str, float], list[pd.DataFrame]] = {
        (model_id, cost): [] for model_id in MODEL_IDS for cost in COST_STRESS_BPS
    }
    sessions = pd.to_datetime(spy["session_date"], utc=True).dt.normalize()
    for model_id in MODEL_IDS:
        strategy, positioning = _model_descriptor(model_id)
        signals = build_model_observations(spy, model_id=model_id)
        for fold in build_walk_forward_folds():
            mask = sessions.ge(fold.test_start) & sessions.lt(fold.test_end_exclusive)
            test_signals = signals.loc[mask].copy(deep=True).reset_index(drop=True)
            if test_signals.empty:
                raise TrendMethodologyError(f"{fold.fold_id} has no test observations.")
            base = apply_next_open_overnight_flat(test_signals, cost_bps_per_turnover=0.0)
            for cost in COST_STRESS_BPS:
                result = base.copy(deep=True)
                result["transaction_cost"] = result["turnover"] * cost / 10_000.0
                result["net_strategy_return"] = result["gross_strategy_return"] - result["transaction_cost"]
                aggregate_inputs[(model_id, cost)].append(result)
                rows.append(
                    {
                        "model_id": model_id,
                        "strategy": strategy,
                        "positioning": positioning,
                        "fold_id": fold.fold_id,
                        "cost_bps_per_turnover": cost,
                        "test_start_timestamp": result["timestamp"].min(),
                        "test_end_timestamp": result["timestamp"].max(),
                        "test_sessions": int(result["session_date"].nunique()),
                        "test_observations": int(len(result)),
                        "initial_position": int(result.iloc[0]["position"]),
                        "initial_turnover": float(result.iloc[0]["turnover"]),
                        **_common_performance(result),
                        "overnight_position_violations": int(
                            result.loc[result["is_session_close"], "ending_position"].ne(0).sum()
                        ),
                    }
                )
    for model_id in MODEL_IDS:
        strategy, positioning = _model_descriptor(model_id)
        for cost in COST_STRESS_BPS:
            combined = pd.concat(aggregate_inputs[(model_id, cost)], ignore_index=True)
            rows.append(
                {
                    "model_id": model_id,
                    "strategy": strategy,
                    "positioning": positioning,
                    "fold_id": "aggregate_2022_2025",
                    "cost_bps_per_turnover": cost,
                    "test_start_timestamp": combined["timestamp"].min(),
                    "test_end_timestamp": combined["timestamp"].max(),
                    "test_sessions": int(combined["session_date"].nunique()),
                    "test_observations": int(len(combined)),
                    "initial_position": int(combined.iloc[0]["position"]),
                    "initial_turnover": float(combined.iloc[0]["turnover"]),
                    **_common_performance(combined),
                    "overnight_position_violations": int(
                        combined.loc[combined["is_session_close"], "ending_position"].ne(0).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows, columns=WALK_FORWARD_COLUMNS)


def _robustness_results(bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frequency in ROBUSTNESS_FREQUENCIES:
        aggregated = aggregate_session_bars(bars, frequency)
        features = build_return_features(
            aggregated,
            expected_symbols=REQUIRED_INPUT_SYMBOLS,
        ).bars
        for symbol in ROBUSTNESS_SYMBOLS:
            symbol_frame = features.loc[features["symbol"].eq(symbol)].reset_index(drop=True)
            for model_id in MODEL_IDS:
                strategy, positioning = _model_descriptor(model_id)
                signals = build_model_observations(symbol_frame, model_id=model_id)
                result = apply_next_open_overnight_flat(signals, cost_bps_per_turnover=1.0)
                rows.append(
                    {
                        "model_id": model_id,
                        "strategy": strategy,
                        "positioning": positioning,
                        "symbol": symbol,
                        "frequency": frequency,
                        "cost_bps_per_turnover": 1.0,
                        "sessions": int(result["session_date"].nunique()),
                        "observations": int(len(result)),
                        **_common_performance(result),
                        "overnight_position_violations": int(
                            result.loc[result["is_session_close"], "ending_position"].ne(0).sum()
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows, columns=ROBUSTNESS_COLUMNS)


def _sensitivity_results(spy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for configuration in build_parameter_grid():
        parameters = TrendRatioParameters(
            short_window=configuration.short_window,
            long_window=configuration.long_window,
            neutral_band=configuration.neutral_band,
            cost_bps_per_turnover=0.0,
            positioning="long_flat",
        )
        signals = build_model_observations(
            spy,
            model_id="price_ratio_long_flat",
            trend_parameters=parameters,
        )
        result = apply_next_open_overnight_flat(signals, cost_bps_per_turnover=1.0)
        rows.append(
            {
                "configuration_id": configuration.configuration_id,
                "short_window": configuration.short_window,
                "long_window": configuration.long_window,
                "neutral_band": configuration.neutral_band,
                "positioning": "long_flat",
                "timing_convention": FINAL_TIMING,
                "cost_bps_per_turnover": 1.0,
                "is_predeclared_baseline": bool(
                    configuration.short_window == TREND_SHORT_WINDOW
                    and configuration.long_window == TREND_LONG_WINDOW
                    and configuration.neutral_band == TREND_NEUTRAL_BAND
                ),
                **_common_performance(result),
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=SENSITIVITY_COLUMNS)
    if len(frame) != 36 or int(frame["is_predeclared_baseline"].sum()) != 1:
        raise RuntimeError("Long-flat sensitivity grid contract changed.")
    return frame


def sequential_next_open_replay(
    observations: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Replay the final timing convention one event at a time."""

    rows: list[dict[str, object]] = []
    for _, source in observations.groupby("symbol", observed=True, sort=True):
        part = source.sort_values("timestamp", kind="stable").reset_index(drop=True)
        sessions = part["session_date"].astype(str).to_numpy()
        signals = part["signal"].to_numpy(dtype="int8")
        opens = part["open"].to_numpy(dtype="float64")
        closes = part["close"].to_numpy(dtype="float64")
        previous_target = 0
        previous_session: str | None = None
        for index in range(len(part)):
            session = sessions[index]
            session_open = previous_session != session
            session_close = index == len(part) - 1 or sessions[index + 1] != session
            position = 0 if index == 0 else int(signals[index - 1])
            prior_end = 0 if session_open else previous_target
            open_turnover = abs(position - prior_end)
            close_turnover = abs(position) if session_close else 0
            if session_close:
                raw_return = closes[index] / opens[index] - 1.0
            else:
                raw_return = opens[index + 1] / opens[index] - 1.0
            turnover = float(open_turnover + close_turnover)
            gross = float(position) * raw_return
            cost = turnover * float(cost_bps_per_turnover) / 10_000.0
            rows.append(
                {
                    "position": position,
                    "ending_position": 0 if session_close else position,
                    "turnover": turnover,
                    "gross_strategy_return": gross,
                    "transaction_cost": cost,
                    "net_strategy_return": gross - cost,
                }
            )
            previous_target = position
            previous_session = session
    return pd.DataFrame.from_records(rows)


def _parity_results(spy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    float_columns = (
        "turnover",
        "gross_strategy_return",
        "transaction_cost",
        "net_strategy_return",
    )
    for model_id in MODEL_IDS:
        signals = build_model_observations(spy, model_id=model_id)
        batch = apply_next_open_overnight_flat(signals, cost_bps_per_turnover=1.0)
        replay = sequential_next_open_replay(signals, cost_bps_per_turnover=1.0)
        differences = {
            column: float(
                np.max(
                    np.abs(
                        batch[column].to_numpy(dtype="float64")
                        - replay[column].to_numpy(dtype="float64")
                    )
                )
            )
            for column in float_columns
        }
        position_exact = bool(
            np.array_equal(
                batch["position"].to_numpy(dtype="int8"),
                replay["position"].to_numpy(dtype="int8"),
            )
        )
        ending_exact = bool(
            np.array_equal(
                batch["ending_position"].to_numpy(dtype="int8"),
                replay["ending_position"].to_numpy(dtype="int8"),
            )
        )
        passed = position_exact and ending_exact and max(differences.values()) <= 1.0e-12
        rows.append(
            {
                "model_id": model_id,
                "observations": len(batch),
                "position_exact": position_exact,
                "ending_position_exact": ending_exact,
                "max_abs_turnover_difference": differences["turnover"],
                "max_abs_gross_return_difference": differences["gross_strategy_return"],
                "max_abs_transaction_cost_difference": differences["transaction_cost"],
                "max_abs_net_return_difference": differences["net_strategy_return"],
                "parity_passed": passed,
            }
        )
    result = pd.DataFrame.from_records(rows, columns=PARITY_COLUMNS)
    if not result["parity_passed"].all():
        raise RuntimeError("Sequential replay did not match batch accounting.")
    return result


def run_trend_methodology_finalization(bars: pd.DataFrame) -> TrendMethodologyResults:
    """Run the complete frozen trend finalization matrix."""

    fifteen = prepare_development_bars(bars)
    features = build_return_features(
        fifteen,
        expected_symbols=REQUIRED_INPUT_SYMBOLS,
    ).bars
    spy = features.loc[features["symbol"].eq("SPY")].reset_index(drop=True)
    results = TrendMethodologyResults(
        timing_comparison=_timing_results(spy),
        walk_forward=_walk_forward_results(spy),
        robustness=_robustness_results(fifteen),
        long_flat_sensitivity=_sensitivity_results(spy),
        replay_parity=_parity_results(spy),
    )
    expected_rows = (6, 60, 27, 36, 3)
    actual_rows = tuple(
        len(frame)
        for frame in (
            results.timing_comparison,
            results.walk_forward,
            results.robustness,
            results.long_flat_sensitivity,
            results.replay_parity,
        )
    )
    if actual_rows != expected_rows:
        raise RuntimeError(f"Finalization row counts changed: {actual_rows}.")
    return results


def _format(value: object, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100.0:+.2f}%" if percent else f"{number:.4f}"


def render_methodology_report(results: TrendMethodologyResults) -> str:
    """Render a concise, result-neutral methodological addendum."""

    timing = results.timing_comparison
    aggregate = results.walk_forward.loc[
        results.walk_forward["fold_id"].eq("aggregate_2022_2025")
        & results.walk_forward["cost_bps_per_turnover"].eq(1.0)
    ]
    lines = [
        "# Day 25 Trend Methodological Finalization",
        "",
        "## Verdict",
        "",
        "The saved trend evidence is preserved as historical accounting evidence. "
        "The causal next-bar-open, overnight-flat convention is the final development "
        "protocol. Long-flat is reported as a required comparison, not a replacement "
        "selected after observing returns. No locked 2026 row was accessed.",
        "",
        "## Full-development timing comparison at 1 bp",
        "",
        "| Model | Timing | Cumulative return | Sharpe | Turnover | Overnight positions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in timing.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.timing_convention} | "
            f"{_format(row.cumulative_return, percent=True)} | "
            f"{_format(row.sharpe_ratio)} | {_format(row.turnover)} | "
            f"{int(row.overnight_positions_held)} |"
        )
    lines.extend(
        [
            "",
            "## Chronological 2022-2025 aggregate at 1 bp",
            "",
            "| Model | Positioning | Cumulative return | Sharpe | Maximum drawdown | Turnover |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.positioning} | "
            f"{_format(row.cumulative_return, percent=True)} | "
            f"{_format(row.sharpe_ratio)} | "
            f"{_format(row.maximum_drawdown, percent=True)} | "
            f"{_format(row.turnover)} |"
        )
    positive = int(results.long_flat_sensitivity["cumulative_return"].gt(0.0).sum())
    lines.extend(
        [
            "",
            "## Sensitivity, robustness, and replay",
            "",
            f"- Long-flat sensitivity reports all 36 frozen grid points; {positive} "
            "were positive after one-basis-point turnover cost. This count is "
            "descriptive and does not select a winner.",
            "- Robustness contains all 27 model-symbol-frequency cases under the "
            "final convention.",
            "- All three batch calculations matched sequential replay, including "
            "positions, forced exits, turnover, costs, and returns.",
            "- All session closes end flat under the final convention.",
            "",
            "## Interpretation boundary",
            "",
            "A causal proxy does not guarantee execution at the first trade of each "
            "bar. The fixed cost stresses remain necessary. Positive cases do not "
            "establish deployable profitability, and negative cases remain part of "
            "the final evidence chain.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n", float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_trend_methodology_artifacts(
    results: TrendMethodologyResults,
    directory: str | Path,
    *,
    source_dataset_id: str,
    source_sha256: str,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact deterministic eight-file methodological bundle."""

    if not isinstance(results, TrendMethodologyResults):
        raise TypeError("results must be TrendMethodologyResults.")
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    methodology = {
        "schema_version": SPECIFICATION_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_sha256": source_sha256,
        "development_start": "2020-01-02",
        "development_end": "2025-12-31",
        "locked_period_accessed": False,
        "models": list(MODEL_IDS),
        "cost_stress_bps": list(COST_STRESS_BPS),
        "robustness_symbols": list(ROBUSTNESS_SYMBOLS),
        "robustness_frequencies": list(ROBUSTNESS_FREQUENCIES),
        "saved_timing": SAVED_TIMING,
        "final_timing": FINAL_TIMING,
        "positioning_comparison_is_selection": False,
        "negative_results_suppressed": False,
    }
    payloads: dict[str, bytes] = {
        TIMING_FILENAME: _csv_bytes(results.timing_comparison),
        WALK_FORWARD_FILENAME: _csv_bytes(results.walk_forward),
        ROBUSTNESS_FILENAME: _csv_bytes(results.robustness),
        SENSITIVITY_FILENAME: _csv_bytes(results.long_flat_sensitivity),
        PARITY_FILENAME: _csv_bytes(results.replay_parity),
        METHODOLOGY_FILENAME: _json_bytes(methodology),
        REPORT_FILENAME: render_methodology_report(results).encode("utf-8"),
    }
    manifest = {
        "schema_version": "day25_methodological_finalization_artifacts_v1",
        "artifact_order": list(APPROVED_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "row_counts": {
            TIMING_FILENAME: len(results.timing_comparison),
            WALK_FORWARD_FILENAME: len(results.walk_forward),
            ROBUSTNESS_FILENAME: len(results.robustness),
            SENSITIVITY_FILENAME: len(results.long_flat_sensitivity),
            PARITY_FILENAME: len(results.replay_parity),
        },
        "locked_period_accessed": False,
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    if tuple(payloads) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Methodological artifact allow-list changed.")
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
    if tuple(sorted(path.name for path in output.iterdir())) != tuple(sorted(APPROVED_ARTIFACT_NAMES)):
        raise RuntimeError("Final methodological artifact allow-list changed.")
    return tuple(output / name for name in APPROVED_ARTIFACT_NAMES)
