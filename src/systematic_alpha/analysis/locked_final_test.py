"""One-time locked 2026 evaluation for the three frozen CQF models."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
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
from systematic_alpha.analysis.reversion_inference import CONFIGURATIONS
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics
from systematic_alpha.analysis.trend_methodology_finalization import (
    FINAL_TIMING,
    apply_next_open_overnight_flat,
    build_model_observations,
)
from systematic_alpha.analysis.trend_family_walk_forward import REQUIRED_INPUT_SYMBOLS
from systematic_alpha.data.session_aggregation import aggregate_session_bars
from systematic_alpha.strategies.ou_vwap_reversion import build_ou_vwap_reversion_strategy


SPECIFICATION_VERSION: Final[str] = "day25_locked_final_test_v1"
AUTHORIZATION_CODE: Final[str] = "ACKNOWLEDGE_FROZEN_ONE_TIME_2026_FINAL_TEST"
LOCKED_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-02", tz="UTC")
LOCKED_END_EXCLUSIVE: Final[pd.Timestamp] = pd.Timestamp("2026-07-01", tz="UTC")
DEVELOPMENT_DATASET_SHA256: Final[str] = (
    "30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2"
)
COST_BPS_PER_TURNOVER: Final[float] = 1.0
FROZEN_MODELS: Final[tuple[str, ...]] = (
    "price_ratio_long_short_neutral",
    "ema_macd_long_short_neutral",
    "ou_vwap_slow_equal_weight",
)
FROZEN_CODE_PATHS: Final[tuple[str, ...]] = (
    "config/base.yaml",
    "src/systematic_alpha/analysis/locked_final_test.py",
    "src/systematic_alpha/analysis/trend_methodology_finalization.py",
    "src/systematic_alpha/analysis/strategy_performance.py",
    "src/systematic_alpha/strategies/trend_ratio.py",
    "src/systematic_alpha/strategies/ema_macd.py",
    "src/systematic_alpha/strategies/ou_vwap_reversion.py",
)

PERFORMANCE_FILENAME: Final[str] = "performance.csv"
SESSION_RETURNS_FILENAME: Final[str] = "session_returns.csv"
LOCKED_BARS_FILENAME: Final[str] = "locked_bars.parquet"
METHODOLOGY_FILENAME: Final[str] = "methodology.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    PERFORMANCE_FILENAME,
    SESSION_RETURNS_FILENAME,
    LOCKED_BARS_FILENAME,
    METHODOLOGY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)

PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "evaluation_series",
    "test_start",
    "test_end",
    "sessions",
    "observations",
    "annualization_factor",
    "cost_bps_per_turnover",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "initial_position",
    "initial_turnover",
    "overnight_position_violations",
)


class LockedFinalTestError(ValueError):
    """Raised when the one-time final test would violate its frozen protocol."""


@dataclass(frozen=True, slots=True)
class LockedFinalTestResults:
    """Frozen one-time performance and daily-return evidence."""

    performance: pd.DataFrame
    session_returns: pd.DataFrame


def require_authorization(value: str | None) -> None:
    """Require the exact acknowledgement code before any locked-data access."""

    if value != AUTHORIZATION_CODE:
        raise LockedFinalTestError(
            "Exact one-time locked-final-test authorization is required."
        )


def verify_frozen_development_state(project_root: str | Path) -> dict[str, str]:
    """Verify the development data and frozen model artifacts before access."""

    root = Path(project_root)
    dataset = root / (
        "data/processed/bars/"
        "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
        "sip_v3_development_canonical.parquet"
    )
    observed_dataset = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if observed_dataset != DEVELOPMENT_DATASET_SHA256:
        raise LockedFinalTestError("Frozen development dataset hash changed.")
    verified: dict[str, str] = {"development_dataset": observed_dataset}
    for label, relative in (
        ("trend_finalization", "artifacts/day25_methodological_finalization"),
        ("reversion", "artifacts/day17"),
    ):
        directory = root / relative
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        hashes = manifest.get("hashes") or manifest.get("artifact_sha256")
        if hashes is None and isinstance(manifest.get("artifacts"), list):
            hashes = {
                item["filename"]: item["sha256"] for item in manifest["artifacts"]
            }
        if not isinstance(hashes, dict) or not hashes:
            raise LockedFinalTestError(f"{label} manifest has no artifact hashes.")
        for filename, expected in hashes.items():
            observed = hashlib.sha256((directory / str(filename)).read_bytes()).hexdigest()
            if observed != str(expected):
                raise LockedFinalTestError(f"Frozen {label} artifact hash changed.")
        verified[label] = hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest()
    inventory_path = root / "artifacts/day25/code_inventory.csv"
    inventory_rows = {
        row["repository_path"]: row["sha256"]
        for row in csv.DictReader(inventory_path.open(newline="", encoding="utf-8"))
    }
    for relative in FROZEN_CODE_PATHS:
        expected = inventory_rows.get(relative)
        if expected is None:
            raise LockedFinalTestError(
                f"Day 25 staging inventory is missing frozen source: {relative}."
            )
        observed = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise LockedFinalTestError(f"Frozen source changed after Day 25 staging: {relative}.")
    verified["day25_staging_manifest"] = hashlib.sha256(
        (root / "artifacts/day25/manifest.json").read_bytes()
    ).hexdigest()
    return verified


def validate_locked_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Require exact symbols and prohibit any row outside the locked interval."""

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise LockedFinalTestError("Locked bars must be a non-empty DataFrame.")
    required = {
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "trade_count", "vwap", "source", "feed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise LockedFinalTestError(f"Locked bars are missing columns: {missing}.")
    result = bars.copy(deep=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    if tuple(sorted(result["symbol"].dropna().unique())) != tuple(
        sorted(REQUIRED_INPUT_SYMBOLS)
    ):
        raise LockedFinalTestError("Locked bars must contain exactly SPY, QQQ, and IWM.")
    outside = result["timestamp"].lt(LOCKED_START) | result["timestamp"].ge(
        LOCKED_END_EXCLUSIVE
    )
    if outside.any():
        raise LockedFinalTestError("A locked-data row lies outside the exact final interval.")
    if result.duplicated(["symbol", "timestamp"]).any():
        raise LockedFinalTestError("Locked bars contain duplicate symbol timestamps.")
    return result.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def _compound(values: pd.Series) -> float:
    numeric = values.to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or np.less_equal(numeric, -1.0).any():
        raise LockedFinalTestError("Final-test returns must be finite and greater than -1.")
    return float(np.prod(1.0 + numeric) - 1.0)


def _performance(
    model_id: str,
    returns: pd.Series,
    *,
    session_dates: pd.Series,
    turnover: float,
    positions: pd.Series,
    initial_turnover: float,
    overnight_violations: int,
    evaluation_series: str,
    annualization_factor: float,
) -> dict[str, object]:
    metrics = calculate_performance_metrics(
        returns.reset_index(drop=True), annualization_factor=annualization_factor
    )
    eligible = positions.astype("float64")
    return {
        "model_id": model_id,
        "evaluation_series": evaluation_series,
        "test_start": pd.Timestamp(session_dates.min()).strftime("%Y-%m-%d"),
        "test_end": pd.Timestamp(session_dates.max()).strftime("%Y-%m-%d"),
        "sessions": int(session_dates.nunique()),
        "observations": int(len(returns)),
        "annualization_factor": float(annualization_factor),
        "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": metrics.max_drawdown,
        "turnover": float(turnover),
        "long_exposure": float(100.0 * eligible.eq(1.0).mean()),
        "short_exposure": float(100.0 * eligible.eq(-1.0).mean()),
        "flat_exposure": float(100.0 * eligible.eq(0.0).mean()),
        "initial_position": int(eligible.iloc[0]),
        "initial_turnover": float(initial_turnover),
        "overnight_position_violations": int(overnight_violations),
    }


def evaluate_locked_final_test(
    development_bars: pd.DataFrame,
    locked_bars: pd.DataFrame,
) -> LockedFinalTestResults:
    """Evaluate every frozen model once, using development history only as warmup."""

    locked = validate_locked_bars(locked_bars)
    development = aggregate_session_bars(development_bars, "15min")
    combined = pd.concat([development, locked], ignore_index=True)
    features = build_return_features(
        combined,
        expected_symbols=REQUIRED_INPUT_SYMBOLS,
    ).bars
    locked_mask = features["timestamp"].ge(LOCKED_START) & features["timestamp"].lt(
        LOCKED_END_EXCLUSIVE
    )
    if int(locked_mask.sum()) != len(locked):
        raise LockedFinalTestError("Feature construction changed the locked row count.")

    performance_rows: list[dict[str, object]] = []
    session_series: dict[str, pd.Series] = {}
    spy = features.loc[features["symbol"].eq("SPY")].reset_index(drop=True)
    for model_id in FROZEN_MODELS[:2]:
        observations = build_model_observations(spy, model_id=model_id)
        test_source = observations.loc[
            observations["timestamp"].ge(LOCKED_START)
            & observations["timestamp"].lt(LOCKED_END_EXCLUSIVE)
        ].reset_index(drop=True)
        timed = apply_next_open_overnight_flat(
            test_source,
            cost_bps_per_turnover=COST_BPS_PER_TURNOVER,
        )
        dates = pd.to_datetime(timed["session_date"], utc=True).dt.normalize()
        sessions = timed.assign(_session=dates).groupby("_session", sort=True)[
            "net_strategy_return"
        ].agg(_compound)
        session_series[model_id] = sessions
        factor = 252.0 * len(timed) / dates.nunique()
        performance_rows.append(
            _performance(
                model_id,
                timed["net_strategy_return"],
                session_dates=dates,
                turnover=float(timed["turnover"].sum()),
                positions=timed["position"],
                initial_turnover=float(timed.iloc[0]["turnover"]),
                overnight_violations=int(
                    timed.loc[timed["is_session_close"], "ending_position"].ne(0).sum()
                ),
                evaluation_series="intraday_bar_net_return",
                annualization_factor=factor,
            )
        )

    slow = next(
        configuration for configuration in CONFIGURATIONS
        if configuration.configuration_id == "ou_vwap_slow"
    )
    reset_timestamps = tuple(
        pd.Timestamp(value)
        for value in features.loc[locked_mask].groupby("symbol", observed=True)[
            "timestamp"
        ].min()
    )
    ou = build_ou_vwap_reversion_strategy(
        features,
        parameters=slow,
        execution_reset_timestamps=reset_timestamps,
    ).observations
    ou_test = ou.loc[
        ou["timestamp"].ge(LOCKED_START) & ou["timestamp"].lt(LOCKED_END_EXCLUSIVE)
    ].reset_index(drop=True)
    ou_daily = (
        ou_test.assign(
            _session=pd.to_datetime(ou_test["session_date"], utc=True).dt.normalize()
        )
        .groupby(["_session", "symbol"], observed=True, sort=True)[
            "net_strategy_return"
        ]
        .agg(_compound)
        .unstack("symbol")
        .reindex(columns=REQUIRED_INPUT_SYMBOLS)
    )
    if ou_daily.isna().any().any():
        raise LockedFinalTestError("OU locked-session returns are incomplete.")
    ou_equal = ou_daily.mean(axis=1)
    session_series[FROZEN_MODELS[2]] = ou_equal
    ou_dates = pd.Series(ou_equal.index, index=ou_equal.index)
    ou_positions = ou_test["position"]
    session_open = ou_test.groupby("symbol", observed=True).cumcount().eq(0) | ou_test[
        "session_date"
    ].ne(ou_test.groupby("symbol", observed=True)["session_date"].shift(1))
    performance_rows.append(
        _performance(
            FROZEN_MODELS[2],
            ou_equal,
            session_dates=ou_dates,
            turnover=float(ou_test.groupby("symbol", observed=True)["turnover"].sum().mean()),
            positions=ou_positions,
            initial_turnover=float(
                ou_test.groupby("symbol", observed=True, sort=True).head(1)["turnover"].sum()
            ),
            overnight_violations=int(ou_test.loc[session_open, "position"].ne(0).sum()),
            evaluation_series="equal_weight_session_return",
            annualization_factor=252.0,
        )
    )
    performance = pd.DataFrame.from_records(performance_rows, columns=PERFORMANCE_COLUMNS)
    if tuple(performance["model_id"]) != FROZEN_MODELS:
        raise RuntimeError("Final-test model order changed.")
    panel = pd.concat(session_series, axis=1).reset_index(names="session_date")
    if panel.isna().any().any() or len(panel) != locked["timestamp"].dt.tz_convert(
        "America/New_York"
    ).dt.normalize().nunique():
        raise LockedFinalTestError("Final-test session panel is incomplete.")
    return LockedFinalTestResults(performance=performance, session_returns=panel)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g",
        date_format="%Y-%m-%d", na_rep="",
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False).rstrip()
        + "\n"
    ).encode("utf-8")


def _report(results: LockedFinalTestResults) -> str:
    lines = [
        "# One-Time Locked 2026 Final Test",
        "",
        "All three frozen results are reported once. No model, parameter, market, "
        "cost, allocation, or exclusion was changed after data access.",
        "",
        "| Model | Cumulative return | Sharpe | Maximum drawdown |",
        "|---|---:|---:|---:|",
    ]
    for row in results.performance.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.cumulative_return * 100.0:+.2f}% | "
            f"{row.sharpe_ratio:.3f} | {row.maximum_drawdown * 100.0:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "These are final-test observations, not a new tuning sample. Every "
            "result remains in the report whether positive or negative.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_locked_final_test_artifacts(
    results: LockedFinalTestResults,
    locked_bars: pd.DataFrame,
    output_directory: str | Path,
    *,
    frozen_hashes: Mapping[str, str],
    request_metadata: Mapping[str, object],
) -> tuple[Path, ...]:
    """Atomically write one immutable final-test bundle and refuse replacement."""

    directory = Path(output_directory)
    if directory.name != "day25_final_test":
        raise LockedFinalTestError("Output directory must be named day25_final_test.")
    if directory.exists():
        raise FileExistsError("The one-time final-test bundle already exists.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".day25-final-test-", dir=directory.parent))
    try:
        performance_bytes = _csv_bytes(results.performance)
        session_bytes = _csv_bytes(results.session_returns)
        (stage / PERFORMANCE_FILENAME).write_bytes(performance_bytes)
        (stage / SESSION_RETURNS_FILENAME).write_bytes(session_bytes)
        locked_bars.to_parquet(stage / LOCKED_BARS_FILENAME, index=False)
        methodology = {
            "schema_version": SPECIFICATION_VERSION,
            "authorization_code_validated": True,
            "one_time_run": True,
            "locked_start": LOCKED_START.isoformat(),
            "locked_end_exclusive": LOCKED_END_EXCLUSIVE.isoformat(),
            "models": list(FROZEN_MODELS),
            "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
            "trend_timing": FINAL_TIMING,
            "reversion_configuration": "ou_vwap_slow",
            "development_history_used_for_warmup_only": True,
            "execution_state_reset_at_test_start": True,
            "ranking_or_retuning_performed": False,
            "all_results_reported": True,
            "frozen_source_hashes": dict(frozen_hashes),
            "request_metadata": dict(request_metadata),
        }
        methodology_bytes = _json_bytes(methodology)
        report_bytes = _report(results).encode("utf-8")
        (stage / METHODOLOGY_FILENAME).write_bytes(methodology_bytes)
        (stage / REPORT_FILENAME).write_bytes(report_bytes)
        hashes = {
            name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
            for name in APPROVED_ARTIFACT_NAMES
            if name != MANIFEST_FILENAME
        }
        manifest = {
            "schema_version": SPECIFICATION_VERSION,
            "artifact_order": list(APPROVED_ARTIFACT_NAMES),
            "hashes": hashes,
            "row_counts": {
                PERFORMANCE_FILENAME: len(results.performance),
                SESSION_RETURNS_FILENAME: len(results.session_returns),
                LOCKED_BARS_FILENAME: len(locked_bars),
            },
            "one_time_run_complete": True,
            "all_results_reported": True,
            "ranking_or_retuning_performed": False,
        }
        (stage / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        if {path.name for path in stage.iterdir()} != set(APPROVED_ARTIFACT_NAMES):
            raise RuntimeError("Final-test artifact allow-list changed.")
        os.replace(stage, directory)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return tuple(directory / name for name in APPROVED_ARTIFACT_NAMES)
