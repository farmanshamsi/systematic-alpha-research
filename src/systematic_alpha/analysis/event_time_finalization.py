"""Representative five-session time-bar versus dollar-bar experiment."""

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
from scipy import stats

from systematic_alpha.analysis.event_bar_diagnostics import (
    build_conservation_table,
    build_time_bars_from_trades,
    normalize_trade_sample,
)
from systematic_alpha.data.resampling import build_dollar_bars


SPECIFICATION_VERSION: Final[str] = "day25_event_time_finalization_v2"
SESSION_DATES: Final[tuple[str, ...]] = (
    "2025-01-15",
    "2025-04-15",
    "2025-07-15",
    "2025-10-15",
    "2025-12-15",
)
TARGET_BARS_PER_SESSION: Final[int] = 26
TIME_RULE: Final[str] = "15min"
SHORT_WINDOW: Final[int] = 4
LONG_WINDOW: Final[int] = 16
NEUTRAL_BAND: Final[float] = 0.001

THRESHOLD_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "source_trades",
    "source_volume",
    "source_dollar_value",
    "target_bars",
    "dollar_threshold",
)
COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "sampling_method",
    "sessions",
    "bars",
    "complete_bars",
    "partial_bars",
    "return_observations",
    "median_duration_seconds",
    "p95_duration_seconds",
    "duration_cv",
    "mean_trade_count",
    "trade_count_cv",
    "mean_volume",
    "volume_cv",
    "mean_dollar_value",
    "dollar_value_cv",
    "lag_one_return_autocorrelation",
    "return_skewness",
    "return_excess_kurtosis",
)
SESSION_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    *COMPARISON_COLUMNS[0:1],
    *COMPARISON_COLUMNS[2:],
)
INDICATOR_COLUMNS: Final[tuple[str, ...]] = (
    "sampling_method",
    "bars",
    "signal_available_observations",
    "long_signals",
    "short_signals",
    "neutral_signals",
    "pearson_signal_forward_return",
    "spearman_signal_forward_return",
    "mean_absolute_continuous_signal",
)
CONSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "sampling_method",
    "input_trade_count",
    "output_trade_count",
    "trade_count_error",
    "input_volume",
    "output_volume",
    "volume_error",
    "input_dollar_value",
    "output_dollar_value",
    "dollar_value_error",
)
BARS_COLUMNS: Final[tuple[str, ...]] = (
    "sampling_method",
    "session_date",
    "bar_sequence",
    "timestamp",
    "start_timestamp",
    "end_timestamp",
    "duration_seconds",
    "is_complete",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "dollar_value",
    "continuous_signal",
    "signal",
    "forward_log_return",
)

THRESHOLDS_FILENAME: Final[str] = "thresholds.csv"
COMPARISON_FILENAME: Final[str] = "sampling_comparison.csv"
SESSION_COMPARISON_FILENAME: Final[str] = "session_comparison.csv"
INDICATOR_FILENAME: Final[str] = "indicator_comparison.csv"
CONSERVATION_FILENAME: Final[str] = "conservation.csv"
BARS_FILENAME: Final[str] = "bars.csv"
METHODOLOGY_FILENAME: Final[str] = "methodology.json"
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"
APPROVED_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    THRESHOLDS_FILENAME,
    COMPARISON_FILENAME,
    SESSION_COMPARISON_FILENAME,
    INDICATOR_FILENAME,
    CONSERVATION_FILENAME,
    BARS_FILENAME,
    METHODOLOGY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)


class EventTimeFinalizationError(ValueError):
    """Raised when the representative event-time experiment is unsafe."""


@dataclass(frozen=True, slots=True)
class EventTimeFinalizationResults:
    """Compact evidence tables for the five-session experiment."""

    thresholds: pd.DataFrame
    sampling_comparison: pd.DataFrame
    session_comparison: pd.DataFrame
    indicator_comparison: pd.DataFrame
    conservation: pd.DataFrame
    bars: pd.DataFrame


def _coefficient_of_variation(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    mean = float(clean.mean())
    if math.isclose(mean, 0.0):
        return float("nan")
    return float(clean.std(ddof=1) / mean)


def validate_representative_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Require the exact predeclared sessions and genuine canonical trades."""

    source = trades.copy(deep=True)
    if "timestamp" not in source.columns:
        raise EventTimeFinalizationError("Event-time sample requires timestamps.")
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="raise")
    if "id" in source.columns:
        # IEX trade identifiers can repeat on different trading dates. Preserve
        # their provider value while making the multi-session sample key explicit.
        session_prefix = (
            source["timestamp"]
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
        )
        source["id"] = session_prefix + ":" + source["id"].astype("string")
    normalized = normalize_trade_sample(source)
    if normalized["symbol"].astype(str).unique().tolist() != ["SPY"]:
        raise EventTimeFinalizationError("Event-time sample must contain only SPY.")
    if not normalized["source"].astype(str).eq("alpaca").all():
        raise EventTimeFinalizationError("Event-time sample must come from Alpaca.")
    if not normalized["feed"].astype(str).eq("iex").all():
        raise EventTimeFinalizationError("Event-time sample must use the IEX feed.")
    normalized["session_date"] = (
        normalized["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    actual_dates = tuple(sorted(normalized["session_date"].unique()))
    if actual_dates != SESSION_DATES:
        raise EventTimeFinalizationError(
            f"Expected predeclared sessions {SESSION_DATES}; received {actual_dates}."
        )
    for session_date, group in normalized.groupby("session_date", observed=True, sort=True):
        local = group["timestamp"].dt.tz_convert("America/New_York")
        first = local.min()
        last = local.max()
        if len(group) < 100 or first.time() > pd.Timestamp("09:35").time() or last.time() < pd.Timestamp("15:55").time():
            raise EventTimeFinalizationError(
                f"Session {session_date} does not span a representative regular session."
            )
        if local.dt.time.lt(pd.Timestamp("09:30").time()).any() or local.dt.time.gt(pd.Timestamp("16:00").time()).any():
            raise EventTimeFinalizationError(f"Session {session_date} contains extended-hours trades.")
    return normalized


def _summarize_bars(frame: pd.DataFrame, *, method: str) -> dict[str, object]:
    returns = frame["log_return"].dropna()
    autocorrelation = float(returns.corr(returns.groupby(frame.loc[returns.index, "session_date"]).shift(1))) if len(returns) >= 3 else float("nan")
    if len(returns) >= 8 and float(returns.std(ddof=1)) > 0.0:
        skewness = float(stats.skew(returns.to_numpy(), bias=False))
        kurtosis = float(stats.kurtosis(returns.to_numpy(), fisher=True, bias=False))
    else:
        skewness = float("nan")
        kurtosis = float("nan")
    return {
        "sampling_method": method,
        "sessions": int(frame["session_date"].nunique()),
        "bars": int(len(frame)),
        "complete_bars": int(frame["is_complete"].sum()),
        "partial_bars": int((~frame["is_complete"]).sum()),
        "return_observations": int(len(returns)),
        "median_duration_seconds": float(frame["duration_seconds"].median()),
        "p95_duration_seconds": float(frame["duration_seconds"].quantile(0.95)),
        "duration_cv": _coefficient_of_variation(frame["duration_seconds"]),
        "mean_trade_count": float(frame["trade_count"].mean()),
        "trade_count_cv": _coefficient_of_variation(frame["trade_count"]),
        "mean_volume": float(frame["volume"].mean()),
        "volume_cv": _coefficient_of_variation(frame["volume"]),
        "mean_dollar_value": float(frame["dollar_value"].mean()),
        "dollar_value_cv": _coefficient_of_variation(frame["dollar_value"]),
        "lag_one_return_autocorrelation": autocorrelation,
        "return_skewness": skewness,
        "return_excess_kurtosis": kurtosis,
    }


def _attach_indicators(frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
    result = frame.sort_values(["session_date", "timestamp", "bar_sequence"], kind="stable").copy(deep=True).reset_index(drop=True)
    result["sampling_method"] = method
    result["duration_seconds"] = (
        pd.to_datetime(result["end_timestamp"], utc=True)
        - pd.to_datetime(result["start_timestamp"], utc=True)
    ).dt.total_seconds()
    result["log_return"] = (
        np.log(result["close"].astype("float64"))
        .groupby(result["session_date"], observed=True)
        .diff()
    )
    close = result["close"].astype("float64")
    short = close.groupby(result["session_date"], observed=True).transform(
        lambda values: values.rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    )
    long = close.groupby(result["session_date"], observed=True).transform(
        lambda values: values.rolling(LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    )
    result["continuous_signal"] = short.div(long).sub(1.0)
    result["signal"] = np.select(
        [
            result["continuous_signal"].gt(NEUTRAL_BAND),
            result["continuous_signal"].lt(-NEUTRAL_BAND),
        ],
        [1, -1],
        default=0,
    ).astype("int8")
    result["forward_log_return"] = result["log_return"].groupby(
        result["session_date"], observed=True
    ).shift(-1)
    return result


def _indicator_summary(frame: pd.DataFrame, *, method: str) -> dict[str, object]:
    available = frame["continuous_signal"].notna()
    paired = frame.loc[available & frame["forward_log_return"].notna(), ["continuous_signal", "forward_log_return"]]
    if len(paired) >= 3 and paired.nunique().min() > 1:
        pearson = float(paired["continuous_signal"].corr(paired["forward_log_return"], method="pearson"))
        spearman = float(paired["continuous_signal"].corr(paired["forward_log_return"], method="spearman"))
    else:
        pearson = float("nan")
        spearman = float("nan")
    signals = frame.loc[available, "signal"]
    return {
        "sampling_method": method,
        "bars": int(len(frame)),
        "signal_available_observations": int(available.sum()),
        "long_signals": int(signals.eq(1).sum()),
        "short_signals": int(signals.eq(-1).sum()),
        "neutral_signals": int(signals.eq(0).sum()),
        "pearson_signal_forward_return": pearson,
        "spearman_signal_forward_return": spearman,
        "mean_absolute_continuous_signal": float(frame.loc[available, "continuous_signal"].abs().mean()),
    }


def run_event_time_finalization(trades: pd.DataFrame) -> EventTimeFinalizationResults:
    """Run the exact five-session descriptive time/event-bar comparison."""

    normalized = validate_representative_trades(trades)
    thresholds: list[dict[str, object]] = []
    conservation_parts: list[pd.DataFrame] = []
    method_parts: dict[str, list[pd.DataFrame]] = {"time_15min": [], "dollar": []}
    for session_date in SESSION_DATES:
        session = normalized.loc[normalized["session_date"].eq(session_date)].copy(deep=True)
        notional = float((session["price"] * session["size"]).sum())
        threshold = notional / TARGET_BARS_PER_SESSION
        time_bars = build_time_bars_from_trades(session, rule=TIME_RULE)
        dollar_bars = build_dollar_bars(session, dollars_per_bar=threshold)
        if len(time_bars) != TARGET_BARS_PER_SESSION:
            raise EventTimeFinalizationError(
                f"Session {session_date} did not produce 26 non-empty 15-minute bars."
            )
        thresholds.append(
            {
                "session_date": session_date,
                "source_trades": len(session),
                "source_volume": float(session["size"].sum()),
                "source_dollar_value": notional,
                "target_bars": TARGET_BARS_PER_SESSION,
                "dollar_threshold": threshold,
            }
        )
        for method, bars in (("time_15min", time_bars), ("dollar", dollar_bars)):
            bars = bars.copy(deep=True)
            bars["session_date"] = session_date
            method_parts[method].append(bars)
        conservation = build_conservation_table(
            session,
            {"time_15min": time_bars, "dollar": dollar_bars},
        )
        conservation.insert(0, "session_date", session_date)
        conservation_parts.append(conservation)

    combined = {
        method: _attach_indicators(pd.concat(parts, ignore_index=True), method=method)
        for method, parts in method_parts.items()
    }
    overall = pd.DataFrame.from_records(
        [_summarize_bars(combined[method], method=method) for method in ("time_15min", "dollar")],
        columns=COMPARISON_COLUMNS,
    )
    session_rows: list[dict[str, object]] = []
    for session_date in SESSION_DATES:
        for method in ("time_15min", "dollar"):
            subset = combined[method].loc[combined[method]["session_date"].eq(session_date)]
            row = _summarize_bars(subset, method=method)
            row.pop("sessions")
            session_rows.append({"session_date": session_date, **row})
    session_comparison = pd.DataFrame.from_records(session_rows, columns=SESSION_COMPARISON_COLUMNS)
    indicator = pd.DataFrame.from_records(
        [_indicator_summary(combined[method], method=method) for method in ("time_15min", "dollar")],
        columns=INDICATOR_COLUMNS,
    )
    conservation = pd.concat(conservation_parts, ignore_index=True)
    conservation = conservation.rename(columns={"sampling_method": "sampling_method"}).loc[:, CONSERVATION_COLUMNS]
    tolerances = conservation["input_dollar_value"].abs().mul(1.0e-12).clip(lower=1.0e-8)
    if (
        conservation["trade_count_error"].ne(0).any()
        or conservation["volume_error"].abs().gt(1.0e-8).any()
        or conservation["dollar_value_error"].abs().gt(tolerances).any()
    ):
        raise RuntimeError("Representative event-bar conservation failed.")
    bars = pd.concat([combined["time_15min"], combined["dollar"]], ignore_index=True)
    bars = bars.loc[:, BARS_COLUMNS]
    results = EventTimeFinalizationResults(
        thresholds=pd.DataFrame.from_records(thresholds, columns=THRESHOLD_COLUMNS),
        sampling_comparison=overall,
        session_comparison=session_comparison,
        indicator_comparison=indicator,
        conservation=conservation,
        bars=bars,
    )
    if tuple(map(len, (results.thresholds, results.sampling_comparison, results.session_comparison, results.indicator_comparison, results.conservation))) != (5, 2, 10, 2, 10):
        raise RuntimeError("Representative event-time row counts changed.")
    return results


def _format(value: object, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100.0:+.2f}%" if percent else f"{number:.4f}"


def render_event_time_report(results: EventTimeFinalizationResults) -> str:
    lines = [
        "# Day 25 Representative Event-Time Experiment",
        "",
        "## Verdict",
        "",
        "Five complete, predeclared 2025 SPY IEX sessions replace the one-minute "
        "smoke sample for the descriptive event-time requirement. Dollar bars are "
        "compared with 15-minute bars without selecting a strategy or claiming "
        "profitability. All trades, shares, and dollar notional reconcile.",
        "",
        "## Sampling comparison",
        "",
        "| Method | Bars | Duration CV | Trade-count CV | Dollar-value CV | Lag-1 return correlation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.sampling_comparison.itertuples(index=False):
        lines.append(
            f"| {row.sampling_method} | {int(row.bars)} | {_format(row.duration_cv)} | "
            f"{_format(row.trade_count_cv)} | {_format(row.dollar_value_cv)} | "
            f"{_format(row.lag_one_return_autocorrelation)} |"
        )
    lines.extend(
        [
            "",
            "## Indicator comparison",
            "",
            "| Method | Available signals | Long | Short | Neutral | Pearson next-event association | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results.indicator_comparison.itertuples(index=False):
        lines.append(
            f"| {row.sampling_method} | {int(row.signal_available_observations)} | "
            f"{int(row.long_signals)} | {int(row.short_signals)} | {int(row.neutral_signals)} | "
            f"{_format(row.pearson_signal_forward_return)} | "
            f"{_format(row.spearman_signal_forward_return)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The comparison is descriptive and covers five deliberately spaced "
            "sessions, not the six-year strategy sample. Dollar thresholds are "
            "calibrated independently inside each session only to match bar counts. "
            "The one-event-ahead horizon differs in clock time between sampling "
            "methods. No result changes the primary 15-minute research frequency.",
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


def write_event_time_artifacts(
    results: EventTimeFinalizationResults,
    directory: str | Path,
    *,
    source_dataset_id: str,
    source_sha256: str,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact deterministic nine-file event-time evidence bundle."""

    if not isinstance(results, EventTimeFinalizationResults):
        raise TypeError("results must be EventTimeFinalizationResults.")
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {output}")
    methodology = {
        "schema_version": SPECIFICATION_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_sha256": source_sha256,
        "provider": "alpaca",
        "feed": "iex",
        "symbol": "SPY",
        "session_dates": list(SESSION_DATES),
        "target_bars_per_session": TARGET_BARS_PER_SESSION,
        "time_rule": TIME_RULE,
        "indicator": {
            "short_window": SHORT_WINDOW,
            "long_window": LONG_WINDOW,
            "neutral_band": NEUTRAL_BAND,
        },
        "trade_identifier_scope": "session_date_plus_provider_trade_id",
        "locked_period_accessed": False,
        "profitability_test": False,
        "primary_sampling_method_changed": False,
    }
    payloads: dict[str, bytes] = {
        THRESHOLDS_FILENAME: _csv_bytes(results.thresholds),
        COMPARISON_FILENAME: _csv_bytes(results.sampling_comparison),
        SESSION_COMPARISON_FILENAME: _csv_bytes(results.session_comparison),
        INDICATOR_FILENAME: _csv_bytes(results.indicator_comparison),
        CONSERVATION_FILENAME: _csv_bytes(results.conservation),
        BARS_FILENAME: _csv_bytes(results.bars),
        METHODOLOGY_FILENAME: _json_bytes(methodology),
        REPORT_FILENAME: render_event_time_report(results).encode("utf-8"),
    }
    manifest = {
        "schema_version": "day25_event_time_finalization_artifacts_v2",
        "artifact_order": list(APPROVED_ARTIFACT_NAMES),
        "hash_algorithm": "sha256",
        "hashes": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "row_counts": {
            THRESHOLDS_FILENAME: len(results.thresholds),
            COMPARISON_FILENAME: len(results.sampling_comparison),
            SESSION_COMPARISON_FILENAME: len(results.session_comparison),
            INDICATOR_FILENAME: len(results.indicator_comparison),
            CONSERVATION_FILENAME: len(results.conservation),
            BARS_FILENAME: len(results.bars),
        },
        "locked_period_accessed": False,
    }
    payloads[MANIFEST_FILENAME] = _json_bytes(manifest)
    if tuple(payloads) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Event-time artifact allow-list changed.")
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
    return tuple(output / name for name in APPROVED_ARTIFACT_NAMES)
