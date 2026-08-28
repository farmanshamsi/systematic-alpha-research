"""Engineering audit of time, tick, volume and dollar bars."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from systematic_alpha.data.resampling import (
    build_dollar_bars,
    build_tick_bars,
    build_volume_bars,
)


REQUIRED_TRADE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "timestamp",
        "symbol",
        "price",
        "size",
        "source",
        "feed",
    }
)


class EventBarDiagnosticError(ValueError):
    """Raised when an event-bar audit cannot be completed safely."""


@dataclass(frozen=True)
class EventBarDiagnosticBundle:
    """Outputs from the event-bar feasibility audit."""

    time_bars: pd.DataFrame
    tick_bars: pd.DataFrame
    volume_bars: pd.DataFrame
    dollar_bars: pd.DataFrame
    thresholds: pd.DataFrame
    comparison: pd.DataFrame
    conservation: pd.DataFrame
    decision: pd.DataFrame


def normalize_trade_sample(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize a canonical trade sample."""

    missing = REQUIRED_TRADE_COLUMNS.difference(
        frame.columns
    )

    if missing:
        raise EventBarDiagnosticError(
            "Trade sample is missing required columns: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise EventBarDiagnosticError(
            "Trade sample cannot be empty."
        )

    trades = frame.copy()

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
        errors="raise",
    )

    trades["symbol"] = (
        trades["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    trades["price"] = pd.to_numeric(
        trades["price"],
        errors="raise",
    )

    trades["size"] = pd.to_numeric(
        trades["size"],
        errors="raise",
    )

    if trades["symbol"].isna().any():
        raise EventBarDiagnosticError(
            "Trade symbols cannot be missing."
        )

    if (trades["symbol"] == "").any():
        raise EventBarDiagnosticError(
            "Trade symbols cannot be empty."
        )

    if (trades["price"] <= 0.0).any():
        raise EventBarDiagnosticError(
            "Trade prices must be positive."
        )

    if (trades["size"] <= 0.0).any():
        raise EventBarDiagnosticError(
            "Trade sizes must be positive."
        )

    if trades[["source", "feed"]].isna().any().any():
        raise EventBarDiagnosticError(
            "Trade source and feed cannot be missing."
        )

    if trades["source"].nunique(dropna=False) != 1:
        raise EventBarDiagnosticError(
            "The audit requires one data source."
        )

    if trades["feed"].nunique(dropna=False) != 1:
        raise EventBarDiagnosticError(
            "The audit requires one data feed."
        )

    if "id" in trades.columns:
        identified = trades.loc[
            trades["id"].notna(),
            ["symbol", "id"],
        ].copy()

        identified["id"] = identified["id"].astype(
            "string"
        )

        if identified.duplicated(
            ["symbol", "id"]
        ).any():
            raise EventBarDiagnosticError(
                "Duplicate trade identifiers detected."
            )

    trades["_source_order"] = np.arange(
        len(trades),
        dtype=np.int64,
    )

    return trades.sort_values(
        ["symbol", "timestamp", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)


def calculate_event_thresholds(
    trades: pd.DataFrame,
    *,
    target_bar_count: int = 10,
) -> dict[str, float]:
    """Calculate matched-count engineering thresholds."""

    normalized = normalize_trade_sample(trades)

    if (
        isinstance(target_bar_count, bool)
        or not isinstance(target_bar_count, int)
        or target_bar_count <= 0
    ):
        raise EventBarDiagnosticError(
            "target_bar_count must be a positive integer."
        )

    total_trades = len(normalized)

    total_volume = float(
        normalized["size"].sum()
    )

    total_notional = float(
        (
            normalized["price"]
            * normalized["size"]
        ).sum()
    )

    return {
        "tick": float(
            max(
                1,
                math.ceil(
                    total_trades
                    / target_bar_count
                ),
            )
        ),
        "volume": max(
            1.0,
            total_volume / target_bar_count,
        ),
        "dollar": max(
            1.0,
            total_notional / target_bar_count,
        ),
    }


def infer_time_rule(
    trades: pd.DataFrame,
    *,
    target_bar_count: int = 10,
) -> str:
    """Choose an approximately matched time interval."""

    normalized = normalize_trade_sample(trades)

    duration_seconds = float(
        (
            normalized["timestamp"].max()
            - normalized["timestamp"].min()
        ).total_seconds()
    )

    interval_seconds = max(
        1,
        math.ceil(
            duration_seconds / target_bar_count
        ),
    )

    return f"{interval_seconds}s"


def build_time_bars_from_trades(
    trades: pd.DataFrame,
    *,
    rule: str,
) -> pd.DataFrame:
    """Aggregate canonical trades into non-empty time bars."""

    frame = normalize_trade_sample(trades)

    try:
        frame["time_bucket"] = (
            frame["timestamp"].dt.floor(rule)
        )
    except ValueError as exc:
        raise EventBarDiagnosticError(
            f"Invalid time-bar rule: {rule}"
        ) from exc

    frame["trade_notional"] = (
        frame["price"] * frame["size"]
    )

    rows: list[dict[str, object]] = []

    for (
        symbol,
        time_bucket,
    ), group in frame.groupby(
        ["symbol", "time_bucket"],
        observed=True,
        sort=True,
    ):
        group = group.sort_values(
            ["timestamp", "_source_order"],
            kind="stable",
        )

        volume = float(
            group["size"].sum()
        )

        dollar_value = float(
            group["trade_notional"].sum()
        )

        rows.append(
            {
                "timestamp": group[
                    "timestamp"
                ].iloc[-1],
                "symbol": str(symbol),
                "bar_type": f"time_{rule}",
                "threshold": np.nan,
                "open": float(
                    group["price"].iloc[0]
                ),
                "high": float(
                    group["price"].max()
                ),
                "low": float(
                    group["price"].min()
                ),
                "close": float(
                    group["price"].iloc[-1]
                ),
                "volume": volume,
                "trade_count": int(
                    len(group)
                ),
                "vwap": (
                    dollar_value / volume
                ),
                "dollar_value": dollar_value,
                "start_timestamp": group[
                    "timestamp"
                ].iloc[0],
                "end_timestamp": group[
                    "timestamp"
                ].iloc[-1],
                "is_complete": True,
                "source": str(
                    group["source"].iloc[0]
                ),
                "feed": str(
                    group["feed"].iloc[0]
                ),
                "time_bucket": time_bucket,
            }
        )

    if not rows:
        raise EventBarDiagnosticError(
            "Time-bar construction produced no bars."
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["symbol", "time_bucket"],
        kind="stable",
    ).reset_index(drop=True)

    result["bar_sequence"] = (
        result.groupby(
            "symbol",
            observed=True,
            sort=False,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    return result[
        [
            "timestamp",
            "symbol",
            "bar_sequence",
            "bar_type",
            "threshold",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "dollar_value",
            "start_timestamp",
            "end_timestamp",
            "is_complete",
            "source",
            "feed",
        ]
    ]


def _coefficient_of_variation(
    values: pd.Series,
) -> float:
    """Calculate standard deviation divided by mean."""

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(clean) < 2:
        return np.nan

    mean = float(clean.mean())

    if np.isclose(mean, 0.0):
        return np.nan

    return float(
        clean.std(ddof=1) / mean
    )


def _bar_comparison_record(
    bars: pd.DataFrame,
    *,
    sampling_method: str,
) -> dict[str, object]:
    """Summarize one sampling method."""

    frame = bars.copy()

    frame["start_timestamp"] = pd.to_datetime(
        frame["start_timestamp"],
        utc=True,
        errors="raise",
    )

    frame["end_timestamp"] = pd.to_datetime(
        frame["end_timestamp"],
        utc=True,
        errors="raise",
    )

    frame["duration_seconds"] = (
        frame["end_timestamp"]
        - frame["start_timestamp"]
    ).dt.total_seconds()

    complete = frame.loc[
        frame["is_complete"]
    ].copy()

    complete = complete.sort_values(
        ["symbol", "bar_sequence"],
        kind="stable",
    )

    complete["log_return"] = (
        complete.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["close"]
        .transform(np.log)
        .groupby(
            complete["symbol"],
            observed=True,
            sort=False,
        )
        .diff()
    )

    returns = complete[
        "log_return"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if (
        len(returns) >= 3
        and returns.std(ddof=1) > 0.0
    ):
        lag_one = float(
            returns.corr(
                returns.shift(1)
            )
        )
    else:
        lag_one = np.nan

    if (
        len(returns) >= 8
        and returns.std(ddof=1) > 0.0
    ):
        skewness = float(
            stats.skew(
                returns.to_numpy(),
                bias=False,
            )
        )

        excess_kurtosis = float(
            stats.kurtosis(
                returns.to_numpy(),
                fisher=True,
                bias=False,
            )
        )

        jarque_bera = stats.jarque_bera(
            returns.to_numpy()
        )

        jarque_bera_pvalue = float(
            jarque_bera.pvalue
        )
    else:
        skewness = np.nan
        excess_kurtosis = np.nan
        jarque_bera_pvalue = np.nan

    return {
        "sampling_method": sampling_method,
        "bars": int(len(frame)),
        "complete_bars": int(
            frame["is_complete"].sum()
        ),
        "partial_bars": int(
            (~frame["is_complete"]).sum()
        ),
        "return_observations": int(
            len(returns)
        ),
        "median_duration_seconds": float(
            frame["duration_seconds"].median()
        ),
        "p95_duration_seconds": float(
            frame["duration_seconds"].quantile(
                0.95
            )
        ),
        "mean_trade_count": float(
            frame["trade_count"].mean()
        ),
        "trade_count_cv": (
            _coefficient_of_variation(
                frame["trade_count"]
            )
        ),
        "mean_volume": float(
            frame["volume"].mean()
        ),
        "volume_cv": (
            _coefficient_of_variation(
                frame["volume"]
            )
        ),
        "mean_dollar_value": float(
            frame["dollar_value"].mean()
        ),
        "dollar_value_cv": (
            _coefficient_of_variation(
                frame["dollar_value"]
            )
        ),
        "lag_one_return_autocorrelation": (
            lag_one
        ),
        "return_skewness": skewness,
        "return_excess_kurtosis": (
            excess_kurtosis
        ),
        "jarque_bera_pvalue": (
            jarque_bera_pvalue
        ),
    }


def build_sampling_comparison(
    bar_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare the mechanics of each bar type."""

    records = [
        _bar_comparison_record(
            frame,
            sampling_method=name,
        )
        for name, frame in bar_frames.items()
    ]

    return pd.DataFrame(records)


def build_conservation_table(
    trades: pd.DataFrame,
    bar_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Verify preservation of trades, shares and notional."""

    source = normalize_trade_sample(trades)

    input_trade_count = int(
        len(source)
    )

    input_volume = float(
        source["size"].sum()
    )

    input_dollar_value = float(
        (
            source["price"]
            * source["size"]
        ).sum()
    )

    records = []

    for method, bars in bar_frames.items():
        output_trade_count = int(
            bars["trade_count"].sum()
        )

        output_volume = float(
            bars["volume"].sum()
        )

        output_dollar_value = float(
            bars["dollar_value"].sum()
        )

        records.append(
            {
                "sampling_method": method,
                "input_trade_count": (
                    input_trade_count
                ),
                "output_trade_count": (
                    output_trade_count
                ),
                "trade_count_error": (
                    output_trade_count
                    - input_trade_count
                ),
                "input_volume": input_volume,
                "output_volume": output_volume,
                "volume_error": (
                    output_volume
                    - input_volume
                ),
                "input_dollar_value": (
                    input_dollar_value
                ),
                "output_dollar_value": (
                    output_dollar_value
                ),
                "dollar_value_error": (
                    output_dollar_value
                    - input_dollar_value
                ),
            }
        )

    return pd.DataFrame(records)


def build_sampling_decision() -> pd.DataFrame:
    """Record the controlled Day 5 sampling decision."""

    return pd.DataFrame(
        [
            {
                "sampling_method": (
                    "15-minute time bars"
                ),
                "project_role": (
                    "primary research frequency"
                ),
                "decision": "retain",
                "reason": (
                    "Complete six-year SIP development "
                    "coverage, exact exchange-calendar "
                    "validation and sufficient observations."
                ),
            },
            {
                "sampling_method": "dollar bars",
                "project_role": (
                    "future event-bar experiment candidate"
                ),
                "decision": (
                    "implementation verified; "
                    "statistical acceptance deferred"
                ),
                "reason": (
                    "Dollar bars normalize economic "
                    "activity, but the available trade "
                    "sample spans only one minute."
                ),
            },
            {
                "sampling_method": "tick bars",
                "project_role": (
                    "engineering robustness"
                ),
                "decision": (
                    "implementation verified only"
                ),
                "reason": (
                    "Trade-count conservation is verified, "
                    "but no representative multi-session "
                    "sample is currently available."
                ),
            },
            {
                "sampling_method": "volume bars",
                "project_role": (
                    "engineering robustness"
                ),
                "decision": (
                    "implementation verified only"
                ),
                "reason": (
                    "Volume conservation is verified, "
                    "but the one-minute sample cannot "
                    "support distributional conclusions."
                ),
            },
        ]
    )


def build_event_bar_diagnostics(
    trades: pd.DataFrame,
    *,
    target_bar_count: int = 10,
) -> EventBarDiagnosticBundle:
    """Run the complete D05-T03 engineering audit."""

    normalized = normalize_trade_sample(
        trades
    )

    thresholds = calculate_event_thresholds(
        normalized,
        target_bar_count=target_bar_count,
    )

    time_rule = infer_time_rule(
        normalized,
        target_bar_count=target_bar_count,
    )

    time_bars = build_time_bars_from_trades(
        normalized,
        rule=time_rule,
    )

    tick_bars = build_tick_bars(
        normalized,
        trades_per_bar=int(
            thresholds["tick"]
        ),
    )

    volume_bars = build_volume_bars(
        normalized,
        shares_per_bar=thresholds[
            "volume"
        ],
    )

    dollar_bars = build_dollar_bars(
        normalized,
        dollars_per_bar=thresholds[
            "dollar"
        ],
    )

    bar_frames = {
        f"time_{time_rule}": time_bars,
        "tick": tick_bars,
        "volume": volume_bars,
        "dollar": dollar_bars,
    }

    threshold_table = pd.DataFrame(
        [
            {
                "bar_type": "time",
                "threshold": time_rule,
                "target_bar_count": (
                    target_bar_count
                ),
                "threshold_method": (
                    "sample duration divided by "
                    "target bar count"
                ),
            },
            {
                "bar_type": "tick",
                "threshold": thresholds["tick"],
                "target_bar_count": (
                    target_bar_count
                ),
                "threshold_method": (
                    "sample trades divided by "
                    "target bar count"
                ),
            },
            {
                "bar_type": "volume",
                "threshold": thresholds["volume"],
                "target_bar_count": (
                    target_bar_count
                ),
                "threshold_method": (
                    "sample shares divided by "
                    "target bar count"
                ),
            },
            {
                "bar_type": "dollar",
                "threshold": thresholds["dollar"],
                "target_bar_count": (
                    target_bar_count
                ),
                "threshold_method": (
                    "sample notional divided by "
                    "target bar count"
                ),
            },
        ]
    )

    return EventBarDiagnosticBundle(
        time_bars=time_bars,
        tick_bars=tick_bars,
        volume_bars=volume_bars,
        dollar_bars=dollar_bars,
        thresholds=threshold_table,
        comparison=build_sampling_comparison(
            bar_frames
        ),
        conservation=build_conservation_table(
            normalized,
            bar_frames,
        ),
        decision=build_sampling_decision(),
    )
