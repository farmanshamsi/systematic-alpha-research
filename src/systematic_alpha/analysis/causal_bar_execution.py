"""Model-independent causal bar-return and turnover accounting."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "timestamp",
        "symbol",
        "session_date",
        "open",
        "close",
        "signal",
        "signal_available",
    }
)


class CausalBarExecutionError(ValueError):
    """Raised when causal bar accounting cannot be applied safely."""


def _validate_input(observations: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the model-independent timing inputs."""

    if not isinstance(observations, pd.DataFrame):
        raise CausalBarExecutionError("Timing input must be a pandas DataFrame.")
    if observations.empty:
        raise CausalBarExecutionError("Timing input cannot be empty.")

    missing = sorted(REQUIRED_COLUMNS.difference(observations.columns))
    if missing:
        raise CausalBarExecutionError(
            f"Timing input is missing columns: {missing}."
        )

    result = observations.copy(deep=True)
    try:
        timestamps = pd.to_datetime(
            result["timestamp"], utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise CausalBarExecutionError(
            "Timing input contains malformed timestamps."
        ) from exc
    if timestamps.isna().any():
        raise CausalBarExecutionError("Timing timestamps cannot be missing.")
    result["timestamp"] = timestamps

    symbols = result["symbol"].astype("string")
    if symbols.isna().any() or symbols.str.strip().eq("").any():
        raise CausalBarExecutionError(
            "Timing symbols cannot be missing or empty."
        )
    sessions = result["session_date"].astype("string")
    if sessions.isna().any() or sessions.str.strip().eq("").any():
        raise CausalBarExecutionError(
            "Timing session dates cannot be missing or empty."
        )

    signal = pd.to_numeric(result["signal"], errors="coerce")
    if signal.isna().any() or not signal.isin((-1, 0, 1)).all():
        raise CausalBarExecutionError(
            "Timing signals must contain only -1, 0, or 1."
        )
    availability = result["signal_available"]
    if availability.isna().any() or not availability.isin((True, False)).all():
        raise CausalBarExecutionError(
            "signal_available must contain only boolean values."
        )

    for column in ("open", "close"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        values = numeric.to_numpy(dtype="float64")
        if (
            numeric.isna().any()
            or not np.isfinite(values).all()
            or numeric.le(0.0).any()
        ):
            raise CausalBarExecutionError(
                f"Timing {column} prices must be finite and strictly positive."
            )

    result = result.sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if result.duplicated(["symbol", "timestamp"], keep=False).any():
        raise CausalBarExecutionError(
            "Timing input contains duplicate symbol-timestamp rows."
        )
    return result


def apply_causal_next_open_overnight_flat(
    observations: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Apply causal next-open returns with same-close liquidation.

    Signal ``t`` becomes the position at the next observed bar open. A
    non-close row earns ``open[t+1] / open[t] - 1``. A session-close row earns
    ``close[t] / open[t] - 1`` and liquidates on that same row.
    """

    try:
        cost_bps = float(cost_bps_per_turnover)
    except (TypeError, ValueError) as exc:
        raise CausalBarExecutionError(
            "Cost must be a finite non-negative number."
        ) from exc
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise CausalBarExecutionError(
            "Cost must be a finite non-negative number."
        )

    validated = _validate_input(observations)
    parts: list[pd.DataFrame] = []
    for _, source in validated.groupby(
        "symbol", observed=True, sort=True
    ):
        part = source.sort_values(
            "timestamp", kind="stable"
        ).copy(deep=True).reset_index(drop=True)
        sessions = part["session_date"].astype("string")
        session_open = sessions.ne(sessions.shift(1)).fillna(True)
        session_close = sessions.ne(sessions.shift(-1)).fillna(True)

        signal = pd.to_numeric(part["signal"], errors="raise").astype(
            "int8"
        )
        position = signal.shift(1, fill_value=0).astype("int8")
        eligible = part["signal_available"].shift(
            1, fill_value=False
        ).astype(bool)

        previous_end = position.shift(1, fill_value=0).astype("int8")
        previous_end = previous_end.mask(session_open, 0).astype("int8")
        open_turnover = position.sub(previous_end).abs().astype("float64")
        close_turnover = position.abs().where(
            session_close, 0
        ).astype("float64")

        current_open = pd.to_numeric(part["open"], errors="raise").astype(
            "float64"
        )
        current_close = pd.to_numeric(
            part["close"], errors="raise"
        ).astype("float64")
        next_open = current_open.shift(-1)
        proxy_return = next_open.div(current_open).sub(1.0)
        proxy_return = proxy_return.where(
            ~session_close,
            current_close.div(current_open).sub(1.0),
        )
        proxy_values = proxy_return.to_numpy(dtype="float64")
        if (
            proxy_return.isna().any()
            or not np.isfinite(proxy_values).all()
            or proxy_return.le(-1.0).any()
        ):
            raise CausalBarExecutionError(
                "Next-open proxy returns are invalid."
            )

        part["position"] = position
        part["position_eligible"] = eligible
        part["is_session_open"] = session_open.astype(bool)
        part["is_session_close"] = session_close.astype(bool)
        part["ending_position"] = position.mask(
            session_close, 0
        ).astype("int8")
        part["open_turnover"] = open_turnover
        part["close_turnover"] = close_turnover
        part["turnover"] = open_turnover.add(close_turnover)
        part["pnl_proxy_return"] = proxy_return.astype("float64")
        part["gross_strategy_return"] = position.astype("float64").mul(
            proxy_return
        )
        part["transaction_cost"] = (
            part["turnover"] * cost_bps / 10_000.0
        )
        part["net_strategy_return"] = (
            part["gross_strategy_return"] - part["transaction_cost"]
        )
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    session_closes = result["is_session_close"].astype(bool)
    if result.loc[session_closes, "ending_position"].ne(0).any():
        raise CausalBarExecutionError(
            "Forced-flat convention left an overnight position."
        )
    return result
