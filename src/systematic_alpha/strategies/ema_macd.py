"""Recursive EMA/MACD foundations for Trend Strategy 2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import make_column_validator


DEFAULT_FAST_WINDOW: Final[int] = 12
DEFAULT_SLOW_WINDOW: Final[int] = 26
DEFAULT_SIGNAL_WINDOW: Final[int] = 9
DEFAULT_NEUTRAL_BAND: Final[float] = 0.0005
DEFAULT_COST_BPS_PER_TURNOVER: Final[float] = 1.0


class EmaMacdError(ValueError):
    """Raised when EMA/MACD features cannot be constructed safely."""


_require_columns = make_column_validator(EmaMacdError)


@dataclass(frozen=True)
class EmaMacdParameters:
    """Validated parameters for the recursive EMA/MACD model."""

    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW
    signal_window: int = DEFAULT_SIGNAL_WINDOW
    neutral_band: float = DEFAULT_NEUTRAL_BAND
    cost_bps_per_turnover: float = DEFAULT_COST_BPS_PER_TURNOVER
    price_column: str = "close"
    return_column: str = "close_to_close_simple_return"

    def __post_init__(self) -> None:
        """Validate without silently coercing parameter values."""

        for name in (
            "fast_window",
            "slow_window",
            "signal_window",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise EmaMacdError(
                    f"{name} must be an integer; received {value!r}."
                )

            if value <= 0:
                raise EmaMacdError(
                    f"{name} must be strictly positive; received {value}."
                )

        if self.fast_window >= self.slow_window:
            raise EmaMacdError(
                "fast_window must be smaller than slow_window."
            )

        for name in (
            "neutral_band",
            "cost_bps_per_turnover",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmaMacdError(
                    f"{name} must be a finite real number; "
                    f"received {value!r}."
                )

            normalized = float(value)

            if not math.isfinite(normalized):
                raise EmaMacdError(
                    f"{name} must be finite; received {value!r}."
                )

            if normalized < 0.0:
                raise EmaMacdError(
                    f"{name} must be non-negative; received {value}."
                )

        for name in ("price_column", "return_column"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise EmaMacdError(
                    f"{name} must be a non-empty string."
                )


def calculate_ema_alpha(window: int) -> float:
    """Return alpha = 2 / (window + 1)."""

    if isinstance(window, bool) or not isinstance(window, Integral):
        raise EmaMacdError(
            f"window must be an integer; received {window!r}."
        )

    if window <= 0:
        raise EmaMacdError(
            f"window must be strictly positive; received {window}."
        )

    return 2.0 / (float(window) + 1.0)


def calculate_ema_half_life_bars(window: int) -> float:
    """Return the EMA weight half-life in observation bars."""

    alpha = calculate_ema_alpha(window)

    if alpha == 1.0:
        return 0.0

    return math.log(0.5) / math.log(1.0 - alpha)


def _coerce_numeric_series(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Convert one series without hiding malformed values."""

    if not isinstance(values, pd.Series):
        raise EmaMacdError(
            f"{context} must be a pandas Series."
        )

    raw = values.copy(deep=True)
    numeric = pd.to_numeric(raw, errors="coerce")
    malformed = raw.notna() & numeric.isna()

    if malformed.any():
        bad_rows = malformed[malformed].index.tolist()[:5]
        raise EmaMacdError(
            f"{context} contains non-numeric values. "
            f"Example row indices: {bad_rows}."
        )

    finite = numeric.dropna().to_numpy(dtype="float64")

    if not np.isfinite(finite).all():
        raise EmaMacdError(
            f"{context} contains infinite values."
        )

    return numeric.astype("float64")


def calculate_recursive_ema(
    values: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Calculate an adjust=False EMA seeded from the first valid value.

    Leading missing values are permitted so the MACD signal-line EMA can
    begin when MACD first becomes available. Missing values after the first
    valid observation are rejected.
    """

    calculate_ema_alpha(window)

    numeric = _coerce_numeric_series(
        values,
        context="EMA input",
    )

    first_valid_position = numeric.first_valid_index()

    result = pd.Series(
        np.nan,
        index=numeric.index,
        dtype="float64",
        name=values.name,
    )

    if first_valid_position is None:
        return result

    first_valid_location = numeric.index.get_loc(
        first_valid_position
    )
    valid_tail = numeric.iloc[first_valid_location:]

    if valid_tail.isna().any():
        raise EmaMacdError(
            "EMA input may contain missing values only before its "
            "first valid observation."
        )

    ema_tail = valid_tail.ewm(
        span=window,
        adjust=False,
        min_periods=window,
    ).mean()

    result.iloc[first_valid_location:] = ema_tail.to_numpy(
        dtype="float64"
    )

    return result


def _normalize_feature_input(
    frame: pd.DataFrame,
    parameters: EmaMacdParameters,
) -> pd.DataFrame:
    """Validate and sort EMA/MACD feature input."""

    if not isinstance(frame, pd.DataFrame):
        raise EmaMacdError(
            "EMA/MACD input must be a pandas DataFrame."
        )

    if frame.empty:
        raise EmaMacdError(
            "EMA/MACD input cannot be empty."
        )

    required = (
        "timestamp",
        "symbol",
        parameters.price_column,
        parameters.return_column,
    )
    _require_columns(
        frame,
        required,
        context="EMA/MACD input",
    )

    result = frame.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise EmaMacdError(
            "EMA/MACD input contains malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise EmaMacdError(
            "EMA/MACD input contains missing timestamps."
        )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        result["symbol"].isna().any()
        or result["symbol"].eq("").any()
    ):
        raise EmaMacdError(
            "EMA/MACD input contains missing or empty symbols."
        )

    result[parameters.price_column] = _coerce_numeric_series(
        result[parameters.price_column],
        context=f"Column {parameters.price_column!r}",
    )

    prices = result[parameters.price_column]

    if prices.isna().any():
        raise EmaMacdError(
            f"Price column {parameters.price_column!r} "
            "contains missing observations."
        )

    if prices.le(0.0).any():
        raise EmaMacdError(
            f"Price column {parameters.price_column!r} "
            "must be strictly positive."
        )

    result[parameters.return_column] = _coerce_numeric_series(
        result[parameters.return_column],
        context=f"Column {parameters.return_column!r}",
    )

    result = result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_mask = result.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        raise EmaMacdError(
            "EMA/MACD input contains duplicate symbol-timestamp "
            "observations."
        )

    first_symbol_observation = (
        result.groupby(
            "symbol",
            observed=True,
            sort=False,
        )
        .cumcount()
        .eq(0)
    )

    missing_returns = result[
        parameters.return_column
    ].isna()

    invalid_missing_returns = (
        missing_returns & ~first_symbol_observation
    )

    if invalid_missing_returns.any():
        raise EmaMacdError(
            "Missing returns are permitted only for the first "
            "observation of each symbol."
        )

    impossible_returns = (
        result[parameters.return_column]
        .dropna()
        .le(-1.0)
    )

    if impossible_returns.any():
        raise EmaMacdError(
            "Simple returns must be greater than -1.0."
        )

    return result


def build_ema_macd_features(
    frame: pd.DataFrame,
    *,
    parameters: EmaMacdParameters,
) -> pd.DataFrame:
    """Build recursive EMA/MACD states without trading positions.

    EMA state continues across exchange-session boundaries. The signal-line
    EMA begins at the first valid MACD observation. The normalized histogram
    is the frozen continuous signal for the Day 8 baseline.
    """

    observations = _normalize_feature_input(
        frame,
        parameters,
    )

    price_column = parameters.price_column

    observations["fast_ema"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )[price_column]
        .transform(
            lambda values: calculate_recursive_ema(
                values,
                window=parameters.fast_window,
            )
        )
        .astype("float64")
    )

    observations["slow_ema"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )[price_column]
        .transform(
            lambda values: calculate_recursive_ema(
                values,
                window=parameters.slow_window,
            )
        )
        .astype("float64")
    )

    observations["macd"] = (
        observations["fast_ema"]
        - observations["slow_ema"]
    )

    observations["macd_signal_line"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["macd"]
        .transform(
            lambda values: calculate_recursive_ema(
                values,
                window=parameters.signal_window,
            )
        )
        .astype("float64")
    )

    observations["macd_histogram"] = (
        observations["macd"]
        - observations["macd_signal_line"]
    )

    observations["normalized_macd_histogram"] = (
        observations["macd_histogram"]
        / observations[price_column]
    )

    observations["histogram_change"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["macd_histogram"]
        .diff()
        .astype("float64")
    )

    observations["histogram_acceleration"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["histogram_change"]
        .diff()
        .astype("float64")
    )

    observations["signal_available"] = (
        observations["normalized_macd_histogram"].notna()
    )

    return observations


EMA_MACD_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "observations",
    "signal_available_observations",
    "position_eligible_observations",
    "signal_warmup_observations",
    "position_warmup_observations",
    "fast_ema_half_life_bars",
    "slow_ema_half_life_bars",
    "signal_ema_half_life_bars",
    "total_turnover",
    "position_changing_bars",
    "long_exposure_pct",
    "short_exposure_pct",
    "neutral_exposure_pct",
)


@dataclass(frozen=True)
class EmaMacdBundle:
    """Complete EMA/MACD strategy output and compact diagnostics."""

    parameters: EmaMacdParameters
    observations: pd.DataFrame
    diagnostics: pd.DataFrame


def _leading_unavailable_observations(
    availability: pd.Series,
) -> int:
    """Count observations before the first available state."""

    if not isinstance(availability, pd.Series):
        raise TypeError(
            "availability must be a pandas Series."
        )

    values = availability.astype(bool).to_numpy()
    available_locations = np.flatnonzero(values)

    if available_locations.size == 0:
        return int(len(values))

    return int(available_locations[0])


def _exposure_percentage(
    eligible_positions: pd.Series,
    *,
    position_value: int,
) -> float:
    """Calculate exposure over position-eligible observations."""

    if eligible_positions.empty:
        return float("nan")

    return float(
        100.0
        * eligible_positions.eq(position_value).mean()
    )


def _build_ema_macd_diagnostics(
    observations: pd.DataFrame,
    *,
    parameters: EmaMacdParameters,
) -> pd.DataFrame:
    """Build one compact diagnostic row per symbol."""

    records: list[dict[str, object]] = []

    for symbol, group in observations.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        ordered = group.sort_values(
            "timestamp",
            kind="stable",
        )

        eligible_positions = ordered.loc[
            ordered["position_eligible"],
            "position",
        ]

        records.append(
            {
                "symbol": str(symbol),
                "observations": int(len(ordered)),
                "signal_available_observations": int(
                    ordered["signal_available"].sum()
                ),
                "position_eligible_observations": int(
                    ordered["position_eligible"].sum()
                ),
                "signal_warmup_observations": (
                    _leading_unavailable_observations(
                        ordered["signal_available"]
                    )
                ),
                "position_warmup_observations": (
                    _leading_unavailable_observations(
                        ordered["position_eligible"]
                    )
                ),
                "fast_ema_half_life_bars": (
                    calculate_ema_half_life_bars(
                        parameters.fast_window
                    )
                ),
                "slow_ema_half_life_bars": (
                    calculate_ema_half_life_bars(
                        parameters.slow_window
                    )
                ),
                "signal_ema_half_life_bars": (
                    calculate_ema_half_life_bars(
                        parameters.signal_window
                    )
                ),
                "total_turnover": float(
                    ordered["turnover"].sum()
                ),
                "position_changing_bars": int(
                    ordered["turnover"].gt(0.0).sum()
                ),
                "long_exposure_pct": _exposure_percentage(
                    eligible_positions,
                    position_value=1,
                ),
                "short_exposure_pct": _exposure_percentage(
                    eligible_positions,
                    position_value=-1,
                ),
                "neutral_exposure_pct": _exposure_percentage(
                    eligible_positions,
                    position_value=0,
                ),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=EMA_MACD_DIAGNOSTIC_COLUMNS,
    )


def build_ema_macd_strategy(
    frame: pd.DataFrame,
    *,
    parameters: EmaMacdParameters = EmaMacdParameters(),
) -> EmaMacdBundle:
    """Build the frozen cost-aware EMA/MACD baseline strategy.

    Signal construction uses the normalized MACD histogram available at bar
    t. The tradable position at bar t is the previous bar's signal. EMA state
    and target positions continue across exchange-session boundaries.
    """

    if not isinstance(parameters, EmaMacdParameters):
        raise EmaMacdError(
            "parameters must be an EmaMacdParameters object."
        )

    observations = build_ema_macd_features(
        frame,
        parameters=parameters,
    )

    continuous_signal = observations[
        "normalized_macd_histogram"
    ]
    available = observations["signal_available"]

    signal = np.select(
        condlist=(
            available
            & continuous_signal.gt(parameters.neutral_band),
            available
            & continuous_signal.lt(-parameters.neutral_band),
        ),
        choicelist=(1, -1),
        default=0,
    )

    observations["signal"] = pd.Series(
        signal,
        index=observations.index,
        dtype="int8",
    )

    observations["position"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["signal"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    observations["position_eligible"] = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["signal_available"]
        .shift(1, fill_value=False)
        .astype(bool)
    )

    prior_position = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["position"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    observations["turnover"] = (
        observations["position"]
        .sub(prior_position)
        .abs()
        .astype("float64")
    )

    pnl_return = observations[
        parameters.return_column
    ].fillna(0.0)

    observations["gross_strategy_return"] = (
        observations["position"].astype("float64")
        * pnl_return
    )

    observations["transaction_cost"] = (
        observations["turnover"]
        * float(parameters.cost_bps_per_turnover)
        / 10_000.0
    )

    observations["net_strategy_return"] = (
        observations["gross_strategy_return"]
        - observations["transaction_cost"]
    )

    diagnostics = _build_ema_macd_diagnostics(
        observations,
        parameters=parameters,
    )

    return EmaMacdBundle(
        parameters=parameters,
        observations=observations,
        diagnostics=diagnostics,
    )
