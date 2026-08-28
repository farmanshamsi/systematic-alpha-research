"""Short/long average price-ratio trend strategy.

The historical Day 6 lineage uses long-short-neutral positioning. Long-flat is
implemented as an explicit comparison mode; it is not the source of the saved
Day 6 through Day 16 results.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import make_column_validator


BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0
DEFAULT_SHORT_WINDOW: Final[int] = 8
DEFAULT_LONG_WINDOW: Final[int] = 32
DEFAULT_NEUTRAL_BAND: Final[float] = 0.001
DEFAULT_COST_BPS_PER_TURNOVER: Final[float] = 1.0
DEFAULT_POSITIONING: Final[str] = "long_short_neutral"
ALLOWED_POSITIONING: Final[tuple[str, ...]] = (
    "long_short_neutral",
    "long_flat",
)


class TrendRatioError(ValueError):
    """Raised when the trend-ratio strategy cannot be constructed safely."""


_require_columns = make_column_validator(TrendRatioError)


@dataclass(frozen=True)
class TrendRatioParameters:
    """Validated parameters for the short/long average price-ratio model."""

    short_window: int
    long_window: int
    neutral_band: float
    cost_bps_per_turnover: float = 0.0
    price_column: str = "close"
    return_column: str = "close_to_close_simple_return"
    positioning: str = DEFAULT_POSITIONING

    def __post_init__(self) -> None:
        """Validate strategy parameters without silently coercing them."""
        for name in ("short_window", "long_window"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TrendRatioError(
                    f"{name} must be an integer; received {value!r}."
                )

            if value <= 0:
                raise TrendRatioError(
                    f"{name} must be strictly positive; received {value}."
                )

        if self.short_window >= self.long_window:
            raise TrendRatioError(
                "short_window must be smaller than long_window."
            )

        for name in ("neutral_band", "cost_bps_per_turnover"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TrendRatioError(
                    f"{name} must be a finite real number; "
                    f"received {value!r}."
                )

            if not np.isfinite(float(value)):
                raise TrendRatioError(
                    f"{name} must be finite; received {value!r}."
                )

            if value < 0:
                raise TrendRatioError(
                    f"{name} must be non-negative; received {value}."
                )

        for name in ("price_column", "return_column"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise TrendRatioError(
                    f"{name} must be a non-empty string."
                )

        if self.positioning not in ALLOWED_POSITIONING:
            raise TrendRatioError(
                "positioning must be one of "
                f"{ALLOWED_POSITIONING}; received {self.positioning!r}."
            )


@dataclass(frozen=True)
class TrendRatioBundle:
    """Trend-ratio observations and compact signal diagnostics."""

    observations: pd.DataFrame
    diagnostics: pd.DataFrame
    parameters: TrendRatioParameters


def _coerce_numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Convert one required column without hiding malformed values."""
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")

    malformed = raw.notna() & numeric.isna()

    if malformed.any():
        bad_rows = malformed[malformed].index.tolist()[:5]
        raise TrendRatioError(
            f"Column {column!r} contains non-numeric values. "
            f"Example row indices: {bad_rows}."
        )

    finite_values = numeric.dropna().to_numpy(dtype="float64")

    if not np.isfinite(finite_values).all():
        raise TrendRatioError(
            f"Column {column!r} contains infinite values."
        )

    return numeric.astype("float64")


def _normalize_strategy_input(
    frame: pd.DataFrame,
    parameters: TrendRatioParameters,
) -> pd.DataFrame:
    """Validate and sort strategy input without imputing observations."""
    if not isinstance(frame, pd.DataFrame):
        raise TrendRatioError(
            "Strategy input must be a pandas DataFrame."
        )

    if frame.empty:
        raise TrendRatioError(
            "Strategy input cannot be empty."
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
        context="Trend-ratio input",
    )

    result = frame.copy()

    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TrendRatioError(
            "Trend-ratio input contains malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise TrendRatioError(
            "Trend-ratio input contains missing timestamps."
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
        raise TrendRatioError(
            "Trend-ratio input contains missing or empty symbols."
        )

    result[parameters.price_column] = _coerce_numeric_column(
        result,
        parameters.price_column,
    )

    prices = result[parameters.price_column]

    if prices.isna().any():
        raise TrendRatioError(
            f"Price column {parameters.price_column!r} "
            "contains missing observations."
        )

    if prices.le(0).any():
        raise TrendRatioError(
            f"Price column {parameters.price_column!r} "
            "must be strictly positive."
        )

    result[parameters.return_column] = _coerce_numeric_column(
        result,
        parameters.return_column,
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
        duplicate_count = int(duplicate_mask.sum())
        raise TrendRatioError(
            "Trend-ratio input contains duplicate symbol-timestamp "
            f"observations: {duplicate_count} rows."
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

    missing_returns = result[parameters.return_column].isna()
    invalid_missing_returns = (
        missing_returns & ~first_symbol_observation
    )

    if invalid_missing_returns.any():
        invalid_count = int(invalid_missing_returns.sum())
        raise TrendRatioError(
            "Missing returns are permitted only for the first "
            "observation of each symbol. "
            f"Invalid missing returns: {invalid_count}."
        )

    impossible_returns = (
        result[parameters.return_column]
        .dropna()
        .le(-1.0)
    )

    if impossible_returns.any():
        raise TrendRatioError(
            "Simple returns must be greater than -1.0."
        )

    return result


def _build_raw_signal(
    price_ratio: pd.Series,
    *,
    neutral_band: float,
    positioning: str,
) -> pd.Series:
    """Apply the explicit neutral band and declared positioning rule."""
    upper_threshold = 1.0 + neutral_band
    lower_threshold = 1.0 - neutral_band

    signal = np.select(
        [
            price_ratio.gt(upper_threshold),
            price_ratio.lt(lower_threshold),
        ],
        [
            1,
            -1 if positioning == "long_short_neutral" else 0,
        ],
        default=0,
    )

    result = pd.Series(
        signal,
        index=price_ratio.index,
        dtype="int8",
        name="signal",
    )

    result.loc[price_ratio.isna()] = 0
    return result


def calculate_turnover(
    position: pd.Series,
    symbols: pd.Series,
) -> pd.Series:
    """Calculate absolute position change, including direct reversals."""
    if len(position) != len(symbols):
        raise TrendRatioError(
            "Position and symbol inputs must have equal lengths."
        )

    numeric_position = pd.to_numeric(
        position,
        errors="coerce",
    )

    if numeric_position.isna().any():
        raise TrendRatioError(
            "Position contains missing or non-numeric values."
        )

    if not numeric_position.isin((-1, 0, 1)).all():
        raise TrendRatioError(
            "Position values must belong to {-1, 0, 1}."
        )

    normalized_symbols = (
        symbols.astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        normalized_symbols.isna().any()
        or normalized_symbols.eq("").any()
    ):
        raise TrendRatioError(
            "Turnover calculation requires valid symbols."
        )

    calculation = pd.DataFrame(
        {
            "symbol": normalized_symbols,
            "position": numeric_position.astype("int8"),
        },
        index=position.index,
    )

    previous_position = (
        calculation.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["position"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    return (
        calculation["position"]
        .sub(previous_position)
        .abs()
        .astype("float64")
        .rename("turnover")
    )


def build_signal_diagnostics(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact signal, exposure and turnover diagnostics."""
    required = (
        "symbol",
        "signal",
        "position",
        "signal_available",
        "position_eligible",
        "turnover",
    )

    _require_columns(
        observations,
        required,
        context="Trend-ratio observations",
    )

    records: list[dict[str, object]] = []

    for symbol, group in observations.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        signal_sample = group.loc[
            group["signal_available"].astype(bool)
        ]
        position_sample = group.loc[
            group["position_eligible"].astype(bool)
        ]

        position_observations = len(position_sample)

        def exposure_percentage(value: int) -> float:
            if position_observations == 0:
                return float("nan")

            count = int(position_sample["position"].eq(value).sum())
            return 100.0 * count / position_observations

        records.append(
            {
                "symbol": str(symbol),
                "observations": int(len(group)),
                "signal_warmup_observations": int(
                    (~group["signal_available"].astype(bool)).sum()
                ),
                "position_warmup_observations": int(
                    (~group["position_eligible"].astype(bool)).sum()
                ),
                "long_signals": int(
                    signal_sample["signal"].eq(1).sum()
                ),
                "short_signals": int(
                    signal_sample["signal"].eq(-1).sum()
                ),
                "neutral_signals": int(
                    signal_sample["signal"].eq(0).sum()
                ),
                "long_exposure_pct": exposure_percentage(1),
                "short_exposure_pct": exposure_percentage(-1),
                "flat_exposure_pct": exposure_percentage(0),
                "total_turnover": float(group["turnover"].sum()),
                "position_changes": int(
                    group["turnover"].gt(0).sum()
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def build_trend_ratio_strategy(
    frame: pd.DataFrame,
    *,
    parameters: TrendRatioParameters,
) -> TrendRatioBundle:
    """Construct the lagged short/long average price-ratio strategy.

    Rolling averages continue across session boundaries. Consequently, a
    signal from the final bar of one session may determine the position for
    the first bar of the next session.

    The signal calculated using bar t is shifted by one observation before
    becoming a position. This prevents same-row signal look-ahead under the
    saved close-to-close accounting convention. It does not prove that a fill
    at the prior close was executable, especially across session boundaries;
    that convention requires a separate execution-timing sensitivity test.
    """
    observations = _normalize_strategy_input(
        frame,
        parameters,
    )

    price_column = parameters.price_column
    return_column = parameters.return_column

    symbol_group = observations.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    observations["short_average"] = (
        symbol_group[price_column]
        .transform(
            lambda values: values.rolling(
                window=parameters.short_window,
                min_periods=parameters.short_window,
            ).mean()
        )
        .astype("float64")
    )

    observations["long_average"] = (
        symbol_group[price_column]
        .transform(
            lambda values: values.rolling(
                window=parameters.long_window,
                min_periods=parameters.long_window,
            ).mean()
        )
        .astype("float64")
    )

    observations["ma_price_ratio"] = (
        observations["short_average"]
        / observations["long_average"]
    )

    observations["signal_available"] = (
        observations["ma_price_ratio"].notna()
    )

    observations["signal"] = _build_raw_signal(
        observations["ma_price_ratio"],
        neutral_band=float(parameters.neutral_band),
        positioning=parameters.positioning,
    )

    signal_group = observations.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    observations["position"] = (
        signal_group["signal"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    observations["position_eligible"] = (
        signal_group["signal_available"]
        .shift(1, fill_value=False)
        .astype(bool)
    )

    observations["turnover"] = calculate_turnover(
        observations["position"],
        observations["symbol"],
    )

    asset_return = observations[return_column].fillna(0.0)

    observations["gross_strategy_return"] = (
        observations["position"].astype("float64")
        * asset_return
    )

    observations["transaction_cost"] = (
        observations["turnover"]
        * float(parameters.cost_bps_per_turnover)
        / BASIS_POINTS_PER_UNIT
    )

    observations["net_strategy_return"] = (
        observations["gross_strategy_return"]
        - observations["transaction_cost"]
    )

    diagnostics = build_signal_diagnostics(observations)

    return TrendRatioBundle(
        observations=observations,
        diagnostics=diagnostics,
        parameters=parameters,
    )
