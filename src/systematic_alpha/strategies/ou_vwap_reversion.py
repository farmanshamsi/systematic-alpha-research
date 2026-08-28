"""Causal OU/VWAP residual mean-reversion strategy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis._validation import make_column_validator


BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0


class OuVwapReversionError(ValueError):
    """Raised when the reversion strategy cannot be built safely."""


_require_columns = make_column_validator(OuVwapReversionError)


@dataclass(frozen=True, slots=True)
class OuVwapReversionParameters:
    """Validated parameters for one frozen sensitivity calibration."""

    configuration_id: str
    reference_window: int
    ou_window: int
    variance_ratio_lag: int
    variance_ratio_threshold: float
    entry_threshold: float
    exit_threshold: float
    minimum_half_life: float
    maximum_half_life: float
    maximum_holding_bars: int
    cost_bps_per_turnover: float = 1.0
    price_column: str = "close"
    vwap_column: str = "vwap"
    volume_column: str = "volume"
    return_column: str = "close_to_close_simple_return"

    def __post_init__(self) -> None:
        """Fail closed on malformed or incoherent strategy parameters."""

        for name in (
            "configuration_id",
            "price_column",
            "vwap_column",
            "volume_column",
            "return_column",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise OuVwapReversionError(f"{name} must be a non-empty string.")

        for name in (
            "reference_window",
            "ou_window",
            "variance_ratio_lag",
            "maximum_holding_bars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise OuVwapReversionError(f"{name} must be an integer.")
            if value <= 0:
                raise OuVwapReversionError(f"{name} must be strictly positive.")

        if self.reference_window < 2:
            raise OuVwapReversionError("reference_window must be at least two.")
        if self.ou_window < 3:
            raise OuVwapReversionError("ou_window must contain at least three transitions.")
        if self.variance_ratio_lag >= self.ou_window:
            raise OuVwapReversionError("variance_ratio_lag must be smaller than ou_window.")

        for name in (
            "variance_ratio_threshold",
            "entry_threshold",
            "exit_threshold",
            "minimum_half_life",
            "maximum_half_life",
            "cost_bps_per_turnover",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise OuVwapReversionError(f"{name} must be a finite real number.")
            if not math.isfinite(float(value)):
                raise OuVwapReversionError(f"{name} must be finite.")

        if not 0.0 < self.variance_ratio_threshold < 1.0:
            raise OuVwapReversionError(
                "variance_ratio_threshold must lie strictly between zero and one."
            )
        if self.entry_threshold <= 0.0:
            raise OuVwapReversionError("entry_threshold must be strictly positive.")
        if not 0.0 <= self.exit_threshold < self.entry_threshold:
            raise OuVwapReversionError(
                "exit_threshold must be non-negative and smaller than entry_threshold."
            )
        if self.minimum_half_life <= 0.0:
            raise OuVwapReversionError("minimum_half_life must be strictly positive.")
        if self.maximum_half_life < self.minimum_half_life:
            raise OuVwapReversionError(
                "maximum_half_life must not be smaller than minimum_half_life."
            )
        if self.cost_bps_per_turnover < 0.0:
            raise OuVwapReversionError("cost_bps_per_turnover must be non-negative.")


@dataclass(frozen=True, slots=True)
class OuVwapReversionBundle:
    """Strategy observations, diagnostics, and frozen parameters."""

    observations: pd.DataFrame
    diagnostics: pd.DataFrame
    parameters: OuVwapReversionParameters


def _coerce_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    malformed = raw.notna() & numeric.isna()
    if malformed.any():
        rows = malformed[malformed].index.tolist()[:5]
        raise OuVwapReversionError(
            f"Column {column!r} contains non-numeric values at rows {rows}."
        )
    finite = numeric.dropna().to_numpy(dtype="float64")
    if not np.isfinite(finite).all():
        raise OuVwapReversionError(f"Column {column!r} contains infinite values.")
    return numeric.astype("float64")


def _normalize_input(
    frame: pd.DataFrame,
    parameters: OuVwapReversionParameters,
    *,
    allow_incomplete_final_session: bool = False,
) -> pd.DataFrame:
    """Validate chronological strategy input without imputing market data."""

    if not isinstance(frame, pd.DataFrame):
        raise OuVwapReversionError("Strategy input must be a pandas DataFrame.")
    if frame.empty:
        raise OuVwapReversionError("Strategy input cannot be empty.")

    required = (
        "timestamp",
        "symbol",
        "session_date",
        "is_session_close_bar",
        parameters.price_column,
        parameters.vwap_column,
        parameters.volume_column,
        parameters.return_column,
    )
    _require_columns(frame, required, context="OU/VWAP reversion input")
    result = frame.copy(deep=True)

    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
        result["session_date"] = pd.to_datetime(
            result["session_date"], utc=True, errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise OuVwapReversionError("Strategy input contains malformed dates.") from exc
    if result[["timestamp", "session_date"]].isna().any().any():
        raise OuVwapReversionError("Strategy dates cannot be missing.")

    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    if result["symbol"].isna().any() or result["symbol"].eq("").any():
        raise OuVwapReversionError("Strategy symbols cannot be missing or empty.")

    close_flag = result["is_session_close_bar"]
    if close_flag.isna().any() or not close_flag.isin((True, False)).all():
        raise OuVwapReversionError("is_session_close_bar must contain booleans.")
    result["is_session_close_bar"] = close_flag.astype(bool)

    for column in (
        parameters.price_column,
        parameters.vwap_column,
        parameters.volume_column,
        parameters.return_column,
    ):
        result[column] = _coerce_numeric(result, column)

    if result[parameters.price_column].isna().any() or result[
        parameters.price_column
    ].le(0.0).any():
        raise OuVwapReversionError("Prices must be present and strictly positive.")
    if result[parameters.vwap_column].isna().any() or result[
        parameters.vwap_column
    ].le(0.0).any():
        raise OuVwapReversionError("VWAP values must be present and strictly positive.")
    if result[parameters.volume_column].isna().any() or result[
        parameters.volume_column
    ].lt(0.0).any():
        raise OuVwapReversionError("Volume must be present and non-negative.")

    result = result.sort_values(["symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    if result.duplicated(["symbol", "timestamp"], keep=False).any():
        raise OuVwapReversionError(
            "Strategy input contains duplicate symbol-timestamp observations."
        )

    first = result.groupby("symbol", observed=True, sort=False).cumcount().eq(0)
    missing_return = result[parameters.return_column].isna()
    if (missing_return & ~first).any():
        raise OuVwapReversionError(
            "Missing returns are permitted only on the first row per symbol."
        )
    if result[parameters.return_column].dropna().le(-1.0).any():
        raise OuVwapReversionError("Simple returns must be greater than -1.0.")

    if type(allow_incomplete_final_session) is not bool:
        raise OuVwapReversionError(
            "allow_incomplete_final_session must be a boolean."
        )
    close_counts = result.groupby(
        ["symbol", "session_date"], observed=True, sort=True
    )["is_session_close_bar"].sum()
    if not allow_incomplete_final_session and not close_counts.eq(1).all():
        raise OuVwapReversionError(
            "Every symbol-session must have exactly one session-close bar."
        )
    if allow_incomplete_final_session:
        for symbol in result["symbol"].unique():
            symbol_counts = close_counts.loc[symbol]
            invalid = symbol_counts.ne(1)
            if invalid.any():
                invalid_sessions = symbol_counts.index[invalid]
                if (
                    len(invalid_sessions) != 1
                    or invalid_sessions[0] != symbol_counts.index.max()
                    or int(symbol_counts.loc[invalid_sessions[0]]) != 0
                ):
                    raise OuVwapReversionError(
                        "Only the final session per symbol may omit its close bar."
                    )
    return result


def _rolling_ou_statistics(values: pd.Series, *, window: int) -> pd.DataFrame:
    """Return vectorized rolling AR(1)/OU diagnostics from past data only."""

    y = values.astype("float64")
    lag = y.shift(1)
    min_periods = window
    sx = lag.rolling(window, min_periods=min_periods).sum()
    sy = y.rolling(window, min_periods=min_periods).sum()
    sxx = lag.pow(2).rolling(window, min_periods=min_periods).sum()
    syy = y.pow(2).rolling(window, min_periods=min_periods).sum()
    sxy = lag.mul(y).rolling(window, min_periods=min_periods).sum()
    n = float(window)

    centered_xx = sxx - sx.pow(2) / n
    centered_xy = sxy - sx.mul(sy) / n
    phi = centered_xy / centered_xx.where(centered_xx.gt(0.0))
    intercept = sy / n - phi * sx / n

    sse = (
        syy
        - 2.0 * intercept * sy
        - 2.0 * phi * sxy
        + n * intercept.pow(2)
        + 2.0 * intercept * phi * sx
        + phi.pow(2) * sxx
    ).clip(lower=0.0)
    innovation_std = np.sqrt(sse / (n - 2.0))
    compatible = phi.gt(0.0) & phi.lt(1.0)
    safe_phi = phi.where(compatible)
    equilibrium = intercept / (1.0 - safe_phi)
    stationary_std = (
        innovation_std / np.sqrt(1.0 - safe_phi.pow(2))
    ).where(innovation_std.gt(0.0))
    half_life = -math.log(2.0) / np.log(safe_phi)

    return pd.DataFrame(
        {
            "ou_intercept": intercept,
            "ou_phi": phi,
            "ou_equilibrium": equilibrium,
            "ou_innovation_std": innovation_std,
            "ou_stationary_std": stationary_std,
            "ou_half_life_bars": half_life,
        },
        index=values.index,
    )


def _target_state(
    zscore: pd.Series,
    regime: pd.Series,
    session_close: pd.Series,
    reset_state: pd.Series,
    *,
    parameters: OuVwapReversionParameters,
) -> tuple[pd.Series, pd.Series]:
    """Build a deterministic stateful target and holding-period series."""

    target = np.zeros(len(zscore), dtype="int8")
    holding = np.zeros(len(zscore), dtype="int64")
    current = 0
    age = 0

    z_values = zscore.to_numpy(dtype="float64")
    regime_values = regime.to_numpy(dtype="bool")
    close_values = session_close.to_numpy(dtype="bool")
    reset_values = reset_state.to_numpy(dtype="bool")

    for index, (z_value, allowed, at_close, reset) in enumerate(
        zip(z_values, regime_values, close_values, reset_values, strict=True)
    ):
        if reset:
            current = 0
            age = 0
        if at_close:
            current = 0
            age = 0
        elif current == 0:
            age = 0
            if allowed and z_value <= -parameters.entry_threshold:
                current = 1
            elif allowed and z_value >= parameters.entry_threshold:
                current = -1
        else:
            age += 1
            exit_for_mean = (
                current == 1 and z_value >= -parameters.exit_threshold
            ) or (current == -1 and z_value <= parameters.exit_threshold)
            if (
                not allowed
                or exit_for_mean
                or age >= parameters.maximum_holding_bars
            ):
                current = 0
                age = 0

        target[index] = current
        holding[index] = age if current != 0 else 0

    return (
        pd.Series(target, index=zscore.index, dtype="int8", name="signal"),
        pd.Series(holding, index=zscore.index, dtype="int64", name="holding_bars"),
    )


def _build_diagnostics(observations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for symbol, group in observations.groupby("symbol", observed=True, sort=True):
        eligible = group["position_eligible"].astype(bool)
        sample = group.loc[eligible]
        denominator = len(sample)

        def pct(value: int) -> float:
            if denominator == 0:
                return float("nan")
            return 100.0 * float(sample["position"].eq(value).sum()) / denominator

        prior_position = group["position"].shift(1, fill_value=0)
        entry_count = int(
            (group["position"].ne(0) & prior_position.eq(0)).sum()
        )
        records.append(
            {
                "symbol": str(symbol),
                "observations": int(len(group)),
                "signal_available_observations": int(group["signal_available"].sum()),
                "regime_eligible_observations": int(group["regime_eligible"].sum()),
                "entries": entry_count,
                "long_exposure_pct": pct(1),
                "short_exposure_pct": pct(-1),
                "flat_exposure_pct": pct(0),
                "total_turnover": float(group["turnover"].sum()),
                "forced_session_flat_count": int(group["is_session_close_bar"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def build_ou_vwap_reversion_strategy(
    frame: pd.DataFrame,
    *,
    parameters: OuVwapReversionParameters,
    execution_reset_timestamps: tuple[pd.Timestamp, ...] = (),
    allow_incomplete_final_session: bool = False,
) -> OuVwapReversionBundle:
    """Construct the lagged, cost-aware OU/VWAP reversion strategy."""

    if not isinstance(parameters, OuVwapReversionParameters):
        raise OuVwapReversionError(
            "parameters must be an OuVwapReversionParameters object."
        )
    if not isinstance(execution_reset_timestamps, tuple):
        raise OuVwapReversionError(
            "execution_reset_timestamps must be a tuple of timestamps."
        )
    normalized_resets: set[pd.Timestamp] = set()
    for value in execution_reset_timestamps:
        if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
            raise OuVwapReversionError(
                "Execution reset timestamps must be timezone-aware pandas Timestamps."
            )
        normalized_resets.add(value.tz_convert("UTC"))

    observations = _normalize_input(
        frame,
        parameters,
        allow_incomplete_final_session=allow_incomplete_final_session,
    )
    pieces: list[pd.DataFrame] = []

    for _, group in observations.groupby("symbol", observed=True, sort=False):
        part = group.copy(deep=True)
        rolling_dollar = part[parameters.vwap_column].mul(
            part[parameters.volume_column]
        ).rolling(
            parameters.reference_window,
            min_periods=parameters.reference_window,
        ).sum()
        rolling_volume = part[parameters.volume_column].rolling(
            parameters.reference_window,
            min_periods=parameters.reference_window,
        ).sum()
        part["volume_weighted_reference"] = (
            rolling_dollar / rolling_volume.where(rolling_volume.gt(0.0))
        )
        part["log_price_residual"] = np.log(
            part[parameters.price_column] / part["volume_weighted_reference"]
        )

        ou = _rolling_ou_statistics(part["log_price_residual"], window=parameters.ou_window)
        for column in ou.columns:
            part[column] = ou[column]

        one_period_change = part["log_price_residual"].diff()
        lagged_change = part["log_price_residual"].diff(parameters.variance_ratio_lag)
        one_variance = one_period_change.rolling(
            parameters.ou_window,
            min_periods=parameters.ou_window,
        ).var(ddof=1)
        lagged_variance = lagged_change.rolling(
            parameters.ou_window,
            min_periods=parameters.ou_window,
        ).var(ddof=1)
        part["variance_ratio"] = lagged_variance / (
            float(parameters.variance_ratio_lag) * one_variance.where(one_variance.gt(0.0))
        )
        part["ou_zscore"] = (
            part["log_price_residual"] - part["ou_equilibrium"]
        ) / part["ou_stationary_std"]
        part["signal_available"] = part[
            ["ou_zscore", "ou_phi", "ou_half_life_bars", "variance_ratio"]
        ].notna().all(axis=1)
        part["regime_eligible"] = (
            part["signal_available"]
            & part["ou_phi"].gt(0.0)
            & part["ou_phi"].lt(1.0)
            & part["ou_half_life_bars"].ge(parameters.minimum_half_life)
            & part["ou_half_life_bars"].le(parameters.maximum_half_life)
            & part["variance_ratio"].lt(parameters.variance_ratio_threshold)
        )
        part["signal_score"] = (-part["ou_zscore"]).where(
            part["regime_eligible"] & ~part["is_session_close_bar"]
        )
        reset_mask = part["timestamp"].isin(normalized_resets)
        signal, holding = _target_state(
            part["ou_zscore"],
            part["regime_eligible"],
            part["is_session_close_bar"],
            reset_mask,
            parameters=parameters,
        )
        part["signal"] = signal
        part["holding_bars"] = holding
        part["position"] = part["signal"].shift(1, fill_value=0).astype("int8")
        part.loc[reset_mask, "position"] = 0
        part["position_eligible"] = part["signal_available"].shift(
            1, fill_value=False
        ).astype(bool)
        previous_position = part["position"].shift(1, fill_value=0).astype("int8")
        part["turnover"] = part["position"].sub(previous_position).abs().astype("float64")
        part.loc[reset_mask, "turnover"] = 0.0
        market_return = part[parameters.return_column].fillna(0.0)
        part["gross_strategy_return"] = part["position"].astype("float64") * market_return
        part["transaction_cost"] = (
            part["turnover"]
            * float(parameters.cost_bps_per_turnover)
            / BASIS_POINTS_PER_UNIT
        )
        part["net_strategy_return"] = (
            part["gross_strategy_return"] - part["transaction_cost"]
        )
        pieces.append(part)

    complete = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    diagnostics = _build_diagnostics(complete)
    return OuVwapReversionBundle(
        observations=complete,
        diagnostics=diagnostics,
        parameters=parameters,
    )
