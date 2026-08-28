"""Development-only diagnostics for the frozen six-sleeve ensemble.

Day 15 Phase 1 measures dependence and covariance concentration.  It does
not select sleeves, optimize weights, or make an allocation recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Final

import numpy as np
import pandas as pd

from systematic_alpha.analysis.eda_features import (
    REQUIRED_COLUMNS as CANONICAL_BAR_COLUMNS,
    build_return_features,
)
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS as DAY10_CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS as DAY10_EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS as DAY10_TREND_RATIO_PARAMETERS,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    build_walk_forward_folds,
)
from systematic_alpha.data.session_aggregation import (
    SessionAggregationError,
    aggregate_session_bars,
)
from systematic_alpha.strategies.ema_macd import (
    build_ema_macd_strategy,
)
from systematic_alpha.strategies.trend_ratio import (
    build_trend_ratio_strategy,
)


DEVELOPMENT_START: Final[pd.Timestamp] = pd.Timestamp(
    "2020-01-02",
    tz="UTC",
)
DEVELOPMENT_END_EXCLUSIVE: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)
FREQUENCY: Final[str] = "15min"
REQUIRED_SYMBOLS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
)

TREND_RATIO_PARAMETERS = DAY10_TREND_RATIO_PARAMETERS
EMA_MACD_PARAMETERS = DAY10_EMA_MACD_PARAMETERS
CONFIGURATION_IDS: Final[dict[str, str]] = dict(
    DAY10_CONFIGURATION_IDS
)

VARIANCE_TOLERANCE: Final[float] = 1e-16
PSD_TOLERANCE: Final[float] = 1e-12
MIN_TRAINING_SESSIONS: Final[int] = 252
MIN_TEST_SESSIONS: Final[int] = 100
MAX_ABSOLUTE_CORRELATION: Final[float] = 0.95
MIN_MEDIAN_EFFECTIVE_RANK: Final[float] = 2.0
MAX_MEDIAN_PC1_SHARE: Final[float] = 0.80
MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO: Final[float] = 1.0


class StrategyDiversificationError(ValueError):
    """Raised when Day 15 diagnostics cannot be built safely."""


@dataclass(frozen=True, slots=True)
class SleeveSpec:
    """One immutable frozen strategy-symbol sleeve."""

    sleeve_id: str
    strategy: str
    symbol: str
    frequency: str
    configuration_id: str

    def __post_init__(self) -> None:
        """Validate membership in the frozen Day 15 contract."""

        if self.strategy not in CONFIGURATION_IDS:
            raise StrategyDiversificationError(
                "Sleeve strategy is outside the frozen Day 10 contract."
            )
        if self.symbol not in REQUIRED_SYMBOLS:
            raise StrategyDiversificationError(
                "Sleeve symbol is outside the frozen Day 15 universe."
            )
        if self.frequency != FREQUENCY:
            raise StrategyDiversificationError(
                "Day 15 supports the 15min frequency only."
            )
        if self.configuration_id != CONFIGURATION_IDS[self.strategy]:
            raise StrategyDiversificationError(
                "Sleeve configuration_id does not match Day 10."
            )
        expected_id = f"{self.strategy}_{self.symbol.lower()}"
        if self.sleeve_id != expected_id:
            raise StrategyDiversificationError(
                "Sleeve identifier does not match strategy and symbol."
            )


FROZEN_SLEEVES: Final[tuple[SleeveSpec, ...]] = (
    SleeveSpec(
        "trend_ratio_spy",
        "trend_ratio",
        "SPY",
        FREQUENCY,
        CONFIGURATION_IDS["trend_ratio"],
    ),
    SleeveSpec(
        "trend_ratio_qqq",
        "trend_ratio",
        "QQQ",
        FREQUENCY,
        CONFIGURATION_IDS["trend_ratio"],
    ),
    SleeveSpec(
        "trend_ratio_iwm",
        "trend_ratio",
        "IWM",
        FREQUENCY,
        CONFIGURATION_IDS["trend_ratio"],
    ),
    SleeveSpec(
        "ema_macd_spy",
        "ema_macd",
        "SPY",
        FREQUENCY,
        CONFIGURATION_IDS["ema_macd"],
    ),
    SleeveSpec(
        "ema_macd_qqq",
        "ema_macd",
        "QQQ",
        FREQUENCY,
        CONFIGURATION_IDS["ema_macd"],
    ),
    SleeveSpec(
        "ema_macd_iwm",
        "ema_macd",
        "IWM",
        FREQUENCY,
        CONFIGURATION_IDS["ema_macd"],
    ),
)
SLEEVE_IDS: Final[tuple[str, ...]] = tuple(
    sleeve.sleeve_id for sleeve in FROZEN_SLEEVES
)
SLEEVE_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    combinations(SLEEVE_IDS, 2)
)

SLEEVE_INPUT_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "sleeve_id",
    "strategy",
    "symbol",
    "frequency",
    "configuration_id",
    "sessions",
    "observations",
    "start_session",
    "end_session",
    "variance",
    "non_degenerate",
    "finite_returns",
    "exact_calendar_aligned",
)

FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS: Final[
    tuple[str, ...]
] = (
    "sleeve_a",
    "sleeve_b",
    "observations",
    "correlation",
)

FOLD_PAIRWISE_CORRELATION_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "sample",
    "sleeve_a",
    "sleeve_b",
    "observations",
    "correlation",
)

PANEL_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "aligned_sessions",
    "non_degenerate_sleeves",
    "minimum_pairwise_correlation",
    "maximum_pairwise_correlation",
    "maximum_absolute_correlation",
    "mean_absolute_correlation",
    "median_absolute_correlation",
    "minimum_correlation_eigenvalue",
    "maximum_correlation_eigenvalue",
    "pc1_share",
    "effective_rank",
    "minimum_covariance_eigenvalue",
    "maximum_covariance_eigenvalue",
    "covariance_condition_number",
    "correlation_psd",
    "covariance_psd",
    "equal_weight_diversification_ratio",
)

FULL_SAMPLE_COVARIANCE_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = PANEL_DIAGNOSTIC_COLUMNS

FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "sample",
    *PANEL_DIAGNOSTIC_COLUMNS,
)

ENSEMBLE_FEASIBILITY_COLUMNS: Final[tuple[str, ...]] = (
    "finite_returns_gate",
    "exact_calendar_alignment_gate",
    "non_degenerate_sleeves_gate",
    "minimum_training_sessions_gate",
    "minimum_test_sessions_gate",
    "maximum_absolute_correlation_gate",
    "median_effective_rank_gate",
    "median_pc1_share_gate",
    "correlation_psd_gate",
    "covariance_psd_gate",
    "realised_test_diversification_gate",
    "maximum_training_absolute_correlation",
    "median_training_effective_rank",
    "median_training_pc1_share",
    "median_test_equal_weight_diversification_ratio",
    "ensemble_feasible",
)


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Return a defensive copy of a result table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    result = frame.copy(deep=True)
    if reset_index:
        result = result.reset_index(drop=True)
    return result


@dataclass(frozen=True, slots=True)
class StrategyDiversificationInputs:
    """Defensively retained six-sleeve session-return inputs."""

    sleeve_session_returns: pd.DataFrame
    session_return_panel: pd.DataFrame
    sleeve_input_diagnostics: pd.DataFrame

    def __post_init__(self) -> None:
        """Retain copies so caller mutation cannot change the inputs."""

        object.__setattr__(
            self,
            "sleeve_session_returns",
            _copy_frame(
                self.sleeve_session_returns,
                name="sleeve_session_returns",
            ),
        )
        panel = _copy_frame(
            self.session_return_panel,
            name="session_return_panel",
            reset_index=False,
        )
        panel.index = panel.index.copy()
        object.__setattr__(self, "session_return_panel", panel)
        object.__setattr__(
            self,
            "sleeve_input_diagnostics",
            _copy_frame(
                self.sleeve_input_diagnostics,
                name="sleeve_input_diagnostics",
            ),
        )

    def copy_sleeve_session_returns(self) -> pd.DataFrame:
        return self.sleeve_session_returns.copy(deep=True)

    def copy_session_return_panel(self) -> pd.DataFrame:
        return self.session_return_panel.copy(deep=True)

    def copy_sleeve_input_diagnostics(self) -> pd.DataFrame:
        return self.sleeve_input_diagnostics.copy(deep=True)


@dataclass(frozen=True, slots=True)
class StrategyDiversificationResults:
    """Defensively retained Day 15 diagnostic tables."""

    session_return_panel: pd.DataFrame
    sleeve_input_diagnostics: pd.DataFrame
    full_sample_pairwise_correlations: pd.DataFrame
    full_sample_covariance_diagnostics: pd.DataFrame
    fold_pairwise_correlations: pd.DataFrame
    fold_covariance_diagnostics: pd.DataFrame
    ensemble_feasibility: pd.DataFrame

    def __post_init__(self) -> None:
        """Retain independent copies of all result tables."""

        for name in (
            "session_return_panel",
            "sleeve_input_diagnostics",
            "full_sample_pairwise_correlations",
            "full_sample_covariance_diagnostics",
            "fold_pairwise_correlations",
            "fold_covariance_diagnostics",
            "ensemble_feasibility",
        ):
            reset_index = name != "session_return_panel"
            retained = _copy_frame(
                getattr(self, name),
                name=name,
                reset_index=reset_index,
            )
            if name == "session_return_panel":
                retained.index = retained.index.copy()
            object.__setattr__(self, name, retained)

    def copy_session_return_panel(self) -> pd.DataFrame:
        return self.session_return_panel.copy(deep=True)

    def copy_sleeve_input_diagnostics(self) -> pd.DataFrame:
        return self.sleeve_input_diagnostics.copy(deep=True)

    def copy_full_sample_pairwise_correlations(
        self,
    ) -> pd.DataFrame:
        return self.full_sample_pairwise_correlations.copy(deep=True)

    def copy_full_sample_covariance_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.full_sample_covariance_diagnostics.copy(deep=True)

    def copy_fold_pairwise_correlations(self) -> pd.DataFrame:
        return self.fold_pairwise_correlations.copy(deep=True)

    def copy_fold_covariance_diagnostics(self) -> pd.DataFrame:
        return self.fold_covariance_diagnostics.copy(deep=True)

    def copy_ensemble_feasibility(self) -> pd.DataFrame:
        return self.ensemble_feasibility.copy(deep=True)


def build_frozen_sleeves() -> tuple[SleeveSpec, ...]:
    """Return the deterministic six-sleeve contract."""

    if len(FROZEN_SLEEVES) != 6 or len(set(SLEEVE_IDS)) != 6:
        raise RuntimeError("Day 15 must contain six unique sleeves.")
    if len(SLEEVE_PAIRS) != 15 or len(set(SLEEVE_PAIRS)) != 15:
        raise RuntimeError("Six sleeves must produce 15 unique pairs.")
    return FROZEN_SLEEVES


def _validate_frozen_contract() -> None:
    """Reject runtime changes to inherited Day 10/11 contracts."""

    if TREND_RATIO_PARAMETERS != DAY10_TREND_RATIO_PARAMETERS:
        raise StrategyDiversificationError(
            "Trend Ratio parameters must remain frozen at Day 10."
        )
    if EMA_MACD_PARAMETERS != DAY10_EMA_MACD_PARAMETERS:
        raise StrategyDiversificationError(
            "EMA/MACD parameters must remain frozen at Day 10."
        )
    if CONFIGURATION_IDS != DAY10_CONFIGURATION_IDS:
        raise StrategyDiversificationError(
            "Configuration identifiers must remain frozen at Day 10."
        )
    folds = build_walk_forward_folds()
    if len(folds) != 4:
        raise StrategyDiversificationError(
            "Day 15 requires the four frozen Day 11 folds."
        )
    build_frozen_sleeves()


def _normalize_development_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate the development-only canonical three-symbol input."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")
    if bars.empty:
        raise StrategyDiversificationError("bars must not be empty.")

    missing = sorted(set(CANONICAL_BAR_COLUMNS).difference(bars.columns))
    if missing:
        raise StrategyDiversificationError(
            "Development bars are missing required columns: "
            f"{missing}."
        )

    result = bars.copy(deep=True)
    try:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
            format="mixed",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyDiversificationError(
            "Development bars contain malformed timestamps."
        ) from exc

    if result["timestamp"].isna().any():
        raise StrategyDiversificationError(
            "Development timestamps cannot be missing."
        )

    normalized_symbols = (
        result["symbol"].astype("string").str.strip().str.upper()
    )
    if (
        normalized_symbols.isna().any()
        or normalized_symbols.eq("").any()
        or not result["symbol"].astype("string").eq(normalized_symbols).all()
    ):
        raise StrategyDiversificationError(
            "Development symbols must be non-missing canonical uppercase "
            "identifiers."
        )

    actual_symbols = set(normalized_symbols.astype(str))
    expected_symbols = set(REQUIRED_SYMBOLS)
    if actual_symbols != expected_symbols:
        raise StrategyDiversificationError(
            "Development bars must contain exactly SPY, QQQ, and IWM. "
            f"Actual symbols: {sorted(actual_symbols)}."
        )

    local_dates = (
        result["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    start = DEVELOPMENT_START.tz_localize(None)
    end_exclusive = DEVELOPMENT_END_EXCLUSIVE.tz_localize(None)
    if local_dates.lt(start).any() or local_dates.ge(end_exclusive).any():
        raise StrategyDiversificationError(
            "Observations must remain within development dates "
            "2020-01-02 through 2025-12-31; 2026 data are forbidden."
        )

    if "bar_frequency" in result.columns:
        frequencies = set(
            result["bar_frequency"].astype("string").dropna().astype(str)
        )
        if frequencies != {FREQUENCY}:
            raise StrategyDiversificationError(
                "Day 15 accepts 15min source bars only."
            )

    return result


def _prepare_return_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate/validate at 15 minutes and build return features."""

    scoped = _normalize_development_bars(bars)
    try:
        aggregated = aggregate_session_bars(scoped, FREQUENCY)
    except SessionAggregationError as exc:
        raise StrategyDiversificationError(
            f"15min session validation failed: {exc}"
        ) from exc

    try:
        return build_return_features(
            aggregated,
            expected_symbols=REQUIRED_SYMBOLS,
        ).bars
    except (TypeError, ValueError) as exc:
        raise StrategyDiversificationError(
            f"Return-feature construction failed: {exc}"
        ) from exc


def _validate_position_delay(observations: pd.DataFrame) -> None:
    """Require position[t] to equal signal[t-1] within one sleeve."""

    required = {"signal", "position"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise StrategyDiversificationError(
            f"Strategy observations are missing delay columns: {missing}."
        )

    signal = pd.to_numeric(observations["signal"], errors="coerce")
    position = pd.to_numeric(observations["position"], errors="coerce")
    if signal.isna().any() or position.isna().any():
        raise StrategyDiversificationError(
            "Strategy signals and positions must be finite."
        )

    expected = signal.shift(1, fill_value=0)
    if not position.eq(expected).all():
        raise StrategyDiversificationError(
            "Each sleeve must preserve position[t] == signal[t-1]."
        )


def _execute_sleeve(
    features: pd.DataFrame,
    sleeve: SleeveSpec,
) -> pd.DataFrame:
    """Execute one frozen strategy against one symbol only."""

    symbol_features = (
        features.loc[features["symbol"].eq(sleeve.symbol)]
        .copy(deep=True)
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    if symbol_features.empty:
        raise StrategyDiversificationError(
            f"{sleeve.sleeve_id} has no feature observations."
        )

    if sleeve.strategy == "trend_ratio":
        bundle = build_trend_ratio_strategy(
            symbol_features,
            parameters=TREND_RATIO_PARAMETERS,
        )
    elif sleeve.strategy == "ema_macd":
        bundle = build_ema_macd_strategy(
            symbol_features,
            parameters=EMA_MACD_PARAMETERS,
        )
    else:
        raise RuntimeError(f"Unexpected sleeve strategy: {sleeve.strategy}.")

    observations = bundle.observations.copy(deep=True)
    if len(observations) != len(symbol_features):
        raise RuntimeError("A frozen strategy changed the observation count.")
    if observations["symbol"].nunique() != 1 or not observations[
        "symbol"
    ].eq(sleeve.symbol).all():
        raise StrategyDiversificationError(
            "A sleeve execution crossed symbol boundaries."
        )

    _validate_position_delay(observations)
    return observations


def compound_intraday_returns(returns: pd.Series) -> float:
    """Compound validated simple intraday net returns into one session."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if returns.empty:
        raise StrategyDiversificationError(
            "A session cannot contain zero intraday returns."
        )
    try:
        numeric = pd.to_numeric(returns.copy(deep=True), errors="raise")
    except (TypeError, ValueError) as exc:
        raise StrategyDiversificationError(
            "Intraday net returns must be numeric."
        ) from exc

    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise StrategyDiversificationError(
            "Intraday net returns must be finite and non-missing."
        )
    if np.less_equal(values, -1.0).any():
        raise StrategyDiversificationError(
            "Simple intraday net returns must be greater than -1."
        )

    result = float(np.prod(1.0 + values) - 1.0)
    if not math.isfinite(result) or result <= -1.0:
        raise StrategyDiversificationError(
            "Compounded session return is invalid."
        )
    return result


def _sleeve_session_frame(
    observations: pd.DataFrame,
    sleeve: SleeveSpec,
) -> pd.DataFrame:
    """Build one sleeve's exact session returns from existing net returns."""

    required = {"session_date", "net_strategy_return"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise StrategyDiversificationError(
            f"Strategy observations are missing return columns: {missing}."
        )

    working = observations[["session_date", "net_strategy_return"]].copy()
    try:
        working["session_date"] = pd.to_datetime(
            working["session_date"],
            utc=True,
            errors="raise",
            format="mixed",
        ).dt.normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyDiversificationError(
            "Strategy observations contain malformed session dates."
        ) from exc
    if working["session_date"].isna().any():
        raise StrategyDiversificationError(
            "Strategy session dates cannot be missing."
        )

    working["net_strategy_return"] = pd.to_numeric(
        working["net_strategy_return"],
        errors="coerce",
    )
    values = working["net_strategy_return"].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise StrategyDiversificationError(
            "net_strategy_return must be finite and non-missing."
        )
    if np.less_equal(values, -1.0).any():
        raise StrategyDiversificationError(
            "net_strategy_return must be greater than -1."
        )

    grouped = working.groupby(
        "session_date",
        observed=True,
        sort=True,
    )
    records = [
        {
            "sleeve_id": sleeve.sleeve_id,
            "strategy": sleeve.strategy,
            "symbol": sleeve.symbol,
            "frequency": sleeve.frequency,
            "configuration_id": sleeve.configuration_id,
            "session_date": session_date,
            "session_return": compound_intraday_returns(
                group["net_strategy_return"]
            ),
            "observations": int(len(group)),
        }
        for session_date, group in grouped
    ]
    return pd.DataFrame.from_records(records)


def _normalize_session_return_rows(
    sleeve_session_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Validate long-form sleeve/session returns without filling gaps."""

    if not isinstance(sleeve_session_returns, pd.DataFrame):
        raise TypeError("sleeve_session_returns must be a pandas DataFrame.")
    if sleeve_session_returns.empty:
        raise StrategyDiversificationError(
            "sleeve_session_returns must not be empty."
        )

    required = {"sleeve_id", "session_date", "session_return"}
    missing = sorted(required.difference(sleeve_session_returns.columns))
    if missing:
        raise StrategyDiversificationError(
            "Sleeve/session returns are missing required columns: "
            f"{missing}."
        )

    result = sleeve_session_returns.copy(deep=True)
    normalized_ids = result["sleeve_id"].astype("string").str.strip()
    if normalized_ids.isna().any() or normalized_ids.eq("").any():
        raise StrategyDiversificationError(
            "sleeve_id cannot be missing or empty."
        )
    actual_ids = set(normalized_ids.astype(str))
    if actual_ids != set(SLEEVE_IDS):
        raise StrategyDiversificationError(
            "Sleeve/session returns must contain exactly the six frozen "
            f"sleeves. Actual sleeves: {sorted(actual_ids)}."
        )
    if not result["sleeve_id"].astype("string").eq(normalized_ids).all():
        raise StrategyDiversificationError(
            "sleeve_id values must already be canonical."
        )

    try:
        result["session_date"] = pd.to_datetime(
            result["session_date"],
            utc=True,
            errors="raise",
            format="mixed",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyDiversificationError(
            "Sleeve/session returns contain malformed dates."
        ) from exc
    if result["session_date"].isna().any():
        raise StrategyDiversificationError(
            "Sleeve/session dates cannot be missing."
        )
    if not result["session_date"].eq(result["session_date"].dt.normalize()).all():
        raise StrategyDiversificationError(
            "session_date values must be normalized whole dates."
        )
    if (
        result["session_date"].lt(DEVELOPMENT_START).any()
        or result["session_date"].ge(DEVELOPMENT_END_EXCLUSIVE).any()
    ):
        raise StrategyDiversificationError(
            "Sleeve/session returns must remain within 2020-01-02 through "
            "2025-12-31; 2026 data are forbidden."
        )
    if result.duplicated(["sleeve_id", "session_date"]).any():
        raise StrategyDiversificationError(
            "Duplicate sleeve/session rows are forbidden."
        )

    try:
        numeric = pd.to_numeric(result["session_return"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise StrategyDiversificationError(
            "Session returns must be numeric."
        ) from exc
    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise StrategyDiversificationError(
            "Session returns must be finite and non-missing."
        )
    if np.less_equal(values, -1.0).any():
        raise StrategyDiversificationError(
            "Simple session returns must be greater than -1."
        )
    result["session_return"] = numeric.astype("float64")

    calendars = {
        sleeve_id: tuple(group["session_date"].sort_values())
        for sleeve_id, group in result.groupby(
            "sleeve_id",
            observed=True,
            sort=False,
        )
    }
    if len(set(calendars.values())) != 1:
        raise StrategyDiversificationError(
            "All six sleeves must use exactly the same session calendar; "
            "filling and interpolation are forbidden."
        )

    return result


def build_exact_return_panel(
    sleeve_session_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build the exact common-date six-column return panel."""

    normalized = _normalize_session_return_rows(sleeve_session_returns)
    panel = normalized.pivot(
        index="session_date",
        columns="sleeve_id",
        values="session_return",
    )
    panel = panel.loc[:, list(SLEEVE_IDS)].sort_index(kind="stable")
    panel.columns.name = None
    panel.index.name = "session_date"

    return validate_session_return_panel(panel)


def _coerce_session_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a panel's session index without filling observations."""

    result = frame.copy(deep=True)
    if "session_date" in result.columns:
        session_values = result.pop("session_date")
    else:
        session_values = result.index

    if not isinstance(session_values, (pd.Series, pd.Index)):
        raise StrategyDiversificationError(
            "Return panel requires a session_date index."
        )
    if pd.api.types.is_numeric_dtype(session_values.dtype):
        raise StrategyDiversificationError(
            "Return panel session dates are malformed."
        )
    try:
        sessions = pd.to_datetime(
            session_values,
            utc=True,
            errors="raise",
            format="mixed",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyDiversificationError(
            "Return panel contains malformed session dates."
        ) from exc

    sessions = pd.DatetimeIndex(sessions, name="session_date")
    if sessions.isna().any():
        raise StrategyDiversificationError(
            "Return panel session dates cannot be missing."
        )
    if not sessions.equals(sessions.normalize()):
        raise StrategyDiversificationError(
            "Return panel sessions must be normalized whole dates."
        )
    if sessions.duplicated().any():
        raise StrategyDiversificationError(
            "Duplicate sleeve/session rows are forbidden."
        )
    if (
        sessions.min() < DEVELOPMENT_START
        or sessions.max() >= DEVELOPMENT_END_EXCLUSIVE
    ):
        raise StrategyDiversificationError(
            "Return panel must remain within 2020-01-02 through "
            "2025-12-31; 2026 data are forbidden."
        )

    result.index = sessions
    return result.sort_index(kind="stable")


def validate_session_return_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy an exact common-date six-sleeve panel."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")
    if panel.empty:
        raise StrategyDiversificationError("panel must not be empty.")

    result = _coerce_session_index(panel)
    actual_columns = set(result.columns)
    expected_columns = set(SLEEVE_IDS)
    if actual_columns != expected_columns or len(result.columns) != 6:
        missing = sorted(expected_columns.difference(actual_columns))
        unexpected = sorted(actual_columns.difference(expected_columns))
        raise StrategyDiversificationError(
            "Return panel must contain exactly the six frozen sleeve "
            f"columns. Missing: {missing}; unexpected: {unexpected}."
        )
    result = result.loc[:, list(SLEEVE_IDS)]

    for column in SLEEVE_IDS:
        try:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            ).astype("float64")
        except (TypeError, ValueError) as exc:
            raise StrategyDiversificationError(
                f"Return panel column {column} must be numeric."
            ) from exc

    values = result.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise StrategyDiversificationError(
            "Return panel cannot contain missing or non-finite returns; "
            "filling and interpolation are forbidden."
        )
    if np.less_equal(values, -1.0).any():
        raise StrategyDiversificationError(
            "All simple returns must be greater than -1."
        )
    if len(result) < 2:
        raise StrategyDiversificationError(
            "At least two aligned sessions are required."
        )

    variances = result.var(axis=0, ddof=1)
    if (
        not np.isfinite(variances.to_numpy(dtype="float64")).all()
        or variances.le(VARIANCE_TOLERANCE).any()
    ):
        failed = variances.index[
            variances.le(VARIANCE_TOLERANCE)
            | ~np.isfinite(variances)
        ].tolist()
        raise StrategyDiversificationError(
            "Zero or near-zero variance sleeves are forbidden. "
            f"Failed sleeves: {failed}."
        )

    return result


def _build_sleeve_input_diagnostics(
    sleeve_session_returns: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize validated input provenance and degeneracy."""

    normalized = _normalize_session_return_rows(sleeve_session_returns)
    common_calendar = tuple(panel.index)
    records: list[dict[str, object]] = []

    for sleeve in FROZEN_SLEEVES:
        rows = normalized.loc[
            normalized["sleeve_id"].eq(sleeve.sleeve_id)
        ].sort_values("session_date", kind="stable")
        returns = rows["session_return"].to_numpy(dtype="float64")
        variance = float(np.var(returns, ddof=1))
        observations = (
            int(rows["observations"].sum())
            if "observations" in rows.columns
            else int(len(rows))
        )
        finite = bool(np.isfinite(returns).all())
        exact_calendar = tuple(rows["session_date"]) == common_calendar

        records.append(
            {
                "sleeve_id": sleeve.sleeve_id,
                "strategy": sleeve.strategy,
                "symbol": sleeve.symbol,
                "frequency": sleeve.frequency,
                "configuration_id": sleeve.configuration_id,
                "sessions": int(len(rows)),
                "observations": observations,
                "start_session": rows["session_date"].min(),
                "end_session": rows["session_date"].max(),
                "variance": variance,
                "non_degenerate": bool(
                    math.isfinite(variance)
                    and variance > VARIANCE_TOLERANCE
                ),
                "finite_returns": finite,
                "exact_calendar_aligned": exact_calendar,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
    )


def build_strategy_diversification_inputs(
    bars: pd.DataFrame,
) -> StrategyDiversificationInputs:
    """Execute all frozen sleeves and build exact session-return inputs."""

    _validate_frozen_contract()
    features = _prepare_return_features(bars)
    frames = [
        _sleeve_session_frame(
            _execute_sleeve(features, sleeve),
            sleeve,
        )
        for sleeve in FROZEN_SLEEVES
    ]
    sleeve_session_returns = pd.concat(frames, ignore_index=True)
    panel = build_exact_return_panel(sleeve_session_returns)
    diagnostics = _build_sleeve_input_diagnostics(
        sleeve_session_returns,
        panel,
    )
    return StrategyDiversificationInputs(
        sleeve_session_returns=sleeve_session_returns,
        session_return_panel=panel,
        sleeve_input_diagnostics=diagnostics,
    )


def calculate_pairwise_correlations(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate all 15 unique unordered Pearson correlations."""

    validated = validate_session_return_panel(panel)
    correlation = validated.corr(method="pearson")
    records = [
        {
            "sleeve_a": sleeve_a,
            "sleeve_b": sleeve_b,
            "observations": int(len(validated)),
            "correlation": float(correlation.loc[sleeve_a, sleeve_b]),
        }
        for sleeve_a, sleeve_b in SLEEVE_PAIRS
    ]
    result = pd.DataFrame.from_records(
        records,
        columns=FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS,
    )
    if len(result) != 15 or result[
        ["sleeve_a", "sleeve_b"]
    ].duplicated().any():
        raise RuntimeError("Pairwise correlations must contain 15 pairs.")
    if not np.isfinite(result["correlation"].to_numpy(dtype=float)).all():
        raise StrategyDiversificationError(
            "Pearson correlations must be finite."
        )
    return result


def calculate_correlation_eigenvalues(
    panel: pd.DataFrame,
) -> np.ndarray:
    """Return correlation eigenvalues in descending order."""

    validated = validate_session_return_panel(panel)
    matrix = validated.corr(method="pearson").to_numpy(dtype="float64")
    return np.linalg.eigvalsh(matrix)[::-1]


def calculate_sample_covariance(
    panel: pd.DataFrame,
) -> np.ndarray:
    """Return ordinary sample covariance using ddof=1."""

    validated = validate_session_return_panel(panel)
    return np.cov(
        validated.to_numpy(dtype="float64"),
        rowvar=False,
        ddof=1,
    )


def calculate_covariance_eigenvalues(
    panel: pd.DataFrame,
) -> np.ndarray:
    """Return ordinary sample-covariance eigenvalues, descending."""

    covariance = calculate_sample_covariance(panel)
    return np.linalg.eigvalsh(covariance)[::-1]


def calculate_entropy_effective_rank(
    descending_eigenvalues: np.ndarray,
) -> float:
    """Calculate entropy effective rank, ignoring zero probabilities."""

    values = np.asarray(descending_eigenvalues, dtype="float64")
    if values.ndim != 1 or values.size == 0:
        raise StrategyDiversificationError(
            "Eigenvalues must be a non-empty one-dimensional array."
        )
    if not np.isfinite(values).all():
        raise StrategyDiversificationError("Eigenvalues must be finite.")

    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    probabilities = clipped / total
    positive = probabilities > np.finfo("float64").eps
    entropy = -float(
        np.sum(
            probabilities[positive]
            * np.log(probabilities[positive])
        )
    )
    return float(np.exp(entropy))


def is_positive_semidefinite(
    eigenvalues: np.ndarray,
) -> bool:
    """Apply the documented absolute PSD eigenvalue tolerance."""

    values = np.asarray(eigenvalues, dtype="float64")
    if values.ndim != 1 or values.size == 0:
        raise StrategyDiversificationError(
            "Eigenvalues must be a non-empty one-dimensional array."
        )
    if not np.isfinite(values).all():
        return False
    return bool(values.min() >= -PSD_TOLERANCE)


def calculate_covariance_condition_number(
    covariance: np.ndarray,
) -> float:
    """Return the spectral condition number, or +inf when singular."""

    matrix = np.asarray(covariance, dtype="float64")
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
    ):
        raise StrategyDiversificationError(
            "covariance must be a non-empty square matrix."
        )
    if not np.isfinite(matrix).all():
        raise StrategyDiversificationError(
            "covariance must contain finite values."
        )

    if not np.array_equal(matrix, matrix.T):
        raise StrategyDiversificationError(
            "covariance must be exactly symmetric; repair is forbidden."
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.linalg.matrix_rank(matrix) < len(matrix):
        return float("inf")
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    if minimum <= 0.0:
        return float("inf")
    return maximum / minimum


def calculate_equal_weight_diversification_ratio(
    covariance: np.ndarray,
) -> float:
    """Calculate the six-sleeve equal-weight diversification ratio."""

    matrix = np.asarray(covariance, dtype="float64")
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise StrategyDiversificationError(
            "Equal-weight diversification requires a finite 6x6 "
            "covariance matrix."
        )

    variances = np.diag(matrix)
    if np.less_equal(variances, VARIANCE_TOLERANCE).any():
        raise StrategyDiversificationError(
            "Diversification ratio requires non-degenerate sleeves."
        )

    weights = np.full(6, 1.0 / 6.0, dtype="float64")
    numerator = float(weights @ np.sqrt(variances))
    portfolio_variance = float(weights @ matrix @ weights)
    if portfolio_variance < -PSD_TOLERANCE:
        raise StrategyDiversificationError(
            "Portfolio variance is materially negative."
        )
    if portfolio_variance <= VARIANCE_TOLERANCE:
        return float("inf")
    return numerator / math.sqrt(portfolio_variance)


def calculate_panel_diagnostics(
    panel: pd.DataFrame,
) -> dict[str, object]:
    """Calculate concentration, spectrum, PSD, and DR diagnostics."""

    validated = validate_session_return_panel(panel)
    pairwise = calculate_pairwise_correlations(validated)
    pair_values = pairwise["correlation"].to_numpy(dtype="float64")

    correlation_eigenvalues = calculate_correlation_eigenvalues(validated)
    covariance = calculate_sample_covariance(validated)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    correlation_sum = float(correlation_eigenvalues.sum())
    if correlation_sum <= 0.0:
        raise StrategyDiversificationError(
            "Correlation eigenvalue sum must be positive."
        )

    variances = validated.var(axis=0, ddof=1)
    return {
        "aligned_sessions": int(len(validated)),
        "non_degenerate_sleeves": int(
            variances.gt(VARIANCE_TOLERANCE).sum()
        ),
        "minimum_pairwise_correlation": float(pair_values.min()),
        "maximum_pairwise_correlation": float(pair_values.max()),
        "maximum_absolute_correlation": float(
            np.abs(pair_values).max()
        ),
        "mean_absolute_correlation": float(
            np.abs(pair_values).mean()
        ),
        "median_absolute_correlation": float(
            np.median(np.abs(pair_values))
        ),
        "minimum_correlation_eigenvalue": float(
            correlation_eigenvalues.min()
        ),
        "maximum_correlation_eigenvalue": float(
            correlation_eigenvalues.max()
        ),
        "pc1_share": float(
            correlation_eigenvalues.max() / correlation_sum
        ),
        "effective_rank": calculate_entropy_effective_rank(
            correlation_eigenvalues
        ),
        "minimum_covariance_eigenvalue": float(
            covariance_eigenvalues.min()
        ),
        "maximum_covariance_eigenvalue": float(
            covariance_eigenvalues.max()
        ),
        "covariance_condition_number": (
            calculate_covariance_condition_number(covariance)
        ),
        "correlation_psd": is_positive_semidefinite(
            correlation_eigenvalues
        ),
        "covariance_psd": is_positive_semidefinite(
            covariance_eigenvalues
        ),
        "equal_weight_diversification_ratio": (
            calculate_equal_weight_diversification_ratio(covariance)
        ),
    }


def _partition_panel(
    panel: pd.DataFrame,
) -> tuple[tuple[str, str, pd.DataFrame], ...]:
    """Partition all four frozen folds into train and test panels."""

    partitions: list[tuple[str, str, pd.DataFrame]] = []
    for fold in build_walk_forward_folds():
        train = panel.loc[
            (panel.index >= fold.train_start)
            & (panel.index < fold.train_end_exclusive)
        ]
        test = panel.loc[
            (panel.index >= fold.test_start)
            & (panel.index < fold.test_end_exclusive)
        ]
        if len(train) < MIN_TRAINING_SESSIONS:
            raise StrategyDiversificationError(
                f"{fold.fold_id} requires at least "
                f"{MIN_TRAINING_SESSIONS} training sessions."
            )
        if len(test) < MIN_TEST_SESSIONS:
            raise StrategyDiversificationError(
                f"{fold.fold_id} requires at least "
                f"{MIN_TEST_SESSIONS} test sessions."
            )
        partitions.extend(
            (
                (fold.fold_id, "train", train.copy(deep=True)),
                (fold.fold_id, "test", test.copy(deep=True)),
            )
        )
    return tuple(partitions)


def _fold_pairwise_table(
    partitions: tuple[tuple[str, str, pd.DataFrame], ...],
) -> pd.DataFrame:
    """Calculate the frozen 4 x 2 x 15 pairwise table."""

    records: list[dict[str, object]] = []
    for fold_id, sample, panel in partitions:
        for record in calculate_pairwise_correlations(panel).to_dict(
            "records"
        ):
            records.append(
                {
                    "fold_id": fold_id,
                    "sample": sample,
                    **record,
                }
            )

    result = pd.DataFrame.from_records(
        records,
        columns=FOLD_PAIRWISE_CORRELATION_COLUMNS,
    )
    if len(result) != 120:
        raise RuntimeError(
            "Fold pairwise correlations must contain exactly 120 rows."
        )
    return result


def _fold_covariance_table(
    partitions: tuple[tuple[str, str, pd.DataFrame], ...],
) -> pd.DataFrame:
    """Calculate the frozen 4 x 2 covariance diagnostic table."""

    records = [
        {
            "fold_id": fold_id,
            "sample": sample,
            **calculate_panel_diagnostics(panel),
        }
        for fold_id, sample, panel in partitions
    ]
    result = pd.DataFrame.from_records(
        records,
        columns=FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS,
    )
    if len(result) != 8:
        raise RuntimeError(
            "Fold covariance diagnostics must contain exactly eight rows."
        )
    return result


def _ensemble_feasibility_table(
    sleeve_input_diagnostics: pd.DataFrame,
    full_sample_covariance_diagnostics: pd.DataFrame,
    fold_covariance_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Apply predeclared ensemble gates without ranking or optimization."""

    training = fold_covariance_diagnostics.loc[
        fold_covariance_diagnostics["sample"].eq("train")
    ]
    test = fold_covariance_diagnostics.loc[
        fold_covariance_diagnostics["sample"].eq("test")
    ]

    maximum_training_correlation = float(
        training["maximum_absolute_correlation"].max()
    )
    median_effective_rank = float(training["effective_rank"].median())
    median_pc1_share = float(training["pc1_share"].median())
    median_test_dr = float(
        test["equal_weight_diversification_ratio"].median()
    )

    gates = {
        "finite_returns_gate": bool(
            sleeve_input_diagnostics["finite_returns"].astype(bool).all()
        ),
        "exact_calendar_alignment_gate": bool(
            sleeve_input_diagnostics[
                "exact_calendar_aligned"
            ].astype(bool).all()
        ),
        "non_degenerate_sleeves_gate": bool(
            sleeve_input_diagnostics["non_degenerate"].astype(bool).all()
        ),
        "minimum_training_sessions_gate": bool(
            training["aligned_sessions"].ge(MIN_TRAINING_SESSIONS).all()
        ),
        "minimum_test_sessions_gate": bool(
            test["aligned_sessions"].ge(MIN_TEST_SESSIONS).all()
        ),
        "maximum_absolute_correlation_gate": bool(
            maximum_training_correlation <= MAX_ABSOLUTE_CORRELATION
        ),
        "median_effective_rank_gate": bool(
            median_effective_rank >= MIN_MEDIAN_EFFECTIVE_RANK
        ),
        "median_pc1_share_gate": bool(
            median_pc1_share <= MAX_MEDIAN_PC1_SHARE
        ),
        "correlation_psd_gate": bool(
            full_sample_covariance_diagnostics[
                "correlation_psd"
            ].astype(bool).all()
            and fold_covariance_diagnostics[
                "correlation_psd"
            ].astype(bool).all()
        ),
        "covariance_psd_gate": bool(
            full_sample_covariance_diagnostics[
                "covariance_psd"
            ].astype(bool).all()
            and fold_covariance_diagnostics[
                "covariance_psd"
            ].astype(bool).all()
        ),
        "realised_test_diversification_gate": bool(
            median_test_dr
            > MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO
        ),
    }
    ensemble_feasible = bool(all(gates.values()))

    return pd.DataFrame.from_records(
        [
            {
                **gates,
                "maximum_training_absolute_correlation": (
                    maximum_training_correlation
                ),
                "median_training_effective_rank": median_effective_rank,
                "median_training_pc1_share": median_pc1_share,
                "median_test_equal_weight_diversification_ratio": (
                    median_test_dr
                ),
                "ensemble_feasible": ensemble_feasible,
            }
        ],
        columns=ENSEMBLE_FEASIBILITY_COLUMNS,
    )


def _diagnostics_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic-provenance diagnostics for direct panel analysis."""

    records = []
    for sleeve in FROZEN_SLEEVES:
        returns = panel[sleeve.sleeve_id]
        variance = float(returns.var(ddof=1))
        records.append(
            {
                "sleeve_id": sleeve.sleeve_id,
                "strategy": sleeve.strategy,
                "symbol": sleeve.symbol,
                "frequency": sleeve.frequency,
                "configuration_id": sleeve.configuration_id,
                "sessions": int(len(panel)),
                "observations": int(len(panel)),
                "start_session": panel.index.min(),
                "end_session": panel.index.max(),
                "variance": variance,
                "non_degenerate": bool(variance > VARIANCE_TOLERANCE),
                "finite_returns": True,
                "exact_calendar_aligned": True,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
    )


def analyze_strategy_diversification_panel(
    panel: pd.DataFrame,
    *,
    sleeve_input_diagnostics: pd.DataFrame | None = None,
) -> StrategyDiversificationResults:
    """Analyze a validated synthetic or strategy-built return panel."""

    _validate_frozen_contract()
    validated = validate_session_return_panel(panel)
    if sleeve_input_diagnostics is None:
        input_diagnostics = _diagnostics_from_panel(validated)
    else:
        input_diagnostics = _copy_frame(
            sleeve_input_diagnostics,
            name="sleeve_input_diagnostics",
        )
        if tuple(input_diagnostics.columns) != (
            SLEEVE_INPUT_DIAGNOSTIC_COLUMNS
        ):
            raise StrategyDiversificationError(
                "sleeve_input_diagnostics has an unexpected schema."
            )
        if tuple(input_diagnostics["sleeve_id"]) != SLEEVE_IDS:
            raise StrategyDiversificationError(
                "sleeve_input_diagnostics must follow frozen sleeve order."
            )

    full_pairwise = calculate_pairwise_correlations(validated)
    full_covariance = pd.DataFrame.from_records(
        [calculate_panel_diagnostics(validated)],
        columns=FULL_SAMPLE_COVARIANCE_DIAGNOSTIC_COLUMNS,
    )
    partitions = _partition_panel(validated)
    fold_pairwise = _fold_pairwise_table(partitions)
    fold_covariance = _fold_covariance_table(partitions)
    feasibility = _ensemble_feasibility_table(
        input_diagnostics,
        full_covariance,
        fold_covariance,
    )

    return StrategyDiversificationResults(
        session_return_panel=validated,
        sleeve_input_diagnostics=input_diagnostics,
        full_sample_pairwise_correlations=full_pairwise,
        full_sample_covariance_diagnostics=full_covariance,
        fold_pairwise_correlations=fold_pairwise,
        fold_covariance_diagnostics=fold_covariance,
        ensemble_feasibility=feasibility,
    )


def run_strategy_diversification(
    bars: pd.DataFrame,
) -> StrategyDiversificationResults:
    """Run Day 15 Phase 1 on canonical development-only bars."""

    inputs = build_strategy_diversification_inputs(bars)
    return analyze_strategy_diversification_panel(
        inputs.session_return_panel,
        sleeve_input_diagnostics=inputs.sleeve_input_diagnostics,
    )
