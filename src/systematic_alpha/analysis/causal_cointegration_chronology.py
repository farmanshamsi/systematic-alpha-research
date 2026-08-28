"""Causal chronology for the frozen Day 14 cointegration design.

The historical Day 14 path compares expanding-fold hedge ratios with an
all-development estimate.  This independent module replaces that look-ahead
comparison with successive expanding-training estimates while retaining the
frozen pair orientation, statistical conventions, thresholds, and gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from systematic_alpha.analysis.cointegration_feasibility import (
    CANDIDATE_PAIRS,
    MAXIMUM_BETA,
    MAXIMUM_BETA_RELATIVE_DEVIATION,
    MAXIMUM_HALF_LIFE_BARS,
    MINIMUM_BETA,
    MINIMUM_HALF_LIFE_BARS,
    MINIMUM_STATIONARY_FOLDS,
    MULTIPLE_TESTING_METHOD,
    PAIR_IDS,
    SIGNIFICANCE_LEVEL,
    CointegrationInputs,
    PairCointegrationInput,
    _adf_result as _frozen_adf_result,
)
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds


METHOD_VERSION: Final[str] = "causal_cointegration_chronology_v1_1_svd_ols"
DEVELOPMENT_END_EXCLUSIVE: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-01", tz="UTC"
)
REFERENCE_BETA_ZERO_TOLERANCE: Final[float] = float(
    64.0 * np.finfo("float64").eps
)

CHRONOLOGY_LEDGER_COLUMNS: Final[tuple[str, ...]] = (
    "pair_id",
    "pair_order",
    "y_symbol",
    "x_symbol",
    "fold_id",
    "fold_order",
    "train_start",
    "train_end_inclusive",
    "train_end_exclusive",
    "test_start",
    "test_end_exclusive",
    "maximum_timestamp_used_for_estimation",
    "train_observations",
    "alpha_estimate",
    "beta_estimate",
    "design_rank",
    "largest_singular_value",
    "smallest_singular_value",
    "design_condition_number",
    "beta_interpretable",
    "reference_fold",
    "reference_beta",
    "reference_maximum_information_timestamp",
    "relative_beta_drift",
    "beta_sign_change",
    "stability_threshold",
    "stability_status",
    "stability_pass",
    "y_plausibly_i1",
    "x_plausibly_i1",
    "engle_granger_p_value",
    "holm_adjusted_p_value",
    "holm_cointegration_pass",
    "residual_adf_regression",
    "residual_adf_statistic",
    "residual_adf_p_value",
    "residual_adf_used_lag",
    "residual_stationary",
    "fold_eligibility",
    "method_version",
)

PAIR_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "pair_id",
    "pair_order",
    "y_symbol",
    "x_symbol",
    "causal_comparisons",
    "maximum_causal_relative_drift",
    "mean_causal_relative_drift",
    "any_beta_sign_change",
    "causal_stability_pass",
    "stationary_fold_count",
    "fold_stationarity_pass",
    "final_fold_id",
    "final_fold_y_plausibly_i1",
    "final_fold_x_plausibly_i1",
    "final_fold_holm_cointegration_pass",
    "final_fold_beta_pass",
    "causal_ou_attempted",
    "causal_ou_pass",
    "causal_ou_rejection_reason",
    "final_pair_eligibility",
    "rejection_reasons",
    "maximum_information_timestamp",
    "method_version",
)

EX_POST_BETA_COLUMNS: Final[tuple[str, ...]] = (
    "pair_id",
    "pair_order",
    "y_symbol",
    "x_symbol",
    "observations",
    "maximum_information_timestamp",
    "alpha_ex_post",
    "beta_ex_post",
    "statistic_role",
    "used_as_fold_reference",
    "used_in_stability_gate",
    "used_in_eligibility",
    "method_version",
)


class CausalCointegrationChronologyError(ValueError):
    """Raised when causal chronology cannot be established safely."""


@dataclass(frozen=True, slots=True)
class CointegratingRegressionEstimate:
    """One intercept-plus-slope OLS estimate and its residual path."""

    alpha: float
    beta: float
    residuals: pd.Series
    observations: int
    design_rank: int
    largest_singular_value: float
    smallest_singular_value: float
    design_condition_number: float

    def __post_init__(self) -> None:
        if not isinstance(self.residuals, pd.Series):
            raise TypeError("residuals must be a pandas Series.")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "residuals", self.residuals.copy(deep=True))
        object.__setattr__(self, "observations", int(self.observations))
        object.__setattr__(self, "design_rank", int(self.design_rank))
        object.__setattr__(
            self,
            "largest_singular_value",
            float(self.largest_singular_value),
        )
        object.__setattr__(
            self,
            "smallest_singular_value",
            float(self.smallest_singular_value),
        )
        object.__setattr__(
            self,
            "design_condition_number",
            float(self.design_condition_number),
        )

    def copy_residuals(self) -> pd.Series:
        return self.residuals.copy(deep=True)


@dataclass(frozen=True, slots=True)
class BetaStabilityComparison:
    """One successive-beta comparison without denominator stabilization."""

    relative_beta_drift: float
    beta_sign_change: bool
    stability_status: str
    stability_pass: bool


@dataclass(frozen=True, slots=True)
class CausalCointegrationChronologyResults:
    """Defensively retained causal ledger, pair summary, and ex-post table."""

    fold_chronology: pd.DataFrame
    pair_summaries: pd.DataFrame
    ex_post_beta_diagnostics: pd.DataFrame
    method_version: str = METHOD_VERSION

    def __post_init__(self) -> None:
        for name, columns in (
            ("fold_chronology", CHRONOLOGY_LEDGER_COLUMNS),
            ("pair_summaries", PAIR_SUMMARY_COLUMNS),
            ("ex_post_beta_diagnostics", EX_POST_BETA_COLUMNS),
        ):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            if tuple(value.columns) != columns:
                raise CausalCointegrationChronologyError(
                    f"{name} schema changed."
                )
            object.__setattr__(
                self,
                name,
                value.copy(deep=True).reset_index(drop=True),
            )
        if self.method_version != METHOD_VERSION:
            raise CausalCointegrationChronologyError(
                "Causal cointegration method version changed."
            )

    def copy_fold_chronology(self) -> pd.DataFrame:
        return self.fold_chronology.copy(deep=True)

    def copy_pair_summaries(self) -> pd.DataFrame:
        return self.pair_summaries.copy(deep=True)

    def copy_ex_post_beta_diagnostics(self) -> pd.DataFrame:
        return self.ex_post_beta_diagnostics.copy(deep=True)


def estimate_cointegrating_regression(
    frame: pd.DataFrame,
) -> CointegratingRegressionEstimate:
    """Estimate gamma by rank-checked SVD least squares for Z=[1,x]."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    required = ("y_log_price", "x_log_price")
    if not set(required).issubset(frame.columns):
        raise CausalCointegrationChronologyError(
            "Cointegrating regression requires y_log_price and x_log_price."
        )
    if len(frame) < 2:
        raise CausalCointegrationChronologyError(
            "Cointegrating regression requires at least two observations."
        )
    try:
        y = pd.to_numeric(frame["y_log_price"], errors="raise").to_numpy(
            dtype="float64", copy=True
        )
        x = pd.to_numeric(frame["x_log_price"], errors="raise").to_numpy(
            dtype="float64", copy=True
        )
    except (TypeError, ValueError) as exc:
        raise CausalCointegrationChronologyError(
            "Cointegrating-regression values must be numeric."
        ) from exc
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise CausalCointegrationChronologyError(
            "Cointegrating-regression values must be finite."
        )
    design = np.column_stack((np.ones(len(x), dtype="float64"), x))
    try:
        gamma, residual_sum_squares, rank, singular_values = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise CausalCointegrationChronologyError(
            "Cointegrating SVD least-squares estimation failed."
        ) from exc
    del residual_sum_squares
    singular_values = np.asarray(singular_values, dtype="float64")
    if int(rank) != 2:
        raise CausalCointegrationChronologyError(
            "Cointegrating regression is rank deficient."
        )
    if (
        singular_values.shape != (2,)
        or not np.isfinite(singular_values).all()
        or singular_values[0] < singular_values[1]
        or singular_values[1] <= 0.0
    ):
        raise CausalCointegrationChronologyError(
            "Cointegrating-regression singular values are invalid."
        )
    condition_number = float(singular_values[0] / singular_values[1])
    residuals = y - design @ gamma
    if (
        not np.isfinite(gamma).all()
        or not np.isfinite(residuals).all()
        or not math.isfinite(condition_number)
    ):
        raise CausalCointegrationChronologyError(
            "Cointegrating-regression output must be finite."
        )
    return CointegratingRegressionEstimate(
        alpha=float(gamma[0]),
        beta=float(gamma[1]),
        residuals=pd.Series(residuals, index=frame.index.copy(), dtype="float64"),
        observations=len(frame),
        design_rank=int(rank),
        largest_singular_value=float(singular_values[0]),
        smallest_singular_value=float(singular_values[1]),
        design_condition_number=condition_number,
    )


def inclusive_training_end_timestamp(
    train_end_exclusive: pd.Timestamp,
) -> pd.Timestamp:
    """Return the final admissible instant under an exclusive fold boundary."""

    if (
        not isinstance(train_end_exclusive, pd.Timestamp)
        or train_end_exclusive.tzinfo is None
    ):
        raise TypeError("train_end_exclusive must be timezone-aware.")
    return train_end_exclusive - pd.Timedelta(nanoseconds=1)


def validate_maximum_information_timestamp(
    *,
    maximum_information_timestamp: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
) -> pd.Timestamp:
    """Require maximum information at or before the inclusive training end."""

    if (
        not isinstance(maximum_information_timestamp, pd.Timestamp)
        or maximum_information_timestamp.tzinfo is None
    ):
        raise TypeError("maximum_information_timestamp must be timezone-aware.")
    train_end_inclusive = inclusive_training_end_timestamp(train_end_exclusive)
    if maximum_information_timestamp > train_end_inclusive:
        raise CausalCointegrationChronologyError(
            "Maximum statistical information timestamp exceeds train end."
        )
    return train_end_inclusive


def compare_successive_betas(
    *,
    current_beta: float,
    reference_beta: float,
) -> BetaStabilityComparison:
    """Apply the frozen relative-drift threshold to one prior-fold reference."""

    current = float(current_beta)
    reference = float(reference_beta)
    if not math.isfinite(reference):
        return BetaStabilityComparison(
            relative_beta_drift=math.nan,
            beta_sign_change=False,
            stability_status="invalid_reference_nonfinite",
            stability_pass=False,
        )
    if abs(reference) <= REFERENCE_BETA_ZERO_TOLERANCE:
        return BetaStabilityComparison(
            relative_beta_drift=math.nan,
            beta_sign_change=False,
            stability_status="invalid_reference_near_zero",
            stability_pass=False,
        )
    if not math.isfinite(current):
        return BetaStabilityComparison(
            relative_beta_drift=math.nan,
            beta_sign_change=False,
            stability_status="invalid_current_nonfinite",
            stability_pass=False,
        )
    if abs(current) <= REFERENCE_BETA_ZERO_TOLERANCE:
        return BetaStabilityComparison(
            relative_beta_drift=abs(current - reference) / abs(reference),
            beta_sign_change=False,
            stability_status="invalid_current_near_zero",
            stability_pass=False,
        )
    drift = abs(current - reference) / abs(reference)
    sign_change = bool(np.sign(current) != np.sign(reference))
    if sign_change:
        status = "beta_sign_change"
        passed = False
    elif drift > MAXIMUM_BETA_RELATIVE_DEVIATION:
        status = "relative_drift_exceeds_threshold"
        passed = False
    else:
        status = "stable"
        passed = True
    return BetaStabilityComparison(
        relative_beta_drift=float(drift),
        beta_sign_change=sign_change,
        stability_status=status,
        stability_pass=passed,
    )


def _baseline_comparison() -> BetaStabilityComparison:
    return BetaStabilityComparison(
        relative_beta_drift=math.nan,
        beta_sign_change=False,
        stability_status="baseline",
        stability_pass=True,
    )


def _validate_pair_frame(
    item: PairCointegrationInput,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_daily = ("session_date", "y_log_price", "x_log_price")
    expected_intraday = (
        "timestamp",
        "session_date",
        "y_log_price",
        "x_log_price",
    )
    daily = item.daily_log_prices.copy(deep=True)
    intraday = item.intraday_log_prices.copy(deep=True)
    if tuple(daily.columns) != expected_daily:
        raise CausalCointegrationChronologyError(
            "Daily pair schema changed."
        )
    if tuple(intraday.columns) != expected_intraday:
        raise CausalCointegrationChronologyError(
            "Intraday pair schema changed."
        )
    daily_sessions = pd.to_datetime(daily["session_date"], utc=True, errors="raise")
    intraday_timestamps = pd.to_datetime(
        intraday["timestamp"], utc=True, errors="raise"
    )
    intraday_sessions = pd.to_datetime(
        intraday["session_date"], utc=True, errors="raise"
    )
    if daily_sessions.empty or intraday_timestamps.empty:
        raise CausalCointegrationChronologyError(
            "Pair observations cannot be empty."
        )
    if not daily_sessions.is_monotonic_increasing:
        raise CausalCointegrationChronologyError(
            "Daily pair timestamps must be ordered."
        )
    if daily_sessions.duplicated().any():
        raise CausalCointegrationChronologyError(
            "Daily pair timestamps must be unique."
        )
    if not intraday_timestamps.is_monotonic_increasing:
        raise CausalCointegrationChronologyError(
            "Intraday pair timestamps must be ordered."
        )
    if intraday_timestamps.duplicated().any():
        raise CausalCointegrationChronologyError(
            "Intraday pair timestamps must be unique."
        )
    if daily_sessions.ge(DEVELOPMENT_END_EXCLUSIVE).any() or intraday_timestamps.ge(
        DEVELOPMENT_END_EXCLUSIVE
    ).any():
        raise CausalCointegrationChronologyError(
            "Causal development chronology forbids every 2026-or-later timestamp."
        )
    if not intraday_timestamps.dt.normalize().equals(intraday_sessions):
        raise CausalCointegrationChronologyError(
            "Intraday timestamp and session_date values are inconsistent."
        )
    daily["session_date"] = daily_sessions
    intraday["timestamp"] = intraday_timestamps
    intraday["session_date"] = intraday_sessions
    return daily, intraday


def _validate_inputs(
    inputs: CointegrationInputs,
) -> tuple[tuple[PairCointegrationInput, pd.DataFrame, pd.DataFrame], ...]:
    if not isinstance(inputs, CointegrationInputs):
        raise TypeError("inputs must be CointegrationInputs.")
    expected = tuple(
        (pair_id, y_symbol, x_symbol)
        for pair_id, (y_symbol, x_symbol) in zip(
            PAIR_IDS, CANDIDATE_PAIRS, strict=True
        )
    )
    actual = tuple(
        (item.pair_id, item.y_symbol, item.x_symbol) for item in inputs.pair_inputs
    )
    if actual != expected:
        raise CausalCointegrationChronologyError(
            "Pair identifiers and fixed Y/X orientations must remain exact."
        )
    return tuple(
        (item, *_validate_pair_frame(item)) for item in inputs.pair_inputs
    )


def _fold_value(fold: object, name: str) -> object:
    try:
        return getattr(fold, name)
    except AttributeError as exc:
        raise CausalCointegrationChronologyError(
            f"Fold is missing {name}."
        ) from exc


def _validate_folds(
    folds: Sequence[object] | None,
) -> tuple[object, ...]:
    selected = tuple(build_walk_forward_folds() if folds is None else folds)
    if not selected:
        raise CausalCointegrationChronologyError(
            "At least one causal fold is required."
        )
    frozen_by_id = {fold.fold_id: fold for fold in build_walk_forward_folds()}
    prior_test_end: pd.Timestamp | None = None
    for order, fold in enumerate(selected, start=1):
        fold_id = str(_fold_value(fold, "fold_id"))
        timestamps = {
            name: _fold_value(fold, name)
            for name in (
                "train_start",
                "train_end_exclusive",
                "test_start",
                "test_end_exclusive",
            )
        }
        if not all(
            isinstance(value, pd.Timestamp) and value.tzinfo is not None
            for value in timestamps.values()
        ):
            raise CausalCointegrationChronologyError(
                "Fold boundaries must be timezone-aware timestamps."
            )
        if timestamps["train_end_exclusive"] > timestamps["test_start"]:
            raise CausalCointegrationChronologyError(
                "Training and test intervals overlap."
            )
        if not (
            timestamps["train_start"] < timestamps["train_end_exclusive"]
            and timestamps["test_start"] < timestamps["test_end_exclusive"]
        ):
            raise CausalCointegrationChronologyError(
                "Fold boundaries are not chronological."
            )
        if prior_test_end is not None and timestamps["test_start"] < prior_test_end:
            raise CausalCointegrationChronologyError(
                "Causal test folds overlap or are unordered."
            )
        prior_test_end = timestamps["test_end_exclusive"]
        frozen = frozen_by_id.get(fold_id)
        if frozen is None or any(
            timestamps[name] != getattr(frozen, name) for name in timestamps
        ):
            raise CausalCointegrationChronologyError(
                "Causal folds must be an ordered subset of the frozen definitions."
            )
        if fold_id != build_walk_forward_folds()[order - 1].fold_id:
            raise CausalCointegrationChronologyError(
                "Causal fold ordering must remain deterministic."
            )
    return selected


def _plausibly_i1(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype="float64")
    level = _frozen_adf_result(numeric, regression="ct")
    difference = _frozen_adf_result(np.diff(numeric), regression="c")
    return bool(
        not level["reject_unit_root"] and difference["reject_unit_root"]
    )


def _engle_granger_p_value(frame: pd.DataFrame) -> float:
    try:
        _, p_value, _ = coint(
            frame["y_log_price"].to_numpy(dtype="float64"),
            frame["x_log_price"].to_numpy(dtype="float64"),
            trend="c",
            autolag="aic",
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return math.nan
    return float(p_value) if math.isfinite(float(p_value)) else math.nan


def _causal_ou_gate(
    *,
    intraday: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
    alpha: float,
    beta: float,
    attempted: bool,
) -> tuple[bool, str]:
    if not attempted:
        return False, "prior_statistical_gates_failed"
    timestamps = pd.to_datetime(intraday["timestamp"], utc=True, errors="raise")
    mask = timestamps.ge(train_start) & timestamps.lt(train_end_exclusive)
    frame = intraday.loc[mask].copy(deep=True)
    timestamps = timestamps.loc[mask].reset_index(drop=True)
    if len(frame) < 3:
        return False, "insufficient_training_intraday_observations"
    residual = (
        frame["y_log_price"].to_numpy(dtype="float64")
        - alpha
        - beta * frame["x_log_price"].to_numpy(dtype="float64")
    )
    sessions = frame["session_date"].astype(str).reset_index(drop=True)
    residual_series = pd.Series(residual).reset_index(drop=True)
    transition_mask = sessions.eq(sessions.shift(1)) & timestamps.diff().eq(
        pd.Timedelta(minutes=15)
    )
    previous = residual_series.shift(1).loc[transition_mask].to_numpy(
        dtype="float64"
    )
    current = residual_series.loc[transition_mask].to_numpy(dtype="float64")
    if len(current) < 3 or np.unique(previous).size < 2:
        return False, "insufficient_consecutive_training_transitions"
    design = np.column_stack((np.ones(len(previous)), previous))
    if np.linalg.matrix_rank(design) != 2:
        return False, "singular_training_ou_regression"
    try:
        parameters = np.linalg.solve(design.T @ design, design.T @ current)
    except np.linalg.LinAlgError:
        return False, "singular_training_ou_regression"
    intercept = float(parameters[0])
    phi = float(parameters[1])
    innovations = current - design @ parameters
    innovation_sigma = float(np.sqrt(np.mean(np.square(innovations))))
    if not 0.0 < phi < 1.0:
        return False, "invalid_phi"
    kappa = float(-np.log(phi))
    theta = float(intercept / (1.0 - phi))
    diffusion_sigma = float(
        innovation_sigma * np.sqrt(2.0 * kappa / (1.0 - phi**2))
    )
    half_life = float(np.log(2.0) / kappa)
    finite = bool(
        np.isfinite(
            (
                intercept,
                phi,
                kappa,
                theta,
                innovation_sigma,
                diffusion_sigma,
                half_life,
            )
        ).all()
        and kappa > 0.0
        and innovation_sigma > 0.0
        and diffusion_sigma > 0.0
    )
    half_life_valid = bool(
        math.isfinite(half_life)
        and MINIMUM_HALF_LIFE_BARS <= half_life <= MAXIMUM_HALF_LIFE_BARS
    )
    if not finite:
        return False, "invalid_parameters"
    if not half_life_valid:
        return False, "invalid_half_life"
    return True, ""


def analyze_causal_cointegration_chronology(
    inputs: CointegrationInputs,
    *,
    folds: Sequence[object] | None = None,
) -> CausalCointegrationChronologyResults:
    """Build training-only fold gates and successive-beta chronology evidence."""

    retained_inputs = _validate_inputs(inputs)
    selected_folds = _validate_folds(folds)
    records: list[dict[str, object]] = []
    ex_post_records: list[dict[str, object]] = []

    for pair_order, (item, daily, _) in enumerate(retained_inputs, start=1):
        sessions = pd.to_datetime(daily["session_date"], utc=True, errors="raise")
        previous_beta: float | None = None
        previous_fold: str | None = None
        previous_information_timestamp: pd.Timestamp | None = None
        for fold_order, fold in enumerate(selected_folds, start=1):
            train_mask = sessions.ge(fold.train_start) & sessions.lt(
                fold.train_end_exclusive
            )
            train = daily.loc[train_mask].copy(deep=True).reset_index(drop=True)
            if len(train) < 20:
                raise CausalCointegrationChronologyError(
                    f"{item.pair_id}/{fold.fold_id} requires at least 20 "
                    "training observations."
                )
            maximum_information_timestamp = pd.to_datetime(
                train["session_date"], utc=True, errors="raise"
            ).max()
            train_end_inclusive = validate_maximum_information_timestamp(
                maximum_information_timestamp=maximum_information_timestamp,
                train_end_exclusive=fold.train_end_exclusive,
            )
            estimate = estimate_cointegrating_regression(train)
            if previous_beta is None:
                comparison = _baseline_comparison()
                reference_beta = math.nan
                reference_fold = ""
                reference_information = pd.NaT
            else:
                comparison = compare_successive_betas(
                    current_beta=estimate.beta,
                    reference_beta=previous_beta,
                )
                reference_beta = previous_beta
                reference_fold = str(previous_fold)
                reference_information = previous_information_timestamp
            residual_adf = _frozen_adf_result(
                estimate.copy_residuals(), regression="n"
            )
            beta_interpretable = bool(
                math.isfinite(estimate.alpha)
                and math.isfinite(estimate.beta)
                and MINIMUM_BETA <= estimate.beta <= MAXIMUM_BETA
            )
            records.append(
                {
                    "pair_id": item.pair_id,
                    "pair_order": pair_order,
                    "y_symbol": item.y_symbol,
                    "x_symbol": item.x_symbol,
                    "fold_id": fold.fold_id,
                    "fold_order": fold_order,
                    "train_start": fold.train_start,
                    "train_end_inclusive": train_end_inclusive,
                    "train_end_exclusive": fold.train_end_exclusive,
                    "test_start": fold.test_start,
                    "test_end_exclusive": fold.test_end_exclusive,
                    "maximum_timestamp_used_for_estimation": (
                        maximum_information_timestamp
                    ),
                    "train_observations": len(train),
                    "alpha_estimate": estimate.alpha,
                    "beta_estimate": estimate.beta,
                    "design_rank": estimate.design_rank,
                    "largest_singular_value": estimate.largest_singular_value,
                    "smallest_singular_value": estimate.smallest_singular_value,
                    "design_condition_number": estimate.design_condition_number,
                    "beta_interpretable": beta_interpretable,
                    "reference_fold": reference_fold,
                    "reference_beta": reference_beta,
                    "reference_maximum_information_timestamp": (
                        reference_information
                    ),
                    "relative_beta_drift": comparison.relative_beta_drift,
                    "beta_sign_change": comparison.beta_sign_change,
                    "stability_threshold": MAXIMUM_BETA_RELATIVE_DEVIATION,
                    "stability_status": comparison.stability_status,
                    "stability_pass": comparison.stability_pass,
                    "y_plausibly_i1": _plausibly_i1(train["y_log_price"]),
                    "x_plausibly_i1": _plausibly_i1(train["x_log_price"]),
                    "engle_granger_p_value": _engle_granger_p_value(train),
                    "holm_adjusted_p_value": math.nan,
                    "holm_cointegration_pass": False,
                    "residual_adf_regression": "n",
                    "residual_adf_statistic": residual_adf["adf_statistic"],
                    "residual_adf_p_value": residual_adf["p_value"],
                    "residual_adf_used_lag": residual_adf["used_lag"],
                    "residual_stationary": residual_adf["reject_unit_root"],
                    "fold_eligibility": False,
                    "method_version": METHOD_VERSION,
                }
            )
            previous_beta = estimate.beta
            previous_fold = fold.fold_id
            previous_information_timestamp = maximum_information_timestamp

        ex_post = estimate_cointegrating_regression(daily)
        ex_post_records.append(
            {
                "pair_id": item.pair_id,
                "pair_order": pair_order,
                "y_symbol": item.y_symbol,
                "x_symbol": item.x_symbol,
                "observations": len(daily),
                "maximum_information_timestamp": sessions.max(),
                "alpha_ex_post": ex_post.alpha,
                "beta_ex_post": ex_post.beta,
                "statistic_role": "ex_post_descriptive_only",
                "used_as_fold_reference": False,
                "used_in_stability_gate": False,
                "used_in_eligibility": False,
                "method_version": METHOD_VERSION,
            }
        )

    ledger = pd.DataFrame.from_records(records, columns=CHRONOLOGY_LEDGER_COLUMNS)
    for fold in selected_folds:
        mask = ledger["fold_id"].eq(fold.fold_id)
        p_values = ledger.loc[mask, "engle_granger_p_value"].to_numpy(
            dtype="float64"
        )
        if np.isfinite(p_values).all():
            reject, adjusted, _, _ = multipletests(
                p_values,
                alpha=SIGNIFICANCE_LEVEL,
                method=MULTIPLE_TESTING_METHOD,
            )
            ledger.loc[mask, "holm_adjusted_p_value"] = adjusted
            ledger.loc[mask, "holm_cointegration_pass"] = reject.astype(bool)
        ledger.loc[mask, "fold_eligibility"] = (
            ledger.loc[mask, "y_plausibly_i1"].astype(bool)
            & ledger.loc[mask, "x_plausibly_i1"].astype(bool)
            & ledger.loc[mask, "holm_cointegration_pass"].astype(bool)
            & ledger.loc[mask, "beta_interpretable"].astype(bool)
            & ledger.loc[mask, "stability_pass"].astype(bool)
            & ledger.loc[mask, "residual_stationary"].astype(bool)
        )

    summary_records: list[dict[str, object]] = []
    retained_by_id = {item.pair_id: (item, intraday) for item, _, intraday in retained_inputs}
    for pair_order, pair_id in enumerate(PAIR_IDS, start=1):
        item, intraday = retained_by_id[pair_id]
        pair_rows = ledger.loc[ledger["pair_id"].eq(pair_id)].sort_values(
            "fold_order", kind="stable"
        )
        comparison_rows = pair_rows.loc[pair_rows["fold_order"].gt(1)]
        valid_drifts = pd.to_numeric(
            comparison_rows["relative_beta_drift"], errors="coerce"
        ).dropna()
        causal_stability_pass = bool(
            len(comparison_rows) == len(selected_folds) - 1
            and comparison_rows["stability_pass"].astype(bool).all()
        )
        stationary_count = int(pair_rows["residual_stationary"].astype(bool).sum())
        fold_stationarity_pass = bool(
            stationary_count >= MINIMUM_STATIONARY_FOLDS
        )
        final = pair_rows.iloc[-1]
        pre_ou_eligible = bool(
            final["y_plausibly_i1"]
            and final["x_plausibly_i1"]
            and final["holm_cointegration_pass"]
            and final["beta_interpretable"]
            and causal_stability_pass
            and fold_stationarity_pass
        )
        final_fold = selected_folds[-1]
        ou_pass, ou_rejection = _causal_ou_gate(
            intraday=intraday,
            train_start=final_fold.train_start,
            train_end_exclusive=final_fold.train_end_exclusive,
            alpha=float(final["alpha_estimate"]),
            beta=float(final["beta_estimate"]),
            attempted=pre_ou_eligible,
        )
        rejection_flags = (
            ("y_not_i1", not bool(final["y_plausibly_i1"])),
            ("x_not_i1", not bool(final["x_plausibly_i1"])),
            (
                "holm_cointegration_failed",
                not bool(final["holm_cointegration_pass"]),
            ),
            ("beta_failed", not bool(final["beta_interpretable"])),
            ("fold_beta_stability_failed", not causal_stability_pass),
            ("fold_stationarity_failed", not fold_stationarity_pass),
            ("ou_not_attempted", not pre_ou_eligible),
            ("ou_failed", pre_ou_eligible and not ou_pass),
        )
        rejection_reasons = [name for name, failed in rejection_flags if failed]
        summary_records.append(
            {
                "pair_id": pair_id,
                "pair_order": pair_order,
                "y_symbol": item.y_symbol,
                "x_symbol": item.x_symbol,
                "causal_comparisons": len(comparison_rows),
                "maximum_causal_relative_drift": (
                    float(valid_drifts.max()) if not valid_drifts.empty else math.nan
                ),
                "mean_causal_relative_drift": (
                    float(valid_drifts.mean()) if not valid_drifts.empty else math.nan
                ),
                "any_beta_sign_change": bool(
                    comparison_rows["beta_sign_change"].astype(bool).any()
                ),
                "causal_stability_pass": causal_stability_pass,
                "stationary_fold_count": stationary_count,
                "fold_stationarity_pass": fold_stationarity_pass,
                "final_fold_id": final["fold_id"],
                "final_fold_y_plausibly_i1": bool(final["y_plausibly_i1"]),
                "final_fold_x_plausibly_i1": bool(final["x_plausibly_i1"]),
                "final_fold_holm_cointegration_pass": bool(
                    final["holm_cointegration_pass"]
                ),
                "final_fold_beta_pass": bool(final["beta_interpretable"]),
                "causal_ou_attempted": pre_ou_eligible,
                "causal_ou_pass": ou_pass,
                "causal_ou_rejection_reason": ou_rejection,
                "final_pair_eligibility": not rejection_reasons,
                "rejection_reasons": "|".join(rejection_reasons),
                "maximum_information_timestamp": final[
                    "maximum_timestamp_used_for_estimation"
                ],
                "method_version": METHOD_VERSION,
            }
        )

    ledger = ledger.loc[:, list(CHRONOLOGY_LEDGER_COLUMNS)].reset_index(drop=True)
    summaries = pd.DataFrame.from_records(
        summary_records, columns=PAIR_SUMMARY_COLUMNS
    )
    ex_post_table = pd.DataFrame.from_records(
        ex_post_records, columns=EX_POST_BETA_COLUMNS
    )
    expected_ledger_order = [
        (pair_id, fold.fold_id) for pair_id in PAIR_IDS for fold in selected_folds
    ]
    if list(zip(ledger["pair_id"], ledger["fold_id"], strict=True)) != (
        expected_ledger_order
    ):
        raise RuntimeError("Causal chronology ordering changed.")
    if tuple(summaries["pair_id"]) != PAIR_IDS or tuple(
        ex_post_table["pair_id"]
    ) != PAIR_IDS:
        raise RuntimeError("Causal pair ordering changed.")
    return CausalCointegrationChronologyResults(
        fold_chronology=ledger,
        pair_summaries=summaries,
        ex_post_beta_diagnostics=ex_post_table,
    )
