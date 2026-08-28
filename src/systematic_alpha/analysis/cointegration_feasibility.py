"""Development-only cointegration and OU feasibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from systematic_alpha.analysis.eda_features import (
    build_return_features,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    build_walk_forward_folds,
)


CANDIDATE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("SPY", "QQQ"),
    ("SPY", "IWM"),
    ("QQQ", "IWM"),
)

PAIR_IDS: Final[tuple[str, ...]] = (
    "SPY_QQQ",
    "SPY_IWM",
    "QQQ_IWM",
)

REQUIRED_SYMBOLS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
)

SIGNIFICANCE_LEVEL: Final[float] = 0.05
MULTIPLE_TESTING_METHOD: Final[str] = "holm"

MINIMUM_BETA: Final[float] = 0.10
MAXIMUM_BETA: Final[float] = 10.00
MAXIMUM_BETA_RELATIVE_DEVIATION: Final[float] = 0.25
MINIMUM_STATIONARY_FOLDS: Final[int] = 3

MINIMUM_HALF_LIFE_BARS: Final[float] = 1.0
MAXIMUM_HALF_LIFE_BARS: Final[float] = 130.0

PAIR_INPUT_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = (
    "pair_id",
    "y_symbol",
    "x_symbol",
    "daily_observations",
    "intraday_observations",
    "daily_start_session",
    "daily_end_session",
    "intraday_start_timestamp",
    "intraday_end_timestamp",
    "forward_fill_used",
    "locked_period_accessed",
)

SERIES_INTEGRATION_COLUMNS: Final[
    tuple[str, ...]
] = (
    "symbol",
    "test",
    "regression",
    "autolag",
    "observations",
    "adf_statistic",
    "p_value",
    "used_lag",
    "critical_value_5pct",
    "reject_unit_root",
    "plausibly_i1",
)

COINTEGRATION_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = (
    "pair_id",
    "y_symbol",
    "x_symbol",
    "observations",
    "alpha",
    "beta",
    "beta_interpretable",
    "ols_r_squared",
    "engle_granger_statistic",
    "engle_granger_p_value",
    "holm_adjusted_p_value",
    "holm_reject",
    "residual_adf_statistic",
    "residual_adf_p_value",
    "residual_adf_used_lag",
    "residual_adf_reject",
)

FOLD_STABILITY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "pair_id",
    "fold_id",
    "train_start",
    "train_end_exclusive",
    "test_start",
    "test_end_exclusive",
    "train_observations",
    "test_observations",
    "train_alpha",
    "train_beta",
    "beta_relative_deviation",
    "beta_sign_stable",
    "test_residual_adf_statistic",
    "test_residual_adf_p_value",
    "test_residual_stationary",
)

OU_DIAGNOSTIC_COLUMNS: Final[
    tuple[str, ...]
] = (
    "pair_id",
    "attempted",
    "intraday_observations",
    "consecutive_transitions",
    "ar_intercept",
    "phi",
    "kappa_per_bar",
    "theta",
    "innovation_sigma",
    "diffusion_sigma",
    "half_life_bars",
    "phi_valid",
    "parameters_finite",
    "half_life_valid",
    "ou_pass",
    "rejection_reason",
)

PAIR_ELIGIBILITY_COLUMNS: Final[
    tuple[str, ...]
] = (
    "pair_id",
    "y_symbol",
    "x_symbol",
    "y_plausibly_i1",
    "x_plausibly_i1",
    "holm_cointegration_pass",
    "beta_pass",
    "fold_beta_stability_pass",
    "stationary_fold_count",
    "fold_stationarity_pass",
    "ou_attempted",
    "ou_pass",
    "eligible",
    "rejection_reasons",
)


class CointegrationFeasibilityError(ValueError):
    """Raised when Day 14 diagnostics cannot be built safely."""

LOCKED_PERIOD_START: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-02",
    tz="UTC",
)


def _validate_development_boundary(
    bars: pd.DataFrame,
) -> None:
    """Reject any observation from the locked evaluation period."""

    if "timestamp" not in bars.columns:
        raise CointegrationFeasibilityError(
            "Input bars must contain a timestamp column."
        )

    timestamps = pd.to_datetime(
        bars["timestamp"],
        utc=True,
        errors="raise",
    )

    if timestamps.ge(
        LOCKED_PERIOD_START
    ).any():
        raise CointegrationFeasibilityError(
            "Day 14 input accessed the locked 2026 period."
        )


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Return a defensive zero-based copy of one table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    return frame.copy(deep=True).reset_index(
        drop=True
    )


def _require_finite_log_prices(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    """Require finite synchronized log-price observations."""

    values = frame[
        [
            "y_log_price",
            "x_log_price",
        ]
    ].to_numpy(dtype=float)

    if (
        len(frame) == 0
        or not np.isfinite(values).all()
    ):
        raise CointegrationFeasibilityError(
            f"{name} must contain finite synchronized "
            "log-price observations."
        )


@dataclass(frozen=True, slots=True)
class PairCointegrationInput:
    """One fixed-orientation pair's synchronized inputs."""

    pair_id: str
    y_symbol: str
    x_symbol: str
    daily_log_prices: pd.DataFrame
    intraday_log_prices: pd.DataFrame

    def __post_init__(self) -> None:
        """Defensively retain synchronized pair observations."""

        daily = _copy_frame(
            self.daily_log_prices,
            name="daily_log_prices",
        )
        intraday = _copy_frame(
            self.intraday_log_prices,
            name="intraday_log_prices",
        )

        _require_finite_log_prices(
            daily,
            name="daily_log_prices",
        )
        _require_finite_log_prices(
            intraday,
            name="intraday_log_prices",
        )

        object.__setattr__(
            self,
            "daily_log_prices",
            daily,
        )
        object.__setattr__(
            self,
            "intraday_log_prices",
            intraday,
        )


@dataclass(frozen=True, slots=True)
class CointegrationInputs:
    """Immutable ordered Day 14 pair-input bundle."""

    pair_inputs: tuple[
        PairCointegrationInput,
        ...,
    ]

    def __post_init__(self) -> None:
        """Freeze and validate pair ordering."""

        retained = tuple(self.pair_inputs)

        if tuple(
            item.pair_id
            for item in retained
        ) != PAIR_IDS:
            raise CointegrationFeasibilityError(
                "Pair inputs must follow the frozen "
                "Day 14 order."
            )

        object.__setattr__(
            self,
            "pair_inputs",
            retained,
        )


@dataclass(frozen=True, slots=True)
class CointegrationFeasibilityResults:
    """Immutable Day 14 diagnostic tables."""

    pair_input_diagnostics: pd.DataFrame
    series_integration_diagnostics: pd.DataFrame
    cointegration_diagnostics: pd.DataFrame
    fold_stability_diagnostics: pd.DataFrame
    ou_diagnostics: pd.DataFrame
    pair_eligibility: pd.DataFrame

    def __post_init__(self) -> None:
        """Defensively retain every diagnostic table."""

        for name in (
            "pair_input_diagnostics",
            "series_integration_diagnostics",
            "cointegration_diagnostics",
            "fold_stability_diagnostics",
            "ou_diagnostics",
            "pair_eligibility",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(
                    getattr(self, name),
                    name=name,
                ),
            )

    def copy_pair_input_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.pair_input_diagnostics.copy(deep=True)

    def copy_series_integration_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.series_integration_diagnostics.copy(deep=True)

    def copy_cointegration_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.cointegration_diagnostics.copy(deep=True)

    def copy_fold_stability_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.fold_stability_diagnostics.copy(deep=True)

    def copy_ou_diagnostics(
        self,
    ) -> pd.DataFrame:
        return self.ou_diagnostics.copy(deep=True)

    def copy_pair_eligibility(
        self,
    ) -> pd.DataFrame:
        return self.pair_eligibility.copy(deep=True)


def _daily_pair_frame(
    sessions: pd.DataFrame,
    *,
    y_symbol: str,
    x_symbol: str,
) -> pd.DataFrame:
    """Align daily session closes by exact date intersection."""

    y_leg = sessions.loc[
        sessions["symbol"].eq(y_symbol),
        [
            "session_date",
            "log_session_close",
        ],
    ].rename(
        columns={
            "log_session_close": (
                "y_log_price"
            ),
        }
    )

    x_leg = sessions.loc[
        sessions["symbol"].eq(x_symbol),
        [
            "session_date",
            "log_session_close",
        ],
    ].rename(
        columns={
            "log_session_close": (
                "x_log_price"
            ),
        }
    )

    result = y_leg.merge(
        x_leg,
        on="session_date",
        how="inner",
        validate="one_to_one",
        sort=True,
    )

    return result[
        [
            "session_date",
            "y_log_price",
            "x_log_price",
        ]
    ].reset_index(drop=True)


def _intraday_pair_frame(
    bars: pd.DataFrame,
    *,
    y_symbol: str,
    x_symbol: str,
) -> pd.DataFrame:
    """Align intraday closes by exact timestamp and session."""

    keys = [
        "timestamp",
        "session_date",
    ]

    y_leg = bars.loc[
        bars["symbol"].eq(y_symbol),
        keys + ["log_close"],
    ].rename(
        columns={
            "log_close": "y_log_price",
        }
    )

    x_leg = bars.loc[
        bars["symbol"].eq(x_symbol),
        keys + ["log_close"],
    ].rename(
        columns={
            "log_close": "x_log_price",
        }
    )

    result = y_leg.merge(
        x_leg,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=True,
    )

    return result[
        [
            "timestamp",
            "session_date",
            "y_log_price",
            "x_log_price",
        ]
    ].reset_index(drop=True)


def build_cointegration_inputs(
    bars: pd.DataFrame,
) -> CointegrationInputs:
    """Build exact synchronized daily and intraday pair inputs."""

    _validate_development_boundary(bars)

    features = build_return_features(
        bars,
        expected_symbols=REQUIRED_SYMBOLS,
    )

    sessions = features.sessions[
        [
            "symbol",
            "session_date",
            "session_close",
        ]
    ].copy()

    sessions["log_session_close"] = np.log(
        pd.to_numeric(
            sessions["session_close"],
            errors="raise",
        )
    )

    intraday = features.bars[
        [
            "timestamp",
            "session_date",
            "symbol",
            "close",
        ]
    ].copy()

    intraday["log_close"] = np.log(
        pd.to_numeric(
            intraday["close"],
            errors="raise",
        )
    )

    pair_inputs = tuple(
        PairCointegrationInput(
            pair_id=pair_id,
            y_symbol=y_symbol,
            x_symbol=x_symbol,
            daily_log_prices=(
                _daily_pair_frame(
                    sessions,
                    y_symbol=y_symbol,
                    x_symbol=x_symbol,
                )
            ),
            intraday_log_prices=(
                _intraday_pair_frame(
                    intraday,
                    y_symbol=y_symbol,
                    x_symbol=x_symbol,
                )
            ),
        )
        for pair_id, (
            y_symbol,
            x_symbol,
        ) in zip(
            PAIR_IDS,
            CANDIDATE_PAIRS,
            strict=True,
        )
    )

    return CointegrationInputs(
        pair_inputs=pair_inputs
    )


def _adf_result(
    values: pd.Series | np.ndarray,
    *,
    regression: str,
) -> dict[str, object]:
    """Run one frozen ADF specification."""

    series = pd.Series(
        values,
        dtype="float64",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(series) < 20 or series.nunique() < 2:
        return {
            "observations": int(len(series)),
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lag": 0,
            "critical_value_5pct": np.nan,
            "reject_unit_root": False,
        }

    statistic, p_value, used_lag, observations, critical, _ = (
        adfuller(
            series.to_numpy(dtype=float),
            regression=regression,
            autolag="AIC",
        )
    )

    return {
        "observations": int(observations),
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "critical_value_5pct": float(
            critical["5%"]
        ),
        "reject_unit_root": bool(
            p_value < SIGNIFICANCE_LEVEL
        ),
    }


def _fit_long_run(
    frame: pd.DataFrame,
) -> tuple[float, float, float, pd.Series]:
    """Fit log(Y) on an intercept and log(X)."""

    y = frame["y_log_price"].to_numpy(
        dtype=float
    )
    x = frame["x_log_price"].to_numpy(
        dtype=float
    )

    model = OLS(
        y,
        add_constant(
            x,
            has_constant="add",
        ),
    ).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    residual = pd.Series(
        model.resid,
        index=frame.index,
        dtype="float64",
    )

    return (
        alpha,
        beta,
        float(model.rsquared),
        residual,
    )


def _pair_input_table(
    inputs: CointegrationInputs,
) -> pd.DataFrame:
    """Build synchronized-input evidence."""

    records: list[dict[str, object]] = []
    locked_start = pd.Timestamp(
        "2026-01-02",
        tz="UTC",
    )

    for item in inputs.pair_inputs:
        daily = item.daily_log_prices
        intraday = item.intraday_log_prices
        timestamps = pd.to_datetime(
            intraday["timestamp"],
            utc=True,
            errors="raise",
        )
        locked_accessed = bool(
            timestamps.ge(locked_start).any()
        )

        if locked_accessed:
            raise CointegrationFeasibilityError(
                "Day 14 input accessed the locked "
                "2026 period."
            )

        records.append(
            {
                "pair_id": item.pair_id,
                "y_symbol": item.y_symbol,
                "x_symbol": item.x_symbol,
                "daily_observations": len(daily),
                "intraday_observations": len(
                    intraday
                ),
                "daily_start_session": str(
                    daily["session_date"].min()
                ),
                "daily_end_session": str(
                    daily["session_date"].max()
                ),
                "intraday_start_timestamp": (
                    timestamps.min().isoformat()
                ),
                "intraday_end_timestamp": (
                    timestamps.max().isoformat()
                ),
                "forward_fill_used": False,
                "locked_period_accessed": False,
            }
        )

    return pd.DataFrame(
        records,
        columns=PAIR_INPUT_DIAGNOSTIC_COLUMNS,
    )


def _integration_table(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Test each daily log-price series for plausible I(1)."""

    sessions = build_return_features(
        bars,
        expected_symbols=REQUIRED_SYMBOLS,
    ).sessions

    records: list[dict[str, object]] = []

    for symbol in REQUIRED_SYMBOLS:
        prices = pd.to_numeric(
            sessions.loc[
                sessions["symbol"].eq(symbol),
                "session_close",
            ],
            errors="raise",
        )
        log_prices = np.log(
            prices.to_numpy(dtype=float)
        )

        level = _adf_result(
            log_prices,
            regression="ct",
        )
        difference = _adf_result(
            np.diff(log_prices),
            regression="c",
        )

        plausibly_i1 = bool(
            not level["reject_unit_root"]
            and difference[
                "reject_unit_root"
            ]
        )

        for test_name, regression, result in (
            ("level", "ct", level),
            (
                "first_difference",
                "c",
                difference,
            ),
        ):
            records.append(
                {
                    "symbol": symbol,
                    "test": test_name,
                    "regression": regression,
                    "autolag": "AIC",
                    **result,
                    "plausibly_i1": (
                        plausibly_i1
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=SERIES_INTEGRATION_COLUMNS,
    )


def _cointegration_table(
    inputs: CointegrationInputs,
) -> pd.DataFrame:
    """Fit all three predeclared Engle–Granger relationships."""

    records: list[dict[str, object]] = []

    for item in inputs.pair_inputs:
        frame = item.daily_log_prices
        alpha, beta, r_squared, residual = (
            _fit_long_run(frame)
        )

        statistic, p_value, _ = coint(
            frame["y_log_price"].to_numpy(
                dtype=float
            ),
            frame["x_log_price"].to_numpy(
                dtype=float
            ),
            trend="c",
            autolag="aic",
        )
        residual_adf = _adf_result(
            residual,
            regression="n",
        )

        records.append(
            {
                "pair_id": item.pair_id,
                "y_symbol": item.y_symbol,
                "x_symbol": item.x_symbol,
                "observations": len(frame),
                "alpha": alpha,
                "beta": beta,
                "beta_interpretable": bool(
                    np.isfinite(alpha)
                    and np.isfinite(beta)
                    and MINIMUM_BETA
                    <= beta
                    <= MAXIMUM_BETA
                ),
                "ols_r_squared": r_squared,
                "engle_granger_statistic": (
                    float(statistic)
                ),
                "engle_granger_p_value": (
                    float(p_value)
                ),
                "holm_adjusted_p_value": (
                    np.nan
                ),
                "holm_reject": False,
                "residual_adf_statistic": (
                    residual_adf[
                        "adf_statistic"
                    ]
                ),
                "residual_adf_p_value": (
                    residual_adf["p_value"]
                ),
                "residual_adf_used_lag": (
                    residual_adf["used_lag"]
                ),
                "residual_adf_reject": (
                    residual_adf[
                        "reject_unit_root"
                    ]
                ),
            }
        )

    table = pd.DataFrame(
        records,
        columns=COINTEGRATION_DIAGNOSTIC_COLUMNS,
    )

    reject, adjusted, _, _ = multipletests(
        table[
            "engle_granger_p_value"
        ].to_numpy(dtype=float),
        alpha=SIGNIFICANCE_LEVEL,
        method=MULTIPLE_TESTING_METHOD,
    )

    table["holm_adjusted_p_value"] = adjusted
    table["holm_reject"] = reject.astype(bool)

    return table


def _fold_table(
    inputs: CointegrationInputs,
    cointegration: pd.DataFrame,
) -> pd.DataFrame:
    """Measure frozen-coefficient stability in four expanding folds."""

    full_beta = (
        cointegration.set_index("pair_id")[
            "beta"
        ].to_dict()
    )
    records: list[dict[str, object]] = []

    for item in inputs.pair_inputs:
        frame = item.daily_log_prices.copy()
        sessions = pd.to_datetime(
            frame["session_date"],
            utc=True,
            errors="raise",
        )

        for fold in build_walk_forward_folds():
            train_mask = (
                sessions.ge(fold.train_start)
                & sessions.lt(
                    fold.train_end_exclusive
                )
            )
            test_mask = (
                sessions.ge(fold.test_start)
                & sessions.lt(
                    fold.test_end_exclusive
                )
            )

            train = frame.loc[
                train_mask
            ].reset_index(drop=True)
            test = frame.loc[
                test_mask
            ].reset_index(drop=True)

            alpha, beta, _, _ = _fit_long_run(
                train
            )
            reference_beta = float(
                full_beta[item.pair_id]
            )

            deviation = (
                abs(beta - reference_beta)
                / abs(reference_beta)
                if reference_beta != 0.0
                else np.inf
            )
            sign_stable = bool(
                beta != 0.0
                and reference_beta != 0.0
                and np.sign(beta)
                == np.sign(reference_beta)
            )

            test_residual = (
                test["y_log_price"]
                - alpha
                - beta
                * test["x_log_price"]
            )
            residual_adf = _adf_result(
                test_residual,
                regression="c",
            )

            records.append(
                {
                    "pair_id": item.pair_id,
                    "fold_id": fold.fold_id,
                    "train_start": (
                        fold.train_start
                        .isoformat()
                    ),
                    "train_end_exclusive": (
                        fold.train_end_exclusive
                        .isoformat()
                    ),
                    "test_start": (
                        fold.test_start
                        .isoformat()
                    ),
                    "test_end_exclusive": (
                        fold.test_end_exclusive
                        .isoformat()
                    ),
                    "train_observations": (
                        len(train)
                    ),
                    "test_observations": (
                        len(test)
                    ),
                    "train_alpha": alpha,
                    "train_beta": beta,
                    "beta_relative_deviation": (
                        float(deviation)
                    ),
                    "beta_sign_stable": (
                        sign_stable
                    ),
                    "test_residual_adf_statistic": (
                        residual_adf[
                            "adf_statistic"
                        ]
                    ),
                    "test_residual_adf_p_value": (
                        residual_adf[
                            "p_value"
                        ]
                    ),
                    "test_residual_stationary": (
                        residual_adf[
                            "reject_unit_root"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=FOLD_STABILITY_COLUMNS,
    )


def _ou_table(
    inputs: CointegrationInputs,
    cointegration: pd.DataFrame,
    folds: pd.DataFrame,
    integration: pd.DataFrame,
) -> pd.DataFrame:
    """Fit OU diagnostics only after all prior statistical gates pass."""

    integration_status = (
        integration.groupby(
            "symbol",
            sort=False,
        )["plausibly_i1"]
        .all()
        .to_dict()
    )
    cointegration_by_pair = (
        cointegration.set_index("pair_id")
    )
    records: list[dict[str, object]] = []

    for item in inputs.pair_inputs:
        row = cointegration_by_pair.loc[
            item.pair_id
        ]
        pair_folds = folds.loc[
            folds["pair_id"].eq(
                item.pair_id
            )
        ]

        beta_stable = bool(
            pair_folds[
                "beta_sign_stable"
            ].astype(bool).all()
            and (
                pair_folds[
                    "beta_relative_deviation"
                ]
                <= MAXIMUM_BETA_RELATIVE_DEVIATION
            ).all()
        )
        stationary_count = int(
            pair_folds[
                "test_residual_stationary"
            ].astype(bool).sum()
        )

        attempted = bool(
            integration_status[
                item.y_symbol
            ]
            and integration_status[
                item.x_symbol
            ]
            and row["holm_reject"]
            and row["beta_interpretable"]
            and beta_stable
            and stationary_count
            >= MINIMUM_STATIONARY_FOLDS
        )

        base = {
            "pair_id": item.pair_id,
            "attempted": attempted,
            "intraday_observations": len(
                item.intraday_log_prices
            ),
        }

        if not attempted:
            records.append(
                {
                    **base,
                    "consecutive_transitions": 0,
                    "ar_intercept": np.nan,
                    "phi": np.nan,
                    "kappa_per_bar": np.nan,
                    "theta": np.nan,
                    "innovation_sigma": np.nan,
                    "diffusion_sigma": np.nan,
                    "half_life_bars": np.nan,
                    "phi_valid": False,
                    "parameters_finite": False,
                    "half_life_valid": False,
                    "ou_pass": False,
                    "rejection_reason": (
                        "prior_statistical_gates_failed"
                    ),
                }
            )
            continue

        frame = item.intraday_log_prices.copy()
        residual = (
            frame["y_log_price"]
            - float(row["alpha"])
            - float(row["beta"])
            * frame["x_log_price"]
        )
        timestamps = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="raise",
        )
        sessions = frame[
            "session_date"
        ].astype(str)

        transition_mask = (
            sessions.eq(sessions.shift(1))
            & timestamps.diff().eq(
                pd.Timedelta(minutes=15)
            )
        )
        previous = residual.shift(1).loc[
            transition_mask
        ].to_numpy(dtype=float)
        current = residual.loc[
            transition_mask
        ].to_numpy(dtype=float)

        ar_model = OLS(
            current,
            add_constant(
                previous,
                has_constant="add",
            ),
        ).fit()

        intercept = float(ar_model.params[0])
        phi = float(ar_model.params[1])
        innovation_sigma = float(
            np.sqrt(
                np.mean(
                    np.square(
                        ar_model.resid
                    )
                )
            )
        )
        phi_valid = bool(
            0.0 < phi < 1.0
        )

        if phi_valid:
            kappa = float(-np.log(phi))
            theta = float(
                intercept / (1.0 - phi)
            )
            diffusion_sigma = float(
                innovation_sigma
                * np.sqrt(
                    2.0
                    * kappa
                    / (1.0 - phi**2)
                )
            )
            half_life = float(
                np.log(2.0) / kappa
            )
        else:
            kappa = np.nan
            theta = np.nan
            diffusion_sigma = np.nan
            half_life = np.nan

        parameters_finite = bool(
            np.isfinite(
                [
                    intercept,
                    phi,
                    kappa,
                    theta,
                    innovation_sigma,
                    diffusion_sigma,
                    half_life,
                ]
            ).all()
            and kappa > 0.0
            and innovation_sigma > 0.0
            and diffusion_sigma > 0.0
        )
        half_life_valid = bool(
            np.isfinite(half_life)
            and MINIMUM_HALF_LIFE_BARS
            <= half_life
            <= MAXIMUM_HALF_LIFE_BARS
        )
        ou_pass = bool(
            phi_valid
            and parameters_finite
            and half_life_valid
        )

        reasons: list[str] = []

        if not phi_valid:
            reasons.append("invalid_phi")
        if not parameters_finite:
            reasons.append(
                "invalid_parameters"
            )
        if not half_life_valid:
            reasons.append(
                "invalid_half_life"
            )

        records.append(
            {
                **base,
                "consecutive_transitions": (
                    len(current)
                ),
                "ar_intercept": intercept,
                "phi": phi,
                "kappa_per_bar": kappa,
                "theta": theta,
                "innovation_sigma": (
                    innovation_sigma
                ),
                "diffusion_sigma": (
                    diffusion_sigma
                ),
                "half_life_bars": half_life,
                "phi_valid": phi_valid,
                "parameters_finite": (
                    parameters_finite
                ),
                "half_life_valid": (
                    half_life_valid
                ),
                "ou_pass": ou_pass,
                "rejection_reason": "|".join(
                    reasons
                ),
            }
        )

    return pd.DataFrame(
        records,
        columns=OU_DIAGNOSTIC_COLUMNS,
    )


def _eligibility_table(
    inputs: CointegrationInputs,
    integration: pd.DataFrame,
    cointegration: pd.DataFrame,
    folds: pd.DataFrame,
    ou: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every predeclared pass/fail gate."""

    integration_status = (
        integration.groupby(
            "symbol",
            sort=False,
        )["plausibly_i1"]
        .all()
        .to_dict()
    )
    cointegration_by_pair = (
        cointegration.set_index("pair_id")
    )
    ou_by_pair = ou.set_index("pair_id")
    records: list[dict[str, object]] = []

    for item in inputs.pair_inputs:
        coint_row = cointegration_by_pair.loc[
            item.pair_id
        ]
        ou_row = ou_by_pair.loc[
            item.pair_id
        ]
        pair_folds = folds.loc[
            folds["pair_id"].eq(
                item.pair_id
            )
        ]

        beta_stability = bool(
            pair_folds[
                "beta_sign_stable"
            ].astype(bool).all()
            and (
                pair_folds[
                    "beta_relative_deviation"
                ]
                <= MAXIMUM_BETA_RELATIVE_DEVIATION
            ).all()
        )
        stationary_count = int(
            pair_folds[
                "test_residual_stationary"
            ].astype(bool).sum()
        )
        fold_stationarity = bool(
            stationary_count
            >= MINIMUM_STATIONARY_FOLDS
        )

        gates = {
            "y_not_i1": bool(
                not integration_status[
                    item.y_symbol
                ]
            ),
            "x_not_i1": bool(
                not integration_status[
                    item.x_symbol
                ]
            ),
            "holm_cointegration_failed": bool(
                not coint_row["holm_reject"]
            ),
            "beta_failed": bool(
                not coint_row[
                    "beta_interpretable"
                ]
            ),
            "fold_beta_stability_failed": bool(
                not beta_stability
            ),
            "fold_stationarity_failed": bool(
                not fold_stationarity
            ),
            "ou_not_attempted": bool(
                not ou_row["attempted"]
            ),
            "ou_failed": bool(
                ou_row["attempted"]
                and not ou_row["ou_pass"]
            ),
        }
        rejection_reasons = [
            name
            for name, failed in gates.items()
            if failed
        ]
        eligible = not rejection_reasons

        records.append(
            {
                "pair_id": item.pair_id,
                "y_symbol": item.y_symbol,
                "x_symbol": item.x_symbol,
                "y_plausibly_i1": (
                    integration_status[
                        item.y_symbol
                    ]
                ),
                "x_plausibly_i1": (
                    integration_status[
                        item.x_symbol
                    ]
                ),
                "holm_cointegration_pass": (
                    bool(
                        coint_row[
                            "holm_reject"
                        ]
                    )
                ),
                "beta_pass": bool(
                    coint_row[
                        "beta_interpretable"
                    ]
                ),
                "fold_beta_stability_pass": (
                    beta_stability
                ),
                "stationary_fold_count": (
                    stationary_count
                ),
                "fold_stationarity_pass": (
                    fold_stationarity
                ),
                "ou_attempted": bool(
                    ou_row["attempted"]
                ),
                "ou_pass": bool(
                    ou_row["ou_pass"]
                ),
                "eligible": eligible,
                "rejection_reasons": "|".join(
                    rejection_reasons
                ),
            }
        )

    return pd.DataFrame(
        records,
        columns=PAIR_ELIGIBILITY_COLUMNS,
    )


def run_cointegration_feasibility(
    bars: pd.DataFrame,
) -> CointegrationFeasibilityResults:
    """Run the frozen Day 14 feasibility study."""

    inputs = build_cointegration_inputs(
        bars
    )
    pair_inputs = _pair_input_table(
        inputs
    )
    integration = _integration_table(
        bars
    )
    cointegration = _cointegration_table(
        inputs
    )
    folds = _fold_table(
        inputs,
        cointegration,
    )
    ou = _ou_table(
        inputs,
        cointegration,
        folds,
        integration,
    )
    eligibility = _eligibility_table(
        inputs,
        integration,
        cointegration,
        folds,
        ou,
    )

    return CointegrationFeasibilityResults(
        pair_input_diagnostics=pair_inputs,
        series_integration_diagnostics=(
            integration
        ),
        cointegration_diagnostics=(
            cointegration
        ),
        fold_stability_diagnostics=folds,
        ou_diagnostics=ou,
        pair_eligibility=eligibility,
    )
