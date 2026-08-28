"""Run the frozen Day 9 development-only EMA/MACD sensitivity study."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Final

import pandas as pd

from systematic_alpha.analysis.day09_ema_macd_sensitivity_report import (
    build_day09_metadata,
    write_day09_ema_macd_sensitivity_artifacts,
)
from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.ema_macd_sensitivity import (
    BASELINE_FAST_WINDOW,
    BASELINE_NEUTRAL_BAND,
    BASELINE_SIGNAL_WINDOW,
    BASELINE_SLOW_WINDOW,
    DAY09_ANNUALIZATION_FACTOR,
    DAY09_COST_BPS_PER_TURNOVER,
    EXPECTED_CONFIGURATION_COUNT,
    run_ema_macd_sensitivity_grid,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
    load_project_config,
)
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.sample_windows import SampleWindow


DEVELOPMENT_DATASET_ID: Final[str] = (
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical"
)

SYMBOL: Final[str] = "SPY"
PRICE_COLUMN: Final[str] = "close"
RETURN_COLUMN: Final[str] = "close_to_close_simple_return"
SESSION_COLUMN: Final[str] = "session_date"

DAILY_VOLATILITY_RELATIVE_PATH: Final[Path] = Path(
    "artifacts/day05/"
    "volatility_seasonality_v1/"
    "data/daily_volatility.parquet"
)

ARTIFACT_RELATIVE_DIRECTORY: Final[Path] = Path(
    "artifacts/day09/ema_macd_sensitivity_v1"
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{64}$"
)


def _validate_development_manifest(
    manifest: dict[str, Any],
    *,
    expected_row_count: int,
) -> str:
    """Validate immutable development-dataset lineage."""

    if not isinstance(manifest, dict):
        raise TypeError(
            "Development dataset manifest must be a dictionary."
        )

    if manifest.get("dataset_id") != DEVELOPMENT_DATASET_ID:
        raise RuntimeError(
            "Development manifest dataset_id does not match "
            "the frozen Day 9 dataset."
        )

    if manifest.get("dataset_kind") != "bars":
        raise RuntimeError(
            "Development manifest dataset_kind must be 'bars'."
        )

    row_count = manifest.get("row_count")

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
    ):
        raise RuntimeError(
            "Development manifest row_count must be an integer."
        )

    if row_count != expected_row_count:
        raise RuntimeError(
            "Development manifest row_count does not match "
            "the verified dataset."
        )

    digest = manifest.get("sha256")

    if not isinstance(digest, str):
        raise RuntimeError(
            "Development manifest has no valid sha256 field."
        )

    normalized_digest = digest.strip().lower()

    if not _SHA256_PATTERN.fullmatch(
        normalized_digest
    ):
        raise RuntimeError(
            "Development manifest sha256 must contain "
            "64 hexadecimal characters."
        )

    return normalized_digest


def _normalize_session_dates(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Return timezone-naive normalized session timestamps."""

    try:
        normalized = (
            pd.to_datetime(
                values.copy(deep=True),
                utc=True,
                errors="raise",
            )
            .dt.tz_convert(None)
            .dt.normalize()
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{context} contain malformed session dates."
        ) from exc

    if normalized.isna().any():
        raise RuntimeError(
            f"{context} contain missing session dates."
        )

    return normalized


def _validate_daily_volatility(
    daily_volatility: pd.DataFrame,
    *,
    development_start: pd.Timestamp,
    development_end_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Validate the trusted Day 5 volatility artifact."""

    if not isinstance(
        daily_volatility,
        pd.DataFrame,
    ):
        raise TypeError(
            "daily_volatility must be a pandas DataFrame."
        )

    if daily_volatility.empty:
        raise RuntimeError(
            "Day 5 daily volatility data must not be empty."
        )

    required_columns = {
        "symbol",
        SESSION_COLUMN,
        "annualized_total_realized_volatility",
    }
    missing_columns = sorted(
        required_columns.difference(
            daily_volatility.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Day 5 daily volatility data are missing "
            f"required columns: {missing_columns}."
        )

    result = daily_volatility.copy(deep=True)

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
        raise RuntimeError(
            "Day 5 daily volatility data contain "
            "invalid symbols."
        )

    result[SESSION_COLUMN] = _normalize_session_dates(
        result[SESSION_COLUMN],
        context="Day 5 daily volatility data",
    )

    try:
        result[
            "annualized_total_realized_volatility"
        ] = pd.to_numeric(
            result[
                "annualized_total_realized_volatility"
            ],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "annualized_total_realized_volatility "
            "must be numeric."
        ) from exc

    volatility = result[
        "annualized_total_realized_volatility"
    ]

    if not volatility.map(math.isfinite).all():
        raise RuntimeError(
            "annualized_total_realized_volatility "
            "must be finite."
        )

    if volatility.lt(0.0).any():
        raise RuntimeError(
            "annualized_total_realized_volatility "
            "must be non-negative."
        )

    if result.duplicated(
        ["symbol", SESSION_COLUMN],
        keep=False,
    ).any():
        raise RuntimeError(
            "Day 5 volatility data contain duplicate "
            "symbol-session rows."
        )

    start = (
        pd.Timestamp(development_start)
        .tz_localize(None)
        .normalize()
    )
    end = (
        pd.Timestamp(development_end_inclusive)
        .tz_localize(None)
        .normalize()
    )

    if (
        result[SESSION_COLUMN].min() < start
        or result[SESSION_COLUMN].max() > end
    ):
        raise RuntimeError(
            "Day 5 volatility data extend outside the "
            "permitted development window."
        )

    if SYMBOL not in set(
        result["symbol"].astype(str)
    ):
        raise RuntimeError(
            "Day 5 volatility data do not contain SPY."
        )

    return result.sort_values(
        ["symbol", SESSION_COLUMN],
        kind="stable",
    ).reset_index(drop=True)


def _extract_spy_features(
    feature_bars: pd.DataFrame,
    daily_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Extract SPY features and confirm regime coverage."""

    if not isinstance(feature_bars, pd.DataFrame):
        raise TypeError(
            "feature_bars must be a pandas DataFrame."
        )

    if feature_bars.empty:
        raise RuntimeError(
            "Return-feature bars cannot be empty."
        )

    required_columns = {
        "timestamp",
        "symbol",
        SESSION_COLUMN,
        PRICE_COLUMN,
        RETURN_COLUMN,
    }
    missing_columns = sorted(
        required_columns.difference(
            feature_bars.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Return-feature bars are missing required "
            f"columns: {missing_columns}."
        )

    normalized_symbols = (
        feature_bars["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    spy_bars = (
        feature_bars.loc[
            normalized_symbols.eq(SYMBOL)
        ]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    if spy_bars.empty:
        raise RuntimeError(
            "SPY return-feature bars cannot be empty."
        )

    spy_bars["symbol"] = SYMBOL

    try:
        spy_bars["timestamp"] = pd.to_datetime(
            spy_bars["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "SPY feature bars contain malformed timestamps."
        ) from exc

    if spy_bars["timestamp"].isna().any():
        raise RuntimeError(
            "SPY feature bars contain missing timestamps."
        )

    spy_bars[SESSION_COLUMN] = (
        _normalize_session_dates(
            spy_bars[SESSION_COLUMN],
            context="SPY return-feature bars",
        )
    )

    try:
        spy_bars[PRICE_COLUMN] = pd.to_numeric(
            spy_bars[PRICE_COLUMN],
            errors="raise",
        ).astype(float)
        spy_bars[RETURN_COLUMN] = pd.to_numeric(
            spy_bars[RETURN_COLUMN],
            errors="coerce",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "SPY price and return columns must be numeric."
        ) from exc

    close = spy_bars[PRICE_COLUMN]

    if close.isna().any():
        raise RuntimeError(
            "SPY close prices contain missing values."
        )

    if not close.map(math.isfinite).all():
        raise RuntimeError(
            "SPY close prices contain non-finite values."
        )

    if close.le(0.0).any():
        raise RuntimeError(
            "SPY close prices must be strictly positive."
        )

    spy_bars = spy_bars.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    if spy_bars.duplicated(
        ["symbol", "timestamp"],
        keep=False,
    ).any():
        raise RuntimeError(
            "SPY feature bars contain duplicate timestamps."
        )

    spy_volatility = daily_volatility.loc[
        daily_volatility["symbol"].eq(SYMBOL)
    ]

    strategy_sessions = set(
        spy_bars[SESSION_COLUMN].drop_duplicates()
    )
    volatility_sessions = set(
        spy_volatility[
            SESSION_COLUMN
        ].drop_duplicates()
    )

    missing_sessions = sorted(
        strategy_sessions - volatility_sessions
    )

    if missing_sessions:
        examples = [
            pd.Timestamp(value).date().isoformat()
            for value in missing_sessions[:10]
        ]

        raise RuntimeError(
            "SPY strategy sessions are missing Day 5 "
            "volatility coverage. "
            f"Examples: {examples}."
        )

    return spy_bars


def _select_baseline_row(
    parameter_results: pd.DataFrame,
) -> pd.Series:
    """Return the exact frozen Day 8 baseline row."""

    required = {
        "fast_window",
        "slow_window",
        "signal_window",
        "neutral_band",
    }

    if not required.issubset(
        parameter_results.columns
    ):
        raise RuntimeError(
            "Parameter results are missing baseline columns."
        )

    baseline = parameter_results.loc[
        parameter_results["fast_window"].eq(
            BASELINE_FAST_WINDOW
        )
        & parameter_results["slow_window"].eq(
            BASELINE_SLOW_WINDOW
        )
        & parameter_results["signal_window"].eq(
            BASELINE_SIGNAL_WINDOW
        )
        & parameter_results["neutral_band"].eq(
            BASELINE_NEUTRAL_BAND
        )
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "The Day 8 baseline must appear exactly once "
            "in Day 9 results."
        )

    return baseline.iloc[0]


def _select_highest_finite_row(
    frame: pd.DataFrame,
    *,
    value_column: str,
) -> pd.Series:
    """Return the deterministic highest finite diagnostic row."""

    required = {
        "configuration_id",
        value_column,
    }

    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Diagnostic table is missing required columns."
        )

    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )

    finite = frame.loc[
        values.map(
            lambda value: (
                pd.notna(value)
                and math.isfinite(float(value))
            )
        )
    ].copy()

    if finite.empty:
        raise RuntimeError(
            f"No finite {value_column} value is available."
        )

    return (
        finite.sort_values(
            [
                value_column,
                "configuration_id",
            ],
            ascending=[False, True],
            kind="stable",
        )
        .iloc[0]
    )


def main() -> None:
    """Execute the complete frozen Day 9 development grid."""

    project_root = find_project_root()
    config = load_project_config()
    sample_window = SampleWindow.from_project_config(
        config
    )

    store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    bars = store.read(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
        verify_hash=True,
    )
    bars = sample_window.validate_development_frame(
        bars
    )

    manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
    )
    manifest_sha256 = _validate_development_manifest(
        manifest,
        expected_row_count=len(bars),
    )

    features = build_return_features(bars)

    volatility_path = (
        project_root
        / DAILY_VOLATILITY_RELATIVE_PATH
    )

    if not volatility_path.exists():
        raise FileNotFoundError(
            "Required Day 5 volatility artifact does not "
            "exist: "
            f"{DAILY_VOLATILITY_RELATIVE_PATH.as_posix()}"
        )

    daily_volatility = pd.read_parquet(
        volatility_path
    )
    daily_volatility = _validate_daily_volatility(
        daily_volatility,
        development_start=(
            sample_window.development_start
        ),
        development_end_inclusive=(
            sample_window.development_end_inclusive
        ),
    )

    spy_bars = _extract_spy_features(
        features.bars,
        daily_volatility,
    )

    tables = run_ema_macd_sensitivity_grid(
        spy_bars,
        cost_bps_per_turnover=(
            DAY09_COST_BPS_PER_TURNOVER
        ),
        annualization_factor=(
            DAY09_ANNUALIZATION_FACTOR
        ),
        price_column=PRICE_COLUMN,
        return_column=RETURN_COLUMN,
        session_column=SESSION_COLUMN,
        daily_volatility=daily_volatility,
        benchmark_symbol=SYMBOL,
        stress_quantile=0.80,
    )

    metadata = build_day09_metadata(
        permitted_dataset_identifier=(
            DEVELOPMENT_DATASET_ID
        ),
        dataset_manifest_sha256=(
            manifest_sha256
        ),
        regime_definition=(
            tables.regime_definition
        ),
        symbol=SYMBOL,
        timeframe="15-minute",
        price_column=PRICE_COLUMN,
        return_column=RETURN_COLUMN,
    )

    output_directory = (
        project_root
        / ARTIFACT_RELATIVE_DIRECTORY
    )

    written_paths = (
        write_day09_ema_macd_sensitivity_artifacts(
            tables,
            metadata=metadata,
            output_directory=output_directory,
        )
    )

    parameter_results = tables.parameter_results
    stability = tables.neighborhood_stability

    if (
        len(parameter_results)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise RuntimeError(
            "Day 9 did not produce exactly 108 "
            "parameter-result rows."
        )

    baseline = _select_baseline_row(
        parameter_results
    )
    isolated = _select_highest_finite_row(
        parameter_results,
        value_column="net_sharpe_ratio",
    )
    plateau = _select_highest_finite_row(
        stability,
        value_column=(
            "median_neighbor_net_sharpe"
        ),
    )

    positive_net_count = int(
        parameter_results[
            "net_cumulative_return"
        ].gt(0.0).sum()
    )
    positive_break_even_count = int(
        parameter_results[
            "break_even_cost_bps"
        ].fillna(0.0).gt(0.0).sum()
    )

    print(
        "===== DAY 9 EMA/MACD SENSITIVITY ====="
    )
    print(
        "Development dataset:",
        DEVELOPMENT_DATASET_ID,
    )
    print("Manifest SHA256:", manifest_sha256)
    print("SPY observations:", len(spy_bars))
    print(
        "Sample start:",
        spy_bars["timestamp"].min().isoformat(),
    )
    print(
        "Sample end:",
        spy_bars["timestamp"].max().isoformat(),
    )
    print("Locked period accessed:", False)
    print(
        "Configurations:",
        len(parameter_results),
    )
    print(
        "Positive net configurations:",
        positive_net_count,
    )
    print(
        "Positive break-even configurations:",
        positive_break_even_count,
    )

    print("\nDay 8 baseline reconciliation:")
    print(
        baseline[
            [
                "configuration_id",
                "gross_cumulative_return",
                "gross_sharpe_ratio",
                "net_cumulative_return",
                "net_sharpe_ratio",
                "net_max_drawdown",
                "total_turnover",
                "break_even_cost_bps",
            ]
        ].to_string()
    )

    print(
        "\nHighest isolated net Sharpe "
        "— diagnostic only:"
    )
    print(
        isolated[
            [
                "configuration_id",
                "net_cumulative_return",
                "net_sharpe_ratio",
                "total_turnover",
                "break_even_cost_bps",
            ]
        ].to_string()
    )

    print(
        "\nHighest neighbouring median Sharpe "
        "— diagnostic only:"
    )
    print(
        plateau[
            [
                "configuration_id",
                "net_sharpe_ratio",
                "median_neighbor_net_sharpe",
                "minimum_neighbor_net_sharpe",
                "neighbor_count",
                "total_turnover",
            ]
        ].to_string()
    )

    print("\nCompact table sizes:")

    for name in (
        "parameter_results",
        "annual_results",
        "annual_consistency",
        "regime_results",
        "regime_definition",
        "holding_diagnostics",
        "signal_validation",
        "signal_buckets",
        "neighborhood_stability",
    ):
        table = getattr(tables, name)

        print(
            f"{name}: rows={len(table)}, "
            f"columns={len(table.columns)}"
        )

    print("\nArtifacts:")

    for path in written_paths:
        print(
            path.relative_to(
                project_root
            ).as_posix()
        )

    print(
        "\nDAY 9 EMA/MACD SENSITIVITY PASSED"
    )


if __name__ == "__main__":
    main()
