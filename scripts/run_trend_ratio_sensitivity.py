"""Run the predeclared Day 7 development-only trend-ratio sensitivity study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import pandas as pd

from systematic_alpha.analysis.day07_trend_sensitivity import (
    build_day07_metadata,
    write_day07_artifacts,
)
from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    BASELINE_COST_BPS_PER_TURNOVER,
    DAY07_ANNUALIZATION_FACTOR,
    run_trend_ratio_sensitivity_grid,
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
    "artifacts/day07/trend_ratio_sensitivity_v1"
)


def _normalize_session_dates(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Return timezone-naive, normalized session timestamps."""

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
        raise ValueError(
            f"{context} contain malformed session dates."
        ) from exc

    if normalized.isna().any():
        raise ValueError(
            f"{context} contain missing session dates."
        )

    return normalized


def _validate_daily_volatility(
    daily_volatility: pd.DataFrame,
    *,
    development_start: pd.Timestamp,
    development_end_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Validate the trusted Day 5 development-only volatility artifact."""

    if not isinstance(daily_volatility, pd.DataFrame):
        raise TypeError(
            "daily_volatility must be a pandas DataFrame."
        )

    if daily_volatility.empty:
        raise ValueError(
            "Day 5 daily volatility data must not be empty."
        )

    required_columns = (
        "symbol",
        "session_date",
        "annualized_total_realized_volatility",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in daily_volatility.columns
    ]

    if missing_columns:
        raise ValueError(
            "Day 5 daily volatility data are missing required columns: "
            f"{missing_columns}."
        )

    result = daily_volatility.copy(deep=True)

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result["symbol"].isna().any() or result["symbol"].eq("").any():
        raise ValueError(
            "Day 5 daily volatility data contain invalid symbols."
        )

    result["session_date"] = _normalize_session_dates(
        result["session_date"],
        context="Day 5 daily volatility data",
    )

    try:
        result["annualized_total_realized_volatility"] = (
            pd.to_numeric(
                result[
                    "annualized_total_realized_volatility"
                ],
                errors="raise",
            ).astype(float)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "annualized_total_realized_volatility must be numeric."
        ) from exc

    volatility_values = result[
        "annualized_total_realized_volatility"
    ]

    if not volatility_values.map(math.isfinite).all():
        raise ValueError(
            "annualized_total_realized_volatility must be finite."
        )

    if volatility_values.lt(0.0).any():
        raise ValueError(
            "annualized_total_realized_volatility must be non-negative."
        )

    if result.duplicated(
        ["symbol", "session_date"],
        keep=False,
    ).any():
        raise ValueError(
            "Day 5 daily volatility data contain duplicate "
            "symbol-session rows."
        )

    start_date = pd.Timestamp(development_start).tz_localize(None).normalize()
    end_date = (
        pd.Timestamp(development_end_inclusive)
        .tz_localize(None)
        .normalize()
    )

    minimum_date = result["session_date"].min()
    maximum_date = result["session_date"].max()

    if minimum_date < start_date or maximum_date > end_date:
        raise RuntimeError(
            "Day 5 volatility data extend outside the permitted "
            "development window."
        )

    symbols = set(result["symbol"].astype(str))

    if SYMBOL not in symbols:
        raise RuntimeError(
            "Day 5 volatility data do not contain SPY."
        )

    return result.sort_values(
        ["symbol", "session_date"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_spy_features(
    bars: pd.DataFrame,
    daily_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Extract SPY and confirm full session-regime coverage."""

    required_columns = {
        "timestamp",
        "symbol",
        SESSION_COLUMN,
        PRICE_COLUMN,
        RETURN_COLUMN,
    }
    missing_columns = sorted(
        required_columns.difference(bars.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Return-feature bars are missing required columns: "
            f"{missing_columns}."
        )

    spy_bars = (
        bars.loc[
            bars["symbol"]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq(SYMBOL)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if spy_bars.empty:
        raise RuntimeError(
            "SPY return-feature bars cannot be empty."
        )

    spy_bars[SESSION_COLUMN] = _normalize_session_dates(
        spy_bars[SESSION_COLUMN],
        context="SPY return-feature bars",
    )

    spy_volatility = daily_volatility.loc[
        daily_volatility["symbol"].eq(SYMBOL)
    ]

    strategy_sessions = set(
        spy_bars[SESSION_COLUMN].drop_duplicates()
    )
    volatility_sessions = set(
        spy_volatility["session_date"].drop_duplicates()
    )

    missing_regime_sessions = sorted(
        strategy_sessions - volatility_sessions
    )

    if missing_regime_sessions:
        examples = [
            pd.Timestamp(value).date().isoformat()
            for value in missing_regime_sessions[:10]
        ]
        raise RuntimeError(
            "SPY strategy sessions are missing Day 5 volatility "
            f"coverage. Examples: {examples}."
        )

    return spy_bars


def main() -> None:
    """Execute the complete frozen Day 7 development grid."""

    project_root = find_project_root()
    config = load_project_config()

    sample_window = SampleWindow.from_project_config(config)

    store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    bars = store.read(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
        verify_hash=True,
    )
    bars = sample_window.validate_development_frame(bars)

    manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
    )

    if "sha256" not in manifest:
        raise RuntimeError(
            "Development dataset manifest has no sha256 field."
        )

    features = build_return_features(bars)

    volatility_path = (
        project_root / DAILY_VOLATILITY_RELATIVE_PATH
    )

    if not volatility_path.exists():
        raise FileNotFoundError(
            "Required Day 5 volatility artifact does not exist: "
            f"{DAILY_VOLATILITY_RELATIVE_PATH.as_posix()}"
        )

    daily_volatility = pd.read_parquet(volatility_path)
    daily_volatility = _validate_daily_volatility(
        daily_volatility,
        development_start=sample_window.development_start,
        development_end_inclusive=(
            sample_window.development_end_inclusive
        ),
    )

    spy_bars = _validate_spy_features(
        features.bars,
        daily_volatility,
    )

    tables = run_trend_ratio_sensitivity_grid(
        spy_bars,
        cost_bps_per_turnover=(
            BASELINE_COST_BPS_PER_TURNOVER
        ),
        annualization_factor=DAY07_ANNUALIZATION_FACTOR,
        price_column=PRICE_COLUMN,
        return_column=RETURN_COLUMN,
        session_column=SESSION_COLUMN,
        daily_volatility=daily_volatility,
        benchmark_symbol=SYMBOL,
        stress_quantile=0.80,
    )

    metadata = build_day07_metadata(
        permitted_dataset_identifier=DEVELOPMENT_DATASET_ID,
        dataset_manifest_sha256=str(manifest["sha256"]),
        regime_definition=tables.regime_definition,
        symbol=SYMBOL,
        timeframe="15-minute",
        price_column=PRICE_COLUMN,
        return_column=RETURN_COLUMN,
    )

    output_directory = (
        project_root / ARTIFACT_RELATIVE_DIRECTORY
    )

    written_paths = write_day07_artifacts(
        tables,
        metadata=metadata,
        output_directory=output_directory,
    )

    parameter_results = tables.parameter_results
    stability = tables.neighborhood_stability

    baseline = parameter_results.loc[
        parameter_results["short_window"].eq(8)
        & parameter_results["long_window"].eq(32)
        & parameter_results["neutral_band"].eq(0.001)
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "The Day 6 baseline does not appear exactly once."
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

    isolated = (
        parameter_results.dropna(
            subset=["net_sharpe_ratio"]
        )
        .sort_values(
            "net_sharpe_ratio",
            ascending=False,
            kind="stable",
        )
        .iloc[0]
    )

    plateau = (
        stability.dropna(
            subset=["median_neighbor_net_sharpe"]
        )
        .sort_values(
            "median_neighbor_net_sharpe",
            ascending=False,
            kind="stable",
        )
        .iloc[0]
    )

    baseline_row = baseline.iloc[0]

    print("===== DAY 7 TREND-RATIO SENSITIVITY =====")
    print("Development dataset:", DEVELOPMENT_DATASET_ID)
    print("Manifest SHA256:", manifest["sha256"])
    print("SPY observations:", len(spy_bars))
    print(
        "Sample start:",
        pd.to_datetime(spy_bars["timestamp"], utc=True).min().isoformat(),
    )
    print(
        "Sample end:",
        pd.to_datetime(spy_bars["timestamp"], utc=True).max().isoformat(),
    )
    print("Locked period accessed:", False)
    print("Configurations:", len(parameter_results))
    print("Positive net configurations:", positive_net_count)
    print(
        "Positive break-even configurations:",
        positive_break_even_count,
    )

    print("\nDay 6 baseline reconciliation:")
    print(
        baseline_row[
            [
                "configuration_id",
                "gross_cumulative_return",
                "gross_sharpe_ratio",
                "net_cumulative_return",
                "net_sharpe_ratio",
                "net_max_drawdown",
                "total_turnover",
            ]
        ].to_string()
    )

    print("\nHighest isolated net Sharpe — diagnostic only:")
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

    print("\nHighest neighbouring median Sharpe — diagnostic only:")
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
        print(path.relative_to(project_root).as_posix())

    print("\nDAY 7 TREND-RATIO SENSITIVITY PASSED")


if __name__ == "__main__":
    main()
