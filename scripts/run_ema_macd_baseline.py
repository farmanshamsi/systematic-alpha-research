"""Run the frozen Day 8 development-only EMA/MACD baseline."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Final

import pandas as pd

from systematic_alpha.analysis.day08_ema_macd_report import (
    write_ema_macd_baseline_artifacts,
)
from systematic_alpha.analysis.eda_features import build_return_features
from systematic_alpha.analysis.ema_macd_baseline import (
    DAY08_ANNUALIZATION_FACTOR,
    analyse_ema_macd_baseline,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
    load_project_config,
)
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.sample_windows import SampleWindow
from systematic_alpha.strategies.ema_macd import EmaMacdParameters


DEVELOPMENT_DATASET_ID: Final[str] = (
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical"
)

SYMBOL: Final[str] = "SPY"
PRICE_COLUMN: Final[str] = "close"
RETURN_COLUMN: Final[str] = "close_to_close_simple_return"
SESSION_COLUMN: Final[str] = "session_date"

ARTIFACT_RELATIVE_DIRECTORY: Final[Path] = Path(
    "artifacts/day08/ema_macd_baseline_v1"
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{64}$"
)


def _validate_development_manifest(
    manifest: dict[str, Any],
    *,
    expected_row_count: int,
) -> str:
    """Validate immutable development-dataset lineage metadata."""

    if not isinstance(manifest, dict):
        raise TypeError(
            "Development dataset manifest must be a dictionary."
        )

    dataset_id = manifest.get("dataset_id")

    if dataset_id != DEVELOPMENT_DATASET_ID:
        raise RuntimeError(
            "Development manifest dataset_id does not match the "
            "predeclared Day 8 dataset."
        )

    dataset_kind = manifest.get("dataset_kind")

    if dataset_kind != "bars":
        raise RuntimeError(
            "Development manifest dataset_kind must be 'bars'."
        )

    manifest_row_count = manifest.get("row_count")

    if (
        isinstance(manifest_row_count, bool)
        or not isinstance(manifest_row_count, int)
    ):
        raise RuntimeError(
            "Development manifest row_count must be an integer."
        )

    if manifest_row_count != expected_row_count:
        raise RuntimeError(
            "Development manifest row_count does not match the "
            "verified dataset."
        )

    digest = manifest.get("sha256")

    if not isinstance(digest, str):
        raise RuntimeError(
            "Development dataset manifest has no valid sha256 field."
        )

    normalized_digest = digest.strip().lower()

    if not _SHA256_PATTERN.fullmatch(normalized_digest):
        raise RuntimeError(
            "Development dataset manifest sha256 must contain "
            "64 hexadecimal characters."
        )

    return normalized_digest


def _extract_spy_features(
    feature_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Extract and validate SPY feature observations."""

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
        required_columns.difference(feature_bars.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Return-feature bars are missing required columns: "
            f"{missing_columns}."
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

    if spy_bars[SESSION_COLUMN].isna().any():
        raise RuntimeError(
            "SPY feature bars contain missing session dates."
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

    prices = spy_bars[PRICE_COLUMN]

    if prices.isna().any():
        raise RuntimeError(
            "SPY close prices contain missing values."
        )

    if not prices.map(math.isfinite).all():
        raise RuntimeError(
            "SPY close prices contain non-finite values."
        )

    if prices.le(0.0).any():
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

    return spy_bars


def _performance_row(
    performance: pd.DataFrame,
    *,
    series: str,
) -> pd.Series:
    """Return exactly one compact performance row."""

    selected = performance.loc[
        performance["series"].eq(series)
    ]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one {series!r} performance row."
        )

    return selected.iloc[0]


def main() -> None:
    """Execute the frozen Day 8 development-only baseline."""

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
    manifest_sha256 = _validate_development_manifest(
        manifest,
        expected_row_count=len(bars),
    )

    features = build_return_features(bars)
    spy_bars = _extract_spy_features(features.bars)

    parameters = EmaMacdParameters()

    analysis = analyse_ema_macd_baseline(
        spy_bars,
        parameters=parameters,
        annualization_factor=DAY08_ANNUALIZATION_FACTOR,
        session_column=SESSION_COLUMN,
    )

    output_directory = (
        project_root / ARTIFACT_RELATIVE_DIRECTORY
    )

    written_paths = write_ema_macd_baseline_artifacts(
        analysis,
        output_directory=output_directory,
        dataset_identifier=DEVELOPMENT_DATASET_ID,
        dataset_manifest_sha256=manifest_sha256,
        development_start=(
            sample_window.development_start.date()
        ),
        development_end=(
            sample_window.development_end_inclusive.date()
        ),
    )

    performance = analysis.performance_summary
    diagnostics = analysis.strategy_bundle.diagnostics.iloc[0]
    holding = analysis.holding_diagnostics.iloc[0]
    break_even = analysis.cost_break_even.iloc[0]

    buy_and_hold = _performance_row(
        performance,
        series="buy_and_hold",
    )
    gross = _performance_row(
        performance,
        series="ema_macd_gross",
    )
    net = _performance_row(
        performance,
        series="ema_macd_net",
    )

    print("===== DAY 8 EMA/MACD BASELINE =====")
    print("Development dataset:", DEVELOPMENT_DATASET_ID)
    print("Manifest SHA256:", manifest_sha256)
    print("Symbol:", SYMBOL)
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

    print("\nFrozen parameters:")
    print("Fast EMA window:", parameters.fast_window)
    print("Slow EMA window:", parameters.slow_window)
    print("Signal EMA window:", parameters.signal_window)
    print("Neutral band:", parameters.neutral_band)
    print(
        "Cost bps per turnover:",
        parameters.cost_bps_per_turnover,
    )

    print("\nPerformance:")
    print(
        "Buy-and-hold cumulative return:",
        buy_and_hold["cumulative_return"],
    )
    print(
        "Buy-and-hold Sharpe:",
        buy_and_hold["sharpe_ratio"],
    )
    print(
        "EMA/MACD gross cumulative return:",
        gross["cumulative_return"],
    )
    print(
        "EMA/MACD gross Sharpe:",
        gross["sharpe_ratio"],
    )
    print(
        "EMA/MACD net cumulative return:",
        net["cumulative_return"],
    )
    print(
        "EMA/MACD net Sharpe:",
        net["sharpe_ratio"],
    )
    print(
        "EMA/MACD net maximum drawdown:",
        net["max_drawdown"],
    )

    print("\nTrading diagnostics:")
    print("Total turnover:", diagnostics["total_turnover"])
    print(
        "Position-changing bars:",
        diagnostics["position_changing_bars"],
    )
    print(
        "Long exposure pct:",
        diagnostics["long_exposure_pct"],
    )
    print(
        "Short exposure pct:",
        diagnostics["short_exposure_pct"],
    )
    print(
        "Neutral exposure pct:",
        diagnostics["neutral_exposure_pct"],
    )

    print("\nHolding diagnostics:")
    print(
        "Non-zero episodes:",
        holding["non_zero_episode_count"],
    )
    print(
        "Median holding bars:",
        holding["median_holding_duration_bars"],
    )
    print(
        "Session-crossing episodes:",
        holding["overnight_carry_episode_count"],
    )

    print("\nCost break-even:")
    print("Status:", break_even["status"])
    print(
        "Break-even cost bps:",
        break_even["break_even_cost_bps"],
    )

    print("\nSignal validation:")
    print(
        analysis.signal_validation.to_string(
            index=False
        )
    )

    print("\nArtifacts:")
    for path in written_paths:
        print(path.relative_to(project_root).as_posix())

    print("\nDAY 8 EMA/MACD BASELINE PASSED")


if __name__ == "__main__":
    main()
