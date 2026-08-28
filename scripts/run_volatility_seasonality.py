"""Run D05-T02C realized-volatility and seasonality analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from systematic_alpha.analysis.volatility_seasonality import (
    build_volatility_seasonality,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate Day 05 realized volatility "
            "and intraday seasonality."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="artifacts/day05/eda_v1",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/day05/"
            "volatility_seasonality_v1"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute D05-T02C."""

    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    data_dir = output_dir / "data"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bars = pd.read_parquet(
        input_dir
        / "data"
        / "bar_return_features.parquet"
    )

    sessions = pd.read_parquet(
        input_dir
        / "data"
        / "session_return_features.parquet"
    )

    bundle = build_volatility_seasonality(
        bars,
        sessions,
    )

    bundle.daily_volatility.to_parquet(
        data_dir / "daily_volatility.parquet",
        index=False,
    )

    bundle.daily_volatility.to_csv(
        table_dir / "daily_volatility.csv",
        index=False,
    )

    bundle.intraday_seasonality.to_csv(
        table_dir / "intraday_seasonality.csv",
        index=False,
    )

    summary = (
        bundle.daily_volatility.groupby(
            "symbol",
            observed=True,
        )
        .agg(
            sessions=("session_date", "size"),
            median_annualized_total_rv=(
                "annualized_total_realized_volatility",
                "median",
            ),
            p95_annualized_total_rv=(
                "annualized_total_realized_volatility",
                lambda values: values.quantile(0.95),
            ),
            maximum_annualized_total_rv=(
                "annualized_total_realized_volatility",
                "max",
            ),
            median_overnight_variance_share=(
                "overnight_variance_share",
                "median",
            ),
            mean_jump_variation_share=(
                "jump_variation_share",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        table_dir / "volatility_summary.csv",
        index=False,
    )

    top_days = (
        bundle.daily_volatility.sort_values(
            "annualized_total_realized_volatility",
            ascending=False,
        )
        .groupby(
            "symbol",
            observed=True,
            sort=False,
        )
        .head(10)
        [
            [
                "symbol",
                "session_date",
                "annualized_total_realized_volatility",
                "annualized_regular_realized_volatility",
                "annualized_parkinson_volatility",
                "overnight_variance_share",
                "jump_variation_share",
                "maximum_absolute_bar_return",
            ]
        ]
    )

    top_days.to_csv(
        table_dir / "highest_volatility_days.csv",
        index=False,
    )

    maximum_reconciliation_error = float(
        bundle.daily_volatility[
            "regular_return_reconciliation_error"
        ]
        .abs()
        .max()
    )

    metadata = {
        "analysis": (
            "day05_volatility_seasonality_v1"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "daily_rows": int(
            len(bundle.daily_volatility)
        ),
        "seasonality_rows": int(
            len(bundle.intraday_seasonality)
        ),
        "maximum_regular_return_reconciliation_error": (
            maximum_reconciliation_error
        ),
        "annualization_days": 252,
        "jump_warning": (
            "RV minus bipower variation is retained "
            "as a descriptive jump/outlier proxy, "
            "not as a formal jump-test conclusion."
        ),
    }

    (
        output_dir / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "===== D05-T02C VOLATILITY AND SEASONALITY ====="
    )

    print("\nVolatility summary:")
    print(summary.to_string(index=False))

    print("\nHighest-volatility sessions:")
    print(top_days.to_string(index=False))

    selected_times = {
        "09:30:00",
        "12:45:00",
        "15:45:00",
    }

    selected_seasonality = (
        bundle.intraday_seasonality.loc[
            bundle.intraday_seasonality[
                "local_time"
            ].isin(selected_times),
            [
                "symbol",
                "bar_number",
                "local_time",
                "observations",
                "mean_absolute_return",
                "mean_squared_return",
                "mean_volume_share",
                "mean_trade_count_share",
                "mean_squared_return_share",
            ],
        ]
    )

    print("\nSelected intraday positions:")
    print(
        selected_seasonality.to_string(
            index=False
        )
    )

    print(
        "\nMaximum return reconciliation error:",
        maximum_reconciliation_error,
    )

    print("\nOutputs:")
    print(output_dir.resolve())

    print("\nD05-T02C PASSED")


if __name__ == "__main__":
    main()
