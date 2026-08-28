"""Generate Day 05 stylized-fact evidence tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from systematic_alpha.analysis.stylized_facts import (
    build_stylized_facts,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate distribution and serial-dependence "
            "statistics for Day 05."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="artifacts/day05/eda_v1",
        help="Directory created by run_development_eda.py.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/day05/"
            "stylized_facts_v1"
        ),
        help="Directory for stylized-fact tables.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the stylized-fact pipeline."""

    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    bar_path = (
        input_dir
        / "data"
        / "bar_return_features.parquet"
    )

    session_path = (
        input_dir
        / "data"
        / "session_return_features.parquet"
    )

    bar_features = pd.read_parquet(
        bar_path
    )

    session_features = pd.read_parquet(
        session_path
    )

    bundle = build_stylized_facts(
        bar_features,
        session_features,
    )

    table_dir = output_dir / "tables"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "return_moments.csv": bundle.moments,
        "return_quantiles.csv": bundle.quantiles,
        "tail_rates.csv": bundle.tail_rates,
        "intraday_acf.csv": bundle.intraday_acf,
        "daily_acf.csv": bundle.daily_acf,
        "daily_ljung_box.csv": (
            bundle.daily_ljung_box
        ),
    }

    for filename, table in outputs.items():
        table.to_csv(
            table_dir / filename,
            index=False,
        )

    metadata = {
        "analysis": (
            "day05_stylized_facts_v1"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "bar_feature_rows": int(
            len(bar_features)
        ),
        "session_feature_rows": int(
            len(session_features)
        ),
        "symbols": sorted(
            str(symbol)
            for symbol in bar_features[
                "symbol"
            ].dropna().unique()
        ),
        "tables": {
            filename: int(len(table))
            for filename, table in outputs.items()
        },
        "interpretation_warning": (
            "Large samples make formal normality tests "
            "highly sensitive. Economic interpretation "
            "must use skewness, kurtosis, quantiles and "
            "tail frequencies alongside p-values."
        ),
    }

    (
        output_dir
        / "analysis_metadata.json"
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
        "===== D05-T02B STYLIZED FACTS ====="
    )

    selected = bundle.moments[
        [
            "symbol",
            "frequency",
            "return_type",
            "observations",
            "mean",
            "standard_deviation",
            "skewness",
            "excess_kurtosis",
            "jarque_bera_pvalue",
        ]
    ]

    print("\nDistribution summary:")
    print(
        selected.to_string(
            index=False
        )
    )

    volatility_clustering = (
        bundle.daily_ljung_box.loc[
            (
                bundle.daily_ljung_box[
                    "transformation"
                ]
                .isin(["absolute", "squared"])
            )
            & (
                bundle.daily_ljung_box[
                    "lag"
                ].eq(20)
            ),
            [
                "symbol",
                "transformation",
                "ljung_box_statistic",
                "ljung_box_pvalue",
                "reject_at_5pct",
            ],
        ]
    )

    print(
        "\nDaily volatility-clustering evidence:"
    )

    print(
        volatility_clustering.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(output_dir.resolve())

    print("\nD05-T02B PASSED")


if __name__ == "__main__":
    main()
