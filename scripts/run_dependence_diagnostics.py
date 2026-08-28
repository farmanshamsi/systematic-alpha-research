"""Run D05-T02D dependence and joint-tail diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from systematic_alpha.analysis.dependence_diagnostics import (
    build_dependence_diagnostics,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate Day 05 linear, rank and "
            "joint-tail dependence."
        )
    )

    parser.add_argument(
        "--eda-dir",
        default="artifacts/day05/eda_v1",
    )

    parser.add_argument(
        "--volatility-dir",
        default=(
            "artifacts/day05/"
            "volatility_seasonality_v1"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/day05/"
            "dependence_v1"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute D05-T02D."""

    args = parse_args()

    eda_dir = Path(args.eda_dir)
    volatility_dir = Path(
        args.volatility_dir
    )
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sessions = pd.read_parquet(
        eda_dir
        / "data"
        / "session_return_features.parquet"
    )

    daily_volatility = pd.read_parquet(
        volatility_dir
        / "data"
        / "daily_volatility.parquet"
    )

    bundle = build_dependence_diagnostics(
        sessions,
        daily_volatility,
    )

    outputs = {
        "pairwise_dependence.csv": (
            bundle.pairwise_dependence
        ),
        "tail_dependence.csv": (
            bundle.tail_dependence
        ),
        "rolling_dependence.csv": (
            bundle.rolling_dependence
        ),
        "regime_dependence.csv": (
            bundle.regime_dependence
        ),
        "regime_tail_dependence.csv": (
            bundle.regime_tail_dependence
        ),
        "regime_definition.csv": (
            bundle.regime_definition
        ),
    }

    for filename, table in outputs.items():
        table.to_csv(
            table_dir / filename,
            index=False,
        )

    metadata = {
        "analysis": (
            "day05_dependence_diagnostics_v1"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "session_rows": int(len(sessions)),
        "daily_volatility_rows": int(
            len(daily_volatility)
        ),
        "tail_probabilities": [
            0.01,
            0.05,
            0.10,
        ],
        "rolling_windows": [
            63,
            126,
        ],
        "stress_definition": (
            "SPY annualized total realized "
            "volatility at or above its development-"
            "sample 80th percentile"
        ),
        "warning": (
            "These are descriptive dependence "
            "diagnostics. They do not establish "
            "cointegration or tradable mean reversion."
        ),
        "tables": {
            filename: int(len(table))
            for filename, table in outputs.items()
        },
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
        "===== D05-T02D DEPENDENCE DIAGNOSTICS ====="
    )

    print("\nFull-development dependence:")
    print(
        bundle.pairwise_dependence.to_string(
            index=False
        )
    )

    print("\nFive-percent tail co-exceedance:")
    print(
        bundle.tail_dependence.loc[
            bundle.tail_dependence[
                "tail_probability"
            ].eq(0.05)
        ].to_string(index=False)
    )

    print("\nRegime definition:")
    print(
        bundle.regime_definition.to_string(
            index=False
        )
    )

    print("\nDependence by volatility regime:")
    print(
        bundle.regime_dependence.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(output_dir.resolve())

    print("\nD05-T02D PASSED")


if __name__ == "__main__":
    main()
