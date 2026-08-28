"""Run Day 05 return construction and data-quality analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from systematic_alpha.analysis.eda_features import (
    assert_analysis_ready,
    build_return_features,
)
from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.local_store import LocalParquetStore


DEFAULT_DATASET_ID = (
    "spy_qqq_iwm_15min_"
    "2020-01-02_2025-12-31_"
    "sip_v3_development_canonical"
)

EXPECTED_SYMBOLS = ("SPY", "QQQ", "IWM")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Construct Day 05 return features and "
            "data-quality evidence."
        )
    )

    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Canonical development dataset ID.",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/day05/eda_v1",
        help="Directory for analytical outputs.",
    )

    return parser.parse_args()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write formatted JSON."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the analytical pipeline."""

    args = parse_args()

    config = load_project_config()

    store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    bars = store.read(
        dataset_kind="bars",
        dataset_id=args.dataset_id,
        verify_hash=True,
    )

    manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=args.dataset_id,
    )

    quality = assert_analysis_ready(bars)

    features = build_return_features(
        bars,
        expected_symbols=EXPECTED_SYMBOLS,
    )

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

    quality.integrity.to_csv(
        table_dir / "data_integrity.csv",
        index=False,
    )

    quality.missingness.to_csv(
        table_dir / "column_missingness.csv",
        index=False,
    )

    quality.symbols.to_csv(
        table_dir / "symbol_coverage.csv",
        index=False,
    )

    quality.sessions.to_csv(
        table_dir / "session_coverage.csv",
        index=False,
    )

    features.bars.to_parquet(
        data_dir / "bar_return_features.parquet",
        index=False,
    )

    features.sessions.to_parquet(
        data_dir / "session_return_features.parquet",
        index=False,
    )

    bars_by_symbol = {
        str(symbol): int(count)
        for symbol, count in (
            features.bars.groupby(
                "symbol",
                observed=True,
            )
            .size()
            .items()
        )
    }

    sessions_by_symbol = {
        str(symbol): int(count)
        for symbol, count in (
            features.sessions.groupby(
                "symbol",
                observed=True,
            )
            .size()
            .items()
        )
    }

    decomposition_error = (
        features.sessions[
            "log_return_decomposition_error"
        ]
        .abs()
        .max(skipna=True)
    )

    metadata = {
        "analysis": "day05_eda_returns_quality_v1",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_dataset_id": args.dataset_id,
        "source_sha256": manifest["sha256"],
        "source_rows": int(len(bars)),
        "feature_rows": int(
            len(features.bars)
        ),
        "session_rows": int(
            len(features.sessions)
        ),
        "symbols": list(EXPECTED_SYMBOLS),
        "bars_by_symbol": bars_by_symbol,
        "sessions_by_symbol": sessions_by_symbol,
        "minimum_timestamp": str(
            features.bars["timestamp"].min()
        ),
        "maximum_timestamp": str(
            features.bars["timestamp"].max()
        ),
        "maximum_log_decomposition_error": (
            None
            if pd.isna(decomposition_error)
            else float(decomposition_error)
        ),
        "return_definitions": {
            "close_to_close_simple": (
                "close_t / close_t-1 - 1"
            ),
            "close_to_close_log": (
                "log(close_t) - log(close_t-1)"
            ),
            "intraday_log": (
                "close-to-close log return only "
                "when both bars are in the same session"
            ),
            "overnight_log": (
                "log(first session open) - "
                "log(previous session close)"
            ),
            "regular_session_log": (
                "log(session close) - "
                "log(session open)"
            ),
        },
    }

    write_json(
        output_dir / "analysis_metadata.json",
        metadata,
    )

    print("===== D05-T02A OUTPUT =====")
    print("Source dataset:", args.dataset_id)
    print("Source SHA256:", manifest["sha256"])
    print("Bar feature rows:", len(features.bars))
    print(
        "Session feature rows:",
        len(features.sessions),
    )

    print("\nBars by symbol:")
    print(
        features.bars.groupby(
            "symbol",
            observed=True,
        ).size().to_string()
    )

    print("\nSessions by symbol:")
    print(
        features.sessions.groupby(
            "symbol",
            observed=True,
        ).size().to_string()
    )

    print(
        "\nMaximum log-return decomposition error:",
        decomposition_error,
    )

    print("\nOutputs:")
    print(output_dir.resolve())

    print("\nD05-T02A PASSED")


if __name__ == "__main__":
    main()
