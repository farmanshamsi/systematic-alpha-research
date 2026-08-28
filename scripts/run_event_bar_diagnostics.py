"""Run the D05-T03 event-bar feasibility audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from systematic_alpha.analysis.event_bar_diagnostics import (
    build_event_bar_diagnostics,
)
from systematic_alpha.data.local_store import LocalParquetStore


DEFAULT_DATASET_ID = (
    "spy_trades_2025-12-15_"
    "1430_1431_iex_canonical"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit time, tick, volume and dollar "
            "bar mechanics on canonical trades."
        )
    )

    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
    )

    parser.add_argument(
        "--target-bar-count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/day05/"
            "event_bar_diagnostics_v1"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute the engineering audit."""

    args = parse_args()

    store = LocalParquetStore.from_project_config(
        tier="processed",
    )

    trades = store.read(
        dataset_kind="trades",
        dataset_id=args.dataset_id,
        verify_hash=True,
    )

    manifest = store.read_manifest(
        dataset_kind="trades",
        dataset_id=args.dataset_id,
    )

    bundle = build_event_bar_diagnostics(
        trades,
        target_bar_count=(
            args.target_bar_count
        ),
    )

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    table_dir = output_dir / "tables"

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bar_outputs = {
        "time_bars.parquet": (
            bundle.time_bars
        ),
        "tick_bars.parquet": (
            bundle.tick_bars
        ),
        "volume_bars.parquet": (
            bundle.volume_bars
        ),
        "dollar_bars.parquet": (
            bundle.dollar_bars
        ),
    }

    for filename, frame in (
        bar_outputs.items()
    ):
        frame.to_parquet(
            data_dir / filename,
            index=False,
        )

    table_outputs = {
        "thresholds.csv": (
            bundle.thresholds
        ),
        "sampling_comparison.csv": (
            bundle.comparison
        ),
        "conservation.csv": (
            bundle.conservation
        ),
        "sampling_decision.csv": (
            bundle.decision
        ),
    }

    for filename, table in (
        table_outputs.items()
    ):
        table.to_csv(
            table_dir / filename,
            index=False,
        )

    trade_count_pass = bool(
        bundle.conservation[
            "trade_count_error"
        ].eq(0).all()
    )

    volume_pass = bool(
        bundle.conservation[
            "volume_error"
        ].abs().le(1.0e-8).all()
    )

    notional_tolerance = (
        bundle.conservation[
            "input_dollar_value"
        ].abs()
        * 1.0e-12
    ).clip(lower=1.0e-8)

    dollar_pass = bool(
        bundle.conservation[
            "dollar_value_error"
        ]
        .abs()
        .le(notional_tolerance)
        .all()
    )

    assert trade_count_pass
    assert volume_pass
    assert dollar_pass

    metadata = {
        "analysis": (
            "day05_event_bar_diagnostics_v1"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_dataset_id": (
            args.dataset_id
        ),
        "source_sha256": manifest[
            "sha256"
        ],
        "source_trade_count": int(
            len(trades)
        ),
        "source_minimum_timestamp": str(
            trades["timestamp"].min()
        ),
        "source_maximum_timestamp": str(
            trades["timestamp"].max()
        ),
        "target_bar_count": int(
            args.target_bar_count
        ),
        "evidence_classification": (
            "engineering smoke test only"
        ),
        "statistical_inference_allowed": (
            False
        ),
        "primary_sampling_decision": (
            "Retain 15-minute SIP time bars."
        ),
        "event_bar_candidate": (
            "Dollar bars, deferred until a "
            "representative multi-session "
            "trade sample exists."
        ),
        "threshold_warning": (
            "Thresholds were derived from the "
            "same one-minute source sample solely "
            "to compare bar mechanics. They must "
            "not be used as strategy parameters."
        ),
        "conservation_gate": {
            "trade_count_pass": (
                trade_count_pass
            ),
            "volume_pass": volume_pass,
            "dollar_value_pass": (
                dollar_pass
            ),
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
        "===== D05-T03 EVENT-BAR AUDIT ====="
    )

    print("\nThresholds:")
    print(
        bundle.thresholds.to_string(
            index=False
        )
    )

    print("\nSampling comparison:")
    print(
        bundle.comparison.to_string(
            index=False
        )
    )

    print("\nConservation:")
    print(
        bundle.conservation.to_string(
            index=False
        )
    )

    print("\nControlled decision:")
    print(
        bundle.decision.to_string(
            index=False
        )
    )

    print("\nEvidence classification:")
    print("Engineering smoke test only")

    print("\nOutputs:")
    print(output_dir.resolve())

    print("\nD05-T03 PASSED")


if __name__ == "__main__":
    main()
