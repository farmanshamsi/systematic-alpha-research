"""Build immutable event bars from the Day 04 SPY trade sample."""

from __future__ import annotations

import math

from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.resampling import (
    build_dollar_bars,
    build_tick_bars,
    build_volume_bars,
)


SOURCE_DATASET_ID = (
    "spy_trades_2025-12-15_1430_1431_iex_canonical"
)
TARGET_BAR_COUNT = 10


def main() -> None:
    store = LocalParquetStore.from_project_config(
        tier="processed",
    )

    trades = store.read(
        dataset_kind="trades",
        dataset_id=SOURCE_DATASET_ID,
    )

    source_manifest = store.read_manifest(
        dataset_kind="trades",
        dataset_id=SOURCE_DATASET_ID,
    )

    total_trades = len(trades)
    total_volume = float(trades["size"].sum())
    total_notional = float(
        (trades["price"] * trades["size"]).sum()
    )

    thresholds = {
        "tick": max(
            1,
            math.ceil(total_trades / TARGET_BAR_COUNT),
        ),
        "volume": max(
            1.0,
            total_volume / TARGET_BAR_COUNT,
        ),
        "dollar": max(
            1.0,
            total_notional / TARGET_BAR_COUNT,
        ),
    }

    datasets = {
        "tick": build_tick_bars(
            trades,
            trades_per_bar=int(thresholds["tick"]),
        ),
        "volume": build_volume_bars(
            trades,
            shares_per_bar=thresholds["volume"],
        ),
        "dollar": build_dollar_bars(
            trades,
            dollars_per_bar=thresholds["dollar"],
        ),
    }

    for bar_type, bars in datasets.items():
        threshold = thresholds[bar_type]

        artifact = store.write(
            bars,
            dataset_kind="event_bars",
            dataset_id=(
                "spy_2025-12-15_1430_1431_iex_"
                f"{bar_type}_bars"
            ),
            schema_version=f"event-{bar_type}-bars-v1",
            metadata={
                "source_dataset_id": SOURCE_DATASET_ID,
                "source_dataset_kind": "trades",
                "source_sha256": source_manifest["sha256"],
                "bar_type": bar_type,
                "threshold": threshold,
                "threshold_method": (
                    "source total divided by target bar count"
                ),
                "target_bar_count": TARGET_BAR_COUNT,
                "whole_trade_assignment": True,
                "retain_final_partial_bar": True,
                "input_trade_count": total_trades,
                "input_volume": total_volume,
                "input_dollar_value": total_notional,
                "output_bar_count": len(bars),
                "complete_bar_count": int(
                    bars["is_complete"].sum()
                ),
                "partial_bar_count": int(
                    (~bars["is_complete"]).sum()
                ),
            },
        )

        print(f"\n{bar_type.upper()} BARS")
        print("Rows:", artifact.row_count)
        print("Threshold:", threshold)
        print("Data:", artifact.data_path)
        print("Manifest:", artifact.manifest_path)
        print("SHA256:", artifact.sha256)


if __name__ == "__main__":
    main()
