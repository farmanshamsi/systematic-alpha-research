"""Validate and resample the Day 04 canonical SPY sample."""

from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.resampling import resample_bars
from systematic_alpha.data.validators import assert_valid_bars


SOURCE_DATASET_ID = "spy_1min_2025-12-15_iex_canonical"


def main() -> None:
    config = load_project_config()

    store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    source = store.read(
        dataset_kind="bars",
        dataset_id=SOURCE_DATASET_ID,
    )

    source_manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=SOURCE_DATASET_ID,
    )

    report = assert_valid_bars(
        source,
        expected_minutes=1,
        exchange_timezone=config["market"]["exchange_timezone"],
    )

    print("Source validation passed:", report.passed)
    print("Source rows:", report.row_count)
    print("Internal missing bars:", report.internal_missing_bars)

    for minutes in (15, 30, 60):
        processed = resample_bars(
            source,
            timeframe_minutes=minutes,
        )

        artifact = store.write(
            processed,
            dataset_kind="bars",
            dataset_id=f"spy_{minutes}min_2025-12-15_iex",
            schema_version=f"canonical-bars-{minutes}min-v1",
            metadata={
                "source_dataset_id": SOURCE_DATASET_ID,
                "source_sha256": source_manifest["sha256"],
                "resampling_minutes": minutes,
                "aggregation": {
                    "open": "first",
                    "high": "maximum",
                    "low": "minimum",
                    "close": "last",
                    "volume": "sum",
                    "trade_count": "sum",
                    "vwap": "volume_weighted",
                },
            },
        )

        print(
            f"{minutes}-minute bars:",
            artifact.row_count,
            artifact.data_path,
        )


if __name__ == "__main__":
    main()
