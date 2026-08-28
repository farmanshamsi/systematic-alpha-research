"""Download and preserve a small Alpaca historical-bar sample."""

from systematic_alpha.data.alpaca_provider import AlpacaBarProvider
from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.local_store import LocalParquetStore


def main() -> None:
    symbols = ["SPY"]
    start = "2025-12-15T14:30:00Z"
    end = "2025-12-15T21:00:00Z"

    config = load_project_config()
    provider = AlpacaBarProvider(config=config)

    raw_store = LocalParquetStore.from_project_config(
        config,
        tier="raw",
    )
    processed_store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    result = provider.fetch_bars_bundle(
        symbols=symbols,
        start=start,
        end=end,
        timeframe_minutes=1,
    )

    raw_artifact = raw_store.write(
        result.raw,
        dataset_kind="bars",
        dataset_id="spy_1min_2025-12-15_iex_raw",
        schema_version="alpaca-stock-bars-v1",
        metadata={
            **result.request_metadata,
            "purpose": "Day 04 historical-bar ingestion test",
            "representation": (
                "Provider-shaped Alpaca frame with index "
                "materialized as columns"
            ),
        },
    )

    canonical_artifact = processed_store.write(
        result.normalized,
        dataset_kind="bars",
        dataset_id="spy_1min_2025-12-15_iex_canonical",
        schema_version="canonical-bars-v1",
        metadata={
            **result.request_metadata,
            "source_dataset": str(raw_artifact.data_path),
            "source_sha256": raw_artifact.sha256,
            "normalization": (
                "systematic_alpha.data.schemas.normalize_bars"
            ),
        },
    )

    print("Raw rows:", raw_artifact.row_count)
    print("Raw data:", raw_artifact.data_path)
    print("Raw manifest:", raw_artifact.manifest_path)
    print("Raw SHA256:", raw_artifact.sha256)
    print()
    print("Canonical rows:", canonical_artifact.row_count)
    print("Canonical data:", canonical_artifact.data_path)
    print("Canonical manifest:", canonical_artifact.manifest_path)
    print("Canonical SHA256:", canonical_artifact.sha256)


if __name__ == "__main__":
    main()
