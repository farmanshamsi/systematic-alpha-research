"""Download immutable raw and canonical quote/trade samples."""

from systematic_alpha.data.alpaca_microstructure import (
    AlpacaMicrostructureProvider,
)
from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.local_store import LocalParquetStore


def main() -> None:
    symbols = ["SPY"]
    start = "2025-12-15T14:30:00Z"
    end = "2025-12-15T14:31:00Z"

    config = load_project_config()
    provider = AlpacaMicrostructureProvider(config=config)

    raw_store = LocalParquetStore.from_project_config(
        config,
        tier="raw",
    )
    processed_store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    quote_result = provider.fetch_quotes_bundle(
        symbols=symbols,
        start=start,
        end=end,
    )

    trade_result = provider.fetch_trades_bundle(
        symbols=symbols,
        start=start,
        end=end,
    )

    quote_raw = raw_store.write(
        quote_result.raw,
        dataset_kind="quotes",
        dataset_id="spy_quotes_2025-12-15_1430_1431_iex_raw",
        schema_version="alpaca-stock-quotes-v1",
        metadata={
            **quote_result.request_metadata,
            "purpose": "Day 04 Level-1 quote ingestion test",
            "representation": (
                "Provider-shaped Alpaca frame with index "
                "materialized as columns"
            ),
        },
    )

    quote_canonical = processed_store.write(
        quote_result.normalized,
        dataset_kind="quotes",
        dataset_id=(
            "spy_quotes_2025-12-15_1430_1431_iex_canonical"
        ),
        schema_version="canonical-quotes-v1",
        metadata={
            **quote_result.request_metadata,
            "source_dataset": str(quote_raw.data_path),
            "source_sha256": quote_raw.sha256,
            "normalization": (
                "systematic_alpha.data.schemas.normalize_quotes"
            ),
        },
    )

    trade_raw = raw_store.write(
        trade_result.raw,
        dataset_kind="trades",
        dataset_id="spy_trades_2025-12-15_1430_1431_iex_raw",
        schema_version="alpaca-stock-trades-v1",
        metadata={
            **trade_result.request_metadata,
            "purpose": "Day 04 trade ingestion test",
            "representation": (
                "Provider-shaped Alpaca frame with index "
                "materialized as columns"
            ),
        },
    )

    trade_canonical = processed_store.write(
        trade_result.normalized,
        dataset_kind="trades",
        dataset_id=(
            "spy_trades_2025-12-15_1430_1431_iex_canonical"
        ),
        schema_version="canonical-trades-v1",
        metadata={
            **trade_result.request_metadata,
            "source_dataset": str(trade_raw.data_path),
            "source_sha256": trade_raw.sha256,
            "normalization": (
                "systematic_alpha.data.schemas.normalize_trades"
            ),
        },
    )

    print("Raw quote rows:", quote_raw.row_count)
    print("Canonical quote rows:", quote_canonical.row_count)
    print("Quote lineage hash:", quote_raw.sha256)
    print()
    print("Raw trade rows:", trade_raw.row_count)
    print("Canonical trade rows:", trade_canonical.row_count)
    print("Trade lineage hash:", trade_raw.sha256)


if __name__ == "__main__":
    main()
