"""Reconcile an Alpaca IEX daily aggregate against Yahoo Finance."""

from __future__ import annotations

from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.external_reconciliation import (
    aggregate_intraday_to_daily,
    canonical_daily_to_comparison,
    reconcile_daily_ohlcv,
)
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.yahoo_provider import YahooDailyProvider


SYMBOL = "SPY"
SESSION_DATE = "2025-12-15"
YAHOO_END_EXCLUSIVE = "2025-12-16"

PRIMARY_INTRADAY_ID = (
    "spy_1min_2025-12-15_iex_canonical"
)

PRIMARY_DAILY_ID = (
    "spy_2025-12-15_alpaca_iex_daily"
)

YAHOO_RAW_ID = (
    "spy_2025-12-15_yahoo_daily_raw"
)

YAHOO_CANONICAL_ID = (
    "spy_2025-12-15_yahoo_daily_canonical"
)

RECONCILIATION_ID = (
    "spy_2025-12-15_alpaca_iex_vs_yahoo"
)


def main() -> None:
    config = load_project_config()

    raw_store = LocalParquetStore.from_project_config(
        config,
        tier="raw",
    )

    processed_store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )

    # ---------------------------------------------------------
    # Primary source: Alpaca canonical one-minute IEX bars
    # ---------------------------------------------------------
    primary_intraday = processed_store.read(
        dataset_kind="bars",
        dataset_id=PRIMARY_INTRADAY_ID,
    )

    primary_intraday_manifest = processed_store.read_manifest(
        dataset_kind="bars",
        dataset_id=PRIMARY_INTRADAY_ID,
    )

    primary_daily = aggregate_intraday_to_daily(
        primary_intraday,
        exchange_timezone=(
            config["market"]["exchange_timezone"]
        ),
    )

    primary_daily_artifact = processed_store.write(
        primary_daily,
        dataset_kind="daily_bars",
        dataset_id=PRIMARY_DAILY_ID,
        schema_version="daily-ohlcv-comparison-v1",
        metadata={
            "provider": "alpaca",
            "feed": "iex",
            "symbol": SYMBOL,
            "session_date": SESSION_DATE,
            "source_dataset_id": PRIMARY_INTRADAY_ID,
            "source_dataset_kind": "bars",
            "source_sha256": (
                primary_intraday_manifest["sha256"]
            ),
            "aggregation": {
                "open": "first",
                "high": "maximum",
                "low": "minimum",
                "close": "last",
                "volume": "sum",
                "session_timezone": (
                    config["market"]["exchange_timezone"]
                ),
            },
            "adjustment_basis": "Alpaca Adjustment.ALL",
        },
    )

    # ---------------------------------------------------------
    # Independent source: Yahoo adjusted daily OHLCV
    # ---------------------------------------------------------
    yahoo_provider = YahooDailyProvider()

    yahoo_result = yahoo_provider.fetch_daily_bundle(
        symbols=[SYMBOL],
        start=SESSION_DATE,
        end=YAHOO_END_EXCLUSIVE,
    )

    yahoo_raw_artifact = raw_store.write(
        yahoo_result.raw,
        dataset_kind="daily_bars",
        dataset_id=YAHOO_RAW_ID,
        schema_version="yfinance-daily-raw-v1",
        metadata={
            **yahoo_result.request_metadata,
            "purpose": (
                "Independent external OHLCV reconciliation"
            ),
            "representation": (
                "Provider-shaped Yahoo Finance daily frame"
            ),
        },
    )

    yahoo_canonical_artifact = processed_store.write(
        yahoo_result.normalized,
        dataset_kind="daily_bars",
        dataset_id=YAHOO_CANONICAL_ID,
        schema_version="canonical-daily-bars-v1",
        metadata={
            **yahoo_result.request_metadata,
            "source_dataset_id": YAHOO_RAW_ID,
            "source_dataset_kind": "daily_bars",
            "source_sha256": yahoo_raw_artifact.sha256,
            "normalization": (
                "systematic_alpha.data.schemas.normalize_bars"
            ),
            "adjustment_basis": (
                "Yahoo auto_adjust=True"
            ),
        },
    )

    yahoo_daily = canonical_daily_to_comparison(
        yahoo_result.normalized,
    )

    # ---------------------------------------------------------
    # Structured outer-join reconciliation
    # ---------------------------------------------------------
    price_tolerance_bps = float(
        config["data"]["external_price_tolerance_bps"]
    )

    volume_tolerance_pct = float(
        config["data"]["external_volume_tolerance_pct"]
    )

    report = reconcile_daily_ohlcv(
        primary_daily,
        yahoo_daily,
        price_tolerance_bps=price_tolerance_bps,
        volume_tolerance_pct=volume_tolerance_pct,
    )

    report_artifact = processed_store.write(
        report,
        dataset_kind="reconciliation",
        dataset_id=RECONCILIATION_ID,
        schema_version="daily-ohlcv-reconciliation-v1",
        metadata={
            "symbol": SYMBOL,
            "session_date": SESSION_DATE,
            "primary_dataset_id": PRIMARY_DAILY_ID,
            "primary_dataset_kind": "daily_bars",
            "primary_sha256": (
                primary_daily_artifact.sha256
            ),
            "external_dataset_id": YAHOO_CANONICAL_ID,
            "external_dataset_kind": "daily_bars",
            "external_sha256": (
                yahoo_canonical_artifact.sha256
            ),
            "price_tolerance_bps": (
                price_tolerance_bps
            ),
            "volume_tolerance_pct": (
                volume_tolerance_pct
            ),
            "price_adjustment_alignment": {
                "primary": "Alpaca Adjustment.ALL",
                "external": "Yahoo auto_adjust=True",
            },
            "interpretation_note": (
                "Price and volume discrepancies are reported, "
                "not silently corrected. IEX and Yahoo may have "
                "different market-volume coverage."
            ),
        },
    )

    print("\n===== STORED ARTIFACTS =====")
    print(
        "Primary daily:",
        primary_daily_artifact.data_path,
    )
    print(
        "Yahoo raw:",
        yahoo_raw_artifact.data_path,
    )
    print(
        "Yahoo canonical:",
        yahoo_canonical_artifact.data_path,
    )
    print(
        "Reconciliation:",
        report_artifact.data_path,
    )

    print("\n===== RECONCILIATION RESULT =====")

    display_columns = [
        "session_date",
        "symbol",
        "open_primary",
        "open_external",
        "open_abs_bps",
        "high_primary",
        "high_external",
        "high_abs_bps",
        "low_primary",
        "low_external",
        "low_abs_bps",
        "close_primary",
        "close_external",
        "close_abs_bps",
        "volume_primary",
        "volume_external",
        "volume_pct_diff",
        "price_pass",
        "volume_pass",
        "overall_pass",
        "status",
    ]

    print(
        report[display_columns].to_string(index=False)
    )

    print("\nReconciliation SHA256:", report_artifact.sha256)


if __name__ == "__main__":
    main()
