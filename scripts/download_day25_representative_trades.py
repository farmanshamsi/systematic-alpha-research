"""Download the predeclared five-session SPY IEX trade sample."""

from __future__ import annotations

from typing import Final

import pandas as pd

from systematic_alpha.data.alpaca_microstructure import AlpacaMicrostructureProvider
from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.local_store import LocalParquetStore


RAW_DATASET_ID: Final[str] = "spy_trades_2025_predeclared_five_session_iex_v1_raw"
CANONICAL_DATASET_ID: Final[str] = (
    "spy_trades_2025_predeclared_five_session_iex_v1_canonical"
)
SESSION_WINDOWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("2025-01-15", "2025-01-15T14:30:00Z", "2025-01-15T21:00:00Z"),
    ("2025-04-15", "2025-04-15T13:30:00Z", "2025-04-15T20:00:00Z"),
    ("2025-07-15", "2025-07-15T13:30:00Z", "2025-07-15T20:00:00Z"),
    ("2025-10-15", "2025-10-15T13:30:00Z", "2025-10-15T20:00:00Z"),
    ("2025-12-15", "2025-12-15T14:30:00Z", "2025-12-15T21:00:00Z"),
)


def main() -> None:
    config = load_project_config()
    provider = AlpacaMicrostructureProvider(config=config)
    raw_parts: list[pd.DataFrame] = []
    canonical_parts: list[pd.DataFrame] = []
    request_records: list[dict[str, object]] = []
    for session_date, start, end in SESSION_WINDOWS:
        result = provider.fetch_trades_bundle(
            symbols=["SPY"],
            start=start,
            end=end,
        )
        raw = result.raw.copy(deep=True)
        raw["requested_session_date"] = session_date
        normalized = result.normalized.copy(deep=True)
        normalized["requested_session_date"] = session_date
        raw_parts.append(raw)
        canonical_parts.append(normalized)
        request_records.append(
            {
                "session_date": session_date,
                "start_utc": start,
                "end_utc": end,
                "rows": len(normalized),
            }
        )
    raw_frame = pd.concat(raw_parts, ignore_index=True)
    canonical_frame = pd.concat(canonical_parts, ignore_index=True)
    raw_store = LocalParquetStore.from_project_config(config, tier="raw")
    processed_store = LocalParquetStore.from_project_config(config, tier="processed")
    raw_result = raw_store.write(
        raw_frame,
        dataset_kind="trades",
        dataset_id=RAW_DATASET_ID,
        schema_version="alpaca-stock-trades-five-session-v1",
        metadata={
            "provider": "alpaca",
            "feed": "iex",
            "symbol": "SPY",
            "purpose": "Predeclared representative event-time experiment",
            "selection_rule": "Fifteenth calendar day in five spaced 2025 months",
            "requests": request_records,
            "locked_period_accessed": False,
        },
    )
    canonical_result = processed_store.write(
        canonical_frame,
        dataset_kind="trades",
        dataset_id=CANONICAL_DATASET_ID,
        schema_version="canonical-trades-five-session-v1",
        metadata={
            "provider": "alpaca",
            "feed": "iex",
            "symbol": "SPY",
            "purpose": "Predeclared representative event-time experiment",
            "selection_rule": "Fifteenth calendar day in five spaced 2025 months",
            "requests": request_records,
            "source_dataset_id": RAW_DATASET_ID,
            "source_sha256": raw_result.sha256,
            "normalization": "systematic_alpha.data.schemas.normalize_trades",
            "locked_period_accessed": False,
        },
    )
    print("===== DAY 25 REPRESENTATIVE TRADES =====")
    print("Raw rows:", raw_result.row_count)
    print("Canonical rows:", canonical_result.row_count)
    print("Canonical SHA256:", canonical_result.sha256)
    print("Locked period accessed: False")


if __name__ == "__main__":
    main()
