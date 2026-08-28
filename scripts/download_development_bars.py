"""Download immutable monthly development-only 15-minute ETF bars."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from systematic_alpha.data.alpaca_provider import AlpacaBarProvider
from systematic_alpha.data.config_loader import load_project_config
from systematic_alpha.data.development_dataset import (
    DevelopmentChunk,
    build_monthly_development_chunks,
    combine_canonical_bar_chunks,
    filter_regular_session_bars,
    validate_chunk_edge_coverage,
    validate_complete_session_grid,
)
from systematic_alpha.data.local_store import (
    ImmutableStoreError,
    LocalParquetStore,
)
from systematic_alpha.data.sample_windows import SampleWindow


SYMBOLS = ("SPY", "QQQ", "IWM")
TIMEFRAME_MINUTES = 15
DATASET_KIND = "bars"
DATA_VERSION = "v3"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download monthly development-only "
            "SPY/QQQ/IWM 15-minute bars."
        )
    )

    parser.add_argument(
        "--month",
        action="append",
        help=(
            "Download one month in YYYY-MM format. "
            "May be supplied more than once."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help=(
            "Download all development months in this year. "
            "May be supplied more than once."
        ),
    )

    parser.add_argument(
        "--feed",
        choices=("iex", "sip"),
        help=(
            "Override the configured Alpaca stock-data feed "
            "for this downloader."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without contacting Alpaca.",
    )

    return parser.parse_args()


def universe_label() -> str:
    """Return the stable dataset-universe label."""

    return "_".join(
        symbol.lower()
        for symbol in SYMBOLS
    )


def monthly_dataset_ids(
    chunk: DevelopmentChunk,
    *,
    feed: str,
) -> tuple[str, str]:
    """Return raw and canonical IDs for one month."""

    prefix = (
        f"{universe_label()}_"
        f"{TIMEFRAME_MINUTES}min_"
        f"{chunk.label}_"
        f"{feed}_"
        f"{DATA_VERSION}"
    )

    return (
        f"{prefix}_raw",
        f"{prefix}_canonical",
    )


def combined_dataset_id(*, feed: str) -> str:
    """Return the full development-dataset ID."""

    return (
        f"{universe_label()}_"
        f"{TIMEFRAME_MINUTES}min_"
        "2020-01-02_2025-12-31_"
        f"{feed}_{DATA_VERSION}_"
        "development_canonical"
    )


def load_existing(
    store: LocalParquetStore,
    *,
    dataset_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load and hash-verify an existing artifact."""

    try:
        manifest = store.read_manifest(
            dataset_kind=DATASET_KIND,
            dataset_id=dataset_id,
        )
    except ImmutableStoreError as exc:
        if "Manifest not found:" not in str(exc):
            raise

        return None

    frame = store.read(
        dataset_kind=DATASET_KIND,
        dataset_id=dataset_id,
        verify_hash=True,
    )

    return frame, manifest


def extract_bundle_frames(
    bundle: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract raw and canonical DataFrames from a provider bundle."""

    frames_by_name: dict[str, pd.DataFrame] = {}

    common_names = (
        "raw",
        "raw_frame",
        "provider",
        "provider_frame",
        "provider_data",
        "canonical",
        "canonical_frame",
        "canonical_data",
        "normalized",
        "normalized_frame",
        "normalized_data",
    )

    for name in common_names:
        value = getattr(bundle, name, None)

        if isinstance(value, pd.DataFrame):
            frames_by_name[name] = value

    if is_dataclass(bundle):
        for field in fields(bundle):
            value = getattr(bundle, field.name)

            if isinstance(value, pd.DataFrame):
                frames_by_name.setdefault(
                    field.name,
                    value,
                )

    try:
        attributes = vars(bundle)
    except TypeError:
        attributes = {}

    for name, value in attributes.items():
        if isinstance(value, pd.DataFrame):
            frames_by_name.setdefault(name, value)

    raw: pd.DataFrame | None = None
    canonical: pd.DataFrame | None = None

    for name in (
        "raw",
        "raw_frame",
        "provider",
        "provider_frame",
        "provider_data",
    ):
        candidate = frames_by_name.get(name)

        if candidate is not None:
            raw = candidate
            break

    for name in (
        "canonical",
        "canonical_frame",
        "canonical_data",
        "normalized",
        "normalized_frame",
        "normalized_data",
    ):
        candidate = frames_by_name.get(name)

        if candidate is not None:
            canonical = candidate
            break

    canonical_columns = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "source",
        "feed",
    }

    if canonical is None:
        for candidate in frames_by_name.values():
            if canonical_columns.issubset(
                candidate.columns
            ):
                canonical = candidate
                break

    if raw is None and canonical is not None:
        for candidate in frames_by_name.values():
            if candidate is not canonical:
                raw = candidate
                break

    if raw is None or canonical is None:
        discovered = {
            name: list(frame.columns)
            for name, frame in frames_by_name.items()
        }

        raise TypeError(
            "Could not identify both raw and canonical "
            "DataFrames in BarFetchResult. "
            f"Discovered attributes: {discovered}"
        )

    if raw is canonical:
        raise TypeError(
            "Raw and canonical frames resolved "
            "to the same DataFrame."
        )

    return raw.copy(), canonical.copy()


def select_chunks(
    all_chunks: list[DevelopmentChunk],
    *,
    requested_months: list[str] | None,
    requested_years: list[int] | None,
) -> list[DevelopmentChunk]:
    """Select and validate requested development months."""

    selected = list(all_chunks)

    if requested_months:
        month_set = {
            str(month).strip()
            for month in requested_months
        }

        valid_months = {
            chunk.label
            for chunk in all_chunks
        }

        invalid_months = month_set.difference(
            valid_months
        )

        if invalid_months:
            raise ValueError(
                "Requested months are outside the "
                "development sample: "
                f"{sorted(invalid_months)}"
            )

        selected = [
            chunk
            for chunk in selected
            if chunk.label in month_set
        ]

    if requested_years:
        year_set = set(requested_years)

        valid_years = {
            int(chunk.label[:4])
            for chunk in all_chunks
        }

        invalid_years = year_set.difference(
            valid_years
        )

        if invalid_years:
            raise ValueError(
                "Requested years are outside the "
                "development sample: "
                f"{sorted(invalid_years)}"
            )

        selected = [
            chunk
            for chunk in selected
            if int(chunk.label[:4]) in year_set
        ]

    if not selected:
        raise ValueError(
            "No development chunks matched the request."
        )

    return selected


def print_chunk_plan(
    chunk: DevelopmentChunk,
    *,
    feed: str,
) -> None:
    """Print one monthly request plan."""

    raw_id, canonical_id = monthly_dataset_ids(
        chunk,
        feed=feed,
    )

    print(
        f"{chunk.label}: "
        f"{chunk.start.isoformat()} -> "
        f"{chunk.end_exclusive.isoformat()}"
    )
    print("  raw:", raw_id)
    print("  canonical:", canonical_id)


def fetch_month(
    *,
    chunk: DevelopmentChunk,
    provider: AlpacaBarProvider,
    window: SampleWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch each symbol separately and validate the month."""

    raw_frames: list[pd.DataFrame] = []
    canonical_frames: list[pd.DataFrame] = []

    start, end = chunk.request_bounds()

    for symbol in SYMBOLS:
        print(
            f"{chunk.label}: requesting {symbol}..."
        )

        bundle = provider.fetch_bars_bundle(
            symbols=[symbol],
            start=start,
            end=end,
            timeframe_minutes=TIMEFRAME_MINUTES,
        )

        raw, canonical = extract_bundle_frames(
            bundle
        )

        if raw.empty:
            raise RuntimeError(
                f"{chunk.label} {symbol}: "
                "provider-shaped response was empty."
            )

        if canonical.empty:
            raise RuntimeError(
                f"{chunk.label} {symbol}: "
                "canonical response was empty."
            )

        regular_session = filter_regular_session_bars(
            canonical
        )

        actual_symbols = set(
            regular_session["symbol"]
            .astype("string")
            .str.upper()
            .dropna()
        )

        if actual_symbols != {symbol}:
            raise RuntimeError(
                f"{chunk.label} {symbol}: "
                "response-universe mismatch. "
                f"Actual symbols: {sorted(actual_symbols)}"
            )

        raw_frames.append(raw)
        canonical_frames.append(regular_session)

        print(
            f"{chunk.label} {symbol}: "
            f"raw rows={len(raw)}; "
            f"regular-session rows="
            f"{len(regular_session)}"
        )

    combined_raw = pd.concat(
        raw_frames,
        ignore_index=True,
        sort=False,
    )

    combined_canonical = combine_canonical_bar_chunks(
        canonical_frames,
        window=window,
        expected_symbols=SYMBOLS,
    )

    validate_complete_session_grid(
        combined_canonical,
        chunk=chunk,
        expected_symbols=SYMBOLS,
        timeframe_minutes=TIMEFRAME_MINUTES,
    )

    validate_chunk_edge_coverage(
        combined_canonical,
        chunk=chunk,
        expected_symbols=SYMBOLS,
    )

    return combined_raw, combined_canonical


def ensure_month(
    *,
    chunk: DevelopmentChunk,
    provider: AlpacaBarProvider,
    raw_store: LocalParquetStore,
    processed_store: LocalParquetStore,
    window: SampleWindow,
    feed: str,
) -> pd.DataFrame:
    """Load or create one immutable V3 monthly chunk."""

    raw_id, canonical_id = monthly_dataset_ids(
        chunk,
        feed=feed,
    )

    existing_raw = load_existing(
        raw_store,
        dataset_id=raw_id,
    )

    existing_canonical = load_existing(
        processed_store,
        dataset_id=canonical_id,
    )

    if (
        existing_raw is None
        and existing_canonical is not None
    ) or (
        existing_raw is not None
        and existing_canonical is None
    ):
        raise RuntimeError(
            "Partial immutable V3 state detected for "
            f"{chunk.label}. Raw and canonical artifacts "
            "must both exist or both be absent."
        )

    if (
        existing_raw is not None
        and existing_canonical is not None
    ):
        canonical, manifest = existing_canonical

        canonical = combine_canonical_bar_chunks(
            [canonical],
            window=window,
            expected_symbols=SYMBOLS,
        )

        validate_complete_session_grid(
            canonical,
            chunk=chunk,
            expected_symbols=SYMBOLS,
            timeframe_minutes=TIMEFRAME_MINUTES,
        )

        validate_chunk_edge_coverage(
            canonical,
            chunk=chunk,
            expected_symbols=SYMBOLS,
        )

        print(
            f"{chunk.label}: existing V3 chunk "
            "passed hash and coverage verification; "
            f"rows={len(canonical)}; "
            f"sha256={manifest['sha256']}"
        )

        return canonical

    combined_raw, combined_canonical = fetch_month(
        chunk=chunk,
        provider=provider,
        window=window,
    )

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    raw_artifact = raw_store.write(
        combined_raw,
        dataset_kind=DATASET_KIND,
        dataset_id=raw_id,
        schema_version="alpaca-bars-raw-v3",
        metadata={
            "provider": "alpaca",
            "feed": feed,
            "symbols": list(SYMBOLS),
            "request_mode": "one-symbol-at-a-time",
            "timeframe_minutes": TIMEFRAME_MINUTES,
            "chunk_type": "calendar_month",
            "chunk_label": chunk.label,
            "start_utc": chunk.start.isoformat(),
            "end_utc_exclusive": (
                chunk.end_exclusive.isoformat()
            ),
            "retrieved_at_utc": retrieved_at,
            "sample_role": "development",
            "representation": "provider-shaped",
            "data_version": DATA_VERSION,
            "purpose": "Day 05 EDA development baseline",
        },
    )

    canonical_artifact = processed_store.write(
        combined_canonical,
        dataset_kind=DATASET_KIND,
        dataset_id=canonical_id,
        schema_version="canonical-bars-v3",
        metadata={
            "provider": "alpaca",
            "feed": feed,
            "symbols": list(SYMBOLS),
            "request_mode": "one-symbol-at-a-time",
            "timeframe_minutes": TIMEFRAME_MINUTES,
            "session_filter": (
                "XNYS exchange calendar; official session open "
                "inclusive and official close exclusive"
            ),
            "chunk_type": "calendar_month",
            "chunk_label": chunk.label,
            "start_utc": chunk.start.isoformat(),
            "end_utc_exclusive": (
                chunk.end_exclusive.isoformat()
            ),
            "retrieved_at_utc": retrieved_at,
            "sample_role": "development",
            "representation": "canonical",
            "data_version": DATA_VERSION,
            "source_dataset_id": raw_id,
            "source_sha256": raw_artifact.sha256,
            "transformation": (
                "canonical normalization followed by "
                "regular-session filtering"
            ),
            "purpose": "Day 05 EDA development baseline",
        },
    )

    print(
        f"{chunk.label}: stored V3 raw rows="
        f"{raw_artifact.row_count}; "
        f"canonical rows="
        f"{canonical_artifact.row_count}"
    )

    print(
        f"{chunk.label}: canonical sha256="
        f"{canonical_artifact.sha256}"
    )

    return combined_canonical


def try_build_combined_dataset(
    *,
    chunks: list[DevelopmentChunk],
    processed_store: LocalParquetStore,
    window: SampleWindow,
    feed: str,
) -> None:
    """Assemble all 72 verified months when available."""

    frames: list[pd.DataFrame] = []
    source_ids: list[str] = []
    source_hashes: list[str] = []
    missing_months: list[str] = []

    for chunk in chunks:
        _, canonical_id = monthly_dataset_ids(
            chunk,
            feed=feed,
        )

        existing = load_existing(
            processed_store,
            dataset_id=canonical_id,
        )

        if existing is None:
            missing_months.append(chunk.label)
            continue

        frame, manifest = existing

        frames.append(frame)
        source_ids.append(canonical_id)
        source_hashes.append(manifest["sha256"])

    if missing_months:
        print(
            "\nCombined V3 development dataset pending."
        )
        print(
            "Available months:",
            len(chunks) - len(missing_months),
        )
        print("Missing months:", len(missing_months))
        print(
            "Next missing months:",
            missing_months[:12],
        )
        return

    dataset_id = combined_dataset_id(
        feed=feed
    )

    existing_combined = load_existing(
        processed_store,
        dataset_id=dataset_id,
    )

    if existing_combined is not None:
        combined, manifest = existing_combined

        print(
            "\nCombined V3 dataset already exists "
            "and passed hash verification."
        )
        print("Rows:", len(combined))
        print("SHA256:", manifest["sha256"])
        return

    combined = combine_canonical_bar_chunks(
        frames,
        window=window,
        expected_symbols=SYMBOLS,
    )

    artifact = processed_store.write(
        combined,
        dataset_kind=DATASET_KIND,
        dataset_id=dataset_id,
        schema_version="development-bars-v3",
        metadata={
            "provider": "alpaca",
            "feed": feed,
            "symbols": list(SYMBOLS),
            "timeframe_minutes": TIMEFRAME_MINUTES,
            "session_filter": (
                "XNYS exchange calendar; official session open "
                "inclusive and official close exclusive"
            ),
            "development_start": (
                window.development_start.isoformat()
            ),
            "development_end_exclusive": (
                window.development_end_exclusive.isoformat()
            ),
            "sample_role": "development",
            "data_version": DATA_VERSION,
            "source_dataset_ids": source_ids,
            "source_sha256": source_hashes,
            "transformation": (
                "chronological concatenation of "
                "72 hash-verified monthly chunks"
            ),
            "purpose": "Day 05 EDA development baseline",
        },
    )

    print("\nCombined V3 development dataset stored.")
    print("Rows:", artifact.row_count)
    print("Data:", artifact.data_path)
    print("Manifest:", artifact.manifest_path)
    print("SHA256:", artifact.sha256)

    print("\nRows by symbol:")
    print(
        combined.groupby(
            "symbol",
            observed=True,
        ).size()
    )

    print("\nTimestamp range:")
    print("Minimum:", combined["timestamp"].min())
    print("Maximum:", combined["timestamp"].max())


def main() -> None:
    """Execute the monthly development download."""

    args = parse_args()

    config = load_project_config()
    window = SampleWindow.from_project_config(
        config
    )

    configured_feed = str(
        config["broker"].get(
            "stock_data_feed",
            "iex",
        )
    ).lower()

    feed = (
        args.feed.lower()
        if args.feed
        else configured_feed
    )

    if feed != "sip":
        raise ValueError(
            "The V3 development baseline requires the "
            "Alpaca SIP feed. Run with --feed sip."
        )

    provider_config = deepcopy(config)
    provider_config["broker"]["stock_data_feed"] = feed

    all_chunks = build_monthly_development_chunks(
        window
    )

    selected_chunks = select_chunks(
        all_chunks,
        requested_months=args.month,
        requested_years=args.year,
    )

    print("===== V3 MONTHLY DEVELOPMENT PLAN =====")
    print("Symbols:", list(SYMBOLS))
    print("Timeframe:", TIMEFRAME_MINUTES, "minutes")
    print("Feed:", feed)
    print("Request mode: one symbol at a time")
    print("Regular session only: True")
    print("Development only: True")
    print("Selected months:", len(selected_chunks))

    for chunk in selected_chunks:
        print_chunk_plan(
            chunk,
            feed=feed,
        )

    if args.dry_run:
        print(
            "\nDry run complete. "
            "No API requests or writes were performed."
        )
        return

    provider = AlpacaBarProvider(
        config=provider_config
    )

    raw_store = (
        LocalParquetStore.from_project_config(
            config,
            tier="raw",
        )
    )

    processed_store = (
        LocalParquetStore.from_project_config(
            config,
            tier="processed",
        )
    )

    for chunk in selected_chunks:
        ensure_month(
            chunk=chunk,
            provider=provider,
            raw_store=raw_store,
            processed_store=processed_store,
            window=window,
            feed=feed,
        )

    try_build_combined_dataset(
        chunks=all_chunks,
        processed_store=processed_store,
        window=window,
        feed=feed,
    )


if __name__ == "__main__":
    main()
