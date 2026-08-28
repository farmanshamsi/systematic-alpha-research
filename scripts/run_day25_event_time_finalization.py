"""Run the predeclared representative event-time experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.analysis.event_time_finalization import (
    run_event_time_finalization,
    write_event_time_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root, load_project_config
from systematic_alpha.data.local_store import LocalParquetStore
DEFAULT_OUTPUT: Final[Path] = Path("artifacts/day25_event_time_finalization")
CANONICAL_DATASET_ID: Final[str] = (
    "spy_trades_2025_predeclared_five_session_iex_v1_canonical"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen five-session time/event-bar comparison."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = parse_args(argv)
    config = load_project_config()
    store = LocalParquetStore.from_project_config(config, tier="processed")
    trades = store.read(
        dataset_kind="trades",
        dataset_id=CANONICAL_DATASET_ID,
        verify_hash=True,
    )
    manifest = store.read_manifest(
        dataset_kind="trades",
        dataset_id=CANONICAL_DATASET_ID,
    )
    results = run_event_time_finalization(trades)
    output = arguments.output_dir
    if not output.is_absolute():
        output = find_project_root() / output
    paths = write_event_time_artifacts(
        results,
        output,
        source_dataset_id=CANONICAL_DATASET_ID,
        source_sha256=str(manifest["sha256"]),
        overwrite=arguments.overwrite,
    )
    print("===== DAY 25 EVENT-TIME FINALIZATION =====")
    print(results.sampling_comparison.to_string(index=False))
    print(results.indicator_comparison.to_string(index=False))
    print("Locked period accessed: False")
    print("Artifacts:")
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
