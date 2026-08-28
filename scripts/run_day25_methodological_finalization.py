"""Run the frozen Day 25 development-only methodological audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from systematic_alpha.analysis.trend_methodology_finalization import (
    run_trend_methodology_finalization,
    write_trend_methodology_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root, load_project_config
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.sample_windows import SampleWindow


DEVELOPMENT_DATASET_ID: Final[str] = (
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical"
)
DEFAULT_OUTPUT: Final[Path] = Path("artifacts/day25_methodological_finalization")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen development-only trend finalization matrix."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = parse_args(argv)
    config = load_project_config()
    sample = SampleWindow.from_project_config(config)
    store = LocalParquetStore.from_project_config(config, tier="processed")
    bars = store.read(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
        verify_hash=True,
    )
    bars = sample.validate_development_frame(bars)
    manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
    )
    results = run_trend_methodology_finalization(bars)
    output = arguments.output_dir
    if not output.is_absolute():
        output = find_project_root() / output
    paths = write_trend_methodology_artifacts(
        results,
        output,
        source_dataset_id=DEVELOPMENT_DATASET_ID,
        source_sha256=str(manifest["sha256"]),
        overwrite=arguments.overwrite,
    )
    aggregate = results.walk_forward.loc[
        results.walk_forward["fold_id"].eq("aggregate_2022_2025")
        & results.walk_forward["cost_bps_per_turnover"].eq(1.0),
        ["model_id", "cumulative_return", "sharpe_ratio"],
    ]
    print("===== DAY 25 METHODOLOGICAL FINALIZATION =====")
    print(aggregate.to_string(index=False))
    print("Locked period accessed: False")
    print("Artifacts:")
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
