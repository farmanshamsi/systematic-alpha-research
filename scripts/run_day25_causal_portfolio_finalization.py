"""Run the development-only causal portfolio dependency correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.causal_portfolio_finalization import (
    APPROVED_ARTIFACT_NAMES,
    run_causal_portfolio_finalization,
    write_causal_portfolio_artifacts,
)
from systematic_alpha.data.config_loader import find_project_root


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day25_causal_portfolio_finalization"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Day 15-16 trend sleeves under final causal timing without "
            "accessing locked data or selecting an allocation rule."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = parse_args(argv)
    root = find_project_root()
    dataset_path = _resolve(arguments.dataset_path, root)
    output = _resolve(arguments.artifact_directory, root)
    if dataset_path.suffix.lower() != ".parquet" or not dataset_path.is_file():
        raise ValueError("The frozen canonical Parquet dataset is required.")
    bars = pd.read_parquet(dataset_path)
    results = run_causal_portfolio_finalization(bars)
    paths = write_causal_portfolio_artifacts(
        results,
        output,
        overwrite=arguments.overwrite,
    )
    if len(paths) != len(APPROVED_ARTIFACT_NAMES):
        raise RuntimeError("Causal portfolio bundle is incomplete.")
    summary = {
        "artifact_files": len(paths),
        "sleeve_session_rows": len(results.sleeve_session_returns),
        "portfolio_test_sessions": len(results.allocation.portfolio_return_panel),
        "aggregate_results": results.allocation.aggregate_portfolio_performance[
            ["allocation_rule", "cumulative_return", "sharpe_ratio"]
        ].to_dict(orient="records"),
        "locked_2026_data_accessed": False,
        "allocation_rule_selected": False,
        "commit_or_push_performed": False,
    }
    print("===== DAY 25 CAUSAL PORTFOLIO FINALIZATION COMPLETE =====")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return paths


if __name__ == "__main__":
    main()
