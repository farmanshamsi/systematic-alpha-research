"""Run the separately authorized one-time locked 2026 final test."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.locked_final_test import (
    APPROVED_ARTIFACT_NAMES,
    AUTHORIZATION_CODE,
    LOCKED_END_EXCLUSIVE,
    LOCKED_START,
    evaluate_locked_final_test,
    require_authorization,
    validate_locked_bars,
    verify_frozen_development_state,
    write_locked_final_test_artifacts,
)
from systematic_alpha.data.alpaca_provider import AlpacaBarProvider
from systematic_alpha.data.config_loader import find_project_root, load_project_config
from systematic_alpha.data.development_dataset import (
    DevelopmentChunk,
    filter_regular_session_bars,
    validate_complete_session_grid,
)


DEFAULT_DEVELOPMENT_DATASET: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
DEFAULT_OUTPUT: Final[Path] = Path("artifacts/day25_final_test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Access and evaluate the locked 2026 interval exactly once under "
            "the frozen three-model protocol."
        )
    )
    parser.add_argument("--authorization-code", required=True)
    parser.add_argument(
        "--development-dataset", type=Path, default=DEFAULT_DEVELOPMENT_DATASET
    )
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = parse_args(argv)
    require_authorization(arguments.authorization_code)
    root = find_project_root()
    output = _resolve(arguments.artifact_directory, root)
    if output.exists():
        raise FileExistsError("The one-time final-test bundle already exists.")
    frozen_hashes = verify_frozen_development_state(root)
    development_path = _resolve(arguments.development_dataset, root)
    development = pd.read_parquet(development_path)

    config = deepcopy(load_project_config())
    config["broker"]["stock_data_feed"] = "sip"
    provider = AlpacaBarProvider(config=config)
    fetched = provider.fetch_bars_bundle(
        symbols=["SPY", "QQQ", "IWM"],
        start=LOCKED_START,
        end=LOCKED_END_EXCLUSIVE,
        timeframe_minutes=15,
    )
    locked = validate_locked_bars(filter_regular_session_bars(fetched.normalized))
    validate_complete_session_grid(
        locked,
        chunk=DevelopmentChunk(
            label="2026-01-02_2026-06-30_locked_final_test",
            start=LOCKED_START,
            end_exclusive=LOCKED_END_EXCLUSIVE,
        ),
        expected_symbols=("SPY", "QQQ", "IWM"),
        timeframe_minutes=15,
    )
    results = evaluate_locked_final_test(development, locked)
    request_metadata = {
        **fetched.request_metadata,
        "purpose": "one_time_locked_final_test",
        "regular_session_only": True,
        "canonical_rows": len(locked),
        "canonical_sha256_in_memory_csv": hashlib.sha256(
            locked.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "authorization_code": AUTHORIZATION_CODE,
    }
    paths = write_locked_final_test_artifacts(
        results,
        locked,
        output,
        frozen_hashes=frozen_hashes,
        request_metadata=request_metadata,
    )
    if len(paths) != len(APPROVED_ARTIFACT_NAMES):
        raise RuntimeError("One-time final-test bundle is incomplete.")
    print("===== ONE-TIME LOCKED 2026 FINAL TEST COMPLETE =====")
    print(results.performance.to_json(orient="records", indent=2))
    print(
        json.dumps(
            {
                "artifact_files": len(paths),
                "all_results_reported": True,
                "ranking_or_retuning_performed": False,
                "broker_orders_or_account_mutation": False,
                "commit_or_push_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()
