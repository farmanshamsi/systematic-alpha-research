"""Run the versioned development-only Day 29 accounting experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day29_fixed_holdings_portfolio_experiment import (
    APPROVED_ARTIFACT_NAMES,
    Day29FixedHoldingsExperimentResults,
    describe_existing_bundle,
    load_comparator_snapshot,
    sha256_file,
    validate_historical_day16_comparators,
    build_day29_experiment,
    write_day29_artifacts,
)
from systematic_alpha.analysis.portfolio_allocation_validation import (
    analyze_portfolio_allocation_panel,
    analyze_portfolio_allocation_panel_fixed_holdings,
)
from systematic_alpha.analysis.strategy_diversification import run_strategy_diversification
from systematic_alpha.data.config_loader import find_project_root

try:
    from scripts.run_day10_trend_robustness import validate_canonical_input
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_day10_trend_robustness import validate_canonical_input


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/processed/bars/"
    "spy_qqq_iwm_15min_2020-01-02_2025-12-31_"
    "sip_v3_development_canonical.parquet"
)
EXPECTED_SOURCE_SHA256: Final[str] = (
    "30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2"
)
DEFAULT_DAY16_COMPARATOR_DIRECTORY: Final[Path] = Path("artifacts/day16")
DEFAULT_DAY25_COMPARATOR_DIRECTORY: Final[Path] = Path(
    "artifacts/day25_causal_portfolio_finalization"
)
DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day29_fixed_holdings_portfolio_experiment"
)


class Day29RunnerError(ValueError):
    """Raised when the Day 29 command-line workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class Day29RunResult:
    """One completed real or synthetic Day 29 runner invocation."""

    dataset_path: Path
    source_sha256: str
    day16_comparator_directory: Path
    day25_comparator_directory: Path
    artifact_directory: Path
    analysis_results: Day29FixedHoldingsExperimentResults
    artifact_paths: tuple[Path, ...]
    evaluation_complete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Day 16 constant-mix accounting with corrected "
            "fixed holdings on development data only."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--day16-comparator-directory",
        type=Path,
        default=DEFAULT_DAY16_COMPARATOR_DIRECTORY,
    )
    parser.add_argument(
        "--day25-comparator-directory",
        type=Path,
        default=DEFAULT_DAY25_COMPARATOR_DIRECTORY,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    return parser.parse_args(argv)


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _read_dataset(path: Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Development dataset does not exist: {path}.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise Day29RunnerError("Development dataset must be Parquet or CSV.")


def audit_canonical_bar_range(bars: pd.DataFrame) -> dict[str, object]:
    """Fail closed unless canonical bars span exactly 2020 through 2025."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")
    if "timestamp" not in bars.columns:
        raise Day29RunnerError("Canonical bars require timestamp.")
    try:
        timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise Day29RunnerError("Canonical timestamps are invalid.") from exc
    if timestamps.isna().any() or timestamps.empty:
        raise Day29RunnerError("Canonical timestamps must be complete.")
    if timestamps.duplicated().any() and "symbol" not in bars.columns:
        raise Day29RunnerError("Canonical observations require symbol identifiers.")
    minimum_session = timestamps.min().normalize()
    maximum_session = timestamps.max().normalize()
    if maximum_session >= pd.Timestamp("2026-01-01", tz="UTC"):
        raise Day29RunnerError(
            "Canonical input contains a prohibited 2026-or-later observation."
        )
    if minimum_session != pd.Timestamp(
        "2020-01-02", tz="UTC"
    ) or maximum_session != pd.Timestamp("2025-12-31", tz="UTC"):
        raise Day29RunnerError(
            "Day 29 canonical data must span exactly 2020-01-02 through "
            "2025-12-31."
        )
    return {
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "session_min": minimum_session.date().isoformat(),
        "session_max": maximum_session.date().isoformat(),
        "rows": int(len(bars)),
        "contains_2026_or_later": False,
    }


def execute_day29(
    *,
    dataset_path: str | Path,
    day16_comparator_directory: str | Path,
    day25_comparator_directory: str | Path,
    artifact_directory: str | Path,
    generation_timestamp: str | None = None,
) -> Day29RunResult:
    """Authenticate inputs, calculate once, and write one isolated bundle."""

    for name, value in (
        ("dataset_path", dataset_path),
        ("day16_comparator_directory", day16_comparator_directory),
        ("day25_comparator_directory", day25_comparator_directory),
        ("artifact_directory", artifact_directory),
    ):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be a path.")
    source_path = Path(dataset_path)
    day16_directory = Path(day16_comparator_directory)
    day25_directory = Path(day25_comparator_directory)
    output_directory = Path(artifact_directory)
    if output_directory.exists():
        raise FileExistsError(
            "Day 29 output directory already exists; refusing overwrite. "
            + describe_existing_bundle(output_directory)
        )
    if source_path.name != DEFAULT_DATASET_PATH.name:
        raise Day29RunnerError(
            "Day 29 requires the exact frozen Day 16 canonical dataset identity."
        )

    comparator_snapshot = load_comparator_snapshot(
        day16_directory=day16_directory,
        day25_directory=day25_directory,
    )
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise Day29RunnerError(
            "Day 29 source SHA-256 differs from the frozen Day 16 canonical "
            "dataset."
        )
    source = _read_dataset(source_path)
    validated_bars = validate_canonical_input(source)
    raw_audit = audit_canonical_bar_range(validated_bars)
    diversification = run_strategy_diversification(validated_bars)
    panel = diversification.copy_session_return_panel()

    historical = analyze_portfolio_allocation_panel(
        panel,
        require_canonical_counts=True,
    )
    validate_historical_day16_comparators(
        historical,
        day16_directory=day16_directory,
    )
    corrected = analyze_portfolio_allocation_panel_fixed_holdings(
        panel,
        require_canonical_counts=True,
    )
    timestamp = generation_timestamp or datetime.now(timezone.utc).isoformat()
    results = build_day29_experiment(
        panel,
        source_dataset_path=source_path.resolve().as_posix(),
        source_sha256=source_sha256,
        comparator_snapshot=comparator_snapshot,
        generation_timestamp=timestamp,
        require_canonical_counts=True,
        require_exact_development_range=True,
        historical_results=historical,
        corrected_results=corrected,
    )
    metadata = dict(results.source_and_method_metadata)
    metadata["raw_bar_audit"] = raw_audit
    results = Day29FixedHoldingsExperimentResults(
        source_and_method_metadata=metadata,
        target_and_covariance_invariance=(
            results.target_and_covariance_invariance
        ),
        fold_performance_comparison=results.fold_performance_comparison,
        aggregate_performance_comparison=results.aggregate_performance_comparison,
        fold_turnover_comparison=results.fold_turnover_comparison,
        ending_weight_drift=results.ending_weight_drift,
        corrected_weight_path=results.corrected_weight_path,
        wealth_identity_checks=results.wealth_identity_checks,
        portfolio_return_comparison=results.portfolio_return_comparison,
        comparator_snapshot=results.comparator_snapshot,
    )
    artifact_paths = write_day29_artifacts(results, output_directory)
    if tuple(path.name for path in artifact_paths) != APPROVED_ARTIFACT_NAMES:
        raise RuntimeError("Day 29 artifact writing did not complete.")
    return Day29RunResult(
        dataset_path=source_path,
        source_sha256=source_sha256,
        day16_comparator_directory=day16_directory,
        day25_comparator_directory=day25_directory,
        artifact_directory=output_directory,
        analysis_results=results,
        artifact_paths=artifact_paths,
        evaluation_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> Day29RunResult:
    arguments = parse_args(argv)
    project_root = find_project_root()
    result = execute_day29(
        dataset_path=_resolve(arguments.dataset_path, project_root=project_root),
        day16_comparator_directory=_resolve(
            arguments.day16_comparator_directory, project_root=project_root
        ),
        day25_comparator_directory=_resolve(
            arguments.day25_comparator_directory, project_root=project_root
        ),
        artifact_directory=_resolve(
            arguments.artifact_directory, project_root=project_root
        ),
    )
    print("===== DAY 29 FIXED-HOLDINGS PORTFOLIO EXPERIMENT COMPLETE =====")
    print("Source dataset:", result.dataset_path)
    print("Source SHA-256:", result.source_sha256)
    print("Day 16 comparator:", result.day16_comparator_directory)
    print("Day 25 preservation comparator:", result.day25_comparator_directory)
    print("Artifact directory:", result.artifact_directory)
    for path in result.artifact_paths:
        print(path)
    print("evaluation_complete:", result.evaluation_complete)
    return result


if __name__ == "__main__":
    main()
