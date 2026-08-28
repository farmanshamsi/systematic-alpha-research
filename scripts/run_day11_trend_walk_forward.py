"""Run and report the frozen Day 11 trend walk-forward study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day11_trend_walk_forward_report import (
    Day11ArtifactResult,
    Day11DatasetAudit,
    Day11ReportError,
    write_day11_artifacts,
)
from systematic_alpha.analysis.trend_family_robustness import (
    DEVELOPMENT_DATASET_ID,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    CONFIGURATION_IDS,
    WALK_FORWARD_FREQUENCY,
    WALK_FORWARD_STRATEGIES,
    WALK_FORWARD_SYMBOL,
    run_trend_family_walk_forward,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
    load_project_config,
)
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.sample_windows import SampleWindow


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day11"
)
CANONICAL_DATASET_RELATIVE_PATH: Final[
    Path
] = Path(
    "data/processed/bars/"
    f"{DEVELOPMENT_DATASET_ID}.parquet"
)
EXPECTED_CANONICAL_ROWS: Final[int] = 117_192
EXPECTED_SPY_ROWS: Final[int] = 39_064
EXPECTED_SPY_SESSIONS: Final[int] = 1_508
EXPECTED_SYMBOLS: Final[frozenset[str]] = (
    frozenset(
        {
            "SPY",
            "QQQ",
            "IWM",
        }
    )
)
_SHA256_PATTERN: Final[re.Pattern[str]] = (
    re.compile(r"^[0-9a-f]{64}$")
)


class Day11RunnerError(ValueError):
    """Raised when the canonical Day 11 runner is unsafe."""


@dataclass(frozen=True, slots=True)
class CanonicalDay11Input:
    """Validated canonical bars and their report audit."""

    bars: pd.DataFrame
    audit: Day11DatasetAudit


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the narrow Day 11 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen SPY 15-minute Day 11 "
            "walk-forward validation."
        )
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help=(
            "Directory for the seven approved Day 11 "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing approved Day 11 "
            "artifact set."
        ),
    )
    parser.add_argument(
        "--generation-timestamp",
        default=None,
        help=(
            "Optional timezone-aware provenance timestamp; "
            "primarily useful for reproducibility audits."
        ),
    )

    return parser.parse_args(argv)


def _validate_manifest(
    manifest: dict[str, Any],
) -> str:
    """Validate immutable canonical dataset lineage."""

    if not isinstance(manifest, dict):
        raise TypeError(
            "Canonical dataset manifest must be a "
            "dictionary."
        )

    if (
        manifest.get("dataset_id")
        != DEVELOPMENT_DATASET_ID
        or manifest.get("dataset_kind")
        != "bars"
    ):
        raise Day11RunnerError(
            "Canonical manifest identity does not match "
            "the frozen development dataset."
        )

    if (
        manifest.get("row_count")
        != EXPECTED_CANONICAL_ROWS
    ):
        raise Day11RunnerError(
            "Canonical manifest row count must be exactly "
            f"{EXPECTED_CANONICAL_ROWS:,}."
        )

    digest = manifest.get("sha256")

    if not isinstance(digest, str):
        raise Day11RunnerError(
            "Canonical manifest has no SHA-256 digest."
        )

    normalized = digest.strip().lower()

    if _SHA256_PATTERN.fullmatch(
        normalized
    ) is None:
        raise Day11RunnerError(
            "Canonical manifest SHA-256 is malformed."
        )

    return normalized


def _normalize_symbols(
    bars: pd.DataFrame,
) -> pd.Series:
    """Normalize canonical symbols without mutating input."""

    if "symbol" not in bars.columns:
        raise Day11RunnerError(
            "Canonical bars must contain symbol."
        )

    symbols = (
        bars["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        symbols.isna().any()
        or symbols.eq("").any()
    ):
        raise Day11RunnerError(
            "Canonical bars contain invalid symbols."
        )

    actual = frozenset(
        symbols.astype(str).unique()
    )

    if actual != EXPECTED_SYMBOLS:
        raise Day11RunnerError(
            "Canonical symbols must be exactly SPY, QQQ "
            f"and IWM; received {sorted(actual)}."
        )

    return symbols


def _build_dataset_audit(
    bars: pd.DataFrame,
    *,
    manifest_sha256: str,
) -> Day11DatasetAudit:
    """Build the canonical Day 11 input audit."""

    if len(bars) != EXPECTED_CANONICAL_ROWS:
        raise Day11RunnerError(
            "Canonical development input must contain "
            f"exactly {EXPECTED_CANONICAL_ROWS:,} rows."
        )

    symbols = _normalize_symbols(bars)
    timestamps = pd.to_datetime(
        bars["timestamp"],
        utc=True,
        errors="raise",
    )
    local_dates = (
        timestamps.dt.tz_convert(
            "America/New_York"
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if (
        local_dates.min()
        != pd.Timestamp("2020-01-02")
        or local_dates.max()
        != pd.Timestamp("2025-12-31")
    ):
        raise Day11RunnerError(
            "Canonical input must cover exactly "
            "2020-01-02 through 2025-12-31 and must not "
            "contain 2026 observations."
        )

    spy = bars.loc[
        symbols.eq(WALK_FORWARD_SYMBOL)
    ].copy(deep=True)

    if len(spy) != EXPECTED_SPY_ROWS:
        raise Day11RunnerError(
            "Canonical SPY input must contain exactly "
            f"{EXPECTED_SPY_ROWS:,} rows."
        )

    spy_timestamps = timestamps.loc[
        symbols.eq(WALK_FORWARD_SYMBOL)
    ]
    session_dates = (
        spy_timestamps
        .dt.tz_convert(
            "America/New_York"
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )
    sessions = int(
        session_dates.nunique()
    )

    if sessions != EXPECTED_SPY_SESSIONS:
        raise Day11RunnerError(
            "Canonical SPY input must contain exactly "
            f"{EXPECTED_SPY_SESSIONS:,} sessions."
        )

    session_sizes = (
        spy.assign(
            _session_date=session_dates
        )
        .groupby(
            "_session_date",
            observed=True,
            sort=False,
        )
        .size()
    )

    if set(
        session_sizes.astype(int).unique()
    ) != {
        14,
        26,
    }:
        raise Day11RunnerError(
            "Canonical SPY sessions must contain 14 or "
            "26 15-minute bars."
        )

    return Day11DatasetAudit(
        dataset_id=DEVELOPMENT_DATASET_ID,
        dataset_path=(
            CANONICAL_DATASET_RELATIVE_PATH
            .as_posix()
        ),
        manifest_sha256=manifest_sha256,
        canonical_row_count=len(bars),
        spy_row_count=len(spy),
        spy_session_count=sessions,
        minimum_timestamp=timestamps.min(),
        maximum_timestamp=timestamps.max(),
    )


def load_canonical_day11_input() -> (
    CanonicalDay11Input
):
    """Load canonical development bars through locked-window controls."""

    config = load_project_config()
    sample_window = SampleWindow.from_project_config(
        config
    )

    if (
        sample_window.development_start
        != pd.Timestamp(
            "2020-01-02",
            tz="UTC",
        )
        or sample_window.development_end_exclusive
        != pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        )
        or sample_window.final_test_start
        < pd.Timestamp(
            "2026-01-02",
            tz="UTC",
        )
    ):
        raise Day11RunnerError(
            "Project sample-window controls do not match "
            "the approved Day 11 development contract."
        )

    store = LocalParquetStore.from_project_config(
        config,
        tier="processed",
    )
    bars = store.read(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
        verify_hash=True,
    )
    validated = (
        sample_window
        .validate_development_frame(
            bars
        )
    )
    manifest = store.read_manifest(
        dataset_kind="bars",
        dataset_id=DEVELOPMENT_DATASET_ID,
    )
    manifest_sha256 = _validate_manifest(
        manifest
    )
    audit = _build_dataset_audit(
        validated,
        manifest_sha256=(
            manifest_sha256
        ),
    )

    return CanonicalDay11Input(
        bars=validated,
        audit=audit,
    )


def _source_git_commit() -> str:
    """Resolve repository provenance for the report manifest."""

    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=find_project_root(),
            text=True,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise Day11RunnerError(
            "Could not resolve the source Git commit."
        ) from exc


def execute_day11(
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
    generation_timestamp: str | None = None,
    source_git_commit: str | None = None,
) -> Day11ArtifactResult:
    """Load, execute and report the canonical Day 11 study."""

    canonical = load_canonical_day11_input()
    results = run_trend_family_walk_forward(
        canonical.bars
    )

    return write_day11_artifacts(
        results,
        canonical.audit,
        artifact_directory=(
            artifact_directory
        ),
        overwrite=overwrite,
        generation_timestamp=(
            generation_timestamp
        ),
        source_git_commit=(
            source_git_commit
            if source_git_commit is not None
            else _source_git_commit()
        ),
    )


def _resolve_from_project_root(
    path: Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve one CLI path relative to the repository."""

    if path.is_absolute():
        return path

    return project_root / path


def _display_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    """Display a repository-relative or absolute path."""

    try:
        return path.relative_to(
            project_root
        ).as_posix()
    except ValueError:
        return path.as_posix()


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the canonical Day 11 command-line workflow."""

    arguments = parse_args(argv)
    project_root = find_project_root()
    artifact_directory = (
        _resolve_from_project_root(
            arguments.artifact_directory,
            project_root=project_root,
        )
    )

    try:
        result = execute_day11(
            artifact_directory=(
                artifact_directory
            ),
            overwrite=arguments.overwrite,
            generation_timestamp=(
                arguments.generation_timestamp
            ),
        )
    except Day11ReportError as exc:
        raise Day11RunnerError(
            str(exc)
        ) from exc

    print(
        "===== DAY 11 TREND-FAMILY "
        "WALK-FORWARD VALIDATION ====="
    )
    print(
        "Source dataset:",
        CANONICAL_DATASET_RELATIVE_PATH
        .as_posix(),
    )
    print(
        "Scope:",
        WALK_FORWARD_SYMBOL,
        WALK_FORWARD_FREQUENCY,
    )
    print(
        "Strategies:",
        list(WALK_FORWARD_STRATEGIES),
    )
    print(
        "Configuration identifiers:",
        CONFIGURATION_IDS,
    )
    print(
        "Fold rows:",
        len(result.fold_results),
    )
    print(
        "Aggregate rows:",
        len(result.aggregate_results),
    )
    print("Artifacts:")

    for path in result.artifact_paths:
        print(
            _display_path(
                path,
                project_root=project_root,
            )
        )

    print(
        "Locked 2026 data accessed:",
        False,
    )
    print(
        "DAY 11 TREND-FAMILY WALK-FORWARD "
        "REPORT PASSED"
    )


if __name__ == "__main__":
    main()
