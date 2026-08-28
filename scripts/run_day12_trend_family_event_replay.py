"""Run and report the canonical Day 12 event-driven replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day12_event_replay_report import (
    Day12ArtifactResult,
    Day12DatasetAudit,
    Day12ReportError,
    run_day12_replay_study,
    write_day12_artifacts,
)
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS,
    DEVELOPMENT_DATASET_ID,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
    load_project_config,
)
from systematic_alpha.data.local_store import LocalParquetStore
from systematic_alpha.data.sample_windows import SampleWindow


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day12"
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


class Day12RunnerError(ValueError):
    """Raised when the canonical Day 12 workflow is unsafe."""


@dataclass(frozen=True, slots=True)
class CanonicalDay12Input:
    """Validated SPY bars and canonical dataset lineage."""

    bars: pd.DataFrame
    audit: Day12DatasetAudit


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the narrow Day 12 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic replay for the two frozen "
            "SPY 15-minute trend baselines."
        )
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help=(
            "Directory for the seven compact Day 12 "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing approved Day 12 "
            "artifact set."
        ),
    )
    parser.add_argument(
        "--generation-timestamp",
        default=None,
        help=(
            "Optional timezone-aware provenance timestamp."
        ),
    )

    return parser.parse_args(argv)


def _validate_manifest(
    manifest: dict[str, Any],
) -> str:
    """Validate immutable canonical development lineage."""

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
        or manifest.get("row_count")
        != EXPECTED_CANONICAL_ROWS
    ):
        raise Day12RunnerError(
            "Canonical manifest identity or row count "
            "does not match the frozen development "
            "dataset."
        )

    digest = manifest.get("sha256")

    if not isinstance(digest, str):
        raise Day12RunnerError(
            "Canonical manifest has no SHA-256 digest."
        )

    normalized = digest.strip().lower()

    if _SHA256_PATTERN.fullmatch(
        normalized
    ) is None:
        raise Day12RunnerError(
            "Canonical manifest SHA-256 is malformed."
        )

    return normalized


def _validate_canonical_bars(
    bars: pd.DataFrame,
    *,
    manifest_sha256: str,
) -> CanonicalDay12Input:
    """Validate the canonical frame, then isolate SPY immediately."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError(
            "Canonical bars must be a pandas DataFrame."
        )

    if len(bars) != EXPECTED_CANONICAL_ROWS:
        raise Day12RunnerError(
            "Canonical development input must contain "
            f"exactly {EXPECTED_CANONICAL_ROWS:,} rows."
        )

    required = {
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
    missing = sorted(
        required.difference(bars.columns)
    )

    if missing:
        raise Day12RunnerError(
            "Canonical bars are missing required columns: "
            f"{missing}."
        )

    source = bars.copy(deep=True)
    symbols = (
        source["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        symbols.isna().any()
        or symbols.eq("").any()
        or frozenset(
            symbols.astype(str).unique()
        )
        != EXPECTED_SYMBOLS
    ):
        raise Day12RunnerError(
            "Canonical symbols must be exactly SPY, QQQ "
            "and IWM."
        )

    source["symbol"] = symbols

    try:
        source["timestamp"] = pd.to_datetime(
            source["timestamp"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise Day12RunnerError(
            "Canonical timestamps are malformed."
        ) from exc

    local_dates = (
        source["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if local_dates.dt.year.eq(2026).any():
        raise Day12RunnerError(
            "Canonical input contains forbidden 2026 "
            "observations."
        )

    if (
        local_dates.min()
        != pd.Timestamp("2020-01-02")
        or local_dates.max()
        != pd.Timestamp("2025-12-31")
    ):
        raise Day12RunnerError(
            "Canonical input must cover exactly the "
            "2020-01-02 through 2025-12-31 development "
            "period."
        )

    spy = source.loc[
        source["symbol"].eq("SPY")
    ].copy(deep=True)
    spy = spy.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    if len(spy) != EXPECTED_SPY_ROWS:
        raise Day12RunnerError(
            "Canonical SPY input must contain exactly "
            f"{EXPECTED_SPY_ROWS:,} rows."
        )

    spy_dates = (
        spy["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    session_sizes = (
        spy.assign(
            session_date=spy_dates
        )
        .groupby(
            "session_date",
            observed=True,
            sort=False,
        )
        .size()
    )

    if (
        len(session_sizes)
        != EXPECTED_SPY_SESSIONS
        or set(
            session_sizes.astype(int).unique()
        )
        != {
            14,
            26,
        }
    ):
        raise Day12RunnerError(
            "Canonical SPY sessions must contain exactly "
            f"{EXPECTED_SPY_SESSIONS:,} sessions of 14 "
            "or 26 bars."
        )

    spy["session_date"] = spy_dates

    return CanonicalDay12Input(
        bars=spy,
        audit=Day12DatasetAudit(
            dataset_id=DEVELOPMENT_DATASET_ID,
            dataset_path=(
                CANONICAL_DATASET_RELATIVE_PATH
                .as_posix()
            ),
            manifest_sha256=(
                manifest_sha256.lower()
            ),
            canonical_row_count=len(source),
            spy_row_count=len(spy),
            spy_session_count=len(
                session_sizes
            ),
            minimum_timestamp=(
                source["timestamp"].min()
            ),
            maximum_timestamp=(
                source["timestamp"].max()
            ),
        ),
    )


def load_canonical_day12_input() -> (
    CanonicalDay12Input
):
    """Load only the named canonical development dataset safely."""

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
    ):
        raise Day12RunnerError(
            "Project development-window controls do not "
            "match the approved Day 12 contract."
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

    return _validate_canonical_bars(
        validated,
        manifest_sha256=(
            _validate_manifest(manifest)
        ),
    )


def _source_git_commit() -> str:
    """Resolve repository provenance for the manifest."""

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
        raise Day12RunnerError(
            "Could not resolve the source Git commit."
        ) from exc


def execute_day12(
    *,
    artifact_directory: str | Path,
    overwrite: bool = False,
    generation_timestamp: str | None = None,
    source_git_commit: str | None = None,
) -> Day12ArtifactResult:
    """Load, replay, validate, and report canonical Day 12."""

    canonical = load_canonical_day12_input()
    study = run_day12_replay_study(
        canonical.bars
    )

    return write_day12_artifacts(
        study,
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
    """Display a repository-relative path when possible."""

    try:
        return path.relative_to(
            project_root
        ).as_posix()
    except ValueError:
        return path.as_posix()


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the canonical Day 12 command-line workflow."""

    arguments = parse_args(argv)
    project_root = find_project_root()
    artifact_directory = (
        _resolve_from_project_root(
            arguments.artifact_directory,
            project_root=project_root,
        )
    )

    try:
        result = execute_day12(
            artifact_directory=(
                artifact_directory
            ),
            overwrite=arguments.overwrite,
            generation_timestamp=(
                arguments.generation_timestamp
            ),
        )
    except Day12ReportError as exc:
        raise Day12RunnerError(
            str(exc)
        ) from exc

    print(
        "===== DAY 12 DETERMINISTIC "
        "TREND-FAMILY EVENT REPLAY ====="
    )
    print(
        "Source dataset:",
        CANONICAL_DATASET_RELATIVE_PATH
        .as_posix(),
    )
    print("Scope: SPY 15min")
    print(
        "Configuration identifiers:",
        CONFIGURATION_IDS,
    )
    print("Replay summary:")
    print(
        result.study.replay_summary[
            [
                "strategy",
                "observations",
                "sessions",
                "events",
                "final_equity",
            ]
        ].to_string(index=False)
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
        "DAY 12 DETERMINISTIC EVENT REPLAY "
        "REPORT PASSED"
    )


if __name__ == "__main__":
    main()
