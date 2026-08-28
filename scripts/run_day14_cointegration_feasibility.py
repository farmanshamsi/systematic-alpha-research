"""Run the canonical Day 14 cointegration feasibility study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.cointegration_feasibility import (
    run_cointegration_feasibility,
)
from systematic_alpha.analysis.day14_cointegration_report import (
    APPROVED_DAY14_ARTIFACT_NAMES,
    Day14CointegrationReport,
    build_day14_cointegration_report,
    write_day14_cointegration_artifacts,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
)
from systematic_alpha.data.sample_windows import (
    SampleWindow,
    SampleWindowError,
)
from scripts.run_day11_trend_walk_forward import (
    CANONICAL_DATASET_RELATIVE_PATH,
    load_canonical_day11_input,
)


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day14"
)


def _parse_args(
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    """Parse the narrow deterministic Day 14 command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Day 14 cointegration "
            "and OU feasibility study."
        )
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args(argv)


def _development_window() -> SampleWindow:
    """Return the canonical development and locked-test boundaries."""

    return SampleWindow(
        development_start=pd.Timestamp(
            "2020-01-02",
            tz="UTC",
        ),
        development_end_exclusive=pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
        final_test_start=pd.Timestamp(
            "2026-01-02",
            tz="UTC",
        ),
        final_test_end_exclusive=pd.Timestamp(
            "2026-07-01",
            tz="UTC",
        ),
    )


def _resolve_output_directory(
    path: Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve one output path relative to the repository root."""

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
    """Run the canonical Day 14 command-line workflow."""

    arguments = _parse_args(argv)
    project_root = find_project_root()
    output_directory = (
        _resolve_output_directory(
            arguments.artifact_directory,
            project_root=project_root,
        )
    )

    canonical = load_canonical_day11_input()
    sample_window = _development_window()

    try:
        development_bars = (
            sample_window.validate_development_frame(
                canonical.bars
            )
        )
    except SampleWindowError:
        print(
            "Day 14 development input rejected."
        )
        return

    results = run_cointegration_feasibility(
        development_bars
    )
    report = build_day14_cointegration_report(
        results
    )
    paths = write_day14_cointegration_artifacts(
        report,
        output_directory,
        overwrite=arguments.overwrite,
    )

    if (
        isinstance(
            report,
            Day14CointegrationReport,
        )
        and len(paths)
        != len(
            APPROVED_DAY14_ARTIFACT_NAMES
        )
    ):
        raise RuntimeError(
            "Day 14 artifact writing did not complete."
        )

    eligible_count = int(
        report.pair_eligibility[
            "eligible"
        ]
        .astype(bool)
        .sum()
    )
    ou_attempted = int(
        report.ou_diagnostics[
            "attempted"
        ]
        .astype(bool)
        .sum()
    )

    print(
        "===== DAY 14 COINTEGRATION "
        "FEASIBILITY COMPLETE ====="
    )
    print(
        "Source dataset:",
        CANONICAL_DATASET_RELATIVE_PATH.as_posix(),
    )
    print("Candidate pairs: 3")
    print("Expanding folds per pair: 4")
    print(
        "OU estimations attempted:",
        ou_attempted,
    )
    print(
        "Eligible pairs:",
        eligible_count,
    )
    print(
        "Artifact directory:",
        _display_path(
            output_directory,
            project_root=project_root,
        ),
    )
    print(
        "DAY 14 COINTEGRATION "
        "FEASIBILITY REPORT PASSED"
    )


if __name__ == "__main__":
    main()
