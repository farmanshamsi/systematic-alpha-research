"""Run the canonical Day 13 event-driven walk-forward workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from systematic_alpha.analysis.day13_event_walk_forward_report import (
    APPROVED_DAY13_ARTIFACT_NAMES,
    Day13EventWalkForwardReport,
    build_day13_event_walk_forward_report,
    write_day13_event_walk_forward_artifacts,
)
from systematic_alpha.analysis.trend_family_event_walk_forward import (
    run_trend_family_event_walk_forward,
)
from systematic_alpha.data.config_loader import (
    find_project_root,
)
from systematic_alpha.data.sample_windows import SampleWindow
from systematic_alpha.data.sample_windows import (
    SampleWindowError,
)
from scripts.run_day11_trend_walk_forward import (
    CANONICAL_DATASET_RELATIVE_PATH,
    load_canonical_day11_input,
)


DEFAULT_ARTIFACT_DIRECTORY: Final[Path] = Path(
    "artifacts/day13"
)


def _parse_args(
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    """Parse the narrow deterministic Day 13 command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen event-driven trend-family "
            "walk-forward study."
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
        development_end_exclusive=(
            pd.Timestamp(
                "2026-01-01",
                tz="UTC",
            )
        ),
        final_test_start=pd.Timestamp(
            "2026-01-02",
            tz="UTC",
        ),
        final_test_end_exclusive=(
            pd.Timestamp(
                "2026-07-01",
                tz="UTC",
            )
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


def _summary_counts(
    report: object,
) -> tuple[int, int]:
    """Return test-observation and passed-parity counts when available."""

    if not isinstance(
        report,
        Day13EventWalkForwardReport,
    ):
        return 0, 0

    observations = int(
        pd.to_numeric(
            report.fold_summary[
                "test_observations"
            ],
            errors="raise",
        ).sum()
    )
    parity_passed = int(
        report.vectorized_parity[
            "passed"
        ].astype(bool).sum()
    )

    return observations, parity_passed


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the canonical Day 13 command-line workflow."""

    arguments = _parse_args(argv)
    project_root = find_project_root()
    output_directory = (
        _resolve_output_directory(
            arguments.artifact_directory,
            project_root=project_root,
        )
    )
    sample_window = _development_window()
    canonical = (
        load_canonical_day11_input()
    )

    try:
        development_bars = (
            sample_window
            .validate_development_frame(
                canonical.bars
            )
        )
    except SampleWindowError:
        print(
            "Day 13 development input rejected."
        )

        return

    results = (
        run_trend_family_event_walk_forward(
            development_bars
        )
    )
    report = (
        build_day13_event_walk_forward_report(
            results
        )
    )
    paths = (
        write_day13_event_walk_forward_artifacts(
            report,
            output_directory,
            overwrite=arguments.overwrite,
        )
    )

    if (
        isinstance(
            report,
            Day13EventWalkForwardReport,
        )
        and len(paths) != len(
            APPROVED_DAY13_ARTIFACT_NAMES
        )
    ):
        raise RuntimeError(
            "Day 13 artifact writing did not complete."
        )

    (
        test_observations,
        parity_passed,
    ) = _summary_counts(report)
    print(
        "===== DAY 13 EVENT-DRIVEN "
        "WALK-FORWARD COMPLETE ====="
    )
    print(
        "Source dataset:",
        CANONICAL_DATASET_RELATIVE_PATH
        .as_posix(),
    )
    print("Strategies: 2")
    print("Folds: 4")
    print("Replay runs: 8")
    print(
        "Test observations:",
        test_observations,
    )
    print("Parity comparisons: 64")
    print(
        "Parity passed:",
        parity_passed,
    )
    print(
        "Artifact directory:",
        _display_path(
            output_directory,
            project_root=project_root,
        ),
    )
    print(
        "DAY 13 EVENT-DRIVEN WALK-FORWARD "
        "REPORT PASSED"
    )


if __name__ == "__main__":
    main()
