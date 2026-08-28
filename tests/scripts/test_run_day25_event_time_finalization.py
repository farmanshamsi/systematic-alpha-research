"""CLI contract tests for the representative event-time runners."""

from __future__ import annotations

from scripts.download_day25_representative_trades import (
    CANONICAL_DATASET_ID,
    SESSION_WINDOWS,
)
from scripts.run_day25_event_time_finalization import DEFAULT_OUTPUT, parse_args


def test_download_scope_is_exactly_five_development_sessions() -> None:
    assert len(SESSION_WINDOWS) == 5
    assert [item[0] for item in SESSION_WINDOWS] == [
        "2025-01-15",
        "2025-04-15",
        "2025-07-15",
        "2025-10-15",
        "2025-12-15",
    ]
    assert "2026" not in CANONICAL_DATASET_ID


def test_event_time_runner_defaults_are_immutable() -> None:
    arguments = parse_args([])
    assert arguments.output_dir == DEFAULT_OUTPUT
    assert arguments.overwrite is False
