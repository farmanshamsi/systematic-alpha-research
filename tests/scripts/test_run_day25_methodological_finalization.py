"""CLI contract tests for the Day 25 methodological finalization runner."""

from __future__ import annotations

from scripts.run_day25_methodological_finalization import (
    DEFAULT_OUTPUT,
    DEVELOPMENT_DATASET_ID,
    parse_args,
)


def test_default_arguments_are_development_only() -> None:
    arguments = parse_args([])
    assert arguments.output_dir == DEFAULT_OUTPUT
    assert arguments.overwrite is False
    assert "2025-12-31" in DEVELOPMENT_DATASET_ID
    assert "2026" not in DEVELOPMENT_DATASET_ID


def test_overwrite_requires_explicit_flag() -> None:
    arguments = parse_args(["--overwrite", "--output-dir", "artifacts/test"])
    assert arguments.overwrite is True
    assert arguments.output_dir.as_posix() == "artifacts/test"
