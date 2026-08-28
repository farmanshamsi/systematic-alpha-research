"""Tests for the safe causal portfolio finalization runner."""

from __future__ import annotations

import pytest

from scripts.run_day25_causal_portfolio_finalization import parse_args


def test_parser_uses_frozen_development_defaults() -> None:
    arguments = parse_args([])
    assert arguments.artifact_directory.as_posix() == (
        "artifacts/day25_causal_portfolio_finalization"
    )
    assert arguments.overwrite is False


def test_parser_rejects_locked_or_broker_authorization() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--authorized-locked-final-test"])
    with pytest.raises(SystemExit):
        parse_args(["--authorized-paper-campaign"])
