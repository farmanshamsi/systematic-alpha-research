"""Tests for the separately authorized locked-test command line."""

from __future__ import annotations

import pytest

from systematic_alpha.analysis.locked_final_test import AUTHORIZATION_CODE
from scripts.run_day25_locked_final_test import parse_args


def test_locked_runner_requires_explicit_authorization_argument() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    arguments = parse_args(["--authorization-code", AUTHORIZATION_CODE])
    assert arguments.authorization_code == AUTHORIZATION_CODE
    assert arguments.artifact_directory.as_posix() == "artifacts/day25_final_test"


def test_locked_runner_has_no_overwrite_or_broker_order_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--authorization-code", AUTHORIZATION_CODE,
                "--overwrite",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--authorization-code", AUTHORIZATION_CODE,
                "--authorized-paper-order",
            ]
        )
