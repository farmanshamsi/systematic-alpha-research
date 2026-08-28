"""Artifact-writing contracts for Day 14 reporting."""

from __future__ import annotations

import hashlib
import json

import pytest

from systematic_alpha.analysis.cointegration_feasibility import (
    run_cointegration_feasibility,
)
from systematic_alpha.analysis.day14_cointegration_report import (
    APPROVED_DAY14_ARTIFACT_NAMES,
    MANIFEST_FILENAME,
    build_day14_cointegration_report,
    write_day14_cointegration_artifacts,
)
from tests.analysis.test_cointegration_statistics import (
    make_cointegrated_bars,
)


def build_report():
    """Build one deterministic synthetic Day 14 report."""

    results = run_cointegration_feasibility(
        make_cointegrated_bars()
    )
    return build_day14_cointegration_report(
        results
    )


def test_writer_creates_exact_hashed_artifact_set(
    tmp_path,
) -> None:
    output = tmp_path / "day14"

    paths = write_day14_cointegration_artifacts(
        build_report(),
        output,
    )

    assert tuple(
        path.name
        for path in paths
    ) == APPROVED_DAY14_ARTIFACT_NAMES

    assert {
        path.name
        for path in output.iterdir()
        if path.is_file()
    } == set(APPROVED_DAY14_ARTIFACT_NAMES)

    manifest = json.loads(
        (output / MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    expected_hash_names = set(
        APPROVED_DAY14_ARTIFACT_NAMES
    ) - {MANIFEST_FILENAME}

    assert set(
        manifest["artifact_sha256"]
    ) == expected_hash_names

    for name, expected_digest in (
        manifest["artifact_sha256"].items()
    ):
        actual_digest = hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()

        assert actual_digest == expected_digest


def test_writer_is_deterministic_and_requires_overwrite(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = build_report()

    write_day14_cointegration_artifacts(
        report,
        first,
    )

    with pytest.raises(FileExistsError):
        write_day14_cointegration_artifacts(
            report,
            first,
        )

    write_day14_cointegration_artifacts(
        report,
        first,
        overwrite=True,
    )
    write_day14_cointegration_artifacts(
        report,
        second,
    )

    for name in APPROVED_DAY14_ARTIFACT_NAMES:
        assert (
            first / name
        ).read_bytes() == (
            second / name
        ).read_bytes()
