"""Focused contracts for the Day 27 OU mathematical revision."""

from __future__ import annotations

from pathlib import Path

import systematic_alpha.analysis.day27_mathematical_report as day27


def test_ou_revision_version_and_single_derivation_group_contract() -> None:
    assert day27.ARTIFACT_VERSION == (
        "day27_mathematical_revision_v0_2_ou_derivation"
    )
    matching_groups = [
        group
        for group in day27.REPORT_MATH_GROUPS
        if group[0] == day27.OU_DERIVATION_MATH_GROUP_ID
    ]
    assert len(matching_groups) == 1
    assert matching_groups[0][1] == 1
    assert len(matching_groups[0][2]) == 20

    marker = day27.report_math_placeholder(day27.OU_DERIVATION_MATH_GROUP_ID)
    source = Path(day27.__file__).read_text(encoding="utf-8")
    assert source.count(marker) == 1


def test_ou_derivation_discloses_implementation_and_failure_boundaries() -> None:
    source = Path(day27.__file__).read_text(encoding="utf-8")
    required_fragments = (
        "_rolling_ou_statistics",
        "n_{\\mathrm{OU}}-2",
        "ou_innovation_std",
        "ou_stationary_std",
        "ou_half_life_bars",
        "variance_ratio",
        "completed-information VWAP reference",
        "positive rolling volume",
        "conditional mean-zero, finite-variance innovations",
        "numerically unstable as `phi` approaches one",
        "heavily overlapping observations",
        "not proof of global stationarity",
    )
    assert all(fragment in source for fragment in required_fragments)
