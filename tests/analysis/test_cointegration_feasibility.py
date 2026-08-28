"""Frozen contracts for Day 14 cointegration feasibility."""

from __future__ import annotations

import inspect

import systematic_alpha.analysis.cointegration_feasibility as feasibility


def test_frozen_candidate_pairs_and_orientation() -> None:
    assert feasibility.CANDIDATE_PAIRS == (
        ("SPY", "QQQ"),
        ("SPY", "IWM"),
        ("QQQ", "IWM"),
    )

    assert feasibility.PAIR_IDS == (
        "SPY_QQQ",
        "SPY_IWM",
        "QQQ_IWM",
    )


def test_frozen_statistical_thresholds() -> None:
    assert feasibility.SIGNIFICANCE_LEVEL == 0.05
    assert feasibility.MULTIPLE_TESTING_METHOD == "holm"
    assert feasibility.MINIMUM_BETA == 0.10
    assert feasibility.MAXIMUM_BETA == 10.00
    assert feasibility.MAXIMUM_BETA_RELATIVE_DEVIATION == 0.25
    assert feasibility.MINIMUM_STATIONARY_FOLDS == 3
    assert feasibility.MINIMUM_HALF_LIFE_BARS == 1.0
    assert feasibility.MAXIMUM_HALF_LIFE_BARS == 130.0


def test_public_core_interfaces_are_narrow() -> None:
    assert tuple(
        inspect.signature(
            feasibility.build_cointegration_inputs
        ).parameters
    ) == ("bars",)

    assert tuple(
        inspect.signature(
            feasibility.run_cointegration_feasibility
        ).parameters
    ) == ("bars",)


def test_result_object_is_frozen_and_slotted() -> None:
    assert feasibility.CointegrationFeasibilityResults.__dataclass_params__.frozen
    assert "__dict__" not in feasibility.CointegrationFeasibilityResults.__slots__
