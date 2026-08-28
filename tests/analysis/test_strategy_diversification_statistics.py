"""Numerical contracts for Day 15 strategy diversification."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt

from systematic_alpha.analysis.strategy_diversification import (
    PSD_TOLERANCE,
    SLEEVE_IDS,
    analyze_strategy_diversification_panel,
    calculate_correlation_eigenvalues,
    calculate_covariance_condition_number,
    calculate_covariance_eigenvalues,
    calculate_entropy_effective_rank,
    calculate_equal_weight_diversification_ratio,
    calculate_pairwise_correlations,
    calculate_panel_diagnostics,
    calculate_sample_covariance,
    is_positive_semidefinite,
)


def make_weakly_correlated_panel(
    *,
    seed: int = 15,
) -> pd.DataFrame:
    """Build six development-only sleeves with weak dependence."""

    sessions = pd.bdate_range(
        "2020-01-02",
        "2025-12-31",
        tz="UTC",
        name="session_date",
    )
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.001, len(sessions))
    independent = rng.normal(
        0.0,
        0.01,
        (len(sessions), len(SLEEVE_IDS)),
    )
    values = independent + common[:, None]
    return pd.DataFrame(
        values,
        index=sessions,
        columns=SLEEVE_IDS,
    )


def make_nearly_identical_panel() -> pd.DataFrame:
    """Build six non-degenerate but redundant sleeves."""

    sessions = pd.bdate_range(
        "2020-01-02",
        "2025-12-31",
        tz="UTC",
        name="session_date",
    )
    rng = np.random.default_rng(1515)
    common = rng.normal(0.0, 0.01, len(sessions))
    noise = rng.normal(
        0.0,
        1e-5,
        (len(sessions), len(SLEEVE_IDS)),
    )
    return pd.DataFrame(
        common[:, None] + noise,
        index=sessions,
        columns=SLEEVE_IDS,
    )


def test_all_15_unordered_pearson_pairs_are_exact() -> None:
    panel = make_weakly_correlated_panel()
    pairs = calculate_pairwise_correlations(panel)

    assert len(pairs) == 15
    assert not pairs[["sleeve_a", "sleeve_b"]].duplicated().any()
    assert (
        pairs["sleeve_a"].map(SLEEVE_IDS.index)
        < pairs["sleeve_b"].map(SLEEVE_IDS.index)
    ).all()

    expected = panel.corr(method="pearson").loc[
        SLEEVE_IDS[0],
        SLEEVE_IDS[1],
    ]
    assert math.isclose(
        pairs.iloc[0]["correlation"],
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert pairs["observations"].eq(len(panel)).all()


def test_correlation_concentration_and_spectrum_are_consistent() -> None:
    panel = make_weakly_correlated_panel()
    pairwise = calculate_pairwise_correlations(panel)
    diagnostics = calculate_panel_diagnostics(panel)
    eigenvalues = calculate_correlation_eigenvalues(panel)

    absolute = pairwise["correlation"].abs()
    assert np.all(np.diff(eigenvalues) <= 0.0)
    assert np.isclose(eigenvalues.sum(), 6.0)
    assert np.isclose(
        diagnostics["minimum_pairwise_correlation"],
        pairwise["correlation"].min(),
    )
    assert np.isclose(
        diagnostics["maximum_pairwise_correlation"],
        pairwise["correlation"].max(),
    )
    assert np.isclose(
        diagnostics["maximum_absolute_correlation"],
        absolute.max(),
    )
    assert np.isclose(
        diagnostics["mean_absolute_correlation"],
        absolute.mean(),
    )
    assert np.isclose(
        diagnostics["median_absolute_correlation"],
        absolute.median(),
    )
    assert np.isclose(
        diagnostics["pc1_share"],
        eigenvalues[0] / eigenvalues.sum(),
    )


def test_effective_rank_ignores_numerically_zero_probabilities() -> None:
    eigenvalues = np.array(
        [3.0, 1.0, 0.0, -1e-16],
        dtype="float64",
    )

    expected = math.exp(
        -(
            0.75 * math.log(0.75)
            + 0.25 * math.log(0.25)
        )
    )

    assert math.isclose(
        calculate_entropy_effective_rank(eigenvalues),
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_sample_covariance_uses_ddof_one_and_descending_eigenvalues() -> None:
    panel = make_weakly_correlated_panel()
    covariance = calculate_sample_covariance(panel)
    expected = np.cov(
        panel.to_numpy(),
        rowvar=False,
        ddof=1,
    )
    eigenvalues = calculate_covariance_eigenvalues(panel)

    np.testing.assert_allclose(covariance, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        eigenvalues,
        np.linalg.eigvalsh(expected)[::-1],
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(np.diff(eigenvalues) <= 0.0)


def test_singular_covariance_has_positive_infinite_condition_number() -> None:
    panel = make_weakly_correlated_panel()
    panel[SLEEVE_IDS[-1]] = panel[SLEEVE_IDS[0]]
    covariance = calculate_sample_covariance(panel)

    assert math.isinf(
        calculate_covariance_condition_number(covariance)
    )
    assert (
        calculate_covariance_condition_number(covariance)
        > 0.0
    )


def test_psd_status_uses_documented_tolerance() -> None:
    assert is_positive_semidefinite(
        np.array([1.0, -0.5 * PSD_TOLERANCE])
    )
    assert not is_positive_semidefinite(
        np.array([1.0, -2.0 * PSD_TOLERANCE])
    )
    assert not is_positive_semidefinite(
        np.array([1.0, np.nan])
    )


def test_equal_weight_diversification_ratio_uses_one_sixth_weights() -> None:
    variances = np.square(
        np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    )
    covariance = np.diag(variances)
    expected = (
        np.sqrt(variances).sum()
        / math.sqrt(variances.sum())
    )

    assert np.isclose(
        calculate_equal_weight_diversification_ratio(covariance),
        expected,
    )


def test_weak_sleeves_pass_and_redundant_sleeves_fail_gates() -> None:
    weak = analyze_strategy_diversification_panel(
        make_weakly_correlated_panel()
    )
    redundant = analyze_strategy_diversification_panel(
        make_nearly_identical_panel()
    )

    weak_gate = weak.ensemble_feasibility.iloc[0]
    redundant_gate = redundant.ensemble_feasibility.iloc[0]

    assert bool(weak_gate["ensemble_feasible"])
    assert not bool(redundant_gate["ensemble_feasible"])
    assert not bool(
        redundant_gate["maximum_absolute_correlation_gate"]
    )
    assert not bool(redundant_gate["median_effective_rank_gate"])
    assert not bool(redundant_gate["median_pc1_share_gate"])


def test_future_test_mutation_cannot_change_training_diagnostics() -> None:
    original = make_weakly_correlated_panel()
    mutated = original.copy(deep=True)
    future = mutated.index.year == 2025
    rng = np.random.default_rng(150)
    common = rng.normal(0.0, 0.02, int(future.sum()))
    mutated.loc[future, :] = (
        common[:, None]
        + rng.normal(
            0.0,
            1e-4,
            (int(future.sum()), len(SLEEVE_IDS)),
        )
    )

    original_results = analyze_strategy_diversification_panel(original)
    mutated_results = analyze_strategy_diversification_panel(mutated)
    original_training = (
        original_results.fold_covariance_diagnostics.loc[
            lambda frame: frame["sample"].eq("train")
        ].reset_index(drop=True)
    )
    mutated_training = (
        mutated_results.fold_covariance_diagnostics.loc[
            lambda frame: frame["sample"].eq("train")
        ].reset_index(drop=True)
    )

    pdt.assert_frame_equal(
        original_training,
        mutated_training,
        check_exact=True,
    )
