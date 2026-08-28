"""Known-answer tests for corrected fixed-holdings portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.portfolio_allocation_validation as allocation
from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS
from tests.day16_fixtures import make_day16_panel


def _two_sleeve_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sleeve_a": [0.10, -0.02],
            "sleeve_b": [-0.05, 0.08],
        },
        index=pd.Index([1, 2], name="session"),
    )


def _two_sleeve_weights() -> pd.Series:
    return pd.Series(
        [0.60, 0.40],
        index=pd.Index(("sleeve_a", "sleeve_b"), name="sleeve_id"),
    )


def test_two_sleeve_two_session_fixed_holdings_matches_hand_calculation() -> None:
    path = allocation.calculate_fixed_holdings_portfolio_path(
        _two_sleeve_returns(),
        _two_sleeve_weights(),
        sleeve_order=("sleeve_a", "sleeve_b"),
    )
    first_return = 0.60 * 0.10 + 0.40 * -0.05
    first_post = np.array([0.60 * 1.10, 0.40 * 0.95]) / (1.0 + first_return)
    second_return = float(first_post @ np.array([-0.02, 0.08]))
    second_post = first_post * np.array([0.98, 1.08]) / (1.0 + second_return)

    np.testing.assert_allclose(
        path.gross_portfolio_returns,
        [first_return, second_return],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        path.pre_return_weights,
        [[0.60, 0.40], first_post],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        path.post_return_weights,
        [first_post, second_post],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        path.ending_weights,
        second_post,
        rtol=0.0,
        atol=1e-15,
    )


def test_terminal_wealth_identity_and_weight_sums_are_exact() -> None:
    returns = _two_sleeve_returns()
    weights = _two_sleeve_weights()
    path = allocation.calculate_fixed_holdings_portfolio_path(
        returns,
        weights,
        sleeve_order=("sleeve_a", "sleeve_b"),
    )
    sleeve_identity = float(
        weights.to_numpy() @ np.prod(1.0 + returns.to_numpy(), axis=0)
    )
    recursive_identity = float(
        np.prod(1.0 + path.gross_portfolio_returns.to_numpy())
    )
    assert sleeve_identity == pytest.approx(1.0572, abs=1e-15)
    assert recursive_identity == pytest.approx(sleeve_identity, abs=1e-15)
    np.testing.assert_allclose(
        path.pre_return_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        path.post_return_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-15
    )


def test_weights_drift_but_identical_sleeve_returns_leave_weights_unchanged() -> None:
    diverging = allocation.calculate_fixed_holdings_portfolio_path(
        _two_sleeve_returns(),
        _two_sleeve_weights(),
        sleeve_order=("sleeve_a", "sleeve_b"),
    )
    assert not np.allclose(
        diverging.post_return_weights.iloc[0],
        _two_sleeve_weights(),
        rtol=0.0,
        atol=1e-15,
    )

    identical_returns = pd.DataFrame(
        {"sleeve_a": [0.03, -0.01], "sleeve_b": [0.03, -0.01]},
        index=pd.Index([1, 2], name="session"),
    )
    unchanged = allocation.calculate_fixed_holdings_portfolio_path(
        identical_returns,
        _two_sleeve_weights(),
        sleeve_order=("sleeve_a", "sleeve_b"),
    )
    expected = np.tile(_two_sleeve_weights().to_numpy(), (2, 1))
    np.testing.assert_allclose(
        unchanged.pre_return_weights, expected, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        unchanged.post_return_weights, expected, rtol=0.0, atol=1e-15
    )


def test_drifting_weights_are_not_clipped_to_target_cap() -> None:
    returns = pd.DataFrame(
        {
            "a": [0.50],
            "b": [-0.10],
            "c": [-0.10],
        }
    )
    weights = pd.Series([0.35, 0.35, 0.30], index=("a", "b", "c"))
    path = allocation.calculate_fixed_holdings_portfolio_path(
        returns,
        weights,
        sleeve_order=("a", "b", "c"),
    )
    assert path.ending_weights["a"] > allocation.MAXIMUM_WEIGHT


def test_fixed_holdings_primitive_does_not_mutate_inputs() -> None:
    returns = _two_sleeve_returns()
    weights = _two_sleeve_weights()
    returns_before = returns.copy(deep=True)
    weights_before = weights.copy(deep=True)
    allocation.calculate_fixed_holdings_portfolio_path(
        returns,
        weights,
        sleeve_order=("sleeve_a", "sleeve_b"),
    )
    pd.testing.assert_frame_equal(returns, returns_before, check_exact=True)
    pd.testing.assert_series_equal(weights, weights_before, check_exact=True)


@pytest.mark.parametrize(
    ("returns", "weights", "order", "message"),
    (
        (
            _two_sleeve_returns().loc[:, ["sleeve_b", "sleeve_a"]],
            _two_sleeve_weights(),
            ("sleeve_a", "sleeve_b"),
            "sleeve order",
        ),
        (
            _two_sleeve_returns(),
            _two_sleeve_weights().iloc[::-1],
            ("sleeve_a", "sleeve_b"),
            "sleeve order",
        ),
        (
            _two_sleeve_returns(),
            pd.Series([1.01, -0.01], index=("sleeve_a", "sleeve_b")),
            ("sleeve_a", "sleeve_b"),
            "nonnegative",
        ),
        (
            _two_sleeve_returns(),
            pd.Series([0.50, 0.40], index=("sleeve_a", "sleeve_b")),
            ("sleeve_a", "sleeve_b"),
            "fully invested",
        ),
        (
            _two_sleeve_returns().assign(sleeve_a=np.nan),
            _two_sleeve_weights(),
            ("sleeve_a", "sleeve_b"),
            "finite",
        ),
        (
            _two_sleeve_returns().assign(sleeve_a=-1.0),
            _two_sleeve_weights(),
            ("sleeve_a", "sleeve_b"),
            "greater than -1",
        ),
        (
            _two_sleeve_returns().sort_index(ascending=False),
            _two_sleeve_weights(),
            ("sleeve_a", "sleeve_b"),
            "ordered",
        ),
    ),
)
def test_invalid_fixed_holdings_inputs_fail_closed(
    returns: pd.DataFrame,
    weights: pd.Series,
    order: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(allocation.PortfolioAllocationValidationError, match=message):
        allocation.calculate_fixed_holdings_portfolio_path(
            returns,
            weights,
            sleeve_order=order,
        )


def test_zero_numerical_portfolio_wealth_fails_closed() -> None:
    near_total_loss = np.nextafter(-1.0, 0.0)
    returns = pd.DataFrame(
        {
            "a": np.full(400, near_total_loss),
            "b": np.full(400, near_total_loss),
        }
    )
    with pytest.raises(
        allocation.PortfolioAllocationValidationError,
        match="wealth must remain strictly positive",
    ):
        allocation.calculate_fixed_holdings_portfolio_path(
            returns,
            pd.Series([0.5, 0.5], index=("a", "b")),
            sleeve_order=("a", "b"),
        )


@pytest.fixture(scope="module")
def historical_results() -> allocation.PortfolioAllocationResults:
    return allocation.analyze_portfolio_allocation_panel(make_day16_panel())


@pytest.fixture(scope="module")
def fixed_results() -> allocation.FixedHoldingsPortfolioAllocationResults:
    return allocation.analyze_portfolio_allocation_panel_fixed_holdings(
        make_day16_panel()
    )


def test_corrected_version_targets_and_ledoit_wolf_are_historically_identical(
    historical_results: allocation.PortfolioAllocationResults,
    fixed_results: allocation.FixedHoldingsPortfolioAllocationResults,
) -> None:
    assert (
        fixed_results.accounting_version
        == allocation.FIXED_HOLDINGS_ACCOUNTING_VERSION
        == "fixed_holdings_fold_rebalance_v1"
    )
    pd.testing.assert_frame_equal(
        fixed_results.allocation_weights,
        historical_results.allocation_weights,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        fixed_results.minimum_variance_covariances,
        historical_results.minimum_variance_covariances,
        check_exact=True,
    )
    historical_target_fields = historical_results.allocation_diagnostics.drop(
        columns=["allocation_turnover", "allocation_cost"]
    )
    corrected_target_fields = fixed_results.allocation_diagnostics.drop(
        columns=["allocation_turnover", "allocation_cost"]
    )
    pd.testing.assert_frame_equal(
        corrected_target_fields,
        historical_target_fields,
        check_exact=True,
    )


def test_no_intrafold_cost_and_cost_occurs_only_on_first_fold_session(
    fixed_results: allocation.FixedHoldingsPortfolioAllocationResults,
) -> None:
    panel = make_day16_panel()
    for row in fixed_results.allocation_diagnostics.itertuples(index=False):
        year = int(row.fold_id[-4:])
        test = panel.loc[
            (panel.index >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & (panel.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ]
        weights = fixed_results.allocation_weights.loc[
            fixed_results.allocation_weights["fold_id"].eq(row.fold_id)
            & fixed_results.allocation_weights["allocation_rule"].eq(
                row.allocation_rule
            )
        ].sort_values("sleeve_order")["weight"]
        gross = allocation.calculate_fixed_holdings_portfolio_path(
            test,
            pd.Series(weights.to_numpy(), index=SLEEVE_IDS),
            sleeve_order=SLEEVE_IDS,
        ).gross_portfolio_returns.to_numpy()
        net = fixed_results.portfolio_return_panel.loc[
            fixed_results.portfolio_return_panel["fold_id"].eq(row.fold_id),
            row.allocation_rule,
        ].to_numpy()
        difference = gross - net
        assert difference[0] == pytest.approx(row.allocation_cost, abs=1e-15)
        np.testing.assert_allclose(
            difference[1:], 0.0, rtol=0.0, atol=1e-15
        )
        assert row.allocation_cost == pytest.approx(
            row.allocation_turnover * allocation.ALLOCATION_COST_RATE
        )


def test_fold_boundary_turnover_uses_previous_ending_drifted_weights(
    fixed_results: allocation.FixedHoldingsPortfolioAllocationResults,
) -> None:
    folds = ("wf_2022", "wf_2023", "wf_2024", "wf_2025")
    for rule in allocation.ALLOCATION_RULES:
        previous = np.zeros(len(SLEEVE_IDS), dtype="float64")
        for fold_id in folds:
            target = fixed_results.allocation_weights.loc[
                fixed_results.allocation_weights["fold_id"].eq(fold_id)
                & fixed_results.allocation_weights["allocation_rule"].eq(rule)
            ].sort_values("sleeve_order")["weight"].to_numpy()
            diagnostic = fixed_results.allocation_diagnostics.loc[
                fixed_results.allocation_diagnostics["fold_id"].eq(fold_id)
                & fixed_results.allocation_diagnostics["allocation_rule"].eq(rule)
            ].iloc[0]
            assert diagnostic["allocation_turnover"] == pytest.approx(
                np.abs(target - previous).sum(), abs=1e-15
            )
            previous = fixed_results.ending_fold_weights.loc[
                fixed_results.ending_fold_weights["fold_id"].eq(fold_id)
                & fixed_results.ending_fold_weights["allocation_rule"].eq(rule)
            ].sort_values("sleeve_order")["ending_weight"].to_numpy()

    equal = fixed_results.allocation_diagnostics.loc[
        fixed_results.allocation_diagnostics["allocation_rule"].eq("equal_weight")
    ]
    assert equal.iloc[0]["allocation_turnover"] == pytest.approx(1.0)
    assert equal.iloc[1:]["allocation_turnover"].gt(0.0).all()


def test_ending_weights_are_exposed_and_reconcile_to_weight_path(
    fixed_results: allocation.FixedHoldingsPortfolioAllocationResults,
) -> None:
    path = fixed_results.fixed_holdings_weight_path
    assert tuple(path.columns) == allocation.FIXED_HOLDINGS_WEIGHT_PATH_COLUMNS
    assert tuple(fixed_results.ending_fold_weights.columns) == (
        allocation.ENDING_FOLD_WEIGHT_COLUMNS
    )
    sums = path.groupby(
        ["session_date", "fold_id", "allocation_rule"], sort=False
    )[["pre_return_weight", "post_return_weight"]].sum()
    np.testing.assert_allclose(sums, 1.0, rtol=0.0, atol=1e-12)
    for ending in fixed_results.ending_fold_weights.itertuples(index=False):
        selected = path.loc[
            path["fold_id"].eq(ending.fold_id)
            & path["allocation_rule"].eq(ending.allocation_rule)
            & path["sleeve_id"].eq(ending.sleeve_id)
        ]
        assert ending.ending_weight == pytest.approx(
            selected.iloc[-1]["post_return_weight"], abs=1e-15
        )


def test_fixed_holdings_differs_from_daily_constant_mix_when_returns_diverge(
    historical_results: allocation.PortfolioAllocationResults,
    fixed_results: allocation.FixedHoldingsPortfolioAllocationResults,
) -> None:
    differences = []
    for rule in allocation.ALLOCATION_RULES:
        historical = historical_results.portfolio_return_panel[rule].to_numpy()
        corrected = fixed_results.portfolio_return_panel[rule].to_numpy()
        differences.append(not np.allclose(historical, corrected, rtol=0.0, atol=1e-15))
    assert all(differences)


def test_future_test_mutation_preserves_targets_and_completed_returns() -> None:
    original = make_day16_panel()
    mutated = original.copy(deep=True)
    mask = (
        mutated.index >= pd.Timestamp("2024-01-01", tz="UTC")
    ) & (mutated.index < pd.Timestamp("2025-01-01", tz="UTC"))
    perturbation = np.linspace(-0.02, 0.02, int(mask.sum()))
    for order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
        mutated.loc[mask, sleeve_id] += perturbation / order

    before = allocation.analyze_portfolio_allocation_panel_fixed_holdings(original)
    after = allocation.analyze_portfolio_allocation_panel_fixed_holdings(mutated)
    before_targets = before.allocation_weights.loc[
        before.allocation_weights["fold_id"].eq("wf_2024")
    ].reset_index(drop=True)
    after_targets = after.allocation_weights.loc[
        after.allocation_weights["fold_id"].eq("wf_2024")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_targets, after_targets, check_exact=True)
    pd.testing.assert_frame_equal(
        before.portfolio_return_panel.loc[
            before.portfolio_return_panel["session_date"].lt("2024-01-01")
        ].reset_index(drop=True),
        after.portfolio_return_panel.loc[
            after.portfolio_return_panel["session_date"].lt("2024-01-01")
        ].reset_index(drop=True),
        check_exact=True,
    )


def test_corrected_analysis_and_paths_do_not_mutate_panel() -> None:
    panel = make_day16_panel()
    before = panel.copy(deep=True)
    result = allocation.analyze_portfolio_allocation_panel_fixed_holdings(panel)
    pd.testing.assert_frame_equal(panel, before, check_exact=True)
    copied = result.copy_fixed_holdings_weight_path()
    copied.iloc[0, -1] = -99.0
    assert result.fixed_holdings_weight_path.iloc[0, -1] != -99.0


def test_historical_public_path_is_exactly_unchanged_by_corrected_analysis() -> None:
    panel = make_day16_panel()
    before = allocation.analyze_portfolio_allocation_panel(panel)
    allocation.analyze_portfolio_allocation_panel_fixed_holdings(panel)
    after = allocation.analyze_portfolio_allocation_panel(panel)
    for field in (
        "allocation_weights",
        "allocation_diagnostics",
        "fold_portfolio_performance",
        "aggregate_portfolio_performance",
        "portfolio_return_panel",
        "minimum_variance_covariances",
    ):
        pd.testing.assert_frame_equal(
            getattr(before, field), getattr(after, field), check_exact=True
        )
    assert before.evaluation_complete is after.evaluation_complete is True
