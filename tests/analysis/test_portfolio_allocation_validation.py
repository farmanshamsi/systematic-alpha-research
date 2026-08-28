"""Core mechanical and economic contracts for Day 16."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.portfolio_allocation_validation as allocation
from systematic_alpha.analysis.strategy_diversification import SLEEVE_IDS
from tests.day16_fixtures import make_day16_panel


def test_frozen_rule_schema_and_canonical_counts_are_exact() -> None:
    assert allocation.ALLOCATION_RULES == (
        "equal_weight",
        "inverse_volatility",
        "constrained_minimum_variance",
    )
    assert allocation.EXPECTED_CANONICAL_FOLD_SESSIONS == {
        "wf_2022": (505, 251),
        "wf_2023": (756, 250),
        "wf_2024": (1006, 252),
        "wf_2025": (1258, 250),
    }
    assert allocation.MAXIMUM_WEIGHT == 0.35
    assert allocation.WEIGHT_TOLERANCE == 1e-12
    assert allocation.SOLVER_CONSTRAINT_TOLERANCE == 1e-10
    assert allocation.SLSQP_FTOL == 1e-12
    assert allocation.SLSQP_MAXITER == 10_000
    assert allocation.ALLOCATION_COST_RATE == 1.0 / 10_000.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda panel: panel.loc[:, list(reversed(SLEEVE_IDS))], "sleeve order"),
        (lambda panel: panel.rename_axis("date"), "named session_date"),
        (
            lambda panel: panel.set_axis(panel.index.tz_localize(None)),
            "must use UTC",
        ),
        (
            lambda panel: panel.set_axis(panel.index + pd.Timedelta(hours=1)),
            "normalized whole dates",
        ),
        (lambda panel: panel.sort_index(ascending=False), "chronological"),
    ),
)
def test_exact_input_schema_order_and_utc_calendar_fail_closed(
    mutation: object,
    message: str,
) -> None:
    panel = make_day16_panel()
    changed = mutation(panel.copy(deep=True))  # type: ignore[operator]
    with pytest.raises(allocation.PortfolioAllocationValidationError, match=message):
        allocation.analyze_portfolio_allocation_panel(changed)


@pytest.mark.parametrize("invalid", [None, [], "panel", 7])
def test_invalid_panel_types_fail_closed(invalid: object) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        allocation.analyze_portfolio_allocation_panel(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (np.nan, "missing or non-finite"),
        (np.inf, "missing or non-finite"),
        (-1.0, "greater than -1"),
    ),
)
def test_missing_nonfinite_and_invalid_simple_returns_are_rejected(
    value: float,
    message: str,
) -> None:
    panel = make_day16_panel()
    panel.iloc[3, 2] = value
    with pytest.raises(allocation.PortfolioAllocationValidationError, match=message):
        allocation.analyze_portfolio_allocation_panel(panel)


def test_locked_period_guard_rejects_any_2026_session() -> None:
    panel = make_day16_panel()
    locked = panel.iloc[[-1]].copy(deep=True)
    locked.index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-02", tz="UTC")],
        name="session_date",
    )
    panel = pd.concat((panel, locked))
    with pytest.raises(allocation.PortfolioAllocationValidationError, match="2026"):
        allocation.analyze_portfolio_allocation_panel(panel)


def test_zero_variance_sleeve_is_rejected() -> None:
    panel = make_day16_panel()
    panel[SLEEVE_IDS[0]] = 0.001
    with pytest.raises(
        allocation.PortfolioAllocationValidationError,
        match="near-zero variance",
    ):
        allocation.analyze_portfolio_allocation_panel(panel)


def test_synthetic_panels_bypass_only_canonical_count_requirement() -> None:
    panel = make_day16_panel()
    result = allocation.analyze_portfolio_allocation_panel(panel)
    assert result.evaluation_complete is True
    with pytest.raises(
        allocation.PortfolioAllocationValidationError,
        match="canonical session counts",
    ):
        allocation.analyze_portfolio_allocation_panel(
            panel,
            require_canonical_counts=True,
        )


def test_require_canonical_counts_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        allocation.analyze_portfolio_allocation_panel(
            make_day16_panel(),
            require_canonical_counts=1,  # type: ignore[arg-type]
        )


def test_equal_weights_are_exact_and_mechanically_valid() -> None:
    weights = allocation.calculate_equal_weights()
    np.testing.assert_array_equal(
        weights,
        np.full(6, 1.0 / 6.0, dtype="float64"),
    )
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(weights).sum() == pytest.approx(1.0, abs=1e-12)
    assert weights.min() >= 0.0
    assert weights.max() <= 0.35


def test_inverse_volatility_uses_ddof_one_and_deterministic_water_filling() -> None:
    panel = make_day16_panel().loc[lambda frame: frame.index < "2022-01-01"]
    weights = allocation.calculate_inverse_volatility_weights(panel)
    volatility = panel.std(axis=0, ddof=1).to_numpy(dtype="float64")
    raw = 1.0 / volatility

    expected = np.zeros(6)
    active = np.ones(6, dtype=bool)
    remaining = 1.0
    while active.any():
        candidates = remaining * raw[active] / raw[active].sum()
        indices = np.flatnonzero(active)
        violating = candidates > 0.35
        if not violating.any():
            expected[indices] = candidates
            break
        expected[indices[violating]] = 0.35
        active[indices[violating]] = False
        remaining = 1.0 - expected.sum()

    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-15)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert weights.max() <= 0.35 + 1e-12


def test_inverse_volatility_rejects_near_zero_training_volatility() -> None:
    panel = make_day16_panel().loc[lambda frame: frame.index < "2022-01-01"]
    panel[SLEEVE_IDS[0]] = 0.001
    with pytest.raises(allocation.PortfolioAllocationValidationError):
        allocation.calculate_inverse_volatility_weights(panel)


def test_minimum_variance_records_finite_symmetric_ledoit_wolf_covariance() -> None:
    training = make_day16_panel().loc[lambda frame: frame.index < "2022-01-01"]
    weights, shrinkage, covariance, status = (
        allocation.calculate_constrained_minimum_variance_weights(training)
    )
    assert status == "success"
    assert 0.0 <= shrinkage <= 1.0
    assert covariance.shape == (6, 6)
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-15)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert weights.min() >= -1e-12
    assert weights.max() <= 0.35 + 1e-12


def test_minimum_variance_solver_failure_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = make_day16_panel().loc[lambda frame: frame.index < "2022-01-01"]
    monkeypatch.setattr(
        allocation,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.full(6, 1.0 / 6.0),
            fun=0.01,
            success=False,
        ),
    )
    with pytest.raises(
        allocation.PortfolioAllocationValidationError,
        match="not successful",
    ):
        allocation.calculate_constrained_minimum_variance_weights(training)


@pytest.mark.parametrize(
    "weights",
    (
        np.array([0.2] * 6),
        np.array([-0.01, 0.21, 0.2, 0.2, 0.2, 0.2]),
        np.array([0.36, 0.128, 0.128, 0.128, 0.128, 0.128]),
        np.array([np.nan, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([0.5, 0.5]),
    ),
)
def test_invalid_weight_vectors_fail_closed(weights: np.ndarray) -> None:
    with pytest.raises(allocation.PortfolioAllocationValidationError):
        allocation._validate_weights(weights)


def test_portfolio_metrics_use_frozen_formulas() -> None:
    returns = pd.Series([-0.10, 0.05, -0.02, 0.03], dtype="float64")
    metrics = allocation.calculate_portfolio_metrics(returns)
    values = returns.to_numpy()
    wealth = np.cumprod(1.0 + values)
    wealth_from_origin = np.concatenate(([1.0], wealth))
    expected_drawdown = np.min(
        wealth_from_origin / np.maximum.accumulate(wealth_from_origin) - 1.0
    )
    quantile = np.quantile(values, 0.05, method="linear")

    assert metrics["observations"] == 4
    assert metrics["cumulative_return"] == pytest.approx(wealth[-1] - 1.0)
    assert metrics["annualized_return"] == pytest.approx(
        wealth[-1] ** (252.0 / 4.0) - 1.0
    )
    assert metrics["annualized_volatility"] == pytest.approx(
        values.std(ddof=1) * np.sqrt(252.0)
    )
    assert metrics["sharpe_ratio"] == pytest.approx(
        values.mean() / values.std(ddof=1) * np.sqrt(252.0)
    )
    assert metrics["maximum_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["historical_var_95"] == pytest.approx(max(0.0, -quantile))
    assert metrics["historical_es_95"] == pytest.approx(
        max(0.0, -values[values <= quantile].mean())
    )


def test_positive_tail_losses_are_floored_at_zero() -> None:
    metrics = allocation.calculate_portfolio_metrics(
        pd.Series([0.01, 0.02, 0.015])
    )
    assert metrics["historical_var_95"] == 0.0
    assert metrics["historical_es_95"] == 0.0


@pytest.mark.parametrize(
    "returns",
    (
        pd.Series([0.01]),
        pd.Series([0.01, np.nan]),
        pd.Series([0.01, -1.0]),
        pd.Series([0.01, 0.01]),
    ),
)
def test_nonfinite_or_undefined_portfolio_metrics_fail_closed(
    returns: pd.Series,
) -> None:
    with pytest.raises(allocation.PortfolioAllocationValidationError):
        allocation.calculate_portfolio_metrics(returns)


@pytest.fixture(scope="module")
def day16_results() -> allocation.PortfolioAllocationResults:
    return allocation.analyze_portfolio_allocation_panel(make_day16_panel())


def test_result_schemas_counts_and_order_are_exact(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    assert tuple(day16_results.allocation_weights.columns) == (
        allocation.ALLOCATION_WEIGHT_COLUMNS
    )
    assert tuple(day16_results.allocation_diagnostics.columns) == (
        allocation.ALLOCATION_DIAGNOSTIC_COLUMNS
    )
    assert tuple(day16_results.fold_portfolio_performance.columns) == (
        allocation.FOLD_PORTFOLIO_PERFORMANCE_COLUMNS
    )
    assert tuple(day16_results.aggregate_portfolio_performance.columns) == (
        allocation.AGGREGATE_PORTFOLIO_PERFORMANCE_COLUMNS
    )
    assert tuple(day16_results.portfolio_return_panel.columns) == (
        allocation.PORTFOLIO_RETURN_PANEL_COLUMNS
    )
    assert tuple(day16_results.minimum_variance_covariances.columns) == (
        allocation.MINIMUM_VARIANCE_COVARIANCE_COLUMNS
    )
    assert len(day16_results.allocation_weights) == 72
    assert len(day16_results.allocation_diagnostics) == 12
    assert len(day16_results.fold_portfolio_performance) == 12
    assert len(day16_results.aggregate_portfolio_performance) == 3
    assert len(day16_results.minimum_variance_covariances) == 24
    assert tuple(
        day16_results.aggregate_portfolio_performance["allocation_rule"]
    ) == allocation.ALLOCATION_RULES


def test_weights_are_training_only_under_test_return_mutation() -> None:
    original = make_day16_panel()
    mutated = original.copy(deep=True)
    test_2024 = (
        (mutated.index >= pd.Timestamp("2024-01-01", tz="UTC"))
        & (mutated.index < pd.Timestamp("2025-01-01", tz="UTC"))
    )
    perturbation = np.linspace(-0.03, 0.03, int(test_2024.sum()))
    for order, sleeve_id in enumerate(SLEEVE_IDS, start=1):
        mutated.loc[test_2024, sleeve_id] += perturbation / order

    before = allocation.analyze_portfolio_allocation_panel(original)
    after = allocation.analyze_portfolio_allocation_panel(mutated)
    before_weights = before.allocation_weights.loc[
        before.allocation_weights["fold_id"].eq("wf_2024"), "weight"
    ].to_numpy()
    after_weights = after.allocation_weights.loc[
        after.allocation_weights["fold_id"].eq("wf_2024"), "weight"
    ].to_numpy()
    np.testing.assert_array_equal(before_weights, after_weights)


def test_fold_weights_are_fixed_and_cost_is_charged_only_on_first_session(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    source = make_day16_panel()
    for diagnostic in day16_results.allocation_diagnostics.itertuples(index=False):
        fold_id = diagnostic.fold_id
        year = int(fold_id[-4:])
        test = source.loc[
            (source.index >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & (source.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ]
        weight_rows = day16_results.allocation_weights.loc[
            day16_results.allocation_weights["fold_id"].eq(fold_id)
            & day16_results.allocation_weights["allocation_rule"].eq(
                diagnostic.allocation_rule
            )
        ]
        weights = weight_rows["weight"].to_numpy(dtype="float64")
        expected = test.to_numpy(dtype="float64") @ weights
        expected[0] -= float(diagnostic.allocation_cost)
        actual = day16_results.portfolio_return_panel.loc[
            day16_results.portfolio_return_panel["fold_id"].eq(fold_id),
            diagnostic.allocation_rule,
        ].to_numpy(dtype="float64")
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_turnover_cost_concentration_and_aggregate_costs_reconcile(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    diagnostics = day16_results.allocation_diagnostics
    weights = day16_results.allocation_weights
    for row in diagnostics.itertuples(index=False):
        vector = weights.loc[
            weights["fold_id"].eq(row.fold_id)
            & weights["allocation_rule"].eq(row.allocation_rule),
            "weight",
        ].to_numpy(dtype="float64")
        assert row.allocation_cost == pytest.approx(
            row.allocation_turnover * allocation.ALLOCATION_COST_RATE
        )
        assert row.herfindahl_concentration == pytest.approx(vector @ vector)
        assert row.effective_sleeve_count == pytest.approx(1.0 / (vector @ vector))
        assert row.maximum_weight == pytest.approx(vector.max())
        assert row.gross_weight == pytest.approx(np.abs(vector).sum())
        assert bool(row.constraint_valid)

    equal = diagnostics.loc[diagnostics["allocation_rule"].eq("equal_weight")]
    assert tuple(equal["allocation_turnover"]) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    for aggregate in day16_results.aggregate_portfolio_performance.itertuples(
        index=False
    ):
        selected = diagnostics.loc[
            diagnostics["allocation_rule"].eq(aggregate.allocation_rule)
        ]
        assert aggregate.total_allocation_turnover == pytest.approx(
            selected["allocation_turnover"].sum()
        )
        assert aggregate.total_allocation_cost == pytest.approx(
            selected["allocation_cost"].sum()
        )


def test_results_are_defensive_frozen_dataclasses(
    day16_results: allocation.PortfolioAllocationResults,
) -> None:
    assert day16_results.__dataclass_params__.frozen
    assert not hasattr(day16_results, "__dict__")
    for name in (
        "allocation_weights",
        "allocation_diagnostics",
        "fold_portfolio_performance",
        "aggregate_portfolio_performance",
        "portfolio_return_panel",
        "minimum_variance_covariances",
    ):
        retained = getattr(day16_results, name)
        copied = getattr(day16_results, f"copy_{name}")()
        original = retained.iloc[0, 0]
        replacement = (
            original + pd.Timedelta(days=1)
            if isinstance(original, pd.Timestamp)
            else "changed"
        )
        copied.iloc[0, 0] = replacement
        assert retained.iloc[0, 0] == original


def test_negative_economic_outcomes_do_not_change_mechanical_completion() -> None:
    results = allocation.analyze_portfolio_allocation_panel(
        make_day16_panel(mean_return=-0.0015)
    )
    assert results.evaluation_complete is True
    assert (results.aggregate_portfolio_performance["cumulative_return"] < 0).any()
    assert (results.aggregate_portfolio_performance["sharpe_ratio"] < 0).any()


def test_canonical_wrapper_rebuilds_the_panel_through_day15(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = make_day16_panel()
    bars = pd.DataFrame({"placeholder": [1]})
    calls: list[pd.DataFrame] = []

    class FakeDay15Results:
        def copy_session_return_panel(self) -> pd.DataFrame:
            return panel.copy(deep=True)

    def day15_spy(frame: pd.DataFrame) -> FakeDay15Results:
        calls.append(frame.copy(deep=True))
        return FakeDay15Results()

    monkeypatch.setattr(allocation, "run_strategy_diversification", day15_spy)
    result = allocation.run_portfolio_allocation(
        bars,
        require_canonical_counts=False,
    )
    assert len(calls) == 1
    assert calls[0].equals(bars)
    assert result.evaluation_complete is True
