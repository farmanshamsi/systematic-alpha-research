"""Synthetic contracts for causal Day 30 cointegration chronology."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.cointegration_feasibility as historical
from systematic_alpha.analysis.causal_cointegration_chronology import (
    CHRONOLOGY_LEDGER_COLUMNS,
    EX_POST_BETA_COLUMNS,
    METHOD_VERSION,
    PAIR_SUMMARY_COLUMNS,
    REFERENCE_BETA_ZERO_TOLERANCE,
    CausalCointegrationChronologyError,
    analyze_causal_cointegration_chronology,
    compare_successive_betas,
    estimate_cointegrating_regression,
    inclusive_training_end_timestamp,
    validate_maximum_information_timestamp,
)
from systematic_alpha.analysis.cointegration_feasibility import (
    CANDIDATE_PAIRS,
    MAXIMUM_BETA_RELATIVE_DEVIATION,
    PAIR_IDS,
    CointegrationInputs,
    PairCointegrationInput,
    build_cointegration_inputs,
    run_cointegration_feasibility,
)
from systematic_alpha.analysis.trend_family_walk_forward import build_walk_forward_folds


HISTORICAL_SOURCE_SHA256 = (
    "0d590983f8d87e550888d8cc7699063c5ee063ca0aebde38b8f09bba54cfeff0"
)
V1_BETA_ESTIMATES = np.array(
    [
        1.2012139138221194,
        1.1981813596898971,
        1.1979477366670888,
        1.1929406564291625,
        0.010565307849681323,
        0.06940513979413701,
        0.027698665174424025,
        -0.3818988309241873,
        0.01347807908403834,
        0.0611187981969795,
        0.02510962131260759,
        -0.3199976005806069,
    ],
    dtype="float64",
)
V1_RELATIVE_DRIFTS = np.array(
    [
        np.nan,
        0.002524574596853468,
        0.0001949813531307537,
        0.004179715094964749,
        np.nan,
        5.569154517937728,
        0.6009133436431192,
        14.78762581226547,
        np.nan,
        3.534681672061158,
        0.5891669657560689,
        13.744023360477183,
    ],
    dtype="float64",
)
V1_FOLD_ELIGIBILITY = (
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
)


def make_causal_inputs() -> CointegrationInputs:
    """Build deterministic daily and intraday log-price pair inputs."""

    sessions = pd.bdate_range(
        "2020-01-02", "2025-12-31", tz="UTC", name="session_date"
    )
    rng = np.random.default_rng(30)
    qqq = 5.0 + np.cumsum(rng.normal(0.0, 0.01, len(sessions)))
    residual = np.empty(len(sessions), dtype="float64")
    residual[0] = 0.0
    for index in range(1, len(sessions)):
        residual[index] = (
            0.60 * residual[index - 1] + rng.normal(0.0, 0.005)
        )
    values = {
        "SPY": 0.20 + 1.20 * qqq + residual,
        "QQQ": qqq,
        "IWM": 4.50 + np.cumsum(rng.normal(0.0, 0.012, len(sessions))),
    }
    pair_inputs: list[PairCointegrationInput] = []
    for pair_id, (y_symbol, x_symbol) in zip(
        PAIR_IDS, CANDIDATE_PAIRS, strict=True
    ):
        daily = pd.DataFrame(
            {
                "session_date": sessions,
                "y_log_price": values[y_symbol],
                "x_log_price": values[x_symbol],
            }
        )
        intraday_records: list[dict[str, object]] = []
        for row_number, session in enumerate(sessions):
            for bar_number, hour_minute in enumerate(("14:30", "14:45")):
                perturbation = (bar_number - 0.5) * 0.0002
                intraday_records.append(
                    {
                        "timestamp": pd.Timestamp(
                            f"{session.date()} {hour_minute}:00", tz="UTC"
                        ),
                        "session_date": session,
                        "y_log_price": float(values[y_symbol][row_number])
                        + perturbation,
                        "x_log_price": float(values[x_symbol][row_number])
                        - perturbation,
                    }
                )
        pair_inputs.append(
            PairCointegrationInput(
                pair_id=pair_id,
                y_symbol=y_symbol,
                x_symbol=x_symbol,
                daily_log_prices=daily,
                intraday_log_prices=pd.DataFrame.from_records(intraday_records),
            )
        )
    return CointegrationInputs(tuple(pair_inputs))


@pytest.fixture(scope="module")
def inputs() -> CointegrationInputs:
    return make_causal_inputs()


@pytest.fixture(scope="module")
def results(inputs: CointegrationInputs):
    return analyze_causal_cointegration_chronology(inputs)


def _replace_pair(
    inputs: CointegrationInputs,
    pair_number: int,
    *,
    daily: pd.DataFrame | None = None,
    intraday: pd.DataFrame | None = None,
    y_symbol: str | None = None,
    x_symbol: str | None = None,
) -> CointegrationInputs:
    items = list(inputs.pair_inputs)
    original = items[pair_number]
    items[pair_number] = PairCointegrationInput(
        pair_id=original.pair_id,
        y_symbol=original.y_symbol if y_symbol is None else y_symbol,
        x_symbol=original.x_symbol if x_symbol is None else x_symbol,
        daily_log_prices=(
            original.daily_log_prices if daily is None else daily
        ),
        intraday_log_prices=(
            original.intraday_log_prices if intraday is None else intraday
        ),
    )
    return CointegrationInputs(tuple(items))


def test_exact_known_answer_ols_intercept_and_beta() -> None:
    x = np.arange(1.0, 11.0)
    frame = pd.DataFrame(
        {"y_log_price": 2.5 + 1.75 * x, "x_log_price": x}
    )
    estimate = estimate_cointegrating_regression(frame)
    assert estimate.alpha == pytest.approx(2.5, abs=1e-12)
    assert estimate.beta == pytest.approx(1.75, abs=1e-12)
    np.testing.assert_allclose(estimate.residuals, 0.0, atol=1e-12, rtol=0.0)
    assert estimate.observations == 10
    assert estimate.design_rank == 2
    assert np.isfinite(estimate.design_condition_number)
    assert estimate.largest_singular_value >= estimate.smallest_singular_value > 0.0


def test_ill_conditioned_full_rank_design_is_retained_as_diagnostic() -> None:
    x = 1.0 + np.linspace(-1e-8, 1e-8, 200)
    frame = pd.DataFrame(
        {
            "y_log_price": -0.75 + 2.25 * x,
            "x_log_price": x,
        }
    )
    estimate = estimate_cointegrating_regression(frame)
    assert estimate.design_rank == 2
    assert estimate.alpha == pytest.approx(-0.75, abs=1e-7)
    assert estimate.beta == pytest.approx(2.25, abs=1e-7)
    assert np.isfinite(estimate.design_condition_number)
    assert estimate.design_condition_number > 1e8
    assert estimate.largest_singular_value >= estimate.smallest_singular_value > 0.0


def test_cointegrating_ols_never_calls_normal_equation_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lstsq = np.linalg.lstsq
    calls: list[object] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("np.linalg.solve must not be called")

    def lstsq_spy(*args, **kwargs):
        calls.append(kwargs.get("rcond", "missing"))
        return original_lstsq(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", lstsq_spy)
    x = np.linspace(-2.0, 2.0, 50)
    estimate = estimate_cointegrating_regression(
        pd.DataFrame(
            {"y_log_price": 1.5 - 0.8 * x, "x_log_price": x}
        )
    )
    assert estimate.alpha == pytest.approx(1.5, abs=1e-12)
    assert estimate.beta == pytest.approx(-0.8, abs=1e-12)
    assert calls == [None]
    estimator_source = inspect.getsource(estimate_cointegrating_regression)
    assert "np.linalg.solve" not in estimator_source
    assert "np.linalg.lstsq" in estimator_source


@pytest.mark.parametrize("failure_mode", ["coefficient", "singular_value"])
def test_nonfinite_svd_outputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    def invalid_lstsq(*args, **kwargs):
        gamma = np.array([1.0, 2.0], dtype="float64")
        singular_values = np.array([2.0, 1.0], dtype="float64")
        if failure_mode == "coefficient":
            gamma[0] = np.nan
        else:
            singular_values[1] = np.nan
        return gamma, np.array([0.0]), 2, singular_values

    monkeypatch.setattr(np.linalg, "lstsq", invalid_lstsq)
    frame = pd.DataFrame(
        {
            "y_log_price": np.linspace(1.0, 2.0, 20),
            "x_log_price": np.linspace(2.0, 3.0, 20),
        }
    )
    with pytest.raises(CausalCointegrationChronologyError, match="finite|invalid"):
        estimate_cointegrating_regression(frame)


def test_first_fold_is_baseline_without_fabricated_reference(results) -> None:
    baseline = results.fold_chronology.loc[
        results.fold_chronology["fold_order"].eq(1)
    ]
    assert len(baseline) == 3
    assert baseline["stability_status"].eq("baseline").all()
    assert baseline["reference_fold"].eq("").all()
    assert baseline["reference_beta"].isna().all()
    assert baseline["relative_beta_drift"].isna().all()
    assert baseline["stability_pass"].all()


def test_later_folds_reference_only_immediately_preceding_beta(results) -> None:
    for _, pair_rows in results.fold_chronology.groupby("pair_id", sort=False):
        ordered = pair_rows.sort_values("fold_order", kind="stable").reset_index(
            drop=True
        )
        for row_number in range(1, len(ordered)):
            current = ordered.iloc[row_number]
            preceding = ordered.iloc[row_number - 1]
            assert current["reference_fold"] == preceding["fold_id"]
            assert current["reference_beta"] == preceding["beta_estimate"]
            assert (
                current["reference_maximum_information_timestamp"]
                == preceding["maximum_timestamp_used_for_estimation"]
            )


def test_known_answer_relative_drift_and_frozen_threshold() -> None:
    comparison = compare_successive_betas(
        current_beta=1.20,
        reference_beta=1.00,
    )
    assert comparison.relative_beta_drift == pytest.approx(0.20, abs=1e-15)
    assert comparison.stability_status == "stable"
    assert comparison.stability_pass
    assert MAXIMUM_BETA_RELATIVE_DEVIATION == 0.25


def test_future_test_mutation_cannot_change_same_fold_estimates_or_eligibility(
    inputs: CointegrationInputs,
) -> None:
    baseline = analyze_causal_cointegration_chronology(inputs)
    daily = inputs.pair_inputs[0].daily_log_prices.copy(deep=True)
    future_test = pd.to_datetime(daily["session_date"], utc=True).between(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-12-31 23:59:59", tz="UTC"),
    )
    daily.loc[future_test, "y_log_price"] += 5.0
    changed = analyze_causal_cointegration_chronology(
        _replace_pair(inputs, 0, daily=daily)
    )
    columns = [
        column
        for column in CHRONOLOGY_LEDGER_COLUMNS
        if column not in ("pair_order", "fold_order")
    ]
    before = baseline.fold_chronology.loc[
        baseline.fold_chronology["fold_id"].eq("wf_2022"), columns
    ].reset_index(drop=True)
    after = changed.fold_chronology.loc[
        changed.fold_chronology["fold_id"].eq("wf_2022"), columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_appending_later_folds_cannot_revise_earlier_rows(
    inputs: CointegrationInputs,
) -> None:
    frozen_folds = build_walk_forward_folds()
    early = analyze_causal_cointegration_chronology(
        inputs, folds=frozen_folds[:2]
    )
    complete = analyze_causal_cointegration_chronology(inputs, folds=frozen_folds)
    retained = complete.fold_chronology.loc[
        complete.fold_chronology["fold_order"].le(2)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(early.fold_chronology, retained, check_exact=True)


def test_maximum_information_timestamp_never_exceeds_train_end(results) -> None:
    maximum = pd.to_datetime(
        results.fold_chronology["maximum_timestamp_used_for_estimation"], utc=True
    )
    train_end_inclusive = pd.to_datetime(
        results.fold_chronology["train_end_inclusive"], utc=True
    )
    assert maximum.le(train_end_inclusive).all()
    references = results.fold_chronology.dropna(
        subset=["reference_maximum_information_timestamp"]
    )
    assert pd.to_datetime(
        references["reference_maximum_information_timestamp"], utc=True
    ).le(pd.to_datetime(references["train_end_inclusive"], utc=True)).all()


def test_inclusive_training_end_timestamp_is_legitimate() -> None:
    exclusive = pd.Timestamp("2022-01-01", tz="UTC")
    inclusive = inclusive_training_end_timestamp(exclusive)
    assert inclusive == pd.Timestamp(
        "2021-12-31 23:59:59.999999999", tz="UTC"
    )
    assert validate_maximum_information_timestamp(
        maximum_information_timestamp=inclusive,
        train_end_exclusive=exclusive,
    ) == inclusive
    with pytest.raises(CausalCointegrationChronologyError, match="exceeds"):
        validate_maximum_information_timestamp(
            maximum_information_timestamp=exclusive,
            train_end_exclusive=exclusive,
        )


def test_no_full_development_beta_enters_fold_gate_and_ex_post_is_labelled(
    results,
) -> None:
    assert "full_beta" not in results.fold_chronology.columns
    assert results.ex_post_beta_diagnostics["statistic_role"].eq(
        "ex_post_descriptive_only"
    ).all()
    for column in (
        "used_as_fold_reference",
        "used_in_stability_gate",
        "used_in_eligibility",
    ):
        assert not results.ex_post_beta_diagnostics[column].astype(bool).any()
    assert results.ex_post_beta_diagnostics[
        "maximum_information_timestamp"
    ].max() > results.pair_summaries["maximum_information_timestamp"].max()


def test_beta_sign_change_is_reported_and_fails_closed() -> None:
    result = compare_successive_betas(current_beta=-0.5, reference_beta=0.5)
    assert result.beta_sign_change
    assert result.stability_status == "beta_sign_change"
    assert not result.stability_pass
    assert result.relative_beta_drift == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("reference, expected_status"),
    [
        (0.0, "invalid_reference_near_zero"),
        (REFERENCE_BETA_ZERO_TOLERANCE / 2.0, "invalid_reference_near_zero"),
        (float("nan"), "invalid_reference_nonfinite"),
        (float("inf"), "invalid_reference_nonfinite"),
    ],
)
def test_zero_near_zero_nan_or_infinite_reference_fails_closed(
    reference: float,
    expected_status: str,
) -> None:
    result = compare_successive_betas(
        current_beta=1.0,
        reference_beta=reference,
    )
    assert np.isnan(result.relative_beta_drift)
    assert result.stability_status == expected_status
    assert not result.stability_pass


def test_singular_cointegrating_regression_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "y_log_price": np.arange(30, dtype="float64"),
            "x_log_price": np.ones(30, dtype="float64"),
        }
    )
    with pytest.raises(CausalCointegrationChronologyError, match="rank deficient"):
        estimate_cointegrating_regression(frame)


def test_unordered_or_duplicate_pair_timestamps_fail_closed(
    inputs: CointegrationInputs,
) -> None:
    daily = inputs.pair_inputs[0].daily_log_prices.copy(deep=True)
    with pytest.raises(CausalCointegrationChronologyError, match="ordered"):
        analyze_causal_cointegration_chronology(
            _replace_pair(inputs, 0, daily=daily.iloc[::-1])
        )
    duplicate = pd.concat([daily.iloc[[0]], daily], ignore_index=True).sort_values(
        "session_date", kind="stable"
    )
    with pytest.raises(CausalCointegrationChronologyError, match="unique"):
        analyze_causal_cointegration_chronology(
            _replace_pair(inputs, 0, daily=duplicate)
        )


def test_train_test_overlap_fails_closed(inputs: CointegrationInputs) -> None:
    frozen = build_walk_forward_folds()[0]
    overlapping = SimpleNamespace(
        fold_id=frozen.fold_id,
        train_start=frozen.train_start,
        train_end_exclusive=frozen.train_end_exclusive,
        test_start=frozen.train_end_exclusive - pd.Timedelta(days=1),
        test_end_exclusive=frozen.test_end_exclusive,
    )
    with pytest.raises(CausalCointegrationChronologyError, match="overlap"):
        analyze_causal_cointegration_chronology(inputs, folds=(overlapping,))


@pytest.mark.parametrize("location", ["daily", "intraday"])
def test_every_2026_timestamp_fails_closed(
    inputs: CointegrationInputs,
    location: str,
) -> None:
    item = inputs.pair_inputs[0]
    if location == "daily":
        daily = item.daily_log_prices.copy(deep=True)
        daily.loc[len(daily)] = {
            "session_date": pd.Timestamp("2026-01-01", tz="UTC"),
            "y_log_price": 5.0,
            "x_log_price": 4.0,
        }
        changed = _replace_pair(inputs, 0, daily=daily)
    else:
        intraday = item.intraday_log_prices.copy(deep=True)
        intraday.loc[len(intraday)] = {
            "timestamp": pd.Timestamp("2026-01-01 14:30", tz="UTC"),
            "session_date": pd.Timestamp("2026-01-01", tz="UTC"),
            "y_log_price": 5.0,
            "x_log_price": 4.0,
        }
        changed = _replace_pair(inputs, 0, intraday=intraday)
    with pytest.raises(CausalCointegrationChronologyError, match="2026"):
        analyze_causal_cointegration_chronology(changed)


def test_pair_orientation_remains_exact(inputs: CointegrationInputs) -> None:
    with pytest.raises(CausalCointegrationChronologyError, match="orientations"):
        analyze_causal_cointegration_chronology(
            _replace_pair(inputs, 0, y_symbol="QQQ", x_symbol="SPY")
        )


def test_pair_fold_and_schema_ordering_is_deterministic(results) -> None:
    assert tuple(results.fold_chronology.columns) == CHRONOLOGY_LEDGER_COLUMNS
    assert tuple(results.pair_summaries.columns) == PAIR_SUMMARY_COLUMNS
    assert tuple(results.ex_post_beta_diagnostics.columns) == EX_POST_BETA_COLUMNS
    assert list(
        zip(
            results.fold_chronology["pair_id"],
            results.fold_chronology["fold_id"],
            strict=True,
        )
    ) == [
        (pair_id, fold.fold_id)
        for pair_id in PAIR_IDS
        for fold in build_walk_forward_folds()
    ]
    assert tuple(results.pair_summaries["pair_id"]) == PAIR_IDS
    assert tuple(results.ex_post_beta_diagnostics["pair_id"]) == PAIR_IDS
    assert tuple(
        zip(
            results.fold_chronology["y_symbol"],
            results.fold_chronology["x_symbol"],
            strict=True,
        )
    ) == tuple(pair for pair in CANDIDATE_PAIRS for _ in range(4))
    assert results.fold_chronology["method_version"].eq(METHOD_VERSION).all()
    assert results.pair_summaries["causal_comparisons"].eq(3).all()
    assert METHOD_VERSION == "causal_cointegration_chronology_v1_1_svd_ols"
    assert results.fold_chronology["design_rank"].eq(2).all()
    assert np.isfinite(results.fold_chronology["design_condition_number"]).all()
    assert (
        results.fold_chronology["largest_singular_value"]
        >= results.fold_chronology["smallest_singular_value"]
    ).all()


def test_svd_refinement_preserves_v1_causal_drift_and_eligibility(results) -> None:
    np.testing.assert_allclose(
        results.fold_chronology["beta_estimate"],
        V1_BETA_ESTIMATES,
        rtol=5e-11,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        results.fold_chronology["relative_beta_drift"],
        V1_RELATIVE_DRIFTS,
        # SVD-level beta roundoff is propagated through a ratio here.  The
        # tolerance remains orders of magnitude below the frozen 0.25 gate.
        rtol=1e-7,
        atol=2e-11,
        equal_nan=True,
    )
    assert tuple(
        results.fold_chronology["fold_eligibility"].astype(bool)
    ) == V1_FOLD_ELIGIBILITY
    assert not results.pair_summaries["final_pair_eligibility"].astype(bool).any()


def test_svd_ols_matches_frozen_day14_ols_with_justified_tolerance(
    inputs: CointegrationInputs,
    results,
) -> None:
    for item in inputs.pair_inputs:
        daily = item.daily_log_prices.copy(deep=True)
        sessions = pd.to_datetime(daily["session_date"], utc=True)
        for fold in build_walk_forward_folds():
            train = daily.loc[
                sessions.ge(fold.train_start)
                & sessions.lt(fold.train_end_exclusive)
            ].reset_index(drop=True)
            causal = estimate_cointegrating_regression(train)
            alpha, beta, _, residuals = historical._fit_long_run(train)
            assert causal.alpha == pytest.approx(alpha, rel=5e-12, abs=5e-12)
            assert causal.beta == pytest.approx(beta, rel=5e-12, abs=5e-12)
            np.testing.assert_allclose(
                causal.residuals,
                residuals,
                rtol=5e-12,
                atol=5e-12,
            )
            frozen_adf = historical._adf_result(residuals, regression="n")
            row = results.fold_chronology.loc[
                results.fold_chronology["pair_id"].eq(item.pair_id)
                & results.fold_chronology["fold_id"].eq(fold.fold_id)
            ].iloc[0]
            assert row["residual_adf_statistic"] == pytest.approx(
                frozen_adf["adf_statistic"], rel=1e-11, abs=1e-11
            )
            assert row["residual_adf_p_value"] == pytest.approx(
                frozen_adf["p_value"], rel=1e-11, abs=1e-11
            )
            assert int(row["residual_adf_used_lag"]) == int(
                frozen_adf["used_lag"]
            )
            assert bool(row["residual_stationary"]) == bool(
                frozen_adf["reject_unit_root"]
            )


def test_inputs_are_not_mutated(inputs: CointegrationInputs) -> None:
    daily_before = [item.daily_log_prices.copy(deep=True) for item in inputs.pair_inputs]
    intraday_before = [
        item.intraday_log_prices.copy(deep=True) for item in inputs.pair_inputs
    ]
    analyze_causal_cointegration_chronology(inputs)
    for item, daily, intraday in zip(
        inputs.pair_inputs, daily_before, intraday_before, strict=True
    ):
        pd.testing.assert_frame_equal(item.daily_log_prices, daily, check_exact=True)
        pd.testing.assert_frame_equal(
            item.intraday_log_prices, intraday, check_exact=True
        )


def test_returned_tables_and_regression_residuals_are_defensive_copies(
    results,
) -> None:
    ledger = results.copy_fold_chronology()
    summaries = results.copy_pair_summaries()
    ex_post = results.copy_ex_post_beta_diagnostics()
    original_eligibility = bool(
        results.pair_summaries.loc[0, "final_pair_eligibility"]
    )
    ledger.loc[0, "beta_estimate"] = 999.0
    summaries.loc[0, "final_pair_eligibility"] = not original_eligibility
    ex_post.loc[0, "beta_ex_post"] = 999.0
    assert results.fold_chronology.loc[0, "beta_estimate"] != 999.0
    assert bool(
        results.pair_summaries.loc[0, "final_pair_eligibility"]
    ) == original_eligibility
    assert results.ex_post_beta_diagnostics.loc[0, "beta_ex_post"] != 999.0

    frame = pd.DataFrame(
        {"y_log_price": [1.0, 3.0, 5.0], "x_log_price": [0.0, 1.0, 2.0]}
    )
    estimate = estimate_cointegrating_regression(frame)
    copied = estimate.copy_residuals()
    copied.iloc[0] = 10.0
    assert estimate.residuals.iloc[0] != 10.0


def test_successive_expanding_estimates_are_reproducible(
    inputs: CointegrationInputs,
) -> None:
    first = analyze_causal_cointegration_chronology(inputs)
    second = analyze_causal_cointegration_chronology(inputs)
    pd.testing.assert_frame_equal(first.fold_chronology, second.fold_chronology)
    pd.testing.assert_frame_equal(first.pair_summaries, second.pair_summaries)
    pd.testing.assert_frame_equal(
        first.ex_post_beta_diagnostics, second.ex_post_beta_diagnostics
    )


def make_frozen_historical_bars() -> pd.DataFrame:
    """Build a stationary fixture that cannot activate the historical OU gate."""

    sessions = pd.bdate_range("2020-01-02", "2025-12-31")
    rows: list[dict[str, object]] = []
    for symbol_order, symbol in enumerate(("SPY", "QQQ", "IWM"), start=1):
        for session_order, session in enumerate(sessions):
            for bar_order, time in enumerate(("14:30:00", "14:45:00")):
                log_price = (
                    5.0
                    + symbol_order * 0.15
                    + 0.02 * np.sin(session_order / (3.0 + symbol_order))
                    + 0.01 * np.cos(session_order / (7.0 + symbol_order))
                    + bar_order * 0.0001
                )
                close = float(np.exp(log_price))
                rows.append(
                    {
                        "timestamp": pd.Timestamp(
                            f"{session.date()} {time}", tz="UTC"
                        ),
                        "symbol": symbol,
                        "open": close,
                        "high": close * 1.0001,
                        "low": close * 0.9999,
                        "close": close,
                        "volume": 1_000.0,
                        "trade_count": 100,
                        "vwap": close,
                        "source": "synthetic",
                        "feed": "test",
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def test_historical_day14_source_and_public_analysis_remain_unchanged() -> None:
    source_path = Path(historical.__file__)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        HISTORICAL_SOURCE_SHA256
    )
    bars = make_frozen_historical_bars()
    before = run_cointegration_feasibility(bars)
    analyze_causal_cointegration_chronology(build_cointegration_inputs(bars))
    after = run_cointegration_feasibility(bars)
    for name in (
        "pair_input_diagnostics",
        "series_integration_diagnostics",
        "cointegration_diagnostics",
        "fold_stability_diagnostics",
        "ou_diagnostics",
        "pair_eligibility",
    ):
        pd.testing.assert_frame_equal(
            getattr(before, f"copy_{name}")(),
            getattr(after, f"copy_{name}")(),
            check_exact=True,
        )
