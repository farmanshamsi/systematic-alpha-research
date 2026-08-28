"""Synthetic contracts for Day 31 slow OU inference robustness."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from systematic_alpha.analysis.day31_slow_ou_inference_robustness import (
    APPROVED_ARTIFACT_NAMES,
    BAR_EXECUTION_COLUMNS,
    BOOTSTRAP_BLOCK_LENGTHS,
    BOOTSTRAP_REPLICATIONS,
    COST_SENSITIVITY_COLUMNS,
    DAY28_TIMING_VERSION,
    DECLARED_DSR_TRIALS,
    EXPERIMENT_VERSION,
    HAC_LAGS_SENSITIVITY,
    LEAVE_ONE_YEAR_OUT_YEARS,
    PRIMARY_BOOTSTRAP_SEED,
    PRIMARY_CONFIGURATION_ID,
    PRIMARY_COST_BPS,
    PRIMARY_SERIES,
    SESSION_PATH_COLUMNS,
    TRANSACTION_COST_BPS,
    Day28Evidence,
    Day31SlowOuRobustnessError,
    FrozenSlowOuPath,
    aggregate_equal_weight_execution_path,
    apply_transaction_costs,
    build_day31_results,
    calculate_hac_statistics,
    calculate_leave_one_year_out,
    circular_block_bootstrap_intervals,
    circular_block_indices,
    sha256_file,
    write_day31_artifacts,
)
from systematic_alpha.analysis.reversion_inference import (
    ANNUALIZATION_FACTOR,
    CONFIGURATION_IDS,
    INFERENCE_COLUMNS,
    ReversionInferenceResults,
    _deflated_benchmark,
    _sharpe_probability,
)
from systematic_alpha.analysis.strategy_performance import calculate_performance_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bar_path() -> pd.DataFrame:
    sessions = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=month, day=2, tz="UTC")
            for year in LEAVE_ONE_YEAR_OUT_YEARS
            for month in range(1, 13)
        ],
        name="session_date",
    )
    records: list[dict[str, object]] = []
    symbol_scale = {"SPY": 1.0, "QQQ": 1.3, "IWM": 0.8}
    for session_order, session in enumerate(sessions):
        fold_id = f"wf_{session.year}"
        direction = -1.0 if session_order % 4 == 1 else 1.0
        for symbol in ("SPY", "QQQ", "IWM"):
            for bar, hour in enumerate((14, 15)):
                gross = (
                    direction
                    * symbol_scale[symbol]
                    * (0.002 + session_order * 0.00015)
                    if bar == 0
                    else direction * 0.0002
                )
                records.append(
                    {
                        "fold_id": fold_id,
                        "timestamp": session + pd.Timedelta(hours=hour),
                        "session_date": session,
                        "symbol": symbol,
                        "gross_return": gross,
                        "turnover": 1.0 if bar == 0 else float(session_order % 3 == 0),
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["timestamp", "symbol"], kind="stable"
    ).reset_index(drop=True).loc[:, BAR_EXECUTION_COLUMNS]


def _inference_row(
    configuration_id: str,
    annualized_sharpe: float,
    *,
    primary: pd.Series,
    dsr_benchmark: float,
    primary_interval: dict[str, float | int],
) -> dict[str, object]:
    per_period = annualized_sharpe / math.sqrt(ANNUALIZATION_FACTOR)
    sample_skewness = float(skew(primary, bias=False))
    sample_kurtosis = float(kurtosis(primary, fisher=False, bias=False))
    is_primary = configuration_id == PRIMARY_CONFIGURATION_ID
    naive_t_statistic = float(
        primary.mean() / (primary.std(ddof=1) / math.sqrt(len(primary)))
    )
    return {
        "configuration_id": configuration_id,
        "series": PRIMARY_SERIES,
        "observations": len(primary),
        "mean_session_return": float(primary.mean()) if is_primary else 0.0,
        "naive_t_statistic": naive_t_statistic if is_primary else 0.0,
        "hac_lags": 5,
        "hac_t_statistic": (
            calculate_hac_statistics(primary, lag=5)["hac_t_statistic"]
            if is_primary
            else 0.0
        ),
        "annualized_sharpe_ratio": annualized_sharpe,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_block_length": 5,
        "bootstrap_mean_ci_lower": primary_interval["mean_ci_lower"],
        "bootstrap_mean_ci_upper": primary_interval["mean_ci_upper"],
        "bootstrap_sharpe_ci_lower": primary_interval["sharpe_ci_lower"],
        "bootstrap_sharpe_ci_upper": primary_interval["sharpe_ci_upper"],
        "information_coefficient": 0.0,
        "information_coefficient_observations": 10,
        "sample_skewness": sample_skewness,
        "sample_kurtosis": sample_kurtosis,
        "probabilistic_sharpe_probability": _sharpe_probability(
            per_period_sharpe=per_period,
            benchmark=0.0,
            observations=len(primary),
            sample_skewness=sample_skewness,
            sample_kurtosis=sample_kurtosis,
        ),
        "deflated_sharpe_benchmark": dsr_benchmark,
        "deflated_sharpe_probability": _sharpe_probability(
            per_period_sharpe=per_period,
            benchmark=dsr_benchmark,
            observations=len(primary),
            sample_skewness=sample_skewness,
            sample_kurtosis=sample_kurtosis,
        ),
        "declared_trials": DECLARED_DSR_TRIALS,
    }


@pytest.fixture(scope="module")
def synthetic_components(tmp_path_factory):
    bar_path = make_bar_path()
    session = aggregate_equal_weight_execution_path(
        bar_path, cost_bps_per_turnover=PRIMARY_COST_BPS
    ).rename(columns={"net_return": "net_return_1bp"})
    frozen_path = FrozenSlowOuPath(
        bar_execution_path=bar_path,
        session_return_path=session.loc[:, SESSION_PATH_COLUMNS],
    )
    primary = pd.Series(
        session["net_return_1bp"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(session["session_date"], name="session_date"),
    )
    annualized_sharpe = float(
        calculate_performance_metrics(primary).sharpe_ratio
    )
    trial_annualized = np.array([-0.4, 0.1, annualized_sharpe], dtype="float64")
    dsr_benchmark = _deflated_benchmark(
        trial_annualized / math.sqrt(ANNUALIZATION_FACTOR)
    )
    interval = circular_block_bootstrap_intervals(
        primary,
        block_length=5,
        replications=BOOTSTRAP_REPLICATIONS,
        seed=PRIMARY_BOOTSTRAP_SEED,
    )
    inference = pd.DataFrame.from_records(
        [
            _inference_row(
                configuration_id,
                float(value),
                primary=primary,
                dsr_benchmark=dsr_benchmark,
                primary_interval=interval,
            )
            for configuration_id, value in zip(
                CONFIGURATION_IDS, trial_annualized, strict=True
            )
        ]
    ).loc[:, INFERENCE_COLUMNS]
    panel = session[["fold_id", "session_date"]].copy(deep=True)
    panel.insert(0, "configuration_id", PRIMARY_CONFIGURATION_ID)
    panel["equal_weight"] = primary.to_numpy(dtype="float64")
    reversion = ReversionInferenceResults(
        signal_diagnostics=pd.DataFrame(),
        fold_performance=pd.DataFrame(),
        aggregate_performance=pd.DataFrame(),
        inference_results=inference,
        cost_sensitivity=pd.DataFrame(),
        session_return_panel=panel,
    )
    gross = aggregate_equal_weight_execution_path(
        bar_path, cost_bps_per_turnover=0.0
    )["net_return"]
    gross_metrics = calculate_performance_metrics(gross)
    primary_metrics = calculate_performance_metrics(primary)
    total_turnover = float(bar_path["turnover"].sum() / 3.0)
    aggregate = pd.DataFrame.from_records(
        [
            {
                "timing_convention": DAY28_TIMING_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "test_sessions": len(primary),
                "start_session": primary.index.min().date().isoformat(),
                "end_session": primary.index.max().date().isoformat(),
                "cumulative_gross_return": gross_metrics.cumulative_return,
                "cumulative_net_return_1bp": primary_metrics.cumulative_return,
                "annualized_volatility_1bp": primary_metrics.annualized_volatility,
                "sharpe_ratio_1bp": primary_metrics.sharpe_ratio,
                "maximum_drawdown_1bp": primary_metrics.max_drawdown,
                "turnover": total_turnover,
                "nonzero_net_sessions_1bp": int(primary.ne(0.0).sum()),
            }
        ]
    )
    cost_rows = []
    for cost in TRANSACTION_COST_BPS:
        returns = aggregate_equal_weight_execution_path(
            bar_path, cost_bps_per_turnover=cost
        )["net_return"]
        metrics = calculate_performance_metrics(returns)
        cost_rows.append(
            {
                "timing_convention": DAY28_TIMING_VERSION,
                "configuration_id": PRIMARY_CONFIGURATION_ID,
                "series": PRIMARY_SERIES,
                "cost_bps_per_turnover": cost,
                "test_sessions": len(returns),
                "cumulative_return": metrics.cumulative_return,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "maximum_drawdown": metrics.max_drawdown,
            }
        )
    saved_inference = inference.copy(deep=True)
    saved_inference.insert(0, "timing_convention", DAY28_TIMING_VERSION)
    comparator = tmp_path_factory.mktemp("day31-comparator") / "sentinel.csv"
    comparator.write_text("immutable\n", encoding="utf-8")
    evidence = Day28Evidence(
        aggregate=aggregate,
        cost_sensitivity=pd.DataFrame.from_records(cost_rows),
        inference=saved_inference,
        manifest={},
        source_hashes={comparator.resolve().as_posix(): sha256_file(comparator)},
    )
    return frozen_path, reversion, evidence, primary


@pytest.fixture(scope="module")
def assembled(synthetic_components):
    frozen_path, reversion, evidence, _ = synthetic_components
    return build_day31_results(
        frozen_path=frozen_path,
        reversion_results=reversion,
        day28_evidence=evidence,
        data_audit={
            "session_min": "2020-01-02",
            "session_max": "2025-12-31",
            "contains_2026": False,
        },
        source_dataset_path="development.parquet",
        source_sha256="0" * 64,
        generation_timestamp="2026-08-15T00:00:00+00:00",
    )


def test_experiment_constants_are_frozen() -> None:
    assert EXPERIMENT_VERSION == "day31_slow_ou_inference_robustness_v1"
    assert DAY28_TIMING_VERSION == "corrected_next_open_overnight_flat"
    assert TRANSACTION_COST_BPS == (0.0, 1.0, 2.0, 5.0)
    assert HAC_LAGS_SENSITIVITY == (1, 5, 10, 20)
    assert BOOTSTRAP_BLOCK_LENGTHS == (5, 10, 20, 40)
    assert DECLARED_DSR_TRIALS == 3


def test_transaction_cost_is_applied_to_actual_turnover() -> None:
    gross = np.array([0.01, -0.02, 0.03])
    turnover = np.array([0.0, 2.0, 0.5])
    observed = apply_transaction_costs(
        gross, turnover, cost_bps_per_turnover=5.0
    )
    np.testing.assert_allclose(observed, gross - turnover * 5.0 / 10_000.0)
    assert observed[0] == gross[0]


def test_bartlett_hac_known_answer_and_lag_zero_naive_relationship() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.04])
    demeaned = values - values.mean()
    gamma0 = float(np.dot(demeaned, demeaned) / len(values))
    gamma1 = float(np.dot(demeaned[1:], demeaned[:-1]) / len(values))
    observed = calculate_hac_statistics(values, lag=1)
    assert observed["long_run_variance"] == pytest.approx(gamma0 + gamma1)
    assert observed["hac_standard_error"] == pytest.approx(
        math.sqrt((gamma0 + gamma1) / len(values))
    )
    lag_zero = calculate_hac_statistics(values, lag=0)
    naive_t = float(values.mean() / (values.std(ddof=1) / math.sqrt(len(values))))
    assert lag_zero["hac_t_statistic"] == pytest.approx(
        naive_t * math.sqrt(len(values) / (len(values) - 1.0))
    )


def test_zero_hac_long_run_variance_fails_closed() -> None:
    with pytest.raises(Day31SlowOuRobustnessError, match="long-run variance"):
        calculate_hac_statistics([0.01, 0.01, 0.01], lag=1)


def test_circular_block_wraparound_and_deterministic_bootstrap() -> None:
    indices = circular_block_indices(
        observations=6, block_length=4, starts=np.array([4, 1])
    )
    np.testing.assert_array_equal(indices, np.array([4, 5, 0, 1, 1, 2]))
    values = np.array([0.01, -0.005, 0.02, -0.01, 0.003, 0.004])
    first = circular_block_bootstrap_intervals(
        values, block_length=3, replications=100, seed=31
    )
    second = circular_block_bootstrap_intervals(
        values, block_length=3, replications=100, seed=31
    )
    assert first == second
    assert first["mean_ci_lower"] <= first["mean_ci_upper"]
    assert first["sharpe_ci_lower"] <= first["sharpe_ci_upper"]


def test_leave_one_year_out_membership_and_annualization(synthetic_components) -> None:
    _, _, _, primary = synthetic_components
    result = calculate_leave_one_year_out(primary)
    assert tuple(result["excluded_year"]) == LEAVE_ONE_YEAR_OUT_YEARS
    for row in result.itertuples(index=False):
        assert str(row.excluded_year) not in row.included_years.split("|")
        sample = primary.loc[primary.index.year != row.excluded_year]
        metrics = calculate_performance_metrics(
            sample.reset_index(drop=True), annualization_factor=252.0
        )
        assert row.observations == len(sample)
        assert row.annualized_volatility == pytest.approx(
            metrics.annualized_volatility
        )


def test_primary_reproduction_and_psr_dsr_are_exact(assembled) -> None:
    assert assembled.primary_day28_reproduction["within_tolerance"].all()
    assert assembled.primary_day28_reproduction["absolute_difference"].max() < 1e-12
    disclosure = assembled.psr_dsr_disclosure.iloc[0]
    saved = assembled.primary_day28_reproduction.set_index("metric")
    assert disclosure["probabilistic_sharpe_probability"] == pytest.approx(
        saved.loc["probabilistic_sharpe_probability", "day28_saved_value"]
    )
    assert disclosure["deflated_sharpe_probability"] == pytest.approx(
        saved.loc["deflated_sharpe_probability", "day28_saved_value"]
    )
    assert disclosure["declared_trials"] == 3
    assert disclosure["globally_corrected_dsr_claimed"] is False or not bool(
        disclosure["globally_corrected_dsr_claimed"]
    )


def test_sensitivity_tables_are_predeclared_and_deterministically_ordered(
    assembled,
) -> None:
    assert tuple(assembled.transaction_cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert tuple(assembled.transaction_cost_sensitivity["cost_bps_per_turnover"]) == (
        0.0,
        1.0,
        2.0,
        5.0,
    )
    assert tuple(assembled.hac_lag_sensitivity["hac_lag"]) == HAC_LAGS_SENSITIVITY
    assert tuple(assembled.block_bootstrap_sensitivity["block_length"]) == (
        BOOTSTRAP_BLOCK_LENGTHS
    )
    assert tuple(assembled.leave_one_year_out["excluded_year"]) == (
        LEAVE_ONE_YEAR_OUT_YEARS
    )


def test_future_mutation_does_not_change_earlier_sessions() -> None:
    original = make_bar_path()
    mutated = original.copy(deep=True)
    mutated.loc[mutated.index[-1], "gross_return"] += 0.1
    before = aggregate_equal_weight_execution_path(
        original, cost_bps_per_turnover=1.0
    )
    after = aggregate_equal_weight_execution_path(
        mutated, cost_bps_per_turnover=1.0
    )
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1], check_exact=True)


def test_2026_rejection_and_input_immutability() -> None:
    source = make_bar_path()
    before = source.copy(deep=True)
    aggregate_equal_weight_execution_path(source, cost_bps_per_turnover=1.0)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    future = source.copy(deep=True)
    future.loc[future.index[-1], "timestamp"] = pd.Timestamp(
        "2026-01-02 15:00", tz="UTC"
    )
    future = future.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    with pytest.raises(Day31SlowOuRobustnessError, match="2026"):
        aggregate_equal_weight_execution_path(
            future, cost_bps_per_turnover=1.0
        )


def test_result_frames_are_defensive_copies(assembled) -> None:
    copy = assembled.transaction_cost_sensitivity.copy(deep=True)
    copy.loc[0, "cumulative_net_return"] = 999.0
    assert assembled.transaction_cost_sensitivity.loc[0, "cumulative_net_return"] != 999.0


def test_artifacts_are_deterministic_hashed_and_allow_listed(
    tmp_path: Path,
    assembled,
) -> None:
    first = tmp_path / "first/day31_slow_ou_inference_robustness"
    second = tmp_path / "second/day31_slow_ou_inference_robustness"
    first_paths = write_day31_artifacts(assembled, first)
    second_paths = write_day31_artifacts(assembled, second)
    assert tuple(path.name for path in first_paths) == APPROVED_ARTIFACT_NAMES
    for filename in APPROVED_ARTIFACT_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
        assert Path(filename).suffix in {".csv", ".json"}
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_version"] == EXPERIMENT_VERSION
    assert manifest["artifact_file_list"] == list(APPROVED_ARTIFACT_NAMES)
    for artifact in manifest["artifacts"]:
        assert artifact["sha256"] == _sha256(first / artifact["filename"])
    assert manifest["report_generated"] is False
    assert manifest["notebook_created"] is False
    assert manifest["chart_created"] is False
    with pytest.raises(FileExistsError):
        write_day31_artifacts(assembled, first)
