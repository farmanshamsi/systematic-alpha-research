"""Contract tests for Day 11 trend-family walk-forward evaluation."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.trend_family_walk_forward as walk_forward
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
)
from systematic_alpha.analysis.trend_family_robustness import (
    CONFIGURATION_IDS as DAY10_CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS as DAY10_EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS as DAY10_TREND_RATIO_PARAMETERS,
)


EXPECTED_FOLD_COLUMNS = (
    "strategy",
    "symbol",
    "frequency",
    "fold_id",
    "configuration_id",
    "train_start_timestamp",
    "train_end_timestamp",
    "test_start_timestamp",
    "test_end_timestamp",
    "train_sessions",
    "test_sessions",
    "train_observations",
    "test_observations",
    "annualization_factor",
    "purge_sessions",
    "embargo_sessions",
    "indicator_history_observations",
    "initial_test_position",
    "initial_test_turnover",
    "warmup_observations",
    "active_observations",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "average_exposure",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "trade_count",
)

EXPECTED_AGGREGATE_COLUMNS = (
    "strategy",
    "symbol",
    "frequency",
    "configuration_id",
    "folds",
    "test_start_timestamp",
    "test_end_timestamp",
    "test_sessions",
    "test_observations",
    "annualization_factor",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "turnover",
    "average_exposure",
    "long_exposure",
    "short_exposure",
    "flat_exposure",
    "trade_count",
)

SESSION_DATES = (
    "2020-01-02",
    "2021-06-15",
    "2022-06-15",
    "2023-06-15",
    "2024-06-17",
    "2025-12-31",
)

SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
)


def make_session(
    *,
    symbol: str,
    session_date: str,
    bar_count: int = 26,
    symbol_offset: float = 0.0,
) -> pd.DataFrame:
    """Create one complete deterministic canonical session."""

    timestamps = pd.date_range(
        f"{session_date} 14:30:00+00:00",
        periods=bar_count,
        freq="15min",
    )
    sequence = np.arange(
        bar_count,
        dtype="float64",
    )
    day_offset = float(
        pd.Timestamp(session_date).year
        - 2020
    )
    close = (
        100.0
        + symbol_offset
        + 2.0 * day_offset
        + 0.20 * sequence
        + 0.50 * np.sin(sequence / 2.0)
    )
    open_price = close - 0.10

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "session_date": session_date,
            "open": open_price,
            "high": close + 0.25,
            "low": open_price - 0.25,
            "close": close,
            "volume": 1_000.0 + sequence,
            "trade_count": 100 + sequence.astype(int),
            "vwap": (
                open_price + close
            ) / 2.0,
            "source": "test",
            "feed": "sip",
        }
    )


def make_development_bars(
    *,
    session_dates: tuple[str, ...] = (
        SESSION_DATES
    ),
    symbols: tuple[str, ...] = SYMBOLS,
    extra_sessions: tuple[
        tuple[str, int],
        ...,
    ] = (),
) -> pd.DataFrame:
    """Create compact complete multi-symbol development bars."""

    offsets = {
        "SPY": 0.0,
        "QQQ": 100.0,
        "IWM": 200.0,
    }
    sessions = (
        *((session_date, 26) for session_date in session_dates),
        *extra_sessions,
    )
    frames = [
        make_session(
            symbol=symbol,
            session_date=session_date,
            bar_count=bar_count,
            symbol_offset=offsets.get(
                symbol,
                300.0,
            ),
        )
        for symbol in symbols
        for session_date, bar_count in sessions
    ]

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "symbol",
                "session_date",
                "timestamp",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def expected_fold_boundaries() -> list[
    tuple[
        str,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:
    """Return the four frozen expanding fold boundaries."""

    return [
        (
            f"wf_{test_year}",
            pd.Timestamp(
                "2020-01-02",
                tz="UTC",
            ),
            pd.Timestamp(
                f"{test_year}-01-01",
                tz="UTC",
            ),
            pd.Timestamp(
                f"{test_year}-01-01",
                tz="UTC",
            ),
            pd.Timestamp(
                f"{test_year + 1}-01-01",
                tz="UTC",
            ),
        )
        for test_year in range(2022, 2026)
    ]


def expected_fold_keys() -> list[
    tuple[str, str]
]:
    """Return deterministic strategy/fold result keys."""

    return [
        (
            strategy,
            f"wf_{test_year}",
        )
        for strategy in (
            "trend_ratio",
            "ema_macd",
        )
        for test_year in range(2022, 2026)
    ]


def test_builds_exact_four_calendar_year_folds() -> None:
    folds = (
        walk_forward.build_walk_forward_folds()
    )

    assert len(folds) == 4
    assert all(
        isinstance(
            fold,
            walk_forward.WalkForwardFold,
        )
        for fold in folds
    )
    assert [
        (
            fold.fold_id,
            fold.train_start,
            fold.train_end_exclusive,
            fold.test_start,
            fold.test_end_exclusive,
        )
        for fold in folds
    ] == expected_fold_boundaries()


def test_training_windows_expand_from_one_fixed_origin() -> None:
    folds = (
        walk_forward.build_walk_forward_folds()
    )

    assert {
        fold.train_start
        for fold in folds
    } == {
        pd.Timestamp(
            "2020-01-02",
            tz="UTC",
        )
    }
    assert [
        fold.train_end_exclusive.year
        for fold in folds
    ] == [
        2022,
        2023,
        2024,
        2025,
    ]
    assert [
        fold.test_end_exclusive
        - fold.train_start
        for fold in folds
    ] == sorted(
        (
            fold.test_end_exclusive
            - fold.train_start
            for fold in folds
        )
    )


def test_fold_boundaries_are_non_overlapping_and_deterministic() -> None:
    first = (
        walk_forward.build_walk_forward_folds()
    )
    second = (
        walk_forward.build_walk_forward_folds()
    )

    assert first == second

    for index, fold in enumerate(first):
        assert (
            fold.train_end_exclusive
            == fold.test_start
        )
        assert (
            fold.train_start
            < fold.train_end_exclusive
            <= fold.test_start
            < fold.test_end_exclusive
        )

        if index:
            assert (
                first[index - 1]
                .test_end_exclusive
                == fold.test_start
            )


def test_scope_and_frozen_baselines_are_locked() -> None:
    assert walk_forward.WALK_FORWARD_STRATEGIES == (
        "trend_ratio",
        "ema_macd",
    )
    assert walk_forward.WALK_FORWARD_SYMBOL == "SPY"
    assert walk_forward.WALK_FORWARD_FREQUENCY == (
        "15min"
    )
    assert (
        walk_forward.TREND_RATIO_PARAMETERS
        == DAY10_TREND_RATIO_PARAMETERS
    )
    assert (
        walk_forward.EMA_MACD_PARAMETERS
        == DAY10_EMA_MACD_PARAMETERS
    )
    assert (
        walk_forward.CONFIGURATION_IDS
        == DAY10_CONFIGURATION_IDS
    )
    assert all(
        fold.purge_sessions == 0
        and fold.embargo_sessions == 0
        for fold in (
            walk_forward.build_walk_forward_folds()
        )
    )


def test_result_schemas_exclude_selection_fields() -> None:
    assert (
        walk_forward.FOLD_RESULT_COLUMNS
        == EXPECTED_FOLD_COLUMNS
    )
    assert (
        walk_forward.AGGREGATE_RESULT_COLUMNS
        == EXPECTED_AGGREGATE_COLUMNS
    )

    forbidden_tokens = (
        "rank",
        "winner",
        "selected",
        "optimal",
        "best",
        "day07",
        "day09",
        "sensitivity",
    )

    for column in (
        *walk_forward.FOLD_RESULT_COLUMNS,
        *walk_forward.AGGREGATE_RESULT_COLUMNS,
    ):
        assert not any(
            token in column.lower()
            for token in forbidden_tokens
        )


def test_complete_orchestration_has_stable_schemas_and_order() -> None:
    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )

    assert isinstance(
        result,
        walk_forward.TrendFamilyWalkForwardResults,
    )
    assert tuple(
        result.fold_results.columns
    ) == EXPECTED_FOLD_COLUMNS
    assert tuple(
        result.aggregate_results.columns
    ) == EXPECTED_AGGREGATE_COLUMNS
    assert len(result.fold_results) == 8
    assert len(result.aggregate_results) == 2
    assert list(
        result.fold_results[
            [
                "strategy",
                "fold_id",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ) == expected_fold_keys()
    assert result.aggregate_results[
        "strategy"
    ].tolist() == [
        "trend_ratio",
        "ema_macd",
    ]
    assert result.fold_results[
        "symbol"
    ].eq("SPY").all()
    assert result.fold_results[
        "frequency"
    ].eq("15min").all()


def test_fold_results_reconcile_sessions_and_observations() -> None:
    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )
    folds = result.fold_results

    expected_train_sessions = {
        "wf_2022": 2,
        "wf_2023": 3,
        "wf_2024": 4,
        "wf_2025": 5,
    }

    assert folds["test_sessions"].eq(1).all()
    assert folds[
        "test_observations"
    ].eq(26).all()
    assert folds[
        "train_sessions"
    ].eq(
        folds["fold_id"].map(
            expected_train_sessions
        )
    ).all()
    assert folds[
        "train_observations"
    ].eq(
        folds["train_sessions"] * 26
    ).all()
    assert folds[
        "indicator_history_observations"
    ].eq(
        folds["train_observations"]
    ).all()
    assert folds[
        "purge_sessions"
    ].eq(0).all()
    assert folds[
        "embargo_sessions"
    ].eq(0).all()


def test_training_history_warms_indicators_but_not_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_lengths: dict[
        str,
        list[int],
    ] = {
        "trend_ratio": [],
        "ema_macd": [],
    }
    metric_lengths: list[int] = []
    metric_factors: list[float] = []
    original_trend = (
        walk_forward.build_trend_ratio_strategy
    )
    original_ema = (
        walk_forward.build_ema_macd_strategy
    )
    original_metrics = (
        walk_forward.calculate_performance_metrics
    )

    def record_trend(
        frame: pd.DataFrame,
        *,
        parameters,
    ):
        strategy_lengths[
            "trend_ratio"
        ].append(len(frame))

        return original_trend(
            frame,
            parameters=parameters,
        )

    def record_ema(
        frame: pd.DataFrame,
        *,
        parameters,
    ):
        strategy_lengths[
            "ema_macd"
        ].append(len(frame))

        return original_ema(
            frame,
            parameters=parameters,
        )

    def record_metrics(
        returns,
        *,
        return_column=None,
        annualization_factor,
    ) -> PerformanceMetrics:
        metric_lengths.append(len(returns))
        metric_factors.append(
            float(annualization_factor)
        )

        return original_metrics(
            returns,
            return_column=return_column,
            annualization_factor=(
                annualization_factor
            ),
        )

    monkeypatch.setattr(
        walk_forward,
        "build_trend_ratio_strategy",
        record_trend,
    )
    monkeypatch.setattr(
        walk_forward,
        "build_ema_macd_strategy",
        record_ema,
    )
    monkeypatch.setattr(
        walk_forward,
        "calculate_performance_metrics",
        record_metrics,
    )

    walk_forward.run_trend_family_walk_forward(
        make_development_bars()
    )

    assert strategy_lengths == {
        "trend_ratio": [
            78,
            104,
            130,
            156,
        ],
        "ema_macd": [
            78,
            104,
            130,
            156,
        ],
    }
    assert metric_lengths.count(26) == 8
    assert metric_lengths.count(104) == 2
    assert len(metric_lengths) == 10
    assert metric_factors == [
        252.0 * 26.0
    ] * 10


def _always_long_bundle(
    frame: pd.DataFrame,
    *,
    parameters,
) -> SimpleNamespace:
    """Return causal observations that carry a long position."""

    del parameters

    observations = frame.copy(deep=True)
    signal = pd.Series(
        1,
        index=observations.index,
        dtype="int8",
    )
    position = signal.shift(
        1,
        fill_value=0,
    ).astype("int8")
    turnover = position.diff().abs().fillna(
        position.abs()
    )
    raw_return = (
        observations[
            "close_to_close_simple_return"
        ]
        .fillna(0.0)
        .astype("float64")
    )
    observations["signal"] = signal
    observations["position"] = position
    observations["position_eligible"] = True
    observations["turnover"] = turnover
    observations[
        "gross_strategy_return"
    ] = position.astype(float) * raw_return
    observations["transaction_cost"] = (
        turnover.astype(float) / 10_000.0
    )
    observations["net_strategy_return"] = (
        observations[
            "gross_strategy_return"
        ]
        - observations["transaction_cost"]
    )

    return SimpleNamespace(
        observations=observations
    )


def test_positions_and_execution_reset_at_every_test_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured_returns: list[pd.Series] = []
    original_metrics = (
        walk_forward.calculate_performance_metrics
    )

    def record_metrics(
        returns,
        *,
        return_column=None,
        annualization_factor,
    ) -> PerformanceMetrics:
        if isinstance(returns, pd.DataFrame):
            measured = returns[
                return_column
            ].copy()
        else:
            measured = returns.copy()

        measured_returns.append(
            measured.reset_index(drop=True)
        )

        return original_metrics(
            returns,
            return_column=return_column,
            annualization_factor=(
                annualization_factor
            ),
        )

    monkeypatch.setattr(
        walk_forward,
        "build_trend_ratio_strategy",
        _always_long_bundle,
    )
    monkeypatch.setattr(
        walk_forward,
        "build_ema_macd_strategy",
        _always_long_bundle,
    )
    monkeypatch.setattr(
        walk_forward,
        "calculate_performance_metrics",
        record_metrics,
    )

    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )

    assert result.fold_results[
        "initial_test_position"
    ].eq(0).all()
    assert result.fold_results[
        "initial_test_turnover"
    ].eq(0.0).all()

    fold_returns = [
        values
        for values in measured_returns
        if len(values) == 26
    ]
    aggregate_returns = [
        values
        for values in measured_returns
        if len(values) == 104
    ]

    assert len(fold_returns) == 8
    assert all(
        values.iloc[0] == pytest.approx(0.0)
        for values in fold_returns
    )
    assert len(aggregate_returns) == 2

    for values in aggregate_returns:
        assert values.iloc[
            [0, 26, 52, 78]
        ].eq(0.0).all()


def test_annualization_uses_observed_test_sessions() -> None:
    bars = make_development_bars(
        extra_sessions=(
            (
                "2022-11-25",
                14,
            ),
        )
    )
    result = (
        walk_forward.run_trend_family_walk_forward(
            bars
        )
    )
    fold_results = result.fold_results
    fold_2022 = fold_results.loc[
        fold_results["fold_id"].eq(
            "wf_2022"
        )
    ]

    assert fold_2022[
        "test_sessions"
    ].eq(2).all()
    assert fold_2022[
        "test_observations"
    ].eq(40).all()
    assert fold_2022[
        "annualization_factor"
    ].eq(
        252.0 * 40.0 / 2.0
    ).all()

    for row in fold_results.itertuples(
        index=False
    ):
        assert (
            row.annualization_factor
            == pytest.approx(
                252.0
                * row.test_observations
                / row.test_sessions
            )
        )

    aggregate = result.aggregate_results

    assert aggregate[
        "annualization_factor"
    ].eq(
        252.0 * 118.0 / 5.0
    ).all()


def test_out_of_period_and_locked_2026_rows_are_rejected() -> None:
    bars = pd.concat(
        [
            make_development_bars(),
            *(
                make_session(
                    symbol=symbol,
                    session_date="2026-01-02",
                    symbol_offset=(
                        100.0 * index
                    ),
                )
                for index, symbol in enumerate(
                    SYMBOLS
                )
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        walk_forward.TrendFamilyWalkForwardError,
        match=(
            "development period|locked"
        ),
    ):
        walk_forward.run_trend_family_walk_forward(
            bars
        )


@pytest.mark.parametrize(
    "invalid_bars",
    (
        lambda bars: bars.loc[
            pd.to_datetime(
                bars["session_date"]
            ).dt.year.ne(2023)
        ].reset_index(drop=True),
        lambda bars: bars.loc[
            bars["session_date"].ne(
                "2020-01-02"
            )
        ].reset_index(drop=True),
        lambda bars: bars.loc[
            bars["session_date"].ne(
                "2025-12-31"
            )
        ].reset_index(drop=True),
    ),
)
def test_incomplete_development_coverage_is_rejected(
    invalid_bars,
) -> None:
    bars = invalid_bars(
        make_development_bars()
    )

    with pytest.raises(
        walk_forward.TrendFamilyWalkForwardError,
        match="complete.*development|coverage",
    ):
        walk_forward.run_trend_family_walk_forward(
            bars
        )


def test_whole_session_boundaries_are_enforced() -> None:
    bars = make_development_bars()
    selected = (
        bars["symbol"].eq("SPY")
        & bars["session_date"].eq(
            "2022-06-15"
        )
    )
    final_index = bars.loc[
        selected
    ].index[-1]
    bars.loc[
        final_index,
        "timestamp",
    ] = pd.Timestamp(
        "2022-06-16 14:30:00+00:00"
    )

    with pytest.raises(
        walk_forward.TrendFamilyWalkForwardError,
        match="session|whole",
    ):
        walk_forward.run_trend_family_walk_forward(
            bars
        )


def test_missing_or_unexpected_symbols_are_rejected() -> None:
    missing = make_development_bars(
        symbols=(
            "SPY",
            "QQQ",
        )
    )
    unexpected = make_development_bars(
        symbols=(
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
        )
    )

    for bars in (
        missing,
        unexpected,
    ):
        with pytest.raises(
            walk_forward.TrendFamilyWalkForwardError,
            match="symbols",
        ):
            (
                walk_forward
                .run_trend_family_walk_forward(
                    bars
                )
            )


def test_symbol_isolation_input_immutability_and_determinism() -> None:
    bars = make_development_bars()
    original = copy.deepcopy(bars)
    altered = bars.copy(deep=True)
    non_spy = altered["symbol"].ne(
        "SPY"
    )
    altered.loc[
        non_spy,
        [
            "open",
            "high",
            "low",
            "close",
            "vwap",
        ],
    ] *= 10.0

    first = (
        walk_forward.run_trend_family_walk_forward(
            bars
        )
    )
    repeated = (
        walk_forward.run_trend_family_walk_forward(
            bars
        )
    )
    isolated = (
        walk_forward.run_trend_family_walk_forward(
            altered
        )
    )

    pd.testing.assert_frame_equal(
        bars,
        original,
    )
    pd.testing.assert_frame_equal(
        first.fold_results,
        repeated.fold_results,
    )
    pd.testing.assert_frame_equal(
        first.aggregate_results,
        repeated.aggregate_results,
    )
    pd.testing.assert_frame_equal(
        first.fold_results,
        isolated.fold_results,
    )
    pd.testing.assert_frame_equal(
        first.aggregate_results,
        isolated.aggregate_results,
    )


def _flat_bundle(
    frame: pd.DataFrame,
    *,
    parameters,
) -> SimpleNamespace:
    """Return eligible but permanently flat strategy observations."""

    del parameters

    observations = frame.copy(deep=True)
    observations["signal"] = 0
    observations["position"] = 0
    observations["position_eligible"] = True
    observations["turnover"] = 0.0
    observations[
        "gross_strategy_return"
    ] = 0.0
    observations["transaction_cost"] = 0.0
    observations["net_strategy_return"] = 0.0

    return SimpleNamespace(
        observations=observations
    )


def test_undefined_sharpe_and_drawdown_follow_shared_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        walk_forward,
        "build_trend_ratio_strategy",
        _flat_bundle,
    )
    monkeypatch.setattr(
        walk_forward,
        "build_ema_macd_strategy",
        _flat_bundle,
    )

    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )

    for table in (
        result.fold_results,
        result.aggregate_results,
    ):
        assert table[
            "annualized_return"
        ].eq(0.0).all()
        assert table[
            "annualized_volatility"
        ].eq(0.0).all()
        assert table[
            "sharpe_ratio"
        ].isna().all()
        assert table[
            "maximum_drawdown"
        ].eq(0.0).all()


def test_negative_performance_is_retained_as_valid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_lengths: list[int] = []

    def negative_metrics(
        returns,
        *,
        return_column=None,
        annualization_factor,
    ) -> PerformanceMetrics:
        del return_column
        del annualization_factor

        observed_lengths.append(len(returns))

        return PerformanceMetrics(
            observations=len(returns),
            cumulative_return=-0.20,
            annualized_return=-0.10,
            annualized_volatility=0.25,
            sharpe_ratio=-0.75,
            max_drawdown=-0.30,
        )

    monkeypatch.setattr(
        walk_forward,
        "calculate_performance_metrics",
        negative_metrics,
    )

    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )

    assert len(result.fold_results) == 8
    assert len(result.aggregate_results) == 2

    for table in (
        result.fold_results,
        result.aggregate_results,
    ):
        assert table[
            "annualized_return"
        ].eq(-0.10).all()
        assert table[
            "sharpe_ratio"
        ].eq(-0.75).all()
        assert table[
            "maximum_drawdown"
        ].eq(-0.30).all()

    assert observed_lengths.count(26) == 8
    assert observed_lengths.count(104) == 2


def test_configuration_ids_are_constant_across_all_folds() -> None:
    result = (
        walk_forward.run_trend_family_walk_forward(
            make_development_bars()
        )
    )

    observed = (
        result.fold_results.groupby(
            "strategy",
            observed=True,
            sort=False,
        )["configuration_id"]
        .agg(
            lambda values: tuple(
                values.unique()
            )
        )
        .to_dict()
    )

    assert observed == {
        strategy: (
            configuration_id,
        )
        for (
            strategy,
            configuration_id,
        ) in DAY10_CONFIGURATION_IDS.items()
    }
    assert len(
        {
            values[0]
            for values in observed.values()
        }
    ) == 2
