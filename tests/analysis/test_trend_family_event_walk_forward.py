"""Contracts for Day 13 event-driven walk-forward orchestration."""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
    fields,
    is_dataclass,
)
import inspect
from typing import cast

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.trend_family_event_walk_forward as day13
from systematic_alpha.analysis.strategy_performance import (
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_event_replay import (
    REPLAY_LEDGER_COLUMNS,
    TrendFamilyEventReplayResult,
    _prepare_replay_bars,
    _resolve_evaluation_window,
    run_trend_family_event_replay,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    _build_strategy_observations,
    _partition_fold,
    _reset_test_execution,
    build_walk_forward_folds,
)


EXPECTED_RUN_KEYS = [
    (
        strategy,
        f"wf_{year}",
    )
    for strategy in (
        "trend_ratio",
        "ema_macd",
    )
    for year in range(2022, 2026)
]
EXPECTED_PARITY_COLUMNS = (
    "strategy",
    "fold_id",
    "comparison",
    "comparison_type",
    "row_count",
    "maximum_absolute_difference",
    "mismatch_count",
    "tolerance",
    "passed",
)
PARITY_MAPPINGS = (
    (
        "target_position",
        "signal",
        "exact",
    ),
    (
        "signal_available",
        "signal_available",
        "exact",
    ),
    (
        "executed_position",
        "position",
        "exact",
    ),
    (
        "position_eligible",
        "position_eligible",
        "exact",
    ),
    (
        "turnover",
        "turnover",
        "numeric",
    ),
    (
        "transaction_cost",
        "transaction_cost",
        "numeric",
    ),
    (
        "gross_strategy_return",
        "gross_strategy_return",
        "numeric",
    ),
    (
        "net_strategy_return",
        "net_strategy_return",
        "numeric",
    ),
)
FORBIDDEN_TOKENS = (
    "winner",
    "ranking",
    "rank",
    "selected_strategy",
    "selected_parameters",
    "best_configuration",
    "optimisation",
    "optimization",
    "sensitivity_winner",
    "profitability_gate",
)


def make_session(
    session_date: str,
    *,
    observations: int = 26,
    start_value: float = 100.0,
) -> pd.DataFrame:
    """Build one valid synthetic SPY 15-minute session."""

    timestamps = pd.date_range(
        f"{session_date} 14:30:00+00:00",
        periods=observations,
        freq="15min",
    )
    close = (
        start_value
        + np.arange(observations, dtype=float)
        * 0.05
        + np.sin(
            np.arange(observations, dtype=float)
            / 3.0
        )
        * 0.10
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": session_date,
            "symbol": "SPY",
            "open": close - 0.02,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": np.arange(
                1_000,
                1_000 + observations,
                dtype=float,
            ),
            "trade_count": np.arange(
                100,
                100 + observations,
                dtype=int,
            ),
            "vwap": close,
            "source": "synthetic",
            "feed": "test",
        }
    )


def make_synthetic_development_bars() -> pd.DataFrame:
    """Build compact 2020-2025 sessions without future-period data."""

    specifications = (
        (
            "2020-01-02",
            26,
        ),
        (
            "2021-06-15",
            26,
        ),
        (
            "2022-01-03",
            26,
        ),
        (
            "2022-11-25",
            14,
        ),
        (
            "2023-01-03",
            26,
        ),
        (
            "2024-01-02",
            26,
        ),
        (
            "2025-01-02",
            26,
        ),
    )

    return pd.concat(
        [
            make_session(
                session_date,
                observations=observations,
                start_value=100.0 + 5.0 * index,
            )
            for index, (
                session_date,
                observations,
            ) in enumerate(specifications)
        ],
        ignore_index=True,
    )


def make_result_tables() -> dict[
    str,
    pd.DataFrame,
]:
    """Return small mutable tables for result-copy contracts."""

    return {
        name: pd.DataFrame(
            {
                "strategy": [
                    "trend_ratio",
                ],
                "value": [
                    float(index),
                ],
            }
        )
        for index, name in enumerate(
            (
                "fold_summary",
                "event_counts",
                "position_diagnostics",
                "performance",
                "vectorized_parity",
                "aggregate_summary",
            )
        )
    }


def install_orchestration_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    pd.DataFrame,
    list[tuple[str, str]],
    list[dict[str, object]],
    dict[str, int],
]:
    """Install deterministic synthetic preparation and replay spies."""

    raw = make_synthetic_development_bars()
    prepared = _prepare_replay_bars(
        raw,
        frequency="15min",
    )
    replay_calls: list[
        dict[str, object]
    ] = []
    fold_calls: list[
        tuple[str, str]
    ] = []
    reference_calls = {
        "build": 0,
        "reset": 0,
    }
    frozen_folds = build_walk_forward_folds()

    def prepare_spy(
        bars: pd.DataFrame,
    ) -> pd.DataFrame:
        assert isinstance(bars, pd.DataFrame)

        return prepared.copy(deep=True)

    def folds_spy():
        fold_calls.extend(
            (
                fold.fold_id,
                fold.test_start.isoformat(),
            )
            for fold in frozen_folds
        )

        return frozen_folds

    def build_reference_spy(
        frame: pd.DataFrame,
        *,
        strategy: str,
    ):
        reference_calls["build"] += 1

        return _build_strategy_observations(
            frame,
            strategy=strategy,
        )

    def reset_reference_spy(
        observations: pd.DataFrame,
        *,
        fold,
        cost_bps_per_turnover: float,
    ) -> pd.DataFrame:
        reference_calls["reset"] += 1

        return _reset_test_execution(
            observations,
            fold=fold,
            cost_bps_per_turnover=(
                cost_bps_per_turnover
            ),
        )

    def replay_spy(
        bars: pd.DataFrame,
        *,
        strategy: str,
        frequency: str,
        evaluation_start: pd.Timestamp,
        evaluation_end_exclusive: pd.Timestamp,
    ) -> TrendFamilyEventReplayResult:
        replay_calls.append(
            {
                "strategy": strategy,
                "frequency": frequency,
                "bars": bars.copy(deep=True),
                "evaluation_start": (
                    evaluation_start
                ),
                "evaluation_end_exclusive": (
                    evaluation_end_exclusive
                ),
            }
        )

        return run_trend_family_event_replay(
            bars,
            strategy=strategy,
            frequency=frequency,
            evaluation_start=evaluation_start,
            evaluation_end_exclusive=(
                evaluation_end_exclusive
            ),
        )

    monkeypatch.setattr(
        day13,
        "_prepare_development_features",
        prepare_spy,
    )
    monkeypatch.setattr(
        day13,
        "build_walk_forward_folds",
        folds_spy,
    )
    monkeypatch.setattr(
        day13,
        "_build_strategy_observations",
        build_reference_spy,
    )
    monkeypatch.setattr(
        day13,
        "_reset_test_execution",
        reset_reference_spy,
    )
    monkeypatch.setattr(
        day13,
        "run_trend_family_event_replay",
        replay_spy,
    )

    return (
        raw,
        fold_calls,
        replay_calls,
        reference_calls,
    )


def test_public_interface_is_frozen_and_narrow() -> None:
    """Freeze ordering, tolerance, and the one-argument API."""

    assert day13.STRATEGY_ORDER == (
        "trend_ratio",
        "ema_macd",
    )
    assert day13.PARITY_TOLERANCE == 1e-12

    signature = inspect.signature(
        day13.run_trend_family_event_walk_forward
    )

    assert tuple(signature.parameters) == (
        "bars",
    )
    parameter = signature.parameters["bars"]
    assert parameter.kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


def test_result_types_are_frozen_and_slotted() -> None:
    """Require immutable dataclass shells without instance dictionaries."""

    reference = pd.DataFrame(
        {
            "signal": [
                0,
            ],
        }
    )
    fold_run = day13.EventWalkForwardFoldRun(
        strategy="trend_ratio",
        fold_id="wf_2022",
        replay_result=cast(
            TrendFamilyEventReplayResult,
            object(),
        ),
        vectorized_observations=reference,
    )
    tables = make_result_tables()
    result = day13.TrendFamilyEventWalkForwardResults(
        fold_runs=(
            fold_run,
        ),
        **tables,
    )

    for value in (
        fold_run,
        result,
    ):
        assert is_dataclass(value)
        assert not hasattr(value, "__dict__")

    with pytest.raises(
        FrozenInstanceError,
    ):
        fold_run.strategy = "ema_macd"

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.fold_runs = ()


def test_fold_run_defensively_copies_vectorized_observations() -> None:
    """Prevent mutable reference observations from aliasing callers."""

    source = pd.DataFrame(
        {
            "signal": [
                0,
                1,
            ],
        },
        index=[
            5,
            6,
        ],
    )
    fold_run = day13.EventWalkForwardFoldRun(
        strategy="trend_ratio",
        fold_id="wf_2022",
        replay_result=cast(
            TrendFamilyEventReplayResult,
            object(),
        ),
        vectorized_observations=source,
    )
    source.loc[
        5,
        "signal",
    ] = -1

    assert isinstance(
        fold_run.vectorized_observations.index,
        pd.RangeIndex,
    )
    assert fold_run.vectorized_observations[
        "signal"
    ].tolist() == [
        0,
        1,
    ]

    copied = (
        fold_run.copy_vectorized_observations()
    )
    copied.loc[
        0,
        "signal",
    ] = -1
    assert fold_run.vectorized_observations.loc[
        0,
        "signal",
    ] == 0


def test_result_tables_are_defensively_copied() -> None:
    """Require copied storage and copied access for every result table."""

    tables = make_result_tables()
    result = day13.TrendFamilyEventWalkForwardResults(
        fold_runs=(),
        **tables,
    )
    copy_methods = {
        "fold_summary": (
            result.copy_fold_summary
        ),
        "event_counts": (
            result.copy_event_counts
        ),
        "position_diagnostics": (
            result.copy_position_diagnostics
        ),
        "performance": (
            result.copy_performance
        ),
        "vectorized_parity": (
            result.copy_vectorized_parity
        ),
        "aggregate_summary": (
            result.copy_aggregate_summary
        ),
    }

    for name, source in tables.items():
        source.loc[
            0,
            "value",
        ] = -1.0
        retained = getattr(result, name)
        assert retained.loc[
            0,
            "value",
        ] >= 0.0

        copied = copy_methods[name]()
        copied.loc[
            0,
            "value",
        ] = -2.0
        assert retained.loc[
            0,
            "value",
        ] >= 0.0


def test_non_dataframe_input_is_rejected() -> None:
    """The skeleton must reject invalid input before its red phase."""

    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        day13.run_trend_family_event_walk_forward(
            [
                1,
                2,
            ]
        )


def test_public_types_contain_no_selection_or_tuning_fields() -> None:
    """Forbid optimisation concepts in the public Day 13 contract."""

    names = [
        field.name
        for result_type in (
            day13.EventWalkForwardFoldRun,
            day13.TrendFamilyEventWalkForwardResults,
        )
        for field in fields(result_type)
    ]
    normalized = " ".join(names).lower()

    assert not any(
        token in normalized
        for token in FORBIDDEN_TOKENS
    )


def test_existing_day12_boundary_contract_supports_both_end_edges() -> None:
    """Document next-session and terminal exclusive boundaries."""

    prepared = _prepare_replay_bars(
        pd.concat(
            [
                make_session("2024-01-02"),
                make_session("2024-01-03"),
            ],
            ignore_index=True,
        ),
        frequency="15min",
    )
    first_start = pd.Timestamp(
        prepared.loc[
            prepared["session_date"].eq(
                "2024-01-02"
            ),
            "timestamp",
        ].iloc[0]
    )
    next_start = pd.Timestamp(
        prepared.loc[
            prepared["session_date"].eq(
                "2024-01-03"
            ),
            "timestamp",
        ].iloc[0]
    )
    terminal = pd.Timestamp(
        prepared["timestamp"].iloc[-1]
    ) + pd.Timedelta(minutes=15)

    start, end, mask = (
        _resolve_evaluation_window(
            prepared,
            evaluation_start=first_start,
            evaluation_end_exclusive=next_start,
        )
    )
    assert start == first_start
    assert end == next_start
    assert int(mask.sum()) == 26

    _, terminal_end, terminal_mask = (
        _resolve_evaluation_window(
            prepared,
            evaluation_start=first_start,
            evaluation_end_exclusive=terminal,
        )
    )
    assert terminal_end == terminal
    assert int(terminal_mask.sum()) == 52


def test_reuses_exact_folds_and_eight_run_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require strategy-major use of the four immutable Day 11 folds."""

    (
        raw,
        fold_calls,
        replay_calls,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )

    assert fold_calls == [
        (
            f"wf_{year}",
            pd.Timestamp(
                f"{year}-01-01",
                tz="UTC",
            ).isoformat(),
        )
        for year in range(2022, 2026)
    ]
    assert [
        (
            run.strategy,
            run.fold_id,
        )
        for run in result.fold_runs
    ] == EXPECTED_RUN_KEYS
    assert [
        (
            call["strategy"],
            f"wf_{pd.Timestamp(call['evaluation_start']).year}",
        )
        for call in replay_calls
    ] == EXPECTED_RUN_KEYS
    assert len(replay_calls) == 8


def test_fold_inputs_use_expanding_history_without_future_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supply each replay with training plus its test year and nothing later."""

    (
        raw,
        _,
        replay_calls,
        _,
    ) = install_orchestration_spies(monkeypatch)
    original = raw.copy(deep=True)

    day13.run_trend_family_event_walk_forward(
        raw
    )

    for call in replay_calls:
        frame = cast(
            pd.DataFrame,
            call["bars"],
        )
        test_year = pd.Timestamp(
            call["evaluation_start"]
        ).year
        local_years = (
            pd.to_datetime(
                frame["timestamp"],
                utc=True,
            )
            .dt.tz_convert(
                "America/New_York"
            )
            .dt.year
        )

        assert local_years.min() == 2020
        assert local_years.max() == test_year
        assert not local_years.ge(
            test_year + 1
        ).any()
        assert not local_years.eq(2026).any()

    pd.testing.assert_frame_equal(
        raw,
        original,
    )


def test_maps_calendar_folds_to_actual_bar_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass real session edges rather than UTC-midnight fold dates."""

    (
        raw,
        _,
        replay_calls,
        _,
    ) = install_orchestration_spies(monkeypatch)

    day13.run_trend_family_event_walk_forward(
        raw
    )

    for call in replay_calls:
        frame = cast(
            pd.DataFrame,
            call["bars"],
        )
        start = pd.Timestamp(
            call["evaluation_start"]
        )
        end = pd.Timestamp(
            call["evaluation_end_exclusive"]
        )
        test_year = start.year
        test_rows = frame.loc[
            pd.to_datetime(
                frame["session_date"]
            ).dt.year.eq(test_year)
        ]

        assert start == pd.Timestamp(
            test_rows["timestamp"].iloc[0]
        )
        assert start.hour != 0
        assert end == (
            pd.Timestamp(
                test_rows["timestamp"].iloc[-1]
            )
            + pd.Timedelta(minutes=15)
        )


def test_calls_only_the_public_day12_replay_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze the exact public Day 12 call contract."""

    (
        raw,
        _,
        replay_calls,
        _,
    ) = install_orchestration_spies(monkeypatch)

    assert not hasattr(
        day13,
        "_run_event_replay_core",
    )
    day13.run_trend_family_event_walk_forward(
        raw
    )

    assert len(replay_calls) == 8
    assert all(
        call["frequency"] == "15min"
        for call in replay_calls
    )
    assert {
        call["strategy"]
        for call in replay_calls
    } == {
        "trend_ratio",
        "ema_macd",
    }


def test_fold_results_are_test_only_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude training events and require a neutral portfolio reset."""

    (
        raw,
        _,
        _,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )

    for run in result.fold_runs:
        replay = run.replay_result
        observations = replay.observations
        first = observations.iloc[0]

        assert observations[
            "timestamp"
        ].ge(
            replay.evaluation_start
        ).all()
        assert observations[
            "timestamp"
        ].lt(
            replay.evaluation_end_exclusive
        ).all()
        assert first[
            "previous_executed_position"
        ] == 0
        assert first["executed_position"] == 0
        assert not bool(
            first["position_eligible"]
        )
        assert first["position_change"] == 0
        assert first["turnover"] == 0.0
        assert first["transaction_cost"] == 0.0
        assert first["previous_equity"] == 1.0
        assert first["cash_balance"] == 1.0
        assert first["holdings_value"] == 0.0
        assert first["ending_equity"] == 1.0
        assert not bool(first["fill_executed"])


def test_first_signal_is_delayed_and_state_persists_across_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one-bar execution delay without ordinary-session resets."""

    (
        raw,
        _,
        _,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )

    for run in result.fold_runs:
        observations = (
            run.replay_result.observations
        )
        assert observations.loc[
            0,
            "executed_position",
        ] == 0
        assert observations.loc[
            1,
            "executed_position",
        ] == observations.loc[
            0,
            "target_position",
        ]

        session_change = observations[
            "session_date"
        ].ne(
            observations[
                "session_date"
            ].shift()
        )
        later_session_starts = list(
            observations.index[
                session_change
                & observations.index.to_series().gt(
                    0
                )
            ]
        )

        for index in later_session_starts:
            assert observations.loc[
                index,
                "executed_position",
            ] == observations.loc[
                index - 1,
                "target_position",
            ]
            assert observations.loc[
                index,
                "previous_equity",
            ] == pytest.approx(
                observations.loc[
                    index - 1,
                    "ending_equity",
                ]
            )


def test_vectorized_reference_is_independent_and_parity_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require independent Day 11 reset observations and eight comparisons."""

    (
        raw,
        _,
        _,
        reference_calls,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )

    assert reference_calls == {
        "build": 8,
        "reset": 8,
    }
    assert tuple(
        result.vectorized_parity.columns
    ) == EXPECTED_PARITY_COLUMNS
    assert len(result.vectorized_parity) == 64
    assert result.vectorized_parity[
        "passed"
    ].astype(bool).all()

    expected_comparisons = [
        (
            strategy,
            fold_id,
            replay_column,
            comparison_type,
        )
        for strategy, fold_id in (
            EXPECTED_RUN_KEYS
        )
        for (
            replay_column,
            _,
            comparison_type,
        ) in PARITY_MAPPINGS
    ]
    assert list(
        result.vectorized_parity[
            [
                "strategy",
                "fold_id",
                "comparison",
                "comparison_type",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ) == expected_comparisons

    for run in result.fold_runs:
        replay = run.replay_result.observations
        reference = (
            run.vectorized_observations
        )
        assert tuple(replay.columns) == (
            REPLAY_LEDGER_COLUMNS
        )

        for (
            replay_column,
            reference_column,
            comparison_type,
        ) in PARITY_MAPPINGS:
            if comparison_type == "exact":
                assert replay[
                    replay_column
                ].reset_index(
                    drop=True
                ).equals(
                    reference[
                        reference_column
                    ].reset_index(drop=True)
                )
            else:
                np.testing.assert_allclose(
                    replay[replay_column],
                    reference[
                        reference_column
                    ],
                    rtol=(
                        day13.PARITY_TOLERANCE
                    ),
                    atol=(
                        day13.PARITY_TOLERANCE
                    ),
                )


def test_fold_performance_uses_observed_session_annualization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recompute gross and net fold metrics using the Day 11 factor."""

    (
        raw,
        _,
        _,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )
    required = {
        "strategy",
        "fold_id",
        "series",
        "observations",
        "sessions",
        "annualization_factor",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    }
    assert required <= set(
        result.performance.columns
    )
    assert len(result.performance) == 16

    fold_2022 = result.performance.loc[
        result.performance["fold_id"].eq(
            "wf_2022"
        )
    ]
    assert fold_2022[
        "observations"
    ].eq(40).all()
    assert fold_2022["sessions"].eq(2).all()
    assert fold_2022[
        "annualization_factor"
    ].eq(
        252.0 * 40.0 / 2.0
    ).all()

    for run in result.fold_runs:
        observations = (
            run.replay_result.observations
        )
        sessions = int(
            observations[
                "session_date"
            ].nunique()
        )
        factor = (
            252.0
            * len(observations)
            / sessions
        )

        for series, column in (
            (
                "gross",
                "gross_strategy_return",
            ),
            (
                "net",
                "net_strategy_return",
            ),
        ):
            expected = (
                calculate_performance_metrics(
                    observations[column],
                    annualization_factor=factor,
                )
            )
            row = result.performance.loc[
                result.performance[
                    "strategy"
                ].eq(run.strategy)
                & result.performance[
                    "fold_id"
                ].eq(run.fold_id)
                & result.performance[
                    "series"
                ].eq(series)
            ].iloc[0]
            assert row[
                "cumulative_return"
            ] == pytest.approx(
                expected.cumulative_return
            )
            assert row[
                "maximum_drawdown"
            ] == pytest.approx(
                expected.max_drawdown
            )


def test_aggregate_metrics_recompute_concatenated_reset_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate chronological fold returns without linking replay state."""

    (
        raw,
        _,
        _,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )
    required = {
        "strategy",
        "series",
        "folds",
        "observations",
        "sessions",
        "annualization_factor",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_wealth",
    }
    assert required <= set(
        result.aggregate_summary.columns
    )
    assert len(result.aggregate_summary) == 4

    for strategy in day13.STRATEGY_ORDER:
        runs = [
            run
            for run in result.fold_runs
            if run.strategy == strategy
        ]
        assert [
            run.fold_id
            for run in runs
        ] == [
            f"wf_{year}"
            for year in range(2022, 2026)
        ]
        assert all(
            run.replay_result.observations.loc[
                0,
                "previous_equity",
            ]
            == 1.0
            for run in runs
        )
        combined = pd.concat(
            [
                run.replay_result.observations
                for run in runs
            ],
            ignore_index=True,
        )
        sessions = int(
            combined["session_date"].nunique()
        )
        factor = (
            252.0 * len(combined) / sessions
        )

        for series, column in (
            (
                "gross",
                "gross_strategy_return",
            ),
            (
                "net",
                "net_strategy_return",
            ),
        ):
            expected = (
                calculate_performance_metrics(
                    combined[column],
                    annualization_factor=factor,
                )
            )
            row = result.aggregate_summary.loc[
                result.aggregate_summary[
                    "strategy"
                ].eq(strategy)
                & result.aggregate_summary[
                    "series"
                ].eq(series)
            ].iloc[0]

            assert row["folds"] == 4
            assert row[
                "observations"
            ] == len(combined)
            assert row[
                "annualization_factor"
            ] == pytest.approx(factor)
            assert row[
                "cumulative_return"
            ] == pytest.approx(
                expected.cumulative_return
            )
            assert row[
                "maximum_drawdown"
            ] == pytest.approx(
                expected.max_drawdown
            )


def test_behavioral_result_schemas_contain_no_selection_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forbid ranking and profitability gates in generated evidence."""

    (
        raw,
        _,
        _,
        _,
    ) = install_orchestration_spies(monkeypatch)
    result = (
        day13.run_trend_family_event_walk_forward(
            raw
        )
    )
    all_columns = " ".join(
        str(column).lower()
        for frame in (
            result.fold_summary,
            result.event_counts,
            result.position_diagnostics,
            result.performance,
            result.vectorized_parity,
            result.aggregate_summary,
        )
        for column in frame.columns
    )

    assert not any(
        token in all_columns
        for token in FORBIDDEN_TOKENS
    )
