"""Synthetic tests for Day 7 trend-ratio sensitivity foundations."""

from __future__ import annotations

import pytest

from systematic_alpha.analysis.trend_ratio_sensitivity import (
    BASELINE_LONG_WINDOW,
    BASELINE_NEUTRAL_BAND,
    BASELINE_SHORT_WINDOW,
    LONG_WINDOWS,
    NEUTRAL_BANDS,
    SHORT_WINDOWS,
    TrendRatioConfiguration,
    build_neighborhood_map,
    build_parameter_grid,
    calculate_filter_lag,
    configurations_are_neighbors,
)


def test_parameter_grid_is_deterministic_and_contains_only_valid_pairs() -> None:
    first_grid = build_parameter_grid()
    second_grid = build_parameter_grid()

    assert first_grid == second_grid
    assert len(first_grid) == 36

    assert all(
        configuration.short_window < configuration.long_window
        for configuration in first_grid
    )

    assert tuple(
        configuration.short_window
        for configuration in first_grid[:12]
    ) == (SHORT_WINDOWS[0],) * 12


def test_day06_baseline_configuration_is_included_exactly_once() -> None:
    parameter_grid = build_parameter_grid()

    baseline_matches = [
        configuration
        for configuration in parameter_grid
        if (
            configuration.short_window == BASELINE_SHORT_WINDOW
            and configuration.long_window == BASELINE_LONG_WINDOW
            and configuration.neutral_band == BASELINE_NEUTRAL_BAND
        )
    ]

    assert len(baseline_matches) == 1


def test_parameter_grid_contains_no_silent_duplicates() -> None:
    parameter_grid = build_parameter_grid()

    configurations = {
        (
            configuration.short_window,
            configuration.long_window,
            configuration.neutral_band,
        )
        for configuration in parameter_grid
    }
    configuration_ids = {
        configuration.configuration_id
        for configuration in parameter_grid
    }

    assert len(configurations) == len(parameter_grid)
    assert len(configuration_ids) == len(parameter_grid)


def test_invalid_declared_grid_is_rejected_instead_of_filtered() -> None:
    with pytest.raises(
        ValueError,
        match="Every declared short window",
    ):
        build_parameter_grid(
            short_windows=(4, 32),
            long_windows=(32, 64),
            neutral_bands=NEUTRAL_BANDS,
        )


def test_filter_lag_diagnostics_follow_predeclared_formula() -> None:
    configuration = TrendRatioConfiguration(
        short_window=8,
        long_window=32,
        neutral_band=0.001,
    )

    diagnostics = calculate_filter_lag(configuration)

    assert diagnostics.short_filter_lag_bars == pytest.approx(3.5)
    assert diagnostics.long_filter_lag_bars == pytest.approx(15.5)
    assert diagnostics.lag_spread_bars == pytest.approx(12.0)


def test_neighborhood_definition_is_axis_adjacent_and_symmetric() -> None:
    center = TrendRatioConfiguration(
        short_window=8,
        long_window=64,
        neutral_band=0.0005,
    )
    short_neighbor = TrendRatioConfiguration(
        short_window=4,
        long_window=64,
        neutral_band=0.0005,
    )
    long_neighbor = TrendRatioConfiguration(
        short_window=8,
        long_window=96,
        neutral_band=0.0005,
    )
    band_neighbor = TrendRatioConfiguration(
        short_window=8,
        long_window=64,
        neutral_band=0.0010,
    )
    diagonal = TrendRatioConfiguration(
        short_window=4,
        long_window=96,
        neutral_band=0.0005,
    )

    assert configurations_are_neighbors(center, short_neighbor)
    assert configurations_are_neighbors(short_neighbor, center)
    assert configurations_are_neighbors(center, long_neighbor)
    assert configurations_are_neighbors(center, band_neighbor)
    assert not configurations_are_neighbors(center, diagonal)
    assert not configurations_are_neighbors(center, center)


def test_edge_and_corner_configurations_have_fewer_neighbors() -> None:
    neighborhood_map = build_neighborhood_map()

    corner = TrendRatioConfiguration(
        short_window=4,
        long_window=32,
        neutral_band=0.0,
    )
    interior = TrendRatioConfiguration(
        short_window=8,
        long_window=64,
        neutral_band=0.0005,
    )
    baseline = TrendRatioConfiguration(
        short_window=8,
        long_window=32,
        neutral_band=0.0010,
    )

    assert len(neighborhood_map[corner.configuration_id]) == 3
    assert len(neighborhood_map[interior.configuration_id]) == 6
    assert len(neighborhood_map[baseline.configuration_id]) == 5


def test_neighborhood_map_is_deterministic_and_symmetric() -> None:
    first_map = build_neighborhood_map()
    second_map = build_neighborhood_map()

    assert first_map == second_map

    for configuration_id, neighbor_ids in first_map.items():
        assert neighbor_ids == tuple(sorted(neighbor_ids))

        for neighbor_id in neighbor_ids:
            assert configuration_id in first_map[neighbor_id]


def test_frozen_grid_axes_remain_exactly_as_declared() -> None:
    assert SHORT_WINDOWS == (4, 8, 16)
    assert LONG_WINDOWS == (32, 64, 96)
    assert NEUTRAL_BANDS == (0.0, 0.0005, 0.0010, 0.0020)


def test_holding_episodes_exclude_neutral_runs_and_split_reversals() -> None:
    import pandas as pd

    positions = pd.Series(
        [0, 1, 1, -1, -1, 0, 1],
        index=pd.date_range(
            "2025-01-02 09:30",
            periods=7,
            freq="15min",
        ),
        dtype=float,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        identify_holding_episodes,
    )

    episodes = identify_holding_episodes(positions)

    assert len(episodes) == 3

    assert episodes[0].position == 1
    assert episodes[0].start_position == 1
    assert episodes[0].end_position == 2
    assert episodes[0].duration_bars == 2

    assert episodes[1].position == -1
    assert episodes[1].start_position == 3
    assert episodes[1].end_position == 4
    assert episodes[1].duration_bars == 2

    assert episodes[2].position == 1
    assert episodes[2].start_position == 6
    assert episodes[2].end_position == 6


def test_episode_is_not_reset_at_calendar_year_boundary() -> None:
    import pandas as pd

    positions = pd.Series(
        [1, 1, 1, 1],
        index=pd.to_datetime(
            [
                "2024-12-31 15:30",
                "2024-12-31 15:45",
                "2025-01-02 09:30",
                "2025-01-02 09:45",
            ]
        ),
        dtype=float,
    )
    sessions = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31",
                "2024-12-31",
                "2025-01-02",
                "2025-01-02",
            ]
        ),
        index=positions.index,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        identify_holding_episodes,
    )

    episodes = identify_holding_episodes(
        positions,
        session_labels=sessions,
    )

    assert len(episodes) == 1
    assert episodes[0].duration_bars == 4
    assert episodes[0].crosses_session_boundary
    assert episodes[0].start_session == pd.Timestamp("2024-12-31")
    assert episodes[0].end_session == pd.Timestamp("2025-01-02")


def test_session_crossing_episode_is_detected_exactly() -> None:
    import pandas as pd

    positions = pd.Series(
        [0, 1, 1, 1, 0, -1],
        index=pd.date_range(
            "2025-01-02 15:30",
            periods=6,
            freq="15min",
        ),
        dtype=float,
    )
    sessions = pd.Series(
        [
            "2025-01-02",
            "2025-01-02",
            "2025-01-02",
            "2025-01-03",
            "2025-01-03",
            "2025-01-03",
        ],
        index=positions.index,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        identify_holding_episodes,
    )

    episodes = identify_holding_episodes(
        positions,
        session_labels=sessions,
    )

    assert len(episodes) == 2
    assert episodes[0].crosses_session_boundary
    assert not episodes[1].crosses_session_boundary


def test_holding_episode_functions_do_not_mutate_inputs() -> None:
    import pandas as pd

    positions = pd.Series([0, 1, 1, 0, -1], dtype=float)
    sessions = pd.Series(["a", "a", "a", "a", "a"])

    original_positions = positions.copy(deep=True)
    original_sessions = sessions.copy(deep=True)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        calculate_holding_diagnostics,
    )

    calculate_holding_diagnostics(
        positions,
        session_labels=sessions,
    )

    pd.testing.assert_series_equal(positions, original_positions)
    pd.testing.assert_series_equal(sessions, original_sessions)


def test_whipsaw_definition_accepts_short_direct_reversal() -> None:
    import pandas as pd

    positions = pd.Series([1, 1, -1, -1, -1], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        classify_whipsaw_episodes,
        identify_holding_episodes,
    )

    episodes = identify_holding_episodes(positions)
    classifications = classify_whipsaw_episodes(episodes)

    assert classifications[0].is_whipsaw
    assert classifications[0].following_start_gap_bars == 1
    assert not classifications[1].is_whipsaw


def test_episode_longer_than_four_bars_is_not_a_whipsaw() -> None:
    import pandas as pd

    positions = pd.Series(
        [1, 1, 1, 1, 1, -1],
        dtype=float,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        classify_whipsaw_episodes,
        identify_holding_episodes,
    )

    classifications = classify_whipsaw_episodes(
        identify_holding_episodes(positions)
    )

    assert not classifications[0].is_whipsaw


def test_opposite_episode_starting_more_than_four_bars_later_is_not_whipsaw(
) -> None:
    import pandas as pd

    positions = pd.Series(
        [1, 1, 0, 0, 0, 0, -1],
        dtype=float,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        classify_whipsaw_episodes,
        identify_holding_episodes,
    )

    classifications = classify_whipsaw_episodes(
        identify_holding_episodes(positions)
    )

    assert classifications[0].following_start_gap_bars == 5
    assert not classifications[0].is_whipsaw


def test_next_same_direction_episode_blocks_later_opposite_episode() -> None:
    import pandas as pd

    positions = pd.Series(
        [1, 0, 1, 0, -1],
        dtype=float,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        classify_whipsaw_episodes,
        identify_holding_episodes,
    )

    classifications = classify_whipsaw_episodes(
        identify_holding_episodes(positions)
    )

    assert not classifications[0].is_whipsaw
    assert classifications[0].following_episode_id == 2
    assert classifications[1].is_whipsaw


def test_holding_diagnostics_report_counts_rates_and_duration_statistics(
) -> None:
    import pandas as pd

    positions = pd.Series(
        [0, 1, 1, -1, -1, -1, 0, 1, 0, -1],
        dtype=float,
    )
    sessions = pd.Series(
        ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"]
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        calculate_holding_diagnostics,
    )

    diagnostics = calculate_holding_diagnostics(
        positions,
        session_labels=sessions,
    )

    assert diagnostics.eligible_observations == 10
    assert diagnostics.non_zero_episode_count == 4
    assert diagnostics.long_episode_count == 2
    assert diagnostics.short_episode_count == 2

    assert diagnostics.median_holding_duration_bars == pytest.approx(1.5)
    assert diagnostics.mean_holding_duration_bars == pytest.approx(1.75)
    assert diagnostics.maximum_holding_duration_bars == pytest.approx(3.0)

    assert diagnostics.overnight_carry_episode_count == 1
    assert diagnostics.session_crossing_episode_proportion == pytest.approx(
        0.25
    )

    assert diagnostics.whipsaw_count == 3
    assert (
        diagnostics.whipsaw_rate_per_1000_eligible_observations
        == pytest.approx(300.0)
    )
    assert diagnostics.whipsaw_episode_proportion == pytest.approx(0.75)


def test_invalid_position_values_are_rejected() -> None:
    import pandas as pd

    positions = pd.Series([0, 1, 0.5, -1], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        identify_holding_episodes,
    )

    with pytest.raises(ValueError, match="only -1, 0, 1"):
        identify_holding_episodes(positions)


def test_cost_break_even_matches_known_compounded_solution() -> None:
    import pandas as pd

    gross_returns = pd.Series([0.005, 0.005], dtype=float)
    turnover = pd.Series([1.0, 1.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_ROOT_FOUND,
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    assert result.status == BREAK_EVEN_STATUS_ROOT_FOUND
    assert result.break_even_cost_bps == pytest.approx(
        50.0,
        abs=1e-9,
    )
    assert result.total_turnover == pytest.approx(2.0)


def test_cost_break_even_reports_non_positive_gross_performance() -> None:
    import pandas as pd

    gross_returns = pd.Series([-0.01, 0.0], dtype=float)
    turnover = pd.Series([1.0, 1.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_NON_POSITIVE_GROSS,
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    assert result.status == BREAK_EVEN_STATUS_NON_POSITIVE_GROSS
    assert result.break_even_cost_bps is None
    assert result.gross_cumulative_return < 0.0


def test_cost_break_even_reports_zero_turnover_explicitly() -> None:
    import pandas as pd

    gross_returns = pd.Series([0.01, 0.02], dtype=float)
    turnover = pd.Series([0.0, 0.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_ZERO_TURNOVER,
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    assert result.status == BREAK_EVEN_STATUS_ZERO_TURNOVER
    assert result.break_even_cost_bps is None
    assert result.total_turnover == pytest.approx(0.0)


def test_cost_break_even_reports_root_above_search_interval() -> None:
    import pandas as pd

    gross_returns = pd.Series([0.02, 0.02], dtype=float)
    turnover = pd.Series([1.0, 1.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_ROOT_ABOVE_INTERVAL,
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    assert result.status == BREAK_EVEN_STATUS_ROOT_ABOVE_INTERVAL
    assert result.break_even_cost_bps is None
    assert result.objective_at_effective_upper > 0.0


def test_break_even_can_be_solved_before_invalid_upper_bound() -> None:
    import pandas as pd

    gross_returns = pd.Series([10.0, -0.5], dtype=float)
    turnover = pd.Series([0.0, 100.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_ROOT_FOUND,
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    expected_root = 100.0 * (0.5 - 1.0 / 11.0)

    assert result.status == BREAK_EVEN_STATUS_ROOT_FOUND
    assert result.break_even_cost_bps == pytest.approx(
        expected_root,
        abs=1e-8,
    )
    assert result.effective_upper_bps < 50.0


def test_break_even_rejects_returns_that_destroy_initial_wealth() -> None:
    import pandas as pd

    gross_returns = pd.Series([0.01, -1.0], dtype=float)
    turnover = pd.Series([1.0, 1.0], dtype=float)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        calculate_cost_break_even,
    )

    with pytest.raises(
        ValueError,
        match="greater than -1.0",
    ):
        calculate_cost_break_even(
            gross_returns,
            turnover,
        )


def test_break_even_inputs_are_not_mutated() -> None:
    import pandas as pd

    gross_returns = pd.Series(
        [float("nan"), 0.005, 0.005],
        index=["warmup", "a", "b"],
        dtype=float,
    )
    turnover = pd.Series(
        [float("nan"), 1.0, 1.0],
        index=["warmup", "a", "b"],
        dtype=float,
    )

    original_gross_returns = gross_returns.copy(deep=True)
    original_turnover = turnover.copy(deep=True)

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        calculate_cost_break_even,
    )

    result = calculate_cost_break_even(
        gross_returns,
        turnover,
    )

    pd.testing.assert_series_equal(
        gross_returns,
        original_gross_returns,
    )
    pd.testing.assert_series_equal(
        turnover,
        original_turnover,
    )
    assert result.eligible_observations == 2


def test_break_even_requires_identical_indexes() -> None:
    import pandas as pd

    gross_returns = pd.Series(
        [0.01, 0.01],
        index=["a", "b"],
        dtype=float,
    )
    turnover = pd.Series(
        [1.0, 1.0],
        index=["a", "c"],
        dtype=float,
    )

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        calculate_cost_break_even,
    )

    with pytest.raises(
        ValueError,
        match="identical indexes",
    ):
        calculate_cost_break_even(
            gross_returns,
            turnover,
        )


def _build_synthetic_strategy_frame(
    close_values: list[float],
):
    import pandas as pd

    close = pd.Series(close_values, dtype=float)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=len(close),
                freq="15min",
                tz="UTC",
            ),
            "symbol": "SPY",
            "close": close,
            "close_to_close_simple_return": close.pct_change(),
        }
    )


def test_single_configuration_run_matches_existing_day06_engine() -> None:
    import numpy as np
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BASELINE_LONG_WINDOW,
        BASELINE_NEUTRAL_BAND,
        BASELINE_SHORT_WINDOW,
        TrendRatioConfiguration,
        run_trend_ratio_configuration,
    )
    from systematic_alpha.strategies.trend_ratio import (
        TrendRatioParameters,
        build_trend_ratio_strategy,
    )

    observation_number = np.arange(64, dtype=float)
    close_values = (
        100.0
        + observation_number * 0.08
        + np.sin(observation_number / 2.0)
    ).tolist()

    frame = _build_synthetic_strategy_frame(close_values)

    configuration = TrendRatioConfiguration(
        short_window=BASELINE_SHORT_WINDOW,
        long_window=BASELINE_LONG_WINDOW,
        neutral_band=BASELINE_NEUTRAL_BAND,
    )

    integrated = run_trend_ratio_configuration(
        frame,
        configuration=configuration,
        cost_bps_per_turnover=1.0,
    )

    direct = build_trend_ratio_strategy(
        frame,
        parameters=TrendRatioParameters(
            short_window=BASELINE_SHORT_WINDOW,
            long_window=BASELINE_LONG_WINDOW,
            neutral_band=BASELINE_NEUTRAL_BAND,
            cost_bps_per_turnover=1.0,
            price_column="close",
            return_column="close_to_close_simple_return",
        ),
    )

    pd.testing.assert_frame_equal(
        integrated.strategy_bundle.observations,
        direct.observations,
    )
    pd.testing.assert_frame_equal(
        integrated.strategy_bundle.diagnostics,
        direct.diagnostics,
    )
    assert integrated.strategy_bundle.parameters == direct.parameters


def test_single_configuration_positions_are_one_bar_lagged() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_configuration,
    )

    frame = _build_synthetic_strategy_frame(
        [1.0, 2.0, 1.0, 1.2, 0.9, 1.1]
    )

    result = run_trend_ratio_configuration(
        frame,
        configuration=TrendRatioConfiguration(
            short_window=1,
            long_window=2,
            neutral_band=0.0,
        ),
    )

    observations = result.strategy_bundle.observations

    expected_position = (
        observations.groupby(
            "symbol",
            observed=True,
            sort=False,
        )["signal"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    pd.testing.assert_series_equal(
        observations["position"],
        expected_position,
        check_names=False,
    )


def test_single_configuration_retains_turnover_two_on_reversal() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_configuration,
    )

    frame = _build_synthetic_strategy_frame(
        [1.0, 2.0, 1.0, 1.0]
    )

    result = run_trend_ratio_configuration(
        frame,
        configuration=TrendRatioConfiguration(
            short_window=1,
            long_window=2,
            neutral_band=0.0,
        ),
    )

    observations = result.strategy_bundle.observations

    assert observations["position"].tolist() == [0, 0, 1, -1]
    assert observations["turnover"].tolist() == [0.0, 0.0, 1.0, 2.0]


def test_future_price_change_does_not_alter_past_strategy_output() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_configuration,
    )

    frame = _build_synthetic_strategy_frame(
        [
            100.0,
            101.0,
            102.0,
            101.5,
            103.0,
            104.0,
            103.5,
            105.0,
            106.0,
            107.0,
        ]
    )
    modified = frame.copy(deep=True)

    final_row = modified.index[-1]
    previous_row = modified.index[-2]

    modified.loc[final_row, "close"] = 250.0
    modified.loc[
        final_row,
        "close_to_close_simple_return",
    ] = (
        modified.loc[final_row, "close"]
        / modified.loc[previous_row, "close"]
        - 1.0
    )

    configuration = TrendRatioConfiguration(
        short_window=2,
        long_window=4,
        neutral_band=0.001,
    )

    original_result = run_trend_ratio_configuration(
        frame,
        configuration=configuration,
    )
    modified_result = run_trend_ratio_configuration(
        modified,
        configuration=configuration,
    )

    pd.testing.assert_frame_equal(
        original_result.strategy_bundle.observations.iloc[:-1],
        modified_result.strategy_bundle.observations.iloc[:-1],
    )


def test_single_configuration_run_does_not_mutate_input_frame() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_configuration,
    )

    frame = _build_synthetic_strategy_frame(
        [100.0, 101.0, 100.5, 102.0, 101.0, 103.0]
    )
    original = frame.copy(deep=True)

    run_trend_ratio_configuration(
        frame,
        configuration=TrendRatioConfiguration(
            short_window=2,
            long_window=4,
            neutral_band=0.001,
        ),
    )

    pd.testing.assert_frame_equal(frame, original)


def test_annual_results_do_not_reset_position_at_year_boundary() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        build_annual_strategy_results,
        run_trend_ratio_configuration,
    )

    close = pd.Series(
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        dtype=float,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-12-31 19:00:00+00:00",
                    "2024-12-31 19:15:00+00:00",
                    "2024-12-31 19:30:00+00:00",
                    "2025-01-02 14:30:00+00:00",
                    "2025-01-02 14:45:00+00:00",
                    "2025-01-02 15:00:00+00:00",
                ]
            ),
            "symbol": "SPY",
            "close": close,
            "close_to_close_simple_return": close.pct_change(),
        }
    )

    run = run_trend_ratio_configuration(
        frame,
        configuration=TrendRatioConfiguration(
            short_window=1,
            long_window=2,
            neutral_band=0.0,
        ),
    )

    observations = run.strategy_bundle.observations
    first_2025_row = observations.loc[
        observations["timestamp"].dt.year.eq(2025)
    ].iloc[0]

    assert first_2025_row["position"] == 1
    assert first_2025_row["position_eligible"]

    annual = build_annual_strategy_results(observations)

    assert annual["calendar_year"].tolist() == [2024, 2025]
    assert annual["observations"].tolist() == [3, 3]


def test_annual_metrics_match_precomputed_full_run_year_slice() -> None:
    import pandas as pd

    from systematic_alpha.analysis.strategy_performance import (
        calculate_performance_metrics,
    )
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        DAY07_ANNUALIZATION_FACTOR,
        TrendRatioConfiguration,
        build_annual_strategy_results,
        run_trend_ratio_configuration,
    )

    close = pd.Series(
        [100.0, 102.0, 104.0, 103.0, 105.0, 107.0],
        dtype=float,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-12-31 19:00:00+00:00",
                    "2024-12-31 19:15:00+00:00",
                    "2024-12-31 19:30:00+00:00",
                    "2025-01-02 14:30:00+00:00",
                    "2025-01-02 14:45:00+00:00",
                    "2025-01-02 15:00:00+00:00",
                ]
            ),
            "symbol": "SPY",
            "close": close,
            "close_to_close_simple_return": close.pct_change(),
        }
    )

    run = run_trend_ratio_configuration(
        frame,
        configuration=TrendRatioConfiguration(
            short_window=1,
            long_window=2,
            neutral_band=0.0,
        ),
    )
    observations = run.strategy_bundle.observations
    year_slice = observations.loc[
        observations["timestamp"].dt.year.eq(2025)
    ]

    expected = calculate_performance_metrics(
        year_slice["net_strategy_return"],
        annualization_factor=DAY07_ANNUALIZATION_FACTOR,
    )

    annual = build_annual_strategy_results(observations)
    row = annual.loc[
        annual["calendar_year"].eq(2025)
    ].iloc[0]

    assert row["net_return"] == pytest.approx(
        expected.cumulative_return
    )
    assert row["net_sharpe_ratio"] == pytest.approx(
        expected.sharpe_ratio
    )
    assert row["net_max_drawdown"] == pytest.approx(
        expected.max_drawdown
    )


def test_annual_exposures_use_position_eligible_observations() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_annual_strategy_results,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=5,
                freq="15min",
                tz="UTC",
            ),
            "symbol": "SPY",
            "position": [0, 0, 1, -1, 0],
            "position_eligible": [False, False, True, True, True],
            "turnover": [0.0, 0.0, 1.0, 2.0, 1.0],
            "gross_strategy_return": [0.0] * 5,
            "net_strategy_return": [0.0] * 5,
        }
    )

    annual = build_annual_strategy_results(observations)
    row = annual.iloc[0]

    assert row["position_eligible_observations"] == 3
    assert row["long_exposure_pct"] == pytest.approx(
        100.0 / 3.0
    )
    assert row["short_exposure_pct"] == pytest.approx(
        100.0 / 3.0
    )
    assert row["neutral_exposure_pct"] == pytest.approx(
        100.0 / 3.0
    )


def test_annual_results_preserve_every_observation() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_annual_strategy_results,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-12-31 19:00:00+00:00",
                    "2024-12-31 19:15:00+00:00",
                    "2025-01-02 14:30:00+00:00",
                    "2025-01-02 14:45:00+00:00",
                ]
            ),
            "symbol": "SPY",
            "position": [0, 1, 1, 0],
            "position_eligible": [False, True, True, True],
            "turnover": [0.0, 1.0, 0.0, 1.0],
            "gross_strategy_return": [0.0, 0.01, 0.02, 0.0],
            "net_strategy_return": [0.0, 0.0099, 0.02, -0.0001],
        }
    )

    annual = build_annual_strategy_results(observations)

    assert annual["observations"].sum() == len(observations)
    assert annual["turnover"].sum() == pytest.approx(
        observations["turnover"].sum()
    )


def test_annual_consistency_summary_uses_strictly_positive_years() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_annual_consistency_summary,
    )

    annual = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "calendar_year": [2022, 2023, 2024, 2025],
            "net_return": [0.10, 0.00, -0.20, 0.30],
        }
    )

    summary = build_annual_consistency_summary(annual)
    row = summary.iloc[0]

    assert row["calendar_years"] == 4
    assert row["positive_net_years"] == 2
    assert row["worst_annual_net_return"] == pytest.approx(-0.20)
    assert row["median_annual_net_return"] == pytest.approx(0.05)
    assert row["positive_net_year_proportion"] == pytest.approx(0.50)
    assert row[
        "standard_deviation_annual_net_return"
    ] == pytest.approx(
        annual["net_return"].std(ddof=1)
    )


def test_annual_summary_functions_do_not_mutate_inputs() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_annual_consistency_summary,
        build_annual_strategy_results,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=3,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["spy", "spy", "spy"],
            "position": [0, 1, 0],
            "position_eligible": [False, True, True],
            "turnover": [0.0, 1.0, 1.0],
            "gross_strategy_return": [0.0, 0.01, 0.0],
            "net_strategy_return": [0.0, 0.0099, -0.0001],
        }
    )
    original_observations = observations.copy(deep=True)

    annual = build_annual_strategy_results(observations)
    original_annual = annual.copy(deep=True)

    build_annual_consistency_summary(annual)

    pd.testing.assert_frame_equal(
        observations,
        original_observations,
    )
    pd.testing.assert_frame_equal(
        annual,
        original_annual,
    )


def _build_synthetic_regime_inputs():
    import pandas as pd

    session_dates = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="D",
    )

    daily_volatility = pd.DataFrame(
        {
            "symbol": ["SPY"] * 5,
            "session_date": session_dates,
            "annualized_total_realized_volatility": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
            ],
        }
    )

    repeated_sessions = session_dates.repeat(2)

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=10,
                freq="15min",
                tz="UTC",
            ),
            "session_date": repeated_sessions,
            "symbol": ["SPY"] * 10,
            "position": [
                0,
                0,
                1,
                1,
                -1,
                -1,
                0,
                1,
                1,
                -1,
            ],
            "position_eligible": [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "turnover": [
                0.0,
                0.0,
                1.0,
                0.0,
                2.0,
                0.0,
                1.0,
                1.0,
                0.0,
                2.0,
            ],
            "gross_strategy_return": [
                0.0,
                0.0,
                0.01,
                0.005,
                -0.004,
                0.006,
                0.0,
                0.003,
                0.002,
                -0.001,
            ],
            "net_strategy_return": [
                0.0,
                0.0,
                0.0099,
                0.005,
                -0.0042,
                0.006,
                -0.0001,
                0.0029,
                0.002,
                -0.0012,
            ],
        }
    )

    return observations, daily_volatility


def test_regime_results_reuse_existing_day05_labels() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_volatility_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )

    results, definition = (
        build_volatility_regime_strategy_results(
            observations,
            daily_volatility,
            benchmark_symbol="SPY",
            stress_quantile=0.80,
        )
    )

    assert set(results["regime"]) == {
        "normal_volatility",
        "high_volatility",
    }
    assert definition.iloc[0]["benchmark_symbol"] == "SPY"
    assert definition.iloc[0]["stress_quantile"] == pytest.approx(
        0.80
    )
    assert definition.iloc[0]["volatility_threshold"] == (
        pytest.approx(0.42)
    )


def test_regime_results_preserve_all_observations_and_turnover() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_volatility_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )

    results, _ = build_volatility_regime_strategy_results(
        observations,
        daily_volatility,
    )

    assert results["observations"].sum() == len(observations)
    assert results["turnover"].sum() == pytest.approx(
        observations["turnover"].sum()
    )


def test_regime_results_reject_missing_session_labels() -> None:
    from systematic_alpha.analysis.dependence_diagnostics import (
        build_volatility_regimes,
    )
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )
    regimes, _ = build_volatility_regimes(daily_volatility)

    incomplete_regimes = regimes.iloc[:-1].copy()

    with pytest.raises(
        ValueError,
        match="No Day 5 volatility regime exists",
    ):
        build_regime_strategy_results(
            observations,
            incomplete_regimes,
        )


def test_regime_results_reject_duplicate_session_definitions() -> None:
    import pandas as pd

    from systematic_alpha.analysis.dependence_diagnostics import (
        build_volatility_regimes,
    )
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )
    regimes, _ = build_volatility_regimes(daily_volatility)

    duplicated = pd.concat(
        [regimes, regimes.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per session_date",
    ):
        build_regime_strategy_results(
            observations,
            duplicated,
        )


def test_regime_exposures_use_position_eligible_observations() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_volatility_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )

    results, _ = build_volatility_regime_strategy_results(
        observations,
        daily_volatility,
    )

    high_row = results.loc[
        results["regime"].eq("high_volatility")
    ].iloc[0]

    assert high_row["observations"] == 2
    assert high_row["position_eligible_observations"] == 2
    assert high_row["invested_exposure_pct"] == pytest.approx(
        100.0
    )
    assert high_row["long_exposure_pct"] == pytest.approx(
        50.0
    )
    assert high_row["short_exposure_pct"] == pytest.approx(
        50.0
    )
    assert high_row["neutral_exposure_pct"] == pytest.approx(
        0.0
    )


def test_regime_summary_functions_do_not_mutate_inputs() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_volatility_regime_strategy_results,
    )

    observations, daily_volatility = (
        _build_synthetic_regime_inputs()
    )
    original_observations = observations.copy(deep=True)
    original_daily_volatility = daily_volatility.copy(deep=True)

    build_volatility_regime_strategy_results(
        observations,
        daily_volatility,
    )

    pd.testing.assert_frame_equal(
        observations,
        original_observations,
    )
    pd.testing.assert_frame_equal(
        daily_volatility,
        original_daily_volatility,
    )


def test_forward_signal_sample_starts_with_t_plus_one_return() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_forward_signal_sample,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=4,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["SPY"] * 4,
            "close": [100.0, 120.0, 126.0, 151.2],
            "ma_price_ratio": [
                float("nan"),
                1.01,
                1.02,
                1.03,
            ],
        }
    )

    sample = build_forward_signal_sample(
        observations,
        horizon_bars=1,
    )

    first_row = sample.iloc[0]

    assert first_row["timestamp"] == observations.loc[1, "timestamp"]
    assert first_row["first_forward_timestamp"] == (
        observations.loc[2, "timestamp"]
    )
    assert first_row["forward_end_timestamp"] == (
        observations.loc[2, "timestamp"]
    )
    assert first_row["forward_return"] == pytest.approx(0.05)


def test_forward_signal_sample_uses_declared_horizon_end_price() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_forward_signal_sample,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=4,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["SPY"] * 4,
            "close": [100.0, 120.0, 126.0, 151.2],
            "ma_price_ratio": [1.00, 1.01, 1.02, 1.03],
        }
    )

    sample = build_forward_signal_sample(
        observations,
        horizon_bars=2,
    )

    first_row = sample.iloc[0]

    assert first_row["first_forward_timestamp"] == (
        observations.loc[1, "timestamp"]
    )
    assert first_row["forward_end_timestamp"] == (
        observations.loc[2, "timestamp"]
    )
    assert first_row["forward_return"] == pytest.approx(0.26)


def test_forward_signal_returns_do_not_cross_symbol_boundaries() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_forward_signal_sample,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 14:30:00+00:00",
                    "2025-01-02 14:30:00+00:00",
                    "2025-01-02 14:45:00+00:00",
                    "2025-01-02 14:45:00+00:00",
                ]
            ),
            "symbol": ["SPY", "QQQ", "SPY", "QQQ"],
            "close": [100.0, 200.0, 110.0, 180.0],
            "ma_price_ratio": [1.01, 0.99, 1.02, 0.98],
        }
    )

    sample = build_forward_signal_sample(
        observations,
        horizon_bars=1,
    )

    spy_return = sample.loc[
        sample["symbol"].eq("SPY"),
        "forward_return",
    ].iloc[0]
    qqq_return = sample.loc[
        sample["symbol"].eq("QQQ"),
        "forward_return",
    ].iloc[0]

    assert spy_return == pytest.approx(0.10)
    assert qqq_return == pytest.approx(-0.10)


def test_signal_information_coefficients_match_perfect_ordering() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_signal_validation_results,
    )

    returns = [
        0.01,
        0.02,
        0.03,
        0.04,
    ]
    close_values = [100.0]

    for return_value in returns:
        close_values.append(
            close_values[-1] * (1.0 + return_value)
        )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=5,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["SPY"] * 5,
            "close": close_values,
            "ma_price_ratio": [
                1.01,
                1.02,
                1.03,
                1.04,
                1.05,
            ],
        }
    )

    summary, _ = build_signal_validation_results(
        observations,
        horizons=(1,),
        signal_bucket_count=2,
    )

    row = summary.iloc[0]

    assert row[
        "pearson_information_coefficient"
    ] == pytest.approx(1.0)
    assert row[
        "spearman_information_coefficient"
    ] == pytest.approx(1.0)


def test_signal_bucket_means_are_monotonic_for_ordered_case() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_signal_validation_results,
    )

    one_bar_returns = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
    ]
    close_values = [100.0]

    for return_value in one_bar_returns:
        close_values.append(
            close_values[-1] * (1.0 + return_value)
        )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=11,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["SPY"] * 11,
            "close": close_values,
            "ma_price_ratio": [
                1.001 + index * 0.001
                for index in range(11)
            ],
        }
    )

    summary, buckets = build_signal_validation_results(
        observations,
        horizons=(1,),
        signal_bucket_count=5,
    )

    summary_row = summary.iloc[0]

    assert summary_row["actual_signal_buckets"] == 5
    assert summary_row[
        "bucket_mean_spearman_monotonicity"
    ] == pytest.approx(1.0)
    assert summary_row[
        "adjacent_increasing_bucket_proportion"
    ] == pytest.approx(1.0)

    assert buckets["signal_bucket"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert buckets["mean_forward_return"].is_monotonic_increasing


def test_signal_validation_rejects_invalid_horizon_grid() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_signal_validation_results,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=4,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["SPY"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
            "ma_price_ratio": [1.0, 1.01, 1.02, 1.03],
        }
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        build_signal_validation_results(
            observations,
            horizons=(4, 1),
        )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_signal_validation_results(
            observations,
            horizons=(1, 1),
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_signal_validation_results(
            observations,
            horizons=(0, 1),
        )


def test_signal_validation_does_not_mutate_input_observations() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_signal_validation_results,
    )

    observations = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=6,
                freq="15min",
                tz="UTC",
            ),
            "symbol": ["spy"] * 6,
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
            "ma_price_ratio": [
                float("nan"),
                1.001,
                1.002,
                1.003,
                1.004,
                1.005,
            ],
        }
    )
    original = observations.copy(deep=True)

    build_signal_validation_results(
        observations,
        horizons=(1, 4),
    )

    pd.testing.assert_frame_equal(
        observations,
        original,
    )


def _build_grid_orchestration_inputs():
    import numpy as np
    import pandas as pd

    session_dates = pd.date_range(
        "2025-01-02",
        periods=13,
        freq="B",
    )

    periods_per_session = 10
    total_observations = (
        len(session_dates) * periods_per_session
    )

    observation_number = np.arange(
        total_observations,
        dtype=float,
    )
    close = (
        100.0
        + 0.04 * observation_number
        + 1.25 * np.sin(observation_number / 4.0)
        + 0.50 * np.cos(observation_number / 9.0)
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-02 14:30",
                periods=total_observations,
                freq="15min",
                tz="UTC",
            ),
            "session_date": session_dates.repeat(
                periods_per_session
            ),
            "symbol": ["SPY"] * total_observations,
            "close": close,
        }
    )
    frame["close_to_close_simple_return"] = (
        frame["close"].pct_change()
    )

    daily_volatility = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(session_dates),
            "session_date": session_dates,
            "annualized_total_realized_volatility": (
                np.linspace(
                    0.10,
                    0.46,
                    len(session_dates),
                )
            ),
        }
    )

    return frame, daily_volatility


def _build_synthetic_parameter_results():
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        BREAK_EVEN_STATUS_NON_POSITIVE_GROSS,
        BREAK_EVEN_STATUS_ROOT_FOUND,
        build_parameter_grid,
    )

    records = []

    for index, configuration in enumerate(
        build_parameter_grid()
    ):
        root_found = index % 3 != 0

        records.append(
            {
                "configuration_id": (
                    configuration.configuration_id
                ),
                "short_window": configuration.short_window,
                "long_window": configuration.long_window,
                "neutral_band": configuration.neutral_band,
                "net_sharpe_ratio": index / 10.0,
                "net_cumulative_return": (
                    0.01
                    if index % 2 == 0
                    else -0.01
                ),
                "total_turnover": 100.0 + index,
                "break_even_status": (
                    BREAK_EVEN_STATUS_ROOT_FOUND
                    if root_found
                    else BREAK_EVEN_STATUS_NON_POSITIVE_GROSS
                ),
                "break_even_cost_bps": (
                    5.0
                    if root_found
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def test_grid_orchestration_runs_complete_frozen_grid() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        run_trend_ratio_sensitivity_grid,
    )

    frame, daily_volatility = (
        _build_grid_orchestration_inputs()
    )

    tables = run_trend_ratio_sensitivity_grid(
        frame,
        daily_volatility=daily_volatility,
    )

    assert len(tables.parameter_results) == 36
    assert (
        tables.parameter_results[
            "configuration_id"
        ].nunique()
        == 36
    )

    baseline = tables.parameter_results.loc[
        tables.parameter_results["short_window"].eq(8)
        & tables.parameter_results["long_window"].eq(32)
        & tables.parameter_results["neutral_band"].eq(0.001)
    ]
    assert len(baseline) == 1

    assert len(tables.annual_results) == 36
    assert len(tables.holding_diagnostics) == 36
    assert len(tables.signal_validation) == 36 * 4
    assert len(tables.regime_results) == 36 * 2
    assert len(tables.neighborhood_stability) == 36
    assert len(tables.regime_definition) == 1


def test_grid_outputs_are_compact_and_exclude_bar_level_rows() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_sensitivity_grid,
    )

    frame, daily_volatility = (
        _build_grid_orchestration_inputs()
    )

    tables = run_trend_ratio_sensitivity_grid(
        frame,
        configurations=(
            TrendRatioConfiguration(
                short_window=8,
                long_window=32,
                neutral_band=0.001,
            ),
        ),
        daily_volatility=daily_volatility,
    )

    compact_tables = (
        tables.parameter_results,
        tables.annual_results,
        tables.regime_results,
        tables.holding_diagnostics,
        tables.signal_validation,
        tables.signal_buckets,
        tables.neighborhood_stability,
    )

    for table in compact_tables:
        assert "gross_strategy_return" not in table.columns
        assert "net_strategy_return" not in table.columns
        assert "position" not in table.columns


def test_grid_orchestration_allows_regime_analysis_to_be_omitted() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_sensitivity_grid,
    )

    frame, _ = _build_grid_orchestration_inputs()

    tables = run_trend_ratio_sensitivity_grid(
        frame,
        configurations=(
            TrendRatioConfiguration(
                short_window=8,
                long_window=32,
                neutral_band=0.001,
            ),
        ),
    )

    assert tables.regime_results.empty
    assert tables.regime_definition.empty
    assert len(tables.parameter_results) == 1


def test_grid_orchestration_does_not_mutate_inputs() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_sensitivity_grid,
    )

    frame, daily_volatility = (
        _build_grid_orchestration_inputs()
    )
    original_frame = frame.copy(deep=True)
    original_daily_volatility = (
        daily_volatility.copy(deep=True)
    )

    run_trend_ratio_sensitivity_grid(
        frame,
        configurations=(
            TrendRatioConfiguration(
                short_window=8,
                long_window=32,
                neutral_band=0.001,
            ),
        ),
        daily_volatility=daily_volatility,
    )

    pd.testing.assert_frame_equal(
        frame,
        original_frame,
    )
    pd.testing.assert_frame_equal(
        daily_volatility,
        original_daily_volatility,
    )


def test_neighborhood_stability_has_expected_corner_and_interior_counts(
) -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_neighborhood_stability,
    )

    parameter_results = (
        _build_synthetic_parameter_results()
    )
    stability = build_neighborhood_stability(
        parameter_results
    )

    corner = stability.loc[
        stability["configuration_id"].eq(
            "s004_l032_d0p0000"
        )
    ].iloc[0]
    interior = stability.loc[
        stability["configuration_id"].eq(
            "s008_l064_d0p0005"
        )
    ].iloc[0]

    assert corner["neighbor_count"] == 3
    assert interior["neighbor_count"] == 6


def test_neighborhood_stability_exposes_isolated_sharpe_spike() -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_neighborhood_stability,
    )

    parameter_results = (
        _build_synthetic_parameter_results()
    )
    target_id = "s008_l064_d0p0010"

    parameter_results.loc[
        parameter_results["configuration_id"].eq(target_id),
        "net_sharpe_ratio",
    ] = 5.0

    stability = build_neighborhood_stability(
        parameter_results
    )
    row = stability.loc[
        stability["configuration_id"].eq(target_id)
    ].iloc[0]

    expected_neighbor_median = (
        parameter_results.set_index("configuration_id")
        .loc[
            row["neighbor_configuration_ids"].split("|"),
            "net_sharpe_ratio",
        ]
        .median()
    )

    assert row["median_neighbor_net_sharpe"] == pytest.approx(
        expected_neighbor_median
    )
    assert row[
        "mean_absolute_one_step_net_sharpe_difference"
    ] > 1.0


def test_neighborhood_stability_is_deterministic() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_neighborhood_stability,
    )

    parameter_results = (
        _build_synthetic_parameter_results()
    )

    first = build_neighborhood_stability(
        parameter_results
    )
    second = build_neighborhood_stability(
        parameter_results.sample(
            frac=1.0,
            random_state=17,
        )
    )

    pd.testing.assert_frame_equal(first, second)


def test_neighborhood_stability_rejects_duplicate_configurations() -> None:
    import pandas as pd

    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        build_neighborhood_stability,
    )

    parameter_results = (
        _build_synthetic_parameter_results()
    )
    duplicated = pd.concat(
        [
            parameter_results,
            parameter_results.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate configuration_id",
    ):
        build_neighborhood_stability(duplicated)


def test_grid_orchestration_rejects_configuration_outside_frozen_grid(
) -> None:
    from systematic_alpha.analysis.trend_ratio_sensitivity import (
        TrendRatioConfiguration,
        run_trend_ratio_sensitivity_grid,
    )

    frame, _ = _build_grid_orchestration_inputs()

    with pytest.raises(
        ValueError,
        match="frozen Day 7 parameter grid",
    ):
        run_trend_ratio_sensitivity_grid(
            frame,
            configurations=(
                TrendRatioConfiguration(
                    short_window=2,
                    long_window=32,
                    neutral_band=0.001,
                ),
            ),
        )
