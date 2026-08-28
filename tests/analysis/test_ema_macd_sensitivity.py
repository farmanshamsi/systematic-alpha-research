"""Synthetic tests for Day 9 EMA/MACD sensitivity foundations."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.ema_macd_sensitivity import (
    BASELINE_FAST_WINDOW,
    BASELINE_NEUTRAL_BAND,
    BASELINE_SIGNAL_WINDOW,
    BASELINE_SLOW_WINDOW,
    EXPECTED_CONFIGURATION_COUNT,
    EmaMacdConfiguration,
    EmaMacdSensitivityError,
    build_ema_macd_neighborhood_stability,
    build_ema_macd_parameter_grid,
    build_neighborhood_map,
    calculate_filter_diagnostics,
    configuration_to_parameters,
    configurations_are_neighbors,
    run_ema_macd_configuration,
    run_ema_macd_sensitivity_grid,
)
from systematic_alpha.analysis.ema_macd_baseline import (
    analyse_ema_macd_baseline,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    calculate_cost_break_even,
    calculate_holding_diagnostics,
)
from systematic_alpha.strategies.ema_macd import (
    EmaMacdParameters,
)


def test_frozen_grid_contains_108_unique_configurations() -> None:
    first = build_ema_macd_parameter_grid()
    second = build_ema_macd_parameter_grid()

    assert first == second
    assert len(first) == EXPECTED_CONFIGURATION_COUNT
    assert len(
        {
            configuration.configuration_id
            for configuration in first
        }
    ) == EXPECTED_CONFIGURATION_COUNT

    assert all(
        configuration.fast_window
        < configuration.slow_window
        for configuration in first
    )


def test_day08_baseline_is_included_exactly_once() -> None:
    grid = build_ema_macd_parameter_grid()

    matches = [
        configuration
        for configuration in grid
        if (
            configuration.fast_window
            == BASELINE_FAST_WINDOW
            and configuration.slow_window
            == BASELINE_SLOW_WINDOW
            and configuration.signal_window
            == BASELINE_SIGNAL_WINDOW
            and configuration.neutral_band
            == BASELINE_NEUTRAL_BAND
        )
    ]

    assert len(matches) == 1
    assert matches[0].configuration_id == (
        "f012_s026_m009_b0p00050"
    )


def test_invalid_declared_axes_are_rejected_not_filtered() -> None:
    with pytest.raises(
        EmaMacdSensitivityError,
        match="Every declared fast window",
    ):
        build_ema_macd_parameter_grid(
            fast_windows=(8, 20),
            slow_windows=(20, 26),
        )

    with pytest.raises(
        EmaMacdSensitivityError,
        match="duplicates",
    ):
        build_ema_macd_parameter_grid(
            signal_windows=(6, 9, 9),
        )

    with pytest.raises(
        EmaMacdSensitivityError,
        match="strictly increasing",
    ):
        build_ema_macd_parameter_grid(
            neutral_bands=(0.0, 0.001, 0.0005),
        )


def test_filter_diagnostics_reuse_declared_ema_formulas() -> None:
    configuration = EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )

    diagnostics = calculate_filter_diagnostics(
        configuration
    )

    assert diagnostics.fast_alpha == pytest.approx(
        2.0 / 13.0
    )
    assert diagnostics.slow_alpha == pytest.approx(
        2.0 / 27.0
    )
    assert diagnostics.signal_alpha == pytest.approx(
        2.0 / 10.0
    )

    assert (
        diagnostics.slow_minus_fast_half_life_bars
        == pytest.approx(
            diagnostics.slow_half_life_bars
            - diagnostics.fast_half_life_bars
        )
    )


def test_neighbors_are_axis_adjacent_and_symmetric() -> None:
    centre = EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )
    fast_neighbor = EmaMacdConfiguration(
        fast_window=8,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )
    slow_neighbor = EmaMacdConfiguration(
        fast_window=12,
        slow_window=32,
        signal_window=9,
        neutral_band=0.0005,
    )
    signal_neighbor = EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=12,
        neutral_band=0.0005,
    )
    band_neighbor = EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.001,
    )
    diagonal = EmaMacdConfiguration(
        fast_window=8,
        slow_window=32,
        signal_window=9,
        neutral_band=0.0005,
    )

    for neighbor in (
        fast_neighbor,
        slow_neighbor,
        signal_neighbor,
        band_neighbor,
    ):
        assert configurations_are_neighbors(
            centre,
            neighbor,
        )
        assert configurations_are_neighbors(
            neighbor,
            centre,
        )

    assert not configurations_are_neighbors(
        centre,
        diagonal,
    )
    assert not configurations_are_neighbors(
        centre,
        centre,
    )


def test_corner_and_interior_neighbor_counts_are_exact() -> None:
    neighborhoods = build_neighborhood_map()

    corner = EmaMacdConfiguration(
        fast_window=8,
        slow_window=20,
        signal_window=6,
        neutral_band=0.0,
    )
    interior = EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )

    assert len(
        neighborhoods[corner.configuration_id]
    ) == 4
    assert len(
        neighborhoods[interior.configuration_id]
    ) == 8


def test_neighborhood_map_is_deterministic_and_symmetric() -> None:
    first = build_neighborhood_map()
    second = build_neighborhood_map()

    assert first == second

    for configuration_id, neighbor_ids in first.items():
        assert neighbor_ids == tuple(sorted(neighbor_ids))

        for neighbor_id in neighbor_ids:
            assert configuration_id in first[neighbor_id]


def test_configuration_validation_rejects_bad_values() -> None:
    with pytest.raises(
        EmaMacdSensitivityError,
        match="smaller than",
    ):
        EmaMacdConfiguration(
            fast_window=26,
            slow_window=26,
            signal_window=9,
            neutral_band=0.0005,
        )

    with pytest.raises(
        EmaMacdSensitivityError,
        match="strictly positive",
    ):
        EmaMacdConfiguration(
            fast_window=12,
            slow_window=26,
            signal_window=0,
            neutral_band=0.0005,
        )

    with pytest.raises(
        EmaMacdSensitivityError,
        match="non-negative",
    ):
        EmaMacdConfiguration(
            fast_window=12,
            slow_window=26,
            signal_window=9,
            neutral_band=-0.0005,
        )


def make_oscillating_frame(
    *,
    observations: int = 320,
    start: str = "2024-01-02 14:30:00+00:00",
) -> pd.DataFrame:
    """Create deterministic oscillating synthetic prices."""

    timestamps = pd.date_range(
        start,
        periods=observations,
        freq="15min",
    )
    primary_phase = np.linspace(
        0.0,
        20.0 * np.pi,
        observations,
    )
    secondary_phase = np.linspace(
        0.0,
        50.0 * np.pi,
        observations,
    )
    close = (
        100.0
        + 4.0 * np.sin(primary_phase)
        + 0.4 * np.sin(secondary_phase)
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "session_date": timestamps.date,
            "close": close,
        }
    )
    frame["close_to_close_simple_return"] = (
        frame["close"].pct_change()
    )

    return frame


def make_year_boundary_trend_frame() -> pd.DataFrame:
    """Create a persistent trend crossing a calendar-year boundary."""

    observations = 200
    timestamps = pd.date_range(
        "2024-12-31 04:00:00+00:00",
        periods=observations,
        freq="15min",
    )
    close = 100.0 * np.power(
        1.001,
        np.arange(observations),
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "session_date": timestamps.date,
            "close": close,
        }
    )
    frame["close_to_close_simple_return"] = (
        frame["close"].pct_change()
    )

    return frame


def baseline_configuration() -> EmaMacdConfiguration:
    """Return the exact Day 8 baseline configuration."""

    return EmaMacdConfiguration(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )


def responsive_configuration() -> EmaMacdConfiguration:
    """Return a zero-band configuration for timing tests."""

    return EmaMacdConfiguration(
        fast_window=8,
        slow_window=20,
        signal_window=6,
        neutral_band=0.0,
    )


def test_configuration_maps_exactly_to_day08_parameters() -> None:
    parameters = configuration_to_parameters(
        baseline_configuration()
    )

    assert parameters == EmaMacdParameters(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
        cost_bps_per_turnover=1.0,
        price_column="close",
        return_column="close_to_close_simple_return",
    )


def test_baseline_run_matches_day08_analysis_engine() -> None:
    frame = make_oscillating_frame()

    sensitivity_run = run_ema_macd_configuration(
        frame,
        configuration=baseline_configuration(),
    )
    baseline = analyse_ema_macd_baseline(
        frame,
        parameters=EmaMacdParameters(),
    )

    pd.testing.assert_frame_equal(
        sensitivity_run.strategy_bundle.observations,
        baseline.strategy_bundle.observations,
    )
    pd.testing.assert_frame_equal(
        sensitivity_run.strategy_bundle.diagnostics,
        baseline.strategy_bundle.diagnostics,
    )

    gross_row = baseline.performance_summary.loc[
        baseline.performance_summary["series"].eq(
            "ema_macd_gross"
        )
    ].iloc[0]
    net_row = baseline.performance_summary.loc[
        baseline.performance_summary["series"].eq(
            "ema_macd_net"
        )
    ].iloc[0]

    assert (
        sensitivity_run.gross_performance.cumulative_return
        == pytest.approx(gross_row["cumulative_return"])
    )
    assert (
        sensitivity_run.gross_performance.sharpe_ratio
        == pytest.approx(
            gross_row["sharpe_ratio"],
            nan_ok=True,
        )
    )
    assert (
        sensitivity_run.net_performance.cumulative_return
        == pytest.approx(net_row["cumulative_return"])
    )
    assert (
        sensitivity_run.net_performance.sharpe_ratio
        == pytest.approx(
            net_row["sharpe_ratio"],
            nan_ok=True,
        )
    )

    baseline_holding = (
        baseline.holding_diagnostics.iloc[0]
    )

    for name, value in asdict(
        sensitivity_run.holding_diagnostics
    ).items():
        assert baseline_holding[name] == pytest.approx(
            value,
            nan_ok=True,
        )

    baseline_break_even = (
        baseline.cost_break_even.iloc[0]
    )

    assert (
        sensitivity_run.cost_break_even.status
        == baseline_break_even["status"]
    )

    if (
        sensitivity_run.cost_break_even.break_even_cost_bps
        is None
    ):
        assert pd.isna(
            baseline_break_even["break_even_cost_bps"]
        )
    else:
        assert baseline_break_even[
            "break_even_cost_bps"
        ] == pytest.approx(
            sensitivity_run.cost_break_even
            .break_even_cost_bps
        )


def test_configuration_run_does_not_mutate_input() -> None:
    frame = make_oscillating_frame()
    original = frame.copy(deep=True)

    run_ema_macd_configuration(
        frame,
        configuration=responsive_configuration(),
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )


def test_future_prices_do_not_change_past_results() -> None:
    frame = make_oscillating_frame()
    cutoff = 210

    altered = frame.copy(deep=True)
    future_length = len(altered) - cutoff

    altered.loc[
        cutoff:,
        "close",
    ] = (
        altered.loc[cutoff:, "close"].to_numpy()
        * np.linspace(
            1.10,
            1.40,
            future_length,
        )
    )
    altered["close_to_close_simple_return"] = (
        altered["close"].pct_change()
    )

    original_run = run_ema_macd_configuration(
        frame,
        configuration=responsive_configuration(),
    )
    altered_run = run_ema_macd_configuration(
        altered,
        configuration=responsive_configuration(),
    )

    comparison_columns = [
        "fast_ema",
        "slow_ema",
        "macd",
        "macd_signal_line",
        "macd_histogram",
        "normalized_macd_histogram",
        "signal",
        "position",
        "turnover",
        "gross_strategy_return",
        "net_strategy_return",
    ]

    pd.testing.assert_frame_equal(
        original_run.strategy_bundle.observations.loc[
            : cutoff - 1,
            comparison_columns,
        ],
        altered_run.strategy_bundle.observations.loc[
            : cutoff - 1,
            comparison_columns,
        ],
    )


def test_all_positions_are_previous_bar_signals() -> None:
    run = run_ema_macd_configuration(
        make_oscillating_frame(),
        configuration=responsive_configuration(),
    )
    observations = run.strategy_bundle.observations

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


def test_direct_reversals_retain_turnover_two() -> None:
    run = run_ema_macd_configuration(
        make_oscillating_frame(),
        configuration=responsive_configuration(),
    )
    observations = run.strategy_bundle.observations

    previous_position = (
        observations["position"]
        .shift(1, fill_value=0)
    )
    direct_reversal = (
        observations["position"].ne(0)
        & previous_position.ne(0)
        & observations["position"].eq(
            -previous_position
        )
    )

    assert direct_reversal.any()
    assert observations.loc[
        direct_reversal,
        "turnover",
    ].eq(2.0).all()


def test_holding_diagnostics_reuse_existing_definition() -> None:
    run = run_ema_macd_configuration(
        make_oscillating_frame(),
        configuration=responsive_configuration(),
    )
    observations = run.strategy_bundle.observations

    expected = calculate_holding_diagnostics(
        observations["position"],
        session_labels=observations["session_date"],
    )

    assert run.holding_diagnostics == expected
    assert (
        run.holding_diagnostics.non_zero_episode_count
        > 0
    )
    assert (
        run.holding_diagnostics.whipsaw_count
        >= 0
    )


def test_cost_break_even_reuses_existing_compounded_solver() -> None:
    run = run_ema_macd_configuration(
        make_oscillating_frame(),
        configuration=responsive_configuration(),
    )
    observations = run.strategy_bundle.observations

    expected = calculate_cost_break_even(
        observations["gross_strategy_return"],
        observations["turnover"],
    )

    assert run.cost_break_even.status == expected.status
    assert (
        run.cost_break_even.eligible_observations
        == expected.eligible_observations
    )
    assert (
        run.cost_break_even.gross_cumulative_return
        == pytest.approx(
            expected.gross_cumulative_return
        )
    )
    assert (
        run.cost_break_even.total_turnover
        == pytest.approx(expected.total_turnover)
    )

    if expected.break_even_cost_bps is None:
        assert (
            run.cost_break_even.break_even_cost_bps
            is None
        )
    else:
        assert (
            run.cost_break_even.break_even_cost_bps
            == pytest.approx(
                expected.break_even_cost_bps
            )
        )


def test_annual_results_preserve_full_sample_positions() -> None:
    run = run_ema_macd_configuration(
        make_year_boundary_trend_frame(),
        configuration=responsive_configuration(),
    )
    observations = run.strategy_bundle.observations
    annual = run.annual_results
    consistency = run.annual_consistency

    assert set(annual["calendar_year"]) == {
        2024,
        2025,
    }
    assert int(annual["observations"].sum()) == len(
        observations
    )
    assert float(annual["turnover"].sum()) == pytest.approx(
        observations["turnover"].sum()
    )

    year_values = observations["timestamp"].dt.year
    first_2025_index = year_values[
        year_values.eq(2025)
    ].index[0]

    assert first_2025_index > 0
    assert (
        observations.loc[
            first_2025_index,
            "position",
        ]
        == observations.loc[
            first_2025_index - 1,
            "position",
        ]
        == 1
    )
    assert observations.loc[
        first_2025_index,
        "turnover",
    ] == pytest.approx(0.0)

    assert len(consistency) == 1
    assert consistency.iloc[0]["calendar_years"] == 2


def adjacent_configurations() -> tuple[
    EmaMacdConfiguration,
    EmaMacdConfiguration,
]:
    """Return two axis-adjacent frozen configurations."""

    return (
        EmaMacdConfiguration(
            fast_window=8,
            slow_window=26,
            signal_window=9,
            neutral_band=0.0005,
        ),
        baseline_configuration(),
    )


def test_subset_grid_returns_only_compact_tables() -> None:
    frame = make_oscillating_frame()
    original = frame.copy(deep=True)

    tables = run_ema_macd_sensitivity_grid(
        frame,
        configurations=adjacent_configurations(),
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )

    assert set(
        tables.__dataclass_fields__
    ) == {
        "parameter_results",
        "annual_results",
        "annual_consistency",
        "regime_results",
        "regime_definition",
        "holding_diagnostics",
        "signal_validation",
        "signal_buckets",
        "neighborhood_stability",
    }

    assert len(tables.parameter_results) == 2
    assert len(tables.holding_diagnostics) == 2
    assert len(tables.annual_consistency) == 2
    assert len(tables.signal_validation) == 8
    assert len(tables.signal_buckets) == 40
    assert len(tables.neighborhood_stability) == 2

    assert tables.regime_results.empty
    assert tables.regime_definition.empty

    assert tables.neighborhood_stability[
        "neighbor_count"
    ].tolist() == [1, 1]

    for table_name in (
        "parameter_results",
        "annual_results",
        "annual_consistency",
        "holding_diagnostics",
        "signal_validation",
        "signal_buckets",
        "neighborhood_stability",
    ):
        table = getattr(tables, table_name)

        assert "timestamp" not in table.columns
        assert (
            "gross_strategy_return"
            not in table.columns
        )
        assert (
            "net_strategy_return"
            not in table.columns
        )
        assert "position" not in table.columns


def test_subset_grid_is_fully_deterministic() -> None:
    frame = make_oscillating_frame()
    configurations = adjacent_configurations()

    first = run_ema_macd_sensitivity_grid(
        frame,
        configurations=configurations,
    )
    second = run_ema_macd_sensitivity_grid(
        frame,
        configurations=configurations,
    )

    for table_name in first.__dataclass_fields__:
        pd.testing.assert_frame_equal(
            getattr(first, table_name),
            getattr(second, table_name),
        )


def test_baseline_parameter_row_matches_single_run() -> None:
    frame = make_oscillating_frame()
    configuration = baseline_configuration()

    grid = run_ema_macd_sensitivity_grid(
        frame,
        configurations=(configuration,),
    )
    single = run_ema_macd_configuration(
        frame,
        configuration=configuration,
    )

    row = grid.parameter_results.iloc[0]
    observations = (
        single.strategy_bundle.observations
    )

    assert row["configuration_id"] == (
        configuration.configuration_id
    )
    assert row["observations"] == len(
        observations
    )
    assert row[
        "gross_cumulative_return"
    ] == pytest.approx(
        single.gross_performance.cumulative_return
    )
    assert row[
        "net_cumulative_return"
    ] == pytest.approx(
        single.net_performance.cumulative_return
    )
    assert row[
        "net_sharpe_ratio"
    ] == pytest.approx(
        single.net_performance.sharpe_ratio,
        nan_ok=True,
    )
    assert row[
        "total_turnover"
    ] == pytest.approx(
        observations["turnover"].sum()
    )
    assert row[
        "break_even_status"
    ] == single.cost_break_even.status


def test_grid_rejects_configuration_outside_frozen_axes() -> None:
    frame = make_oscillating_frame()
    invalid = EmaMacdConfiguration(
        fast_window=10,
        slow_window=26,
        signal_window=9,
        neutral_band=0.0005,
    )

    with pytest.raises(
        EmaMacdSensitivityError,
        match="frozen Day 9",
    ):
        run_ema_macd_sensitivity_grid(
            frame,
            configurations=(invalid,),
        )


def test_neighborhood_statistics_use_only_axis_neighbors() -> None:
    configurations = (
        EmaMacdConfiguration(
            fast_window=8,
            slow_window=26,
            signal_window=9,
            neutral_band=0.0005,
        ),
        EmaMacdConfiguration(
            fast_window=12,
            slow_window=26,
            signal_window=9,
            neutral_band=0.0005,
        ),
        EmaMacdConfiguration(
            fast_window=16,
            slow_window=26,
            signal_window=9,
            neutral_band=0.0005,
        ),
    )

    parameter_results = pd.DataFrame(
        {
            "configuration_id": [
                configuration.configuration_id
                for configuration in configurations
            ],
            "fast_window": [8, 12, 16],
            "slow_window": [26, 26, 26],
            "signal_window": [9, 9, 9],
            "neutral_band": [
                0.0005,
                0.0005,
                0.0005,
            ],
            "net_sharpe_ratio": [
                0.10,
                0.50,
                0.30,
            ],
            "net_cumulative_return": [
                0.02,
                0.04,
                -0.01,
            ],
            "total_turnover": [
                100.0,
                110.0,
                120.0,
            ],
            "break_even_status": [
                "root_found",
                "root_found",
                "non_positive_gross",
            ],
            "break_even_cost_bps": [
                1.2,
                1.8,
                np.nan,
            ],
        }
    )

    stability = (
        build_ema_macd_neighborhood_stability(
            parameter_results
        )
    )
    centre = stability.loc[
        stability["fast_window"].eq(12)
    ].iloc[0]

    assert centre["neighbor_count"] == 2
    assert centre[
        "median_neighbor_net_sharpe"
    ] == pytest.approx(0.20)
    assert centre[
        "minimum_neighbor_net_sharpe"
    ] == pytest.approx(0.10)
    assert centre[
        "standard_deviation_neighbor_net_sharpe"
    ] == pytest.approx(0.10)
    assert centre[
        "proportion_neighbors_positive_net_return"
    ] == pytest.approx(0.50)
    assert centre[
        "proportion_neighbors_positive_break_even"
    ] == pytest.approx(0.50)
    assert centre[
        "median_neighbor_turnover"
    ] == pytest.approx(110.0)
    assert centre[
        "mean_absolute_one_step_net_sharpe_difference"
    ] == pytest.approx(0.30)
    assert centre[
        "mean_absolute_one_step_turnover_difference"
    ] == pytest.approx(10.0)


def test_optional_regime_tables_preserve_each_run() -> None:
    frame = make_oscillating_frame()
    session_dates = (
        pd.Series(frame["session_date"])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    daily_volatility = pd.DataFrame(
        {
            "symbol": "SPY",
            "session_date": session_dates,
            "annualized_total_realized_volatility": (
                np.linspace(
                    0.10,
                    0.40,
                    len(session_dates),
                )
            ),
        }
    )

    tables = run_ema_macd_sensitivity_grid(
        frame,
        configurations=adjacent_configurations(),
        daily_volatility=daily_volatility,
    )

    assert len(tables.regime_definition) == 1
    assert set(
        tables.regime_results["regime"]
    ) == {
        "normal_volatility",
        "high_volatility",
    }

    for configuration_id, group in (
        tables.regime_results.groupby(
            "configuration_id",
            observed=True,
            sort=True,
        )
    ):
        assert group[
            "observations"
        ].sum() == len(frame)

        expected_turnover = (
            tables.parameter_results.loc[
                tables.parameter_results[
                    "configuration_id"
                ].eq(configuration_id),
                "total_turnover",
            ].iloc[0]
        )

        assert group[
            "turnover"
        ].sum() == pytest.approx(
            expected_turnover
        )
