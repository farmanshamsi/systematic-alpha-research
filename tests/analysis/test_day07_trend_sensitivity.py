"""Synthetic tests for the compact Day 7 artifact writer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from systematic_alpha.analysis.day07_trend_sensitivity import (
    APPROVED_DAY07_ARTIFACT_FILENAMES,
    build_day07_metadata,
    write_day07_artifacts,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    ANNUAL_CONSISTENCY_COLUMNS,
    ANNUAL_RESULT_COLUMNS,
    BREAK_EVEN_STATUS_NON_POSITIVE_GROSS,
    BREAK_EVEN_STATUS_ROOT_FOUND,
    CONFIGURATION_RESULT_COLUMNS,
    HOLDING_RESULT_COLUMNS,
    REGIME_RESULT_COLUMNS,
    SIGNAL_BUCKET_RESULT_COLUMNS,
    SIGNAL_VALIDATION_SUMMARY_COLUMNS,
    TrendRatioSensitivityTables,
    build_neighborhood_stability,
    build_parameter_grid,
)


def _configured_record(
    configuration,
) -> dict[str, object]:
    return {
        "configuration_id": configuration.configuration_id,
        "short_window": configuration.short_window,
        "long_window": configuration.long_window,
        "neutral_band": configuration.neutral_band,
    }


def _build_synthetic_tables() -> TrendRatioSensitivityTables:
    parameter_records = []
    annual_records = []
    annual_consistency_records = []
    regime_records = []
    holding_records = []
    signal_records = []
    signal_bucket_records = []

    for index, configuration in enumerate(
        build_parameter_grid()
    ):
        prefix = _configured_record(configuration)
        net_return = 0.01 if index % 2 == 0 else -0.01
        root_found = index % 3 != 0

        parameter_records.append(
            {
                "symbol": "SPY",
                **prefix,
                "short_filter_lag_bars": (
                    configuration.short_window - 1
                )
                / 2.0,
                "long_filter_lag_bars": (
                    configuration.long_window - 1
                )
                / 2.0,
                "lag_spread_bars": (
                    configuration.long_window
                    - configuration.short_window
                )
                / 2.0,
                "observations": 100,
                "position_eligible_observations": 96,
                "gross_cumulative_return": net_return + 0.02,
                "gross_sharpe_ratio": index / 20.0,
                "net_cumulative_return": net_return,
                "net_sharpe_ratio": index / 25.0,
                "net_max_drawdown": -0.10,
                "total_turnover": 100.0 + index,
                "position_changing_bars": 20 + index,
                "long_exposure_pct": 40.0,
                "short_exposure_pct": 35.0,
                "neutral_exposure_pct": 25.0,
                "break_even_status": (
                    BREAK_EVEN_STATUS_ROOT_FOUND
                    if root_found
                    else BREAK_EVEN_STATUS_NON_POSITIVE_GROSS
                ),
                "break_even_cost_bps": (
                    5.0 + index / 10.0
                    if root_found
                    else float("nan")
                ),
                "calendar_years": 6,
                "positive_net_years": 3,
                "worst_annual_net_return": -0.08,
                "median_annual_net_return": 0.01,
                "standard_deviation_annual_net_return": 0.05,
                "positive_net_year_proportion": 0.50,
            }
        )

        annual_records.append(
            {
                **prefix,
                "symbol": "SPY",
                "calendar_year": 2025,
                "observations": 100,
                "position_eligible_observations": 96,
                "gross_return": 0.03,
                "net_return": net_return,
                "net_annualized_volatility": 0.15,
                "net_sharpe_ratio": index / 25.0,
                "net_max_drawdown": -0.10,
                "turnover": 100.0 + index,
                "long_exposure_pct": 40.0,
                "short_exposure_pct": 35.0,
                "neutral_exposure_pct": 25.0,
            }
        )

        annual_consistency_records.append(
            {
                **prefix,
                "symbol": "SPY",
                "calendar_years": 6,
                "positive_net_years": 3,
                "worst_annual_net_return": -0.08,
                "median_annual_net_return": 0.01,
                "standard_deviation_annual_net_return": 0.05,
                "positive_net_year_proportion": 0.50,
            }
        )

        for regime in (
            "normal_volatility",
            "high_volatility",
        ):
            regime_records.append(
                {
                    **prefix,
                    "symbol": "SPY",
                    "regime": regime,
                    "observations": 50,
                    "position_eligible_observations": 48,
                    "net_return": net_return / 2.0,
                    "net_annualized_volatility": 0.15,
                    "net_sharpe_ratio": index / 30.0,
                    "invested_exposure_pct": 75.0,
                    "long_exposure_pct": 40.0,
                    "short_exposure_pct": 35.0,
                    "neutral_exposure_pct": 25.0,
                    "turnover": (100.0 + index) / 2.0,
                }
            )

        holding_records.append(
            {
                "symbol": "SPY",
                **prefix,
                "eligible_observations": 96,
                "non_zero_episode_count": 20,
                "long_episode_count": 10,
                "short_episode_count": 10,
                "median_holding_duration_bars": 4.0,
                "mean_holding_duration_bars": 5.0,
                "holding_duration_25th_percentile_bars": 2.0,
                "holding_duration_75th_percentile_bars": 7.0,
                "maximum_holding_duration_bars": 20.0,
                "overnight_carry_episode_count": 3,
                "session_crossing_episode_proportion": 0.15,
                "whipsaw_count": 4,
                "whipsaw_rate_per_1000_eligible_observations": 41.67,
                "whipsaw_episode_proportion": 0.20,
            }
        )

        for horizon in (1, 4, 8, 16):
            signal_records.append(
                {
                    **prefix,
                    "symbol": "SPY",
                    "horizon_bars": horizon,
                    "observations": 80,
                    "pearson_information_coefficient": 0.02,
                    "spearman_information_coefficient": 0.03,
                    "requested_signal_buckets": 5,
                    "actual_signal_buckets": 5,
                    "bucket_mean_spearman_monotonicity": 0.50,
                    "adjacent_increasing_bucket_proportion": 0.75,
                }
            )

            for bucket in range(1, 6):
                signal_bucket_records.append(
                    {
                        **prefix,
                        "symbol": "SPY",
                        "horizon_bars": horizon,
                        "signal_bucket": bucket,
                        "observations": 16,
                        "signal_minimum": -0.01,
                        "signal_maximum": 0.01,
                        "signal_mean": (
                            bucket - 3
                        )
                        * 0.002,
                        "mean_forward_return": (
                            bucket - 3
                        )
                        * 0.0005,
                        "median_forward_return": (
                            bucket - 3
                        )
                        * 0.0004,
                    }
                )

    parameter_results = pd.DataFrame.from_records(
        parameter_records,
        columns=CONFIGURATION_RESULT_COLUMNS,
    )
    annual_results = pd.DataFrame.from_records(
        annual_records,
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *ANNUAL_RESULT_COLUMNS,
        ),
    )
    annual_consistency = pd.DataFrame.from_records(
        annual_consistency_records,
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *ANNUAL_CONSISTENCY_COLUMNS,
        ),
    )
    regime_results = pd.DataFrame.from_records(
        regime_records,
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *REGIME_RESULT_COLUMNS,
        ),
    )
    holding_diagnostics = pd.DataFrame.from_records(
        holding_records,
        columns=HOLDING_RESULT_COLUMNS,
    )
    signal_validation = pd.DataFrame.from_records(
        signal_records,
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *SIGNAL_VALIDATION_SUMMARY_COLUMNS,
        ),
    )
    signal_buckets = pd.DataFrame.from_records(
        signal_bucket_records,
        columns=(
            "configuration_id",
            "short_window",
            "long_window",
            "neutral_band",
            *SIGNAL_BUCKET_RESULT_COLUMNS,
        ),
    )
    neighborhood_stability = build_neighborhood_stability(
        parameter_results
    )

    regime_definition = pd.DataFrame(
        {
            "benchmark_symbol": ["SPY"],
            "stress_quantile": [0.80],
            "volatility_threshold": [0.30],
            "normal_regime_label": ["normal_volatility"],
            "stress_regime_label": ["high_volatility"],
        }
    )

    return TrendRatioSensitivityTables(
        parameter_results=parameter_results,
        annual_results=annual_results,
        annual_consistency=annual_consistency,
        regime_results=regime_results,
        regime_definition=regime_definition,
        holding_diagnostics=holding_diagnostics,
        signal_validation=signal_validation,
        signal_buckets=signal_buckets,
        neighborhood_stability=neighborhood_stability,
    )


def _build_metadata(
    tables: TrendRatioSensitivityTables,
):
    return build_day07_metadata(
        permitted_dataset_identifier="spy_15m_development_v1",
        dataset_manifest_sha256="a" * 64,
        regime_definition=tables.regime_definition,
    )


def test_writer_creates_only_approved_compact_outputs(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)

    written = write_day07_artifacts(
        tables,
        metadata=metadata,
        output_directory=tmp_path,
    )

    assert {
        path.name
        for path in written
    } == APPROVED_DAY07_ARTIFACT_FILENAMES

    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.is_file()
    } == APPROVED_DAY07_ARTIFACT_FILENAMES

    assert not list(tmp_path.glob("*.parquet"))
    assert not any(
        "bar" in path.name.lower()
        for path in tmp_path.iterdir()
    )


def test_writer_creates_nonempty_figures(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day07_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    figure_names = (
        "net_sharpe_surface.png",
        "net_return_surface.png",
        "turnover_surface.png",
        "cost_break_even_surface.png",
        "stability_surface.png",
    )

    for filename in figure_names:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 100


def test_signal_artifact_contains_summary_and_bucket_records(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day07_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    signal_artifact = pd.read_csv(
        tmp_path / "signal_validation.csv"
    )

    assert set(signal_artifact["record_type"]) == {
        "horizon_summary",
        "signal_bucket",
    }


def test_writer_rejects_locked_period_access(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)
    metadata["locked_period_accessed"] = True

    with pytest.raises(
        ValueError,
        match="locked_period_accessed must be false",
    ):
        write_day07_artifacts(
            tables,
            metadata=metadata,
            output_directory=tmp_path,
        )


def test_writer_rejects_absolute_local_paths(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)
    metadata["notes"] = (
        "/Users/example/private/market-data.parquet"
    )

    with pytest.raises(
        ValueError,
        match="absolute local paths",
    ):
        write_day07_artifacts(
            tables,
            metadata=metadata,
            output_directory=tmp_path,
        )


def test_writer_rejects_unapproved_existing_files(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()
    (tmp_path / "full_bar_output.csv").write_text(
        "forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unapproved files",
    ):
        write_day07_artifacts(
            tables,
            metadata=_build_metadata(tables),
            output_directory=tmp_path,
        )


def test_writer_does_not_mutate_input_tables(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    originals = {
        name: getattr(tables, name).copy(deep=True)
        for name in (
            "parameter_results",
            "annual_results",
            "annual_consistency",
            "regime_results",
            "regime_definition",
            "holding_diagnostics",
            "signal_validation",
            "signal_buckets",
            "neighborhood_stability",
        )
    }

    write_day07_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    for name, original in originals.items():
        pd.testing.assert_frame_equal(
            getattr(tables, name),
            original,
        )


def test_metadata_confirms_frozen_development_scope() -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)

    assert metadata["development_sample_start"] == "2020-01-02"
    assert metadata["development_sample_end"] == "2025-12-31"
    assert metadata["locked_period_accessed"] is False
    assert (
        metadata["parameter_selected_using_locked_period"]
        is False
    )
    assert metadata["short_window_grid"] == [4, 8, 16]
    assert metadata["long_window_grid"] == [32, 64, 96]
    assert metadata["neutral_band_grid"] == [
        0.0,
        0.0005,
        0.001,
        0.002,
    ]


def test_written_text_artifacts_contain_no_absolute_paths(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day07_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    metadata_text = (
        tmp_path / "metadata.json"
    ).read_text(encoding="utf-8")
    findings_text = (
        tmp_path / "findings.md"
    ).read_text(encoding="utf-8")

    assert "/Users/" not in metadata_text
    assert "/Users/" not in findings_text

    payload = json.loads(metadata_text)
    assert payload["locked_period_accessed"] is False
