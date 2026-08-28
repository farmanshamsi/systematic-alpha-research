"""Synthetic tests for the compact Day 9 artifact writer."""

from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from systematic_alpha.analysis.day09_ema_macd_sensitivity_report import (
    APPROVED_DAY09_ARTIFACT_FILENAMES,
    Day09ReportError,
    build_day09_metadata,
    write_day09_ema_macd_sensitivity_artifacts,
)
from systematic_alpha.analysis.ema_macd_baseline import (
    DAY08_FORWARD_HORIZONS,
    EMA_MACD_SIGNAL_BUCKET_COLUMNS,
    EMA_MACD_SIGNAL_SUMMARY_COLUMNS,
)
from systematic_alpha.analysis.ema_macd_sensitivity import (
    CONFIGURATION_PARAMETER_COLUMNS,
    EMA_MACD_CONFIGURATION_RESULT_COLUMNS,
    EMA_MACD_HOLDING_RESULT_COLUMNS,
    EmaMacdSensitivityTables,
    build_ema_macd_neighborhood_stability,
    build_ema_macd_parameter_grid,
    calculate_filter_diagnostics,
)
from systematic_alpha.analysis.trend_ratio_sensitivity import (
    ANNUAL_CONSISTENCY_COLUMNS,
    ANNUAL_RESULT_COLUMNS,
    BREAK_EVEN_STATUS_NON_POSITIVE_GROSS,
    BREAK_EVEN_STATUS_ROOT_FOUND,
    REGIME_RESULT_COLUMNS,
)


def _configured_record(
    configuration,
) -> dict[str, object]:
    return {
        "configuration_id": (
            configuration.configuration_id
        ),
        "fast_window": (
            configuration.fast_window
        ),
        "slow_window": (
            configuration.slow_window
        ),
        "signal_window": (
            configuration.signal_window
        ),
        "neutral_band": (
            configuration.neutral_band
        ),
    }


def _blank_record(
    columns,
) -> dict[str, object]:
    return {
        column: 0.0
        for column in columns
    }


def _build_synthetic_tables() -> (
    EmaMacdSensitivityTables
):
    parameter_records = []
    annual_records = []
    annual_consistency_records = []
    regime_records = []
    holding_records = []
    signal_records = []
    signal_bucket_records = []

    for index, configuration in enumerate(
        build_ema_macd_parameter_grid()
    ):
        prefix = _configured_record(
            configuration
        )
        filters = calculate_filter_diagnostics(
            configuration
        )

        net_return = (
            0.02
            if index % 2 == 0
            else -0.01
        )
        root_found = index % 3 != 0

        parameter_records.append(
            {
                "symbol": "SPY",
                **prefix,
                "fast_alpha": (
                    filters.fast_alpha
                ),
                "slow_alpha": (
                    filters.slow_alpha
                ),
                "signal_alpha": (
                    filters.signal_alpha
                ),
                "fast_half_life_bars": (
                    filters.fast_half_life_bars
                ),
                "slow_half_life_bars": (
                    filters.slow_half_life_bars
                ),
                "signal_half_life_bars": (
                    filters.signal_half_life_bars
                ),
                "slow_minus_fast_half_life_bars": (
                    filters
                    .slow_minus_fast_half_life_bars
                ),
                "observations": 1_000,
                "position_eligible_observations": (
                    950
                ),
                "gross_cumulative_return": (
                    net_return + 0.03
                ),
                "gross_sharpe_ratio": (
                    index / 100.0
                ),
                "net_cumulative_return": (
                    net_return
                ),
                "net_sharpe_ratio": (
                    index / 120.0
                ),
                "net_max_drawdown": -0.10,
                "total_turnover": (
                    100.0 + index
                ),
                "position_changing_bars": (
                    50 + index
                ),
                "long_exposure_pct": 40.0,
                "short_exposure_pct": 35.0,
                "neutral_exposure_pct": 25.0,
                "break_even_status": (
                    BREAK_EVEN_STATUS_ROOT_FOUND
                    if root_found
                    else (
                        BREAK_EVEN_STATUS_NON_POSITIVE_GROSS
                    )
                ),
                "break_even_cost_bps": (
                    1.5 + index / 100.0
                    if root_found
                    else float("nan")
                ),
                "calendar_years": 2,
                "positive_net_years": (
                    2 if net_return > 0.0 else 0
                ),
                "worst_annual_net_return": (
                    net_return - 0.01
                ),
                "median_annual_net_return": (
                    net_return
                ),
                "standard_deviation_annual_net_return": (
                    0.02
                ),
                "positive_net_year_proportion": (
                    1.0
                    if net_return > 0.0
                    else 0.0
                ),
            }
        )

        holding_records.append(
            {
                "symbol": "SPY",
                **prefix,
                "eligible_observations": 950,
                "non_zero_episode_count": 40,
                "long_episode_count": 22,
                "short_episode_count": 18,
                "median_holding_duration_bars": (
                    6.0
                ),
                "mean_holding_duration_bars": (
                    7.0
                ),
                "holding_duration_25th_percentile_bars": (
                    3.0
                ),
                "holding_duration_75th_percentile_bars": (
                    10.0
                ),
                "maximum_holding_duration_bars": (
                    30
                ),
                "overnight_carry_episode_count": (
                    5
                ),
                "session_crossing_episode_proportion": (
                    0.125
                ),
                "whipsaw_count": 8,
                "whipsaw_rate_per_1000_eligible_observations": (
                    8.421
                ),
                "whipsaw_episode_proportion": (
                    0.20
                ),
            }
        )

        for year in (2024, 2025):
            annual = _blank_record(
                ANNUAL_RESULT_COLUMNS
            )
            annual.update(prefix)

            if "symbol" in annual:
                annual["symbol"] = "SPY"
            if "calendar_year" in annual:
                annual["calendar_year"] = year
            if "observations" in annual:
                annual["observations"] = 500
            if "net_cumulative_return" in annual:
                annual[
                    "net_cumulative_return"
                ] = net_return

            annual_records.append(annual)

        annual_consistency = _blank_record(
            ANNUAL_CONSISTENCY_COLUMNS
        )
        annual_consistency.update(prefix)

        if "symbol" in annual_consistency:
            annual_consistency["symbol"] = "SPY"
        if "calendar_years" in annual_consistency:
            annual_consistency[
                "calendar_years"
            ] = 2

        annual_consistency_records.append(
            annual_consistency
        )

        for regime_name in (
            "normal_volatility",
            "high_volatility",
        ):
            regime = _blank_record(
                REGIME_RESULT_COLUMNS
            )
            regime.update(prefix)

            if "symbol" in regime:
                regime["symbol"] = "SPY"
            if "regime" in regime:
                regime["regime"] = regime_name
            if "observations" in regime:
                regime["observations"] = 500

            regime_records.append(regime)

        for horizon in DAY08_FORWARD_HORIZONS:
            summary = _blank_record(
                EMA_MACD_SIGNAL_SUMMARY_COLUMNS
            )
            summary.update(prefix)

            if "horizon_bars" in summary:
                summary["horizon_bars"] = horizon
            if "observations" in summary:
                summary["observations"] = 900
            if (
                "actual_signal_buckets"
                in summary
            ):
                summary[
                    "actual_signal_buckets"
                ] = 5

            signal_records.append(summary)

            for bucket in range(1, 6):
                bucket_record = _blank_record(
                    EMA_MACD_SIGNAL_BUCKET_COLUMNS
                )
                bucket_record.update(prefix)

                if (
                    "horizon_bars"
                    in bucket_record
                ):
                    bucket_record[
                        "horizon_bars"
                    ] = horizon
                if (
                    "signal_bucket"
                    in bucket_record
                ):
                    bucket_record[
                        "signal_bucket"
                    ] = bucket
                if (
                    "observations"
                    in bucket_record
                ):
                    bucket_record[
                        "observations"
                    ] = 180

                signal_bucket_records.append(
                    bucket_record
                )

    parameter_results = (
        pd.DataFrame.from_records(
            parameter_records,
            columns=(
                EMA_MACD_CONFIGURATION_RESULT_COLUMNS
            ),
        )
    )

    annual_results = pd.DataFrame.from_records(
        annual_records,
        columns=(
            *CONFIGURATION_PARAMETER_COLUMNS,
            *ANNUAL_RESULT_COLUMNS,
        ),
    )

    annual_consistency = (
        pd.DataFrame.from_records(
            annual_consistency_records,
            columns=(
                *CONFIGURATION_PARAMETER_COLUMNS,
                *ANNUAL_CONSISTENCY_COLUMNS,
            ),
        )
    )

    regime_results = pd.DataFrame.from_records(
        regime_records,
        columns=(
            *CONFIGURATION_PARAMETER_COLUMNS,
            *REGIME_RESULT_COLUMNS,
        ),
    )

    holding_diagnostics = (
        pd.DataFrame.from_records(
            holding_records,
            columns=(
                EMA_MACD_HOLDING_RESULT_COLUMNS
            ),
        )
    )

    signal_validation = (
        pd.DataFrame.from_records(
            signal_records,
            columns=(
                *CONFIGURATION_PARAMETER_COLUMNS,
                *EMA_MACD_SIGNAL_SUMMARY_COLUMNS,
            ),
        )
    )

    signal_buckets = pd.DataFrame.from_records(
        signal_bucket_records,
        columns=(
            *CONFIGURATION_PARAMETER_COLUMNS,
            *EMA_MACD_SIGNAL_BUCKET_COLUMNS,
        ),
    )

    neighborhood_stability = (
        build_ema_macd_neighborhood_stability(
            parameter_results
        )
    )

    regime_definition = pd.DataFrame(
        {
            "benchmark_symbol": ["SPY"],
            "stress_quantile": [0.80],
            "volatility_threshold": [0.30],
            "normal_regime_label": [
                "normal_volatility"
            ],
            "stress_regime_label": [
                "high_volatility"
            ],
        }
    )

    return EmaMacdSensitivityTables(
        parameter_results=parameter_results,
        annual_results=annual_results,
        annual_consistency=annual_consistency,
        regime_results=regime_results,
        regime_definition=regime_definition,
        holding_diagnostics=holding_diagnostics,
        signal_validation=signal_validation,
        signal_buckets=signal_buckets,
        neighborhood_stability=(
            neighborhood_stability
        ),
    )


def _build_metadata(
    tables: EmaMacdSensitivityTables,
):
    return build_day09_metadata(
        permitted_dataset_identifier=(
            "spy_15m_development_v1"
        ),
        dataset_manifest_sha256="a" * 64,
        regime_definition=(
            tables.regime_definition
        ),
    )


def test_writer_creates_exact_approved_set(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    written = (
        write_day09_ema_macd_sensitivity_artifacts(
            tables,
            metadata=_build_metadata(tables),
            output_directory=tmp_path,
        )
    )

    assert {
        path.name
        for path in written
    } == APPROVED_DAY09_ARTIFACT_FILENAMES

    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.is_file()
    } == APPROVED_DAY09_ARTIFACT_FILENAMES

    assert not list(
        tmp_path.glob("*.parquet")
    )


def test_writer_creates_nonempty_figures(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day09_ema_macd_sensitivity_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    figure_names = (
        "net_sharpe_slices.png",
        "net_return_slices.png",
        "turnover_slices.png",
        "cost_break_even_slices.png",
        "stability_slices.png",
    )

    for filename in figure_names:
        path = tmp_path / filename

        assert path.exists()
        assert path.stat().st_size > 1_000
        assert path.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )


def test_signal_artifact_combines_summary_and_buckets(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day09_ema_macd_sensitivity_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    signal = pd.read_csv(
        tmp_path / "signal_validation.csv"
    )

    assert set(signal["record_type"]) == {
        "horizon_summary",
        "signal_bucket",
    }

    assert (
        signal["record_type"]
        .eq("horizon_summary")
        .sum()
        == 108 * 4
    )

    assert (
        signal["record_type"]
        .eq("signal_bucket")
        .sum()
        == 108 * 4 * 5
    )


def test_metadata_preserves_frozen_scope() -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)

    assert (
        metadata["development_sample_start"]
        == "2020-01-02"
    )
    assert (
        metadata["development_sample_end"]
        == "2025-12-31"
    )
    assert (
        metadata["locked_period_accessed"]
        is False
    )
    assert (
        metadata[
            "parameter_selected_using_locked_period"
        ]
        is False
    )
    assert (
        metadata[
            "full_bar_level_artifacts_written"
        ]
        is False
    )
    assert metadata["configuration_count"] == 108
    assert metadata["fast_window_grid"] == [
        8,
        12,
        16,
    ]
    assert metadata["slow_window_grid"] == [
        20,
        26,
        32,
    ]
    assert metadata["signal_window_grid"] == [
        6,
        9,
        12,
    ]
    assert metadata["neutral_band_grid"] == [
        0.0,
        0.00025,
        0.0005,
        0.001,
    ]


def test_writer_rejects_locked_period_access(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()
    metadata = _build_metadata(tables)
    metadata["locked_period_accessed"] = True

    with pytest.raises(
        Day09ReportError,
        match="locked_period_accessed must be false",
    ):
        write_day09_ema_macd_sensitivity_artifacts(
            tables,
            metadata=metadata,
            output_directory=tmp_path,
        )


def test_writer_rejects_unapproved_existing_files(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    (
        tmp_path / "full_bar_output.csv"
    ).write_text(
        "forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(
        Day09ReportError,
        match="unapproved files",
    ):
        write_day09_ema_macd_sensitivity_artifacts(
            tables,
            metadata=_build_metadata(tables),
            output_directory=tmp_path,
        )


def test_writer_rejects_observation_level_columns(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    invalid = replace(
        tables,
        parameter_results=(
            tables.parameter_results.assign(
                timestamp="2025-01-01"
            )
        ),
    )

    with pytest.raises(
        Day09ReportError,
        match="prohibited observation-level columns",
    ):
        write_day09_ema_macd_sensitivity_artifacts(
            invalid,
            metadata=_build_metadata(tables),
            output_directory=tmp_path,
        )


def test_writer_does_not_mutate_tables(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    originals = {
        name: getattr(
            tables,
            name,
        ).copy(deep=True)
        for name in tables.__dataclass_fields__
    }

    write_day09_ema_macd_sensitivity_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    for name, original in originals.items():
        pd.testing.assert_frame_equal(
            getattr(tables, name),
            original,
        )


def test_findings_and_metadata_are_conservative(
    tmp_path,
) -> None:
    tables = _build_synthetic_tables()

    write_day09_ema_macd_sensitivity_artifacts(
        tables,
        metadata=_build_metadata(tables),
        output_directory=tmp_path,
    )

    findings = (
        tmp_path / "findings.md"
    ).read_text(
        encoding="utf-8"
    ).lower()

    metadata_text = (
        tmp_path / "metadata.json"
    ).read_text(
        encoding="utf-8"
    )

    for phrase in (
        "parameter sensitivity, not parameter",
        "do not establish alpha",
        "locked final period",
        "no parameter was selected",
    ):
        assert phrase in findings

    assert "/Users/" not in findings
    assert "/Users/" not in metadata_text

    payload = json.loads(metadata_text)

    assert (
        payload["locked_period_accessed"]
        is False
    )
