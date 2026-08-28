"""Tests for the Day 6 trend-ratio baseline report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.day06_trend_baseline import (
    ANNUALIZATION_FACTOR,
    ARTIFACT_FILENAMES,
    Day06TrendBaselineError,
    build_annual_exposure,
    build_baseline_summary,
    write_trend_baseline_artifacts,
)
from systematic_alpha.analysis.strategy_performance import (
    calculate_performance_metrics,
)


def make_observations() -> pd.DataFrame:
    """Create compact synthetic baseline observations."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-12-30T14:30:00Z",
                    "2024-12-30T14:45:00Z",
                    "2024-12-30T15:00:00Z",
                    "2024-12-30T15:15:00Z",
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T14:45:00Z",
                    "2025-01-02T15:00:00Z",
                    "2025-01-02T15:15:00Z",
                ],
                utc=True,
            ),
            "close_to_close_simple_return": [
                np.nan,
                -0.02,
                0.03,
                0.01,
                -0.01,
                0.02,
                0.01,
                -0.015,
            ],
            "gross_strategy_return": [
                0.0,
                0.01,
                -0.02,
                0.0,
                0.02,
                -0.01,
                0.015,
                -0.005,
            ],
            "net_strategy_return": [
                0.0,
                0.0098,
                -0.0202,
                -0.0001,
                0.0199,
                -0.0101,
                0.0148,
                -0.0051,
            ],
            "position": [
                0,
                1,
                -1,
                0,
                1,
                1,
                -1,
                0,
            ],
            "position_eligible": [
                False,
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
                1.0,
                2.0,
                0.0,
                1.0,
                0.0,
                2.0,
                0.0,
            ],
        }
    )


def make_diagnostics() -> pd.DataFrame:
    """Create one existing-style trend diagnostics row."""
    return pd.DataFrame(
        {
            "symbol": ["SPY"],
            "observations": [8],
            "signal_warmup_observations": [1],
            "position_warmup_observations": [1],
            "long_signals": [3],
            "short_signals": [2],
            "neutral_signals": [2],
            "long_exposure_pct": [
                100.0 * 3.0 / 7.0
            ],
            "short_exposure_pct": [
                100.0 * 2.0 / 7.0
            ],
            "flat_exposure_pct": [
                100.0 * 2.0 / 7.0
            ],
            "total_turnover": [6.0],
            "position_changes": [4],
        }
    )


def make_metadata() -> dict[str, object]:
    """Create portable fixed-baseline metadata."""
    return {
        "artifact_version": (
            "trend_ratio_baseline_v1"
        ),
        "dataset_identifier": (
            "synthetic_development_only"
        ),
        "dataset_manifest_sha256": "a" * 64,
        "symbol": "SPY",
        "frequency": "15-minute",
        "sample_start": (
            "2024-12-30T14:30:00+00:00"
        ),
        "sample_end": (
            "2025-01-02T15:15:00+00:00"
        ),
        "observations": 8,
        "development_window_start": "2020-01-02",
        "development_window_end": "2025-12-31",
        "locked_period_accessed": False,
        "short_window": 8,
        "long_window": 32,
        "neutral_band": 0.001,
        "price_column": "close",
        "return_column": (
            "close_to_close_simple_return"
        ),
        "transaction_cost_convention": (
            "One basis point per unit of turnover."
        ),
        "cost_bps_per_turnover": 1.0,
        "annualization_factor": (
            ANNUALIZATION_FACTOR
        ),
        "risk_free_rate_convention": "zero",
        "signal_timing": (
            "Signal uses prices through bar t."
        ),
        "position_timing": (
            "Position uses the prior signal."
        ),
        "overnight_position_rule": (
            "Positions may carry overnight."
        ),
        "session_boundary_rule": (
            "Rolling averages do not reset."
        ),
        "signal_warmup_observations": 31,
        "position_warmup_observations": 32,
    }


def test_summary_has_required_order() -> None:
    summary = build_baseline_summary(
        make_observations()
    )

    assert summary["series"].tolist() == [
        "spy_buy_and_hold",
        "trend_ratio_gross",
        "trend_ratio_net",
    ]


def test_gross_and_net_use_their_return_columns() -> None:
    observations = make_observations()
    summary = build_baseline_summary(
        observations
    ).set_index("series")

    expected_gross = calculate_performance_metrics(
        observations["gross_strategy_return"],
        annualization_factor=(
            ANNUALIZATION_FACTOR
        ),
    )
    expected_net = calculate_performance_metrics(
        observations["net_strategy_return"],
        annualization_factor=(
            ANNUALIZATION_FACTOR
        ),
    )

    assert np.isclose(
        summary.loc[
            "trend_ratio_gross",
            "cumulative_return",
        ],
        expected_gross.cumulative_return,
    )
    assert np.isclose(
        summary.loc[
            "trend_ratio_net",
            "cumulative_return",
        ],
        expected_net.cumulative_return,
    )


def test_annual_exposure_uses_eligible_positions() -> None:
    exposure = build_annual_exposure(
        make_observations()
    ).set_index("year")

    assert np.isclose(
        exposure.loc[
            2024,
            "long_exposure_pct",
        ],
        100.0 / 3.0,
    )
    assert np.isclose(
        exposure.loc[
            2024,
            "short_exposure_pct",
        ],
        100.0 / 3.0,
    )
    assert np.isclose(
        exposure.loc[
            2024,
            "flat_exposure_pct",
        ],
        100.0 / 3.0,
    )

    assert np.isclose(
        exposure.loc[
            2025,
            "long_exposure_pct",
        ],
        50.0,
    )


def test_annual_exposure_percentages_sum_to_100() -> None:
    exposure = build_annual_exposure(
        make_observations()
    )

    totals = exposure[
        [
            "long_exposure_pct",
            "short_exposure_pct",
            "flat_exposure_pct",
        ]
    ].sum(axis=1)

    np.testing.assert_allclose(
        totals,
        100.0,
    )


def test_annual_turnover_and_changes_are_correct() -> None:
    exposure = build_annual_exposure(
        make_observations()
    ).set_index("year")

    assert exposure.loc[
        2024,
        "total_turnover",
    ] == 3.0
    assert exposure.loc[
        2025,
        "total_turnover",
    ] == 3.0
    assert exposure.loc[
        2024,
        "position_changes",
    ] == 2
    assert exposure.loc[
        2025,
        "position_changes",
    ] == 2


def test_artifact_writer_creates_seven_nonempty_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "baseline"

    paths = write_trend_baseline_artifacts(
        output_dir=output_dir,
        observations=make_observations(),
        diagnostics=make_diagnostics(),
        metadata=make_metadata(),
    )

    assert tuple(paths) == ARTIFACT_FILENAMES

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    png_files = sorted(
        output_dir.glob("*.png")
    )

    assert len(png_files) == 2
    assert all(
        path.stat().st_size > 1_000
        for path in png_files
    )


def test_metadata_confirms_locked_period_not_accessed(
    tmp_path: Path,
) -> None:
    paths = write_trend_baseline_artifacts(
        output_dir=tmp_path,
        observations=make_observations(),
        diagnostics=make_diagnostics(),
        metadata=make_metadata(),
    )

    metadata = json.loads(
        paths["metadata.json"].read_text(
            encoding="utf-8"
        )
    )

    assert metadata[
        "locked_period_accessed"
    ] is False


def test_artifacts_do_not_contain_output_absolute_path(
    tmp_path: Path,
) -> None:
    paths = write_trend_baseline_artifacts(
        output_dir=tmp_path,
        observations=make_observations(),
        diagnostics=make_diagnostics(),
        metadata=make_metadata(),
    )

    for filename in (
        "metadata.json",
        "findings.md",
    ):
        content = paths[filename].read_text(
            encoding="utf-8"
        )
        assert str(tmp_path) not in content


def test_no_bar_level_or_parquet_output_is_written(
    tmp_path: Path,
) -> None:
    write_trend_baseline_artifacts(
        output_dir=tmp_path,
        observations=make_observations(),
        diagnostics=make_diagnostics(),
        metadata=make_metadata(),
    )

    assert not list(
        tmp_path.rglob("*.parquet")
    )

    csv_names = {
        path.name
        for path in tmp_path.glob("*.csv")
    }

    assert csv_names == {
        "summary.csv",
        "signal_diagnostics.csv",
        "annual_exposure.csv",
    }


def test_input_frames_are_not_mutated(
    tmp_path: Path,
) -> None:
    observations = make_observations()
    diagnostics = make_diagnostics()
    original_observations = observations.copy(
        deep=True
    )
    original_diagnostics = diagnostics.copy(
        deep=True
    )

    write_trend_baseline_artifacts(
        output_dir=tmp_path,
        observations=observations,
        diagnostics=diagnostics,
        metadata=make_metadata(),
    )

    pd.testing.assert_frame_equal(
        observations,
        original_observations,
    )
    pd.testing.assert_frame_equal(
        diagnostics,
        original_diagnostics,
    )


@pytest.mark.parametrize(
    ("builder", "column"),
    [
        (
            build_baseline_summary,
            "net_strategy_return",
        ),
        (
            build_annual_exposure,
            "position_eligible",
        ),
    ],
)
def test_missing_required_columns_fail_clearly(
    builder: object,
    column: str,
) -> None:
    observations = make_observations().drop(
        columns=[column]
    )

    with pytest.raises(
        Day06TrendBaselineError,
        match="missing required columns",
    ):
        builder(observations)


def test_empty_eligible_position_sample_fails_clearly() -> None:
    observations = make_observations()
    observations["position_eligible"] = False

    with pytest.raises(
        Day06TrendBaselineError,
        match="at least one eligible",
    ):
        build_annual_exposure(observations)


def test_existing_outputs_are_replaced(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in ARTIFACT_FILENAMES:
        (
            tmp_path / filename
        ).write_bytes(b"stale")

    paths = write_trend_baseline_artifacts(
        output_dir=tmp_path,
        observations=make_observations(),
        diagnostics=make_diagnostics(),
        metadata=make_metadata(),
    )

    for path in paths.values():
        assert path.read_bytes() != b"stale"
