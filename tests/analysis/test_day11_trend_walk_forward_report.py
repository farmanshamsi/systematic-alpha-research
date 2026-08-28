"""Contract tests for deterministic Day 11 reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from systematic_alpha.analysis.day11_trend_walk_forward_report import (
    APPROVED_DAY11_ARTIFACT_NAMES,
    DAY11_ARTIFACT_VERSION,
    Day11DatasetAudit,
    Day11ReportError,
    build_day11_report,
    write_day11_artifacts,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    AGGREGATE_RESULT_COLUMNS,
    CONFIGURATION_IDS,
    FOLD_RESULT_COLUMNS,
    WALK_FORWARD_STRATEGIES,
    TrendFamilyWalkForwardResults,
)


FIXED_GENERATION_TIMESTAMP = (
    "2026-07-26T12:00:00Z"
)
FIXED_GIT_COMMIT = "a" * 40


def make_results(
    *,
    undefined_sharpe: bool = False,
) -> TrendFamilyWalkForwardResults:
    """Create deterministic compact Day 11 result tables."""

    fold_records: list[
        dict[str, object]
    ] = []

    for strategy_index, strategy in enumerate(
        WALK_FORWARD_STRATEGIES
    ):
        for fold_index, test_year in enumerate(
            range(2022, 2026)
        ):
            test_observations = (
                26 + fold_index
            )
            cumulative_return = (
                -0.04
                + 0.02 * fold_index
                - 0.01 * strategy_index
            )
            fold_records.append(
                {
                    "strategy": strategy,
                    "symbol": "SPY",
                    "frequency": "15min",
                    "fold_id": (
                        f"wf_{test_year}"
                    ),
                    "configuration_id": (
                        CONFIGURATION_IDS[
                            strategy
                        ]
                    ),
                    "train_start_timestamp": (
                        pd.Timestamp(
                            "2020-01-02 "
                            "14:30:00+00:00"
                        )
                    ),
                    "train_end_timestamp": (
                        pd.Timestamp(
                            f"{test_year - 1}"
                            "-12-31 20:45:00+00:00"
                        )
                    ),
                    "test_start_timestamp": (
                        pd.Timestamp(
                            f"{test_year}"
                            "-01-03 14:30:00+00:00"
                        )
                    ),
                    "test_end_timestamp": (
                        pd.Timestamp(
                            f"{test_year}"
                            "-12-30 20:45:00+00:00"
                        )
                    ),
                    "train_sessions": (
                        500 + 250 * fold_index
                    ),
                    "test_sessions": 1,
                    "train_observations": (
                        13_000
                        + 6_500 * fold_index
                    ),
                    "test_observations": (
                        test_observations
                    ),
                    "annualization_factor": (
                        252.0
                        * test_observations
                    ),
                    "purge_sessions": 0,
                    "embargo_sessions": 0,
                    "indicator_history_observations": (
                        13_000
                        + 6_500 * fold_index
                    ),
                    "initial_test_position": 0,
                    "initial_test_turnover": 0.0,
                    "warmup_observations": 2,
                    "active_observations": (
                        test_observations - 2
                    ),
                    "cumulative_return": (
                        cumulative_return
                    ),
                    "annualized_return": (
                        cumulative_return
                    ),
                    "annualized_volatility": (
                        0.10
                        + 0.01 * fold_index
                    ),
                    "sharpe_ratio": (
                        float("nan")
                        if undefined_sharpe
                        else (
                            -0.4
                            + 0.2 * fold_index
                            - 0.1
                            * strategy_index
                        )
                    ),
                    "maximum_drawdown": (
                        -0.08
                        - 0.01 * fold_index
                    ),
                    "turnover": (
                        10.0 + fold_index
                    ),
                    "average_exposure": 75.0,
                    "long_exposure": 40.0,
                    "short_exposure": 35.0,
                    "flat_exposure": 25.0,
                    "trade_count": (
                        8 + fold_index
                    ),
                }
            )

    folds = pd.DataFrame.from_records(
        fold_records,
        columns=FOLD_RESULT_COLUMNS,
    )
    aggregate_records: list[
        dict[str, object]
    ] = []

    for strategy_index, strategy in enumerate(
        WALK_FORWARD_STRATEGIES
    ):
        selected = folds.loc[
            folds["strategy"].eq(strategy)
        ]
        aggregate_records.append(
            {
                "strategy": strategy,
                "symbol": "SPY",
                "frequency": "15min",
                "configuration_id": (
                    CONFIGURATION_IDS[
                        strategy
                    ]
                ),
                "folds": 4,
                "test_start_timestamp": (
                    pd.Timestamp(
                        "2022-01-03 "
                        "14:30:00+00:00"
                    )
                ),
                "test_end_timestamp": (
                    pd.Timestamp(
                        "2025-12-30 "
                        "20:45:00+00:00"
                    )
                ),
                "test_sessions": int(
                    selected[
                        "test_sessions"
                    ].sum()
                ),
                "test_observations": int(
                    selected[
                        "test_observations"
                    ].sum()
                ),
                "annualization_factor": (
                    252.0
                    * selected[
                        "test_observations"
                    ].sum()
                    / selected[
                        "test_sessions"
                    ].sum()
                ),
                "cumulative_return": (
                    -0.02
                    - 0.03 * strategy_index
                ),
                "annualized_return": (
                    -0.01
                    - 0.02 * strategy_index
                ),
                "annualized_volatility": (
                    0.12
                ),
                "sharpe_ratio": (
                    float("nan")
                    if undefined_sharpe
                    else (
                        -0.1
                        - 0.2 * strategy_index
                    )
                ),
                "maximum_drawdown": -0.15,
                "turnover": float(
                    selected[
                        "turnover"
                    ].sum()
                ),
                "average_exposure": 75.0,
                "long_exposure": 40.0,
                "short_exposure": 35.0,
                "flat_exposure": 25.0,
                "trade_count": int(
                    selected[
                        "trade_count"
                    ].sum()
                ),
            }
        )

    aggregate = pd.DataFrame.from_records(
        aggregate_records,
        columns=AGGREGATE_RESULT_COLUMNS,
    )

    return TrendFamilyWalkForwardResults(
        fold_results=folds,
        aggregate_results=aggregate,
    )


def make_audit() -> Day11DatasetAudit:
    """Create canonical-shaped synthetic lineage."""

    return Day11DatasetAudit(
        dataset_id=(
            "spy_qqq_iwm_15min_"
            "2020-01-02_2025-12-31_"
            "sip_v3_development_canonical"
        ),
        dataset_path=(
            "data/processed/bars/"
            "development.parquet"
        ),
        manifest_sha256="b" * 64,
        canonical_row_count=117_192,
        spy_row_count=39_064,
        spy_session_count=1_508,
        minimum_timestamp=pd.Timestamp(
            "2020-01-02 14:30:00+00:00"
        ),
        maximum_timestamp=pd.Timestamp(
            "2025-12-31 20:45:00+00:00"
        ),
    )


def write_compact(
    tmp_path: Path,
    *,
    name: str = "artifacts",
    overwrite: bool = False,
    undefined_sharpe: bool = False,
):
    """Write one fixed-provenance synthetic artifact set."""

    return write_day11_artifacts(
        make_results(
            undefined_sharpe=(
                undefined_sharpe
            )
        ),
        make_audit(),
        artifact_directory=(
            tmp_path / name
        ),
        overwrite=overwrite,
        generation_timestamp=(
            FIXED_GENERATION_TIMESTAMP
        ),
        source_git_commit=FIXED_GIT_COMMIT,
    )


def test_writes_exact_stable_artifact_schemas(
    tmp_path: Path,
) -> None:
    result = write_compact(tmp_path)
    actual = {
        path.name
        for path in (
            result.artifact_directory
            .iterdir()
        )
        if path.is_file()
    }

    assert actual == set(
        APPROVED_DAY11_ARTIFACT_NAMES
    )
    fold_csv = pd.read_csv(
        result.artifact_directory
        / "fold_results.csv"
    )
    aggregate_csv = pd.read_csv(
        result.artifact_directory
        / "aggregate_results.csv"
    )

    assert tuple(
        fold_csv.columns
    ) == FOLD_RESULT_COLUMNS
    assert tuple(
        aggregate_csv.columns
    ) == AGGREGATE_RESULT_COLUMNS
    assert len(fold_csv) == 8
    assert len(aggregate_csv) == 2
    assert list(
        fold_csv[
            [
                "strategy",
                "fold_id",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ) == [
        (
            strategy,
            f"wf_{year}",
        )
        for strategy in (
            WALK_FORWARD_STRATEGIES
        )
        for year in range(2022, 2026)
    ]


def test_machine_artifacts_preserve_scope_resets_and_counts(
    tmp_path: Path,
) -> None:
    result = write_compact(tmp_path)
    folds = pd.read_csv(
        result.artifact_directory
        / "fold_results.csv"
    )
    aggregate = pd.read_csv(
        result.artifact_directory
        / "aggregate_results.csv"
    )

    assert folds["symbol"].eq("SPY").all()
    assert folds[
        "frequency"
    ].eq("15min").all()
    assert folds[
        "initial_test_position"
    ].eq(0).all()
    assert folds[
        "initial_test_turnover"
    ].eq(0.0).all()
    assert folds["purge_sessions"].eq(0).all()
    assert folds["embargo_sessions"].eq(0).all()

    timestamps = pd.to_datetime(
        folds[
            [
                "train_start_timestamp",
                "train_end_timestamp",
                "test_start_timestamp",
                "test_end_timestamp",
            ]
        ].stack(),
        utc=True,
    )

    assert timestamps.max() < pd.Timestamp(
        "2026-01-01",
        tz="UTC",
    )

    for row in aggregate.itertuples(
        index=False
    ):
        selected = folds.loc[
            folds["strategy"].eq(
                row.strategy
            )
        ]

        assert row.test_observations == (
            selected[
                "test_observations"
            ].sum()
        )
        assert row.test_sessions == (
            selected[
                "test_sessions"
            ].sum()
        )

    forbidden = {
        column
        for column in folds.columns
        if (
            "train_return" in column
            or "training_return" in column
        )
    }

    assert not forbidden


def test_manifest_serializes_frozen_contract_and_hashes(
    tmp_path: Path,
) -> None:
    result = write_compact(tmp_path)
    manifest = json.loads(
        (
            result.artifact_directory
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert manifest["day"] == 11
    assert manifest[
        "artifact_version"
    ] == DAY11_ARTIFACT_VERSION
    assert manifest[
        "generation_timestamp"
    ] == FIXED_GENERATION_TIMESTAMP
    assert manifest["symbol"] == "SPY"
    assert manifest["frequency"] == "15min"
    assert manifest[
        "locked_2026_period_accessed"
    ] is False
    assert manifest[
        "parameter_selection_performed"
    ] is False
    assert manifest[
        "profitability_acceptance_gate"
    ] is False
    assert manifest[
        "strategy_configuration_identifiers"
    ] == CONFIGURATION_IDS
    assert len(
        manifest["fold_definitions"]
    ) == 4
    assert manifest[
        "fold_result_rows"
    ] == 8
    assert manifest[
        "aggregate_result_rows"
    ] == 2

    hashes = manifest[
        "artifact_sha256"
    ]
    expected_names = {
        name
        for name in (
            APPROVED_DAY11_ARTIFACT_NAMES
        )
        if name != "manifest.json"
    }

    assert set(hashes) == expected_names

    for name, digest in hashes.items():
        assert digest == hashlib.sha256(
            (
                result.artifact_directory
                / name
            ).read_bytes()
        ).hexdigest()


def test_report_contains_required_methodology_and_no_selection(
    tmp_path: Path,
) -> None:
    report = write_compact(
        tmp_path
    ).report.lower()

    for heading in (
        "1. title and scope",
        "2. scientific purpose",
        "3. locked scope",
        "4. fold design",
        "5. leakage controls",
        "6. frozen configurations",
        "7. dataset audit",
        "8. per-fold results",
        "9. aggregate out-of-sample results",
        "10. stability interpretation",
        "11. acceptance criteria",
        "12. limitations and next step",
    ):
        assert heading in report

    for phrase in (
        "whole trading sessions",
        "training history is available only to warm "
        "indicators",
        "training returns, turnover, costs and drawdowns "
        "are excluded",
        "resets position and delayed execution state",
        "one-observation execution delay",
        "purge and embargo are both zero sessions",
        "2026 locked final-test period was not accessed",
        "profitability is not an acceptance condition",
        "do not declare a winning strategy",
        "does not select either strategy for deployment",
    ):
        assert phrase in report


def test_report_generation_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = write_compact(
        tmp_path,
        name="first",
    )
    second = write_compact(
        tmp_path,
        name="second",
    )

    for name in (
        APPROVED_DAY11_ARTIFACT_NAMES
    ):
        assert (
            first.artifact_directory
            / name
        ).read_bytes() == (
            second.artifact_directory
            / name
        ).read_bytes()


def test_undefined_sharpe_uses_empty_csv_convention(
    tmp_path: Path,
) -> None:
    result = write_compact(
        tmp_path,
        undefined_sharpe=True,
    )
    folds = pd.read_csv(
        result.artifact_directory
        / "fold_results.csv"
    )
    aggregate = pd.read_csv(
        result.artifact_directory
        / "aggregate_results.csv"
    )

    assert folds["sharpe_ratio"].isna().all()
    assert aggregate[
        "sharpe_ratio"
    ].isna().all()


def test_refuses_unsafe_overwrite_and_allows_explicit_replacement(
    tmp_path: Path,
) -> None:
    write_compact(tmp_path)

    with pytest.raises(
        Day11ReportError,
        match="--overwrite",
    ):
        write_compact(tmp_path)

    replaced = write_compact(
        tmp_path,
        overwrite=True,
    )

    assert len(
        replaced.artifact_paths
    ) == len(
        APPROVED_DAY11_ARTIFACT_NAMES
    )


def test_figures_are_nonempty_png_test_fold_comparisons(
    tmp_path: Path,
) -> None:
    result = write_compact(tmp_path)

    for name in (
        "net_return.png",
        "sharpe.png",
        "drawdown.png",
    ):
        content = (
            result.artifact_directory
            / name
        ).read_bytes()

        assert len(content) > 1_000
        assert content.startswith(
            b"\x89PNG\r\n\x1a\n"
        )


def test_report_builder_does_not_mutate_inputs() -> None:
    results = make_results()
    original_folds = (
        results.fold_results.copy(
            deep=True
        )
    )
    original_aggregate = (
        results.aggregate_results.copy(
            deep=True
        )
    )

    first = build_day11_report(
        results,
        make_audit(),
    )
    second = build_day11_report(
        results,
        make_audit(),
    )

    assert first == second
    pd.testing.assert_frame_equal(
        results.fold_results,
        original_folds,
    )
    pd.testing.assert_frame_equal(
        results.aggregate_results,
        original_aggregate,
    )
