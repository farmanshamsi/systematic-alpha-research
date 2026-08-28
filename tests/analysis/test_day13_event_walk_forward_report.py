"""Contracts for deterministic Day 13 reporting and artifacts."""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
    fields,
    is_dataclass,
)
import inspect
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import systematic_alpha.analysis.day13_event_walk_forward_report as day13_report
from systematic_alpha.analysis.trend_family_event_replay import (
    TrendFamilyEventReplayResult,
)
from systematic_alpha.analysis.trend_family_event_walk_forward import (
    AGGREGATE_SUMMARY_COLUMNS,
    EVENT_COUNT_COLUMNS,
    FOLD_SUMMARY_COLUMNS,
    PARITY_COLUMNS,
    PARITY_MAPPINGS,
    PARITY_TOLERANCE,
    PERFORMANCE_COLUMNS,
    POSITION_DIAGNOSTIC_COLUMNS,
    STRATEGY_ORDER,
    EventWalkForwardFoldRun,
    TrendFamilyEventWalkForwardResults,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    CONFIGURATION_IDS,
)


FOLD_IDS = (
    "wf_2022",
    "wf_2023",
    "wf_2024",
    "wf_2025",
)
SERIES_ORDER = (
    "gross",
    "net",
)
EXPECTED_ARTIFACT_NAMES = (
    "fold_summary.csv",
    "event_counts.csv",
    "position_diagnostics.csv",
    "performance.csv",
    "vectorized_parity.csv",
    "aggregate_summary.csv",
    "manifest.json",
    "report.md",
)
PROHIBITED_PUBLIC_FIELDS = (
    "winner",
    "ranking",
    "recommendation",
    "selected_strategy",
    "selected_parameters",
    "best_configuration",
    "profitability_gate",
)
PROHIBITED_CONCLUSIONS = (
    "best strategy",
    "preferred strategy",
    "selected strategy",
    "outperforming strategy",
    "profitability gate",
    "production recommendation",
)


def _fold_year(
    fold_id: str,
) -> int:
    """Return the calendar test year encoded in one fold identifier."""

    return int(
        fold_id.removeprefix("wf_")
    )


def _fold_summary_records() -> list[
    dict[str, object]
]:
    """Build eight deterministic synthetic fold-summary records."""

    records: list[
        dict[str, object]
    ] = []

    for strategy in STRATEGY_ORDER:
        for fold_id in FOLD_IDS:
            year = _fold_year(fold_id)
            records.append(
                {
                    "strategy": strategy,
                    "fold_id": fold_id,
                    "symbol": "SPY",
                    "frequency": "15min",
                    "configuration_id": (
                        CONFIGURATION_IDS[
                            strategy
                        ]
                    ),
                    "train_start_timestamp": pd.Timestamp(
                        "2020-01-02",
                        tz="UTC",
                    ),
                    "train_end_exclusive": pd.Timestamp(
                        f"{year}-01-01",
                        tz="UTC",
                    ),
                    "test_start_timestamp": pd.Timestamp(
                        f"{year}-01-01",
                        tz="UTC",
                    ),
                    "test_end_exclusive": pd.Timestamp(
                        f"{year + 1}-01-01",
                        tz="UTC",
                    ),
                    "evaluation_start": pd.Timestamp(
                        f"{year}-01-03 "
                        "14:30:00+00:00",
                    ),
                    "evaluation_end_exclusive": pd.Timestamp(
                        f"{year}-12-30 "
                        "21:00:00+00:00",
                    ),
                    "train_observations": (
                        100
                        + (year - 2022) * 40
                    ),
                    "test_observations": 40,
                    "train_sessions": (
                        4 + year - 2022
                    ),
                    "test_sessions": 2,
                    "indicator_history_observations": (
                        100
                        + (year - 2022) * 40
                    ),
                    "initial_position": 0,
                    "initial_equity": 1.0,
                    "parity_comparisons": 8,
                    "parity_passed": True,
                }
            )

    return records


def _event_count_records() -> list[
    dict[str, object]
]:
    """Build eight reconciled synthetic event-count records."""

    return [
        {
            "strategy": strategy,
            "fold_id": fold_id,
            "market_bar_events": 40,
            "signal_events": 40,
            "order_events": 3,
            "fill_events": 3,
            "portfolio_snapshots": 40,
            "total_events": 126,
            "observations": 40,
        }
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
    ]


def _position_records() -> list[
    dict[str, object]
]:
    """Build neutral synthetic fold-boundary diagnostics."""

    return [
        {
            "strategy": strategy,
            "fold_id": fold_id,
            "initial_previous_position": 0,
            "initial_position": 0,
            "initial_position_eligible": False,
            "initial_turnover": 0.0,
            "initial_transaction_cost": 0.0,
            "initial_previous_equity": 1.0,
            "initial_cash_balance": 1.0,
            "initial_holdings_value": 0.0,
            "initial_ending_equity": 1.0,
            "initial_fill_executed": False,
            "total_turnover": 3.0,
            "total_fractional_transaction_cost": 0.0003,
            "total_transaction_cost_amount": 0.0003,
            "final_equity": 1.01,
        }
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
    ]


def _performance_records() -> list[
    dict[str, object]
]:
    """Build sixteen neutral gross/net fold-performance rows."""

    return [
        {
            "strategy": strategy,
            "fold_id": fold_id,
            "series": series,
            "observations": 40,
            "sessions": 2,
            "annualization_factor": 5_040.0,
            "cumulative_return": (
                0.011
                if series == "gross"
                else 0.010
            ),
            "annualized_return": (
                0.15
                if series == "gross"
                else 0.14
            ),
            "annualized_volatility": 0.20,
            "sharpe_ratio": (
                0.75
                if series == "gross"
                else 0.70
            ),
            "maximum_drawdown": -0.03,
            "final_wealth": (
                1.011
                if series == "gross"
                else 1.010
            ),
        }
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
        for series in SERIES_ORDER
    ]


def _parity_records() -> list[
    dict[str, object]
]:
    """Build all sixty-four frozen parity records."""

    return [
        {
            "strategy": strategy,
            "fold_id": fold_id,
            "comparison": replay_column,
            "comparison_type": (
                comparison_type
            ),
            "row_count": 40,
            "maximum_absolute_difference": 0.0,
            "mismatch_count": 0,
            "tolerance": (
                0.0
                if comparison_type == "exact"
                else PARITY_TOLERANCE
            ),
            "passed": True,
        }
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
        for (
            replay_column,
            _,
            comparison_type,
        ) in PARITY_MAPPINGS
    ]


def _aggregate_records() -> list[
    dict[str, object]
]:
    """Build four recomputed aggregate evidence rows."""

    return [
        {
            "strategy": strategy,
            "series": series,
            "folds": 4,
            "observations": 160,
            "sessions": 8,
            "annualization_factor": 5_040.0,
            "cumulative_return": (
                0.045
                if series == "gross"
                else 0.041
            ),
            "annualized_return": (
                0.16
                if series == "gross"
                else 0.145
            ),
            "annualized_volatility": 0.20,
            "sharpe_ratio": (
                0.80
                if series == "gross"
                else 0.72
            ),
            "maximum_drawdown": -0.05,
            "final_wealth": (
                1.045
                if series == "gross"
                else 1.041
            ),
        }
        for strategy in STRATEGY_ORDER
        for series in SERIES_ORDER
    ]


def make_synthetic_results() -> (
    TrendFamilyEventWalkForwardResults
):
    """Construct compact valid Day 13 scientific evidence."""

    fold_summary = pd.DataFrame.from_records(
        _fold_summary_records(),
        columns=FOLD_SUMMARY_COLUMNS,
    )
    event_counts = pd.DataFrame.from_records(
        _event_count_records(),
        columns=EVENT_COUNT_COLUMNS,
    )
    position_diagnostics = (
        pd.DataFrame.from_records(
            _position_records(),
            columns=(
                POSITION_DIAGNOSTIC_COLUMNS
            ),
        )
    )
    performance = pd.DataFrame.from_records(
        _performance_records(),
        columns=PERFORMANCE_COLUMNS,
    )
    vectorized_parity = (
        pd.DataFrame.from_records(
            _parity_records(),
            columns=PARITY_COLUMNS,
        )
    )
    aggregate_summary = (
        pd.DataFrame.from_records(
            _aggregate_records(),
            columns=(
                AGGREGATE_SUMMARY_COLUMNS
            ),
        )
    )
    fold_runs = tuple(
        EventWalkForwardFoldRun(
            strategy=strategy,
            fold_id=fold_id,
            replay_result=cast(
                TrendFamilyEventReplayResult,
                object(),
            ),
            vectorized_observations=pd.DataFrame(
                {
                    "signal": [
                        0,
                    ],
                }
            ),
        )
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
    )

    return TrendFamilyEventWalkForwardResults(
        fold_runs=fold_runs,
        fold_summary=fold_summary,
        event_counts=event_counts,
        position_diagnostics=(
            position_diagnostics
        ),
        performance=performance,
        vectorized_parity=vectorized_parity,
        aggregate_summary=aggregate_summary,
    )


def make_report_bundle() -> (
    day13_report.Day13EventWalkForwardReport
):
    """Construct one directly valid report-bundle shell."""

    results = make_synthetic_results()

    return (
        day13_report.Day13EventWalkForwardReport(
            fold_summary=results.fold_summary,
            event_counts=results.event_counts,
            position_diagnostics=(
                results.position_diagnostics
            ),
            performance=results.performance,
            vectorized_parity=(
                results.vectorized_parity
            ),
            aggregate_summary=(
                results.aggregate_summary
            ),
            manifest={
                "report_id": (
                    "day13_event_walk_forward"
                ),
                "row_counts": {
                    "fold_summary": 8,
                },
                "strategies": list(
                    STRATEGY_ORDER
                ),
            },
            report="# Day 13 synthetic report\n",
        )
    )


def test_public_report_interfaces_and_filenames_are_frozen() -> None:
    """Freeze narrow call signatures and the exact artifact set."""

    build_signature = inspect.signature(
        day13_report
        .build_day13_event_walk_forward_report
    )
    writer_signature = inspect.signature(
        day13_report
        .write_day13_event_walk_forward_artifacts
    )

    assert tuple(
        build_signature.parameters
    ) == ("results",)
    assert tuple(
        writer_signature.parameters
    ) == (
        "report",
        "output_directory",
        "overwrite",
    )
    assert (
        day13_report
        .APPROVED_DAY13_ARTIFACT_NAMES
        == EXPECTED_ARTIFACT_NAMES
    )
    assert not any(
        name.endswith(".png")
        for name in EXPECTED_ARTIFACT_NAMES
    )


def test_report_bundle_is_frozen_and_slotted() -> None:
    """Require an immutable report shell without an instance dictionary."""

    report = make_report_bundle()

    assert is_dataclass(report)
    assert not hasattr(report, "__dict__")

    with pytest.raises(
        FrozenInstanceError,
    ):
        report.report = "changed"


def test_report_bundle_defensively_copies_tables_and_manifest() -> None:
    """Protect retained evidence and recursively freeze the manifest."""

    results = make_synthetic_results()
    source_manifest = {
        "row_counts": {
            "fold_summary": 8,
        },
        "strategies": list(
            STRATEGY_ORDER
        ),
    }
    report = (
        day13_report.Day13EventWalkForwardReport(
            fold_summary=results.fold_summary,
            event_counts=results.event_counts,
            position_diagnostics=(
                results.position_diagnostics
            ),
            performance=results.performance,
            vectorized_parity=(
                results.vectorized_parity
            ),
            aggregate_summary=(
                results.aggregate_summary
            ),
            manifest=source_manifest,
            report="# Synthetic\n",
        )
    )
    results.fold_summary.loc[
        0,
        "fold_id",
    ] = "changed"
    source_manifest[
        "row_counts"
    ]["fold_summary"] = 0

    assert report.fold_summary.loc[
        0,
        "fold_id",
    ] == "wf_2022"

    with pytest.raises(TypeError):
        cast(
            dict[str, object],
            report.manifest[
                "row_counts"
            ],
        )["fold_summary"] = 0

    copied_frame = (
        report.copy_fold_summary()
    )
    copied_frame.loc[
        0,
        "fold_id",
    ] = "changed"
    copied_manifest = (
        report.copy_manifest()
    )
    cast(
        dict[str, object],
        copied_manifest[
            "row_counts"
        ],
    )["fold_summary"] = 0

    assert report.fold_summary.loc[
        0,
        "fold_id",
    ] == "wf_2022"
    assert cast(
        dict[str, object],
        report.copy_manifest()[
            "row_counts"
        ],
    )["fold_summary"] == 8

    for copy_method in (
        report.copy_event_counts,
        report.copy_position_diagnostics,
        report.copy_performance,
        report.copy_vectorized_parity,
        report.copy_aggregate_summary,
    ):
        copied = copy_method()
        copied.iloc[
            0,
            0,
        ] = "changed"
        assert (
            copy_method().iloc[
                0,
                0,
            ]
            != "changed"
        )


def test_invalid_public_inputs_are_rejected_before_red_phase(
    tmp_path: Path,
) -> None:
    """Reject invalid builder, bundle, writer, and path inputs."""

    with pytest.raises(
        TypeError,
        match=(
            "TrendFamilyEventWalkForwardResults"
        ),
    ):
        day13_report.build_day13_event_walk_forward_report(
            object()
        )

    with pytest.raises(
        TypeError,
        match="Day13EventWalkForwardReport",
    ):
        day13_report.write_day13_event_walk_forward_artifacts(
            object(),
            tmp_path,
        )

    with pytest.raises(
        TypeError,
        match="path",
    ):
        day13_report.write_day13_event_walk_forward_artifacts(
            make_report_bundle(),
            13,
        )


def test_public_report_types_contain_no_selection_fields() -> None:
    """Keep ranking and strategy-selection concepts out of public fields."""

    names = " ".join(
        field.name.lower()
        for field in fields(
            day13_report
            .Day13EventWalkForwardReport
        )
    )

    assert not any(
        token in names
        for token in (
            PROHIBITED_PUBLIC_FIELDS
        )
    )


def test_report_preserves_schemas_and_deterministic_order() -> None:
    """Preserve all six core schemas and frozen scientific row order."""

    report = (
        day13_report
        .build_day13_event_walk_forward_report(
            make_synthetic_results()
        )
    )

    assert tuple(
        report.fold_summary.columns
    ) == FOLD_SUMMARY_COLUMNS
    assert tuple(
        report.event_counts.columns
    ) == EVENT_COUNT_COLUMNS
    assert tuple(
        report.position_diagnostics.columns
    ) == POSITION_DIAGNOSTIC_COLUMNS
    assert tuple(
        report.performance.columns
    ) == PERFORMANCE_COLUMNS
    assert tuple(
        report.vectorized_parity.columns
    ) == PARITY_COLUMNS
    assert tuple(
        report.aggregate_summary.columns
    ) == AGGREGATE_SUMMARY_COLUMNS
    assert list(
        report.fold_summary[
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
            fold_id,
        )
        for strategy in STRATEGY_ORDER
        for fold_id in FOLD_IDS
    ]


def test_report_exposes_fold_reset_event_and_leakage_evidence() -> None:
    """Retain explicit neutral-reset and actual event-count diagnostics."""

    report = (
        day13_report
        .build_day13_event_walk_forward_report(
            make_synthetic_results()
        )
    )

    assert len(report.fold_summary) == 8
    assert report.fold_summary[
        "parity_comparisons"
    ].eq(8).all()
    assert report.fold_summary[
        "initial_position"
    ].eq(0).all()
    assert report.fold_summary[
        "initial_equity"
    ].eq(1.0).all()
    assert len(report.event_counts) == 8
    assert report.event_counts[
        "market_bar_events"
    ].eq(
        report.event_counts[
            "observations"
        ]
    ).all()
    assert report.event_counts[
        "signal_events"
    ].eq(
        report.event_counts[
            "observations"
        ]
    ).all()
    assert report.event_counts[
        "portfolio_snapshots"
    ].eq(
        report.event_counts[
            "observations"
        ]
    ).all()
    diagnostics = (
        report.position_diagnostics
    )
    assert diagnostics[
        "initial_previous_position"
    ].eq(0).all()
    assert diagnostics[
        "initial_position"
    ].eq(0).all()
    assert ~diagnostics[
        "initial_position_eligible"
    ].astype(bool).any()
    assert diagnostics[
        "initial_previous_equity"
    ].eq(1.0).all()
    assert diagnostics[
        "initial_cash_balance"
    ].eq(1.0).all()
    assert diagnostics[
        "initial_holdings_value"
    ].eq(0.0).all()
    assert diagnostics[
        "initial_ending_equity"
    ].eq(1.0).all()
    assert ~diagnostics[
        "initial_fill_executed"
    ].astype(bool).any()


def test_report_retains_all_parity_and_performance_diagnostics() -> None:
    """Show failures truthfully and present recomputed gross/net evidence."""

    report = (
        day13_report
        .build_day13_event_walk_forward_report(
            make_synthetic_results()
        )
    )
    parity = report.vectorized_parity

    assert len(parity) == 64
    assert parity.groupby(
        [
            "strategy",
            "fold_id",
        ],
        sort=False,
    ).size().eq(8).all()
    assert parity.loc[
        parity[
            "comparison_type"
        ].eq("exact"),
        "tolerance",
    ].eq(0.0).all()
    assert parity.loc[
        parity[
            "comparison_type"
        ].eq("numeric"),
        "tolerance",
    ].eq(PARITY_TOLERANCE).all()
    assert {
        "row_count",
        "mismatch_count",
        "maximum_absolute_difference",
        "passed",
    } <= set(parity.columns)
    assert len(report.performance) == 16
    assert len(
        report.aggregate_summary
    ) == 4
    assert list(
        report.performance[
            "series"
        ].drop_duplicates()
    ) == list(SERIES_ORDER)
    assert report.aggregate_summary[
        "folds"
    ].eq(4).all()


def test_markdown_freezes_methodology_limitations_and_neutral_language() -> None:
    """Require all methodology sections without selecting a strategy."""

    report = (
        day13_report
        .build_day13_event_walk_forward_report(
            make_synthetic_results()
        )
    )
    lowered = report.report.lower()
    sections = (
        "scope and frozen protocol",
        "scientific question",
        "fold definitions",
        "event-driven replay and warm-up design",
        "fold-boundary reset and leakage controls",
        "event counts and execution diagnostics",
        "vectorised/event-driven parity",
        "fold-level gross and net performance",
        "aggregate walk-forward evidence",
        "interpretation and limitations",
        "reproducibility and manifest",
    )
    statements = (
        "development data only",
        "spy only",
        "15-minute bars only",
        "trend ratio",
        "ema/macd",
        "four fixed expanding folds",
        "training history warms indicators",
        "execution events are test-only",
        "neutral position",
        "pending execution state",
        "equity reset to 1.0",
        "one-observation execution delay",
        "no training p&l leakage",
        "no fold-boundary position leakage",
        "chronological out-of-sample aggregation",
        "no tuning",
        "no ranking",
        "no winner selection",
        "no 2026 data",
        "not final locked-period performance",
    )

    assert all(
        section in lowered
        for section in sections
    )
    assert all(
        statement in lowered
        for statement in statements
    )
    assert not any(
        phrase in lowered
        for phrase in (
            PROHIBITED_CONCLUSIONS
        )
    )


def test_manifest_is_deterministic_complete_and_neutral() -> None:
    """Freeze scope, row counts, hashes, and explicit false controls."""

    report = (
        day13_report
        .build_day13_event_walk_forward_report(
            make_synthetic_results()
        )
    )
    manifest = report.copy_manifest()

    assert manifest[
        "report_id"
    ] == "day13_event_walk_forward"
    assert manifest[
        "artifact_version"
    ] == (
        day13_report
        .DAY13_ARTIFACT_VERSION
    )
    assert manifest[
        "artifact_filenames"
    ] == list(EXPECTED_ARTIFACT_NAMES)
    assert manifest["development_only"] is True
    assert manifest["symbol"] == "SPY"
    assert manifest["frequency"] == "15min"
    assert manifest["strategies"] == list(
        STRATEGY_ORDER
    )
    assert manifest["folds"] == list(
        FOLD_IDS
    )
    assert manifest[
        "locked_period_accessed"
    ] is False
    assert manifest[
        "tuning_performed"
    ] is False
    assert manifest[
        "ranking_performed"
    ] is False
    assert manifest[
        "winner_selection_performed"
    ] is False
    assert manifest[
        "row_counts"
    ] == {
        "fold_summary": 8,
        "event_counts": 8,
        "position_diagnostics": 8,
        "performance": 16,
        "vectorized_parity": 64,
        "aggregate_summary": 4,
    }
    assert manifest[
        "parity_comparison_count"
    ] == 64
    assert manifest[
        "parity_passed_count"
    ] == 64
    assert set(
        cast(
            dict[str, str],
            manifest[
                "artifact_sha256"
            ],
        )
    ) == set(
        EXPECTED_ARTIFACT_NAMES
    ).difference(
        {
            "manifest.json",
        }
    )
    assert "generation_timestamp" not in manifest
    json.dumps(
        manifest,
        sort_keys=True,
        allow_nan=False,
    )


def test_writer_freezes_exact_atomic_deterministic_artifact_set(
    tmp_path: Path,
) -> None:
    """Write exactly eight stable files and safely replace approved output."""

    report = make_report_bundle()
    output = tmp_path / "day13"
    first = (
        day13_report
        .write_day13_event_walk_forward_artifacts(
            report,
            output,
        )
    )
    first_bytes = {
        path.name: path.read_bytes()
        for path in first
    }
    second = (
        day13_report
        .write_day13_event_walk_forward_artifacts(
            report,
            output,
            overwrite=True,
        )
    )

    assert tuple(
        path.name
        for path in first
    ) == EXPECTED_ARTIFACT_NAMES
    assert tuple(
        path.name
        for path in second
    ) == EXPECTED_ARTIFACT_NAMES
    assert {
        path.name: path.read_bytes()
        for path in second
    } == first_bytes
    assert {
        path.name
        for path in output.iterdir()
    } == set(EXPECTED_ARTIFACT_NAMES)
    assert not any(
        path.suffix == ".png"
        for path in output.iterdir()
    )


def test_writer_failure_leaves_no_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Surface write failure without exposing a partial artifact set."""

    report = make_report_bundle()
    output = tmp_path / "day13"
    original_write_text = (
        Path.write_text
    )

    def fail_report(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        if path.name == "report.md":
            raise OSError(
                "synthetic write failure"
            )

        return original_write_text(
            path,
            data,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "write_text",
        fail_report,
    )
    day13_report.write_day13_event_walk_forward_artifacts(
        report,
        output,
    )

    assert (
        not output.exists()
        or not any(output.iterdir())
    )
