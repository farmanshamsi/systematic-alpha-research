"""Deterministic reporting for Day 15 strategy diversification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Final, Mapping

import pandas as pd

from systematic_alpha.analysis.strategy_diversification import (
    CONFIGURATION_IDS,
    ENSEMBLE_FEASIBILITY_COLUMNS,
    FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS,
    FOLD_PAIRWISE_CORRELATION_COLUMNS,
    FREQUENCY,
    FROZEN_SLEEVES,
    FULL_SAMPLE_COVARIANCE_DIAGNOSTIC_COLUMNS,
    FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS,
    MAX_ABSOLUTE_CORRELATION,
    MAX_MEDIAN_PC1_SHARE,
    MIN_MEDIAN_EFFECTIVE_RANK,
    MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO,
    MIN_TEST_SESSIONS,
    MIN_TRAINING_SESSIONS,
    PSD_TOLERANCE,
    SLEEVE_IDS,
    SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
    SLEEVE_PAIRS,
    VARIANCE_TOLERANCE,
    StrategyDiversificationResults,
)
from systematic_alpha.analysis.trend_family_robustness import (
    DEVELOPMENT_DATASET_ID,
)
from systematic_alpha.analysis.trend_family_walk_forward import (
    build_walk_forward_folds,
)


DAY15_ARTIFACT_VERSION: Final[str] = (
    "day15_strategy_diversification_v1"
)

SLEEVE_INPUT_FILENAME: Final[str] = (
    "sleeve_input_diagnostics.csv"
)
FULL_SAMPLE_PAIRWISE_FILENAME: Final[str] = (
    "full_sample_pairwise_correlations.csv"
)
FOLD_PAIRWISE_FILENAME: Final[str] = (
    "fold_pairwise_correlations.csv"
)
FOLD_COVARIANCE_FILENAME: Final[str] = (
    "fold_covariance_diagnostics.csv"
)
ENSEMBLE_FEASIBILITY_FILENAME: Final[str] = (
    "ensemble_feasibility.csv"
)
REPORT_FILENAME: Final[str] = "report.md"
MANIFEST_FILENAME: Final[str] = "manifest.json"

APPROVED_DAY15_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    SLEEVE_INPUT_FILENAME,
    FULL_SAMPLE_PAIRWISE_FILENAME,
    FOLD_PAIRWISE_FILENAME,
    FOLD_COVARIANCE_FILENAME,
    ENSEMBLE_FEASIBILITY_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
)

EXPECTED_ROW_COUNTS: Final[dict[str, int]] = {
    "sleeve_input_diagnostics": 6,
    "full_sample_pairwise_correlations": 15,
    "fold_pairwise_correlations": 120,
    "fold_covariance_diagnostics": 8,
    "ensemble_feasibility": 1,
}


class Day15StrategyDiversificationReportError(ValueError):
    """Raised when Day 15 reporting cannot proceed safely."""


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Return a defensive zero-based copy of one evidence table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame.copy(deep=True).reset_index(drop=True)


def _validated_table(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
    rows: int,
) -> pd.DataFrame:
    """Validate one exact Day 15 evidence table."""

    retained = _copy_frame(frame, name=name)
    if tuple(retained.columns) != columns:
        raise Day15StrategyDiversificationReportError(
            f"{name} has an unexpected schema."
        )
    if len(retained) != rows:
        raise Day15StrategyDiversificationReportError(
            f"{name} must contain exactly {rows} rows."
        )
    return retained


def _validate_row_order(
    *,
    sleeve_inputs: pd.DataFrame,
    full_pairwise: pd.DataFrame,
    fold_pairwise: pd.DataFrame,
    fold_covariance: pd.DataFrame,
) -> None:
    """Require the exact Phase 1 deterministic row order."""

    if tuple(sleeve_inputs["sleeve_id"]) != SLEEVE_IDS:
        raise Day15StrategyDiversificationReportError(
            "Sleeve diagnostics do not follow frozen sleeve order."
        )

    full_keys = tuple(
        full_pairwise[["sleeve_a", "sleeve_b"]].itertuples(
            index=False,
            name=None,
        )
    )
    if full_keys != SLEEVE_PAIRS:
        raise Day15StrategyDiversificationReportError(
            "Full-sample pairs do not follow frozen pair order."
        )

    expected_fold_pair_keys: list[tuple[str, str, str, str]] = []
    expected_fold_keys: list[tuple[str, str]] = []
    for fold in build_walk_forward_folds():
        for sample in ("train", "test"):
            expected_fold_keys.append((fold.fold_id, sample))
            expected_fold_pair_keys.extend(
                (
                    fold.fold_id,
                    sample,
                    sleeve_a,
                    sleeve_b,
                )
                for sleeve_a, sleeve_b in SLEEVE_PAIRS
            )

    actual_fold_pair_keys = tuple(
        fold_pairwise[
            ["fold_id", "sample", "sleeve_a", "sleeve_b"]
        ].itertuples(index=False, name=None)
    )
    if actual_fold_pair_keys != tuple(expected_fold_pair_keys):
        raise Day15StrategyDiversificationReportError(
            "Fold pairs do not follow frozen fold/sample/pair order."
        )

    actual_fold_keys = tuple(
        fold_covariance[["fold_id", "sample"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual_fold_keys != tuple(expected_fold_keys):
        raise Day15StrategyDiversificationReportError(
            "Fold diagnostics do not follow frozen fold/sample order."
        )


def _freeze_manifest(value: object) -> object:
    """Recursively freeze mutable manifest containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_manifest(item)
                for key, item in deepcopy(dict(value)).items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_manifest(item)
            for item in deepcopy(value)
        )
    return deepcopy(value)


def _copy_manifest(value: object) -> object:
    """Return mutable copies of frozen manifest values."""

    if isinstance(value, Mapping):
        return {
            str(key): _copy_manifest(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_copy_manifest(item) for item in value]
    return deepcopy(value)


def _iso_date(value: pd.Timestamp) -> str:
    """Render one frozen fold boundary as an ISO date."""

    return value.strftime("%Y-%m-%d")


def _fold_definitions() -> list[dict[str, object]]:
    """Build deterministic manifest descriptions of all four folds."""

    return [
        {
            "fold_id": fold.fold_id,
            "train_start": _iso_date(fold.train_start),
            "train_end_exclusive": _iso_date(
                fold.train_end_exclusive
            ),
            "test_start": _iso_date(fold.test_start),
            "test_end_exclusive": _iso_date(
                fold.test_end_exclusive
            ),
            "purge_sessions": fold.purge_sessions,
            "embargo_sessions": fold.embargo_sessions,
        }
        for fold in build_walk_forward_folds()
    ]


def _sleeve_universe() -> list[dict[str, str]]:
    """Build the exact ordered manifest sleeve universe."""

    return [
        {
            "sleeve_id": sleeve.sleeve_id,
            "strategy": sleeve.strategy,
            "symbol": sleeve.symbol,
            "frequency": sleeve.frequency,
            "configuration_id": sleeve.configuration_id,
        }
        for sleeve in FROZEN_SLEEVES
    ]


def _format_number(value: object, *, digits: int = 6) -> str:
    """Render one finite diagnostic deterministically."""

    number = float(value)
    if math.isinf(number):
        return "positive infinity" if number > 0.0 else "negative infinity"
    if math.isnan(number):
        return "undefined"
    return f"{number:.{digits}f}"


def _sleeve_lines(sleeve_inputs: pd.DataFrame) -> str:
    """Render the exact frozen sleeve universe."""

    lines = [
        "| Sleeve | Strategy | Symbol | Configuration |",
        "|---|---|---|---|",
    ]
    for row in sleeve_inputs.itertuples(index=False):
        lines.append(
            f"| {row.sleeve_id} | {row.strategy} | {row.symbol} | "
            f"{row.configuration_id} |"
        )
    return "\n".join(lines)


def _fold_lines(fold_covariance: pd.DataFrame) -> str:
    """Render training and realised test-fold evidence."""

    lines = [
        "| Fold | Sample | Sessions | Max abs. correlation | "
        "PC1 share | Effective rank | Equal-weight DR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fold_covariance.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.sample} | "
            f"{int(row.aligned_sessions)} | "
            f"{_format_number(row.maximum_absolute_correlation)} | "
            f"{_format_number(row.pc1_share)} | "
            f"{_format_number(row.effective_rank)} | "
            f"{_format_number(row.equal_weight_diversification_ratio)} |"
        )
    return "\n".join(lines)


def _gate_lines(feasibility: pd.DataFrame) -> str:
    """Render every predeclared gate and its pass/fail outcome."""

    row = feasibility.iloc[0]
    labels = (
        ("finite_returns_gate", "Finite sleeve returns"),
        (
            "exact_calendar_alignment_gate",
            "Exact common-session alignment",
        ),
        (
            "non_degenerate_sleeves_gate",
            "Non-degenerate sleeve variance",
        ),
        (
            "minimum_training_sessions_gate",
            "Minimum training sessions",
        ),
        ("minimum_test_sessions_gate", "Minimum test sessions"),
        (
            "maximum_absolute_correlation_gate",
            "Training absolute-correlation ceiling",
        ),
        (
            "median_effective_rank_gate",
            "Median training effective-rank floor",
        ),
        (
            "median_pc1_share_gate",
            "Median training PC1-share ceiling",
        ),
        ("correlation_psd_gate", "Correlation PSD status"),
        ("covariance_psd_gate", "Covariance PSD status"),
        (
            "realised_test_diversification_gate",
            "Median realised test diversification-ratio floor",
        ),
    )
    lines = [
        "| Gate | Outcome |",
        "|---|---:|",
    ]
    for column, label in labels:
        lines.append(
            f"| {label} | "
            f"{'Pass' if bool(row[column]) else 'Fail'} |"
        )
    return "\n".join(lines)


def _render_report(
    *,
    sleeve_inputs: pd.DataFrame,
    full_pairwise: pd.DataFrame,
    full_covariance: pd.DataFrame,
    fold_covariance: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> str:
    """Render neutral development-only Markdown with one final newline."""

    full = full_covariance.iloc[0]
    feasible = bool(feasibility.iloc[0]["ensemble_feasible"])
    outcome_word = "sufficient" if feasible else "insufficient"
    outcome_value = "true" if feasible else "false"

    minimum_pair = full_pairwise.loc[
        full_pairwise["correlation"].idxmin()
    ]
    maximum_pair = full_pairwise.loc[
        full_pairwise["correlation"].idxmax()
    ]

    markdown = f"""# Day 15 — Strategy Diversification Feasibility

## 1. Objective and scope

This development-only study measures return-stream diversification across six
frozen strategy sleeves. Statistical diversification, economic performance,
and portfolio allocation are distinct questions: this phase addresses only
the first.

Low correlation does not automatically create alpha.
No profitability improvement is claimed.

## 2. Frozen six-sleeve universe

{_sleeve_lines(sleeve_inputs)}

Each strategy uses its frozen Day 10 configuration.
No strategy was ranked or removed.

## 3. Data and alignment protocol

The source frequency is 15 minutes and the permitted development dates are
2020-01-02 through 2025-12-31. Existing net strategy returns include the
frozen transaction-cost convention and preserve position[t] = signal[t-1].
Intraday simple net returns are compounded within each session as
product(1 + intraday net return) - 1. The six session series are aligned on
exact common session dates. Forward filling, backward filling, interpolation,
or asynchronous calendar matching is not used.

The locked January–June 2026 period was not accessed.

## 4. Mathematical diagnostics

Pearson correlations are calculated for all 15 unique unordered sleeve pairs.
Correlation eigenvalues are evaluated in descending order; first-principal-
component share and entropy effective rank describe concentration. Covariance
uses ordinary sample covariance with ddof=1, without repair or shrinkage.
Positive-semidefinite status uses the frozen numerical tolerance.
Equal weights of 1/6 are used only as a neutral diagnostic.
They are not a portfolio allocation.

## 5. Predeclared thresholds

- Sleeve variance must exceed {VARIANCE_TOLERANCE:.17g}.
- PSD eigenvalues may not be below -{PSD_TOLERANCE:.17g}.
- Every fold requires at least {MIN_TRAINING_SESSIONS} training sessions and
  {MIN_TEST_SESSIONS} test sessions.
- Maximum training absolute correlation may not exceed
  {MAX_ABSOLUTE_CORRELATION:.2f}.
- Median training effective rank must be at least
  {MIN_MEDIAN_EFFECTIVE_RANK:.2f}.
- Median training PC1 share may not exceed
  {MAX_MEDIAN_PC1_SHARE:.2f}.
- Median realised test equal-weight diversification ratio must be strictly
  greater than {MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO:.2f}.

## 6. Full-sample descriptive evidence

- Aligned sessions: {int(full["aligned_sessions"])}
- Minimum pair correlation: {_format_number(full["minimum_pairwise_correlation"])}
  ({minimum_pair["sleeve_a"]} / {minimum_pair["sleeve_b"]})
- Maximum pair correlation: {_format_number(full["maximum_pairwise_correlation"])}
  ({maximum_pair["sleeve_a"]} / {maximum_pair["sleeve_b"]})
- Maximum absolute correlation:
  {_format_number(full["maximum_absolute_correlation"])}
- First-PC share: {_format_number(full["pc1_share"])}
- Entropy effective rank: {_format_number(full["effective_rank"])}
- Equal-weight diversification ratio:
  {_format_number(full["equal_weight_diversification_ratio"])}
- Correlation PSD: {'Yes' if bool(full["correlation_psd"]) else 'No'}
- Covariance PSD: {'Yes' if bool(full["covariance_psd"]) else 'No'}

These statistics describe dependence; they do not establish economic
performance.

## 7. Training and realised test-fold evidence

{_fold_lines(fold_covariance)}

Training evidence is computed independently of future test rows. Realised test
diversification is descriptive and is not an outperformance claim.

## 8. Gate-by-gate feasibility conclusion

{_gate_lines(feasibility)}

The recorded ensemble_feasible outcome is **{outcome_value}**. The frozen
sleeves exhibit {outcome_word} return-stream diversification under the
predeclared correlation, eigenstructure, covariance, and walk-forward gates.
A false feasibility outcome is valid and is retained without changing the
sleeve universe.

## 9. Limitations and next-stage restrictions

This phase does not test alpha creation, economic utility, capacity, market
impact, or profitability improvement. Equal weights are diagnostic only.
No portfolio weights were optimised.
No leverage was used, and no allocation decision is provided.
Subsequent economic-performance or allocation work
must remain separate and must not use the locked period without explicit
authorization.
"""
    return markdown.rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class Day15StrategyDiversificationReport:
    """Immutable Day 15 report and approved evidence bundle."""

    sleeve_input_diagnostics: pd.DataFrame
    full_sample_pairwise_correlations: pd.DataFrame
    fold_pairwise_correlations: pd.DataFrame
    fold_covariance_diagnostics: pd.DataFrame
    ensemble_feasibility: pd.DataFrame
    manifest: Mapping[str, object]
    report: str

    def __post_init__(self) -> None:
        """Defensively retain all report evidence."""

        for name in (
            "sleeve_input_diagnostics",
            "full_sample_pairwise_correlations",
            "fold_pairwise_correlations",
            "fold_covariance_diagnostics",
            "ensemble_feasibility",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(getattr(self, name), name=name),
            )
        object.__setattr__(self, "manifest", _freeze_manifest(self.manifest))
        object.__setattr__(
            self,
            "report",
            str(self.report).rstrip() + "\n",
        )

    def copy_sleeve_input_diagnostics(self) -> pd.DataFrame:
        return self.sleeve_input_diagnostics.copy(deep=True)

    def copy_full_sample_pairwise_correlations(self) -> pd.DataFrame:
        return self.full_sample_pairwise_correlations.copy(deep=True)

    def copy_fold_pairwise_correlations(self) -> pd.DataFrame:
        return self.fold_pairwise_correlations.copy(deep=True)

    def copy_fold_covariance_diagnostics(self) -> pd.DataFrame:
        return self.fold_covariance_diagnostics.copy(deep=True)

    def copy_ensemble_feasibility(self) -> pd.DataFrame:
        return self.ensemble_feasibility.copy(deep=True)

    def copy_manifest(self) -> dict[str, object]:
        copied = _copy_manifest(self.manifest)
        if not isinstance(copied, dict):
            raise TypeError("Copied manifest must be a dictionary.")
        return copied


def _build_manifest(
    *,
    ensemble_feasible: bool,
) -> dict[str, object]:
    """Build deterministic Day 15 provenance and safety metadata."""

    manifest: dict[str, object] = {
        "report_id": "day15_strategy_diversification",
        "artifact_version": DAY15_ARTIFACT_VERSION,
        "schema_version": 1,
        "artifact_filenames": list(APPROVED_DAY15_ARTIFACT_NAMES),
        "development_only": True,
        "dataset_id": DEVELOPMENT_DATASET_ID,
        "frequency": FREQUENCY,
        "development_start": "2020-01-02",
        "development_end": "2025-12-31",
        "locked_period_start": "2026-01-02",
        "locked_period_end": "2026-06-30",
        "locked_period_accessed": False,
        "sleeve_universe": _sleeve_universe(),
        "sleeve_count": 6,
        "strategy_configuration_ids": dict(CONFIGURATION_IDS),
        "fold_definitions": _fold_definitions(),
        "covariance_estimator": "ordinary sample covariance",
        "covariance_ddof": 1,
        "alignment_method": (
            "exact common session dates across all six sleeves"
        ),
        "intraday_to_session_return_method": (
            "product(1 + intraday_net_return) - 1"
        ),
        "transaction_cost_convention": (
            "existing frozen per-turnover costs included in "
            "net_strategy_return"
        ),
        "execution_delay_convention": (
            "position[t] equals signal[t-1] within each sleeve"
        ),
        "numerical_tolerances": {
            "variance_tolerance": VARIANCE_TOLERANCE,
            "psd_tolerance": PSD_TOLERANCE,
        },
        "feasibility_thresholds": {
            "minimum_training_sessions": MIN_TRAINING_SESSIONS,
            "minimum_test_sessions": MIN_TEST_SESSIONS,
            "maximum_absolute_correlation": MAX_ABSOLUTE_CORRELATION,
            "minimum_median_effective_rank": MIN_MEDIAN_EFFECTIVE_RANK,
            "maximum_median_pc1_share": MAX_MEDIAN_PC1_SHARE,
            "minimum_median_test_diversification_ratio": (
                MIN_MEDIAN_TEST_DIVERSIFICATION_RATIO
            ),
        },
        "ensemble_feasible": bool(ensemble_feasible),
        "row_counts": dict(EXPECTED_ROW_COUNTS),
        "artifact_sha256": {},
        "forward_fill_used": False,
        "backward_fill_used": False,
        "interpolation_used": False,
        "covariance_repair_used": False,
        "covariance_shrinkage_used": False,
        "optimisation_performed": False,
        "ranking_performed": False,
        "winner_selection_performed": False,
        "sleeve_removal_performed": False,
        "leverage_used": False,
        "profitability_claimed": False,
    }
    _manifest_bytes(manifest)
    return manifest


def build_day15_strategy_diversification_report(
    results: StrategyDiversificationResults,
) -> Day15StrategyDiversificationReport:
    """Build deterministic neutral Day 15 evidence and Markdown."""

    if not isinstance(results, StrategyDiversificationResults):
        raise TypeError(
            "results must be a StrategyDiversificationResults object."
        )

    sleeve_inputs = _validated_table(
        results.sleeve_input_diagnostics,
        name="sleeve_input_diagnostics",
        columns=SLEEVE_INPUT_DIAGNOSTIC_COLUMNS,
        rows=6,
    )
    full_pairwise = _validated_table(
        results.full_sample_pairwise_correlations,
        name="full_sample_pairwise_correlations",
        columns=FULL_SAMPLE_PAIRWISE_CORRELATION_COLUMNS,
        rows=15,
    )
    full_covariance = _validated_table(
        results.full_sample_covariance_diagnostics,
        name="full_sample_covariance_diagnostics",
        columns=FULL_SAMPLE_COVARIANCE_DIAGNOSTIC_COLUMNS,
        rows=1,
    )
    fold_pairwise = _validated_table(
        results.fold_pairwise_correlations,
        name="fold_pairwise_correlations",
        columns=FOLD_PAIRWISE_CORRELATION_COLUMNS,
        rows=120,
    )
    fold_covariance = _validated_table(
        results.fold_covariance_diagnostics,
        name="fold_covariance_diagnostics",
        columns=FOLD_COVARIANCE_DIAGNOSTIC_COLUMNS,
        rows=8,
    )
    feasibility = _validated_table(
        results.ensemble_feasibility,
        name="ensemble_feasibility",
        columns=ENSEMBLE_FEASIBILITY_COLUMNS,
        rows=1,
    )

    _validate_row_order(
        sleeve_inputs=sleeve_inputs,
        full_pairwise=full_pairwise,
        fold_pairwise=fold_pairwise,
        fold_covariance=fold_covariance,
    )

    ensemble_feasible = bool(
        feasibility.iloc[0]["ensemble_feasible"]
    )
    manifest = _build_manifest(
        ensemble_feasible=ensemble_feasible
    )
    markdown = _render_report(
        sleeve_inputs=sleeve_inputs,
        full_pairwise=full_pairwise,
        full_covariance=full_covariance,
        fold_covariance=fold_covariance,
        feasibility=feasibility,
    )

    return Day15StrategyDiversificationReport(
        sleeve_input_diagnostics=sleeve_inputs,
        full_sample_pairwise_correlations=full_pairwise,
        fold_pairwise_correlations=fold_pairwise,
        fold_covariance_diagnostics=fold_covariance,
        ensemble_feasibility=feasibility,
        manifest=manifest,
        report=markdown,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize one evidence table deterministically."""

    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def _report_bytes(report: str) -> bytes:
    """Serialize Markdown with exactly one final newline."""

    return (report.rstrip() + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    """Return one SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def _artifact_payloads(
    report: Day15StrategyDiversificationReport,
) -> dict[str, bytes]:
    """Build every approved non-manifest payload in fixed order."""

    return {
        SLEEVE_INPUT_FILENAME: _csv_bytes(
            report.sleeve_input_diagnostics
        ),
        FULL_SAMPLE_PAIRWISE_FILENAME: _csv_bytes(
            report.full_sample_pairwise_correlations
        ),
        FOLD_PAIRWISE_FILENAME: _csv_bytes(
            report.fold_pairwise_correlations
        ),
        FOLD_COVARIANCE_FILENAME: _csv_bytes(
            report.fold_covariance_diagnostics
        ),
        ENSEMBLE_FEASIBILITY_FILENAME: _csv_bytes(
            report.ensemble_feasibility
        ),
        REPORT_FILENAME: _report_bytes(report.report),
    }


def _manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Serialize the manifest with deterministic strict JSON."""

    try:
        text = json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Day15StrategyDiversificationReportError(
            "Manifest must contain finite JSON-compatible values."
        ) from exc
    return (text.rstrip() + "\n").encode("utf-8")


def _validate_output_directory(
    output_directory: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Validate one exact destination without broad deletion."""

    if not isinstance(output_directory, (str, Path)):
        raise TypeError("output_directory must be a path.")
    if isinstance(output_directory, str) and not output_directory.strip():
        raise ValueError("output_directory cannot be empty.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    directory = Path(output_directory)
    if directory.name != "day15":
        raise ValueError(
            "output_directory must have the final name 'day15'."
        )
    if directory.is_symlink():
        raise ValueError("output_directory cannot be a symbolic link.")
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(
                "Day 15 output path exists but is not a directory."
            )
        if not overwrite:
            raise FileExistsError(
                f"Day 15 output already exists: {directory}."
            )
    return directory


def _replace_directory(*, staged: Path, destination: Path) -> None:
    """Atomically replace a complete directory with rollback protection."""

    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=".day15-backup-",
                dir=destination.parent,
            )
        )
        backup.rmdir()
        os.replace(destination, backup)

    try:
        os.replace(staged, destination)
    except Exception:
        if (
            backup is not None
            and backup.exists()
            and not destination.exists()
        ):
            os.replace(backup, destination)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def write_day15_strategy_diversification_artifacts(
    report: Day15StrategyDiversificationReport,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly seven deterministic Day 15 artifacts atomically."""

    if not isinstance(report, Day15StrategyDiversificationReport):
        raise TypeError(
            "report must be a Day15StrategyDiversificationReport."
        )
    directory = _validate_output_directory(
        output_directory,
        overwrite=overwrite,
    )
    directory.parent.mkdir(parents=True, exist_ok=True)

    payloads = _artifact_payloads(report)
    expected_non_manifest = set(APPROVED_DAY15_ARTIFACT_NAMES) - {
        MANIFEST_FILENAME
    }
    if set(payloads) != expected_non_manifest:
        raise RuntimeError("Day 15 artifact payload set is incomplete.")

    artifact_hashes = {
        name: _sha256_bytes(payload)
        for name, payload in payloads.items()
    }
    manifest = report.copy_manifest()
    manifest["artifact_sha256"] = {
        name: artifact_hashes[name]
        for name in sorted(artifact_hashes)
    }
    manifest_payload = _manifest_bytes(manifest)
    expected_payloads = {
        **payloads,
        MANIFEST_FILENAME: manifest_payload,
    }

    with tempfile.TemporaryDirectory(
        prefix=".day15-stage-",
        dir=directory.parent,
    ) as temporary:
        staged = Path(temporary) / "day15"
        staged.mkdir()

        for name in APPROVED_DAY15_ARTIFACT_NAMES:
            (staged / name).write_bytes(expected_payloads[name])

        staged_entries = tuple(item.name for item in staged.iterdir())
        if set(staged_entries) != set(APPROVED_DAY15_ARTIFACT_NAMES):
            raise RuntimeError("Staged Day 15 artifact set is incomplete.")
        if any(not (staged / name).is_file() for name in staged_entries):
            raise RuntimeError("Staged Day 15 entries must all be files.")

        for name, expected in expected_payloads.items():
            if (staged / name).read_bytes() != expected:
                raise RuntimeError(
                    f"Staged Day 15 bytes differ for {name}."
                )
        for name, digest in artifact_hashes.items():
            if _sha256_bytes((staged / name).read_bytes()) != digest:
                raise RuntimeError(
                    f"Staged Day 15 hash mismatch for {name}."
                )

        _replace_directory(staged=staged, destination=directory)

    final_entries = tuple(item.name for item in directory.iterdir())
    if set(final_entries) != set(APPROVED_DAY15_ARTIFACT_NAMES):
        raise RuntimeError("Final Day 15 artifact set is not approved.")
    for name, expected in expected_payloads.items():
        path = directory / name
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(
                f"Final Day 15 bytes differ for {name}."
            )

    return tuple(
        directory / name for name in APPROVED_DAY15_ARTIFACT_NAMES
    )
