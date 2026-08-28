"""Deterministic reporting for Day 14 cointegration feasibility."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Final, Mapping

import pandas as pd

from systematic_alpha.analysis.cointegration_feasibility import (
    CANDIDATE_PAIRS,
    COINTEGRATION_DIAGNOSTIC_COLUMNS,
    FOLD_STABILITY_COLUMNS,
    OU_DIAGNOSTIC_COLUMNS,
    PAIR_ELIGIBILITY_COLUMNS,
    PAIR_INPUT_DIAGNOSTIC_COLUMNS,
    SERIES_INTEGRATION_COLUMNS,
    CointegrationFeasibilityResults,
)


DAY14_ARTIFACT_VERSION: Final[str] = (
    "cointegration_ou_feasibility_v1"
)

PAIR_INPUT_FILENAME: Final[str] = (
    "pair_input_diagnostics.csv"
)
SERIES_INTEGRATION_FILENAME: Final[str] = (
    "series_integration_diagnostics.csv"
)
COINTEGRATION_FILENAME: Final[str] = (
    "cointegration_diagnostics.csv"
)
FOLD_STABILITY_FILENAME: Final[str] = (
    "fold_stability_diagnostics.csv"
)
OU_DIAGNOSTICS_FILENAME: Final[str] = (
    "ou_diagnostics.csv"
)
PAIR_ELIGIBILITY_FILENAME: Final[str] = (
    "pair_eligibility.csv"
)
MANIFEST_FILENAME: Final[str] = "manifest.json"
REPORT_FILENAME: Final[str] = "report.md"

APPROVED_DAY14_ARTIFACT_NAMES: Final[
    tuple[str, ...]
] = (
    PAIR_INPUT_FILENAME,
    SERIES_INTEGRATION_FILENAME,
    COINTEGRATION_FILENAME,
    FOLD_STABILITY_FILENAME,
    OU_DIAGNOSTICS_FILENAME,
    PAIR_ELIGIBILITY_FILENAME,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
)


def _copy_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Return a defensive zero-based copy."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    return frame.copy(deep=True).reset_index(
        drop=True
    )


def _validated_table(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
    rows: int,
) -> pd.DataFrame:
    """Validate one exact Day 14 evidence table."""

    retained = _copy_frame(
        frame,
        name=name,
    )

    if tuple(retained.columns) != columns:
        raise ValueError(
            f"{name} has an unexpected schema."
        )

    if len(retained) != rows:
        raise ValueError(
            f"{name} must contain exactly {rows} rows."
        )

    return retained


def _freeze_manifest(
    value: object,
) -> object:
    """Recursively freeze mutable manifest containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_manifest(item)
                for key, item in deepcopy(
                    dict(value)
                ).items()
            }
        )

    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_manifest(item)
            for item in deepcopy(value)
        )

    return deepcopy(value)


def _copy_manifest(
    value: object,
) -> object:
    """Return mutable copies of frozen manifest values."""

    if isinstance(value, Mapping):
        return {
            str(key): _copy_manifest(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            _copy_manifest(item)
            for item in value
        ]

    return deepcopy(value)


def _eligibility_lines(
    eligibility: pd.DataFrame,
) -> str:
    """Render the three predeclared pair outcomes."""

    lines = [
        "| Pair | Eligible | Rejection reasons |",
        "|---|---:|---|",
    ]

    for row in eligibility.itertuples(
        index=False
    ):
        reasons = (
            row.rejection_reasons
            if row.rejection_reasons
            else "None"
        )
        lines.append(
            f"| {row.pair_id} | "
            f"{'Yes' if row.eligible else 'No'} | "
            f"{reasons} |"
        )

    return "\n".join(lines)


def _render_report(
    *,
    eligibility: pd.DataFrame,
    integration: pd.DataFrame,
    cointegration: pd.DataFrame,
    ou: pd.DataFrame,
) -> str:
    """Render a neutral development-only Day 14 report."""

    eligible_count = int(
        eligibility["eligible"]
        .astype(bool)
        .sum()
    )
    holm_count = int(
        cointegration["holm_reject"]
        .astype(bool)
        .sum()
    )
    ou_attempted_count = int(
        ou["attempted"]
        .astype(bool)
        .sum()
    )
    i1_symbols = int(
        integration.groupby(
            "symbol"
        )["plausibly_i1"]
        .all()
        .sum()
    )

    return f"""# Day 14 — Cointegration and OU Feasibility

## 1. Purpose

This study evaluates whether any predeclared ETF pair satisfies the frozen
development-period statistical and OU feasibility gates. It does not define,
simulate or assess a trading strategy.

## 2. Frozen candidate universe

- SPY / QQQ: SPY is Y and QQQ is X
- SPY / IWM: SPY is Y and IWM is X
- QQQ / IWM: QQQ is Y and IWM is X

Reverse orientations were not tested for eligibility.

## 3. Data and alignment contract

Engle–Granger estimation uses daily log session-close prices derived from the
canonical 15-minute development data. Daily observations are joined by exact
session date. Intraday observations are joined by exact timestamp and session.
No forward filling, interpolation or asynchronous matching is permitted.

## 4. Statistical contract

Individual series are assessed for plausible I(1) behaviour using the frozen
ADF level and first-difference specifications. Engle–Granger inference uses an
intercept and Holm correction across the three candidate tests. Hedge-ratio
and expanding-fold stability gates are applied before OU estimation.

## 5. Aggregate diagnostic counts

- Plausibly I(1) symbols: {i1_symbols} of 3
- Holm-adjusted cointegration passes: {holm_count} of 3
- OU estimations attempted: {ou_attempted_count} of 3
- Fully eligible pairs: {eligible_count} of 3

## 6. Pair eligibility

{_eligibility_lines(eligibility)}

## 7. Interpretation boundary

No profitability criterion, transaction-cost criterion, trading threshold,
position rule, pair ranking or winner selection is included in Day 14.
Eligibility is a statistical feasibility outcome only. A result in which no
pair qualifies is valid.

## 8. Locked-period protection

The January–June 2026 final evaluation period was not accessed.

## 9. Conclusion

The evidence records pass/fail outcomes for every predeclared pair without
using returns or profitability to rank the candidates. Any subsequent ECM,
signal, execution or portfolio work remains outside Day 14.
"""


@dataclass(frozen=True, slots=True)
class Day14CointegrationReport:
    """Immutable Day 14 report and evidence bundle."""

    pair_input_diagnostics: pd.DataFrame
    series_integration_diagnostics: pd.DataFrame
    cointegration_diagnostics: pd.DataFrame
    fold_stability_diagnostics: pd.DataFrame
    ou_diagnostics: pd.DataFrame
    pair_eligibility: pd.DataFrame
    manifest: Mapping[str, object]
    report: str

    def __post_init__(self) -> None:
        """Defensively retain report evidence."""

        for name in (
            "pair_input_diagnostics",
            "series_integration_diagnostics",
            "cointegration_diagnostics",
            "fold_stability_diagnostics",
            "ou_diagnostics",
            "pair_eligibility",
        ):
            object.__setattr__(
                self,
                name,
                _copy_frame(
                    getattr(self, name),
                    name=name,
                ),
            )

        object.__setattr__(
            self,
            "manifest",
            _freeze_manifest(self.manifest),
        )
        object.__setattr__(
            self,
            "report",
            str(self.report),
        )

    def copy_manifest(
        self,
    ) -> dict[str, object]:
        """Return a mutable manifest copy."""

        copied = _copy_manifest(self.manifest)

        if not isinstance(copied, dict):
            raise TypeError(
                "Copied manifest must be a dictionary."
            )

        return copied


def build_day14_cointegration_report(
    results: CointegrationFeasibilityResults,
) -> Day14CointegrationReport:
    """Build deterministic neutral Day 14 evidence."""

    if not isinstance(
        results,
        CointegrationFeasibilityResults,
    ):
        raise TypeError(
            "results must be a "
            "CointegrationFeasibilityResults object."
        )

    pair_inputs = _validated_table(
        results.pair_input_diagnostics,
        name="pair_input_diagnostics",
        columns=PAIR_INPUT_DIAGNOSTIC_COLUMNS,
        rows=3,
    )
    integration = _validated_table(
        results.series_integration_diagnostics,
        name="series_integration_diagnostics",
        columns=SERIES_INTEGRATION_COLUMNS,
        rows=6,
    )
    cointegration = _validated_table(
        results.cointegration_diagnostics,
        name="cointegration_diagnostics",
        columns=COINTEGRATION_DIAGNOSTIC_COLUMNS,
        rows=3,
    )
    folds = _validated_table(
        results.fold_stability_diagnostics,
        name="fold_stability_diagnostics",
        columns=FOLD_STABILITY_COLUMNS,
        rows=12,
    )
    ou = _validated_table(
        results.ou_diagnostics,
        name="ou_diagnostics",
        columns=OU_DIAGNOSTIC_COLUMNS,
        rows=3,
    )
    eligibility = _validated_table(
        results.pair_eligibility,
        name="pair_eligibility",
        columns=PAIR_ELIGIBILITY_COLUMNS,
        rows=3,
    )

    eligible_count = int(
        eligibility["eligible"]
        .astype(bool)
        .sum()
    )

    manifest = {
        "report_id": (
            "day14_cointegration_feasibility"
        ),
        "artifact_version": (
            DAY14_ARTIFACT_VERSION
        ),
        "schema_version": 1,
        "artifact_filenames": list(
            APPROVED_DAY14_ARTIFACT_NAMES
        ),
        "development_only": True,
        "frequency": "15min",
        "candidate_pairs": [
            {
                "y_symbol": y_symbol,
                "x_symbol": x_symbol,
            }
            for y_symbol, x_symbol in (
                CANDIDATE_PAIRS
            )
        ],
        "candidate_pair_count": 3,
        "eligible_pair_count": (
            eligible_count
        ),
        "locked_period_accessed": False,
        "forward_fill_used": False,
        "tuning_performed": False,
        "ranking_performed": False,
        "winner_selection_performed": False,
        "row_counts": {
            "pair_input_diagnostics": 3,
            "series_integration_diagnostics": 6,
            "cointegration_diagnostics": 3,
            "fold_stability_diagnostics": 12,
            "ou_diagnostics": 3,
            "pair_eligibility": 3,
        },
        "artifact_sha256": {},
    }

    markdown = _render_report(
        eligibility=eligibility,
        integration=integration,
        cointegration=cointegration,
        ou=ou,
    )

    return Day14CointegrationReport(
        pair_input_diagnostics=pair_inputs,
        series_integration_diagnostics=(
            integration
        ),
        cointegration_diagnostics=(
            cointegration
        ),
        fold_stability_diagnostics=folds,
        ou_diagnostics=ou,
        pair_eligibility=eligibility,
        manifest=manifest,
        report=markdown,
    )

def _csv_bytes(
    frame: pd.DataFrame,
) -> bytes:
    """Serialise one evidence table deterministically."""

    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def _report_bytes(
    report: str,
) -> bytes:
    """Serialise Markdown with one final newline."""

    return (
        report.rstrip()
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(
    payload: bytes,
) -> str:
    """Return one SHA-256 digest."""

    return hashlib.sha256(
        payload
    ).hexdigest()


def _artifact_payloads(
    report: Day14CointegrationReport,
) -> dict[str, bytes]:
    """Build every non-manifest payload in fixed order."""

    return {
        PAIR_INPUT_FILENAME: _csv_bytes(
            report.pair_input_diagnostics
        ),
        SERIES_INTEGRATION_FILENAME: _csv_bytes(
            report.series_integration_diagnostics
        ),
        COINTEGRATION_FILENAME: _csv_bytes(
            report.cointegration_diagnostics
        ),
        FOLD_STABILITY_FILENAME: _csv_bytes(
            report.fold_stability_diagnostics
        ),
        OU_DIAGNOSTICS_FILENAME: _csv_bytes(
            report.ou_diagnostics
        ),
        PAIR_ELIGIBILITY_FILENAME: _csv_bytes(
            report.pair_eligibility
        ),
        REPORT_FILENAME: _report_bytes(
            report.report
        ),
    }


def _manifest_bytes(
    manifest: Mapping[str, object],
) -> bytes:
    """Serialise the manifest canonically."""

    return (
        json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_output_directory(
    directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Validate one requested Day 14 destination."""

    if directory.exists():
        if not directory.is_dir():
            raise ValueError(
                "Day 14 output path exists but is not a directory."
            )

        if not overwrite:
            raise FileExistsError(
                f"Day 14 output already exists: {directory}."
            )


def _replace_directory(
    *,
    staged: Path,
    destination: Path,
) -> None:
    """Replace a complete directory with rollback protection."""

    backup: Path | None = None

    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=".day14-backup-",
                dir=destination.parent,
            )
        )
        backup.rmdir()

        os.replace(
            destination,
            backup,
        )

    try:
        os.replace(
            staged,
            destination,
        )
    except Exception:
        if (
            backup is not None
            and backup.exists()
            and not destination.exists()
        ):
            os.replace(
                backup,
                destination,
            )

        raise
    else:
        if (
            backup is not None
            and backup.exists()
        ):
            shutil.rmtree(backup)


def write_day14_cointegration_artifacts(
    report: Day14CointegrationReport,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the exact approved Day 14 artifact set safely."""

    if not isinstance(
        report,
        Day14CointegrationReport,
    ):
        raise TypeError(
            "report must be a Day14CointegrationReport."
        )

    if not isinstance(
        output_directory,
        (str, Path),
    ):
        raise TypeError(
            "output_directory must be a path."
        )

    if not isinstance(overwrite, bool):
        raise TypeError(
            "overwrite must be a boolean."
        )

    directory = Path(output_directory)

    _validate_output_directory(
        directory,
        overwrite=overwrite,
    )

    directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payloads = _artifact_payloads(
        report
    )
    expected_payload_names = set(
        APPROVED_DAY14_ARTIFACT_NAMES
    ) - {MANIFEST_FILENAME}

    if set(payloads) != expected_payload_names:
        raise RuntimeError(
            "Day 14 artifact payload set is incomplete."
        )

    artifact_hashes = {
        name: _sha256_bytes(payload)
        for name, payload in payloads.items()
    }

    manifest = report.copy_manifest()
    manifest["artifact_sha256"] = {
        name: artifact_hashes[name]
        for name in sorted(
            artifact_hashes
        )
    }
    manifest_payload = _manifest_bytes(
        manifest
    )

    with tempfile.TemporaryDirectory(
        prefix=".day14-stage-",
        dir=directory.parent,
    ) as temporary:
        staged = (
            Path(temporary)
            / "day14"
        )
        staged.mkdir()

        for name, payload in payloads.items():
            (staged / name).write_bytes(
                payload
            )

        (
            staged / MANIFEST_FILENAME
        ).write_bytes(
            manifest_payload
        )

        staged_names = {
            item.name
            for item in staged.iterdir()
            if item.is_file()
        }

        if staged_names != set(
            APPROVED_DAY14_ARTIFACT_NAMES
        ):
            raise RuntimeError(
                "Staged Day 14 artifact set is incomplete."
            )

        for name, digest in artifact_hashes.items():
            actual = _sha256_bytes(
                (staged / name).read_bytes()
            )

            if actual != digest:
                raise RuntimeError(
                    f"Staged artifact hash mismatch: {name}."
                )

        _replace_directory(
            staged=staged,
            destination=directory,
        )

    paths = tuple(
        directory / name
        for name in (
            APPROVED_DAY14_ARTIFACT_NAMES
        )
    )

    if any(
        not artifact.is_file()
        or artifact.stat().st_size <= 0
        for artifact in paths
    ):
        raise RuntimeError(
            "Written Day 14 artifact set is incomplete."
        )

    return paths
