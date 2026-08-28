"""Shared input-validation helpers for analytical modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import numpy as np
import pandas as pd


ColumnValidator = Callable[
    [pd.DataFrame, Iterable[str]],
    None,
]

NumericCleaner = Callable[..., pd.Series]


def make_column_validator(
    error_type: type[ValueError],
) -> Callable[..., None]:
    """Bind a module-specific exception to column validation."""

    def require_columns(
        frame: pd.DataFrame,
        required: Iterable[str],
        *,
        context: str,
    ) -> None:
        missing = set(required).difference(
            frame.columns
        )

        if missing:
            raise error_type(
                f"{context} is missing required columns: "
                f"{sorted(missing)}"
            )

    return require_columns


def make_numeric_cleaner(
    error_type: type[ValueError],
) -> Callable[..., pd.Series]:
    """Bind a module-specific exception to numeric cleaning."""

    def clean_numeric_series(
        values: pd.Series,
        *,
        minimum_observations: int = 8,
    ) -> pd.Series:
        clean = pd.to_numeric(
            values,
            errors="coerce",
        )

        clean = (
            clean.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .astype("float64")
        )

        if len(clean) < minimum_observations:
            raise error_type(
                "Insufficient finite observations. "
                f"Required at least {minimum_observations}; "
                f"received {len(clean)}."
            )

        return clean

    return clean_numeric_series
