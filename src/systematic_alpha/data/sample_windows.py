"""Development and locked-test sample-window controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


class SampleWindowError(ValueError):
    """Raised when sample boundaries or observations are invalid."""


def _as_utc_midnight(
    value: str | datetime | pd.Timestamp,
) -> pd.Timestamp:
    """Convert a date-like value to UTC midnight."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SampleWindowError(
            f"Invalid sample boundary: {value!r}."
        ) from exc

    if pd.isna(timestamp):
        raise SampleWindowError(
            f"Invalid sample boundary: {value!r}."
        )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.normalize()


@dataclass(frozen=True)
class SampleWindow:
    """Development and locked final-test sample boundaries.

    End dates supplied in configuration are inclusive calendar dates.
    Internally, end boundaries are represented as exclusive timestamps.
    """

    development_start: pd.Timestamp
    development_end_exclusive: pd.Timestamp
    final_test_start: pd.Timestamp
    final_test_end_exclusive: pd.Timestamp

    @classmethod
    def from_project_config(
        cls,
        config: dict[str, Any],
    ) -> "SampleWindow":
        """Create validated boundaries from project configuration."""

        try:
            sample = config["sample"]

            development_start = _as_utc_midnight(
                sample["development_start"]
            )

            development_end_exclusive = (
                _as_utc_midnight(sample["development_end"])
                + pd.Timedelta(days=1)
            )

            final_test_start = _as_utc_midnight(
                sample["final_test_start"]
            )

            final_test_end_exclusive = (
                _as_utc_midnight(sample["final_test_end"])
                + pd.Timedelta(days=1)
            )
        except KeyError as exc:
            raise SampleWindowError(
                f"Missing sample configuration key: {exc.args[0]}"
            ) from exc

        if development_start >= development_end_exclusive:
            raise SampleWindowError(
                "Development start must precede development end."
            )

        if final_test_start >= final_test_end_exclusive:
            raise SampleWindowError(
                "Final-test start must precede final-test end."
            )

        if development_end_exclusive > final_test_start:
            raise SampleWindowError(
                "Development and final-test periods overlap."
            )

        return cls(
            development_start=development_start,
            development_end_exclusive=development_end_exclusive,
            final_test_start=final_test_start,
            final_test_end_exclusive=final_test_end_exclusive,
        )

    @property
    def development_end_inclusive(self) -> pd.Timestamp:
        """Return the final included development calendar date."""

        return (
            self.development_end_exclusive
            - pd.Timedelta(days=1)
        )

    @property
    def final_test_end_inclusive(self) -> pd.Timestamp:
        """Return the final included locked-test calendar date."""

        return (
            self.final_test_end_exclusive
            - pd.Timedelta(days=1)
        )

    def development_request_bounds(
        self,
    ) -> tuple[datetime, datetime]:
        """Return inclusive-start and exclusive-end API boundaries."""

        return (
            self.development_start.to_pydatetime(),
            self.development_end_exclusive.to_pydatetime(),
        )

    def validate_development_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """Validate that every row belongs to development data.

        The function does not silently remove out-of-window rows.
        Any observation outside the development period raises an error.
        """

        if frame.empty:
            raise SampleWindowError(
                "Development dataset cannot be empty."
            )

        if timestamp_column not in frame.columns:
            raise SampleWindowError(
                f"Timestamp column not found: {timestamp_column!r}"
            )

        result = frame.copy()

        result[timestamp_column] = pd.to_datetime(
            result[timestamp_column],
            utc=True,
            errors="raise",
        )

        timestamps = result[timestamp_column]

        before_development = (
            timestamps < self.development_start
        )

        after_development = (
            timestamps >= self.development_end_exclusive
        )

        outside_development = (
            before_development | after_development
        )

        locked_final_test = (
            (timestamps >= self.final_test_start)
            & (timestamps < self.final_test_end_exclusive)
        )

        if outside_development.any():
            outside_count = int(outside_development.sum())
            locked_count = int(locked_final_test.sum())

            outside_rows = timestamps[outside_development]

            raise SampleWindowError(
                "Dataset contains observations outside the "
                "development period. "
                f"Outside rows: {outside_count}; "
                f"locked final-test rows: {locked_count}; "
                f"minimum outside timestamp: "
                f"{outside_rows.min().isoformat()}; "
                f"maximum outside timestamp: "
                f"{outside_rows.max().isoformat()}."
            )

        sort_columns = [timestamp_column]

        if "symbol" in result.columns:
            sort_columns = ["symbol", timestamp_column]

        return result.sort_values(
            sort_columns,
            kind="stable",
        ).reset_index(drop=True)
