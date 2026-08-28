import pandas as pd
import pytest

from systematic_alpha.data.sample_windows import (
    SampleWindow,
    SampleWindowError,
)


def sample_config() -> dict:
    return {
        "sample": {
            "development_start": "2020-01-02",
            "development_end": "2025-12-31",
            "final_test_start": "2026-01-02",
            "final_test_end": "2026-06-30",
        }
    }


def test_config_dates_become_exclusive_end_boundaries() -> None:
    window = SampleWindow.from_project_config(
        sample_config()
    )

    assert window.development_start == pd.Timestamp(
        "2020-01-02T00:00:00Z"
    )

    assert window.development_end_exclusive == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )

    assert window.final_test_start == pd.Timestamp(
        "2026-01-02T00:00:00Z"
    )

    assert window.final_test_end_exclusive == pd.Timestamp(
        "2026-07-01T00:00:00Z"
    )


def test_valid_development_rows_are_accepted_and_sorted() -> None:
    window = SampleWindow.from_project_config(
        sample_config()
    )

    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-31T20:45:00Z",
                "2020-01-02T14:30:00Z",
            ],
            "symbol": ["SPY", "SPY"],
            "close": [680.0, 320.0],
        }
    )

    result = window.validate_development_frame(frame)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2020-01-02T14:30:00Z"),
        pd.Timestamp("2025-12-31T20:45:00Z"),
    ]


def test_locked_final_test_row_is_rejected() -> None:
    window = SampleWindow.from_project_config(
        sample_config()
    )

    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-31T20:45:00Z",
                "2026-01-02T14:30:00Z",
            ],
            "symbol": ["SPY", "SPY"],
        }
    )

    with pytest.raises(
        SampleWindowError,
        match="locked final-test rows: 1",
    ):
        window.validate_development_frame(frame)


def test_predevelopment_row_is_rejected() -> None:
    window = SampleWindow.from_project_config(
        sample_config()
    )

    frame = pd.DataFrame(
        {
            "timestamp": [
                "2019-12-31T20:45:00Z",
            ],
            "symbol": ["SPY"],
        }
    )

    with pytest.raises(
        SampleWindowError,
        match="outside the development period",
    ):
        window.validate_development_frame(frame)


def test_overlapping_sample_periods_are_rejected() -> None:
    config = sample_config()

    config["sample"]["final_test_start"] = "2025-12-01"

    with pytest.raises(
        SampleWindowError,
        match="periods overlap",
    ):
        SampleWindow.from_project_config(config)


@pytest.mark.parametrize("invalid_value", [None, "not-a-date", pd.NaT])
def test_invalid_sample_boundaries_fail_closed(invalid_value: object) -> None:
    config = sample_config()
    config["sample"]["development_start"] = invalid_value

    with pytest.raises(SampleWindowError, match="Invalid sample boundary"):
        SampleWindow.from_project_config(config)


def test_empty_development_frame_is_rejected() -> None:
    window = SampleWindow.from_project_config(
        sample_config()
    )

    with pytest.raises(
        SampleWindowError,
        match="cannot be empty",
    ):
        window.validate_development_frame(
            pd.DataFrame(columns=["timestamp"])
        )
