import pandas as pd
import pytest

from systematic_alpha.data.development_dataset import (
    DevelopmentDatasetError,
    build_yearly_development_chunks,
    combine_canonical_bar_chunks,
)
from systematic_alpha.data.sample_windows import SampleWindow


def sample_window() -> SampleWindow:
    return SampleWindow.from_project_config(
        {
            "sample": {
                "development_start": "2020-01-02",
                "development_end": "2025-12-31",
                "final_test_start": "2026-01-02",
                "final_test_end": "2026-06-30",
            }
        }
    )


def make_bars(
    *,
    symbol: str,
    timestamps: list[str],
) -> pd.DataFrame:
    row_count = len(timestamps)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * row_count,
            "open": [100.0] * row_count,
            "high": [101.0] * row_count,
            "low": [99.0] * row_count,
            "close": [100.5] * row_count,
            "volume": [1000.0] * row_count,
            "trade_count": [100] * row_count,
            "vwap": [100.25] * row_count,
            "source": ["alpaca"] * row_count,
            "feed": ["iex"] * row_count,
        }
    )


def test_yearly_chunks_cover_development_window() -> None:
    chunks = build_yearly_development_chunks(
        sample_window()
    )

    assert [chunk.label for chunk in chunks] == [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]

    assert chunks[0].start == pd.Timestamp(
        "2020-01-02T00:00:00Z"
    )

    assert chunks[-1].end_exclusive == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )

    for previous, following in zip(
        chunks,
        chunks[1:],
    ):
        assert previous.end_exclusive == following.start


def test_chunks_are_combined_and_sorted() -> None:
    frames = [
        make_bars(
            symbol="SPY",
            timestamps=["2021-01-04T14:30:00Z"],
        ),
        make_bars(
            symbol="QQQ",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
        make_bars(
            symbol="IWM",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
    ]

    result = combine_canonical_bar_chunks(
        frames,
        window=sample_window(),
        expected_symbols=["SPY", "QQQ", "IWM"],
    )

    assert result["symbol"].tolist() == [
        "IWM",
        "QQQ",
        "SPY",
    ]

    assert set(result["symbol"]) == {
        "SPY",
        "QQQ",
        "IWM",
    }


def test_duplicate_symbol_timestamp_is_rejected() -> None:
    duplicate = make_bars(
        symbol="SPY",
        timestamps=[
            "2020-01-02T14:30:00Z",
            "2020-01-02T14:30:00Z",
        ],
    )

    frames = [
        duplicate,
        make_bars(
            symbol="QQQ",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
        make_bars(
            symbol="IWM",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
    ]

    with pytest.raises(
        DevelopmentDatasetError,
        match="Duplicate symbol/timestamp",
    ):
        combine_canonical_bar_chunks(
            frames,
            window=sample_window(),
            expected_symbols=["SPY", "QQQ", "IWM"],
        )


def test_missing_expected_symbol_is_rejected() -> None:
    frames = [
        make_bars(
            symbol="SPY",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
        make_bars(
            symbol="QQQ",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
    ]

    with pytest.raises(
        DevelopmentDatasetError,
        match="Missing: \\['IWM'\\]",
    ):
        combine_canonical_bar_chunks(
            frames,
            window=sample_window(),
            expected_symbols=["SPY", "QQQ", "IWM"],
        )


def test_locked_test_observation_is_rejected() -> None:
    frames = [
        make_bars(
            symbol="SPY",
            timestamps=["2026-01-02T14:30:00Z"],
        ),
        make_bars(
            symbol="QQQ",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
        make_bars(
            symbol="IWM",
            timestamps=["2020-01-02T14:30:00Z"],
        ),
    ]

    with pytest.raises(
        DevelopmentDatasetError,
        match="violated the development window",
    ):
        combine_canonical_bar_chunks(
            frames,
            window=sample_window(),
            expected_symbols=["SPY", "QQQ", "IWM"],
        )


def test_monthly_chunks_cover_entire_development_window() -> None:
    from systematic_alpha.data.development_dataset import (
        build_monthly_development_chunks,
    )

    chunks = build_monthly_development_chunks(
        sample_window()
    )

    assert len(chunks) == 72
    assert chunks[0].label == "2020-01"
    assert chunks[-1].label == "2025-12"

    assert chunks[0].start == pd.Timestamp(
        "2020-01-02T00:00:00Z"
    )

    assert chunks[-1].end_exclusive == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )

    for previous, following in zip(
        chunks,
        chunks[1:],
    ):
        assert previous.end_exclusive == following.start


def test_regular_session_filter_removes_extended_hours() -> None:
    from systematic_alpha.data.development_dataset import (
        filter_regular_session_bars,
    )

    frame = make_bars(
        symbol="SPY",
        timestamps=[
            "2020-01-02T14:15:00Z",
            "2020-01-02T14:30:00Z",
            "2020-01-02T20:45:00Z",
            "2020-01-02T21:00:00Z",
        ],
    )

    result = filter_regular_session_bars(frame)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2020-01-02T14:30:00Z"),
        pd.Timestamp("2020-01-02T20:45:00Z"),
    ]


def test_complete_month_passes_edge_coverage() -> None:
    from systematic_alpha.data.development_dataset import (
        DevelopmentChunk,
        validate_chunk_edge_coverage,
    )

    chunk = DevelopmentChunk(
        label="2020-01",
        start=pd.Timestamp("2020-01-02T00:00:00Z"),
        end_exclusive=pd.Timestamp(
            "2020-02-01T00:00:00Z"
        ),
    )

    frames = []

    for symbol in ("SPY", "QQQ", "IWM"):
        frames.append(
            make_bars(
                symbol=symbol,
                timestamps=[
                    "2020-01-02T14:30:00Z",
                    "2020-01-31T20:45:00Z",
                ],
            )
        )

    validate_chunk_edge_coverage(
        pd.concat(frames, ignore_index=True),
        chunk=chunk,
        expected_symbols=["SPY", "QQQ", "IWM"],
    )


def test_truncated_year_fails_edge_coverage() -> None:
    from systematic_alpha.data.development_dataset import (
        DevelopmentChunk,
        validate_chunk_edge_coverage,
    )

    chunk = DevelopmentChunk(
        label="2020",
        start=pd.Timestamp("2020-01-02T00:00:00Z"),
        end_exclusive=pd.Timestamp(
            "2021-01-01T00:00:00Z"
        ),
    )

    frames = []

    for symbol in ("SPY", "QQQ", "IWM"):
        frames.append(
            make_bars(
                symbol=symbol,
                timestamps=[
                    "2020-07-27T13:30:00Z",
                    "2020-12-31T20:45:00Z",
                ],
            )
        )

    with pytest.raises(
        DevelopmentDatasetError,
        match="failed edge-coverage",
    ):
        validate_chunk_edge_coverage(
            pd.concat(frames, ignore_index=True),
            chunk=chunk,
            expected_symbols=["SPY", "QQQ", "IWM"],
        )


def test_exchange_calendar_filter_handles_early_close() -> None:
    from systematic_alpha.data.development_dataset import (
        filter_regular_session_bars,
    )

    frame = make_bars(
        symbol="SPY",
        timestamps=[
            "2020-11-27T14:30:00Z",
            "2020-11-27T17:45:00Z",
            "2020-11-27T18:00:00Z",
            "2020-11-27T20:45:00Z",
        ],
    )

    result = filter_regular_session_bars(frame)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2020-11-27T14:30:00Z"),
        pd.Timestamp("2020-11-27T17:45:00Z"),
    ]


def test_exchange_calendar_filter_removes_holiday() -> None:
    from systematic_alpha.data.development_dataset import (
        filter_regular_session_bars,
    )

    frame = pd.concat(
        [
            make_bars(
                symbol="SPY",
                timestamps=[
                    "2020-12-24T14:30:00Z",
                ],
            ),
            make_bars(
                symbol="SPY",
                timestamps=[
                    "2020-12-25T14:30:00Z",
                ],
            ),
        ],
        ignore_index=True,
    )

    result = filter_regular_session_bars(frame)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2020-12-24T14:30:00Z"),
    ]


def test_complete_grid_accepts_early_close_session() -> None:
    from systematic_alpha.data.development_dataset import (
        DevelopmentChunk,
        validate_complete_session_grid,
    )

    timestamps = pd.date_range(
        start="2020-11-27T14:30:00Z",
        end="2020-11-27T18:00:00Z",
        freq="15min",
        inclusive="left",
    )

    frame = pd.concat(
        [
            make_bars(
                symbol=symbol,
                timestamps=[
                    timestamp.isoformat()
                    for timestamp in timestamps
                ],
            )
            for symbol in ("SPY", "QQQ", "IWM")
        ],
        ignore_index=True,
    )

    chunk = DevelopmentChunk(
        label="2020-11-27",
        start=pd.Timestamp(
            "2020-11-27T00:00:00Z"
        ),
        end_exclusive=pd.Timestamp(
            "2020-11-28T00:00:00Z"
        ),
    )

    validate_complete_session_grid(
        frame,
        chunk=chunk,
        expected_symbols=("SPY", "QQQ", "IWM"),
    )


def test_complete_grid_rejects_missing_bar() -> None:
    from systematic_alpha.data.development_dataset import (
        DevelopmentChunk,
        DevelopmentDatasetError,
        validate_complete_session_grid,
    )

    timestamps = pd.date_range(
        start="2020-11-27T14:30:00Z",
        end="2020-11-27T18:00:00Z",
        freq="15min",
        inclusive="left",
    )

    frames = []

    for symbol in ("SPY", "QQQ", "IWM"):
        symbol_timestamps = list(timestamps)

        if symbol == "IWM":
            symbol_timestamps = symbol_timestamps[:-1]

        frames.append(
            make_bars(
                symbol=symbol,
                timestamps=[
                    timestamp.isoformat()
                    for timestamp in symbol_timestamps
                ],
            )
        )

    chunk = DevelopmentChunk(
        label="2020-11-27",
        start=pd.Timestamp(
            "2020-11-27T00:00:00Z"
        ),
        end_exclusive=pd.Timestamp(
            "2020-11-28T00:00:00Z"
        ),
    )

    with pytest.raises(
        DevelopmentDatasetError,
        match="Incomplete official exchange-session grid",
    ):
        validate_complete_session_grid(
            pd.concat(frames, ignore_index=True),
            chunk=chunk,
            expected_symbols=("SPY", "QQQ", "IWM"),
        )
