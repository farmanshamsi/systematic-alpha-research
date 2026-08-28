import pandas as pd

from systematic_alpha.data.yahoo_provider import YahooDailyProvider


def test_yahoo_provider_normalizes_daily_data() -> None:
    calls: list[tuple[str, dict]] = []

    def fake_download(
        ticker: str,
        **kwargs,
    ) -> pd.DataFrame:
        calls.append((ticker, kwargs))

        return pd.DataFrame(
            {
                "Open": [680.0],
                "High": [686.0],
                "Low": [678.0],
                "Close": [685.0],
                "Volume": [50_000_000],
            },
            index=pd.DatetimeIndex(
                ["2025-12-15"],
                name="Date",
            ),
        )

    provider = YahooDailyProvider(
        downloader=fake_download,
    )

    result = provider.fetch_daily_bundle(
        symbols=["spy"],
        start="2025-12-15",
        end="2025-12-16",
    )

    assert len(result.raw) == 1
    assert result.raw.loc[0, "requested_symbol"] == "SPY"

    assert len(result.normalized) == 1
    assert result.normalized.loc[0, "symbol"] == "SPY"
    assert result.normalized.loc[0, "source"] == "yfinance"
    assert result.normalized.loc[0, "feed"] == "yahoo"

    ticker, arguments = calls[0]

    assert ticker == "SPY"
    assert arguments["start"] == "2025-12-15"
    assert arguments["end"] == "2025-12-16"
    assert arguments["interval"] == "1d"
    assert arguments["auto_adjust"] is True
    assert arguments["multi_level_index"] is False

    assert (
        result.request_metadata["end_exclusive"]
        == "2025-12-16"
    )
