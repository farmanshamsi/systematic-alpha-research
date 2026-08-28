import pandas as pd

from systematic_alpha.data.alpaca_provider import AlpacaBarProvider


class FakeResponse:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.df = frame


class FakeClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_stock_bars(self, request):
        return FakeResponse(self.frame)


def test_fetch_bars_normalizes_alpaca_frame() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (
                "SPY",
                pd.Timestamp("2025-12-15 14:30:00", tz="UTC"),
            )
        ],
        names=["symbol", "timestamp"],
    )

    alpaca_frame = pd.DataFrame(
        {
            "open": [680.0],
            "high": [681.0],
            "low": [679.5],
            "close": [680.5],
            "volume": [1000],
            "trade_count": [50],
            "vwap": [680.4],
        },
        index=index,
    )

    provider = AlpacaBarProvider(
        config={
            "broker": {
                "stock_data_feed": "iex",
            }
        },
        client=FakeClient(alpaca_frame),
    )

    result = provider.fetch_bars(
        symbols=["SPY"],
        start="2025-12-15T14:30:00Z",
        end="2025-12-15T14:31:00Z",
    )

    assert len(result) == 1
    assert result.loc[0, "symbol"] == "SPY"
    assert result.loc[0, "source"] == "alpaca"
    assert result.loc[0, "feed"] == "iex"
