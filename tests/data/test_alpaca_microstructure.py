import pandas as pd

from systematic_alpha.data.alpaca_microstructure import (
    AlpacaMicrostructureProvider,
)


class FakeResponse:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.df = frame


class FakeClient:
    def __init__(
        self,
        quotes: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> None:
        self.quotes = quotes
        self.trades = trades

    def get_stock_quotes(self, request):
        return FakeResponse(self.quotes)

    def get_stock_trades(self, request):
        return FakeResponse(self.trades)


def make_provider() -> AlpacaMicrostructureProvider:
    shared_timestamp = pd.Timestamp(
        "2025-12-15T14:30:00Z"
    )

    quote_index = pd.MultiIndex.from_tuples(
        [
            ("SPY", shared_timestamp),
            ("SPY", shared_timestamp),
        ],
        names=["symbol", "timestamp"],
    )

    trade_index = pd.MultiIndex.from_tuples(
        [
            ("SPY", shared_timestamp),
            ("SPY", shared_timestamp),
        ],
        names=["symbol", "timestamp"],
    )

    quotes = pd.DataFrame(
        {
            "bid_price": [679.49, 679.50],
            "bid_size": [100, 120],
            "bid_exchange": ["V", "V"],
            "ask_price": [679.50, 679.51],
            "ask_size": [200, 180],
            "ask_exchange": ["V", "V"],
            "conditions": [["R"], ["R"]],
            "tape": ["C", "C"],
        },
        index=quote_index,
    )

    trades = pd.DataFrame(
        {
            "id": ["trade-1", "trade-2"],
            "price": [679.50, 679.51],
            "size": [50, 25],
            "exchange": ["V", "V"],
            "conditions": [["@"], ["I"]],
            "tape": ["C", "C"],
        },
        index=trade_index,
    )

    return AlpacaMicrostructureProvider(
        config={
            "broker": {
                "stock_data_feed": "iex",
            }
        },
        client=FakeClient(quotes, trades),
    )


def test_fetch_quotes_bundle_preserves_raw_and_optional_fields() -> None:
    provider = make_provider()

    result = provider.fetch_quotes_bundle(
        symbols=["spy"],
        start="2025-12-15T14:30:00Z",
        end="2025-12-15T14:31:00Z",
    )

    assert len(result.raw) == 2
    assert "source" not in result.raw
    assert "bid_exchange" in result.raw

    assert len(result.normalized) == 2
    assert result.normalized.loc[0, "symbol"] == "SPY"
    assert result.normalized.loc[0, "bid_exchange"] == "V"
    assert result.normalized.loc[0, "ask_exchange"] == "V"
    assert result.normalized.loc[0, "tape"] == "C"

    assert result.request_metadata["symbols"] == ["SPY"]
    assert result.request_metadata["data_kind"] == "quotes"


def test_fetch_trades_bundle_preserves_trade_ids() -> None:
    provider = make_provider()

    result = provider.fetch_trades_bundle(
        symbols=["SPY"],
        start="2025-12-15T14:30:00Z",
        end="2025-12-15T14:31:00Z",
    )

    assert len(result.raw) == 2
    assert "source" not in result.raw

    assert len(result.normalized) == 2
    assert result.normalized["id"].tolist() == [
        "trade-1",
        "trade-2",
    ]
    assert result.normalized["exchange"].tolist() == ["V", "V"]
    assert result.request_metadata["data_kind"] == "trades"


def test_same_timestamp_events_are_not_removed() -> None:
    provider = make_provider()

    quotes = provider.fetch_quotes(
        symbols=["SPY"],
        start="2025-12-15T14:30:00Z",
        end="2025-12-15T14:31:00Z",
    )

    trades = provider.fetch_trades(
        symbols=["SPY"],
        start="2025-12-15T14:30:00Z",
        end="2025-12-15T14:31:00Z",
    )

    assert len(quotes) == 2
    assert quotes["timestamp"].nunique() == 1

    assert len(trades) == 2
    assert trades["timestamp"].nunique() == 1
