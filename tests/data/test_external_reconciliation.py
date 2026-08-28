import pandas as pd

from systematic_alpha.data.external_reconciliation import (
    aggregate_intraday_to_daily,
    reconcile_daily_ohlcv,
)


def make_daily(
    *,
    dates: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    source: str,
    feed: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                dates,
                utc=True,
            ),
            "session_date": dates,
            "symbol": ["SPY"] * len(dates),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "source": [source] * len(dates),
            "feed": [feed] * len(dates),
        }
    )


def test_intraday_aggregation_uses_session_ohlcv() -> None:
    intraday = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-12-15T14:30:00Z",
                    "2025-12-15T14:31:00Z",
                ],
                utc=True,
            ),
            "symbol": ["SPY", "SPY"],
            "open": [100.0, 101.0],
            "high": [102.0, 104.0],
            "low": [99.0, 100.0],
            "close": [101.0, 103.0],
            "volume": [1000, 2000],
            "source": ["alpaca", "alpaca"],
            "feed": ["iex", "iex"],
        }
    )

    result = aggregate_intraday_to_daily(intraday)

    assert len(result) == 1
    assert result.loc[0, "session_date"] == "2025-12-15"
    assert result.loc[0, "open"] == 100.0
    assert result.loc[0, "high"] == 104.0
    assert result.loc[0, "low"] == 99.0
    assert result.loc[0, "close"] == 103.0
    assert result.loc[0, "volume"] == 3000


def test_matching_daily_prices_pass() -> None:
    primary = make_daily(
        dates=["2025-12-15"],
        opens=[100.00],
        highs=[102.00],
        lows=[99.00],
        closes=[101.00],
        volumes=[1_000_000],
        source="alpaca",
        feed="iex",
    )

    external = make_daily(
        dates=["2025-12-15"],
        opens=[100.01],
        highs=[102.01],
        lows=[99.01],
        closes=[101.01],
        volumes=[1_050_000],
        source="yfinance",
        feed="yahoo",
    )

    report = reconcile_daily_ohlcv(
        primary,
        external,
        price_tolerance_bps=50,
        volume_tolerance_pct=10,
    )

    assert bool(report.loc[0, "price_pass"])
    assert bool(report.loc[0, "volume_pass"])
    assert bool(report.loc[0, "overall_pass"])
    assert report.loc[0, "status"] == "pass"


def test_volume_mismatch_is_reported_separately() -> None:
    primary = make_daily(
        dates=["2025-12-15"],
        opens=[100],
        highs=[102],
        lows=[99],
        closes=[101],
        volumes=[100_000],
        source="alpaca",
        feed="iex",
    )

    external = make_daily(
        dates=["2025-12-15"],
        opens=[100],
        highs=[102],
        lows=[99],
        closes=[101],
        volumes=[1_000_000],
        source="yfinance",
        feed="yahoo",
    )

    report = reconcile_daily_ohlcv(
        primary,
        external,
        price_tolerance_bps=10,
        volume_tolerance_pct=25,
    )

    assert bool(report.loc[0, "price_pass"])
    assert not bool(report.loc[0, "volume_pass"])
    assert not bool(report.loc[0, "overall_pass"])
    assert report.loc[0, "status"] == "volume_mismatch"


def test_missing_external_date_is_preserved() -> None:
    primary = make_daily(
        dates=["2025-12-15"],
        opens=[100],
        highs=[102],
        lows=[99],
        closes=[101],
        volumes=[100_000],
        source="alpaca",
        feed="iex",
    )

    external = make_daily(
        dates=["2025-12-16"],
        opens=[101],
        highs=[103],
        lows=[100],
        closes=[102],
        volumes=[1_000_000],
        source="yfinance",
        feed="yahoo",
    )

    report = reconcile_daily_ohlcv(
        primary,
        external,
        price_tolerance_bps=10,
        volume_tolerance_pct=25,
    )

    assert len(report) == 2

    statuses = set(report["status"])

    assert statuses == {
        "missing_external",
        "missing_primary",
    }
