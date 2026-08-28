"""Scalar contract tests for Day 12 event-driven replay records."""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
    is_dataclass,
    replace,
)
from types import SimpleNamespace
from typing import get_args

import numpy as np
import pandas as pd
import pytest

import systematic_alpha.analysis.trend_family_event_replay as replay
from systematic_alpha.analysis.trend_family_event_replay import (
    FillEvent,
    MarketBarEvent,
    PortfolioSnapshot,
    REPLAY_LEDGER_COLUMNS,
    ReplayEvent,
    SignalEvent,
    TargetPositionOrderEvent,
    TrendFamilyEventReplayResult,
    TrendFamilyEventReplayError,
    _ReplayState,
    _build_frozen_signal_observations,
    _prepare_replay_bars,
    _run_event_replay_core,
    run_trend_family_event_replay,
)
from systematic_alpha.analysis.strategy_performance import (
    PerformanceMetrics,
    build_wealth_index,
    calculate_performance_metrics,
)
from systematic_alpha.analysis.trend_family_robustness import (
    ANNUALIZATION_FACTORS,
    CONFIGURATION_IDS,
    EMA_MACD_PARAMETERS,
    TREND_RATIO_PARAMETERS,
)


UTC_TIMESTAMP = pd.Timestamp(
    "2025-01-02 14:30:00",
    tz="UTC",
)


def make_market_event(
    **overrides: object,
) -> MarketBarEvent:
    """Build one valid completed-bar event."""

    values: dict[str, object] = {
        "event_sequence": 0,
        "bar_index": 0,
        "timestamp": UTC_TIMESTAMP,
        "session_date": "2025-01-02",
        "symbol": "SPY",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1_000.0,
        "trade_count": 42,
        "vwap": 100.5,
        "source": "alpaca",
        "feed": "sip",
        "asset_return": 0.01,
    }
    values.update(overrides)

    return MarketBarEvent(**values)


def make_signal_event(
    **overrides: object,
) -> SignalEvent:
    """Build one valid frozen-strategy signal."""

    values: dict[str, object] = {
        "event_sequence": 1,
        "bar_index": 0,
        "timestamp": UTC_TIMESTAMP,
        "symbol": "SPY",
        "strategy": "trend_ratio",
        "configuration_id": (
            CONFIGURATION_IDS["trend_ratio"]
        ),
        "target_position": 1,
        "signal_available": True,
    }
    values.update(overrides)

    return SignalEvent(**values)


def make_order_event(
    **overrides: object,
) -> TargetPositionOrderEvent:
    """Build one valid next-observation target order."""

    values: dict[str, object] = {
        "event_sequence": 2,
        "submitted_bar_index": 0,
        "execute_bar_index": 1,
        "submitted_timestamp": UTC_TIMESTAMP,
        "symbol": "SPY",
        "strategy": "trend_ratio",
        "configuration_id": (
            CONFIGURATION_IDS["trend_ratio"]
        ),
        "current_executed_position": 0,
        "target_position": 1,
    }
    values.update(overrides)

    return TargetPositionOrderEvent(**values)


def make_fill_event(
    **overrides: object,
) -> FillEvent:
    """Build one valid target-position fill."""

    values: dict[str, object] = {
        "event_sequence": 3,
        "bar_index": 1,
        "timestamp": UTC_TIMESTAMP
        + pd.Timedelta(minutes=15),
        "symbol": "SPY",
        "strategy": "trend_ratio",
        "configuration_id": (
            CONFIGURATION_IDS["trend_ratio"]
        ),
        "submitted_bar_index": 0,
        "previous_position": 0,
        "executed_position": 1,
        "position_change": 1,
        "turnover": 1.0,
        "cost_bps_per_turnover": 1.0,
        "transaction_cost": 0.0001,
    }
    values.update(overrides)

    return FillEvent(**values)


def make_snapshot(
    **overrides: object,
) -> PortfolioSnapshot:
    """Build one valid normalized-notional portfolio record."""

    values: dict[str, object] = {
        "event_sequence": 4,
        "bar_index": 1,
        "timestamp": UTC_TIMESTAMP
        + pd.Timedelta(minutes=15),
        "symbol": "SPY",
        "strategy": "trend_ratio",
        "configuration_id": (
            CONFIGURATION_IDS["trend_ratio"]
        ),
        "position_eligible": True,
        "previous_position": 0,
        "executed_position": 1,
        "position_change": 1,
        "turnover": 1.0,
        "asset_return": 0.01,
        "gross_strategy_return": 0.01,
        "transaction_cost": 0.0001,
        "transaction_cost_amount": 0.0001,
        "net_strategy_return": 0.0099,
        "previous_equity": 1.0,
        "gross_ending_equity": 1.01,
        "cash_balance": -0.0001,
        "holdings_value": 1.01,
        "ending_equity": 1.0099,
    }
    values.update(overrides)

    return PortfolioSnapshot(**values)


def valid_events() -> tuple[
    MarketBarEvent,
    SignalEvent,
    TargetPositionOrderEvent,
    FillEvent,
    PortfolioSnapshot,
]:
    """Return one valid instance of every public record."""

    return (
        make_market_event(),
        make_signal_event(),
        make_order_event(),
        make_fill_event(),
        make_snapshot(),
    )


def test_event_records_are_frozen_slotted_dataclasses() -> None:
    for event in valid_events():
        assert is_dataclass(event)
        assert not hasattr(event, "__dict__")
        assert "__slots__" in type(event).__dict__

        with pytest.raises(FrozenInstanceError):
            event.event_sequence = 100


def test_replay_event_union_contains_all_public_records() -> None:
    assert set(get_args(ReplayEvent)) == {
        MarketBarEvent,
        SignalEvent,
        TargetPositionOrderEvent,
        FillEvent,
        PortfolioSnapshot,
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: make_market_event(
            event_sequence=-1
        ),
        lambda: make_signal_event(
            bar_index=-1
        ),
        lambda: make_order_event(
            submitted_bar_index=-1,
            execute_bar_index=0,
        ),
        lambda: make_fill_event(
            event_sequence=True
        ),
        lambda: make_snapshot(
            bar_index=1.5
        ),
    ),
)
def test_invalid_event_sequences_and_indexes_are_rejected(
    factory,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="index|sequence",
    ):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda timestamp: make_market_event(
            timestamp=timestamp
        ),
        lambda timestamp: make_signal_event(
            timestamp=timestamp
        ),
        lambda timestamp: make_order_event(
            submitted_timestamp=timestamp
        ),
        lambda timestamp: make_fill_event(
            timestamp=timestamp
        ),
        lambda timestamp: make_snapshot(
            timestamp=timestamp
        ),
    ),
)
def test_timestamps_require_timezone_awareness_and_normalize_to_utc(
    factory,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="timezone-aware",
    ):
        factory(
            pd.Timestamp(
                "2025-01-02 14:30:00"
            )
        )

    event = factory(
        pd.Timestamp(
            "2025-01-02 19:30:00",
            tz="Asia/Karachi",
        )
    )
    timestamp = getattr(
        event,
        "submitted_timestamp",
        getattr(event, "timestamp", None),
    )

    assert timestamp == UTC_TIMESTAMP
    assert str(timestamp.tz) == "UTC"


@pytest.mark.parametrize(
    "invalid_session",
    (
        "2025-1-2",
        "2025-02-30",
        "02-01-2025",
        "",
    ),
)
def test_market_session_date_must_be_valid_iso_date(
    invalid_session: str,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="session_date",
    ):
        make_market_event(
            session_date=invalid_session
        )


def test_symbols_normalize_to_uppercase_and_only_spy_is_accepted() -> None:
    signal = make_signal_event(
        symbol=" spy "
    )

    assert signal.symbol == "SPY"

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="SPY",
    ):
        make_signal_event(symbol="QQQ")


@pytest.mark.parametrize(
    ("strategy", "configuration_id"),
    tuple(CONFIGURATION_IDS.items()),
)
def test_frozen_strategy_and_configuration_pairs_are_accepted(
    strategy: str,
    configuration_id: str,
) -> None:
    event = make_signal_event(
        strategy=strategy,
        configuration_id=configuration_id,
    )

    assert event.strategy == strategy
    assert (
        event.configuration_id
        == configuration_id
    )


@pytest.mark.parametrize(
    ("strategy", "configuration_id"),
    (
        (
            "day09_winner",
            CONFIGURATION_IDS[
                "trend_ratio"
            ],
        ),
        (
            "trend_ratio",
            CONFIGURATION_IDS[
                "ema_macd"
            ],
        ),
        (
            "ema_macd",
            "best_parameter",
        ),
    ),
)
def test_unknown_or_mismatched_strategy_configuration_is_rejected(
    strategy: str,
    configuration_id: str,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="strategy|configuration",
    ):
        make_signal_event(
            strategy=strategy,
            configuration_id=configuration_id,
        )


@pytest.mark.parametrize(
    "position",
    (-2, 2, 0.5, True),
)
def test_positions_accept_only_minus_one_zero_and_one(
    position: object,
) -> None:
    factories = (
        lambda: make_signal_event(
            target_position=position
        ),
        lambda: make_order_event(
            target_position=position
        ),
        lambda: make_fill_event(
            executed_position=position
        ),
        lambda: make_snapshot(
            executed_position=position
        ),
    )

    for factory in factories:
        with pytest.raises(
            TrendFamilyEventReplayError,
            match="position",
        ):
            factory()


def test_unavailable_signal_requires_neutral_target() -> None:
    neutral = make_signal_event(
        signal_available=False,
        target_position=0,
    )

    assert neutral.target_position == 0

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="unavailable|neutral",
    ):
        make_signal_event(
            signal_available=False,
            target_position=1,
        )


def test_order_executes_exactly_one_observation_later() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="one observation",
    ):
        make_order_event(
            submitted_bar_index=4,
            execute_bar_index=6,
        )


def test_order_must_request_an_actual_position_change() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="position change",
    ):
        make_order_event(
            current_executed_position=1,
            target_position=1,
        )


def test_fill_position_change_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="position_change",
    ):
        make_fill_event(
            previous_position=-1,
            executed_position=1,
            position_change=1,
            turnover=2.0,
            transaction_cost=0.0002,
        )


def test_fill_turnover_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="turnover",
    ):
        make_fill_event(
            turnover=2.0,
            transaction_cost=0.0002,
        )


def test_fill_transaction_cost_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="transaction_cost",
    ):
        make_fill_event(
            cost_bps_per_turnover=2.0,
            transaction_cost=0.0001,
        )


def test_fill_must_execute_on_the_next_bar() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="submitted_bar_index|next",
    ):
        make_fill_event(
            bar_index=2,
            submitted_bar_index=0,
        )


def test_portfolio_position_and_turnover_identities_are_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="position_change",
    ):
        make_snapshot(position_change=0)

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="turnover",
    ):
        make_snapshot(
            turnover=2.0
        )


def test_portfolio_gross_return_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="gross_strategy_return",
    ):
        make_snapshot(
            gross_strategy_return=0.02
        )


def test_portfolio_net_return_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="net_strategy_return",
    ):
        make_snapshot(
            net_strategy_return=0.0098
        )


def test_portfolio_transaction_cost_amount_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="transaction_cost_amount",
    ):
        make_snapshot(
            previous_equity=2.0,
        )


def test_portfolio_gross_equity_identity_is_enforced() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="gross_ending_equity",
    ):
        make_snapshot(
            gross_ending_equity=1.02
        )


def test_portfolio_cash_holdings_and_ending_equity_identity() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="cash|holdings|ending_equity",
    ):
        make_snapshot(
            cash_balance=0.0,
        )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="ending_equity",
    ):
        make_snapshot(
            ending_equity=1.02,
        )


def test_negative_returns_and_declining_equity_are_valid() -> None:
    snapshot = make_snapshot(
        previous_position=1,
        executed_position=1,
        position_change=0,
        turnover=0.0,
        asset_return=-0.10,
        gross_strategy_return=-0.10,
        transaction_cost=0.0,
        transaction_cost_amount=0.0,
        net_strategy_return=-0.10,
        gross_ending_equity=0.90,
        cash_balance=0.0,
        holdings_value=0.90,
        ending_equity=0.90,
    )

    assert snapshot.net_strategy_return < 0.0
    assert (
        snapshot.ending_equity
        < snapshot.previous_equity
    )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: make_market_event(
            open=float("nan")
        ),
        lambda: make_market_event(
            asset_return=float("inf")
        ),
        lambda: make_fill_event(
            turnover=float("nan")
        ),
        lambda: make_fill_event(
            cost_bps_per_turnover=float("inf")
        ),
        lambda: make_snapshot(
            holdings_value=float("nan")
        ),
        lambda: make_snapshot(
            ending_equity=float("inf")
        ),
    ),
)
def test_nan_and_infinite_numeric_values_are_rejected(
    factory,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="finite",
    ):
        factory()


@pytest.mark.parametrize(
    "overrides",
    (
        {"open": 98.0},
        {"close": 103.0},
        {"low": 103.0},
        {"high": 98.0},
        {"volume": -1.0},
        {"trade_count": -1},
        {"trade_count": 1.5},
        {"vwap": 0.0},
        {"source": " "},
        {"feed": ""},
    ),
)
def test_market_bar_economic_validation(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
    ):
        make_market_event(**overrides)


def test_accounting_tolerance_accepts_benign_roundoff() -> None:
    snapshot = make_snapshot(
        ending_equity=(
            1.0099
            + np.finfo(float).eps
        )
    )

    assert snapshot.ending_equity == pytest.approx(
        1.0099
    )


def test_replace_cannot_bypass_validation() -> None:
    event = make_fill_event()

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="turnover",
    ):
        replace(event, turnover=2.0)


def make_replay_bars(
    *,
    session_dates: tuple[str, ...] = (
        "2024-12-31",
        "2025-01-02",
    ),
    bars_per_session: int = 26,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Build compact canonical SPY 15-minute sessions."""

    records: list[dict[str, object]] = []
    observation = 0

    for session_date in session_dates:
        timestamps = pd.date_range(
            f"{session_date} 14:30:00",
            periods=bars_per_session,
            freq="15min",
            tz="UTC",
        )

        for timestamp in timestamps:
            close = float(
                100.0
                + observation * 0.08
                + np.sin(
                    observation / 2.5
                )
            )
            open_price = close - 0.10

            records.append(
                {
                    "timestamp": timestamp,
                    "session_date": (
                        session_date
                    ),
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(
                        open_price,
                        close,
                    )
                    + 0.20,
                    "low": min(
                        open_price,
                        close,
                    )
                    - 0.20,
                    "close": close,
                    "volume": float(
                        1_000 + observation
                    ),
                    "trade_count": (
                        20 + observation
                    ),
                    "vwap": (
                        open_price + close
                    )
                    / 2.0,
                    "source": "alpaca",
                    "feed": "sip",
                }
            )
            observation += 1

    return pd.DataFrame.from_records(
        records
    )


def test_prepare_replay_bars_rebuilds_15min_return_features() -> None:
    bars = make_replay_bars()
    original = bars.copy(deep=True)

    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )

    pd.testing.assert_frame_equal(
        bars,
        original,
    )
    assert len(prepared) == len(bars)
    assert prepared["symbol"].eq(
        "SPY"
    ).all()
    assert prepared["bar_frequency"].eq(
        "15min"
    ).all()
    assert prepared[
        "source_frequency"
    ].eq("15min").all()
    assert prepared[
        "source_bar_count"
    ].eq(1).all()
    assert not prepared[
        "is_partial_bar"
    ].any()
    assert str(
        prepared["timestamp"].dt.tz
    ) == "UTC"
    assert prepared[
        "timestamp"
    ].is_monotonic_increasing
    assert pd.isna(
        prepared[
        "close_to_close_simple_return"
        ].iloc[0]
    )
    assert prepared[
        "close_to_close_simple_return"
    ].iloc[26:].notna().all()
    expected_returns = (
        prepared["close"].pct_change(
            fill_method=None
        )
    )
    pd.testing.assert_series_equal(
        prepared[
            "close_to_close_simple_return"
        ],
        expected_returns.rename(
            "close_to_close_simple_return"
        ),
    )
    assert prepared[
        "source"
    ].eq("alpaca").all()
    assert prepared[
        "feed"
    ].eq("sip").all()


def test_prepare_replay_bars_is_deterministic() -> None:
    bars = make_replay_bars()

    first = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    second = _prepare_replay_bars(
        bars,
        frequency="15min",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


@pytest.mark.parametrize(
    "frequency",
    (
        "30min",
        "60min",
        "15-minute",
        "",
    ),
)
def test_prepare_replay_bars_rejects_non_primary_frequency(
    frequency: str,
) -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="15min|frequency",
    ):
        _prepare_replay_bars(
            make_replay_bars(),
            frequency=frequency,
        )


def test_prepare_replay_bars_rejects_missing_or_non_spy_input() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="missing|required",
    ):
        _prepare_replay_bars(
            make_replay_bars().drop(
                columns="close"
            ),
            frequency="15min",
        )

    for invalid_symbol in (
        "QQQ",
        "IWM",
    ):
        with pytest.raises(
            TrendFamilyEventReplayError,
            match="SPY|symbol",
        ):
            _prepare_replay_bars(
                make_replay_bars(
                    symbol=invalid_symbol
                ),
                frequency="15min",
            )

    multiple = pd.concat(
        [
            make_replay_bars(),
            make_replay_bars(
                symbol="IWM"
            ),
        ],
        ignore_index=True,
    ).sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="SPY|symbol",
    ):
        _prepare_replay_bars(
            multiple,
            frequency="15min",
        )


def test_prepare_replay_bars_normalizes_spy_symbol() -> None:
    bars = make_replay_bars()
    bars["symbol"] = " spy "

    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )

    assert prepared["symbol"].eq(
        "SPY"
    ).all()


def test_prepare_replay_bars_requires_aware_timestamps() -> None:
    bars = make_replay_bars()
    bars["timestamp"] = (
        bars["timestamp"]
        .dt.tz_localize(None)
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="timezone-aware",
    ):
        _prepare_replay_bars(
            bars,
            frequency="15min",
        )


def test_prepare_replay_bars_converts_aware_timestamps_to_utc() -> None:
    bars = make_replay_bars()
    original_instants = bars[
        "timestamp"
    ].copy(deep=True)
    bars["timestamp"] = (
        bars["timestamp"]
        .dt.tz_convert("Asia/Karachi")
    )

    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )

    assert str(
        prepared["timestamp"].dt.tz
    ) == "UTC"
    pd.testing.assert_series_equal(
        prepared["timestamp"],
        original_instants,
    )


def test_prepare_replay_bars_rejects_out_of_development_dates() -> None:
    bars = make_replay_bars(
        session_dates=(
            "2019-12-30",
        ),
        bars_per_session=14,
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="development period",
    ):
        _prepare_replay_bars(
            bars,
            frequency="15min",
        )


def test_prepare_replay_bars_explicitly_rejects_2026_rows() -> None:
    bars = make_replay_bars(
        session_dates=(
            "2026-01-02",
        ),
        bars_per_session=14,
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="2026|development period",
    ):
        _prepare_replay_bars(
            bars,
            frequency="15min",
        )


def test_prepare_replay_bars_preserves_session_isolation() -> None:
    bars = pd.concat(
        [
            make_replay_bars(
                session_dates=(
                    "2024-11-29",
                ),
                bars_per_session=14,
            ),
            make_replay_bars(
                session_dates=(
                    "2025-01-02",
                ),
                bars_per_session=26,
            ),
        ],
        ignore_index=True,
    )

    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    observed_sizes = (
        prepared.groupby(
            "session_date",
            observed=True,
            sort=True,
        )
        .size()
        .to_dict()
    )

    assert observed_sizes == {
        "2024-11-29": 14,
        "2025-01-02": 26,
    }
    assert prepared[
        "bar_number"
    ].iloc[[0, 14]].tolist() == [
        1,
        1,
    ]
    assert prepared[
        "timestamp"
    ].is_monotonic_increasing


def test_prepare_replay_bars_rejects_duplicate_or_unsorted_input() -> None:
    duplicate = make_replay_bars()
    duplicate.loc[
        1,
        "timestamp",
    ] = duplicate.loc[
        0,
        "timestamp",
    ]

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="duplicate",
    ):
        _prepare_replay_bars(
            duplicate,
            frequency="15min",
        )

    unsorted = make_replay_bars()
    unsorted.iloc[
        [0, 1]
    ] = unsorted.iloc[
        [1, 0]
    ].to_numpy()

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="chronological|sorted",
    ):
        _prepare_replay_bars(
            unsorted,
            frequency="15min",
        )


@pytest.mark.parametrize(
    "invalid",
    (
        pd.DataFrame(),
        "not-a-frame",
    ),
)
def test_prepare_replay_bars_rejects_invalid_container(
    invalid: object,
) -> None:
    with pytest.raises(
        (TypeError, TrendFamilyEventReplayError),
        match="DataFrame|empty",
    ):
        _prepare_replay_bars(
            invalid,
            frequency="15min",
        )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_frozen_signal_observations_have_compact_stable_schema(
    strategy: str,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    original = prepared.copy(deep=True)

    signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )

    pd.testing.assert_frame_equal(
        prepared,
        original,
    )
    assert tuple(signals.columns) == (
        "timestamp",
        "symbol",
        "signal",
        "signal_available",
    )
    assert len(signals) == len(
        prepared
    )
    assert signals[
        "signal"
    ].isin((-1, 0, 1)).all()
    assert signals.loc[
        ~signals[
            "signal_available"
        ],
        "signal",
    ].eq(0).all()

    prohibited = {
        "position",
        "position_eligible",
        "turnover",
        "transaction_cost",
        "gross_strategy_return",
        "net_strategy_return",
    }
    assert not prohibited.intersection(
        signals.columns
    )


def test_signal_adapter_routes_exact_frozen_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    observed: list[
        tuple[str, object]
    ] = []
    original_trend = (
        replay.build_trend_ratio_strategy
    )
    original_ema = (
        replay.build_ema_macd_strategy
    )

    def record_trend(
        frame: pd.DataFrame,
        *,
        parameters,
    ):
        observed.append(
            (
                "trend_ratio",
                parameters,
            )
        )

        return original_trend(
            frame,
            parameters=parameters,
        )

    def record_ema(
        frame: pd.DataFrame,
        *,
        parameters,
    ):
        observed.append(
            (
                "ema_macd",
                parameters,
            )
        )

        return original_ema(
            frame,
            parameters=parameters,
        )

    monkeypatch.setattr(
        replay,
        "build_trend_ratio_strategy",
        record_trend,
    )
    monkeypatch.setattr(
        replay,
        "build_ema_macd_strategy",
        record_ema,
    )

    for strategy in (
        "trend_ratio",
        "ema_macd",
    ):
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )

    assert observed == [
        (
            "trend_ratio",
            TREND_RATIO_PARAMETERS,
        ),
        (
            "ema_macd",
            EMA_MACD_PARAMETERS,
        ),
    ]


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_signal_adapter_has_exact_vectorized_signal_parity(
    strategy: str,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )

    if strategy == "trend_ratio":
        bundle = (
            replay.build_trend_ratio_strategy(
                prepared,
                parameters=(
                    TREND_RATIO_PARAMETERS
                ),
            )
        )
    else:
        bundle = (
            replay.build_ema_macd_strategy(
                prepared,
                parameters=EMA_MACD_PARAMETERS,
            )
        )

    expected = bundle.observations[
        [
            "timestamp",
            "symbol",
            "signal",
            "signal_available",
        ]
    ].reset_index(drop=True)
    actual = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )
    assert actual["timestamp"].equals(
        prepared["timestamp"]
    )


def test_signal_adapter_ignores_vectorized_execution_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    original_builder = (
        replay.build_trend_ratio_strategy
    )

    def poisoned_builder(
        frame: pd.DataFrame,
        *,
        parameters,
    ):
        bundle = original_builder(
            frame,
            parameters=parameters,
        )
        observations = (
            bundle.observations.copy(
                deep=True
            )
        )
        observations["position"] = 99
        observations["turnover"] = -99.0
        observations[
            "transaction_cost"
        ] = -99.0
        observations[
            "gross_strategy_return"
        ] = -99.0
        observations[
            "net_strategy_return"
        ] = -99.0

        return replace(
            bundle,
            observations=observations,
        )

    monkeypatch.setattr(
        replay,
        "build_trend_ratio_strategy",
        poisoned_builder,
    )

    signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy="trend_ratio",
        )
    )

    assert signals[
        "signal"
    ].isin((-1, 0, 1)).all()


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_completed_bar_signals_are_causal(
    strategy: str,
) -> None:
    original_bars = make_replay_bars()
    modified_bars = original_bars.copy(
        deep=True
    )
    future = (
        modified_bars.index >= 45
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "vwap",
    ):
        modified_bars.loc[
            future,
            column,
        ] *= 3.0

    original = (
        _build_frozen_signal_observations(
            _prepare_replay_bars(
                original_bars,
                frequency="15min",
            ),
            strategy=strategy,
        )
    )
    modified = (
        _build_frozen_signal_observations(
            _prepare_replay_bars(
                modified_bars,
                frequency="15min",
            ),
            strategy=strategy,
        )
    )

    pd.testing.assert_frame_equal(
        original.loc[:44],
        modified.loc[:44],
    )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_completed_bar_causality_persists_across_session_boundary(
    strategy: str,
) -> None:
    original_bars = make_replay_bars()
    modified_bars = original_bars.copy(
        deep=True
    )
    cutoff = 30
    future = (
        modified_bars.index > cutoff
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "vwap",
    ):
        modified_bars.loc[
            future,
            column,
        ] *= 4.0

    original = (
        _build_frozen_signal_observations(
            _prepare_replay_bars(
                original_bars,
                frequency="15min",
            ),
            strategy=strategy,
        )
    )
    modified = (
        _build_frozen_signal_observations(
            _prepare_replay_bars(
                modified_bars,
                frequency="15min",
            ),
            strategy=strategy,
        )
    )

    pd.testing.assert_frame_equal(
        original.loc[:cutoff],
        modified.loc[:cutoff],
    )
    assert (
        original.loc[
            cutoff,
            "timestamp",
        ].date().isoformat()
        == "2025-01-02"
    )


def test_signal_adapter_rejects_unknown_strategy() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="strategy",
    ):
        _build_frozen_signal_observations(
            prepared,
            strategy="day09_winner",
        )


def test_signal_adapter_rejects_unprepared_input() -> None:
    with pytest.raises(
        TrendFamilyEventReplayError,
        match="prepared|required|columns",
    ):
        _build_frozen_signal_observations(
            make_replay_bars(),
            strategy="trend_ratio",
        )


def test_signal_adapter_rejects_invalid_builder_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )

    def invalid_builder(
        frame: pd.DataFrame,
        *,
        parameters,
    ) -> SimpleNamespace:
        del parameters

        observations = frame.copy(
            deep=True
        )
        observations["signal"] = 2
        observations[
            "signal_available"
        ] = True

        return SimpleNamespace(
            observations=observations
        )

    monkeypatch.setattr(
        replay,
        "build_trend_ratio_strategy",
        invalid_builder,
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="signal|position",
    ):
        _build_frozen_signal_observations(
            prepared,
            strategy="trend_ratio",
        )


def test_signal_adapter_is_deterministic() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )

    for strategy in (
        "trend_ratio",
        "ema_macd",
    ):
        first = (
            _build_frozen_signal_observations(
                prepared,
                strategy=strategy,
            )
        )
        second = (
            _build_frozen_signal_observations(
                prepared,
                strategy=strategy,
            )
        )

        pd.testing.assert_frame_equal(
            first,
            second,
        )


def make_manual_signal_observations(
    prepared: pd.DataFrame,
    *,
    targets: tuple[int, ...],
    availability: tuple[bool, ...] | None = None,
) -> pd.DataFrame:
    """Build aligned scalar decisions for replay-core tests."""

    if len(targets) > len(prepared):
        raise ValueError(
            "targets cannot exceed prepared observations."
        )

    target_values = np.zeros(
        len(prepared),
        dtype=np.int8,
    )
    target_values[:len(targets)] = (
        np.asarray(
            targets,
            dtype=np.int8,
        )
    )
    signals = pd.Series(
        target_values,
        name="signal",
    )
    availability_values = np.ones(
        len(prepared),
        dtype=bool,
    )

    if availability is not None:
        if len(availability) != len(targets):
            raise ValueError(
                "availability must align with targets."
            )

        availability_values[
            :len(availability)
        ] = np.asarray(
            availability,
            dtype=bool,
        )

    available = pd.Series(
        availability_values,
        name="signal_available",
    )

    return pd.DataFrame(
        {
            "timestamp": (
                prepared["timestamp"]
                .reset_index(drop=True)
            ),
            "symbol": (
                prepared["symbol"]
                .reset_index(drop=True)
            ),
            "signal": signals,
            "signal_available": available,
        }
    )


def run_manual_replay(
    targets: tuple[int, ...],
    *,
    availability: tuple[bool, ...] | None = None,
    prepared: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[ReplayEvent, ...],
    pd.DataFrame,
]:
    """Run the private core with aligned synthetic decisions."""

    if prepared is None:
        prepared = _prepare_replay_bars(
            make_replay_bars(),
            frequency="15min",
        )

    signals = make_manual_signal_observations(
        prepared,
        targets=targets,
        availability=availability,
    )
    events, ledger = _run_event_replay_core(
        prepared,
        signals,
        strategy="trend_ratio",
    )

    return prepared, signals, events, ledger


def event_bar_index(
    event: ReplayEvent,
) -> int:
    """Return the bar that dispatches one replay event."""

    if isinstance(
        event,
        TargetPositionOrderEvent,
    ):
        return event.submitted_bar_index

    return event.bar_index


def test_replay_state_is_frozen_slotted_and_neutral() -> None:
    state = _ReplayState(
        executed_position=0,
        pending_target_position=0,
        pending_signal_available=False,
        pending_order=None,
        equity=1.0,
    )

    assert is_dataclass(_ReplayState)
    assert not hasattr(state, "__dict__")
    assert state.executed_position == 0
    assert state.pending_target_position == 0
    assert state.pending_signal_available is False
    assert state.pending_order is None
    assert state.equity == 1.0

    with pytest.raises(FrozenInstanceError):
        state.equity = 2.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("executed_position", 2),
        ("pending_target_position", -2),
        ("equity", 0.0),
        ("equity", -1.0),
        ("equity", float("nan")),
        ("equity", float("inf")),
    ),
)
def test_replay_state_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "executed_position": 0,
        "pending_target_position": 0,
        "pending_signal_available": False,
        "pending_order": None,
        "equity": 1.0,
    }
    values[field] = value

    with pytest.raises(
        TrendFamilyEventReplayError,
    ):
        _ReplayState(**values)


def test_replay_state_validates_pending_order_identity() -> None:
    order = make_order_event()
    state = _ReplayState(
        executed_position=0,
        pending_target_position=1,
        pending_signal_available=True,
        pending_order=order,
        equity=1.0,
    )

    assert state.pending_order == order

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="pending|order|target",
    ):
        replace(
            state,
            pending_target_position=-1,
        )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="pending|order",
    ):
        replace(
            state,
            pending_order=None,
        )


@pytest.mark.parametrize(
    "invalid",
    (
        pd.DataFrame(),
        "not-a-frame",
    ),
)
def test_replay_core_rejects_invalid_containers(
    invalid: object,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = make_manual_signal_observations(
        prepared,
        targets=(0,),
    )

    with pytest.raises(
        (TypeError, TrendFamilyEventReplayError),
        match="DataFrame|empty",
    ):
        _run_event_replay_core(
            invalid,
            signals,
            strategy="trend_ratio",
        )

    with pytest.raises(
        (TypeError, TrendFamilyEventReplayError),
        match="DataFrame|empty",
    ):
        _run_event_replay_core(
            prepared,
            invalid,
            strategy="trend_ratio",
        )


def test_replay_core_rejects_missing_and_misaligned_inputs() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = make_manual_signal_observations(
        prepared,
        targets=(0,),
    )

    invalid_pairs = (
        (
            prepared.drop(columns="open"),
            signals,
        ),
        (
            prepared,
            signals.drop(columns="signal"),
        ),
        (
            prepared.iloc[:-1].copy(),
            signals,
        ),
        (
            prepared,
            signals.assign(
                timestamp=lambda frame: (
                    frame["timestamp"]
                    + pd.Timedelta(minutes=1)
                )
            ),
        ),
        (
            prepared,
            signals.assign(symbol="QQQ"),
        ),
    )

    for invalid_bars, invalid_signals in invalid_pairs:
        with pytest.raises(
            TrendFamilyEventReplayError,
            match=(
                "missing|required|count|align|"
                "timestamp|symbol"
            ),
        ):
            _run_event_replay_core(
                invalid_bars,
                invalid_signals,
                strategy="trend_ratio",
            )


def test_replay_core_rejects_invalid_ordering_and_signals() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = make_manual_signal_observations(
        prepared,
        targets=(0,),
    )
    unsorted = prepared.iloc[
        [1, 0, *range(2, len(prepared))]
    ].reset_index(drop=True)
    duplicate = prepared.copy(deep=True)
    duplicate.loc[
        1,
        "timestamp",
    ] = duplicate.loc[
        0,
        "timestamp",
    ]

    for invalid in (
        unsorted,
        duplicate,
    ):
        with pytest.raises(
            TrendFamilyEventReplayError,
            match="chronological|duplicate|ordered",
        ):
            _run_event_replay_core(
                invalid,
                signals,
                strategy="trend_ratio",
            )

    invalid_target = signals.copy(deep=True)
    invalid_target.loc[0, "signal"] = 2

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="signal|position",
    ):
        _run_event_replay_core(
            prepared,
            invalid_target,
            strategy="trend_ratio",
        )

    unavailable_active = signals.copy(deep=True)
    unavailable_active.loc[
        0,
        ["signal", "signal_available"],
    ] = [1, False]

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="Unavailable|neutral",
    ):
        _run_event_replay_core(
            prepared,
            unavailable_active,
            strategy="trend_ratio",
        )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="strategy",
    ):
        _run_event_replay_core(
            prepared,
            signals,
            strategy="day09_winner",
        )


def test_replay_core_is_non_mutating_and_deterministic() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = make_manual_signal_observations(
        prepared,
        targets=(1, 1, -1, 0),
    )
    original_prepared = prepared.copy(deep=True)
    original_signals = signals.copy(deep=True)

    first_events, first_ledger = (
        _run_event_replay_core(
            prepared,
            signals,
            strategy="trend_ratio",
        )
    )
    second_events, second_ledger = (
        _run_event_replay_core(
            prepared,
            signals,
            strategy="trend_ratio",
        )
    )

    assert first_events == second_events
    pd.testing.assert_frame_equal(
        first_ledger,
        second_ledger,
    )
    pd.testing.assert_frame_equal(
        prepared,
        original_prepared,
    )
    pd.testing.assert_frame_equal(
        signals,
        original_signals,
    )


def test_first_bar_is_neutral_and_not_position_eligible() -> None:
    _, _, events, ledger = run_manual_replay(
        (1, 1),
    )
    first = ledger.iloc[0]

    assert first[
        "previous_executed_position"
    ] == 0
    assert first["executed_position"] == 0
    assert first["position_change"] == 0
    assert first["turnover"] == 0.0
    assert first["asset_return"] == 0.0
    assert first["transaction_cost"] == 0.0
    assert not bool(
        first["position_eligible"]
    )
    assert not bool(first["fill_executed"])
    assert not any(
        isinstance(event, FillEvent)
        and event.bar_index == 0
        for event in events
    )

    market = next(
        event
        for event in events
        if isinstance(event, MarketBarEvent)
        and event.bar_index == 0
    )
    assert market.asset_return == 0.0


def test_signal_executes_exactly_one_observation_later() -> None:
    _, signals, _, ledger = run_manual_replay(
        (1, -1, 0, 1),
    )
    expected = (
        signals["signal"]
        .shift(1, fill_value=0)
        .astype("int8")
    )

    pd.testing.assert_series_equal(
        ledger["executed_position"],
        expected.rename("executed_position"),
    )
    assert ledger.loc[
        0,
        "executed_position",
    ] != signals.loc[0, "signal"]


def test_orders_fills_turnover_and_cost_follow_position_changes() -> None:
    _, _, events, ledger = run_manual_replay(
        (1, -1, 0, 0),
    )
    expected_positions = [0, 1, -1, 0]
    expected_changes = [0, 1, -2, 1]
    expected_turnover = [0.0, 1.0, 2.0, 1.0]

    assert ledger[
        "executed_position"
    ].iloc[:4].tolist() == expected_positions
    assert ledger[
        "position_change"
    ].iloc[:4].tolist() == expected_changes
    assert ledger[
        "turnover"
    ].iloc[:4].tolist() == expected_turnover
    np.testing.assert_allclose(
        ledger[
            "transaction_cost"
        ].iloc[:4],
        np.array(expected_turnover)
        * TREND_RATIO_PARAMETERS.cost_bps_per_turnover
        / 10_000.0,
    )

    orders = [
        event
        for event in events
        if isinstance(
            event,
            TargetPositionOrderEvent,
        )
    ]
    fills = [
        event
        for event in events
        if isinstance(event, FillEvent)
    ]

    assert [
        event.submitted_bar_index
        for event in orders[:3]
    ] == [0, 1, 2]
    assert [
        event.bar_index
        for event in fills[:3]
    ] == [1, 2, 3]
    assert [
        event.turnover
        for event in fills[:3]
    ] == [1.0, 2.0, 1.0]


def test_unchanged_targets_emit_no_redundant_orders_or_fills() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    _, _, events, ledger = run_manual_replay(
        tuple(
            1 for _ in range(len(prepared))
        ),
        prepared=prepared,
    )
    orders = [
        event
        for event in events
        if isinstance(
            event,
            TargetPositionOrderEvent,
        )
    ]
    fills = [
        event
        for event in events
        if isinstance(event, FillEvent)
    ]

    assert [
        event.submitted_bar_index
        for event in orders
    ] == [0]
    assert [
        event.bar_index
        for event in fills
    ] == [1]
    assert ledger.loc[
        2:,
        "position_change",
    ].eq(0).all()


def test_unavailable_neutral_signal_can_exit_active_position() -> None:
    _, _, events, ledger = run_manual_replay(
        (1, 0, 0),
        availability=(True, False, False),
    )

    assert ledger[
        "executed_position"
    ].iloc[:3].tolist() == [0, 1, 0]
    assert ledger[
        "position_eligible"
    ].iloc[:3].tolist() == [
        False,
        True,
        False,
    ]
    assert [
        event.bar_index
        for event in events
        if isinstance(event, FillEvent)
    ][:2] == [1, 2]


def test_unavailable_neutral_warmup_emits_no_orders() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = make_manual_signal_observations(
        prepared,
        targets=tuple(
            0 for _ in range(len(prepared))
        ),
        availability=tuple(
            False for _ in range(len(prepared))
        ),
    )
    events, ledger = _run_event_replay_core(
        prepared,
        signals,
        strategy="trend_ratio",
    )

    assert not any(
        isinstance(
            event,
            (
                TargetPositionOrderEvent,
                FillEvent,
            ),
        )
        for event in events
    )
    assert not ledger[
        "position_eligible"
    ].any()
    assert ledger[
        "executed_position"
    ].eq(0).all()


def test_normalized_notional_accounting_compounds_equity() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    prepared.loc[
        1,
        "close_to_close_simple_return",
    ] = -0.02
    prepared.loc[
        2,
        "close_to_close_simple_return",
    ] = 0.03
    _, _, _, ledger = run_manual_replay(
        (1, -1, 0),
        prepared=prepared,
    )

    np.testing.assert_allclose(
        ledger["gross_strategy_return"],
        ledger["executed_position"]
        * ledger["asset_return"],
    )
    np.testing.assert_allclose(
        ledger["net_strategy_return"],
        ledger["gross_strategy_return"]
        - ledger["transaction_cost"],
    )
    np.testing.assert_allclose(
        ledger["ending_equity"],
        ledger["previous_equity"]
        * (
            1.0
            + ledger["net_strategy_return"]
        ),
    )
    np.testing.assert_allclose(
        ledger["cash_balance"]
        + ledger["holdings_value"],
        ledger["ending_equity"],
    )
    np.testing.assert_allclose(
        ledger["previous_equity"].iloc[1:],
        ledger["ending_equity"].iloc[:-1],
    )
    assert ledger.loc[
        1,
        "ending_equity",
    ] < ledger.loc[
        1,
        "previous_equity",
    ]
    assert ledger.loc[
        1,
        "cash_balance",
    ] < 0.0
    assert ledger.loc[
        2,
        "holdings_value",
    ] < 0.0


def test_replay_rejects_nonpositive_ending_equity() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    prepared.loc[
        1,
        "close_to_close_simple_return",
    ] = -1.0
    signals = make_manual_signal_observations(
        prepared,
        targets=(1, 1),
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="equity|positive",
    ):
        _run_event_replay_core(
            prepared,
            signals,
            strategy="trend_ratio",
        )


def test_position_and_equity_persist_across_session_boundary() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    targets = tuple(
        1
        if index == 25
        else 0
        for index in range(len(prepared))
    )
    _, _, _, ledger = run_manual_replay(
        targets,
        prepared=prepared,
    )

    assert ledger.loc[
        25,
        "session_date",
    ] == "2024-12-31"
    assert ledger.loc[
        26,
        "session_date",
    ] == "2025-01-02"
    assert ledger.loc[
        26,
        "executed_position",
    ] == 1
    assert ledger.loc[
        26,
        "previous_equity",
    ] == pytest.approx(
        ledger.loc[
            25,
            "ending_equity",
        ]
    )


def test_event_sequence_and_per_bar_order_are_frozen() -> None:
    _, _, events, _ = run_manual_replay(
        (1, -1, 0, 0),
    )

    assert [
        event.event_sequence
        for event in events
    ] == list(range(len(events)))

    for bar_index in range(4):
        bar_events = [
            event
            for event in events
            if event_bar_index(event)
            == bar_index
        ]
        event_types = [
            type(event)
            for event in bar_events
        ]

        assert event_types[0] is MarketBarEvent
        if ledger_fill := any(
            event_type is FillEvent
            for event_type in event_types
        ):
            assert event_types[1] is FillEvent

        assert event_types[
            1 + int(ledger_fill)
        ] is PortfolioSnapshot

        if (
            event_types[-1]
            is not SignalEvent
        ):
            assert event_types[-2] is SignalEvent
            assert (
                event_types[-1]
                is TargetPositionOrderEvent
            )
        else:
            assert event_types[-1] is SignalEvent


def test_every_order_fills_on_next_bar() -> None:
    _, _, events, _ = run_manual_replay(
        (1, -1, 0, 0),
    )
    orders = {
        event.execute_bar_index: event
        for event in events
        if isinstance(
            event,
            TargetPositionOrderEvent,
        )
    }
    fills = {
        event.bar_index: event
        for event in events
        if isinstance(event, FillEvent)
    }

    assert set(orders) == set(fills)

    for bar_index, fill in fills.items():
        order = orders[bar_index]

        assert (
            fill.submitted_bar_index
            == order.submitted_bar_index
        )
        assert (
            fill.executed_position
            == order.target_position
        )


def test_terminal_signal_is_retained_without_unexecutable_order() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    targets = tuple(
        1
        if index == len(prepared) - 1
        else 0
        for index in range(len(prepared))
    )
    _, _, events, ledger = run_manual_replay(
        targets,
        prepared=prepared,
    )
    final_index = len(prepared) - 1

    assert any(
        isinstance(event, SignalEvent)
        and event.bar_index == final_index
        and event.target_position == 1
        for event in events
    )
    assert not any(
        isinstance(
            event,
            TargetPositionOrderEvent,
        )
        and event.submitted_bar_index
        == final_index
        for event in events
    )
    assert not bool(
        ledger.loc[
            final_index,
            "order_submitted",
        ]
    )


def test_ledger_schema_types_and_event_flags_are_exact() -> None:
    prepared, _, events, ledger = run_manual_replay(
        (1, -1, 0, 0),
    )

    assert tuple(ledger.columns) == (
        REPLAY_LEDGER_COLUMNS
    )
    assert len(ledger) == len(prepared)
    assert isinstance(
        ledger.index,
        pd.RangeIndex,
    )
    assert ledger[
        [
            "previous_executed_position",
            "executed_position",
            "position_change",
        ]
    ].dtypes.map(
        pd.api.types.is_integer_dtype
    ).all()
    assert ledger[
        [
            "signal_available",
            "position_eligible",
            "order_submitted",
            "fill_executed",
        ]
    ].dtypes.map(
        pd.api.types.is_bool_dtype
    ).all()
    assert str(ledger["timestamp"].dt.tz) == "UTC"

    order_bars = {
        event.submitted_bar_index
        for event in events
        if isinstance(
            event,
            TargetPositionOrderEvent,
        )
    }
    fill_bars = {
        event.bar_index
        for event in events
        if isinstance(event, FillEvent)
    }

    assert set(
        ledger.index[
            ledger["order_submitted"]
        ]
    ) == order_bars
    assert set(
        ledger.index[
            ledger["fill_executed"]
        ]
    ) == fill_bars


def test_portfolio_snapshots_match_ledger_rows() -> None:
    _, _, events, ledger = run_manual_replay(
        (1, -1, 0, 0),
    )
    snapshots = [
        event
        for event in events
        if isinstance(
            event,
            PortfolioSnapshot,
        )
    ]

    assert len(snapshots) == len(ledger)

    numeric_pairs = (
        (
            "previous_executed_position",
            "previous_position",
        ),
        (
            "executed_position",
            "executed_position",
        ),
        ("position_change", "position_change"),
        ("turnover", "turnover"),
        ("asset_return", "asset_return"),
        (
            "gross_strategy_return",
            "gross_strategy_return",
        ),
        (
            "transaction_cost",
            "transaction_cost",
        ),
        (
            "transaction_cost_amount",
            "transaction_cost_amount",
        ),
        (
            "net_strategy_return",
            "net_strategy_return",
        ),
        ("previous_equity", "previous_equity"),
        (
            "gross_ending_equity",
            "gross_ending_equity",
        ),
        ("cash_balance", "cash_balance"),
        ("holdings_value", "holdings_value"),
        ("ending_equity", "ending_equity"),
    )

    for snapshot, row in zip(
        snapshots,
        ledger.itertuples(index=False),
        strict=True,
    ):
        assert (
            snapshot.position_eligible
            == row.position_eligible
        )

        for ledger_name, event_name in numeric_pairs:
            assert getattr(
                snapshot,
                event_name,
            ) == pytest.approx(
                getattr(row, ledger_name)
            )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_replay_has_bar_for_bar_vectorized_parity(
    strategy: str,
) -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )
    _, ledger = _run_event_replay_core(
        prepared,
        signals,
        strategy=strategy,
    )

    if strategy == "trend_ratio":
        reference = (
            replay.build_trend_ratio_strategy(
                prepared,
                parameters=(
                    TREND_RATIO_PARAMETERS
                ),
            ).observations
        )
    else:
        reference = (
            replay.build_ema_macd_strategy(
                prepared,
                parameters=EMA_MACD_PARAMETERS,
            ).observations
        )

    pd.testing.assert_series_equal(
        ledger["executed_position"],
        reference["position"].rename(
            "executed_position"
        ),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        ledger["position_eligible"],
        reference[
            "position_eligible"
        ].rename("position_eligible"),
        check_dtype=False,
    )

    for ledger_column, reference_column in (
        ("turnover", "turnover"),
        (
            "transaction_cost",
            "transaction_cost",
        ),
        (
            "gross_strategy_return",
            "gross_strategy_return",
        ),
        (
            "net_strategy_return",
            "net_strategy_return",
        ),
    ):
        np.testing.assert_allclose(
            ledger[ledger_column],
            reference[reference_column],
            rtol=0.0,
            atol=replay.ACCOUNTING_TOLERANCE,
        )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_replay_remains_causal_when_future_bars_change(
    strategy: str,
) -> None:
    original_bars = make_replay_bars()
    modified_bars = original_bars.copy(
        deep=True
    )
    cutoff = 39

    for column in (
        "open",
        "high",
        "low",
        "close",
        "vwap",
    ):
        modified_bars.loc[
            modified_bars.index > cutoff,
            column,
        ] *= 2.5

    def run(
        bars: pd.DataFrame,
    ) -> tuple[
        tuple[ReplayEvent, ...],
        pd.DataFrame,
    ]:
        prepared = _prepare_replay_bars(
            bars,
            frequency="15min",
        )
        signals = (
            _build_frozen_signal_observations(
                prepared,
                strategy=strategy,
            )
        )

        return _run_event_replay_core(
            prepared,
            signals,
            strategy=strategy,
        )

    original_events, original_ledger = run(
        original_bars
    )
    modified_events, modified_ledger = run(
        modified_bars
    )

    pd.testing.assert_frame_equal(
        original_ledger.loc[:cutoff],
        modified_ledger.loc[:cutoff],
    )
    original_prefix = tuple(
        event
        for event in original_events
        if event_bar_index(event) <= cutoff
    )
    modified_prefix = tuple(
        event
        for event in modified_events
        if event_bar_index(event) <= cutoff
    )

    assert original_prefix == modified_prefix


def make_public_replay_bars() -> pd.DataFrame:
    """Build three complete sessions with evaluation warm-up."""

    return make_replay_bars(
        session_dates=(
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
        ),
    )


def public_boundaries(
    bars: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a warmed one-session final evaluation interval."""

    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )

    return (
        pd.Timestamp(
            prepared.loc[52, "timestamp"]
        ),
        pd.Timestamp(
            prepared["timestamp"].max()
        )
        + pd.Timedelta(minutes=15),
    )


def assert_performance_equal(
    actual: PerformanceMetrics,
    expected: PerformanceMetrics,
) -> None:
    """Compare shared metrics while preserving NaN conventions."""

    assert actual.observations == (
        expected.observations
    )

    for name in (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ):
        actual_value = getattr(actual, name)
        expected_value = getattr(
            expected,
            name,
        )

        if np.isnan(expected_value):
            assert np.isnan(actual_value)
        else:
            assert actual_value == pytest.approx(
                expected_value
            )


def test_public_result_is_frozen_slotted_and_copies_observations() -> None:
    bars = make_public_replay_bars()
    result = run_trend_family_event_replay(
        bars,
        strategy="trend_ratio",
    )

    assert is_dataclass(
        TrendFamilyEventReplayResult
    )
    assert not hasattr(result, "__dict__")
    assert isinstance(result.events, tuple)

    with pytest.raises(FrozenInstanceError):
        result.strategy = "ema_macd"

    copied = result.copy_observations()
    copied.loc[
        0,
        "ending_equity",
    ] = -999.0

    assert (
        result.observations.loc[
            0,
            "ending_equity",
        ]
        != -999.0
    )

    external = result.observations.copy(
        deep=True
    )
    reconstructed = replace(
        result,
        observations=external,
    )
    external.loc[
        0,
        "ending_equity",
    ] = -777.0

    assert (
        reconstructed.observations.loc[
            0,
            "ending_equity",
        ]
        != -777.0
    )


def test_public_result_rejects_invalid_contract_fields() -> None:
    result = run_trend_family_event_replay(
        make_public_replay_bars(),
        strategy="trend_ratio",
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="strategy|configuration",
    ):
        replace(
            result,
            strategy="ema_macd",
        )

    raw_without_snapshot = tuple(
        event
        for event in result.events
        if not (
            isinstance(
                event,
                PortfolioSnapshot,
            )
            and event.bar_index == 0
        )
    )
    without_snapshot = tuple(
        replace(
            event,
            event_sequence=index,
        )
        for index, event in enumerate(
            raw_without_snapshot
        )
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="count|snapshot|observation",
    ):
        replace(
            result,
            events=without_snapshot,
        )


def test_public_replay_rejects_frequency_and_naive_boundaries() -> None:
    bars = make_public_replay_bars()
    start, end = public_boundaries(bars)

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="15min|frequency",
    ):
        run_trend_family_event_replay(
            bars,
            strategy="trend_ratio",
            frequency="30min",
        )

    for naive_start, naive_end in (
        (
            start.tz_localize(None),
            end,
        ),
        (
            start,
            end.tz_localize(None),
        ),
    ):
        with pytest.raises(
            TrendFamilyEventReplayError,
            match="timezone-aware|boundary",
        ):
            run_trend_family_event_replay(
                bars,
                strategy="trend_ratio",
                evaluation_start=(
                    naive_start
                ),
                evaluation_end_exclusive=(
                    naive_end
                ),
            )


def test_non_utc_evaluation_boundaries_normalize_to_utc() -> None:
    bars = make_public_replay_bars()
    start, end = public_boundaries(bars)
    result = run_trend_family_event_replay(
        bars,
        strategy="trend_ratio",
        evaluation_start=(
            start.tz_convert(
                "America/New_York"
            )
        ),
        evaluation_end_exclusive=(
            end.tz_convert(
                "America/New_York"
            )
        ),
    )

    assert result.evaluation_start == start
    assert (
        result.evaluation_end_exclusive
        == end
    )
    assert str(
        result.evaluation_start.tz
    ) == "UTC"


def test_public_replay_rejects_invalid_or_partial_windows() -> None:
    bars = make_public_replay_bars()
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    start, end = public_boundaries(bars)
    invalid_windows = (
        (end, start),
        (start, start),
        (
            start
            + pd.Timedelta(minutes=15),
            end,
        ),
        (
            start,
            end
            - pd.Timedelta(minutes=15),
        ),
        (
            prepared["timestamp"].min()
            - pd.Timedelta(minutes=15),
            end,
        ),
        (
            start,
            prepared["timestamp"].max()
            + pd.Timedelta(minutes=30),
        ),
    )

    for invalid_start, invalid_end in invalid_windows:
        with pytest.raises(
            TrendFamilyEventReplayError,
            match=(
                "start|end|empty|session|boundary|"
                "outside|window"
            ),
        ):
            run_trend_family_event_replay(
                bars,
                strategy="trend_ratio",
                evaluation_start=(
                    invalid_start
                ),
                evaluation_end_exclusive=(
                    invalid_end
                ),
            )


def test_public_replay_explicitly_rejects_2026_input() -> None:
    bars = make_replay_bars(
        session_dates=(
            "2026-01-02",
        ),
        bars_per_session=14,
    )

    with pytest.raises(
        TrendFamilyEventReplayError,
        match="2026|development",
    ):
        run_trend_family_event_replay(
            bars,
            strategy="trend_ratio",
        )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_omitted_boundaries_equal_explicit_full_frame(
    strategy: str,
) -> None:
    bars = make_public_replay_bars()
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    start = pd.Timestamp(
        prepared["timestamp"].min()
    )
    end = pd.Timestamp(
        prepared["timestamp"].max()
    ) + pd.Timedelta(minutes=15)

    omitted = run_trend_family_event_replay(
        bars,
        strategy=strategy,
    )
    explicit = run_trend_family_event_replay(
        bars,
        strategy=strategy,
        evaluation_start=start,
        evaluation_end_exclusive=end,
    )

    assert omitted.events == explicit.events
    pd.testing.assert_frame_equal(
        omitted.observations,
        explicit.observations,
    )
    assert_performance_equal(
        omitted.gross_performance,
        explicit.gross_performance,
    )
    assert_performance_equal(
        omitted.net_performance,
        explicit.net_performance,
    )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_pre_evaluation_history_warms_signals_but_not_execution(
    strategy: str,
) -> None:
    bars = make_public_replay_bars()
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    full_signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )
    start, end = public_boundaries(bars)
    result = run_trend_family_event_replay(
        bars,
        strategy=strategy,
        evaluation_start=start,
        evaluation_end_exclusive=end,
    )
    first = result.observations.iloc[0]

    assert (
        first["signal_available"]
        == full_signals.loc[
            52,
            "signal_available",
        ]
    )
    assert bool(
        full_signals.loc[
            52,
            "signal_available",
        ]
    )
    assert full_signals.loc[
        51,
        "signal",
    ] != 0
    assert result.observations[
        "timestamp"
    ].ge(start).all()
    assert result.observations[
        "timestamp"
    ].lt(end).all()
    assert all(
        (
            event.timestamp
            if not isinstance(
                event,
                TargetPositionOrderEvent,
            )
            else event.submitted_timestamp
        )
        >= start
        for event in result.events
    )
    assert all(
        (
            event.timestamp
            if not isinstance(
                event,
                TargetPositionOrderEvent,
            )
            else event.submitted_timestamp
        )
        < end
        for event in result.events
    )
    assert first[
        "executed_position"
    ] == 0
    assert not bool(
        first["position_eligible"]
    )
    assert first["turnover"] == 0.0
    assert first["transaction_cost"] == 0.0
    assert first["previous_equity"] == 1.0
    assert result.observations.loc[
        1,
        "executed_position",
    ] == result.observations.loc[
        0,
        "target_position",
    ]


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_public_replay_matches_reset_direct_core(
    strategy: str,
) -> None:
    bars = make_public_replay_bars()
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    signals = (
        _build_frozen_signal_observations(
            prepared,
            strategy=strategy,
        )
    )
    start, end = public_boundaries(bars)
    mask = (
        prepared["timestamp"].ge(start)
        & prepared["timestamp"].lt(end)
    )
    evaluation_bars = (
        prepared.loc[mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    evaluation_signals = (
        signals.loc[mask]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    expected_events, expected_ledger = (
        _run_event_replay_core(
            evaluation_bars,
            evaluation_signals,
            strategy=strategy,
        )
    )
    result = run_trend_family_event_replay(
        bars,
        strategy=strategy,
        evaluation_start=start,
        evaluation_end_exclusive=end,
    )

    assert result.events == expected_events
    pd.testing.assert_frame_equal(
        result.observations,
        expected_ledger,
    )
    assert_performance_equal(
        result.gross_performance,
        calculate_performance_metrics(
            expected_ledger[
                "gross_strategy_return"
            ],
            annualization_factor=(
                ANNUALIZATION_FACTORS[
                    "15min"
                ]
            ),
        ),
    )
    assert_performance_equal(
        result.net_performance,
        calculate_performance_metrics(
            expected_ledger[
                "net_strategy_return"
            ],
            annualization_factor=(
                ANNUALIZATION_FACTORS[
                    "15min"
                ]
            ),
        ),
    )


def test_public_replay_preserves_state_across_evaluation_sessions() -> None:
    bars = make_public_replay_bars()
    prepared = _prepare_replay_bars(
        bars,
        frequency="15min",
    )
    start = pd.Timestamp(
        prepared.loc[26, "timestamp"]
    )
    end = pd.Timestamp(
        prepared["timestamp"].max()
    ) + pd.Timedelta(minutes=15)
    result = run_trend_family_event_replay(
        bars,
        strategy="trend_ratio",
        evaluation_start=start,
        evaluation_end_exclusive=end,
    )
    second_session_first = 26

    assert (
        result.observations.loc[
            second_session_first,
            "session_date",
        ]
        == "2025-01-02"
    )
    assert (
        result.observations.loc[
            second_session_first,
            "executed_position",
        ]
        == result.observations.loc[
            second_session_first - 1,
            "target_position",
        ]
    )
    assert result.observations.loc[
        second_session_first,
        "previous_equity",
    ] == pytest.approx(
        result.observations.loc[
            second_session_first - 1,
            "ending_equity",
        ]
    )


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_public_performance_and_wealth_use_replay_returns(
    strategy: str,
) -> None:
    result = run_trend_family_event_replay(
        make_public_replay_bars(),
        strategy=strategy,
    )
    gross = calculate_performance_metrics(
        result.observations[
            "gross_strategy_return"
        ],
        annualization_factor=(
            ANNUALIZATION_FACTORS["15min"]
        ),
    )
    net = calculate_performance_metrics(
        result.observations[
            "net_strategy_return"
        ],
        annualization_factor=(
            ANNUALIZATION_FACTORS["15min"]
        ),
    )
    net_wealth = build_wealth_index(
        result.observations[
            "net_strategy_return"
        ]
    )

    assert_performance_equal(
        result.gross_performance,
        gross,
    )
    assert_performance_equal(
        result.net_performance,
        net,
    )
    np.testing.assert_allclose(
        result.observations[
            "ending_equity"
        ],
        net_wealth,
        rtol=0.0,
        atol=replay.ACCOUNTING_TOLERANCE,
    )


def test_public_replay_preserves_undefined_sharpe_convention() -> None:
    bars = make_public_replay_bars()
    bars["open"] = 100.0
    bars["high"] = 100.5
    bars["low"] = 99.5
    bars["close"] = 100.0
    bars["vwap"] = 100.0
    result = run_trend_family_event_replay(
        bars,
        strategy="trend_ratio",
    )

    assert np.isnan(
        result.gross_performance.sharpe_ratio
    )
    assert np.isnan(
        result.net_performance.sharpe_ratio
    )
    assert (
        result.net_performance
        .annualized_volatility
        == 0.0
    )


def test_public_result_accepts_negative_performance_metrics() -> None:
    prepared = _prepare_replay_bars(
        make_replay_bars(),
        frequency="15min",
    )
    prepared.loc[
        1,
        "close_to_close_simple_return",
    ] = -0.10
    signals = make_manual_signal_observations(
        prepared,
        targets=(1, 0),
    )
    events, observations = (
        _run_event_replay_core(
            prepared,
            signals,
            strategy="trend_ratio",
        )
    )
    gross = calculate_performance_metrics(
        observations[
            "gross_strategy_return"
        ],
        annualization_factor=(
            ANNUALIZATION_FACTORS["15min"]
        ),
    )
    net = calculate_performance_metrics(
        observations[
            "net_strategy_return"
        ],
        annualization_factor=(
            ANNUALIZATION_FACTORS["15min"]
        ),
    )
    accepted = TrendFamilyEventReplayResult(
        strategy="trend_ratio",
        symbol="SPY",
        frequency="15min",
        configuration_id=(
            CONFIGURATION_IDS[
                "trend_ratio"
            ]
        ),
        evaluation_start=pd.Timestamp(
            prepared["timestamp"].min()
        ),
        evaluation_end_exclusive=(
            pd.Timestamp(
                prepared["timestamp"].max()
            )
            + pd.Timedelta(minutes=15)
        ),
        events=events,
        observations=observations,
        gross_performance=gross,
        net_performance=net,
    )

    assert (
        accepted.net_performance
        .cumulative_return
        < 0.0
    )


def test_public_replay_schema_boundaries_and_bar_indexes() -> None:
    bars = make_public_replay_bars()
    start, end = public_boundaries(bars)
    result = run_trend_family_event_replay(
        bars,
        strategy="ema_macd",
        evaluation_start=start,
        evaluation_end_exclusive=end,
    )

    assert tuple(
        result.observations.columns
    ) == REPLAY_LEDGER_COLUMNS
    assert isinstance(
        result.observations.index,
        pd.RangeIndex,
    )
    assert result.observations[
        "bar_index"
    ].tolist() == list(
        range(len(result.observations))
    )
    assert result.observations[
        "timestamp"
    ].ge(start).all()
    assert result.observations[
        "timestamp"
    ].lt(end).all()


@pytest.mark.parametrize(
    "strategy",
    (
        "trend_ratio",
        "ema_macd",
    ),
)
def test_public_replay_is_deterministic_non_mutating_and_unaliased(
    strategy: str,
) -> None:
    bars = make_public_replay_bars()
    original = bars.copy(deep=True)
    first = run_trend_family_event_replay(
        bars,
        strategy=strategy,
    )
    second = run_trend_family_event_replay(
        bars,
        strategy=strategy,
    )

    assert first.events == second.events
    pd.testing.assert_frame_equal(
        first.observations,
        second.observations,
    )
    assert_performance_equal(
        first.gross_performance,
        second.gross_performance,
    )
    assert_performance_equal(
        first.net_performance,
        second.net_performance,
    )
    pd.testing.assert_frame_equal(
        bars,
        original,
    )

    first.observations.loc[
        0,
        "ending_equity",
    ] = -123.0

    assert (
        second.observations.loc[
            0,
            "ending_equity",
        ]
        != -123.0
    )
    assert (
        bars.loc[0, "close"]
        == original.loc[0, "close"]
    )
