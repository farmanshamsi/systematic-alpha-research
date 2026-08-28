"""Synthetic, network-free fixtures for the frozen Day 18 contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from systematic_alpha.broker.paper_boundary import (
    ALPACA_PAPER_BASE_URL,
    CORE_SYMBOLS,
    AlpacaPaperBroker,
    PreflightResult,
    run_paper_preflight,
)


def safe_day18_config() -> dict[str, Any]:
    return {
        "project": {"environment": "paper"},
        "safety": {
            "allow_live_trading": False,
            "require_paper_mode": True,
            "require_manual_order_confirmation": True,
            "kill_switch_enabled": True,
        },
        "broker": {
            "provider": "alpaca",
            "paper": True,
            "paper_base_url": ALPACA_PAPER_BASE_URL,
            "credentials_file": ".env",
            "api_key_env": "ALPACA_API_KEY",
            "secret_key_env": "ALPACA_SECRET_KEY",
        },
        "execution": {"order_submission_enabled": False},
    }


def account_response(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "status": "ACTIVE",
        "trading_blocked": False,
        "account_blocked": False,
        "trade_suspended_by_user": False,
        "shorting_enabled": True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def clock_response(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "timestamp": datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
        "is_open": False,
        "next_open": datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc),
        "next_close": datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def asset_response(symbol: str, **overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "symbol": symbol,
        "asset_class": "us_equity",
        "status": "active",
        "tradable": True,
        "shortable": True,
        "easy_to_borrow": True,
        "fractionable": True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class FakeTradingClient:
    """Read-only fake that records the exact broker call order."""

    def __init__(
        self,
        *,
        account: object | None = None,
        clock: object | None = None,
        assets: dict[str, object] | None = None,
        endpoint: str = ALPACA_PAPER_BASE_URL,
    ) -> None:
        self._base_url = endpoint
        self.account = account if account is not None else account_response()
        self.clock = clock if clock is not None else clock_response()
        self.assets = assets or {
            symbol: asset_response(symbol) for symbol in CORE_SYMBOLS
        }
        self.calls: list[str] = []

    def get_account(self) -> object:
        self.calls.append("account")
        return self.account

    def get_clock(self) -> object:
        self.calls.append("clock")
        return self.clock

    def get_asset(self, symbol: str) -> object:
        self.calls.append(f"asset:{symbol}")
        return self.assets[symbol]


def passing_preflight_result() -> PreflightResult:
    client = FakeTradingClient()
    broker = AlpacaPaperBroker(
        config=deepcopy(safe_day18_config()),
        client=client,
    )
    return run_paper_preflight(broker)
