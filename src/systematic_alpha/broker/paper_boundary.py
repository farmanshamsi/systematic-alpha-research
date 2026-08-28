"""Fail-closed, read-only Alpaca paper-broker boundary for Day 18."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Final, Mapping, Protocol, TypeVar

from alpaca.trading.client import TradingClient

from systematic_alpha.data.config_loader import (
    AlpacaCredentials,
    ProjectConfigError,
    load_alpaca_credentials,
    load_project_config,
)


ALPACA_PAPER_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"
CORE_SYMBOLS: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM")
PROVIDER_ORDER_TYPES: Final[tuple[str, ...]] = (
    "market",
    "limit",
    "stop",
    "stop_limit",
    "trailing_stop",
)
PROVIDER_TIME_IN_FORCE: Final[tuple[str, ...]] = (
    "day",
    "gtc",
    "opg",
    "cls",
    "ioc",
    "fok",
)
PREFLIGHT_CALL_ORDER: Final[tuple[str, ...]] = (
    "account",
    "clock",
    "asset:SPY",
    "asset:QQQ",
    "asset:IWM",
)


class PaperBrokerError(RuntimeError):
    """Base class for safe-to-persist Day 18 broker failures."""


class PaperBrokerConfigurationError(PaperBrokerError):
    """Raised when the paper-only safety configuration is invalid."""


class PaperBrokerCredentialError(PaperBrokerError):
    """Raised when local Alpaca credentials are incomplete or unavailable."""


class PaperBrokerConnectionError(PaperBrokerError):
    """Raised when a read-only Alpaca request cannot be completed."""


class PaperBrokerResponseError(PaperBrokerError):
    """Raised when a broker response is missing or malformed."""


class PaperBrokerPreflightError(PaperBrokerError):
    """Raised when normalized broker state fails a frozen preflight gate."""


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Minimal non-financial account state needed by the Day 18 gate."""

    status: str
    trading_blocked: bool
    account_blocked: bool
    trade_suspended_by_user: bool
    shorting_enabled: bool


@dataclass(frozen=True, slots=True)
class MarketClockSnapshot:
    """Normalized market-clock state."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    """Minimal US-equity eligibility state for one requested symbol."""

    symbol: str
    asset_class: str
    status: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool

    @property
    def long_eligible(self) -> bool:
        return (
            self.asset_class == "us_equity"
            and self.status == "active"
            and self.tradable
        )

    @property
    def short_eligible(self) -> bool:
        return self.long_eligible and self.shortable and self.easy_to_borrow

    @property
    def fractional_eligible(self) -> bool:
        return self.long_eligible and self.fractionable

    @property
    def gate_passed(self) -> bool:
        return (
            self.long_eligible
            and self.short_eligible
            and self.fractional_eligible
        )


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """One documented provider capability with Day 18 authorization state."""

    capability_kind: str
    capability_value: str
    provider_supported: bool
    day18_authorized: bool
    constraint: str
    evidence_source: str


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Complete normalized result of one read-only paper preflight."""

    schema_version: str
    paper_endpoint: str
    sdk_name: str
    sdk_version: str
    credentials_loaded: bool
    credential_values_persisted: bool
    order_submission_enabled: bool
    order_submission_occurred: bool
    call_order: tuple[str, ...]
    core_symbols: tuple[str, ...]
    account: AccountSnapshot
    clock: MarketClockSnapshot
    assets: tuple[AssetSnapshot, ...]
    capabilities: tuple[CapabilitySnapshot, ...]
    account_gate_passed: bool
    clock_gate_passed: bool
    asset_gate_passed: bool
    preflight_passed: bool


class ReadOnlyPaperBroker(Protocol):
    """Broker-neutral surface deliberately excluding order mutation."""

    @property
    def paper_endpoint(self) -> str: ...

    @property
    def credentials_loaded(self) -> bool: ...

    def get_account_snapshot(self) -> AccountSnapshot: ...

    def get_market_clock_snapshot(self) -> MarketClockSnapshot: ...

    def get_asset_snapshot(self, symbol: str) -> AssetSnapshot: ...


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperBrokerConfigurationError(f"{name} must be a mapping.")
    return value


def _exact_bool(value: object, *, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise PaperBrokerConfigurationError(
            f"{name} must be exactly {str(expected).lower()}."
        )


def validate_day18_paper_config(config: Mapping[str, Any]) -> str:
    """Validate every frozen Day 18 paper-only configuration invariant."""

    root = _mapping(config, name="config")
    project = _mapping(root.get("project"), name="project")
    safety = _mapping(root.get("safety"), name="safety")
    broker = _mapping(root.get("broker"), name="broker")
    execution = _mapping(root.get("execution"), name="execution")

    if project.get("environment") != "paper":
        raise PaperBrokerConfigurationError(
            "Project environment must be exactly paper."
        )
    if broker.get("provider") != "alpaca":
        raise PaperBrokerConfigurationError(
            "Broker provider must be exactly alpaca."
        )
    _exact_bool(broker.get("paper"), name="broker.paper", expected=True)
    _exact_bool(
        safety.get("allow_live_trading"),
        name="safety.allow_live_trading",
        expected=False,
    )
    _exact_bool(
        safety.get("require_paper_mode"),
        name="safety.require_paper_mode",
        expected=True,
    )
    _exact_bool(
        safety.get("require_manual_order_confirmation"),
        name="safety.require_manual_order_confirmation",
        expected=True,
    )
    _exact_bool(
        safety.get("kill_switch_enabled"),
        name="safety.kill_switch_enabled",
        expected=True,
    )
    _exact_bool(
        execution.get("order_submission_enabled"),
        name="execution.order_submission_enabled",
        expected=False,
    )

    endpoint_value = broker.get("paper_base_url")
    if not isinstance(endpoint_value, str) or not endpoint_value.strip():
        raise PaperBrokerConfigurationError(
            "Broker paper endpoint must be a non-empty string."
        )
    endpoint = endpoint_value.strip().rstrip("/")
    if endpoint != ALPACA_PAPER_BASE_URL:
        raise PaperBrokerConfigurationError(
            "Broker endpoint is not the frozen Alpaca paper endpoint."
        )
    if broker.get("credentials_file") != ".env":
        raise PaperBrokerConfigurationError(
            "Day 18 credentials file must be the gitignored local .env file."
        )
    for key in ("api_key_env", "secret_key_env"):
        value = broker.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PaperBrokerConfigurationError(
                f"broker.{key} must name a non-empty environment variable."
            )
    if broker["api_key_env"] == broker["secret_key_env"]:
        raise PaperBrokerConfigurationError(
            "API-key and secret environment-variable names must differ."
        )
    return endpoint


def _safe_text(value: object, *, field: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise PaperBrokerResponseError(
            f"Broker response field {field} must be non-empty text."
        )
    return value.strip()


def _safe_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise PaperBrokerResponseError(
            f"Broker response field {field} must be boolean."
        )
    return value


def _safe_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise PaperBrokerResponseError(
            f"Broker response field {field} must be a datetime."
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise PaperBrokerResponseError(
            f"Broker response field {field} must be timezone-aware."
        )
    return value.astimezone(timezone.utc)


def _field(response: object, field: str) -> object:
    if isinstance(response, Mapping):
        if field not in response:
            raise PaperBrokerResponseError(
                f"Broker response is missing required field {field}."
            )
        return response[field]
    if not hasattr(response, field):
        raise PaperBrokerResponseError(
            f"Broker response is missing required field {field}."
        )
    return getattr(response, field)


T = TypeVar("T")


def _request(operation: str, function: Callable[[], T]) -> T:
    try:
        response = function()
    except PaperBrokerError:
        raise
    except Exception as exc:
        raise PaperBrokerConnectionError(
            f"Read-only Alpaca {operation} request failed."
        ) from exc
    if response is None:
        raise PaperBrokerResponseError(
            f"Read-only Alpaca {operation} response was empty."
        )
    return response


def _endpoint_text(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise PaperBrokerConfigurationError(
            "Trading client endpoint cannot be verified."
        )
    return value.strip().rstrip("/")


def _client_endpoint(client: object) -> str:
    if hasattr(client, "_base_url"):
        return _endpoint_text(getattr(client, "_base_url"))
    if hasattr(client, "paper_base_url"):
        return _endpoint_text(getattr(client, "paper_base_url"))
    raise PaperBrokerConfigurationError(
        "Trading client does not expose a verifiable paper endpoint."
    )


class AlpacaPaperBroker:
    """Read-only Alpaca adapter that cannot submit or mutate orders."""

    __slots__ = ("_client", "_credentials_loaded", "_paper_endpoint")

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        client: object | None = None,
        credentials: AlpacaCredentials | None = None,
    ) -> None:
        project_config = config or load_project_config()
        endpoint = validate_day18_paper_config(project_config)

        if client is not None and credentials is not None:
            raise PaperBrokerConfigurationError(
                "Injected client and credentials cannot be supplied together."
            )
        if client is None:
            try:
                loaded = credentials or load_alpaca_credentials(
                    dict(project_config)
                )
            except ProjectConfigError as exc:
                raise PaperBrokerCredentialError(
                    "Required Alpaca paper credentials are unavailable."
                ) from exc
            try:
                client = TradingClient(
                    loaded.api_key,
                    loaded.secret_key,
                    paper=True,
                    raw_data=False,
                    url_override=endpoint,
                )
            except Exception as exc:
                raise PaperBrokerConfigurationError(
                    "Alpaca paper client construction failed."
                ) from exc
            credentials_loaded = True
        else:
            credentials_loaded = False

        actual_endpoint = _client_endpoint(client)
        if actual_endpoint != ALPACA_PAPER_BASE_URL:
            raise PaperBrokerConfigurationError(
                "Trading client is not bound to the frozen paper endpoint."
            )
        self._client = client
        self._credentials_loaded = credentials_loaded
        self._paper_endpoint = endpoint

    @property
    def paper_endpoint(self) -> str:
        return self._paper_endpoint

    @property
    def credentials_loaded(self) -> bool:
        return self._credentials_loaded

    def get_account_snapshot(self) -> AccountSnapshot:
        response = _request("account", self._client.get_account)
        status = _safe_text(_field(response, "status"), field="status").upper()
        return AccountSnapshot(
            status=status,
            trading_blocked=_safe_bool(
                _field(response, "trading_blocked"),
                field="trading_blocked",
            ),
            account_blocked=_safe_bool(
                _field(response, "account_blocked"),
                field="account_blocked",
            ),
            trade_suspended_by_user=_safe_bool(
                _field(response, "trade_suspended_by_user"),
                field="trade_suspended_by_user",
            ),
            shorting_enabled=_safe_bool(
                _field(response, "shorting_enabled"),
                field="shorting_enabled",
            ),
        )

    def get_market_clock_snapshot(self) -> MarketClockSnapshot:
        response = _request("clock", self._client.get_clock)
        snapshot = MarketClockSnapshot(
            timestamp=_safe_datetime(
                _field(response, "timestamp"), field="timestamp"
            ),
            is_open=_safe_bool(_field(response, "is_open"), field="is_open"),
            next_open=_safe_datetime(
                _field(response, "next_open"), field="next_open"
            ),
            next_close=_safe_datetime(
                _field(response, "next_close"), field="next_close"
            ),
        )
        if snapshot.next_open >= snapshot.next_close:
            raise PaperBrokerResponseError(
                "Broker clock next_open must precede next_close."
            )
        return snapshot

    def get_asset_snapshot(self, symbol: str) -> AssetSnapshot:
        if not isinstance(symbol, str) or not symbol.strip():
            raise PaperBrokerConfigurationError(
                "Asset symbol must be non-empty text."
            )
        normalized_symbol = symbol.strip().upper()
        response = _request(
            f"asset {normalized_symbol}",
            lambda: self._client.get_asset(normalized_symbol),
        )
        returned_symbol = _safe_text(
            _field(response, "symbol"), field="symbol"
        ).upper()
        if returned_symbol != normalized_symbol:
            raise PaperBrokerResponseError(
                "Broker asset symbol does not match the requested symbol."
            )
        return AssetSnapshot(
            symbol=returned_symbol,
            asset_class=_safe_text(
                _field(response, "asset_class"), field="asset_class"
            ).lower(),
            status=_safe_text(
                _field(response, "status"), field="status"
            ).lower(),
            tradable=_safe_bool(
                _field(response, "tradable"), field="tradable"
            ),
            shortable=_safe_bool(
                _field(response, "shortable"), field="shortable"
            ),
            easy_to_borrow=_safe_bool(
                _field(response, "easy_to_borrow"), field="easy_to_borrow"
            ),
            fractionable=_safe_bool(
                _field(response, "fractionable"), field="fractionable"
            ),
        )


def _capability_snapshot() -> tuple[CapabilitySnapshot, ...]:
    order_constraints = {
        "market": "whole: day/gtc/opg/cls/ioc/fok; fractional: day",
        "limit": "whole: day/gtc/opg/cls/ioc/fok; fractional: day",
        "stop": "whole: day/gtc; fractional: day",
        "stop_limit": "whole: day/gtc; fractional: day",
        "trailing_stop": (
            "whole: day/gtc; fractional support requires fresh validation"
        ),
    }
    rows = [
        CapabilitySnapshot(
            capability_kind="order_type",
            capability_value=value,
            provider_supported=True,
            day18_authorized=False,
            constraint=order_constraints[value],
            evidence_source="alpaca_orders_documentation",
        )
        for value in PROVIDER_ORDER_TYPES
    ]
    rows.extend(
        CapabilitySnapshot(
            capability_kind="time_in_force",
            capability_value=value,
            provider_supported=True,
            day18_authorized=False,
            constraint=(
                "provider equity support; valid combination depends on "
                "order type, quantity mode, and session"
            ),
            evidence_source="alpaca_orders_documentation",
        )
        for value in PROVIDER_TIME_IN_FORCE
    )
    return tuple(rows)


def _sdk_version() -> str:
    try:
        return version("alpaca-py")
    except PackageNotFoundError:
        return "unknown"


def run_paper_preflight(
    broker: ReadOnlyPaperBroker,
    *,
    symbols: tuple[str, ...] = CORE_SYMBOLS,
) -> PreflightResult:
    """Run the frozen account, clock, and symbol checks in exact order."""

    if tuple(symbols) != CORE_SYMBOLS:
        raise PaperBrokerConfigurationError(
            "Day 18 symbols must be exactly SPY, QQQ, IWM in frozen order."
        )
    account = broker.get_account_snapshot()
    clock = broker.get_market_clock_snapshot()
    assets = tuple(broker.get_asset_snapshot(symbol) for symbol in symbols)

    account_passed = (
        account.status == "ACTIVE"
        and not account.trading_blocked
        and not account.account_blocked
        and not account.trade_suspended_by_user
        and account.shorting_enabled
    )
    clock_passed = clock.next_open < clock.next_close
    asset_passed = all(asset.gate_passed for asset in assets)
    endpoint_passed = broker.paper_endpoint == ALPACA_PAPER_BASE_URL
    preflight_passed = (
        endpoint_passed and account_passed and clock_passed and asset_passed
    )

    return PreflightResult(
        schema_version="day18_alpaca_paper_preflight_v1",
        paper_endpoint=broker.paper_endpoint,
        sdk_name="alpaca-py",
        sdk_version=_sdk_version(),
        credentials_loaded=broker.credentials_loaded,
        credential_values_persisted=False,
        order_submission_enabled=False,
        order_submission_occurred=False,
        call_order=PREFLIGHT_CALL_ORDER,
        core_symbols=CORE_SYMBOLS,
        account=account,
        clock=clock,
        assets=assets,
        capabilities=_capability_snapshot(),
        account_gate_passed=account_passed,
        clock_gate_passed=clock_passed,
        asset_gate_passed=asset_passed,
        preflight_passed=preflight_passed,
    )
