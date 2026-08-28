from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

from systematic_alpha.broker.paper_boundary import (
    ALPACA_PAPER_BASE_URL,
    CORE_SYMBOLS,
    PREFLIGHT_CALL_ORDER,
    PROVIDER_ORDER_TYPES,
    PROVIDER_TIME_IN_FORCE,
    AlpacaPaperBroker,
    PaperBrokerConfigurationError,
    PaperBrokerConnectionError,
    PaperBrokerResponseError,
    run_paper_preflight,
    validate_day18_paper_config,
)
from systematic_alpha.data.config_loader import (
    AlpacaCredentials,
    ProjectConfigError,
    load_alpaca_credentials,
)
from tests.day18_fixtures import (
    FakeTradingClient,
    account_response,
    asset_response,
    clock_response,
    safe_day18_config,
)


def _set(config: dict[str, object], path: str, value: object) -> None:
    section_name, key = path.split(".", maxsplit=1)
    section = config[section_name]
    assert isinstance(section, dict)
    section[key] = value


def test_exact_safe_configuration_passes() -> None:
    assert validate_day18_paper_config(safe_day18_config()) == (
        ALPACA_PAPER_BASE_URL
    )


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("project.environment", "live"),
        ("broker.provider", "other"),
        ("broker.paper", False),
        ("broker.paper_base_url", "https://api.alpaca.markets"),
        ("safety.allow_live_trading", True),
        ("safety.require_paper_mode", False),
        ("safety.require_manual_order_confirmation", False),
        ("safety.kill_switch_enabled", False),
        ("execution.order_submission_enabled", True),
        ("broker.credentials_file", "credentials.txt"),
        ("broker.api_key_env", ""),
        ("broker.secret_key_env", ""),
    ],
)
def test_unsafe_configuration_fails_closed(
    path: str,
    unsafe_value: object,
) -> None:
    config = deepcopy(safe_day18_config())
    _set(config, path, unsafe_value)
    with pytest.raises(PaperBrokerConfigurationError):
        validate_day18_paper_config(config)


def test_same_credential_variable_name_is_rejected() -> None:
    config = safe_day18_config()
    config["broker"]["secret_key_env"] = "ALPACA_API_KEY"
    with pytest.raises(PaperBrokerConfigurationError):
        validate_day18_paper_config(config)


def test_paper_endpoint_accepts_one_trailing_slash() -> None:
    config = safe_day18_config()
    config["broker"]["paper_base_url"] += "/"
    assert validate_day18_paper_config(config) == ALPACA_PAPER_BASE_URL


def test_injected_live_client_endpoint_is_rejected() -> None:
    with pytest.raises(PaperBrokerConfigurationError):
        AlpacaPaperBroker(
            config=safe_day18_config(),
            client=FakeTradingClient(
                endpoint="https://api.alpaca.markets"
            ),
        )


def test_unverifiable_client_endpoint_is_rejected() -> None:
    class ClientWithoutEndpoint:
        pass

    with pytest.raises(PaperBrokerConfigurationError):
        AlpacaPaperBroker(
            config=safe_day18_config(),
            client=ClientWithoutEndpoint(),
        )


def test_client_and_credentials_cannot_both_be_injected() -> None:
    with pytest.raises(PaperBrokerConfigurationError):
        AlpacaPaperBroker(
            config=safe_day18_config(),
            client=FakeTradingClient(),
            credentials=AlpacaCredentials("key", "secret"),
        )


def test_credentials_are_redacted_from_repr() -> None:
    credentials = AlpacaCredentials("visible-key", "visible-secret")
    representation = repr(credentials)
    assert "visible-key" not in representation
    assert "visible-secret" not in representation


def test_process_environment_credentials_take_precedence(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "process-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "process-secret")
    credentials = load_alpaca_credentials(safe_day18_config())
    assert credentials.api_key == "process-key"
    assert credentials.secret_key == "process-secret"


def test_partial_process_environment_credentials_fail(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "do-not-print")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ProjectConfigError) as captured:
        load_alpaca_credentials(safe_day18_config())
    assert "do-not-print" not in str(captured.value)


def test_exact_read_only_call_order_and_normalization() -> None:
    client = FakeTradingClient()
    broker = AlpacaPaperBroker(config=safe_day18_config(), client=client)
    result = run_paper_preflight(broker)

    assert tuple(client.calls) == PREFLIGHT_CALL_ORDER
    assert result.call_order == PREFLIGHT_CALL_ORDER
    assert result.core_symbols == CORE_SYMBOLS
    assert result.account.status == "ACTIVE"
    assert result.clock.timestamp.tzinfo is not None
    assert tuple(asset.symbol for asset in result.assets) == CORE_SYMBOLS
    assert result.preflight_passed is True
    assert result.order_submission_enabled is False
    assert result.order_submission_occurred is False


def test_mapping_responses_are_normalized() -> None:
    client = FakeTradingClient(
        account=vars(account_response()),
        clock=vars(clock_response()),
        assets={
            symbol: vars(asset_response(symbol)) for symbol in CORE_SYMBOLS
        },
    )
    result = run_paper_preflight(
        AlpacaPaperBroker(config=safe_day18_config(), client=client)
    )
    assert result.preflight_passed is True


def test_closed_market_is_valid() -> None:
    client = FakeTradingClient(clock=clock_response(is_open=False))
    result = run_paper_preflight(
        AlpacaPaperBroker(config=safe_day18_config(), client=client)
    )
    assert result.clock.is_open is False
    assert result.clock_gate_passed is True
    assert result.preflight_passed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "INACTIVE"),
        ("trading_blocked", True),
        ("account_blocked", True),
        ("trade_suspended_by_user", True),
        ("shorting_enabled", False),
    ],
)
def test_account_gate_failures_remain_explicit(
    field: str,
    value: object,
) -> None:
    client = FakeTradingClient(
        account=account_response(**{field: value})
    )
    result = run_paper_preflight(
        AlpacaPaperBroker(config=safe_day18_config(), client=client)
    )
    assert result.account_gate_passed is False
    assert result.preflight_passed is False
    assert tuple(client.calls) == PREFLIGHT_CALL_ORDER


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_class", "crypto"),
        ("status", "inactive"),
        ("tradable", False),
        ("shortable", False),
        ("easy_to_borrow", False),
        ("fractionable", False),
    ],
)
def test_asset_gate_failures_remain_explicit(
    field: str,
    value: object,
) -> None:
    assets = {symbol: asset_response(symbol) for symbol in CORE_SYMBOLS}
    assets["SPY"] = asset_response("SPY", **{field: value})
    result = run_paper_preflight(
        AlpacaPaperBroker(
            config=safe_day18_config(),
            client=FakeTradingClient(assets=assets),
        )
    )
    assert result.asset_gate_passed is False
    assert result.preflight_passed is False


def test_wrong_returned_symbol_fails_closed() -> None:
    assets = {symbol: asset_response(symbol) for symbol in CORE_SYMBOLS}
    assets["SPY"] = asset_response("QQQ")
    broker = AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(assets=assets),
    )
    with pytest.raises(PaperBrokerResponseError):
        run_paper_preflight(broker)


def test_missing_response_field_fails_closed() -> None:
    account = vars(account_response())
    del account["trading_blocked"]
    broker = AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(account=account),
    )
    with pytest.raises(PaperBrokerResponseError):
        run_paper_preflight(broker)


def test_non_boolean_response_field_fails_closed() -> None:
    broker = AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(account=account_response(trading_blocked=0)),
    )
    with pytest.raises(PaperBrokerResponseError):
        run_paper_preflight(broker)


def test_naive_clock_datetime_fails_closed() -> None:
    broker = AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(
            clock=clock_response(timestamp=datetime(2026, 8, 2, 12, 0))
        ),
    )
    with pytest.raises(PaperBrokerResponseError):
        run_paper_preflight(broker)


def test_invalid_next_open_close_order_fails_closed() -> None:
    clock = clock_response()
    broker = AlpacaPaperBroker(
        config=safe_day18_config(),
        client=FakeTradingClient(
            clock=clock_response(next_open=clock.next_close)
        ),
    )
    with pytest.raises(PaperBrokerResponseError):
        run_paper_preflight(broker)


def test_connection_error_is_generic_and_does_not_copy_secret() -> None:
    class FailingClient(FakeTradingClient):
        def get_account(self) -> object:
            raise RuntimeError("network failed secret-value")

    broker = AlpacaPaperBroker(
        config=safe_day18_config(), client=FailingClient()
    )
    with pytest.raises(PaperBrokerConnectionError) as captured:
        run_paper_preflight(broker)
    assert str(captured.value) == "Read-only Alpaca account request failed."
    assert "secret-value" not in str(captured.value)


def test_symbol_order_is_frozen() -> None:
    broker = AlpacaPaperBroker(
        config=safe_day18_config(), client=FakeTradingClient()
    )
    with pytest.raises(PaperBrokerConfigurationError):
        run_paper_preflight(broker, symbols=("QQQ", "SPY", "IWM"))


def test_capability_order_and_authorization_are_frozen() -> None:
    result = run_paper_preflight(
        AlpacaPaperBroker(
            config=safe_day18_config(), client=FakeTradingClient()
        )
    )
    values = tuple(row.capability_value for row in result.capabilities)
    assert values == PROVIDER_ORDER_TYPES + PROVIDER_TIME_IN_FORCE
    assert all(row.provider_supported for row in result.capabilities)
    assert not any(row.day18_authorized for row in result.capabilities)


def test_public_adapter_exposes_no_order_mutation_methods() -> None:
    public_names = {
        name for name in dir(AlpacaPaperBroker) if not name.startswith("_")
    }
    prohibited = {
        "submit_order",
        "replace_order",
        "cancel_order",
        "cancel_orders",
        "close_position",
        "close_all_positions",
    }
    assert public_names.isdisjoint(prohibited)
