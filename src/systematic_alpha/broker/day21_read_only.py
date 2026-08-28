"""Mutation-free live-state capture for a blocked Day 21 session."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from systematic_alpha.broker.controlled_paper_execution import (
    ALPACA_PAPER_BASE_URL,
    DAY21_AUTHORIZATION_SCOPE,
    DAY21_SCHEMA_VERSION,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    Day21Authorization,
    Day21ExecutionResult,
    PositionCashSnapshot,
    _decimal,
    _enum_text,
    _field,
    _order_snapshot,
    day21_client_order_ids,
    evaluate_day21_startup_gates,
)
from systematic_alpha.broker.day21_signal import Day21SignalSnapshot
from systematic_alpha.broker.paper_boundary import (
    AlpacaPaperBroker,
    PreflightResult,
    run_paper_preflight,
    validate_day18_paper_config,
)
from systematic_alpha.data.config_loader import load_alpaca_credentials, load_project_config


@dataclass(frozen=True, slots=True)
class Day21ReadOnlyState:
    preflight: PreflightResult
    open_spy_orders: tuple[BrokerOrderSnapshot, ...]
    spy_position: BrokerPositionSnapshot | None
    cash: Decimal
    duplicate_signal_order: bool


class AlpacaDay21ReadOnlyReader:
    """Read account/order/position state without exposing mutation methods."""

    def __init__(self, *, config: Mapping[str, object] | None = None) -> None:
        project_config = dict(config or load_project_config())
        endpoint = validate_day18_paper_config(project_config)
        credentials = load_alpaca_credentials(project_config)
        client = TradingClient(
            credentials.api_key,
            credentials.secret_key,
            paper=True,
            raw_data=False,
            url_override=endpoint,
        )
        actual_endpoint = str(getattr(client, "_base_url", "")).rstrip("/")
        if actual_endpoint != ALPACA_PAPER_BASE_URL:
            raise RuntimeError("Read-only client is not bound to the paper endpoint.")
        self._config = project_config
        self._client = client

    def capture(self, *, entry_client_order_id: str) -> Day21ReadOnlyState:
        preflight = run_paper_preflight(
            AlpacaPaperBroker(config=self._config, client=self._client)
        )
        orders = self._client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                symbols=["SPY"],
                limit=500,
            )
        )
        normalized_orders = tuple(_order_snapshot(item) for item in orders)
        open_orders = tuple(
            item
            for item in normalized_orders
            if item.status not in {"filled", "canceled", "expired", "rejected"}
        )
        positions = tuple(
            item
            for item in self._client.get_all_positions()
            if _enum_text(_field(item, "symbol")).upper() == "SPY"
        )
        if len(positions) > 1:
            raise RuntimeError("Duplicate SPY position returned by broker.")
        position = None
        if positions:
            item = positions[0]
            position = BrokerPositionSnapshot(
                symbol="SPY",
                quantity=_decimal(_field(item, "qty"), name="position qty"),  # type: ignore[arg-type]
                average_entry_price=_decimal(
                    _field(item, "avg_entry_price"),
                    name="average entry price",
                    allow_none=True,
                ),
                market_value=_decimal(
                    _field(item, "market_value"),
                    name="market value",
                    allow_none=True,
                ),
                unrealized_pl=_decimal(
                    _field(item, "unrealized_pl"),
                    name="unrealized P&L",
                    allow_none=True,
                ),
            )
        account = self._client.get_account()
        cash = _decimal(_field(account, "cash"), name="cash")
        return Day21ReadOnlyState(
            preflight=preflight,
            open_spy_orders=open_orders,
            spy_position=position,
            cash=cash,  # type: ignore[arg-type]
            duplicate_signal_order=any(
                item.client_order_id == entry_client_order_id
                for item in normalized_orders
            ),
        )


def build_day21_read_only_result(
    *,
    signal: Day21SignalSnapshot,
    state: Day21ReadOnlyState,
    day20_gate_passed: bool,
) -> Day21ExecutionResult:
    """Create a complete abort/eligibility record with zero mutation capability."""

    authorization = Day21Authorization(
        approved=True,
        scope=DAY21_AUTHORIZATION_SCOPE,
        paper_endpoint=ALPACA_PAPER_BASE_URL,
        kill_switch_armed=True,
    )
    gates = evaluate_day21_startup_gates(
        paper_endpoint=ALPACA_PAPER_BASE_URL,
        preflight=state.preflight,
        clock=state.preflight.clock,
        signal=signal,
        authorization=authorization,
        day20_gate_passed=day20_gate_passed,
        open_orders=state.open_spy_orders,
        initial_position=state.spy_position,
        duplicate_signal_order=state.duplicate_signal_order,
    )
    entry_id, flatten_id = day21_client_order_ids(signal)
    failed = tuple(item.gate_id for item in gates if not item.passed)
    quantity = Decimal("0") if state.spy_position is None else state.spy_position.quantity
    clean_startup = quantity == 0 and not state.open_spy_orders
    return Day21ExecutionResult(
        schema_version=DAY21_SCHEMA_VERSION,
        signal=signal,
        gates=gates,
        order_events=(),
        position_cash=(
            PositionCashSnapshot(
                phase="read_only_live_probe",
                observed_at=state.preflight.clock.timestamp,
                spy_quantity=quantity,
                cash=state.cash,
            ),
        ),
        entry_client_order_id=entry_id,
        flatten_client_order_id=flatten_id,
        order_submission_occurred=False,
        entry_filled_quantity=Decimal("0"),
        flatten_filled_quantity=Decimal("0"),
        realized_round_trip_pnl=None,
        execution_complete=False,
        shutdown_reconciled=clean_startup,
        manual_recovery_required=False,
        outcome=(
            "read_only_eligible_no_submission"
            if not failed
            else "aborted_before_submission"
        ),
        abort_reasons=failed,
    )

