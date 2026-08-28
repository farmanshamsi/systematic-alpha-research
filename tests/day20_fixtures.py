from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from systematic_alpha.broker.order_state import (
    OrderIntent,
    OrderState,
    OrderStateMachine,
    OrderStatus,
    OrderUpdate,
)
from systematic_alpha.broker.reconciliation import (
    BrokerAccountSnapshot,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    OpeningPosition,
    ReconciliationInput,
)


BASE_TIMESTAMP = datetime(2025, 12, 16, 14, 30, tzinfo=timezone.utc)
AS_OF = BASE_TIMESTAMP + timedelta(seconds=10)


def client_id(order_number: int = 1) -> str:
    return f"axiom-fixture-order-{order_number:04d}"


def broker_id(order_number: int = 1) -> str:
    return f"broker-fixture-order-{order_number:04d}"


def make_local_state(
    *,
    order_number: int = 1,
    status: OrderStatus = OrderStatus.FILLED,
    requested: str = "10",
    filled: str = "10",
    average: str | None = "100",
    symbol: str = "SPY",
    side: str = "buy",
    recovery_required: bool = False,
) -> OrderState:
    machine = OrderStateMachine(f"fixture_{order_number}")
    machine.register_intent(
        OrderIntent(
            client_order_id=client_id(order_number),
            symbol=symbol,
            side=side,
            order_type="market",
            time_in_force="day",
            requested_quantity=Decimal(requested),
            submitted_at=BASE_TIMESTAMP,
        )
    )
    event_at = BASE_TIMESTAMP + timedelta(seconds=1)
    machine.apply(
        OrderUpdate(
            event_id=f"fixture-event-{order_number:04d}",
            provider_sequence=1,
            client_order_id=client_id(order_number),
            broker_order_id=broker_id(order_number),
            status=status,
            requested_quantity=Decimal(requested),
            filled_quantity=Decimal(filled),
            filled_average_price=(
                None if average is None else Decimal(average)
            ),
            event_at=event_at,
            received_at=event_at + timedelta(seconds=1),
        )
    )
    state = machine.states[client_id(order_number)]
    return (
        replace(state, recovery_required=True)
        if recovery_required
        else state
    )


def make_broker_order(
    local: OrderState,
    **overrides: object,
) -> BrokerOrderSnapshot:
    values: dict[str, object] = {
        "client_order_id": local.client_order_id,
        "broker_order_id": local.broker_order_id,
        "symbol": local.symbol,
        "side": local.side,
        "status": local.status,
        "requested_quantity": local.requested_quantity,
        "filled_quantity": local.filled_quantity,
        "filled_average_price": local.filled_average_price,
        "snapshot_at": BASE_TIMESTAMP + timedelta(seconds=5),
    }
    values.update(overrides)
    return BrokerOrderSnapshot(**values)  # type: ignore[arg-type]


def make_fill(
    local: OrderState,
    *,
    fill_number: int = 1,
    quantity: str | None = None,
    price: str = "100",
    **overrides: object,
) -> BrokerFillSnapshot:
    values: dict[str, object] = {
        "fill_id": f"fixture-fill-{fill_number:04d}",
        "client_order_id": local.client_order_id,
        "broker_order_id": local.broker_order_id,
        "symbol": local.symbol,
        "side": local.side,
        "quantity": local.filled_quantity if quantity is None else Decimal(quantity),
        "price": Decimal(price),
        "executed_at": BASE_TIMESTAMP + timedelta(seconds=4),
    }
    values.update(overrides)
    return BrokerFillSnapshot(**values)  # type: ignore[arg-type]


def matching_input(
    *,
    local: OrderState | None = None,
    broker_orders: tuple[BrokerOrderSnapshot, ...] | None = None,
    broker_fills: tuple[BrokerFillSnapshot, ...] | None = None,
    broker_positions: tuple[BrokerPositionSnapshot, ...] | None = None,
    opening_positions: tuple[OpeningPosition, ...] | None = None,
    broker_cash: str = "99000",
    opening_cash: str = "100000",
    as_of: datetime = AS_OF,
) -> ReconciliationInput:
    normalized_local = local or make_local_state()
    normalized_orders = (
        (make_broker_order(normalized_local),)
        if broker_orders is None
        else broker_orders
    )
    normalized_fills = (
        (make_fill(normalized_local),)
        if broker_fills is None
        else broker_fills
    )
    normalized_positions = (
        (
            BrokerPositionSnapshot(
                symbol="SPY",
                quantity=Decimal("10"),
                mark_price=Decimal("100"),
                snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        )
        if broker_positions is None
        else broker_positions
    )
    normalized_opening = (
        (OpeningPosition(symbol="SPY", quantity=Decimal("0")),)
        if opening_positions is None
        else opening_positions
    )
    return ReconciliationInput(
        local_states=(normalized_local,),
        broker_orders=normalized_orders,
        broker_fills=normalized_fills,
        broker_positions=normalized_positions,
        opening_positions=normalized_opening,
        opening_cash=Decimal(opening_cash),
        broker_account=BrokerAccountSnapshot(
            cash=Decimal(broker_cash),
            snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
        ),
        as_of=as_of,
    )

