from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from systematic_alpha.broker.order_state import (
    OrderIntent,
    OrderStateMachine,
    OrderStatus,
    OrderUpdate,
)


BASE_TIMESTAMP = datetime(2025, 12, 15, 14, 30, tzinfo=timezone.utc)
CLIENT_ORDER_ID = "axiom-test-order-0001"
BROKER_ORDER_ID = "broker-test-order-0001"
REQUESTED_QUANTITY = Decimal("10")


def make_intent(**overrides: object) -> OrderIntent:
    values: dict[str, object] = {
        "client_order_id": CLIENT_ORDER_ID,
        "symbol": "spy",
        "side": "buy",
        "order_type": "market",
        "time_in_force": "day",
        "requested_quantity": REQUESTED_QUANTITY,
        "submitted_at": BASE_TIMESTAMP,
    }
    values.update(overrides)
    return OrderIntent(**values)  # type: ignore[arg-type]


def make_update(
    *,
    event_number: int = 1,
    sequence: int | None = None,
    status: OrderStatus = OrderStatus.NEW,
    filled: str = "0",
    average_price: str | None = None,
    event_at: datetime | None = None,
    received_at: datetime | None = None,
    replacement_order_id: str | None = None,
    **overrides: object,
) -> OrderUpdate:
    normalized_event_at = event_at or (
        BASE_TIMESTAMP + timedelta(seconds=event_number)
    )
    normalized_received_at = received_at or (
        normalized_event_at + timedelta(seconds=1)
    )
    values: dict[str, object] = {
        "event_id": f"event-test-{event_number}",
        "provider_sequence": event_number if sequence is None else sequence,
        "client_order_id": CLIENT_ORDER_ID,
        "broker_order_id": BROKER_ORDER_ID,
        "status": status,
        "requested_quantity": REQUESTED_QUANTITY,
        "filled_quantity": Decimal(filled),
        "filled_average_price": (
            None if average_price is None else Decimal(average_price)
        ),
        "event_at": normalized_event_at,
        "received_at": normalized_received_at,
        "replacement_order_id": replacement_order_id,
    }
    values.update(overrides)
    return OrderUpdate(**values)  # type: ignore[arg-type]


def registered_machine(
    scenario_id: str = "test_scenario",
) -> OrderStateMachine:
    machine = OrderStateMachine(scenario_id)
    machine.register_intent(make_intent())
    return machine

