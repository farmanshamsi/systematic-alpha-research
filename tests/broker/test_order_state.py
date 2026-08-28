from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from itertools import product

import pytest

from systematic_alpha.broker.order_state import (
    ACKNOWLEDGMENT_TIMEOUT,
    EVENT_STALE_AFTER,
    LEGAL_TRANSITIONS,
    MAX_FUTURE_SKEW,
    NO_FILL_STATUSES,
    PROVIDER_STATUSES,
    REASON_CODES,
    STATUS_ORDER,
    TERMINAL_STATUSES,
    UPDATE_TIMEOUT,
    OrderIntent,
    OrderStateError,
    OrderStateMachine,
    OrderStatus,
    OrderUpdate,
    is_transition_allowed,
    transition_matrix_rows,
)
from tests.day19_fixtures import (
    BASE_TIMESTAMP,
    BROKER_ORDER_ID,
    CLIENT_ORDER_ID,
    make_intent,
    make_update,
    registered_machine,
)


EXPECTED_STATUS_VALUES = (
    "intent_created",
    "pending_review",
    "pending_new",
    "held",
    "accepted",
    "accepted_for_bidding",
    "new",
    "partially_filled",
    "pending_cancel",
    "pending_replace",
    "done_for_day",
    "stopped",
    "suspended",
    "calculated",
    "filled",
    "canceled",
    "expired",
    "replaced",
    "rejected",
)


def _economic_values(
    status: OrderStatus,
    *,
    current_filled: Decimal = Decimal("0"),
    current_average: Decimal | None = None,
) -> tuple[str, str | None, str | None]:
    if status is OrderStatus.PARTIALLY_FILLED:
        filled = max(current_filled, Decimal("3"))
        return str(filled), str(current_average or Decimal("100")), None
    if status is OrderStatus.FILLED:
        return "10", "100", None
    if status in NO_FILL_STATUSES:
        return "0", None, None
    replacement = "broker-replacement-child" if status is OrderStatus.REPLACED else None
    return (
        str(current_filled),
        None if current_average is None else str(current_average),
        replacement,
    )


def _apply_status(
    machine: OrderStateMachine,
    status: OrderStatus,
    *,
    event_number: int,
) -> None:
    state = machine.states[CLIENT_ORDER_ID]
    filled, average, replacement_id = _economic_values(
        status,
        current_filled=state.filled_quantity,
        current_average=state.filled_average_price,
    )
    if status is OrderStatus.REPLACED and state.replacement_order_id is not None:
        replacement_id = state.replacement_order_id
    machine.apply(
        make_update(
            event_number=event_number,
            status=status,
            filled=filled,
            average_price=average,
            replacement_order_id=replacement_id,
        )
    )


def _assert_reason(
    machine: OrderStateMachine,
    update: OrderUpdate,
    reason: str,
) -> None:
    before = machine.states[CLIENT_ORDER_ID]
    with pytest.raises(OrderStateError) as captured:
        machine.apply(update)
    after = machine.states[CLIENT_ORDER_ID]
    assert captured.value.reason_code == reason
    assert str(captured.value) == f"Order state rejected: {reason}."
    assert after.status is before.status
    assert after.filled_quantity == before.filled_quantity
    assert after.filled_average_price == before.filled_average_price
    assert after.last_provider_sequence == before.last_provider_sequence
    assert after.recovery_required is True
    assert machine.audit_entries[-1].reason_code == reason


def test_frozen_status_and_reason_vocabularies() -> None:
    assert tuple(status.value for status in STATUS_ORDER) == EXPECTED_STATUS_VALUES
    assert len(PROVIDER_STATUSES) == 18
    assert tuple(REASON_CODES) == (
        "unknown_order",
        "idempotency_conflict",
        "duplicate_event_conflict",
        "broker_order_id_mismatch",
        "requested_quantity_mismatch",
        "provider_sequence_not_increasing",
        "event_time_regressed",
        "event_arrived_stale",
        "event_time_future_skew",
        "illegal_status_transition",
        "filled_quantity_decreased",
        "filled_quantity_exceeds_order",
        "status_quantity_inconsistent",
        "average_fill_price_inconsistent",
        "replacement_id_missing",
        "invalid_event",
        "acknowledgment_timeout",
        "update_timeout",
    )
    assert TERMINAL_STATUSES == {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REPLACED,
        OrderStatus.REJECTED,
    }


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    tuple(product(STATUS_ORDER, repeat=2)),
    ids=lambda value: value.value,
)
def test_every_transition_matrix_cell_matches_frozen_graph(
    from_status: OrderStatus,
    to_status: OrderStatus,
) -> None:
    expected = to_status is not OrderStatus.INTENT_CREATED and (
        from_status is to_status or to_status in LEGAL_TRANSITIONS[from_status]
    )
    assert is_transition_allowed(from_status, to_status) is expected


def test_transition_matrix_rows_are_complete_and_ordered() -> None:
    rows = transition_matrix_rows()
    assert len(rows) == len(STATUS_ORDER) ** 2 == 361
    assert tuple(rows[0]) == (
        "from_status",
        "to_status",
        "allowed",
        "terminal_from_status",
    )
    assert tuple((row["from_status"], row["to_status"]) for row in rows) == tuple(
        (left.value, right.value) for left, right in product(STATUS_ORDER, repeat=2)
    )


@pytest.mark.parametrize(
    "status",
    tuple(PROVIDER_STATUSES),
    ids=lambda status: status.value,
)
def test_every_provider_status_is_valid_as_restart_fast_forward(
    status: OrderStatus,
) -> None:
    machine = registered_machine()
    _apply_status(machine, status, event_number=1)
    assert machine.states[CLIENT_ORDER_ID].status is status


LEGAL_EDGES = tuple(
    (left, right)
    for left, right in product(STATUS_ORDER, repeat=2)
    if right is not OrderStatus.INTENT_CREATED
    and (left is right or right in LEGAL_TRANSITIONS[left])
)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    LEGAL_EDGES,
    ids=lambda status: status.value,
)
def test_every_legal_edge_is_accepted(
    from_status: OrderStatus,
    to_status: OrderStatus,
) -> None:
    machine = registered_machine()
    if from_status is not OrderStatus.INTENT_CREATED:
        _apply_status(machine, from_status, event_number=1)
        event_number = 2
    else:
        event_number = 1
    _apply_status(machine, to_status, event_number=event_number)
    assert machine.states[CLIENT_ORDER_ID].status is to_status


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("client_order_id", "bad"),
        ("symbol", "SPY1"),
        ("side", "hold"),
        ("order_type", "iceberg"),
        ("time_in_force", "week"),
        ("requested_quantity", Decimal("0")),
        ("requested_quantity", Decimal("NaN")),
        ("requested_quantity", True),
        ("submitted_at", BASE_TIMESTAMP.replace(tzinfo=None)),
    ),
)
def test_invalid_intent_fields_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_intent(**{field: value})


def test_intent_normalizes_symbol_decimal_and_timestamp() -> None:
    intent = make_intent(requested_quantity="10.0")
    assert intent.symbol == "SPY"
    assert intent.requested_quantity == Decimal("10.0")
    assert intent.submitted_at == BASE_TIMESTAMP


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("event_id", "event with spaces"),
        ("provider_sequence", -1),
        ("provider_sequence", True),
        ("client_order_id", "bad"),
        ("broker_order_id", " broker"),
        ("status", "not_a_status"),
        ("status", OrderStatus.INTENT_CREATED),
        ("requested_quantity", Decimal("0")),
        ("filled_quantity", Decimal("-1")),
        ("filled_quantity", Decimal("Infinity")),
        ("filled_average_price", Decimal("0")),
        ("event_at", BASE_TIMESTAMP.replace(tzinfo=None)),
        ("received_at", BASE_TIMESTAMP.replace(tzinfo=None)),
        ("replacement_order_id", "bad id"),
    ),
)
def test_invalid_update_fields_fail_closed(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        make_update(**{field: value})


def test_registration_and_exact_intent_duplicate_are_idempotent() -> None:
    machine = registered_machine()
    duplicate = machine.register_intent(make_intent())
    assert duplicate.action == "duplicate_intent_ignored"
    assert len(machine.states) == 1
    assert machine.audit_entries[-1].action == "duplicate_intent_ignored"


def test_conflicting_intent_sets_recovery() -> None:
    machine = registered_machine()
    with pytest.raises(OrderStateError) as captured:
        machine.register_intent(make_intent(side="sell"))
    assert captured.value.reason_code == "idempotency_conflict"
    assert machine.states[CLIENT_ORDER_ID].recovery_required is True


def test_cumulative_fill_converts_to_incremental_fill() -> None:
    machine = registered_machine()
    _apply_status(machine, OrderStatus.NEW, event_number=1)
    first = machine.apply(
        make_update(
            event_number=2,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="4",
            average_price="100",
        )
    )
    second = machine.apply(
        make_update(
            event_number=3,
            status=OrderStatus.FILLED,
            filled="10",
            average_price="101",
        )
    )
    assert first.incremental_fill == Decimal("4")
    assert second.incremental_fill == Decimal("6")
    assert machine.states[CLIENT_ORDER_ID].filled_quantity == Decimal("10")


def test_identical_event_delivery_is_ignored() -> None:
    machine = registered_machine()
    update = make_update(status=OrderStatus.NEW)
    machine.apply(update)
    duplicate = machine.apply(update)
    assert duplicate.action == "duplicate_event_ignored"
    assert duplicate.incremental_fill == 0


def test_later_receipt_time_does_not_change_event_identity() -> None:
    machine = registered_machine()
    update = make_update(status=OrderStatus.NEW)
    machine.apply(update)
    duplicate = machine.apply(
        replace(update, received_at=update.received_at + timedelta(seconds=5))
    )
    assert duplicate.action == "duplicate_event_ignored"


def test_conflicting_event_id_is_rejected() -> None:
    machine = registered_machine()
    update = make_update(status=OrderStatus.NEW)
    machine.apply(update)
    _assert_reason(
        machine,
        replace(update, provider_sequence=2),
        "duplicate_event_conflict",
    )


def test_rejected_event_redelivery_is_idempotently_ignored() -> None:
    machine = registered_machine()
    machine.apply(make_update(event_number=1, status=OrderStatus.NEW))
    rejected = make_update(event_number=2, sequence=1)
    _assert_reason(
        machine,
        rejected,
        "provider_sequence_not_increasing",
    )
    duplicate = machine.apply(rejected)
    assert duplicate.action == "duplicate_event_ignored"
    assert duplicate.state is not None
    assert duplicate.state.status is OrderStatus.NEW


def test_unknown_event_redelivery_is_idempotently_ignored() -> None:
    machine = OrderStateMachine("unknown_duplicate")
    update = make_update()
    with pytest.raises(OrderStateError) as captured:
        machine.apply(update)
    assert captured.value.reason_code == "unknown_order"
    duplicate = machine.apply(update)
    assert duplicate.action == "duplicate_event_ignored"
    assert duplicate.state is None
    assert machine.global_recovery_required is True


@pytest.mark.parametrize(
    ("update", "reason"),
    (
        (
            make_update(event_number=2, broker_order_id="broker-other"),
            "broker_order_id_mismatch",
        ),
        (
            make_update(event_number=2, requested_quantity=Decimal("11")),
            "requested_quantity_mismatch",
        ),
        (
            make_update(event_number=2, sequence=1),
            "provider_sequence_not_increasing",
        ),
        (
            make_update(
                event_number=2,
                event_at=BASE_TIMESTAMP,
                received_at=BASE_TIMESTAMP + timedelta(seconds=2),
            ),
            "event_time_regressed",
        ),
        (
            make_update(
                event_number=2,
                event_at=BASE_TIMESTAMP + timedelta(seconds=2),
                received_at=BASE_TIMESTAMP + EVENT_STALE_AFTER + timedelta(seconds=3),
            ),
            "event_arrived_stale",
        ),
        (
            make_update(
                event_number=2,
                event_at=BASE_TIMESTAMP + MAX_FUTURE_SKEW + timedelta(seconds=3),
                received_at=BASE_TIMESTAMP + timedelta(seconds=2),
            ),
            "event_time_future_skew",
        ),
        (
            make_update(event_number=2, status=OrderStatus.ACCEPTED),
            "illegal_status_transition",
        ),
        (
            make_update(
                event_number=2,
                status=OrderStatus.FILLED,
                filled="11",
                average_price="100",
            ),
            "filled_quantity_exceeds_order",
        ),
        (
            make_update(event_number=2, status=OrderStatus.PARTIALLY_FILLED),
            "status_quantity_inconsistent",
        ),
        (
            make_update(
                event_number=2,
                status=OrderStatus.PARTIALLY_FILLED,
                filled="3",
            ),
            "average_fill_price_inconsistent",
        ),
    ),
)
def test_known_order_rejection_paths_preserve_economic_state(
    update: OrderUpdate,
    reason: str,
) -> None:
    machine = registered_machine()
    machine.apply(make_update(event_number=1, status=OrderStatus.NEW))
    _assert_reason(machine, update, reason)


def test_decreasing_cumulative_fill_is_rejected() -> None:
    machine = registered_machine()
    _apply_status(machine, OrderStatus.NEW, event_number=1)
    machine.apply(
        make_update(
            event_number=2,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="5",
            average_price="100",
        )
    )
    _assert_reason(
        machine,
        make_update(
            event_number=3,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="4",
            average_price="100",
        ),
        "filled_quantity_decreased",
    )


def test_missing_replacement_identifier_is_rejected() -> None:
    machine = registered_machine()
    _apply_status(machine, OrderStatus.NEW, event_number=1)
    _apply_status(machine, OrderStatus.PENDING_REPLACE, event_number=2)
    _assert_reason(
        machine,
        make_update(event_number=3, status=OrderStatus.REPLACED),
        "replacement_id_missing",
    )


def test_terminal_same_status_confirmation_cannot_change_replacement_id() -> None:
    machine = registered_machine()
    _apply_status(machine, OrderStatus.REPLACED, event_number=1)
    _assert_reason(
        machine,
        make_update(
            event_number=2,
            status=OrderStatus.REPLACED,
            replacement_order_id="broker-different-child",
        ),
        "replacement_id_missing",
    )


def test_unknown_order_sets_only_global_recovery() -> None:
    machine = OrderStateMachine("unknown")
    with pytest.raises(OrderStateError) as captured:
        machine.apply(make_update())
    assert captured.value.reason_code == "unknown_order"
    assert machine.global_recovery_required is True
    assert not machine.states
    assert machine.audit_entries[-1].resulting_status == "unknown"


def test_invalid_normalized_event_is_audited_without_raw_payload() -> None:
    machine = registered_machine()
    with pytest.raises(OrderStateError) as captured:
        machine.record_invalid_event(
            client_order_id=CLIENT_ORDER_ID,
            event_id="safe-event-id",
            received_at=BASE_TIMESTAMP + timedelta(seconds=1),
        )
    assert captured.value.reason_code == "invalid_event"
    entry = machine.audit_entries[-1]
    assert entry.event_id == "safe-event-id"
    assert entry.provider_sequence is None
    assert entry.incoming_status == "invalid_event"
    assert machine.states[CLIENT_ORDER_ID].recovery_required is True
    duplicate = machine.record_invalid_event(
        client_order_id=CLIENT_ORDER_ID,
        event_id="safe-event-id",
        received_at=BASE_TIMESTAMP + timedelta(seconds=1),
    )
    assert duplicate.action == "duplicate_event_ignored"


def test_acknowledgment_timeout_boundary_and_idempotency() -> None:
    machine = registered_machine()
    assert machine.check_timeouts(BASE_TIMESTAMP + ACKNOWLEDGMENT_TIMEOUT) == ()
    created = machine.check_timeouts(
        BASE_TIMESTAMP + ACKNOWLEDGMENT_TIMEOUT + timedelta(seconds=1)
    )
    assert len(created) == 1
    assert created[0].reason_code == "acknowledgment_timeout"
    assert machine.check_timeouts(
        BASE_TIMESTAMP + ACKNOWLEDGMENT_TIMEOUT + timedelta(seconds=2)
    ) == ()


def test_update_timeout_boundary_new_sequence_and_terminal_exemption() -> None:
    machine = registered_machine()
    accepted = machine.apply(make_update(status=OrderStatus.NEW))
    reference = accepted.state.last_received_at
    assert reference is not None
    assert machine.check_timeouts(reference + UPDATE_TIMEOUT) == ()
    created = machine.check_timeouts(
        reference + UPDATE_TIMEOUT + timedelta(seconds=1)
    )
    assert len(created) == 1
    assert created[0].reason_code == "update_timeout"
    assert machine.check_timeouts(
        reference + UPDATE_TIMEOUT + timedelta(seconds=2)
    ) == ()

    terminal = registered_machine("terminal")
    _apply_status(terminal, OrderStatus.FILLED, event_number=1)
    assert terminal.check_timeouts(BASE_TIMESTAMP + timedelta(days=1)) == ()


def test_timeout_as_of_before_reference_fails_closed() -> None:
    machine = registered_machine()
    with pytest.raises(ValueError, match="cannot precede"):
        machine.check_timeouts(BASE_TIMESTAMP - timedelta(seconds=1))


def test_public_methods_reject_wrong_types() -> None:
    machine = OrderStateMachine("types")
    with pytest.raises(TypeError):
        machine.register_intent(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        machine.apply(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_transition_allowed("new", OrderStatus.NEW)  # type: ignore[arg-type]
