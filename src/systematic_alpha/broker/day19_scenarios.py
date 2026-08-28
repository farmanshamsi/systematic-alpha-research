"""Frozen synthetic Day 19 order-state scenarios and result tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Mapping

from systematic_alpha.broker.order_state import (
    AuditEntry,
    OrderIntent,
    OrderState,
    OrderStateError,
    OrderStateMachine,
    OrderStatus,
    OrderUpdate,
    TimeoutDiagnostic,
    transition_matrix_rows,
)


SCENARIO_ORDER: Final[tuple[str, ...]] = (
    "complete_fill",
    "cancel_after_partial",
    "broker_rejection",
    "replacement",
    "duplicate_delivery",
    "decreasing_fill_rejected",
    "out_of_order_rejected",
    "acknowledgment_timeout",
    "unknown_order_rejected",
)
BASE_TIMESTAMP: Final[datetime] = datetime(
    2025, 12, 15, 14, 30, tzinfo=timezone.utc
)
REQUESTED_QUANTITY: Final[Decimal] = Decimal("10")


@dataclass(frozen=True, slots=True)
class Day19ScenarioResults:
    """Frozen tabular result of all nine synthetic scenarios."""

    scenario_summary: tuple[Mapping[str, object], ...]
    final_states: tuple[Mapping[str, object], ...]
    transition_log: tuple[Mapping[str, object], ...]
    rejection_diagnostics: tuple[Mapping[str, object], ...]
    timeout_diagnostics: tuple[Mapping[str, object], ...]
    state_transition_matrix: tuple[Mapping[str, object], ...]
    evaluation_complete: bool

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_summary",
            "final_states",
            "transition_log",
            "rejection_diagnostics",
            "timeout_diagnostics",
            "state_transition_matrix",
        ):
            rows = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                tuple(MappingProxyType(dict(row)) for row in rows),
            )


def _intent(scenario_id: str) -> OrderIntent:
    return OrderIntent(
        client_order_id=f"axiom-{scenario_id.replace('_', '-')}",
        symbol="SPY",
        side="buy",
        order_type="market",
        time_in_force="day",
        requested_quantity=REQUESTED_QUANTITY,
        submitted_at=BASE_TIMESTAMP,
    )


def _update(
    scenario_id: str,
    *,
    event_number: int,
    sequence: int,
    status: OrderStatus,
    filled: str = "0",
    average_price: str | None = None,
    event_offset_seconds: int | None = None,
    client_order_id: str | None = None,
    replacement_order_id: str | None = None,
) -> OrderUpdate:
    offset = event_number if event_offset_seconds is None else event_offset_seconds
    event_at = BASE_TIMESTAMP + timedelta(seconds=offset)
    return OrderUpdate(
        event_id=f"event-{scenario_id}-{event_number}",
        provider_sequence=sequence,
        client_order_id=client_order_id or _intent(scenario_id).client_order_id,
        broker_order_id=f"broker-{scenario_id}",
        status=status,
        requested_quantity=REQUESTED_QUANTITY,
        filled_quantity=Decimal(filled),
        filled_average_price=(
            None if average_price is None else Decimal(average_price)
        ),
        event_at=event_at,
        received_at=event_at + timedelta(seconds=1),
        replacement_order_id=replacement_order_id,
    )


def _apply_expected_rejection(
    machine: OrderStateMachine,
    update: OrderUpdate,
    reason_code: str,
) -> None:
    try:
        machine.apply(update)
    except OrderStateError as exc:
        if exc.reason_code != reason_code:
            raise RuntimeError(
                f"Scenario expected {reason_code}, received {exc.reason_code}."
            ) from exc
    else:
        raise RuntimeError(f"Scenario expected rejection {reason_code}.")


def _run_complete_fill() -> OrderStateMachine:
    scenario = "complete_fill"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    updates = (
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.PENDING_NEW,
        ),
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.ACCEPTED,
        ),
        _update(
            scenario,
            event_number=3,
            sequence=3,
            status=OrderStatus.NEW,
        ),
        _update(
            scenario,
            event_number=4,
            sequence=4,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="4",
            average_price="100.00",
        ),
        _update(
            scenario,
            event_number=5,
            sequence=5,
            status=OrderStatus.FILLED,
            filled="10",
            average_price="100.50",
        ),
    )
    for update in updates:
        machine.apply(update)
    return machine


def _run_cancel_after_partial() -> OrderStateMachine:
    scenario = "cancel_after_partial"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    updates = (
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.NEW,
        ),
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="3",
            average_price="101.25",
        ),
        _update(
            scenario,
            event_number=3,
            sequence=3,
            status=OrderStatus.PENDING_CANCEL,
            filled="3",
            average_price="101.25",
        ),
        _update(
            scenario,
            event_number=4,
            sequence=4,
            status=OrderStatus.CANCELED,
            filled="3",
            average_price="101.25",
        ),
    )
    for update in updates:
        machine.apply(update)
    return machine


def _run_broker_rejection() -> OrderStateMachine:
    scenario = "broker_rejection"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    machine.apply(
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.PENDING_NEW,
        )
    )
    machine.apply(
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.REJECTED,
        )
    )
    return machine


def _run_replacement() -> OrderStateMachine:
    scenario = "replacement"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    updates = (
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.NEW,
        ),
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.PENDING_REPLACE,
        ),
        _update(
            scenario,
            event_number=3,
            sequence=3,
            status=OrderStatus.REPLACED,
            replacement_order_id="broker-replacement-child",
        ),
    )
    for update in updates:
        machine.apply(update)
    return machine


def _run_duplicate_delivery() -> OrderStateMachine:
    scenario = "duplicate_delivery"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    update = _update(
        scenario,
        event_number=1,
        sequence=1,
        status=OrderStatus.NEW,
    )
    machine.apply(update)
    machine.apply(update)
    return machine


def _run_decreasing_fill_rejected() -> OrderStateMachine:
    scenario = "decreasing_fill_rejected"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    machine.apply(
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.NEW,
        )
    )
    machine.apply(
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="5",
            average_price="99.75",
        )
    )
    _apply_expected_rejection(
        machine,
        _update(
            scenario,
            event_number=3,
            sequence=3,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="4",
            average_price="99.75",
        ),
        "filled_quantity_decreased",
    )
    return machine


def _run_out_of_order_rejected() -> OrderStateMachine:
    scenario = "out_of_order_rejected"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    machine.apply(
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.NEW,
        )
    )
    machine.apply(
        _update(
            scenario,
            event_number=3,
            sequence=3,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="4",
            average_price="102.00",
        )
    )
    _apply_expected_rejection(
        machine,
        _update(
            scenario,
            event_number=2,
            sequence=2,
            status=OrderStatus.PARTIALLY_FILLED,
            filled="2",
            average_price="101.50",
        ),
        "provider_sequence_not_increasing",
    )
    return machine


def _run_acknowledgment_timeout() -> OrderStateMachine:
    scenario = "acknowledgment_timeout"
    machine = OrderStateMachine(scenario)
    machine.register_intent(_intent(scenario))
    diagnostics = machine.check_timeouts(
        BASE_TIMESTAMP + timedelta(seconds=31)
    )
    if len(diagnostics) != 1:
        raise RuntimeError("Acknowledgment-timeout scenario did not fire once.")
    return machine


def _run_unknown_order_rejected() -> OrderStateMachine:
    scenario = "unknown_order_rejected"
    machine = OrderStateMachine(scenario)
    _apply_expected_rejection(
        machine,
        _update(
            scenario,
            event_number=1,
            sequence=1,
            status=OrderStatus.NEW,
            client_order_id="axiom-unknown-event",
        ),
        "unknown_order",
    )
    return machine


def _scenario_machines() -> tuple[OrderStateMachine, ...]:
    return (
        _run_complete_fill(),
        _run_cancel_after_partial(),
        _run_broker_rejection(),
        _run_replacement(),
        _run_duplicate_delivery(),
        _run_decreasing_fill_rejected(),
        _run_out_of_order_rejected(),
        _run_acknowledgment_timeout(),
        _run_unknown_order_rejected(),
    )


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _state_row(
    scenario_order: int,
    scenario_id: str,
    state: OrderState,
) -> dict[str, object]:
    return {
        "scenario_order": scenario_order,
        "scenario_id": scenario_id,
        "client_order_id": state.client_order_id,
        "broker_order_id": state.broker_order_id or "",
        "symbol": state.symbol,
        "side": state.side,
        "order_type": state.order_type,
        "time_in_force": state.time_in_force,
        "requested_quantity": str(state.requested_quantity),
        "status": state.status.value,
        "filled_quantity": str(state.filled_quantity),
        "filled_average_price": (
            ""
            if state.filled_average_price is None
            else str(state.filled_average_price)
        ),
        "last_provider_sequence": (
            ""
            if state.last_provider_sequence is None
            else state.last_provider_sequence
        ),
        "last_event_at": _iso(state.last_event_at),
        "last_received_at": _iso(state.last_received_at),
        "replacement_order_id": state.replacement_order_id or "",
        "terminal": state.terminal,
        "recovery_required": state.recovery_required,
    }


def _audit_row(entry: AuditEntry) -> dict[str, object]:
    return {
        "scenario_id": entry.scenario_id,
        "audit_sequence": entry.audit_sequence,
        "client_order_id": entry.client_order_id,
        "broker_order_id": entry.broker_order_id,
        "event_id": entry.event_id,
        "provider_sequence": (
            "" if entry.provider_sequence is None else entry.provider_sequence
        ),
        "previous_status": entry.previous_status,
        "incoming_status": entry.incoming_status,
        "resulting_status": entry.resulting_status,
        "action": entry.action,
        "incremental_fill": str(entry.incremental_fill),
        "cumulative_filled_quantity": str(
            entry.cumulative_filled_quantity
        ),
        "reason_code": entry.reason_code,
        "event_at": _iso(entry.event_at),
        "received_at": _iso(entry.received_at),
        "recovery_required": entry.recovery_required,
    }


def _timeout_row(row: TimeoutDiagnostic) -> dict[str, object]:
    return {
        "scenario_id": row.scenario_id,
        "client_order_id": row.client_order_id,
        "reason_code": row.reason_code,
        "as_of": _iso(row.as_of),
        "reference_time": _iso(row.reference_time),
        "elapsed_seconds": row.elapsed_seconds,
        "last_provider_sequence": (
            "" if row.last_provider_sequence is None else row.last_provider_sequence
        ),
        "recovery_required": row.recovery_required,
    }


def run_day19_scenarios() -> Day19ScenarioResults:
    """Run and validate the exact nine-scenario synthetic suite."""

    machines = _scenario_machines()
    if tuple(machine.scenario_id for machine in machines) != SCENARIO_ORDER:
        raise RuntimeError("Day 19 scenario order changed.")

    expected = {
        "complete_fill": ("filled", "", False, False),
        "cancel_after_partial": ("canceled", "", False, False),
        "broker_rejection": ("rejected", "", False, False),
        "replacement": ("replaced", "", False, False),
        "duplicate_delivery": ("new", "", False, False),
        "decreasing_fill_rejected": (
            "partially_filled",
            "filled_quantity_decreased",
            True,
            False,
        ),
        "out_of_order_rejected": (
            "partially_filled",
            "provider_sequence_not_increasing",
            True,
            False,
        ),
        "acknowledgment_timeout": (
            "intent_created",
            "acknowledgment_timeout",
            True,
            False,
        ),
        "unknown_order_rejected": (
            "",
            "unknown_order",
            False,
            True,
        ),
    }

    scenario_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    timeout_rows: list[dict[str, object]] = []

    for scenario_order, machine in enumerate(machines, start=1):
        states = tuple(machine.states.values())
        final_status = states[0].status.value if states else ""
        recovery_required = any(state.recovery_required for state in states)
        rejections = tuple(
            entry for entry in machine.audit_entries if entry.action == "rejected"
        )
        duplicates = tuple(
            entry
            for entry in machine.audit_entries
            if entry.action in {
                "duplicate_event_ignored",
                "duplicate_intent_ignored",
            }
        )
        expected_status, expected_reason, expected_recovery, expected_global = (
            expected[machine.scenario_id]
        )
        observed_reason = (
            rejections[-1].reason_code
            if rejections
            else (
                machine.timeout_diagnostics[-1].reason_code
                if machine.timeout_diagnostics
                else ""
            )
        )
        scenario_passed = (
            final_status == expected_status
            and observed_reason == expected_reason
            and recovery_required is expected_recovery
            and machine.global_recovery_required is expected_global
        )
        if not scenario_passed:
            raise RuntimeError(
                f"Day 19 scenario failed: {machine.scenario_id}."
            )
        scenario_rows.append(
            {
                "scenario_order": scenario_order,
                "scenario_id": machine.scenario_id,
                "expected_final_status": expected_status,
                "observed_final_status": final_status,
                "expected_reason_code": expected_reason,
                "observed_reason_code": observed_reason,
                "accepted_events": sum(
                    entry.action == "applied" for entry in machine.audit_entries
                ),
                "duplicate_events": len(duplicates),
                "rejected_events": len(rejections),
                "timeout_events": len(machine.timeout_diagnostics),
                "recovery_required": recovery_required,
                "global_recovery_required": (
                    machine.global_recovery_required
                ),
                "scenario_passed": scenario_passed,
            }
        )
        state_rows.extend(
            _state_row(scenario_order, machine.scenario_id, state)
            for state in states
        )
        machine_audit_rows = tuple(
            _audit_row(entry) for entry in machine.audit_entries
        )
        audit_rows.extend(machine_audit_rows)
        rejection_rows.extend(
            row for row in machine_audit_rows if row["action"] == "rejected"
        )
        timeout_rows.extend(
            _timeout_row(row) for row in machine.timeout_diagnostics
        )

    return Day19ScenarioResults(
        scenario_summary=tuple(scenario_rows),
        final_states=tuple(state_rows),
        transition_log=tuple(audit_rows),
        rejection_diagnostics=tuple(rejection_rows),
        timeout_diagnostics=tuple(timeout_rows),
        state_transition_matrix=transition_matrix_rows(),
        evaluation_complete=all(row["scenario_passed"] for row in scenario_rows),
    )
