"""Frozen synthetic Day 20 reconciliation and monitoring scenarios."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Mapping

from systematic_alpha.broker.monitoring import (
    OperationalDecision,
    StreamHealthMonitor,
    evaluate_operational_gate,
)
from systematic_alpha.broker.order_state import (
    OrderIntent,
    OrderState,
    OrderStateMachine,
    OrderStatus,
    OrderUpdate,
)
from systematic_alpha.broker.reconciliation import (
    BalanceComparison,
    BrokerAccountSnapshot,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    OpeningPosition,
    ReconciliationInput,
    ReconciliationResult,
    reconcile_snapshot,
)


SCENARIO_ORDER: Final[tuple[str, ...]] = (
    "fully_reconciled",
    "partial_fill_reconciled",
    "missed_stream_update",
    "orphan_broker_order",
    "position_mismatch",
    "cash_mismatch",
    "local_recovery_required",
    "stale_stream_recovered",
    "reconnect_exhausted",
    "single_order_limit_breach",
    "gross_exposure_breach",
    "kill_switch_latched",
)
BASE_TIMESTAMP: Final[datetime] = datetime(
    2025, 12, 16, 14, 30, tzinfo=timezone.utc
)
OPENING_CASH: Final[Decimal] = Decimal("100000")


SCENARIO_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "expected_operational_gate_passed",
    "observed_operational_gate_passed",
    "expected_reason_codes",
    "observed_reason_codes",
    "reconciliation_diagnostic_count",
    "stream_audit_count",
    "stream_final_state",
    "day20_order_submission_authorized",
    "can_submit_orders",
    "scenario_passed",
)
RECONCILIATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "local_order_count",
    "broker_order_count",
    "unique_fill_count",
    "exact_duplicate_fill_count",
    "broker_position_count",
    "expected_cash",
    "broker_cash",
    "gross_notional",
    "open_order_count",
    "core_reconciliation_passed",
    "limits_passed",
    "reconciliation_passed",
)
RECONCILIATION_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "diagnostic_sequence",
    "category",
    "reason_code",
    "client_order_id",
    "broker_order_id",
    "symbol",
    "local_value",
    "broker_value",
    "required_action",
)
POSITION_CASH_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "balance_type",
    "symbol",
    "opening_value",
    "signed_fill_flow",
    "expected_value",
    "broker_value",
    "difference",
    "tolerance",
    "reconciled",
)
STREAM_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "audit_sequence",
    "stream_id",
    "action",
    "previous_state",
    "resulting_state",
    "event_id",
    "occurred_at",
    "reconnect_attempt",
    "next_retry_at",
    "reason_code",
    "submission_blocked",
)
OPERATIONAL_DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_order",
    "scenario_id",
    "reconciliation_passed",
    "limits_passed",
    "stream_state",
    "stream_safe",
    "circuit_breaker_open",
    "kill_switch_latched",
    "active_reason_codes",
    "operational_gate_passed",
    "day20_order_submission_authorized",
    "can_submit_orders",
)


@dataclass(frozen=True, slots=True)
class Day20ScenarioResults:
    scenario_summary: tuple[Mapping[str, object], ...]
    reconciliation_summary: tuple[Mapping[str, object], ...]
    reconciliation_diagnostics: tuple[Mapping[str, object], ...]
    position_cash_reconciliation: tuple[Mapping[str, object], ...]
    stream_transition_log: tuple[Mapping[str, object], ...]
    operational_decisions: tuple[Mapping[str, object], ...]
    evaluation_complete: bool

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_summary",
            "reconciliation_summary",
            "reconciliation_diagnostics",
            "position_cash_reconciliation",
            "stream_transition_log",
            "operational_decisions",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(
                    MappingProxyType(dict(row))
                    for row in getattr(self, field_name)
                ),
            )


@dataclass(frozen=True, slots=True)
class _ScenarioOutcome:
    scenario_id: str
    reconciliation: ReconciliationResult
    monitor: StreamHealthMonitor
    decision: OperationalDecision


def _client_id(order_number: int) -> str:
    return f"axiom-day20-order-{order_number:04d}"


def _broker_id(order_number: int) -> str:
    return f"broker-day20-order-{order_number:04d}"


def _local_state(
    *,
    order_number: int = 1,
    status: OrderStatus,
    requested: str = "10",
    filled: str = "0",
    average: str | None = None,
    side: str = "buy",
    symbol: str = "SPY",
    recovery_required: bool = False,
) -> OrderState:
    client_order_id = _client_id(order_number)
    machine = OrderStateMachine(f"day20_local_{order_number}")
    machine.register_intent(
        OrderIntent(
            client_order_id=client_order_id,
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
            event_id=f"event-day20-local-{order_number:04d}",
            provider_sequence=1,
            client_order_id=client_order_id,
            broker_order_id=_broker_id(order_number),
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
    state = machine.states[client_order_id]
    return (
        replace(state, recovery_required=True)
        if recovery_required
        else state
    )


def _broker_order(
    local: OrderState,
    *,
    status: OrderStatus | None = None,
    filled: str | None = None,
    average: str | None | object = ...,
    broker_order_id: str | None = None,
) -> BrokerOrderSnapshot:
    normalized_filled = (
        local.filled_quantity if filled is None else Decimal(filled)
    )
    normalized_average = (
        local.filled_average_price
        if average is ...
        else None if average is None else Decimal(str(average))
    )
    return BrokerOrderSnapshot(
        client_order_id=local.client_order_id,
        broker_order_id=broker_order_id or local.broker_order_id or "",
        symbol=local.symbol,
        side=local.side,
        status=status or local.status,
        requested_quantity=local.requested_quantity,
        filled_quantity=normalized_filled,
        filled_average_price=normalized_average,
        snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
    )


def _fill(
    local: OrderState,
    *,
    fill_number: int = 1,
    quantity: str,
    price: str,
) -> BrokerFillSnapshot:
    return BrokerFillSnapshot(
        fill_id=f"fill-day20-{fill_number:04d}",
        client_order_id=local.client_order_id,
        broker_order_id=local.broker_order_id or "",
        symbol=local.symbol,
        side=local.side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        executed_at=BASE_TIMESTAMP + timedelta(seconds=3 + fill_number),
    )


def _input(
    *,
    local_states: tuple[OrderState, ...] = (),
    broker_orders: tuple[BrokerOrderSnapshot, ...] = (),
    broker_fills: tuple[BrokerFillSnapshot, ...] = (),
    position_quantity: str = "0",
    position_price: str = "100",
    opening_position: str = "0",
    broker_cash: str = "100000",
) -> ReconciliationInput:
    return ReconciliationInput(
        local_states=local_states,
        broker_orders=broker_orders,
        broker_fills=broker_fills,
        broker_positions=(
            BrokerPositionSnapshot(
                symbol="SPY",
                quantity=Decimal(position_quantity),
                mark_price=Decimal(position_price),
                snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(
            OpeningPosition(symbol="SPY", quantity=Decimal(opening_position)),
        ),
        opening_cash=OPENING_CASH,
        broker_account=BrokerAccountSnapshot(
            cash=Decimal(broker_cash),
            snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
        ),
        as_of=BASE_TIMESTAMP + timedelta(seconds=10),
    )


def _healthy_monitor(scenario_id: str) -> StreamHealthMonitor:
    monitor = StreamHealthMonitor(
        f"stream-{scenario_id}", started_at=BASE_TIMESTAMP
    )
    monitor.record_message(
        event_id=f"stream-event-{scenario_id}-1",
        event_at=BASE_TIMESTAMP + timedelta(seconds=2),
        received_at=BASE_TIMESTAMP + timedelta(seconds=3),
    )
    return monitor


def _outcome(
    scenario_id: str,
    reconciliation_input: ReconciliationInput,
    *,
    monitor: StreamHealthMonitor | None = None,
) -> _ScenarioOutcome:
    reconciliation = reconcile_snapshot(reconciliation_input)
    normalized_monitor = monitor or _healthy_monitor(scenario_id)
    return _ScenarioOutcome(
        scenario_id=scenario_id,
        reconciliation=reconciliation,
        monitor=normalized_monitor,
        decision=evaluate_operational_gate(
            reconciliation, normalized_monitor
        ),
    )


def _fully_reconciled(
    scenario_id: str,
    *,
    recovery_required: bool = False,
) -> _ScenarioOutcome:
    local = _local_state(
        status=OrderStatus.FILLED,
        filled="10",
        average="100",
        recovery_required=recovery_required,
    )
    return _outcome(
        scenario_id,
        _input(
            local_states=(local,),
            broker_orders=(_broker_order(local),),
            broker_fills=(_fill(local, quantity="10", price="100"),),
            position_quantity="10",
            broker_cash="99000",
        ),
    )


def _run_scenarios() -> tuple[_ScenarioOutcome, ...]:
    complete = _fully_reconciled("fully_reconciled")

    partial_local = _local_state(
        status=OrderStatus.PARTIALLY_FILLED,
        filled="3",
        average="100",
    )
    partial = _outcome(
        "partial_fill_reconciled",
        _input(
            local_states=(partial_local,),
            broker_orders=(_broker_order(partial_local),),
            broker_fills=(
                _fill(partial_local, quantity="3", price="100"),
            ),
            position_quantity="3",
            broker_cash="99700",
        ),
    )

    missed_local = _local_state(status=OrderStatus.NEW)
    missed = _outcome(
        "missed_stream_update",
        _input(
            local_states=(missed_local,),
            broker_orders=(
                _broker_order(
                    missed_local,
                    status=OrderStatus.PARTIALLY_FILLED,
                    filled="3",
                    average="100",
                ),
            ),
            broker_fills=(
                _fill(missed_local, quantity="3", price="100"),
            ),
            position_quantity="3",
            broker_cash="99700",
        ),
    )

    orphan_local = _local_state(order_number=2, status=OrderStatus.NEW)
    orphan = _outcome(
        "orphan_broker_order",
        _input(broker_orders=(_broker_order(orphan_local),)),
    )

    position_local = _local_state(
        order_number=3,
        status=OrderStatus.FILLED,
        filled="10",
        average="100",
    )
    position = _outcome(
        "position_mismatch",
        _input(
            local_states=(position_local,),
            broker_orders=(_broker_order(position_local),),
            broker_fills=(
                _fill(
                    position_local,
                    fill_number=3,
                    quantity="10",
                    price="100",
                ),
            ),
            position_quantity="9",
            broker_cash="99000",
        ),
    )

    cash_local = _local_state(
        order_number=4,
        status=OrderStatus.FILLED,
        filled="10",
        average="100",
    )
    cash = _outcome(
        "cash_mismatch",
        _input(
            local_states=(cash_local,),
            broker_orders=(_broker_order(cash_local),),
            broker_fills=(
                _fill(
                    cash_local,
                    fill_number=4,
                    quantity="10",
                    price="100",
                ),
            ),
            position_quantity="10",
            broker_cash="99001",
        ),
    )

    recovery = _fully_reconciled(
        "local_recovery_required", recovery_required=True
    )

    recovered_monitor = _healthy_monitor("stale_stream_recovered")
    stale_at = BASE_TIMESTAMP + timedelta(seconds=34)
    recovered_monitor.evaluate(as_of=stale_at)
    recovered_monitor.begin_reconnect(at=stale_at)
    recovered_monitor.record_reconnect_failure(
        at=stale_at + timedelta(seconds=1)
    )
    recovered_monitor.record_reconnect_success(
        at=stale_at + timedelta(seconds=3)
    )
    recovered_monitor.record_message(
        event_id="stream-event-stale_stream_recovered-2",
        event_at=stale_at + timedelta(seconds=3),
        received_at=stale_at + timedelta(seconds=3),
    )
    recovered_base = _fully_reconciled("stale_stream_recovered")
    stale_recovered = replace(
        recovered_base,
        monitor=recovered_monitor,
        decision=evaluate_operational_gate(
            recovered_base.reconciliation, recovered_monitor
        ),
    )

    exhausted_monitor = _healthy_monitor("reconnect_exhausted")
    exhausted_stale_at = BASE_TIMESTAMP + timedelta(seconds=34)
    exhausted_monitor.evaluate(as_of=exhausted_stale_at)
    exhausted_monitor.begin_reconnect(at=exhausted_stale_at)
    exhausted_monitor.record_reconnect_failure(
        at=exhausted_stale_at + timedelta(seconds=1)
    )
    exhausted_monitor.record_reconnect_failure(
        at=exhausted_stale_at + timedelta(seconds=3)
    )
    exhausted_monitor.record_reconnect_failure(
        at=exhausted_stale_at + timedelta(seconds=7)
    )
    exhausted_base = _fully_reconciled("reconnect_exhausted")
    exhausted = replace(
        exhausted_base,
        monitor=exhausted_monitor,
        decision=evaluate_operational_gate(
            exhausted_base.reconciliation, exhausted_monitor
        ),
    )

    large_local = _local_state(
        order_number=5,
        status=OrderStatus.NEW,
        requested="30",
    )
    single_limit = _outcome(
        "single_order_limit_breach",
        _input(
            local_states=(large_local,),
            broker_orders=(_broker_order(large_local),),
        ),
    )

    gross = _outcome(
        "gross_exposure_breach",
        _input(
            position_quantity="80",
            position_price="400",
            opening_position="80",
        ),
    )

    killed_base = _fully_reconciled("kill_switch_latched")
    killed_monitor = killed_base.monitor
    killed_monitor.engage_kill_switch(
        at=BASE_TIMESTAMP + timedelta(seconds=4)
    )
    killed = replace(
        killed_base,
        decision=evaluate_operational_gate(
            killed_base.reconciliation, killed_monitor
        ),
    )

    outcomes = (
        complete,
        partial,
        missed,
        orphan,
        position,
        cash,
        recovery,
        stale_recovered,
        exhausted,
        single_limit,
        gross,
        killed,
    )
    if tuple(row.scenario_id for row in outcomes) != SCENARIO_ORDER:
        raise RuntimeError("Day 20 scenario order changed.")
    return outcomes


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _balance_row(
    scenario_order: int,
    scenario_id: str,
    row: BalanceComparison,
) -> dict[str, object]:
    return {
        "scenario_order": scenario_order,
        "scenario_id": scenario_id,
        "balance_type": row.balance_type,
        "symbol": row.symbol,
        "opening_value": str(row.opening_value),
        "signed_fill_flow": str(row.signed_fill_flow),
        "expected_value": str(row.expected_value),
        "broker_value": str(row.broker_value),
        "difference": str(row.difference),
        "tolerance": str(row.tolerance),
        "reconciled": row.reconciled,
    }


def run_day20_scenarios() -> Day20ScenarioResults:
    """Run and validate all frozen Day 20 synthetic scenarios."""

    outcomes = _run_scenarios()
    expected = {
        "fully_reconciled": (True, ()),
        "partial_fill_reconciled": (True, ()),
        "missed_stream_update": (
            False,
            (
                "order_status_mismatch",
                "cumulative_fill_mismatch",
                "weighted_average_price_mismatch",
                "fill_quantity_mismatch",
            ),
        ),
        "orphan_broker_order": (False, ("unknown_broker_order",)),
        "position_mismatch": (False, ("position_quantity_mismatch",)),
        "cash_mismatch": (False, ("cash_balance_mismatch",)),
        "local_recovery_required": (False, ("local_recovery_required",)),
        "stale_stream_recovered": (True, ()),
        "reconnect_exhausted": (
            False,
            ("stream_stale", "reconnect_exhausted"),
        ),
        "single_order_limit_breach": (
            False,
            ("single_order_limit_exceeded",),
        ),
        "gross_exposure_breach": (
            False,
            ("gross_notional_limit_exceeded",),
        ),
        "kill_switch_latched": (False, ("kill_switch_latched",)),
    }
    scenario_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for scenario_order, outcome in enumerate(outcomes, start=1):
        expected_gate, expected_reasons = expected[outcome.scenario_id]
        observed_reasons = outcome.decision.active_reason_codes
        scenario_passed = (
            outcome.decision.operational_gate_passed is expected_gate
            and observed_reasons == expected_reasons
            and outcome.decision.day20_order_submission_authorized is False
            and outcome.decision.can_submit_orders is False
        )
        if not scenario_passed:
            raise RuntimeError(
                f"Day 20 scenario failed: {outcome.scenario_id}; "
                f"observed={observed_reasons}."
            )
        scenario_rows.append(
            {
                "scenario_order": scenario_order,
                "scenario_id": outcome.scenario_id,
                "expected_operational_gate_passed": expected_gate,
                "observed_operational_gate_passed": (
                    outcome.decision.operational_gate_passed
                ),
                "expected_reason_codes": "|".join(expected_reasons),
                "observed_reason_codes": "|".join(observed_reasons),
                "reconciliation_diagnostic_count": len(
                    outcome.reconciliation.diagnostics
                ),
                "stream_audit_count": len(outcome.monitor.audit_entries),
                "stream_final_state": outcome.monitor.state.value,
                "day20_order_submission_authorized": False,
                "can_submit_orders": False,
                "scenario_passed": True,
            }
        )
        result = outcome.reconciliation
        reconciliation_rows.append(
            {
                "scenario_order": scenario_order,
                "scenario_id": outcome.scenario_id,
                "local_order_count": result.local_order_count,
                "broker_order_count": result.broker_order_count,
                "unique_fill_count": result.unique_fill_count,
                "exact_duplicate_fill_count": (
                    result.exact_duplicate_fill_count
                ),
                "broker_position_count": result.broker_position_count,
                "expected_cash": str(result.expected_cash),
                "broker_cash": str(result.broker_cash),
                "gross_notional": str(result.gross_notional),
                "open_order_count": result.open_order_count,
                "core_reconciliation_passed": (
                    result.core_reconciliation_passed
                ),
                "limits_passed": result.limits_passed,
                "reconciliation_passed": result.reconciliation_passed,
            }
        )
        diagnostic_rows.extend(
            {
                "scenario_order": scenario_order,
                "scenario_id": outcome.scenario_id,
                "diagnostic_sequence": row.diagnostic_sequence,
                "category": row.category,
                "reason_code": row.reason_code,
                "client_order_id": row.client_order_id,
                "broker_order_id": row.broker_order_id,
                "symbol": row.symbol,
                "local_value": row.local_value,
                "broker_value": row.broker_value,
                "required_action": row.required_action,
            }
            for row in result.diagnostics
        )
        balance_rows.extend(
            _balance_row(scenario_order, outcome.scenario_id, row)
            for row in result.balance_comparisons
        )
        stream_rows.extend(
            {
                "scenario_order": scenario_order,
                "scenario_id": outcome.scenario_id,
                "audit_sequence": row.audit_sequence,
                "stream_id": row.stream_id,
                "action": row.action,
                "previous_state": row.previous_state,
                "resulting_state": row.resulting_state,
                "event_id": row.event_id,
                "occurred_at": _iso(row.occurred_at),
                "reconnect_attempt": row.reconnect_attempt,
                "next_retry_at": _iso(row.next_retry_at),
                "reason_code": row.reason_code,
                "submission_blocked": row.submission_blocked,
            }
            for row in outcome.monitor.audit_entries
        )
        decision = outcome.decision
        decision_rows.append(
            {
                "scenario_order": scenario_order,
                "scenario_id": outcome.scenario_id,
                "reconciliation_passed": decision.reconciliation_passed,
                "limits_passed": decision.limits_passed,
                "stream_state": decision.stream_state.value,
                "stream_safe": decision.stream_safe,
                "circuit_breaker_open": decision.circuit_breaker_open,
                "kill_switch_latched": decision.kill_switch_latched,
                "active_reason_codes": "|".join(
                    decision.active_reason_codes
                ),
                "operational_gate_passed": decision.operational_gate_passed,
                "day20_order_submission_authorized": False,
                "can_submit_orders": False,
            }
        )

    return Day20ScenarioResults(
        scenario_summary=tuple(scenario_rows),
        reconciliation_summary=tuple(reconciliation_rows),
        reconciliation_diagnostics=tuple(diagnostic_rows),
        position_cash_reconciliation=tuple(balance_rows),
        stream_transition_log=tuple(stream_rows),
        operational_decisions=tuple(decision_rows),
        evaluation_complete=all(row["scenario_passed"] for row in scenario_rows),
    )
