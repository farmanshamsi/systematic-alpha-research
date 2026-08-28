from __future__ import annotations

from datetime import timedelta

import pytest

from systematic_alpha.broker.monitoring import (
    MAX_RECONNECT_ATTEMPTS,
    MONITOR_REASON_CODES,
    RECONNECT_BACKOFF_SECONDS,
    STREAM_STALE_AFTER,
    StreamHealthMonitor,
    StreamState,
    evaluate_operational_gate,
)
from systematic_alpha.broker.reconciliation import reconcile_snapshot
from tests.day20_fixtures import BASE_TIMESTAMP, matching_input


def _healthy_monitor(stream_id: str = "stream-test") -> StreamHealthMonitor:
    monitor = StreamHealthMonitor(stream_id, started_at=BASE_TIMESTAMP)
    monitor.record_message(
        event_id="event-1",
        event_at=BASE_TIMESTAMP + timedelta(seconds=1),
        received_at=BASE_TIMESTAMP + timedelta(seconds=2),
    )
    return monitor


def test_frozen_monitor_constants_and_initial_fail_closed_state() -> None:
    monitor = StreamHealthMonitor("stream-test", started_at=BASE_TIMESTAMP)
    assert STREAM_STALE_AFTER == timedelta(seconds=30)
    assert RECONNECT_BACKOFF_SECONDS == (1, 2, 4)
    assert MAX_RECONNECT_ATTEMPTS == 3
    assert MONITOR_REASON_CODES == (
        "stream_stale",
        "reconnect_exhausted",
        "kill_switch_latched",
    )
    assert monitor.state is StreamState.STALE
    assert monitor.active_reason_codes == ("stream_stale",)


def test_new_message_makes_stream_healthy_but_audit_remains_blocked() -> None:
    monitor = _healthy_monitor()
    assert monitor.state is StreamState.HEALTHY
    assert monitor.active_reason_codes == ()
    assert monitor.reconnect_attempts == 0
    assert monitor.audit_entries[-1].action == "message_received"
    assert monitor.audit_entries[-1].submission_blocked is True


def test_exact_duplicate_message_is_idempotent() -> None:
    monitor = _healthy_monitor()
    before = monitor.audit_entries
    duplicate = monitor.record_message(
        event_id="event-1",
        event_at=BASE_TIMESTAMP + timedelta(seconds=1),
        received_at=BASE_TIMESTAMP + timedelta(seconds=5),
    )
    assert duplicate is None
    assert monitor.audit_entries == before
    assert monitor.last_received_at == BASE_TIMESTAMP + timedelta(seconds=2)


def test_conflicting_event_id_reuse_fails_closed() -> None:
    monitor = _healthy_monitor()
    with pytest.raises(ValueError, match="Conflicting"):
        monitor.record_message(
            event_id="event-1",
            event_at=BASE_TIMESTAMP + timedelta(seconds=2),
            received_at=BASE_TIMESTAMP + timedelta(seconds=3),
        )


def test_stale_boundary_and_repeated_check_are_idempotent() -> None:
    monitor = _healthy_monitor()
    reference = monitor.last_received_at
    assert reference is not None
    assert monitor.evaluate(as_of=reference + STREAM_STALE_AFTER) is None
    created = monitor.evaluate(
        as_of=reference + STREAM_STALE_AFTER + timedelta(seconds=1)
    )
    assert created is not None
    assert created.reason_code == "stream_stale"
    assert monitor.state is StreamState.STALE
    count = len(monitor.audit_entries)
    assert monitor.evaluate(
        as_of=reference + STREAM_STALE_AFTER + timedelta(seconds=2)
    ) is None
    assert len(monitor.audit_entries) == count


def test_reconnect_schedule_is_exactly_one_two_four_then_circuit() -> None:
    monitor = _healthy_monitor()
    reference = monitor.last_received_at
    assert reference is not None
    stale_at = reference + STREAM_STALE_AFTER + timedelta(seconds=1)
    monitor.evaluate(as_of=stale_at)
    scheduled = monitor.begin_reconnect(at=stale_at)
    assert scheduled.next_retry_at == stale_at + timedelta(seconds=1)

    first_at = scheduled.next_retry_at
    assert first_at is not None
    first = monitor.record_reconnect_failure(at=first_at)
    assert first.reconnect_attempt == 1
    assert first.next_retry_at == first_at + timedelta(seconds=2)

    second_at = first.next_retry_at
    assert second_at is not None
    second = monitor.record_reconnect_failure(at=second_at)
    assert second.reconnect_attempt == 2
    assert second.next_retry_at == second_at + timedelta(seconds=4)

    third_at = second.next_retry_at
    assert third_at is not None
    third = monitor.record_reconnect_failure(at=third_at)
    assert third.reconnect_attempt == 3
    assert third.next_retry_at is None
    assert monitor.state is StreamState.CIRCUIT_OPEN
    assert monitor.active_reason_codes == (
        "stream_stale",
        "reconnect_exhausted",
    )


def test_early_reconnect_attempt_fails_closed() -> None:
    monitor = StreamHealthMonitor("stream-early", started_at=BASE_TIMESTAMP)
    scheduled = monitor.begin_reconnect(at=BASE_TIMESTAMP)
    assert scheduled.next_retry_at is not None
    with pytest.raises(ValueError, match="before scheduled"):
        monitor.record_reconnect_failure(at=BASE_TIMESTAMP)


def test_successful_reconnect_waits_for_new_message() -> None:
    monitor = _healthy_monitor()
    reference = monitor.last_received_at
    assert reference is not None
    stale_at = reference + STREAM_STALE_AFTER + timedelta(seconds=1)
    monitor.evaluate(as_of=stale_at)
    scheduled = monitor.begin_reconnect(at=stale_at)
    retry_at = scheduled.next_retry_at
    assert retry_at is not None
    success = monitor.record_reconnect_success(at=retry_at)
    assert success.resulting_state == "stale"
    assert monitor.active_reason_codes == ("stream_stale",)
    monitor.record_message(
        event_id="event-2",
        event_at=retry_at,
        received_at=retry_at,
    )
    assert monitor.state is StreamState.HEALTHY
    assert monitor.active_reason_codes == ()


def test_kill_switch_is_latched_and_idempotent() -> None:
    monitor = _healthy_monitor()
    created = monitor.engage_kill_switch(
        at=BASE_TIMESTAMP + timedelta(seconds=3)
    )
    assert created is not None
    assert monitor.state is StreamState.KILLED
    assert monitor.active_reason_codes == ("kill_switch_latched",)
    count = len(monitor.audit_entries)
    assert monitor.engage_kill_switch(
        at=BASE_TIMESTAMP + timedelta(seconds=4)
    ) is None
    assert len(monitor.audit_entries) == count
    with pytest.raises(RuntimeError, match="terminal"):
        monitor.record_message(
            event_id="event-2",
            event_at=BASE_TIMESTAMP + timedelta(seconds=4),
            received_at=BASE_TIMESTAMP + timedelta(seconds=4),
        )


def test_circuit_open_is_terminal() -> None:
    monitor = StreamHealthMonitor("stream-circuit", started_at=BASE_TIMESTAMP)
    scheduled = monitor.begin_reconnect(at=BASE_TIMESTAMP)
    first_at = scheduled.next_retry_at
    assert first_at is not None
    first = monitor.record_reconnect_failure(at=first_at)
    second_at = first.next_retry_at
    assert second_at is not None
    second = monitor.record_reconnect_failure(at=second_at)
    third_at = second.next_retry_at
    assert third_at is not None
    monitor.record_reconnect_failure(at=third_at)
    with pytest.raises(RuntimeError, match="terminal"):
        monitor.begin_reconnect(at=third_at + timedelta(seconds=1))


@pytest.mark.parametrize(
    ("event_at", "received_at", "message"),
    (
        (
            BASE_TIMESTAMP + timedelta(seconds=8),
            BASE_TIMESTAMP + timedelta(seconds=2),
            "future-skew",
        ),
        (
            BASE_TIMESTAMP,
            BASE_TIMESTAMP + timedelta(seconds=3),
            "event time regressed",
        ),
    ),
)
def test_invalid_stream_times_fail_closed(
    event_at,
    received_at,
    message: str,
) -> None:
    monitor = _healthy_monitor()
    with pytest.raises(ValueError, match=message):
        monitor.record_message(
            event_id="event-2",
            event_at=event_at,
            received_at=received_at,
        )


def test_operational_gate_passes_synthetic_safety_but_never_authorizes() -> None:
    reconciliation = reconcile_snapshot(matching_input())
    monitor = _healthy_monitor()
    decision = evaluate_operational_gate(reconciliation, monitor)
    assert decision.operational_gate_passed is True
    assert decision.stream_safe is True
    assert decision.day20_order_submission_authorized is False
    assert decision.can_submit_orders is False


def test_operational_gate_combines_reconciliation_and_monitor_reasons() -> None:
    local = matching_input().local_states[0]
    reconciliation = reconcile_snapshot(
        matching_input(local=local, broker_orders=())
    )
    monitor = StreamHealthMonitor("stream-stale", started_at=BASE_TIMESTAMP)
    decision = evaluate_operational_gate(reconciliation, monitor)
    assert decision.active_reason_codes == (
        "local_order_missing_at_broker",
        "stream_stale",
    )
    assert decision.operational_gate_passed is False


def test_public_monitor_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        StreamHealthMonitor("bad stream", started_at=BASE_TIMESTAMP)
    monitor = _healthy_monitor()
    with pytest.raises(ValueError, match="cannot precede"):
        monitor.evaluate(as_of=BASE_TIMESTAMP)
    with pytest.raises(TypeError):
        evaluate_operational_gate(object(), monitor)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_operational_gate(
            reconcile_snapshot(matching_input()), object()  # type: ignore[arg-type]
        )
