from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from systematic_alpha.broker.order_state import OrderStatus
from systematic_alpha.broker.reconciliation import (
    CASH_TOLERANCE,
    MAX_ABSOLUTE_SYMBOL_POSITION,
    MAX_GROSS_NOTIONAL,
    MAX_OPEN_ORDERS,
    MAX_SINGLE_ORDER_QUANTITY,
    PRICE_TOLERANCE,
    RECONCILIATION_REASON_CODES,
    SNAPSHOT_STALE_AFTER,
    BrokerAccountSnapshot,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    OpeningPosition,
    ReconciliationInput,
    freeze_reason_mapping,
    reconcile_snapshot,
)
from tests.day20_fixtures import (
    AS_OF,
    BASE_TIMESTAMP,
    make_broker_order,
    make_fill,
    make_local_state,
    matching_input,
)


EXPECTED_REASON_CODES = (
    "local_recovery_required",
    "local_order_missing_at_broker",
    "unknown_broker_order",
    "broker_order_id_mismatch",
    "duplicate_broker_order_id",
    "order_status_mismatch",
    "requested_quantity_mismatch",
    "cumulative_fill_mismatch",
    "weighted_average_price_mismatch",
    "duplicate_fill_conflict",
    "unknown_order_fill",
    "fill_broker_order_id_mismatch",
    "fill_symbol_side_mismatch",
    "fill_quantity_mismatch",
    "broker_position_missing",
    "unexpected_broker_position",
    "position_quantity_mismatch",
    "cash_balance_mismatch",
    "snapshot_stale",
    "single_order_limit_exceeded",
    "symbol_position_limit_exceeded",
    "gross_notional_limit_exceeded",
    "open_order_limit_exceeded",
)


def _reasons(snapshot: ReconciliationInput) -> tuple[str, ...]:
    return reconcile_snapshot(snapshot).active_reason_codes


def _flat_input(
    *,
    broker_positions: tuple[BrokerPositionSnapshot, ...],
    opening_positions: tuple[OpeningPosition, ...],
    broker_cash: str = "100000",
) -> ReconciliationInput:
    return ReconciliationInput(
        local_states=(),
        broker_orders=(),
        broker_fills=(),
        broker_positions=broker_positions,
        opening_positions=opening_positions,
        opening_cash=Decimal("100000"),
        broker_account=BrokerAccountSnapshot(
            cash=Decimal(broker_cash),
            snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
        ),
        as_of=AS_OF,
    )


def _open_orders_input(count: int, *, requested: str = "10") -> ReconciliationInput:
    locals_ = tuple(
        make_local_state(
            order_number=number,
            status=OrderStatus.NEW,
            requested=requested,
            filled="0",
            average=None,
        )
        for number in range(1, count + 1)
    )
    return ReconciliationInput(
        local_states=locals_,
        broker_orders=tuple(make_broker_order(row) for row in locals_),
        broker_fills=(),
        broker_positions=(
            BrokerPositionSnapshot(
                symbol="SPY",
                quantity=Decimal("0"),
                mark_price=Decimal("100"),
                snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("0")),),
        opening_cash=Decimal("100000"),
        broker_account=BrokerAccountSnapshot(
            cash=Decimal("100000"),
            snapshot_at=BASE_TIMESTAMP + timedelta(seconds=5),
        ),
        as_of=AS_OF,
    )


def test_frozen_constants_and_reason_vocabulary() -> None:
    assert RECONCILIATION_REASON_CODES == EXPECTED_REASON_CODES
    assert SNAPSHOT_STALE_AFTER == timedelta(seconds=30)
    assert CASH_TOLERANCE == Decimal("0.01")
    assert PRICE_TOLERANCE == Decimal("0.0001")
    assert MAX_SINGLE_ORDER_QUANTITY == Decimal("25")
    assert MAX_ABSOLUTE_SYMBOL_POSITION == Decimal("100")
    assert MAX_GROSS_NOTIONAL == Decimal("25000")
    assert MAX_OPEN_ORDERS == 5


def test_fully_matched_snapshot_reconciles_exactly() -> None:
    result = reconcile_snapshot(matching_input())
    assert result.reconciliation_passed is True
    assert result.core_reconciliation_passed is True
    assert result.limits_passed is True
    assert result.active_reason_codes == ()
    assert result.expected_cash == Decimal("99000")
    assert result.gross_notional == Decimal("1000")
    assert len(result.balance_comparisons) == 2
    assert all(row.reconciled for row in result.balance_comparisons)


def test_exact_duplicate_fill_is_counted_once() -> None:
    local = make_local_state()
    fill = make_fill(local)
    result = reconcile_snapshot(
        matching_input(
            local=local,
            broker_fills=(fill, fill),
        )
    )
    assert result.reconciliation_passed is True
    assert result.unique_fill_count == 1
    assert result.exact_duplicate_fill_count == 1


def test_conflicting_duplicate_fill_blocks_without_double_counting() -> None:
    local = make_local_state()
    fill = make_fill(local)
    conflicting = replace(fill, price=Decimal("101"))
    result = reconcile_snapshot(
        matching_input(local=local, broker_fills=(fill, conflicting))
    )
    assert result.active_reason_codes == ("duplicate_fill_conflict",)
    assert result.unique_fill_count == 1
    assert result.expected_cash == Decimal("99000")


def test_local_recovery_flag_blocks_reconciliation() -> None:
    local = make_local_state(recovery_required=True)
    assert _reasons(matching_input(local=local)) == (
        "local_recovery_required",
    )


def test_local_order_missing_at_broker_blocks() -> None:
    assert _reasons(matching_input(broker_orders=())) == (
        "local_order_missing_at_broker",
    )


def test_unknown_broker_order_blocks() -> None:
    local = make_local_state()
    snapshot = ReconciliationInput(
        local_states=(),
        broker_orders=(make_broker_order(local),),
        broker_fills=(),
        broker_positions=(
            BrokerPositionSnapshot(
                "SPY",
                Decimal("0"),
                Decimal("100"),
                BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("0")),),
        opening_cash=Decimal("100000"),
        broker_account=BrokerAccountSnapshot(
            Decimal("100000"), BASE_TIMESTAMP + timedelta(seconds=5)
        ),
        as_of=AS_OF,
    )
    assert _reasons(snapshot) == ("unknown_broker_order",)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"broker_order_id": "broker-other"}, "broker_order_id_mismatch"),
        ({"status": OrderStatus.CANCELED}, "order_status_mismatch"),
        ({"requested_quantity": Decimal("11")}, "requested_quantity_mismatch"),
        ({"filled_quantity": Decimal("9")}, "cumulative_fill_mismatch"),
        ({"filled_average_price": Decimal("101")}, "weighted_average_price_mismatch"),
        ({"symbol": "QQQ"}, "fill_symbol_side_mismatch"),
    ),
)
def test_broker_order_mismatch_reason_is_explicit(
    overrides: dict[str, object],
    reason: str,
) -> None:
    local = make_local_state()
    order = make_broker_order(local, **overrides)
    assert reason in _reasons(
        matching_input(local=local, broker_orders=(order,))
    )


def test_duplicate_broker_order_id_is_explicit() -> None:
    first = make_local_state(
        order_number=1,
        status=OrderStatus.NEW,
        filled="0",
        average=None,
    )
    second = make_local_state(
        order_number=2,
        status=OrderStatus.NEW,
        filled="0",
        average=None,
    )
    orders = (
        make_broker_order(first),
        make_broker_order(second, broker_order_id=first.broker_order_id),
    )
    snapshot = _open_orders_input(2)
    snapshot = replace(snapshot, broker_orders=orders)
    assert "duplicate_broker_order_id" in _reasons(snapshot)


def test_unknown_order_fill_is_explicit() -> None:
    local = make_local_state()
    fill = make_fill(local)
    snapshot = ReconciliationInput(
        local_states=(),
        broker_orders=(),
        broker_fills=(fill,),
        broker_positions=(
            BrokerPositionSnapshot(
                "SPY",
                Decimal("10"),
                Decimal("100"),
                BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("0")),),
        opening_cash=Decimal("100000"),
        broker_account=BrokerAccountSnapshot(
            Decimal("99000"), BASE_TIMESTAMP + timedelta(seconds=5)
        ),
        as_of=AS_OF,
    )
    assert _reasons(snapshot) == ("unknown_order_fill",)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"broker_order_id": "broker-other"}, "fill_broker_order_id_mismatch"),
        ({"side": "sell"}, "fill_symbol_side_mismatch"),
    ),
)
def test_fill_identity_mismatches_are_explicit(
    overrides: dict[str, object],
    reason: str,
) -> None:
    local = make_local_state()
    fill = make_fill(local, **overrides)
    assert reason in _reasons(
        matching_input(local=local, broker_fills=(fill,))
    )


def test_fill_quantity_mismatch_is_explicit_without_balance_noise() -> None:
    local = make_local_state()
    fill = make_fill(local, quantity="9")
    positions = (
        BrokerPositionSnapshot(
            "SPY",
            Decimal("9"),
            Decimal("100"),
            BASE_TIMESTAMP + timedelta(seconds=5),
        ),
    )
    reasons = _reasons(
        matching_input(
            local=local,
            broker_fills=(fill,),
            broker_positions=positions,
            broker_cash="99100",
        )
    )
    assert reasons == ("fill_quantity_mismatch",)


def test_weighted_fill_price_tolerance_and_mismatch() -> None:
    local = make_local_state()
    within = make_fill(local, price="100.0001")
    within_result = reconcile_snapshot(
        matching_input(
            local=local,
            broker_fills=(within,),
            broker_cash="98999.999",
        )
    )
    assert within_result.reconciliation_passed is True

    outside = make_fill(local, price="100.0002")
    reasons = _reasons(
        matching_input(
            local=local,
            broker_fills=(outside,),
            broker_cash="98999.998",
        )
    )
    assert reasons == ("weighted_average_price_mismatch",)


def test_missing_unexpected_and_wrong_position_reasons() -> None:
    assert "broker_position_missing" in _reasons(
        matching_input(broker_positions=())
    )
    assert "position_quantity_mismatch" in _reasons(
        matching_input(
            broker_positions=(
                BrokerPositionSnapshot(
                    "SPY",
                    Decimal("9"),
                    Decimal("100"),
                    BASE_TIMESTAMP + timedelta(seconds=5),
                ),
            )
        )
    )
    unexpected = _flat_input(
        broker_positions=(
            BrokerPositionSnapshot(
                "QQQ",
                Decimal("1"),
                Decimal("400"),
                BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(),
    )
    assert _reasons(unexpected) == ("unexpected_broker_position",)


def test_cash_tolerance_boundary() -> None:
    assert "cash_balance_mismatch" not in _reasons(
        matching_input(broker_cash="98999.99")
    )
    assert "cash_balance_mismatch" in _reasons(
        matching_input(broker_cash="98999.989")
    )


def test_snapshot_staleness_boundary() -> None:
    local = make_local_state()
    boundary_time = AS_OF - SNAPSHOT_STALE_AFTER
    boundary = make_broker_order(local, snapshot_at=boundary_time)
    assert "snapshot_stale" not in _reasons(
        matching_input(local=local, broker_orders=(boundary,))
    )
    stale = replace(boundary, snapshot_at=boundary_time - timedelta(seconds=1))
    assert "snapshot_stale" in _reasons(
        matching_input(local=local, broker_orders=(stale,))
    )


def test_single_order_limit_boundary() -> None:
    at_limit = _open_orders_input(1, requested="25")
    assert "single_order_limit_exceeded" not in _reasons(at_limit)
    above = _open_orders_input(1, requested="25.0001")
    assert "single_order_limit_exceeded" in _reasons(above)


def test_open_order_limit_boundary() -> None:
    assert "open_order_limit_exceeded" not in _reasons(
        _open_orders_input(5)
    )
    assert "open_order_limit_exceeded" in _reasons(
        _open_orders_input(6)
    )


def test_symbol_position_limit_boundary() -> None:
    at_limit = _flat_input(
        broker_positions=(
            BrokerPositionSnapshot(
                "SPY",
                Decimal("100"),
                Decimal("100"),
                BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("100")),),
    )
    assert "symbol_position_limit_exceeded" not in _reasons(at_limit)
    above = replace(
        at_limit,
        broker_positions=(
            replace(
                at_limit.broker_positions[0], quantity=Decimal("100.0001")
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("100.0001")),),
    )
    assert "symbol_position_limit_exceeded" in _reasons(above)


def test_gross_notional_limit_boundary() -> None:
    at_limit = _flat_input(
        broker_positions=(
            BrokerPositionSnapshot(
                "SPY",
                Decimal("100"),
                Decimal("250"),
                BASE_TIMESTAMP + timedelta(seconds=5),
            ),
        ),
        opening_positions=(OpeningPosition("SPY", Decimal("100")),),
    )
    assert "gross_notional_limit_exceeded" not in _reasons(at_limit)
    above = replace(
        at_limit,
        broker_positions=(
            replace(
                at_limit.broker_positions[0], mark_price=Decimal("250.01")
            ),
        ),
    )
    assert "gross_notional_limit_exceeded" in _reasons(above)


def test_reason_count_mapping_is_frozen_and_immutable() -> None:
    local = make_local_state(recovery_required=True)
    result = reconcile_snapshot(matching_input(local=local, broker_orders=()))
    counts = freeze_reason_mapping(result.diagnostics)
    assert tuple(counts) == RECONCILIATION_REASON_CODES
    assert counts["local_recovery_required"] == 1
    assert counts["local_order_missing_at_broker"] == 1
    with pytest.raises(TypeError):
        counts["local_recovery_required"] = 0  # type: ignore[index]


@pytest.mark.parametrize(
    "builder",
    (
        lambda: BrokerOrderSnapshot(
            "bad",
            "broker",
            "SPY",
            "buy",
            OrderStatus.NEW,
            Decimal("10"),
            Decimal("0"),
            None,
            BASE_TIMESTAMP,
        ),
        lambda: BrokerFillSnapshot(
            "fill id",
            make_local_state().client_order_id,
            "broker",
            "SPY",
            "buy",
            Decimal("1"),
            Decimal("100"),
            BASE_TIMESTAMP,
        ),
        lambda: BrokerPositionSnapshot(
            "SPY1", Decimal("1"), Decimal("100"), BASE_TIMESTAMP
        ),
        lambda: BrokerAccountSnapshot(Decimal("NaN"), BASE_TIMESTAMP),
    ),
)
def test_invalid_snapshot_values_fail_closed(builder) -> None:
    with pytest.raises(ValueError):
        builder()


def test_future_snapshot_and_duplicate_keys_fail_closed() -> None:
    valid = matching_input()
    with pytest.raises(ValueError, match="future"):
        replace(
            valid,
            broker_account=replace(
                valid.broker_account,
                snapshot_at=AS_OF + timedelta(seconds=1),
            ),
        )
    with pytest.raises(ValueError, match="Duplicate broker position"):
        replace(
            valid,
            broker_positions=(
                valid.broker_positions[0],
                valid.broker_positions[0],
            ),
        )


def test_reconcile_rejects_wrong_public_type() -> None:
    with pytest.raises(TypeError):
        reconcile_snapshot(object())  # type: ignore[arg-type]

