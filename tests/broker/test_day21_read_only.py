from datetime import datetime, timezone
from decimal import Decimal

from systematic_alpha.broker.controlled_paper_execution import day21_client_order_ids
from systematic_alpha.broker.day21_read_only import (
    Day21ReadOnlyState,
    build_day21_read_only_result,
)
from systematic_alpha.broker.day21_scenarios import run_day21_synthetic_happy_path
from tests.broker.test_controlled_paper_execution import signal
from tests.day18_fixtures import passing_preflight_result


def test_read_only_probe_records_abort_without_order_events() -> None:
    snapshot = signal(
        computed_at=datetime(2026, 8, 2, 19, 54, tzinfo=timezone.utc),
        signal_fresh=False,
        position=0,
        raw_signal=0,
    )
    entry_id, _ = day21_client_order_ids(snapshot)
    assert entry_id.startswith("axiom-day21-spy-e-")
    state = Day21ReadOnlyState(
        preflight=passing_preflight_result(),
        open_spy_orders=(),
        spy_position=None,
        cash=Decimal("100000"),
        duplicate_signal_order=False,
    )
    result = build_day21_read_only_result(
        signal=snapshot,
        state=state,
        day20_gate_passed=True,
    )
    assert result.outcome == "aborted_before_submission"
    assert not result.order_submission_occurred
    assert not result.order_events
    assert result.shutdown_reconciled
    assert set(result.abort_reasons) >= {
        "market_open_window",
        "signal_available",
        "signal_nonzero",
    }


def test_synthetic_happy_path_has_exact_known_answer() -> None:
    result = run_day21_synthetic_happy_path()
    assert result.execution_complete
    assert result.shutdown_reconciled
    assert result.realized_round_trip_pnl == Decimal("0.0001")

