# Day 20 Synthetic Reconciliation and Monitoring Specification

## 1. Status and authority

- Specification version: `day20_reconciliation_monitoring_v1`
- Status: frozen before implementation
- Scope: deterministic synthetic broker reconciliation and operational gating
- Broker network access: prohibited
- Credential access: prohibited
- Order submission, replacement, cancellation, and position mutation: prohibited
- Canonical market-data access: prohibited
- Locked 2026 research data: prohibited

This document is the Day 20 implementation contract. Changes to identifiers,
numeric tolerances, snapshot semantics, diagnostic codes, monitoring states,
limits, scenario order, schemas, or artifact order require an explicit
specification revision.

## 2. Objective

Build a broker-neutral, restart-safe reconciliation layer that compares Day 19
local order state with independent synthetic broker order, fill, position, and
cash snapshots. Combine the result with deterministic stream-health monitoring,
bounded reconnect/backoff, exposure limits, a circuit breaker, and a latched
kill switch.

Day 20 is diagnostic and fail-closed. It never repairs broker state, invents a
fill, cancels an order, submits an order, or clears a recovery flag.

## 3. Frozen timestamps and numeric rules

All canonical scenarios use fixed timezone-aware UTC timestamps in December
2025. The system clock is never called. Every evaluation receives an explicit
`as_of` timestamp.

- Quantities, prices, cash, and notionals use finite `decimal.Decimal` values.
- Order and fill quantities and prices are strictly positive where present.
- Position quantities and cash may be zero and use signed decimal values.
- Broker order, position, and account snapshots must not be in the future and
  are stale more than 30 seconds after `snapshot_at`.
- Fill execution timestamps may be historical but must not be in the future.
- Cash comparisons use an absolute tolerance of USD `0.01`.
- Weighted-average-price comparisons use an absolute tolerance of `0.0001`.
- Position and order quantities reconcile exactly.

## 4. Frozen reconciliation inputs

The reconciler consumes immutable tuples of:

1. Day 19 `OrderState` objects keyed by `client_order_id`;
2. broker order snapshots keyed by client and broker order IDs;
3. broker fills keyed by immutable `fill_id`;
4. broker positions keyed by symbol;
5. opening local positions keyed by symbol;
6. opening local cash and one broker cash snapshot; and
7. one explicit reconciliation `as_of` timestamp.

Exact duplicate fill rows are counted once. Reuse of a fill ID with different
normalized content is contradictory and blocks the gate. Unknown broker orders
or fills are never adopted silently.

## 5. Order and fill reconciliation

For every local order, the broker snapshot must agree on client ID, broker ID,
symbol, side, status, requested quantity, cumulative filled quantity, and
average fill price. Local `recovery_required` always blocks the gate even when
the broker snapshot otherwise agrees.

For every broker order, exactly one local order must exist. Broker order IDs
must be unique. Unique fills for an order must reconcile to the local cumulative
filled quantity. Their quantity-weighted price must reconcile to the local
average fill price. Fill client ID, broker ID, symbol, and side must match the
local order.

No mismatch automatically changes Day 19 state. Recovery requires a later,
separately authorized reconciliation action.

## 6. Position and cash reconciliation

Expected position for symbol `s` is:

```text
opening_position[s]
+ sum(buy_fill_quantity[s])
- sum(sell_fill_quantity[s])
```

Expected cash is:

```text
opening_cash
- sum(buy_fill_quantity * fill_price)
+ sum(sell_fill_quantity * fill_price)
```

Every expected non-zero position must exist at the broker. Every unexpected
non-zero broker position blocks the gate. Position quantities reconcile
exactly. Broker cash must reconcile to expected cash within one cent.

Commissions, fees, corporate actions, transfers, and margin interest are absent
from these synthetic fixtures. Day 22 must extend cash reconciliation before
interpreting real paper results.

## 7. Frozen operational limits

The Day 20 synthetic controller uses these exact limits:

- maximum requested quantity for one order: `25` shares;
- maximum absolute broker position per symbol: `100` shares;
- maximum gross marked notional: USD `25,000`;
- maximum non-terminal local orders: `5`;
- leverage: prohibited; and
- kill switch: latched once engaged and not resettable by Day 20.

Limit diagnostics do not liquidate or resize anything. They block the
operational gate and require manual review.

## 8. Stream health, reconnect, and circuit breaker

The exact stream states are:

1. `healthy`;
2. `stale`;
3. `reconnecting`;
4. `circuit_open`; and
5. `killed`.

A stream is stale more than 30 seconds after its last received message. Stale
state blocks the operational gate. Reconnect failures use the exact bounded
backoff schedule `1`, `2`, then `4` seconds. The third consecutive failure
opens the circuit. A successful transport reconnect remains `stale` until a new
valid message arrives. A valid message resets the failure count and returns the
stream to `healthy` unless the circuit or kill switch is already latched.

Repeated checks that do not change stream state do not append duplicate audit
rows. Circuit-open and killed states are terminal within Day 20.

## 9. Exact diagnostic vocabulary

The exact Day 20 reason-code order is:

1. `local_recovery_required`;
2. `local_order_missing_at_broker`;
3. `unknown_broker_order`;
4. `broker_order_id_mismatch`;
5. `duplicate_broker_order_id`;
6. `order_status_mismatch`;
7. `requested_quantity_mismatch`;
8. `cumulative_fill_mismatch`;
9. `weighted_average_price_mismatch`;
10. `duplicate_fill_conflict`;
11. `unknown_order_fill`;
12. `fill_broker_order_id_mismatch`;
13. `fill_symbol_side_mismatch`;
14. `fill_quantity_mismatch`;
15. `broker_position_missing`;
16. `unexpected_broker_position`;
17. `position_quantity_mismatch`;
18. `cash_balance_mismatch`;
19. `snapshot_stale`;
20. `single_order_limit_exceeded`;
21. `symbol_position_limit_exceeded`;
22. `gross_notional_limit_exceeded`;
23. `open_order_limit_exceeded`;
24. `stream_stale`;
25. `reconnect_exhausted`; and
26. `kill_switch_latched`.

Diagnostics store only normalized values and safe identifiers. Raw provider
payloads and exceptions are outside Day 20.

## 10. Operational decision semantics

The operational gate passes only when reconciliation, positions, cash, limits,
and stream health all pass and neither circuit breaker nor kill switch is
latched. Every active reason code is retained in frozen order.

Even when the gate passes:

- `day20_order_submission_authorized` is always `false`; and
- `can_submit_orders` is always `false`.

Thus a passing Day 20 result means the synthetic state is internally safe for a
future controlled paper loop. It is not order authorization.

## 11. Frozen synthetic scenarios

The canonical scenario order is:

1. `fully_reconciled`;
2. `partial_fill_reconciled`;
3. `missed_stream_update`;
4. `orphan_broker_order`;
5. `position_mismatch`;
6. `cash_mismatch`;
7. `local_recovery_required`;
8. `stale_stream_recovered`;
9. `reconnect_exhausted`;
10. `single_order_limit_breach`;
11. `gross_exposure_breach`; and
12. `kill_switch_latched`.

Expected mismatches, stale states, reconnect failures, limit breaches, and kill
switch engagement are valid known-answer outcomes. They do not make the Day 20
implementation evaluation incomplete.

## 12. Exact artifact bundle

The writer emits exactly eight files in this order:

1. `scenario_summary.csv`;
2. `reconciliation_summary.csv`;
3. `reconciliation_diagnostics.csv`;
4. `position_cash_reconciliation.csv`;
5. `stream_transition_log.csv`;
6. `operational_decisions.csv`;
7. `report.md`; and
8. `manifest.json`.

The manifest hashes every non-manifest artifact. The writer uses strict JSON,
deterministic row and column order, sibling staging, atomic replacement,
rollback, overwrite protection, and a final exact allow-list check.

## 13. Test and acceptance contract

Focused tests must cover all 26 reason codes, exact and conflicting fill
duplicates, Day 19 recovery propagation, order/fill/position/cash comparisons,
snapshot staleness, every frozen limit boundary, stream timeout boundaries,
the `1/2/4` reconnect schedule, reconnect recovery, terminal circuit/kill
states, exact schemas, manifest hashes, rollback, and byte replay.

Day 20 is complete only when:

1. the focused tests pass;
2. the complete repository suite passes;
3. all 12 scenarios meet their known answers;
4. artifact hashes and fixed-input byte replay pass;
5. `git diff --check` and repository status checks pass;
6. the cumulative progress report and output copy are updated;
7. no credentials, broker network, canonical market data, or locked data are
   accessed; and
8. zero orders or account mutations occur.

Day 21 may use the gate only after a separately frozen controlled-paper
protocol and explicit user authorization to submit paper orders.
