# Day 19 Synthetic Order-State Machine Specification

## 1. Status and authority

- Specification version: `day19_order_state_machine_v1`
- Status: frozen before implementation
- Scope: deterministic synthetic order-state processing and audit replay
- Broker network access: prohibited
- Credential access: prohibited
- Order submission, replacement, cancellation, and position mutation: prohibited
- Locked 2026 research data: prohibited

This document is the Day 19 implementation contract. Any change to status
semantics, transition rules, identifiers, quantities, timestamps, timeout
thresholds, schemas, or artifact order requires a documented specification
revision before implementation.

## 2. Objective

Build a broker-neutral state machine that converts an append-only sequence of
synthetic broker order messages into a deterministic local order state while
failing closed on duplicates with conflicting content, stale or out-of-order
messages, illegal status transitions, contradictory identifiers, and invalid
cumulative fill information.

Day 19 is not a broker adapter and must not call Alpaca. It supplies the state
logic that later reconciliation and paper-execution work can use.

## 3. Governing evidence

The CQF Algorithmic Trading brief requires verification of broker responses and
explicit handling of partial fills and incorrect or inconsistent fill
information. Alpaca's official order documentation recommends streaming updates
for order-state maintenance and documents common lifecycle states including
`new`, `partially_filled`, `filled`, `done_for_day`, `canceled`, `expired`, and
`replaced`. The installed `alpaca-py 0.43.5` SDK exposes the complete provider
status vocabulary used below.

Official reference:

- <https://docs.alpaca.markets/us/docs/orders-at-alpaca>

## 4. Frozen identifiers and numeric types

Every local intent has one immutable `client_order_id`. Every broker update has
one immutable `event_id`, a non-negative integer `provider_sequence`, and a
`broker_order_id`.

- Client order IDs must match `axiom-[a-z0-9-]{8,48}`.
- Broker order IDs and event IDs are non-empty opaque text with no whitespace.
- An order's first valid broker update binds its broker order ID permanently.
- Reuse of a client order ID with identical intent content is idempotent.
- Reuse with different content is an idempotency conflict and recovery trigger.
- Reuse of an event ID with identical normalized content is a duplicate and is
  ignored without changing economic state.
- Local `received_at` is transport metadata, not provider event content, and is
  excluded from the event fingerprint so a later redelivery remains idempotent.
- Reuse of an event ID with different content is contradictory and fails
  closed.

Order and fill quantities and prices use `decimal.Decimal`, never binary float.
Quantities must be finite and positive for intents, and cumulative fill
quantities must lie in `[0, requested_quantity]`.

## 5. Frozen status vocabulary

The exact internal status order is:

1. `intent_created`;
2. `pending_review`;
3. `pending_new`;
4. `held`;
5. `accepted`;
6. `accepted_for_bidding`;
7. `new`;
8. `partially_filled`;
9. `pending_cancel`;
10. `pending_replace`;
11. `done_for_day`;
12. `stopped`;
13. `suspended`;
14. `calculated`;
15. `filled`;
16. `canceled`;
17. `expired`;
18. `replaced`; and
19. `rejected`.

Terminal statuses are `filled`, `canceled`, `expired`, `replaced`, and
`rejected`. A terminal state may receive a byte-equivalent duplicate or an
identical same-status confirmation, but it may not transition to a different
status.

## 6. Legal transition graph

Same-status confirmations are allowed when identifiers, quantity, cumulative
fill, and average price remain consistent. `partially_filled` may repeat only
with non-decreasing cumulative fill.

The non-self transition graph is:

| From | Allowed next statuses |
|---|---|
| `intent_created` | any provider status in the frozen vocabulary |
| `pending_review` | `held`, `pending_new`, `accepted`, `new`, `rejected`, `canceled`, `expired` |
| `pending_new` | `held`, `accepted`, `new`, `rejected`, `canceled`, `expired` |
| `held` | `pending_new`, `accepted`, `new`, `rejected`, `canceled`, `expired` |
| `accepted` | `accepted_for_bidding`, `new`, `partially_filled`, `filled`, `pending_cancel`, `pending_replace`, `canceled`, `expired`, `rejected`, `held` |
| `accepted_for_bidding` | `new`, `partially_filled`, `filled`, `pending_cancel`, `canceled`, `expired`, `rejected` |
| `new` | `partially_filled`, `filled`, `pending_cancel`, `pending_replace`, `done_for_day`, `canceled`, `expired`, `rejected`, `stopped`, `suspended`, `calculated` |
| `partially_filled` | `filled`, `pending_cancel`, `pending_replace`, `done_for_day`, `canceled`, `expired`, `replaced`, `stopped`, `suspended`, `calculated` |
| `pending_cancel` | `new`, `partially_filled`, `filled`, `canceled`, `done_for_day`, `expired` |
| `pending_replace` | `new`, `partially_filled`, `filled`, `replaced`, `canceled`, `expired` |
| `done_for_day` | `new`, `partially_filled`, `filled`, `pending_cancel`, `pending_replace`, `canceled`, `expired` |
| `stopped` | `new`, `partially_filled`, `filled`, `canceled`, `expired`, `rejected` |
| `suspended` | `new`, `partially_filled`, `filled`, `canceled`, `expired`, `rejected` |
| `calculated` | `filled`, `canceled`, `expired` |

The first update may contain a terminal status because a local process can
restart after the broker has already completed the order. That fast-forward is
allowed only when every identifier and quantity invariant is satisfied.

## 7. Cumulative fill accounting

Provider `filled_quantity` is cumulative. For accepted event `t`:

```text
incremental_fill[t] = cumulative_fill[t] - cumulative_fill[t-1]
```

Required invariants:

- cumulative fill never decreases;
- cumulative fill never exceeds requested quantity;
- `partially_filled` requires `0 < filled < requested`;
- `filled` requires `filled = requested`;
- `rejected` requires zero filled quantity;
- a positive filled quantity requires a finite positive average fill price;
- a zero filled quantity requires no average fill price;
- when cumulative fill is unchanged, average fill price must also be unchanged;
- replacement status requires a non-empty replacement order ID distinct from
  the current broker order ID; and
- a terminal partial quantity remains visible for canceled, expired, or
  replaced orders.

Day 19 does not infer individual fill prices from cumulative average price.
That decomposition requires actual fill records and belongs to later work.

## 8. Event ordering, duplicates, and staleness

For each order, accepted non-duplicate updates must have strictly increasing
provider sequence and non-decreasing event timestamps. All timestamps must be
timezone-aware and are normalized to UTC.

An update is stale when its receipt time is more than 120 seconds after its
event time. A stale update is rejected, economic state is unchanged, the event
is appended to the rejection audit, and the order is marked
`recovery_required`.

Provider timestamps may be at most five seconds ahead of receipt time. A larger
future skew is invalid and fails closed.

The first normalized fingerprint for an event ID is retained whether the
message is accepted or rejected. Therefore an identical redelivery of a
previously rejected or unknown-order message is ignored idempotently, while
conflicting reuse of that event ID still fails closed.

## 9. Timeout semantics

Timeout checks use an explicitly supplied timezone-aware `as_of` timestamp;
the state machine must not call the system clock.

- Acknowledgment timeout: 30 seconds after intent submission while status
  remains `intent_created`.
- Update timeout: 120 seconds since the last accepted broker message for a
  non-terminal order.
- Terminal orders do not time out.
- Repeating the same timeout check is idempotent for the same order, timeout
  kind, and last accepted provider sequence.

A timeout does not invent a broker status or cancel an order. It sets
`recovery_required`, creates an append-only timeout audit row, and requires
later broker reconciliation.

## 10. Failure semantics

Rejected messages leave status, cumulative filled quantity, average fill price,
and last accepted provider sequence unchanged. Known-order failures set
`recovery_required`. Unknown-order messages set the global recovery flag.

The exact reason-code vocabulary is:

1. `unknown_order`;
2. `idempotency_conflict`;
3. `duplicate_event_conflict`;
4. `broker_order_id_mismatch`;
5. `requested_quantity_mismatch`;
6. `provider_sequence_not_increasing`;
7. `event_time_regressed`;
8. `event_arrived_stale`;
9. `event_time_future_skew`;
10. `illegal_status_transition`;
11. `filled_quantity_decreased`;
12. `filled_quantity_exceeds_order`;
13. `status_quantity_inconsistent`;
14. `average_fill_price_inconsistent`;
15. `replacement_id_missing`;
16. `invalid_event`;
17. `acknowledgment_timeout`; and
18. `update_timeout`.

Errors expose only reason codes and safe identifiers; raw broker exception text
is outside Day 19.

When an upstream adapter can safely identify the client order and event but
cannot normalize the remaining message, it calls `record_invalid_event`. The
raw message is not retained; the state machine writes only those safe
identifiers, marks recovery, and emits `invalid_event`.

## 11. Frozen synthetic scenarios

The canonical scenario order is:

1. `complete_fill`;
2. `cancel_after_partial`;
3. `broker_rejection`;
4. `replacement`;
5. `duplicate_delivery`;
6. `decreasing_fill_rejected`;
7. `out_of_order_rejected`;
8. `acknowledgment_timeout`; and
9. `unknown_order_rejected`.

Each scenario uses fixed UTC timestamps, decimal quantities, identifiers, and
event sequences. Expected rejections and timeouts are valid scenario outcomes;
they do not make the Day 19 implementation evaluation fail.

## 12. Exact artifact bundle

The writer emits exactly eight files in this order:

1. `scenario_summary.csv`;
2. `final_states.csv`;
3. `transition_log.csv`;
4. `rejection_diagnostics.csv`;
5. `timeout_diagnostics.csv`;
6. `state_transition_matrix.csv`;
7. `report.md`; and
8. `manifest.json`.

The manifest hashes every non-manifest artifact. The writer uses strict JSON,
deterministic row and column order, sibling staging, atomic replacement,
rollback, overwrite protection, and a final exact allow-list check.

## 13. Test and acceptance contract

Focused tests must cover every status, every legal edge, every reason code,
terminal immutability, partial-fill arithmetic, event-id idempotency, client-ID
idempotency, ordering, staleness, future skew, timeout idempotency, recovery
flags, exact schemas, manifest hashes, rollback, and fixed-input byte replay.

Day 19 is complete only when:

1. the focused tests pass;
2. the complete repository suite passes;
3. the nine frozen scenarios meet their expected outcomes;
4. artifact hashes and deterministic replay pass;
5. `git diff --check` and repository status checks pass;
6. the progress report is updated and copied to the project output folder;
7. no credentials, broker network, canonical market data, or locked data were
   accessed; and
8. zero orders or account mutations occurred.

Day 20 may consume the Day 19 recovery flags and audit trail for reconciliation
and monitoring. Day 19 does not authorize paper-order submission.
