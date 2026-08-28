# Day 21 Controlled Alpaca Paper-Execution Specification

## 1. Status and authority

- Specification version: `day21_controlled_paper_execution_v1`
- Status: frozen before implementation
- Scope: one deliberately bounded Alpaca paper-only execution session
- Implementation and read-only authorization: received
- Exact authorization for the 0.01-share SPY paper entry and same-run flatten:
  pending before the first possible order-capable command
- Real-money trading: prohibited
- Locked 2026 research interval (2026-01-02 through 2026-06-30): prohibited
- Canonical research-data mutation: prohibited

This document is the Day 21 implementation and execution contract. A change to
the candidate, symbol, signal mapping, size, order type, session rule, endpoint,
entry/exit behavior, timeout, gate, evidence schema, or abort behavior requires
an explicit specification revision and fresh authorization before another paper
order can be submitted.

## 2. Objective and non-claim

Connect one predeclared research candidate to a narrow paper-only loop and
observe real provider order, fill, position, cash, and reconciliation behavior.
The run tests operational implementability. It does not prove alpha, estimate a
stable Sharpe ratio, or promote the candidate to production.

Day 17 found positive net development evidence for the slow OU/VWAP family,
but its inference remained inconclusive. Day 21 therefore labels it an
`operational_probe_candidate`, not a selected winner or profitable strategy.

## 3. Frozen candidate and live signal

- Candidate ID: `ou_vwap_slow`
- Symbol: `SPY`
- Frequency: 15-minute bars
- Data feed: configured Alpaca IEX feed
- Operational-data start floor: `2026-07-01T00:00:00Z`
- Frozen Day 17 parameters:
  - reference window: 52 bars;
  - OU window: 208 transitions;
  - variance-ratio lag: 4;
  - variance-ratio threshold: 0.95;
  - entry z-score: 2.25;
  - exit z-score: 0.25;
  - half-life range: 1 through 39 bars; and
  - maximum holding period: 39 bars.

Only official XNYS regular-session bars are retained. A bar is usable only
after its scheduled 15-minute interval has completed according to the broker
clock. The last complete bar drives the already-lagged Day 17 `position` value.
That bar is stale when its scheduled end is more than 20 minutes before the
broker clock.
The exact mapping is `position +1 -> buy`, `position -1 -> sell`, and
`position 0 -> no order`. Missing diagnostics, an incomplete warm-up, a
non-finite value, or a stale final bar aborts the run.

Operational bars after 2026-06-30 are isolated from the locked final-test
dataset. They are not appended to canonical research data and are not used to
score or retune the strategy.

## 4. Frozen broker and order boundary

- Endpoint: `https://paper-api.alpaca.markets`
- Account environment: Alpaca paper only
- Order type: market
- Time in force: day
- Extended hours: false
- Quantity: exactly 0.01 SPY share
- Maximum entry notional guard: USD 10.00 using the last complete bar close
- Maximum Day 21 entry orders: one
- Maximum Day 21 flatten orders: one
- Client-order prefix: `axiom-day21-spy-`
- Regular market hours only
- Minimum time to scheduled close at entry: 30 minutes
- Entry fill wait: 30 seconds
- Cancel/reconciliation wait after an unfilled or partial entry: 30 seconds
- Flatten fill wait: 30 seconds
- Overnight exposure: prohibited

Fractional market orders are used only because the live asset gate must confirm
that SPY remains fractionable. Market execution price is not guaranteed; the
last complete bar close is only a pre-submit notional guard, not a fill-price
promise.

## 5. Startup gates and abort conditions

All of the following must pass immediately before entry submission:

1. the one-time authorization object is present and exact;
2. the Day 18 configuration still enforces paper mode, the paper endpoint, live
   trading disabled, manual confirmation, and the kill switch;
3. the Day 18 live account, clock, and SPY eligibility gates pass;
4. the broker clock reports the regular market open and at least 30 minutes
   remain until the scheduled close;
5. the Day 20 synthetic operational gate remains a tested prerequisite;
6. the live operational-data window starts after the locked final-test period;
7. the signal is available, current, finite, and non-zero;
8. the last close times 0.01 does not exceed USD 10.00;
9. there is no pre-existing SPY paper position;
10. there is no open SPY paper order;
11. there is no prior Day 21 entry order for the same signal timestamp; and
12. no kill-switch or reconciliation reason is active.

Any failure aborts without submitting or queuing an order. In particular, a
closed market is an abort, never permission to queue for the next session.
Other account positions and orders are observed for scope but are never
modified by Day 21.

## 6. Entry, observation, and shutdown

After the gates pass, the controller submits at most one 0.01-share SPY market
entry. It polls the specific returned broker order ID only. If the order becomes
fully filled, the controller records the fill fields and immediately submits an
opposite 0.01-share market flatten order. If the entry is rejected, canceled,
or expired, no flatten is sent.

If the entry is partially filled or remains non-terminal at timeout, the
controller cancels only that specific Day 21 order, re-reads it, and flattens
only the confirmed filled quantity. The controller never uses cancel-all or
close-all operations.

Shutdown re-reads the two specific order IDs, all open SPY orders, the SPY
position, account cash, and broker clock. Success requires no open Day 21 or SPY
order, zero SPY position, and fill arithmetic consistent with the observed
orders. Any residual exposure or uncertainty is recorded as
`manual_recovery_required`; the controller does not pretend the run reconciled.

## 7. Safe evidence contract

The evidence bundle contains only normalized, credential-free fields:

1. `protocol.json`;
2. `gate_results.csv`;
3. `signal_snapshot.csv`;
4. `order_events.csv`;
5. `fill_summary.csv`;
6. `position_cash_snapshots.csv`;
7. `reconciliation.json`;
8. `report.md`; and
9. `manifest.json`.

The manifest hashes every non-manifest artifact. API keys, secrets, request
headers, raw provider payloads, full account identifiers, and exception text are
prohibited. An abort bundle is valid evidence only when it clearly reports
`order_submission_occurred=false`; it does not satisfy the controlled-fill gate.

## 8. Testing and completion gate

Synthetic tests must cover paper-endpoint enforcement, authorization absence,
closed market, near-close, flat signal, stale data, warm-up failure, notional
cap, existing SPY exposure, existing SPY order, duplicate signal ID, entry
rejection, full fill and flatten, partial fill then cancel/flatten, timeout,
residual exposure, exact artifact schemas, secret scanning, hashes, and replay.

Day 21 is complete only when:

1. focused tests and the full repository suite pass;
2. the deterministic synthetic happy path and all abort/failure paths pass;
3. the live paper endpoint and account gates pass;
4. one eligible strategy-driven paper entry and its shutdown flatten are
   observed, or the day remains explicitly incomplete because a frozen gate
   prevented submission;
5. no unresolved SPY order or position remains;
6. the live evidence bundle verifies by hash;
7. `git diff --check` and repository status checks pass; and
8. no real-money endpoint, locked research data, or credential value is used or
   persisted.
