# Day 22 Execution and Live-Performance Validation Specification

## 1. Status and authority

- Specification version: `day22_execution_live_performance_v1`
- Status: frozen implementation contract; live campaign separately authorized
- Scope: execution-quality, realized-P&L, dynamic-risk, and prospective
  multi-trade campaign validation
- Day 22 live multi-order authorization: granted by explicit user approval on
  2026-08-03 for this exact bounded calibration campaign only
- Day 21 authorization inheritance: prohibited
- Broker network, credentials, order submission, cancellation, replacement,
  and position mutation during implementation validation: prohibited
- Locked 2026 research interval: prohibited

The user's exact Day 21 authorization permits at most its single 0.01-share SPY
entry and same-run flatten when every Day 21 gate passes. It does not authorize
the Day 22 multi-trade campaign. The separately granted Day 22 authorization is
limited to up to ten 0.01-share SPY calibration entries and up to ten immediate
opposite flatten orders, no more than two round trips per session, at only the
frozen 10:15 and 14:15 New York slots, and only when every safety gate passes.
It does not authorize real-money trading, strategy-signal orders, rescheduling,
larger quantities, other symbols, or discretionary additional orders.

## 2. Objective and evidence separation

Day 22 answers two different questions without mixing their evidence:

1. **Strategy evidence:** Did genuine frozen `ou_vwap_slow` position changes
   produce positive realized paper P&L after observed fills and commissions?
2. **Execution-calibration evidence:** How much delay, quoted spread, residual
   fill slippage, latency, rejection, and partial-fill behavior did the paper
   broker exhibit under pre-scheduled small probes?

Only records labeled `strategy_signal` may contribute to strategy hit rate,
profit factor, drawdown, VaR/ES, Beta-to-SPY, Sharpe, or profitability claims.
Records labeled `calibration_probe` can estimate execution behavior but cannot
be counted as alpha trades, even if their round-trip P&L is positive.

## 3. Frozen normalized execution record

Every filled execution leg contains:

- immutable execution and round-trip IDs;
- purpose: `strategy_signal` or `calibration_probe`;
- leg: `entry` or `exit`;
- SPY symbol, `buy` or `sell` side, and positive decimal quantity;
- decision timestamp and price;
- quote timestamp, bid, and ask;
- local submit timestamp;
- broker-submitted timestamp;
- fill timestamp and fill price; and
- non-negative commission.

All timestamps are timezone-aware UTC. Their required order is:

```text
decision_at <= quote_at <= submitted_at <= broker_submitted_at <= filled_at
```

The quote may be no more than two seconds old at local submission. Bid, ask,
decision, and fill prices must be finite and positive; bid cannot exceed ask.
The symbol is exactly SPY. Raw payloads, credentials, headers, and account IDs
are prohibited.

## 4. Exact shortfall decomposition

Let `s = +1` for a buy and `s = -1` for a sell. Let `D` be decision price, `M`
arrival mid, `T` arrival touch (`ask` for buy and `bid` for sell), and `F` fill
price. All basis-point components use decision price as the common denominator:

```text
total_shortfall_bps = s * (F - D) / D * 10,000
delay_bps           = s * (M - D) / D * 10,000
spread_bps          = s * (T - M) / D * 10,000
residual_bps        = s * (F - T) / D * 10,000
```

The identity `total = delay + spread + residual` must reconcile to `1e-9`
basis points. Negative residual represents price improvement relative to touch.
Decision-to-submit, broker acknowledgment, and fill latency are stored in
milliseconds.

## 5. Round-trip and P&L contract

Each round trip contains exactly one entry and one later exit with the same
purpose, symbol, and quantity; sides must be opposite. Long gross P&L is
`quantity * (exit_fill - entry_fill)`. Short gross P&L is
`quantity * (entry_fill - exit_fill)`. Net P&L subtracts both recorded
commissions. Fill prices already contain execution effects, so shortfall is an
attribution and is not subtracted a second time.

Unmatched, overlapping, same-side, unequal-quantity, time-reversed, or mixed-
purpose legs fail closed. Partial live fills enter Day 22 only after Day 21/22
reconciliation has established the exact final filled quantity.

## 6. Dynamic daily performance and risk

Daily snapshots are pre-labeled `strategy_signal` and contain session date,
net P&L, gross exposure, net exposure, turnover notional, and aligned SPY simple
return. Starting equity is USD 100,000. Equity, running peak, and drawdown are
calculated chronologically without scaling the observed paper P&L.

The strategy risk summary contains:

- cumulative P&L and return;
- annualized volatility and zero-rate Sharpe;
- maximum drawdown;
- 95% historical VaR and Expected Shortfall;
- Beta-to-SPY using sample covariance divided by sample SPY variance;
- total turnover notional;
- maximum gross and absolute net exposure;
- strategy round-trip hit rate and profit factor; and
- an exact evidence-availability flag.

VaR/ES, volatility, Sharpe, and Beta require at least 20 daily observations.
Beta additionally requires positive finite SPY sample variance. Until those
conditions hold, values are blank and the reason is
`insufficient_daily_observations`; they are never fabricated from a tiny
sample. Twenty-day rolling volatility and VaR/ES are blank before observation
20.

## 7. Prospective multi-trade campaign template

The calibration template contains ten possible round trips: 10:15 and 14:15
America/New_York on the first five consecutive XNYS sessions after a separately
authorized activation date. Entry sides alternate buy, sell, buy, sell in
schedule order. Quantity is exactly 0.01 SPY share, order type market, time in
force day, and extended hours false.

Frozen controls:

- at most two calibration round trips per session and ten total;
- one SPY position and one active round trip maximum;
- decision-price notional no more than USD 10;
- no order in the final 30 minutes;
- immediate opposite flatten after confirmed entry fill;
- skip, do not reschedule, a calibration probe when a genuine strategy order,
  position, stale-data state, mismatch, limit breach, or kill switch is active;
- every skipped probe remains in the evidence; and
- no optional stopping because fills or P&L look favorable.

This schedule is an execution experiment, not a trading strategy. A live
campaign manifest must freeze its activation date and exact ten timestamps
before the first campaign order.

### 7.1 Authorized live activation freeze

- Campaign ID: `day22_calibration_v1`
- Activation date: `2026-08-03`
- XNYS sessions: `2026-08-03` through `2026-08-07`
- Slot-entry window: scheduled timestamp inclusive through 60 seconds after
  the scheduled timestamp exclusive; a missed slot is recorded and is not
  rescheduled
- Exact UTC slots: `14:15` and `18:15` UTC on each frozen session
- Exact New York slots: `10:15` and `14:15` America/New_York on each frozen
  session
- Entry-side order: buy, sell, buy, sell, buy, sell, buy, sell, buy, sell
- Maximum authorized submissions: ten entries plus at most one opposite
  flatten for each confirmed positive entry fill
- Campaign evidence purpose: `calibration_probe`; alpha eligible: `false`

The authorization is consumed only by an in-window slot attempt. Read-only
preflight checks outside a slot do not consume it. Any failed gate records a
skip for that frozen slot without an entry submission. A positive entry fill
always prioritizes the authorized opposite flatten and shutdown reconciliation.
Any unresolved order, non-zero SPY position, fill mismatch, or ambiguous broker
response latches manual recovery and blocks every later campaign entry.

## 8. Exact artifact bundle

The deterministic writer emits exactly eight files:

1. `execution_shortfall.csv`;
2. `round_trip_pnl.csv`;
3. `daily_performance.csv`;
4. `risk_summary.csv`;
5. `campaign_schedule.csv`;
6. `campaign_summary.csv`;
7. `report.md`; and
8. `manifest.json`.

The manifest hashes every non-manifest artifact. The writer uses strict JSON,
fixed row and column order, sibling staging, atomic replacement and rollback,
overwrite protection, and an exact final allow-list. Synthetic artifacts must
state that broker network, credentials, orders, canonical data, and locked data
were not accessed.

The separately authorized live campaign is isolated under
`artifacts/day22/live_campaign/` and does not overwrite or relabel the
synthetic eight-file bundle. Its deterministic activation manifest hashes the
exact ten-slot schedule. Its mutable campaign state is atomically replaced and
records every authorized, missed, skipped, completed, or recovery-latched slot.
Before any entry submission, the state is changed to `in_progress` and the
manual-recovery latch is armed. After a safe result is persisted, the slot
state is finalized and the latch is cleared only when shutdown reconciliation
succeeds.

Each attempted slot is immutable and contains exactly seven files:

1. `gate_results.csv`;
2. `quote_snapshots.csv`;
3. `execution_records.csv`;
4. `order_events.csv`;
5. `position_cash_snapshots.csv`;
6. `result.json`; and
7. `manifest.json`.

The per-slot manifest hashes every non-manifest file. Execution records are
emitted only for positively filled legs with valid quote, local-submit,
broker-submit, fill-time, quantity, and fill-price evidence. A missing or stale
flatten quote never blocks the safety-priority flatten, but it prevents that
leg from being treated as complete shortfall evidence.

## 9. Test and completion contract

Focused tests cover malformed records, timestamp order, quote staleness, crossed
quotes, both sides of the exact shortfall identity, price improvement, latency,
long and short P&L, commissions, round-trip mismatch rejection, calibration/
strategy separation, 19/20-observation risk boundaries, VaR/ES, Beta,
drawdown, rolling metrics, schedule times and side order, hashes, secret scan,
rollback, byte replay, exact live slot timing, entry and session caps, duplicate
prevention, existing-position/order conflicts, atomic pre-submit intent,
single-consumption state, missed-slot handling, and ambiguous-submit recovery
latching.

Day 22 implementation is complete when focused tests, the full repository
suite, synthetic known answers, hashes, replay, progress output, and worktree
checks pass. Day 22 empirical validation remains incomplete until live
campaign records exist. Authorization and the August 3-7 activation freeze now
exist, but no campaign slot had occurred when this specification was updated.
Synthetic fills and Day 21's read-only abort cannot be represented as realized
execution or profitability evidence.
