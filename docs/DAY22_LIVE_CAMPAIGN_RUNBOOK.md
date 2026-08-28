# Day 22 Live Paper Calibration Campaign Runbook

## Frozen scope

- Alpaca paper endpoint only; real-money trading remains disabled.
- SPY only, 0.01 share per entry, market/day orders, no extended hours.
- Ten maximum entries and ten maximum immediate opposite flattens.
- Two maximum round trips per XNYS session.
- Frozen sessions: 3-7 August 2026.
- Frozen New York slots: 10:15 and 14:15.
- Equivalent Pakistan times: 19:15 and 23:15.
- Entry window: the scheduled timestamp through 59.999 seconds after it.
- Missed or failed-gate slots are recorded and never moved.
- Purpose: `calibration_probe`; alpha eligible: `false`.

## Normal unattended operation

The one-shot watcher is started through `caffeinate` so the Mac remains awake.
It polls every 15 seconds, performs no broker read outside a due slot except an
explicit preflight, and terminates after the final slot or immediately after a
manual-recovery latch. It does not restart itself.

The source of truth is:

```text
artifacts/day22/live_campaign/campaign_state.json
```

At initial activation the required state is:

```text
state_revision = 0
manual_recovery_required = false
remaining authorized_scheduled slots = 10
entry submissions = 0
flatten submissions = 0
```

## Per-slot safety sequence

1. Verify exact authorization, campaign identity, unused slot, total/session
   caps, Day 20 hashes, paper endpoint/account/asset, market clock, and at least
   30 minutes to close.
2. Require a flat SPY position, no open SPY order, no duplicate deterministic
   client-order ID, and a clear recovery latch.
3. Read an IEX SPY quote and require positive non-crossed prices, no more than
   two seconds of quote age, and no more than USD 10 arrival-mid notional.
4. Recheck slot time, clock synchronization, position, open orders, and
   duplicate ID.
5. Atomically record `in_progress` and arm recovery before the entry API call.
6. Submit at most one 0.01-share entry and immediately submit the opposite
   flatten for exactly the confirmed positive filled quantity.
7. Require no open SPY order, zero SPY position, and matching entry/flatten
   quantity at shutdown before clearing the recovery latch.
8. Write the immutable seven-file slot bundle and its SHA-256 manifest.

## Evidence interpretation

The synthetic Day 22 bundle remains unchanged and is not realized evidence.
Live slot bundles appear as `slot_01` through `slot_10`. Calibration P&L may be
reported as an execution side effect but can never enter strategy profitability,
Sharpe, hit-rate, drawdown, VaR/ES, Beta, or promotion claims.

## Failure and recovery

If `manual_recovery_required` becomes `true`, do not restart the watcher and do
not submit a later campaign entry. First inspect the Alpaca paper dashboard for
an open SPY order or non-zero SPY position and compare it with the relevant
deterministic client IDs in campaign state and per-slot evidence.

An ambiguous submission is deliberately treated as potentially live. Never
retry an entry under a new client ID. Any recovery action must be limited to
the paper account and must prioritize canceling the scoped unresolved order or
flattening the confirmed residual SPY quantity. Save the broker result before
clearing a latch. If the exact broker state cannot be established, stop and
request user direction.

Runtime output is isolated in:

```text
/private/tmp/axiom_day22_campaign.stdout.log
/private/tmp/axiom_day22_campaign.stderr.log
```

Credentials, request headers, account IDs, and raw broker payloads must never be
copied into project artifacts or progress reports.

