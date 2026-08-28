# Day 23 Paper-Safe Operations Runbook

## Safe default

From the repository root, the offline health check is:

```bash
python scripts/run_day23_operational_validation.py --check-only
```

It loads neither `.env` nor process credentials, creates no broker client,
accesses no network or market data, and cannot submit an order. A successful run
reports all 14 checks passed. Failure is non-zero and names the failed gates.

The matching container default is:

```bash
docker compose run --rm axiom-smoke
```

No container engine was available on the development host during Day 23. This
command is therefore a future operator/CI check, not claimed local evidence.

## Locked environment

Use Python 3.11 in a new virtual environment. Install the exact transitive lock
first, then the local package without dependency resolution:

```bash
python -m venv .venv-day23
.venv-day23/bin/python -m pip install --require-hashes -r requirements.lock
.venv-day23/bin/python -m pip install --no-deps --no-build-isolation .
.venv-day23/bin/python -m pip check
.venv-day23/bin/python scripts/run_day23_operational_validation.py --check-only
```

Do not copy `.env` into the environment or image. The lock is regenerated only
from `pyproject.toml` with the `dev` extra and SHA-256 hashes, then reviewed and
validated in another clean environment.

## Scheduling boundary

The dispatcher accepts only these exact jobs:

- `health-smoke`: automatic-safe and not order-capable.
- `day22-campaign-once`: requires `--authorized-paper-campaign`, runs at most
  one state-aware due-slot check, and preserves the existing frozen slots.
- `day21-strategy-once`: manual-only and requires
  `--authorized-paper-order` plus every existing Day 21 gate.

Unknown jobs, missing flags, extra authorization flags, and retries fail closed.
While the authorized Day 22 watcher is active, do not launch a second external
scheduler. The watcher remains the sole live campaign scheduler; the one-shot
entry point is retained for future supervised operation and recovery.

## Day 22 operator checks

Keep the host awake, powered, and online through the frozen New York slots.
Inspect status read-only; do not edit `campaign_state.json`, remove the lock,
move a slot, rerun a consumed slot, or manually flatten unless the evidence says
`manual_recovery_required`. If that latch appears, stop automation, inspect the
paper account's SPY position and open SPY orders, and recover under explicit
human supervision before clearing any state.

## Persistence and backups

Deterministic day bundles are immutable after freeze. Day 22 mutable state uses
atomic replacement and each consumed slot writes an immutable evidence bundle.
Logs and mutable state are not research evidence.

A backup must use a new timestamped directory. Before copying, verify every
source file against its manifest; after copying, replay every hash. Never back
up `.env`, credential values, request headers, raw broker payloads, or account
identifiers into the repository. Restore read-only first, then verify hashes and
run a clean paper endpoint/account/position/order preflight before automation.

## CI and incident handling

CI installs `requirements.lock`, runs `pip check`, the offline smoke, focused
Day 23 tests, and the full suite with no broker secrets. A failed health check,
dependency conflict, hash mismatch, unexpected order-capable default, Day 22
recovery latch, or non-flat shutdown is a stop condition—not a retry signal.

