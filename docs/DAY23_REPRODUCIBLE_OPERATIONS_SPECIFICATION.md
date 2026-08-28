# Day 23 Reproducible Operations Specification

## 1. Status and authority

- Specification version: `day23_reproducible_operations_v1`
- Status: frozen before implementation
- Scope: dependency locking, paper-safe runtime validation, containerization,
  scheduled entry points, persistence policy, runbook, and CI smoke coverage
- Real-money trading: prohibited
- New broker orders, cancellations, replacements, or position mutation during
  Day 23 validation: prohibited
- Day 22 campaign process, schedule, authorization, and mutable state: must not
  be stopped, reset, broadened, or consumed by Day 23 validation
- Locked 2026 final-test data: prohibited
- Git commit and push: prohibited until separately requested

Day 23 is an operations and reproducibility day. It cannot tune a strategy,
select a result, inspect the locked test, manufacture paper fills, or turn
calibration P&L into alpha evidence.

## 2. Objective

A clean Python 3.11 environment must be able to install the exact dependency
set, validate the repository configuration, run a network-free and credential-
free paper-safe startup/shutdown smoke check, reproduce the Day 23 evidence
bundle, and run the test suite. A container definition and CI workflow must use
the same lock and the same default smoke path.

Container execution is not treated as verified unless a compatible container
runtime is actually available. Static Dockerfile tests and a clean host virtual
environment are separate evidence and must be labeled separately.

## 3. Runtime contract

- Python: `3.11.x`, with `3.11.15` as the development reference runtime
- Package installation: exact transitive `requirements.lock`, followed by
  editable project installation with `--no-deps`
- Default mode: offline `smoke`
- Default broker environment: `paper`
- Alpaca endpoint: exactly `https://paper-api.alpaca.markets`
- `allow_live_trading`: exactly `false`
- `order_submission_enabled`: exactly `false`
- `require_paper_mode`, manual confirmation, and kill switch: exactly `true`
- Credentials: optional and unread in smoke mode; local environment or ignored
  `.env` only for explicit read-only/live paper modes
- Repository-root discovery: independent of the caller's current directory
- Time zones: exchange `America/New_York`, storage/scheduler `UTC`

Smoke mode imports the package and required libraries, validates configuration,
checks the dependency lock and scheduled-job contract, verifies required paths,
creates and removes a temporary persistence probe, and exits cleanly. It does
not construct a broker adapter, read credentials, access the network, inspect
market data, or call any order method.

## 4. Dependency lock

`requirements.lock` must pin every resolved runtime and development dependency
with exact versions and SHA-256 hashes. It is generated from `pyproject.toml`
including the `dev` extra. The project itself is not embedded as an editable
line in the lock.

Validation must prove:

- every direct runtime dependency appears in the lock;
- `pytest` and `pytest-cov` appear in the lock;
- every resolved requirement is exact rather than a lower bound or wildcard;
- the lock contains hashes;
- `exchange-calendars==4.13.2` and `alpaca-py==0.43.5` remain exact; and
- the installed environment has no dependency conflicts under `pip check`.

## 5. Container contract

The repository contains:

- `Dockerfile` using an exact Python 3.11 slim-bookworm tag or digest;
- `.dockerignore` excluding credentials, Git state, caches, local raw/processed
  data, outputs, artifacts, logs, and backups;
- `compose.yaml` with a default non-order-capable smoke service; and
- a container health check that runs the same network-free runtime validator.

The image must install `requirements.lock` before project source so dependency
layers are cacheable, install the package with `--no-deps`, run as a non-root
user, expose no port, and default to the offline smoke command. Secrets cannot
be copied into the image or declared in build arguments, environment defaults,
or Compose configuration.

Persistent `artifacts` and `logs` volumes are documented for explicit runtime
use, but the default smoke service writes only to a temporary container path.
The Day 22 order-capable watcher is deliberately absent from the default Compose
service.

## 6. Scheduled entry-point contract

`config/operations.yaml` declares exact, ordered jobs and UTC scheduling policy.
The generic scheduled runner exposes only:

1. `health-smoke`: offline, credential-free, non-order-capable;
2. `day22-campaign-once`: one state-aware due-slot check, requiring the exact
   `--authorized-paper-campaign` flag; and
3. `day21-strategy-once`: manual-only, requiring the exact Day 21 authorization
   flag and every existing Day 21 strategy/safety gate.

Unknown jobs fail closed. The runner never retries, reschedules, changes a slot,
or converts a manual job into an automatic job. Day 22's currently running
one-shot watcher remains the active campaign scheduler.

## 7. Health and lifecycle checks

The frozen smoke checks are ordered and fail closed:

1. supported Python version;
2. required package imports;
3. exact project identity and paper environment;
4. real-money disabled;
5. exact paper endpoint;
6. order submission disabled by default;
7. manual confirmation and kill switch enabled;
8. locked final-test flag and exact date boundary;
9. UTC/New York time-zone contract;
10. dependency-lock integrity;
11. scheduled-entrypoint integrity;
12. required repository files;
13. temporary persistence write/read/hash/remove round trip; and
14. clean shutdown with zero broker/network/credential/order access.

An explicit paper-read-only mode may recheck Alpaca connectivity later, but it
must be separate from deterministic Day 23 evidence and cannot consume a Day 22
slot.

## 8. Persistence and backup policy

- Research artifacts are immutable after their day is frozen.
- Live Day 22 state uses atomic replacement plus immutable per-slot bundles.
- Logs and mutable runtime state are outside deterministic research bundles.
- Secrets, headers, raw broker payloads, and account identifiers are never
  backed up into the repository.
- A backup is accepted only after source hashes verify under the relevant day
  manifest; the backup uses a new timestamped directory and never overwrites an
  earlier backup.
- Restore is read-only until hashes and a clean paper account/position/order
  preflight pass.
- Backups and logs are ignored by Git.

## 9. CI contract

GitHub Actions must:

- use Python 3.11;
- install the exact lock, then the package with `--no-deps`;
- run `pip check`;
- run the offline health check;
- run focused Day 23 tests; and
- run the full repository suite.

CI receives no Alpaca secret, accesses no broker endpoint, submits no order,
and does not load the locked final-test data.

## 10. Exact Day 23 evidence bundle

The deterministic writer emits exactly seven files:

1. `dependency_audit.csv`;
2. `health_checks.csv`;
3. `runtime_contract.json`;
4. `schedule_entrypoints.csv`;
5. `persistence_policy.json`;
6. `report.md`; and
7. `manifest.json`.

The manifest hashes every non-manifest artifact. The writer uses strict JSON,
fixed row/column order, sibling staging, atomic replacement and rollback,
overwrite protection, credential scanning, and an exact final allow-list.

## 11. Completion gate

Day 23 implementation is complete when:

- the frozen smoke and artifact tests pass;
- the affected operations/broker regression suite passes;
- the full repository suite passes;
- `pip check`, lock audit, hashes, byte replay, credential scan, and
  `git diff --check` pass;
- a fresh isolated Python 3.11 environment installs from the lock and runs the
  smoke check and focused tests;
- container definitions pass static validation; and
- the project progress file and saved outputs copy are updated.

If Docker is unavailable locally, image build/runtime verification remains an
explicit environmental limitation rather than being falsely claimed.

