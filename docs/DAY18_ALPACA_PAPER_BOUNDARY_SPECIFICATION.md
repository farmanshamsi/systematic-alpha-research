# Day 18 Alpaca Paper-Broker Boundary and Preflight Specification

## 1. Status and authority

- Specification version: `day18_alpaca_paper_boundary_v1`
- Status: frozen before implementation
- Scope: read-only Alpaca paper-account boundary and execution-capability audit
- Core symbols: SPY, QQQ, IWM, in that exact order
- Order submission: prohibited
- Position mutation: prohibited
- Locked 2026 research data: prohibited

This document is the Day 18 implementation contract. Code, tests, reports, and
artifacts must follow it exactly. A change requires an explicit documented
revision before implementation.

## 2. Objective

Establish a broker-neutral, fail-closed boundary around the Alpaca paper
Trading API and prove that the configured account, endpoint, market clock, and
core-symbol eligibility can be inspected without submitting, replacing, or
canceling an order.

Day 18 is an operational-safety and connectivity exercise. It does not select
a strategy, authorize paper execution, access locked data, or claim that a
successful preflight makes the system deployment-ready.

## 3. Governing evidence

The current CQF Algorithmic Trading brief requires broker-API live testing,
exception handling, order-type and time-in-force analysis, verification of
broker responses, and operational discipline. Alpaca's current official Trading
API documentation identifies the paper base URL as
`https://paper-api.alpaca.markets`, lists the supported US-equity order types,
and documents the associated time-in-force values.

Official references used for the Day 18 capability snapshot:

- <https://docs.alpaca.markets/us/docs/getting-started-with-trading-api>
- <https://docs.alpaca.markets/us/docs/orders-at-alpaca>
- <https://docs.alpaca.markets/us/v1.1/reference/postorder>

## 4. Hard safety boundary

The Day 18 adapter must reject configuration unless all of the following hold:

1. project environment is exactly `paper`;
2. broker provider is exactly `alpaca`;
3. broker paper mode is exactly `true`;
4. the configured endpoint is exactly
   `https://paper-api.alpaca.markets` after removal of a trailing slash;
5. live trading is exactly `false`;
6. paper mode is required;
7. manual order confirmation is required;
8. the kill switch is enabled; and
9. Day 18 order submission is exactly `false`.

The public Day 18 interface must expose only:

- `get_account_snapshot()`;
- `get_market_clock_snapshot()`; and
- `get_asset_snapshot(symbol)`.

It must not expose submit, replace, cancel, close-position, or transfer
operations. The implementation may construct Alpaca's official
`TradingClient`, but it must do so with `paper=True` and the exact paper URL.

## 5. Credential contract

The YAML configuration may contain credential variable names, never credential
values. Credentials may be loaded from the named process-environment variables
or the gitignored local `.env` file. Both the API key and secret must be present
and non-empty; partial credentials are a hard failure.

Credential values must never appear in:

- exceptions or console output;
- logs or test snapshots;
- Markdown, JSON, CSV, or manifests;
- source code, configuration, or fixtures; or
- Git history.

Canonical outputs may state only whether credentials were loaded and whether
credential values were persisted. They may not store account identifiers,
account numbers, balances, buying power, cash, equity, or portfolio value.

## 6. Frozen read-only preflight

The canonical call order is:

1. account;
2. market clock;
3. SPY asset;
4. QQQ asset; and
5. IWM asset.

No retry is permitted on authentication, authorization, unsafe configuration,
or invalid-response errors. Day 18 does not implement a general retry policy;
that belongs to the later monitoring work.

### 6.1 Account gate

The normalized account snapshot contains only:

- status;
- trading blocked;
- account blocked;
- trade suspended by user; and
- shorting enabled.

The account gate passes only when status is `ACTIVE`, every blocking flag is
false, and shorting is enabled.

### 6.2 Clock gate

The normalized clock snapshot contains:

- broker timestamp;
- market-open flag;
- next open; and
- next close.

All timestamps must be timezone-aware, finite datetimes and next open must
precede next close. A currently closed market is a valid state and is not a
preflight failure.

### 6.3 Asset gate

Each requested asset must match its requested uppercase symbol and must be:

- US equity;
- active;
- tradable;
- shortable;
- easy to borrow; and
- fractionable.

The output records separate long, short, and fractional eligibility flags. The
canonical Day 18 gate requires all three flags for all three core symbols
because the research system contains both long/flat and long/short strategies.

### 6.4 Overall gate

`preflight_passed` is true only when the paper endpoint, account, clock, and all
three asset gates pass. Every diagnostic must be retained even if the overall
gate fails. A failed gate must produce a non-zero runner exit and must never be
reinterpreted as success.

## 7. Order-type and time-in-force capability snapshot

The provider's documented US-equity order types are:

1. `market`;
2. `limit`;
3. `stop`;
4. `stop_limit`; and
5. `trailing_stop`.

The provider's documented US-equity time-in-force values are:

1. `day`;
2. `gtc`;
3. `opg`;
4. `cls`;
5. `ioc`; and
6. `fok`.

The capability artifact is informational. Every row must set
`day18_authorized` to false because order submission is outside Day 18. The
project defaults remain `market` and `day`, regular trading hours only, with
manual confirmation required. Future order schemas must validate type/TIF,
quantity mode, session, and advanced-order constraints before submission.

## 8. Error taxonomy

The public broker boundary uses the following fail-closed categories:

- `PaperBrokerConfigurationError`: unsafe or incomplete configuration;
- `PaperBrokerCredentialError`: missing or incomplete credentials;
- `PaperBrokerConnectionError`: SDK/network request failure;
- `PaperBrokerResponseError`: missing, malformed, contradictory, or unexpected
  response data; and
- `PaperBrokerPreflightError`: normalized data failed a frozen Day 18 gate.

Public exception messages must be generic and safe to persist. The original
exception may be chained for local debugging, but its text must not be copied
into the artifact bundle.

## 9. Exact artifact bundle

The writer emits exactly five files in this order:

1. `preflight_summary.json`;
2. `asset_eligibility.csv`;
3. `capability_matrix.csv`;
4. `report.md`; and
5. `manifest.json`.

The manifest hashes every non-manifest artifact with SHA-256 and records the
schema version, exact file allow-list, core-symbol order, paper endpoint,
redaction flags, and order-submission prohibition. The writer must use sibling
staging, atomic replacement, rollback on failure, strict JSON, deterministic
row ordering, and final allow-list verification.

The live artifact is a timestamped broker-state snapshot and is not expected to
match a future live run byte for byte. For a fixed synthetic preflight object,
the writer must reproduce identical bytes.

## 10. Test contract

Focused tests must prove:

- exact safe configuration passes;
- every unsafe configuration flag fails before client construction;
- live and arbitrary URL overrides are rejected;
- credentials load without appearing in representations or errors;
- exact read-only call order;
- normalization of model-like and mapping responses;
- missing and malformed response fields fail closed;
- closed-market clock is valid;
- account and asset gate failures remain explicit;
- public adapter has no order-mutation methods;
- exact capability ordering and all `day18_authorized` values false;
- exact artifact schemas, ordering, hashes, redaction, overwrite, rollback, and
  deterministic replay; and
- the runner makes one preflight call and one writer call.

Synthetic tests must not make network calls, submit orders, load canonical
market data, or read any 2026 data.

## 11. Canonical-run acceptance

Day 18 is complete only when:

1. the frozen specification exists before implementation;
2. focused tests pass;
3. the full repository test suite passes;
4. `git diff --check` passes;
5. the live read-only Alpaca paper preflight either passes or is reported as an
   explicit external-access failure;
6. the artifact bundle hashes and allow-list verify;
7. a fixed-input replay is byte-identical;
8. a repository secret-pattern scan finds no credential value or credential
   file staged for commit;
9. `PROJECT_PROGRESS.md` records the result and next gate; and
10. no order, position, account, or research-data mutation occurred.

Day 18 does not authorize Day 19 order submission. Day 19 may model order state
with synthetic messages, but any later paper-order submission still requires
its own frozen contract and explicit user authorization.
