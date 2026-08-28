# AL Project Decision Log

## Governance

The frozen AL V1.0 plan remains the canonical baseline.

This repository is the working implementation branch. Any changes caused by
Tutorial III or later CQF instructions must be documented before implementation.

---

## D-001 — Core Broker and Venue

**Decision:** Use Alpaca paper trading for the core project.

**Reasoning:**
- Paper environment is operational.
- Authentication has been tested successfully.
- Market clock and asset-information endpoints work.
- SPY, QQQ, and IWM are active and tradable.
- The architecture can later support additional venue adapters.

**Excluded from core scope:**
- Binance
- Kalshi
- Polymarket
- Real-money trading

These may be discussed as future extensions only.

---

## D-002 — Core Instrument Universe

**Primary trend instrument:** SPY

**Robustness instruments:**
- QQQ
- IWM

**Mean-reversion feasibility candidate pairs:**
- SPY / QQQ
- SPY / IWM
- QQQ / IWM

The candidate universe is evaluated through a predeclared pass/fail framework.
Correlation, profitability, trading costs and return ranking cannot determine
Day 14 eligibility. The fixed pair orientations and complete statistical and
OU gate definitions are recorded in D-014.

---

## D-003 — Trading Session and Timezones

**Trading session:** US regular market hours only

**Exchange timezone:** America/New_York

**Storage timezone:** UTC

**Extended-hours data:** Excluded from the core experiment

---

## D-004 — Sampling Frequencies

**Raw historical bars:** 1 minute

**Primary strategy frequency:** 15 minutes

**Robustness frequencies:**
- 30 minutes
- 60 minutes

**Long-run cointegration estimation:** Daily data where appropriate

**Execution-level pair analysis:** Intraday data after the long-run relationship
passes the statistical gate

---

## D-005 — Sample Design

**Development sample:**
2 January 2020 through 31 December 2025

**Locked final test sample:**
2 January 2026 through 30 June 2026

The locked test period must not be used for:
- strategy selection
- parameter tuning
- pair selection
- transaction-cost calibration
- acceptance-rule design

The project will use rolling walk-forward evaluation before the final test.

---

## D-006 — Positioning Assumptions

**Trend baseline:** Long or flat

**Trend long-short version:** Separate ablation

**Mean-reversion strategy:** One long leg and one short leg

**Maximum gross exposure:** 1.0 times capital

**Leverage:** Not used in the core experiment

**Pair target net exposure:** Approximately zero

---

## D-007 — Event-Bar Analysis

The required non-time-bar comparison will primarily use dollar bars.

If trade-level data is insufficient, volume bars will be used as the documented
fallback.

The benchmark comparison is:

15-minute time bars versus event bars.

Synthetic event bars created from coarse OHLCV data must not be represented as
genuine trade-level event bars.

---

## D-008 — Paper-Trading Safety

The system must:
- use paper mode only
- prevent live-order submission
- maintain a kill switch
- handle rejected and partially filled orders
- reconcile broker positions against internal positions
- record all order-state transitions

No real-money order is permitted in the core project.

---

## D-009 — Alpaca Access Test

The following tests passed on 19 July 2026:

- Paper authentication
- Active account status
- Account information retrieval
- Historical market-data script
- Latest market-data script
- Market clock retrieval
- SPY eligibility
- QQQ eligibility
- IWM eligibility

SPY, QQQ, and IWM were reported as:
- active
- tradable
- shortable
- easy to borrow
- fractionable

These properties must be checked again at execution time because broker
eligibility can change.

---

## D-010 — Tutorial III Governance

The project may begin before Tutorial III.

After Tutorial III:
1. Extract new instructions.
2. Compare them against the frozen V1.0 baseline.
3. Classify changes as mandatory, recommended, optional, or irrelevant.
4. Update the working plan only after documenting the delta.

## 2026-07-20 — Day 03 Data Architecture

### Decision

Use Alpaca as the primary provider for US equity bars, quotes, and trades. Store all normalized timestamps in UTC and convert to America/New_York only for session analysis and reporting.

### Storage

Raw and processed market data are stored locally as Parquet files. Raw datasets are immutable by default. Each stored dataset has a JSON provenance manifest and SHA-256 file hash.

### Bar hierarchy

- Raw research frequency: 1 minute
- Primary strategy frequency: 15 minutes
- Robustness frequencies: 30 and 60 minutes

The project will build strategy bars internally rather than relying exclusively on provider-generated aggregated bars.

### Microstructure

Historical Level-1 quotes and individual trades will support:

- quoted spread
- relative spread
- bid/ask imbalance
- microprice
- trade-flow analysis
- slippage and execution-cost modelling

Microstructure variables will initially act as filters and execution diagnostics rather than independent alpha strategies.

### Data governance

The final January–June 2026 test period remains locked. Day 03 pipeline testing used 15 December 2025, which belongs to the development period.

---

## D-014 — Cointegration and OU Feasibility Contract

Day 14 uses only the canonical 2020-01-02 through 2025-12-31
SPY, QQQ and IWM development dataset. The locked 2026 period remains
inaccessible.

**Candidate pairs and fixed orientation:**
- SPY / QQQ: SPY is Y; QQQ is X
- SPY / IWM: SPY is Y; IWM is X
- QQQ / IWM: QQQ is Y; IWM is X

The reverse orientation will not be tested for eligibility.

**Data contract:**
- Engle-Granger estimation uses log daily session-close prices derived from
  canonical 15-minute bars.
- Daily legs are aligned by exact session-date intersection.
- Intraday legs are aligned by exact timestamp and session intersection.
- Forward filling, interpolation and asynchronous matching are prohibited.

**Integration diagnostics:**
- ADF level test: intercept and deterministic trend, AIC lag selection.
- ADF first-difference test: intercept, AIC lag selection.
- A series is plausibly I(1) only when the 5% level test does not reject and
  the 5% first-difference test rejects.

**Cointegration inference:**
- Long-run regression: log(Y) = alpha + beta log(X) + residual.
- The regression contains an intercept and no deterministic trend.
- Statsmodels Engle-Granger residual-based inference is the eligibility test.
- A no-constant ADF test on fitted residuals is retained as a diagnostic only.
- Holm family-wise correction at 5% is applied across the three candidate
  Engle-Granger tests.

**Hedge-ratio and stability gates:**
- Alpha and beta must be finite.
- Beta must be positive and lie between 0.10 and 10.00.
- All four expanding-fold training betas must retain the full-sample sign.
- Maximum fold-beta deviation from the full-sample beta must not exceed 25%.
- At least three of four fixed-training-coefficient test residuals must reject
  a unit root at 5%.
- Fold test-residual ADF diagnostics include an intercept and use AIC lag
  selection because training coefficients do not guarantee zero-mean
  out-of-sample residuals.

**OU gate:**
- OU estimation is attempted only after cointegration and stability pass.
- The full-sample hedge ratio is applied to synchronized 15-minute log prices.
- AR(1) estimation uses within-session consecutive observations only.
- The discrete coefficient must satisfy 0 < phi < 1.
- Kappa, theta and sigma must be finite, with kappa and sigma positive.
- Half-life must be at least 1 and at most 130 fifteen-minute bars.

Pair eligibility requires every predeclared statistical and OU gate to pass.
Profitability, costs, thresholds, positions, ranking and winner selection are
not Day 14 eligibility criteria. A valid outcome is that no pair qualifies.
ECM estimation and all trading logic are deferred to later work.

---

## D-018 — Read-Only Alpaca Paper-Broker Boundary

**Decision:** Day 18 uses a broker-neutral read-only adapter bound to the exact
Alpaca paper endpoint `https://paper-api.alpaca.markets`.

The Day 18 public interface exposes account, market-clock and asset-information
requests only. It deliberately exposes no method to submit, replace, cancel, or
otherwise mutate an order, position, transfer, or account.

**Safety requirements:**

- project environment is `paper`;
- broker paper mode is required;
- live trading and Day 18 order submission are disabled;
- manual confirmation and the kill switch remain required;
- credential values are loaded from environment-backed local storage and are
  excluded from representations and artifacts;
- the canonical artifact excludes account identifiers and financial balances;
- no locked 2026 research data is read; and
- every provider order type and time-in-force capability remains unauthorized
  on Day 18.

**Canonical read-only result on 2 August 2026:**

- Alpaca paper endpoint verified;
- account status active and no trading/account/user-suspension block reported;
- market-clock response valid; market closed at the snapshot time;
- SPY, QQQ and IWM reported active, tradable, shortable, easy to borrow and
  fractionable;
- all frozen mechanical preflight gates passed; and
- zero orders were submitted.

Broker and asset state can change. The preflight must be rerun before any later
paper-order session. Day 18 success does not authorize Day 19 or later order
submission.
