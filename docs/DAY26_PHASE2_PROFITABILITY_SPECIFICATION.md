# Day 26 Phase II Profitability Specification

## 1. Status and evidence boundary

- Specification version: `day26_phase2_profitability_v1`
- Status: frozen before any Phase II return was calculated
- Primary purpose: test two theory-led profitability hypotheses without
  weakening the completed CQF evidence boundary
- Development data: canonical Alpaca SIP 15-minute bars for SPY, QQQ, and IWM,
  2020-01-02 through 2025-12-31 inclusive
- Chronological evaluation: expanding-history annual test folds for 2022,
  2023, 2024, and 2025
- Consumed final test: 2026-01-02 through 2026-06-30; prohibited for Phase II
  design, tuning, cost calibration, instrument selection, or evaluation
- Untouched post-lock holdout: unavailable in the local repository at freeze
  time
- Broker access, orders, campaign mutation, leverage, commit, and push:
  prohibited

All prior positive, negative, and rejected results remain permanent baselines.
This experiment may establish a development-period change in turnover, gross
return, or net return. It cannot establish improved out-of-sample
profitability because no untouched future holdout is locally available.

## 2. Research questions and declared trials

Exactly two Phase II configurations are evaluated. Both are retained and
reported regardless of their result. There is no grid search, ranking, winner
selection, post-result threshold change, portfolio-weight optimization, or
promotion decision.

### Trial 1: persistent-hysteresis price-ratio long-flat

The retained baseline is the SPY 8/32 simple-moving-average price ratio with a
0.001 entry band and long-flat positioning. The Phase II target uses the same
ratio and risk limit, but enters long only after four consecutive completed
15-minute bars have ratios strictly above 1.001. Once long, the target remains
long until the ratio is at or below 1.0005. The four-bar confirmation is one
hour and the exit band is half the entry band. These values are fixed as one
compound turnover-control hypothesis; they are not selected from alternatives.

The final causal timing convention remains
`next_bar_open_overnight_flat_v1`: a target observed at bar close can first be
entered at the next bar open, positions are forced flat at every session close,
and costs apply to every unit of open, reversal, and forced-close turnover.

### Trial 2: cost-margin-gated slow OU/VWAP

The retained baseline is the exact `ou_vwap_slow` calibration: 52-bar
volume-weighted reference, 208-transition OU and variance-ratio windows,
variance-ratio lag 4 and threshold 0.95, entry z-score 2.25, exit z-score 0.25,
half-life range 1-39 bars, maximum holding period 39 bars, one-bar execution
delay, and forced session-flat state.

The Phase II target adds one ex-ante gate and changes nothing else. At an entry
signal, let `x_t` be the log price/reference residual, `mu_t` the rolling OU
equilibrium, `phi_t` the rolling AR(1) coefficient, and `H_t` the smaller of
39 bars and the number of investable bars remaining before the forced session
close. Define the expected residual-convergence proxy

\[
E_t = |x_t-\mu_t|\left(1-\phi_t^{H_t}\right).
\]

An entry is permitted only when `H_t >= 1` and `E_t >= 0.0010`. The ten-basis-
point threshold equals the two-turnover round-trip cost at the predeclared
five-basis-point-per-turnover stress. It is a model-based residual proxy, not a
guaranteed asset return or fill. The gate applies only to new entries; the
baseline exit, invalid-regime, maximum-holding, reset, and session-close rules
remain unchanged.

## 3. Comparators, costs, and chronology

Four configurations are reported in fixed order:

1. `price_ratio_long_flat_baseline`;
2. `price_ratio_persistent_hysteresis_phase2`;
3. `ou_vwap_slow_baseline`;
4. `ou_vwap_slow_cost_margin_phase2`.

Trend is evaluated on SPY, matching the retained trend baseline. OU/VWAP is
reported for SPY, QQQ, IWM, and their contemporaneous equal-weight session
return series, matching Day 17. Cost stresses are fixed at 0, 1, 2, and 5 basis
points per unit turnover. Signals and positions are held constant across cost
stresses.

For every annual fold, pre-test history may warm rolling indicators, but
execution state resets to flat on the first test row. Aggregate 2022-2025
statistics are recomputed from concatenated, non-overlapping test observations
only. No data from 2026 or later may be read.

## 4. Required evidence

For each eligible configuration, series, fold, and cost, report cumulative and
annualized return, annualized volatility, Sharpe ratio, maximum drawdown,
turnover, trade count, and long, short, and flat exposure. Also report:

- gross-versus-net aggregate results and absolute cost drag;
- fold-by-fold sign and stability;
- turnover and entry-count change against the matching retained baseline;
- the log-linear break-even cost approximation per unit turnover, where gross
  return and turnover are both positive;
- Newey-West/HAC t-statistic with five lags for aggregate session returns;
- deterministic moving-block-bootstrap 95% intervals for mean session return
  and annualized Sharpe, using 2,000 replications, five-session blocks, and a
  declared seed of 2601; and
- exact batch-versus-sequential or independent-accounting reconciliation for
  the new stateful rules.

The source-data audit must record row count, symbol coverage, timestamp range,
duplicate symbol-timestamps, null count, price/volume domain checks, source,
feed, and the SHA-256 hash of the canonical development file.

## 5. Interpretation rules

A Phase II development result may be described as improved only with the
qualifier `development-period` and only when the new configuration is compared
with its matching baseline on identical folds, cost, timing, symbol scope, and
risk limits. A higher return alone is insufficient: turnover, drawdown,
exposure, fold stability, and uncertainty must be shown beside it.

The report must not say that profitability has been established, that a model
is ready for deployment, or that the Phase II configuration passed an
out-of-sample profitability gate. Such a claim requires a separately frozen
design evaluated once on a new untouched future holdout under empirically
grounded execution cost, with all trials disclosed.

## 6. Deterministic artifact contract

The exact Day 26 bundle contains only:

1. `data_quality.json`;
2. `aggregate_performance.csv`;
3. `fold_performance.csv`;
4. `cost_sensitivity.csv`;
5. `comparison.csv`;
6. `inference.csv`;
7. `session_returns.csv`;
8. `methodology.json`;
9. `report.md`; and
10. `manifest.json`.

The manifest hashes every non-manifest file. Output is written through sibling
staging and atomic replacement. Existing frozen Day 17, Day 24, Day 25, and
locked-final-test bundles are never edited by this runner.

## 7. Acceptance criteria

- the implementation rejects any row on or after 2026-01-01;
- the two Phase II configurations and two retained baselines are exact and in
  the declared order;
- all four cost stresses and all four chronological folds are present;
- state resets, one-bar delays, session-flat behavior, and turnover costs
  reconcile independently;
- every declared trial and every result is reported without ranking;
- uncertainty intervals and undefined statistics remain explicit;
- the deterministic artifact bundle replays byte-for-byte;
- focused and full repository tests pass;
- `git diff --check` passes; and
- no broker action, campaign mutation, locked-data access, commit, or push
  occurs.
