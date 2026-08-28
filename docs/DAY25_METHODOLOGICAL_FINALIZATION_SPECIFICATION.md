# Day 25 Methodological Finalization Specification

## 1. Status and boundaries

- Specification version: `day25_methodological_finalization_v1`
- Frozen before any result produced under this protocol
- Research priority: CQF validity before repository presentation
- Development sample: 2020-01-02 through 2025-12-31 inclusive
- Locked final-test sample: 2026-01-02 through 2026-06-30; prohibited
- Strategy tuning, winner selection, and result suppression: prohibited
- Broker orders and campaign mutation: prohibited
- Commit and push: prohibited until separately requested

This protocol closes the two development-evidence gaps identified by the
Chunks 0-3 audit: the price-ratio positioning mismatch and the idealized trend
return timing. It also predeclares the representative event-time experiment.
All historical negative results and rejected hypotheses remain part of the
project.

## 2. Trend configurations

The audit evaluates exactly three fixed configurations:

1. price-ratio trend, short window 8, long window 32, neutral band 0.001,
   `long_short_neutral` positioning;
2. the same price-ratio rule with `long_flat` positioning; and
3. EMA/MACD with fast window 12, slow window 26, signal window 9, and
   normalized-histogram neutral band 0.0005.

The long-flat case is a predeclared comparison required by the original
decision record. It does not replace the implemented historical lineage and
cannot be promoted merely because its return is higher.

## 3. Executable timing convention

The final development convention is `next_bar_open_overnight_flat_v1`:

1. features and signal `z_t` use information through the close of bar `t`;
2. target `q_(t+1) = z_t` is entered at the next regular-session bar open;
3. within a session, `q_t` earns the simple return from `open_t` to
   `open_(t+1)`;
4. on the final bar of a session, `q_t` earns `open_t` to `close_t` and is
   forced flat at that close;
5. the next session begins flat before any new target is entered at its first
   observed open;
6. turnover includes position changes at each bar open and the forced exit at
   each session close; and
7. cost is charged per unit of absolute turnover, including direct reversals.

This convention is causal and removes unobserved overnight attribution. The
bar open and close remain backtest price proxies rather than guaranteed fills,
so 0, 1, 2.5, and 5 basis-point-per-turnover stresses are all reported.

The saved `close_to_close_one_row_lag_overnight_carry` results remain visible
as historical accounting evidence. They are not relabeled as executable.

## 4. Trend experiment matrix

The audit produces, without ranking:

- SPY 15-minute full-development results for all three configurations under
  the saved and final timing conventions at one basis point;
- four fixed annual test folds, 2022 through 2025, for all three
  configurations and all four costs under the final convention;
- full-development robustness for SPY, QQQ, and IWM at 15, 30, and 60 minutes
  for all three configurations at one basis point;
- the existing 36-point price-ratio parameter grid for the long-flat case at
  one basis point, reported as sensitivity rather than optimization; and
- batch-versus-sequential replay parity for positions, turnover, gross return,
  costs, and net return.

Training history may warm indicators for each walk-forward fold, but execution
state resets to flat at the first test row. Aggregate walk-forward statistics
are recomputed from the concatenated test rows only.

## 5. Representative event-time experiment

The predeclared source is SPY Alpaca IEX trade data for five complete regular
sessions selected by calendar rule before download: the fifteenth calendar day
of January, April, July, October, and December 2025, moved to the next NYSE
session only if the date is not a trading day. The intended dates are
2025-01-15, 2025-04-15, 2025-07-15, 2025-10-15, and 2025-12-15.

For each session:

- 15-minute time bars are the benchmark;
- dollar-bar thresholds are calibrated from that session's total notional to
  target the same number of bars as the benchmark;
- whole trades remain atomic and all trade count, volume, and notional must
  reconcile;
- a fixed price-ratio indicator uses 4/16 windows and a 0.001 band, with
  rolling state reset independently inside each session;
- comparison statistics are bar count, duration dispersion, activity
  dispersion, lag-one return autocorrelation, return skewness and excess
  kurtosis, signal availability, signal distribution, and one-event-ahead
  signal/return association; and
- results are descriptive across five predeclared sessions, not a trading
  profitability test and not a basis for selecting the primary bar type.

If provider access cannot supply the complete sample, the one-minute Day 5
sample remains an engineering smoke test and the final report must retain the
event-time evidence blocker. Coarse OHLCV bars must never be converted into
purported trade-level event bars.

### Pre-result design correction recorded after the first engineering run

The initial event-time implementation used 8/32 windows carried across the
five non-contiguous sessions. Because a regular session contains only 26
15-minute bars, a 32-bar long window cannot warm within a session; carrying it
across multi-month gaps creates an artificial state transition. That first
bundle is preserved as `day25_event_time_finalization_v1_invalid` and is
excluded from final claims. The corrected 4/16 windows above are the smallest
Day 7 grid pair with the same four-to-one long/short ratio, fit inside one
session, reset per session, and apply identically to time and dollar bars. The
correction is based only on data topology and was not chosen by comparing
returns or correlations.

## 6. Acceptance criteria

- no row on or after 2026-01-01 is read by the development runners;
- exact configurations, folds, cost scenarios, symbols, frequencies, and
  event-session dates are enforced in code;
- all tables are written by exact allow-list to atomic deterministic bundles
  with non-self SHA-256 manifests;
- saved-convention results reconcile to the existing baseline within declared
  tolerances;
- forced-flat positions and turnover reconcile at every session boundary;
- sequential replay equals vectorized output exactly or within `1e-12` for
  floating-point returns;
- negative, zero, and positive results are retained without ranking;
- focused tests, the full suite, package build, and `git diff --check` pass;
  and
- the technical report and Day 25 staging bundle are regenerated only from
  verified source artifacts.
