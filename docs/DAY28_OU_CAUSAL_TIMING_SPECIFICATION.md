# Day 28 OU Causal Timing Specification

## Version and claim boundary

- Specification version: `day28_ou_causal_timing_v1`.
- Corrected timing label: `corrected_next_open_overnight_flat`.
- Historical comparator label: `historical_close_to_close`.
- This is development-period evidence only. It does not select, rank, promote, or
  validate a configuration.
- The consumed 2026 holdout, report generators, reports, notebooks, charts,
  brokers, and live-operation boundaries are outside scope.

## Frozen inputs

The only empirical input is the canonical 15-minute SIP development dataset:

`data/processed/bars/spy_qqq_iwm_15min_2020-01-02_2025-12-31_sip_v3_development_canonical.parquet`

The calculation must fail closed unless the input:

- starts no earlier than 2020-01-02 and ends no later than 2025-12-31;
- contains no timestamp or session in 2026;
- contains exactly SPY, QQQ, and IWM;
- contains only 15-minute within-session intervals;
- retains the frozen 2022-2025 chronological expanding folds; and
- retains `ou_vwap_fast`, `ou_vwap_base`, and `ou_vwap_slow` exactly once.

All Day 17 parameters, reported series, cost stresses, HAC lag, block length,
bootstrap seed and replications, and DSR trial count remain frozen. The Phase II
slow baseline and cost-margin-gated candidate also retain their frozen signal
construction and inference settings.

## Timing intervention

Only performance attribution changes. A signal observed at bar t close becomes
the target at bar t+1 open. A non-close row earns `open[t+1] / open[t] - 1`; a
session-close row earns `close[t] / open[t] - 1`. Entry, reversal, ordinary exit,
and same-row close liquidation turnover are charged, and ending position is zero
at every session close. The implementation uses
`apply_causal_next_open_overnight_flat` through the accepted Phase 1 engines.

Raw VWAP references, OU estimates, diagnostics, thresholds, raw signals,
configurations, and fold definitions are not changed.

## Immutable historical evidence

Historical results are read from `artifacts/day17` and `artifacts/day26`. The old
close-to-close convention is not reconstructed. Every comparator file used is
checked against its original manifest and SHA-256 snapshot. The snapshot is
checked again after Day 28 writing.

## Output contract

The only output directory is `artifacts/day28_ou_causal_timing/`, created once
without overwrite. It contains exactly:

1. `corrected_fold_performance.csv`
2. `corrected_aggregate_performance.csv`
3. `corrected_cost_sensitivity.csv`
4. `corrected_inference_results.csv`
5. `historical_vs_corrected_timing.csv`
6. `corrected_phase2_ou_comparison.csv`
7. `annual_concentration_diagnostics.csv`
8. `manifest.json`

The aggregate comparison reports cumulative gross return at zero cost and
cumulative net return, annualized volatility, Sharpe ratio, maximum drawdown,
turnover, and non-zero net sessions at one basis point per unit turnover. Changes
are arithmetic corrected-minus-historical differences, not selection scores.

The corrected slow equal-weight inference row reuses the Day 17 implementation
for the naive t-statistic, HAC(5), circular block-bootstrap intervals, PSR, and
DSR. Annual concentration is computed from the corrected one-basis-point slow
equal-weight session series. Absolute log-return shares and their HHI describe
calendar-year concentration; they do not establish independent evidence.

The Phase II artifact reports the frozen baseline and cost-margin candidate for
all previously reported OU series. `signal_entry_count` is the existing
execution-statistics trade count. `execution_path_difference_sessions` counts
sessions where baseline and candidate gross return, one-basis-point net return,
or turnover differ. These are observable execution-path effects, not a claim that
every raw signal row was independently serialized.

## Interpretation boundary

A positive corrected development result remains an unvalidated candidate result.
Overlapping rolling OU estimates, correlated symbols, multiple research choices,
cost/model risk, and development-period reuse remain material. Day 28 preserves
negative and inconclusive results and contains no winner, rank, promotion, or
replacement field.
