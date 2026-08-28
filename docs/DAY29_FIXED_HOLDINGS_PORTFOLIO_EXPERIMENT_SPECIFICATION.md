# Day 29 Fixed-Holdings Portfolio Experiment Specification

Version: `day29_fixed_holdings_portfolio_experiment_v1`  
Accounting version: `fixed_holdings_fold_rebalance_v1`

## Purpose and boundary

Day 29 measures, on development data only, the empirical difference between
the frozen Day 16 daily constant-mix accounting convention and the corrected
fixed-holdings-with-fold-rebalancing convention. It does not select or promote
an allocation rule, tune a target estimator, or constitute final validation.

The canonical input is the exact Day 16 development source
`data/processed/bars/spy_qqq_iwm_15min_2020-01-02_2025-12-31_sip_v3_development_canonical.parquet`.
The return panel is rebuilt through
`run_strategy_diversification(validated_bars).copy_session_return_panel()`.
The Day 25 causal return panel is authenticated as preservation evidence but is
not substituted for the original Day 16 analytical input.

No observation on or after 2026-01-01, locked-final runner or artifact, broker,
network, report generator, notebook, chart, or parameter-selection workflow is
in scope.

## Frozen invariants

Both methods use the same:

- 2020-01-02 through 2025-12-31 development return panel;
- six sleeves and their exact order;
- four chronological expanding folds (`wf_2022` through `wf_2025`);
- equal-weight, inverse-volatility, and Ledoit-Wolf constrained
  minimum-variance targets;
- training observations, covariance matrices, shrinkage coefficients,
  optimizer inputs and constraints;
- long-only, fully invested 35% target cap;
- 252-session annualization and one-basis-point allocation cost rate.

Target weights, covariance estimates, shrinkage estimates, and fold signatures
must be exactly equal between methods. The rebuilt historical path must also
match the saved Day 16 machine-readable evidence. Any mismatch fails closed.

## Accounting methods

For the historical method in fold (f), the target weights remain constant:

\[
r_{H,t}=w_f^\top r_t.
\]

Historical fold-boundary turnover remains the frozen Day 16 quantity:

\[
TO_{H,f}=\sum_i\left|w_{f,i}-w_{f-1,i}\right|,
\]

with a zero vector before the first fold.

For corrected fixed holdings, the pre-return weights drift with prior sleeve
returns:

\[
r_{C,t}=(w_t^-)^\top r_t,
\qquad
w_t^+=\frac{w_t^-\odot(1+r_t)}{1+r_{C,t}},
\qquad
w_{t+1}^-=w_t^+.
\]

At the next fold boundary:

\[
TO_{C,f}=\sum_i\left|w_{f,i}^{\mathrm{target}}
-w_{f-1,i}^{\mathrm{ending}}\right|.
\]

For either method (M), the external cost deduction is

\[
cost_{M,f}=TO_{M,f}\times10^{-4},
\]

and is charged only on the first test session of the fold. It does not alter
the relative corrected sleeve weights.

The corrected pre-cost path must satisfy in every fold and rule:

\[
\prod_t(1+r_{C,t})
=\sum_i w_{f,i}^{\mathrm{target}}\prod_t(1+r_{i,t})
\]

within `1e-12` relative or absolute tolerance. This identity is not asserted
for the cost-adjusted path.

## Evidence tables

The non-overwriting bundle is written only to
`artifacts/day29_fixed_holdings_portfolio_experiment/` and contains:

- `source_and_method_metadata.json`;
- `target_and_covariance_invariance.csv`;
- `fold_performance_comparison.csv`;
- `aggregate_performance_comparison.csv`;
- `fold_turnover_comparison.csv`;
- `ending_weight_drift.csv`;
- `corrected_weight_path.csv`;
- `wealth_identity_checks.csv`;
- `portfolio_return_comparison.csv`;
- `manifest.json`.

`portfolio_return_comparison.csv` is the one additional machine-readable file.
It exposes session-level gross returns, net returns, and first-session cost
deductions, making the turnover/cost decomposition auditable without a report.

All differences use `corrected minus historical`. Gross-return differences are
the return-accounting effect. Turnover differences are reported separately.
The cost effect is the aggregate net-return difference minus the aggregate
gross-return difference; because costs compound with returns, it need not equal
the negative difference in nominal cost deductions.

Drift above the 35% target cap is retained and labelled as an expected result
of fixed holdings. The cap is a fold-entry target constraint, not an intrafold
clipping rule.

## Provenance and failure policy

The runner authenticates the exact required Day 16 and Day 25 machine-readable
comparators against their manifests, records their file paths and SHA-256
hashes, and requires identical hashes after artifact writing. The source file
path, source hash, deterministic panel hash, date range, row count, sleeve
order, frequency, folds, cost, versions, and safety declarations are recorded.

An existing Day 29 directory is never overwritten. Invalid dates, schemas,
ordering, duplicate observations, incomplete sleeves, nonfinite returns,
returns at or below -100%, invariant failures, historical comparator mismatch,
wealth-identity failure, or comparator mutation aborts execution. A failure
after directory creation leaves partial evidence in place for diagnosis; it is
not deleted or overwritten.

## Interpretation

Results are development evidence about accounting consequences only. A metric
improvement does not establish superiority, justify promotion, or change the
frozen allocation targets. Economic materiality must be judged separately for
the return-accounting, turnover, and compounded cost effects, and must be
qualified by sampling uncertainty not estimated by this experiment.
