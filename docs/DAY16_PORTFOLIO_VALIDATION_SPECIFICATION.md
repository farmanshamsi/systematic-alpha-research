# Day 16 Portfolio Allocation and Economic Validation Specification

## 1. Status and authority

- Specification version: `day16_portfolio_validation_v1`
- Status: frozen for implementation after repository audit
- Scope: development-only portfolio allocation and chronological economic validation
- Permitted data dates: 2020-01-02 through 2025-12-31
- Locked final-test dates: 2026-01-02 through 2026-06-30

This document is the Day 16 implementation contract. Code, tests, reports,
and artifacts must follow it exactly. Changes require an explicit documented
revision before implementation.

## 2. Research question

Do three predeclared, long-only allocation rules produce mechanically valid
and economically interpretable portfolios from the six frozen Day 15 strategy
sleeves when weights are estimated using training data only and evaluated on
the existing chronological test folds?

Day 16 measures portfolio behaviour. It does not select a preferred rule,
claim profitability, or authorize deployment.

## 3. Inherited evidence and exclusions

Day 16 inherits without modification:

- the six Day 15 sleeves and their exact order;
- the Day 10 strategy configurations;
- the Day 11 expanding walk-forward folds;
- the Day 15 exact common-session return panel;
- the existing one-basis-point strategy cost embedded in each sleeve's
  `net_strategy_return`;
- position timing `position[t] = signal[t-1]`;
- the development-only and locked-period boundaries.

Day 14 found zero eligible cointegration pairs. No mean-reversion sleeve may
be added, substituted, or manufactured for Day 16. The Day 16 universe is
therefore the six frozen trend sleeves only.

Day 16 excludes:

- strategy or sleeve removal;
- parameter tuning;
- allocation-rule ranking or winner selection;
- return, Sharpe, drawdown, VaR, ES, or profitability acceptance gates;
- leverage, short allocation weights, or borrowing;
- cost-aware optimization;
- use of the locked 2026 period;
- paper or live order submission.

## 4. Frozen sleeve order

1. `trend_ratio_spy`
2. `trend_ratio_qqq`
3. `trend_ratio_iwm`
4. `ema_macd_spy`
5. `ema_macd_qqq`
6. `ema_macd_iwm`

Every matrix, vector, table, manifest entry, and artifact must retain this
order. Alphabetical reordering is prohibited.

## 5. Input contract

The analytical input is the validated Day 15 session-return panel:

- index: UTC-normalized `session_date`;
- columns: the six sleeves in the frozen order;
- values: finite simple session returns strictly greater than `-1.0`;
- calendar: exact common dates with no filling or interpolation;
- minimum variance: strictly greater than the inherited Day 15 tolerance;
- permitted dates: development period only.

Canonical execution must rebuild this panel through the Day 15 analysis from
the selected canonical development dataset. Committed Day 15 CSV artifacts
must not be treated as a substitute analytical dataset.

Synthetic panels may be used in unit tests. Implementation-only tests must not
load canonical data or any 2026 data.

## 6. Frozen walk-forward protocol

Use the four inherited expanding folds:

| Fold | Training interval | Test interval |
|---|---|---|
| `wf_2022` | 2020-01-02 to 2022-01-01 exclusive | 2022-01-01 to 2023-01-01 exclusive |
| `wf_2023` | 2020-01-02 to 2023-01-01 exclusive | 2023-01-01 to 2024-01-01 exclusive |
| `wf_2024` | 2020-01-02 to 2024-01-01 exclusive | 2024-01-01 to 2025-01-01 exclusive |
| `wf_2025` | 2020-01-02 to 2025-01-01 exclusive | 2025-01-01 to 2026-01-01 exclusive |

For the canonical development panel, the frozen training/test session counts
are `505/251`, `756/250`, `1006/252`, and `1258/250` in fold order. Synthetic
tests may use smaller panels while preserving the same chronological logic.

For each allocation rule and fold:

1. estimate weights from that fold's training panel only;
2. freeze the resulting vector before reading test returns;
3. apply the vector unchanged throughout the test interval;
4. retain every test session in chronological order;
5. concatenate the four non-overlapping test intervals for aggregate
   out-of-sample reporting.

Monthly, quarterly, intrafold, or result-triggered rebalancing is prohibited.

## 7. Frozen allocation rules

The exact rule order is:

1. `equal_weight`
2. `inverse_volatility`
3. `constrained_minimum_variance`

All rules must satisfy:

- six finite weights;
- `sum(weights) = 1.0` within `1e-12`;
- `0.0 <= weight <= 0.35` within `1e-12`;
- gross allocation equals `1.0` within `1e-12`;
- no leverage and no short allocation weights.

### 7.1 Equal weight

Set every sleeve weight to exactly `1/6`. This rule uses no estimated return,
volatility, or covariance input.

### 7.2 Inverse volatility

For each training sleeve, calculate the ordinary sample standard deviation
with `ddof=1`. Define the raw score as the reciprocal standard deviation.
Normalize raw scores to sum to one, then apply the `0.35` cap through a
deterministic iterative water-filling procedure:

1. cap every violating weight at `0.35`;
2. redistribute the remaining mass across uncapped sleeves in proportion to
   their original raw scores;
3. repeat until no cap is violated;
4. validate the final vector against all constraints.

Zero, near-zero, missing, or non-finite training volatility is a hard failure.

### 7.3 Constrained minimum variance

Fit `sklearn.covariance.LedoitWolf(assume_centered=False)` to the fold's
training return matrix in the frozen sleeve order. Record the shrinkage
coefficient and the complete finite symmetric covariance matrix.

Solve:

```text
minimize      w' Sigma w
subject to    sum(w) = 1
              0 <= w_i <= 0.35
```

Use SciPy SLSQP with:

- initial vector: exact equal weights;
- analytical objective gradient `2 * Sigma * w`;
- `ftol = 1e-12`;
- `maxiter = 10_000`;
- no random initialization.

The solver result must be finite, successful, and satisfy every constraint
within `1e-10`. Failure is a hard Day 16 analysis error. Substitution of equal
or inverse-volatility weights is prohibited because it would silently change
the rule being evaluated.

Cost-aware optimization and expected-return inputs are outside Day 16.

## 8. Portfolio returns and allocation costs

Sleeve session returns already include frozen strategy-level transaction
costs. Those costs must not be added a second time.

For rule `a`, fold `f`, and test session `t`, the pre-allocation-cost return is:

```text
r_portfolio[a,f,t] = sum_i weight[a,f,i] * sleeve_return[i,t]
```

Allocation turnover is charged only on the first test session of each fold:

```text
turnover[a,f] = sum_i abs(weight[a,f,i] - previous_weight[a,i])
allocation_cost[a,f] = turnover[a,f] * 1.0 / 10_000
```

For `wf_2022`, `previous_weight` is the six-element zero vector. For each
later fold it is the same rule's weight vector from the preceding fold.

Subtract `allocation_cost` from the first test-session return only. All later
test sessions in the fold have zero allocation cost. This convention is
separate from the strategy costs already embedded in sleeve returns. Every
reported portfolio return and performance metric must use this
post-allocation-cost net return.

## 9. Frozen economic metrics

Calculate metrics independently for every rule/fold and for each rule's
concatenated 2022-2025 out-of-sample series. Use 252 sessions per year and a
zero risk-free rate.

Required metrics:

- observations;
- cumulative return;
- annualized return;
- annualized volatility using `ddof=1`;
- Sharpe ratio;
- maximum drawdown;
- historical 95% Value at Risk, reported as a non-negative loss;
- historical 95% Expected Shortfall, reported as a non-negative loss;
- allocation turnover;
- allocation cost;
- maximum sleeve weight;
- Herfindahl concentration `sum(w_i^2)`;
- effective sleeve count `1 / sum(w_i^2)`.

Historical VaR uses the linear 5% return quantile. Historical ES is the
negative mean of observations at or below that quantile. Loss metrics are
floored at zero.

All finite outcomes, including negative returns or Sharpe ratios, must be
retained without reinterpretation.

## 10. Mechanical validation gates

Day 16 may report only mechanical completeness gates:

- exact frozen input schema and sleeve order;
- development-only dates;
- no missing or non-finite input returns;
- train/test separation for all folds;
- weights estimated without test rows;
- finite weights and covariance inputs;
- weight sum, bounds, gross exposure, and no-leverage constraints;
- successful minimum-variance solver status;
- finite portfolio returns strictly greater than `-1.0`;
- complete fold and aggregate row counts;
- deterministic artifact serialization and hashes.

The final mechanical field is `evaluation_complete`. It must not depend on
return, Sharpe, drawdown, VaR, ES, profitability, or relative performance.

## 11. Frozen result schemas

### 11.1 `allocation_weights`

One row per fold, rule, and sleeve: `4 * 3 * 6 = 72` rows.

```text
fold_id, allocation_rule, sleeve_id, sleeve_order, training_sessions,
weight, weight_sum, maximum_weight, gross_weight, constraint_valid
```

### 11.2 `allocation_diagnostics`

One row per fold and rule: `4 * 3 = 12` rows.

```text
fold_id, allocation_rule, training_sessions, test_sessions,
covariance_estimator, shrinkage_coefficient, solver_status,
allocation_turnover, allocation_cost, weight_sum, minimum_weight,
maximum_weight, gross_weight, herfindahl_concentration,
effective_sleeve_count, constraint_valid
```

Non-applicable covariance, shrinkage, and solver fields must use explicit
neutral strings or empty CSV fields; JSON artifacts may not contain NaN or
Infinity.

### 11.3 `fold_portfolio_performance`

One row per fold and rule: `4 * 3 = 12` rows.

```text
fold_id, allocation_rule, observations, start_session, end_session,
cumulative_return, annualized_return, annualized_volatility, sharpe_ratio,
maximum_drawdown, historical_var_95, historical_es_95,
allocation_turnover, allocation_cost
```

### 11.4 `aggregate_portfolio_performance`

One row per rule: `3` rows.

```text
allocation_rule, observations, start_session, end_session,
cumulative_return, annualized_return, annualized_volatility, sharpe_ratio,
maximum_drawdown, historical_var_95, historical_es_95,
total_allocation_turnover, total_allocation_cost
```

### 11.5 `portfolio_return_panel`

One row per out-of-sample session. The exact expected count is the sum of the
four inherited test-fold session counts. For the canonical development panel,
this is `251 + 250 + 252 + 250 = 1003` rows. Columns:

```text
session_date, fold_id, equal_weight,
inverse_volatility, constrained_minimum_variance
```

## 12. Frozen artifact bundle

The writer must emit exactly seven files in this order:

1. `allocation_weights.csv`
2. `allocation_diagnostics.csv`
3. `fold_portfolio_performance.csv`
4. `aggregate_portfolio_performance.csv`
5. `portfolio_return_panel.csv`
6. `report.md`
7. `manifest.json`

Requirements:

- deterministic CSV, Markdown, and strict JSON bytes;
- exactly one final newline in every text artifact;
- SHA-256 hashes for every non-manifest artifact;
- no manifest self-hash;
- sibling staging and atomic directory replacement with rollback;
- overwrite protection restricted to an exact `artifacts/day16` destination
  basename of `day16`;
- exact final allow-list verification;
- no timestamps or environment-dependent absolute paths in artifacts.

## 13. Report interpretation contract

The report must:

- distinguish statistical diversification from economic performance;
- state that allocation rules were predeclared;
- show all three rules in frozen order;
- describe training-only estimation and fixed test-fold weights;
- report every fold and aggregate result;
- state that negative results are valid;
- avoid `best`, `winner`, `optimal strategy`, or deployment language;
- disclose that constrained minimum variance is an optimization rule but that
  no rule was selected using realized performance;
- state that the zero-pair Day 14 result is retained;
- state that the locked 2026 period was not accessed.

## 14. Testing and acceptance

Implementation acceptance requires:

- synthetic tests for every input, weight, solver, metric, cost, schema,
  ordering, writer, rollback, and locked-period guardrail;
- tests proving test-return mutation cannot change training-derived weights;
- tests proving fold weights remain constant inside each test fold;
- tests proving non-day16 overwrite destinations are rejected before writes;
- focused Day 16 tests passing;
- the complete repository test suite passing;
- compilation of all new modules and runners;
- `git diff --check` passing;
- exact changed-file and repository-status audits;
- no commit or push until implementation and artifact audits pass.

The canonical development run is a separate, explicit step after synthetic
implementation validation. It may use only the selected 2020-2025 canonical
dataset and must never access the locked 2026 period.

## 15. Post-Day 16 boundary

Day 16 does not authorize final allocation selection or paper execution.
After Day 16, the project may separately address reproducible data rebuilding,
submission packaging, and paper-execution integration.
