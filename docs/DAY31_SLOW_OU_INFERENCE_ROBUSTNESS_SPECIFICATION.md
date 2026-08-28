# Day 31 Slow OU Inference Robustness Specification

Version: `day31_slow_ou_inference_robustness_v1`

## Purpose and claim boundary

Day 31 is a development-only sensitivity analysis of the corrected Day 28 slow
OU/VWAP equal-weight return path. It does not tune parameters, select a cost,
select an inference convention, rank a strategy, or promote the candidate. The
slow configuration, SPY/QQQ/IWM universe, 2020--2025 canonical input, 2022--2025
walk-forward tests, causal next-open/session-close execution, primary one-basis-
point cost, HAC(5), 2,000-replication circular bootstrap, and 252-session
annualization remain frozen.

The source must be
`data/processed/bars/spy_qqq_iwm_15min_2020-01-02_2025-12-31_sip_v3_development_canonical.parquet`
with SHA-256
`30212cd6414e506fe397df6eae23455214b40c26099096d3f8fe9f3d2c29c3f2`.
No timestamp on or after 2026-01-01 is admissible.

## Immutable Day 28 reproduction gate

All Day 28 files are authenticated against the saved Day 28 manifest and hashed
before and after artifact generation. Day 31 reconstructs the slow bar-level
execution path using the frozen Day 28 strategy and causal execution functions.
The reconstructed one-basis-point equal-weight session path must agree with the
frozen engine's session path, and its aggregate, inference, PSR, DSR, and primary
bootstrap values must reproduce the saved machine-readable Day 28 evidence
within a declared numerical tolerance. Sensitivities are rejected if this gate
fails.

## Transaction-cost sensitivity

For every executed bar (t), with frozen gross return (r_t^g), actual turnover
(mathrm{TO}_t), and cost (c\in\{0,1,2,5\}) basis points,

\[
r_t^n(c)=r_t^g-\mathrm{TO}_t\frac{c}{10{,}000}.
\]

Costs are not a constant daily fee. Bar net returns are compounded within each
symbol/session, and the three resulting symbol session returns are equally
weighted exactly as in Day 28. Total disclosed cost is

\[
\mathrm{TotalCost}(c)
=\frac{1}{3}\sum_t\mathrm{TO}_t\frac{c}{10{,}000}.
\]

The primary specification remains one basis point regardless of sensitivity
outcomes.

## HAC lag sensitivity

For primary one-basis-point session returns `r_t`, `T` observations, and sample
mean `r_bar = T^{-1} sum_t r_t`, the implemented autocovariance convention is

\[
\widehat\gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}(r_t-\bar r)(r_{t-\ell}-\bar r).
\]

For `L` in `{1, 5, 10, 20}`, Bartlett weights give

\[
\widehat\Omega_L
=\widehat\gamma_0
+2\sum_{\ell=1}^{L}
\left(1-\frac{\ell}{L+1}\right)\widehat\gamma_\ell,
\qquad
\mathrm{SE}_{HAC}=\sqrt{\frac{\widehat\Omega_L}{T}},
\qquad
t_{HAC}=\frac{\bar r}{\mathrm{SE}_{HAC}}.
\]

A zero or negative long-run variance fails closed. HAC depends on the lag choice
and asymptotic approximation. Overlapping rolling OU estimation windows create
dependence that a single lag convention cannot eliminate. HAC(5) remains primary.

## Circular block bootstrap

For block lengths (b\in\{5,10,20,40\}), block starts are sampled uniformly
with the frozen effective Day 28 slow/equal-weight seed. Each block uses circular
indices

\[
(s,s+1,\ldots,s+b-1)\pmod T,
\]

and enough blocks are concatenated to obtain (T) observations. The existing
2,000 replication count and percentile 2.5%/97.5% interval convention are used
for the mean and annualized Sharpe. Block-bootstrap conclusions depend on block
length and stationarity assumptions. No favourable block length is selected;
length five remains primary.

## Leave-one-year-out concentration

For each year in 2022--2025, the complete year is removed and the remaining
primary returns retain chronological order. Cumulative return, annualized
volatility, annualized Sharpe, maximum drawdown, HAC(5), and sample size are
reported. These are dependent subsamples, not independent replications, a new
backtest, or a selection rule.

## PSR and DSR scope

The Probabilistic Sharpe Ratio uses the frozen sample skewness and non-Fisher
kurtosis convention to adjust the Sharpe sampling approximation for
non-normality. It does not correct an undocumented research process.

The reproduced Deflated Sharpe Ratio declares exactly the three pre-existing OU
trials: fast, base, and slow. It excludes trend, EMA/MACD, portfolio, and other
Axiom research choices. Consequently it is a local diagnostic that likely
understates total research multiplicity. No effective independent-trial count
is invented and no globally corrected DSR is claimed. A defensible DSR depends
on a defensible trial universe.

## Limitations

- Rolling-window overlap and persistent positions induce serial dependence.
- HAC results depend on lag selection and large-sample reasoning.
- Block-bootstrap intervals depend on block length and approximate stationarity.
- Leave-one-year-out rows share most observations and are dependent.
- PSR addresses selected non-normal moments, not undisclosed researcher degrees
  of freedom.
- The three-configuration DSR is narrower than the complete Axiom search space.
- Sensitivity analysis cannot convert development evidence into holdout evidence.
- Passing reproduction and numerical tests does not validate profitability.

## Artifact contract

Only these machine-readable files are allowed under
`artifacts/day31_slow_ou_inference_robustness/`:

1. `source_and_method_metadata.json`
2. `primary_day28_reproduction.csv`
3. `transaction_cost_sensitivity.csv`
4. `hac_lag_sensitivity.csv`
5. `block_bootstrap_sensitivity.csv`
6. `leave_one_year_out.csv`
7. `psr_dsr_disclosure.csv`
8. `manifest.json`

The directory is created atomically and is never overwritten. The manifest
hashes every other artifact and records the canonical source, data range,
primary and sensitivity conventions, Day 28 comparator paths and hashes, and
explicit negative declarations for holdout, report, notebook, chart, broker,
network, parameter-selection, promotion, commit, and push actions.
