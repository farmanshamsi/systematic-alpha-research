# Day 17 Reversion Strategy and Statistical Inference Specification

## 1. Status and purpose

This document freezes the Day 17 development-only research contract before
canonical performance is inspected. Day 17 supplies the non-trivial reversion
strategy required by the Algorithmic Trading project brief and attaches formal
uncertainty estimates to its economic results.

Day 17 is an evaluation, not a strategy-selection exercise. It does not rank
calibrations, choose a winner, authorize paper trading, or access locked 2026
data.

## 2. Research question

Does a causal, intraday mean-reversion rule based on a volume-weighted price
residual, rolling Ornstein-Uhlenbeck diagnostics, and a variance-ratio regime
gate retain economically meaningful net performance across predeclared
chronological development folds and transaction-cost assumptions?

## 3. Data and leakage boundary

- Dataset: canonical Alpaca SIP 15-minute bars for SPY, QQQ, and IWM.
- Development interval: 2020-01-02 through 2025-12-31.
- Locked interval: timestamps on or after 2026-01-02 are rejected.
- Four expanding-history test folds: calendar years 2022, 2023, 2024, and
  2025, reusing the Day 11 boundaries.
- Rolling indicators may use observations preceding a test fold as history.
- Position and holding-period state reset to flat at every test boundary.
- A signal calculated from completed bar t can first affect position t+1.
- Positions are forced flat across session boundaries; overnight returns are
  not attributed to the strategy.

## 4. Strategy definition

For each symbol independently, define the rolling volume-weighted reference

\[
R_t = \frac{\sum_{i=t-n+1}^{t} V_i\,VWAP_i}
           {\sum_{i=t-n+1}^{t} V_i}
\]

and transformed residual

\[
x_t = \log(C_t/R_t).
\]

On a rolling window of transition pairs, estimate

\[
x_t = a + \phi x_{t-1} + \epsilon_t.
\]

An OU-compatible window requires \(0 < \phi < 1\). Its equilibrium level,
innovation volatility, stationary volatility, and half-life are

\[
\mu = \frac{a}{1-\phi},\qquad
\sigma_x = \frac{\sigma_\epsilon}{\sqrt{1-\phi^2}},\qquad
h = -\frac{\log 2}{\log \phi}.
\]

The standardized deviation is \(z_t=(x_t-\mu)/\sigma_x\). A rolling variance
ratio on residual changes supplies a separate regime gate:

\[
VR_t(q)=\frac{Var(x_t-x_{t-q})}{q\,Var(x_t-x_{t-1})}.
\]

Trading is allowed only when the OU half-life lies inside the configuration's
predeclared interval and \(VR_t(4) < 0.95\). A flat strategy enters long when
\(z_t\) is below the negative entry threshold and short when it is above the
positive threshold. An active position exits near equilibrium, when the
regime gate fails, at the maximum holding period, or at session close.

## 5. Frozen calibrations

| configuration_id | reference bars | OU/VR transitions | entry | exit | half-life bars | maximum holding bars |
|---|---:|---:|---:|---:|---:|---:|
| ou_vwap_fast | 26 | 104 | 1.75 | 0.25 | [1, 20] | 20 |
| ou_vwap_base | 32 | 130 | 2.00 | 0.25 | [1, 26] | 26 |
| ou_vwap_slow | 52 | 208 | 2.25 | 0.25 | [1, 39] | 39 |

All configurations use variance-ratio lag 4, threshold 0.95, the canonical
close-to-close simple return, and 1 basis point per unit turnover as the base
cost. These calibrations are sensitivity cases, not optimization candidates.

## 6. Portfolio and cost evidence

Results are reported for SPY, QQQ, IWM, and a contemporaneous equal-weight
three-symbol return series. No asset or calibration is removed because of its
performance. Cost stresses are fixed at 0, 1, 2, and 5 basis points per unit
turnover and reuse identical signals and positions.

## 7. Statistical inference

For each calibration and reported return series, Day 17 reports:

- number of test sessions and arithmetic mean session return;
- naive and Newey-West/HAC t-statistics, with five lags;
- annualized Sharpe ratio;
- deterministic moving-block bootstrap 95% confidence intervals for the mean
  and annualized Sharpe, using 2,000 replications, five-session blocks, and
  seed 1701;
- signal information coefficient between the causal signal score at t and
  the next intraday bar return;
- probabilistic Sharpe ratio against zero;
- Deflated Sharpe Ratio using the three predeclared calibration trials and
  the observed cross-calibration Sharpe dispersion for the same return series.

Undefined statistics remain explicit; they are never replaced by zero.

## 8. Acceptance and interpretation

Implementation acceptance requires deterministic tables, exact schema and
ordering checks, synthetic known-answer tests, no-look-ahead tests, cost
reconciliation, development-boundary rejection, artifact hashes, replayable
output, focused tests, full-suite tests, and clean Git checks.

There is no profitability gate. Negative returns, confidence intervals
spanning zero, weak information coefficients, or low Deflated Sharpe
probabilities are valid economic findings and must be discussed plainly.
