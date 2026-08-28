# Day 16 — Portfolio Allocation and Economic Validation

## 1. Objective and boundary

This development-only evaluation distinguishes statistical diversification
from economic performance. It measures mechanically valid portfolios; it does
not select a preferred rule, claim profitability, or authorize paper or live
orders. Negative returns, Sharpe ratios, and other negative results are valid
outcomes and are retained without reinterpretation.

The Day 14 zero-pair cointegration result is retained, so no mean-reversion
sleeve was added. The locked January–June 2026 period was not accessed.

## 2. Frozen inputs and rule order

The six Day 15 sleeves remain in this exact order:
trend_ratio_spy, trend_ratio_qqq, trend_ratio_iwm, ema_macd_spy, ema_macd_qqq, ema_macd_iwm.

The three allocation rules were predeclared and remain in this exact order:
equal_weight, inverse_volatility, constrained_minimum_variance.

Every fold estimates weights from training rows only, freezes the vector before
test-return use, and holds those weights fixed throughout that fold. There is
no monthly, quarterly, intrafold, or result-triggered rebalancing.

## 3. Allocation methods and costs

Equal weight assigns exactly 1/6 to every sleeve. Inverse volatility uses
ordinary training-sample standard deviations with ddof=1 and deterministic
water filling at the 0.35 cap. Constrained minimum variance is an optimization rule
using LedoitWolf(assume_centered=False) covariance and the frozen SLSQP settings.
No rule was selected using realized performance.

Sleeve returns already contain the frozen one-basis-point strategy cost.
Allocation turnover is charged separately at one basis point on the first test
session of each fold only; strategy costs are not counted twice.

## 4. Fold allocations and concentration

Weights below are listed in the frozen sleeve order.

| Fold | Rule | Weights in frozen sleeve order | Turnover | Cost | Max weight | HHI | Effective sleeves |
|---|---|---|---:|---:|---:|---:|---:|
| wf_2022 | equal_weight | 0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667 | 1.000000 | 0.0001000000 | 0.166667 | 0.166667 | 6.000000 |
| wf_2022 | inverse_volatility | 0.185274, 0.169468, 0.130814, 0.180958, 0.177958, 0.155529 | 1.000000 | 0.0001000000 | 0.185274 | 0.168762 | 5.925507 |
| wf_2022 | constrained_minimum_variance | 0.208707, 0.191474, 0.089826, 0.183990, 0.166442, 0.159562 | 1.000000 | 0.0001000000 | 0.208707 | 0.175304 | 5.704361 |
| wf_2023 | equal_weight | 0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667 | 0.000000 | 0.0000000000 | 0.166667 | 0.166667 | 6.000000 |
| wf_2023 | inverse_volatility | 0.185378, 0.153893, 0.135062, 0.191107, 0.173726, 0.160834 | 0.039615 | 0.0000039615 | 0.191107 | 0.168860 | 5.922065 |
| wf_2023 | constrained_minimum_variance | 0.332213, 0.072315, 0.056930, 0.247067, 0.139879, 0.151597 | 0.373167 | 0.0000373167 | 0.332213 | 0.222425 | 4.495888 |
| wf_2024 | equal_weight | 0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667 | 0.000000 | 0.0000000000 | 0.166667 | 0.166667 | 6.000000 |
| wf_2024 | inverse_volatility | 0.185744, 0.150519, 0.132466, 0.196455, 0.174385, 0.160430 | 0.012746 | 0.0000012746 | 0.196455 | 0.169447 | 5.901555 |
| wf_2024 | constrained_minimum_variance | 0.344734, 0.047552, 0.058446, 0.280944, 0.132376, 0.135948 | 0.095829 | 0.0000095829 | 0.344734 | 0.239453 | 4.176181 |
| wf_2025 | equal_weight | 0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667 | 0.000000 | 0.0000000000 | 0.166667 | 0.166667 | 6.000000 |
| wf_2025 | inverse_volatility | 0.187013, 0.149265, 0.131060, 0.200323, 0.173781, 0.158559 | 0.010272 | 0.0000010272 | 0.200323 | 0.169900 | 5.885802 |
| wf_2025 | constrained_minimum_variance | 0.350000, 0.036819, 0.060457, 0.314337, 0.112946, 0.125440 | 0.081342 | 0.0000081342 | 0.350000 | 0.254811 | 3.924482 |

## 5. Fold economic performance

| Fold | Rule | N | Start | End | Cumulative | Annualized | Ann. vol. | Sharpe | Max drawdown | VaR 95 | ES 95 | Turnover | Cost |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wf_2022 | equal_weight | 251 | 2022-01-03 | 2022-12-30 | -0.098425 | -0.098798 | 0.165043 | -0.548000 | -0.250067 | 0.017305 | 0.022732 | 1.000000 | 0.0001000000 |
| wf_2022 | inverse_volatility | 251 | 2022-01-03 | 2022-12-30 | -0.094966 | -0.095326 | 0.163546 | -0.531026 | -0.247230 | 0.017483 | 0.022506 | 1.000000 | 0.0001000000 |
| wf_2022 | constrained_minimum_variance | 251 | 2022-01-03 | 2022-12-30 | -0.088422 | -0.088758 | 0.163750 | -0.485998 | -0.242190 | 0.017912 | 0.022560 | 1.000000 | 0.0001000000 |
| wf_2023 | equal_weight | 250 | 2023-01-03 | 2023-12-29 | -0.010237 | -0.010319 | 0.090471 | -0.069576 | -0.097101 | 0.009325 | 0.013023 | 0.000000 | 0.0000000000 |
| wf_2023 | inverse_volatility | 250 | 2023-01-03 | 2023-12-29 | -0.007722 | -0.007784 | 0.088188 | -0.044676 | -0.094478 | 0.009068 | 0.012801 | 0.039615 | 0.0000039615 |
| wf_2023 | constrained_minimum_variance | 250 | 2023-01-03 | 2023-12-29 | -0.001791 | -0.001806 | 0.083329 | 0.019829 | -0.085467 | 0.008556 | 0.012545 | 0.373167 | 0.0000373167 |
| wf_2024 | equal_weight | 252 | 2024-01-02 | 2024-12-31 | -0.002344 | -0.002344 | 0.079897 | 0.010371 | -0.072112 | 0.007374 | 0.010436 | 0.000000 | 0.0000000000 |
| wf_2024 | inverse_volatility | 252 | 2024-01-02 | 2024-12-31 | -0.009748 | -0.009748 | 0.077551 | -0.087718 | -0.072061 | 0.007136 | 0.010315 | 0.012746 | 0.0000012746 |
| wf_2024 | constrained_minimum_variance | 252 | 2024-01-02 | 2024-12-31 | -0.021549 | -0.021549 | 0.070934 | -0.271768 | -0.073386 | 0.006290 | 0.009647 | 0.095829 | 0.0000095829 |
| wf_2025 | equal_weight | 250 | 2025-01-02 | 2025-12-31 | -0.209502 | -0.210987 | 0.139019 | -1.632972 | -0.217012 | 0.011431 | 0.023268 | 0.000000 | 0.0000000000 |
| wf_2025 | inverse_volatility | 250 | 2025-01-02 | 2025-12-31 | -0.206974 | -0.208444 | 0.140330 | -1.593378 | -0.215126 | 0.011264 | 0.023007 | 0.010272 | 0.0000010272 |
| wf_2025 | constrained_minimum_variance | 250 | 2025-01-02 | 2025-12-31 | -0.198122 | -0.199538 | 0.137983 | -1.541854 | -0.205960 | 0.009890 | 0.022359 | 0.081342 | 0.0000081342 |

All reported portfolio returns and metrics are net of the specified allocation
cost. Historical VaR and Expected Shortfall are non-negative loss measures.

## 6. Concatenated 2022–2025 out-of-sample performance

| Rule | N | Start | End | Cumulative | Annualized | Ann. vol. | Sharpe | Max drawdown | VaR 95 | ES 95 | Total turnover | Total cost |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equal_weight | 1003 | 2022-01-03 | 2025-12-31 | -0.296256 | -0.084489 | 0.123553 | -0.652175 | -0.386079 | 0.011238 | 0.019198 | 1.000000 | 0.0001000000 |
| inverse_volatility | 1003 | 2022-01-03 | 2025-12-31 | -0.294770 | -0.084004 | 0.122629 | -0.653656 | -0.382538 | 0.011407 | 0.019019 | 1.062633 | 0.0001062633 |
| constrained_minimum_variance | 1003 | 2022-01-03 | 2025-12-31 | -0.286059 | -0.081174 | 0.120146 | -0.644009 | -0.373295 | 0.010910 | 0.018811 | 1.550337 | 0.0001550337 |

## 7. Mechanical evaluation

The frozen schema, development-only scope, chronological train/test separation,
training-only weight estimation, weight constraints, finite covariance inputs,
successful minimum-variance solver status, finite portfolio returns, complete
row counts, and deterministic artifact contract are satisfied.

The final evaluation_complete field is true. It depends only on mechanical
completeness and does not depend on return, Sharpe ratio, drawdown, VaR,
Expected Shortfall, profitability, or relative performance.

## 8. Interpretation and limitations

These results describe allocation behaviour under three fixed rules. Statistical
diversification does not imply positive economic performance. The constrained
minimum-variance calculation uses optimization, but realized results were not
used to rank or select a rule. Leverage, short weights, borrowing, cost-aware
optimization, expected-return inputs, strategy removal, and parameter tuning
remain outside this evaluation.
