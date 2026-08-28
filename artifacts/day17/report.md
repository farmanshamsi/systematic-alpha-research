# Day 17 OU/VWAP Reversion and Statistical Inference

## Scope

This is a development-only evaluation of three predeclared sensitivity calibrations on SPY, QQQ, and IWM. It uses rolling volume-weighted price residuals, rolling OU diagnostics, a variance-ratio regime gate, one-bar execution delay, forced overnight flatness, and explicit costs.

No calibration or asset is ranked, selected, or authorized for paper trading. Locked 2026 data were not accessed.

## Aggregate walk-forward performance

| Configuration | Series | Cumulative return | Annualized return | Sharpe | Maximum drawdown | Turnover |
|---|---|---:|---:|---:|---:|---:|
| ou_vwap_fast | SPY | -0.126555 | -0.033425 | -0.601770 | -0.152524 | 493.00 |
| ou_vwap_fast | QQQ | -0.183281 | -0.049595 | -0.679387 | -0.226101 | 494.00 |
| ou_vwap_fast | IWM | -0.001800 | -0.000453 | 0.029864 | -0.105957 | 508.00 |
| ou_vwap_fast | equal_weight | -0.105102 | -0.027514 | -0.451570 | -0.133637 | 498.33 |
| ou_vwap_base | SPY | -0.120054 | -0.031622 | -0.853923 | -0.142045 | 262.00 |
| ou_vwap_base | QQQ | -0.015934 | -0.004028 | -0.045907 | -0.078351 | 262.00 |
| ou_vwap_base | IWM | 0.004548 | 0.001141 | 0.048368 | -0.067526 | 244.00 |
| ou_vwap_base | equal_weight | -0.043596 | -0.011137 | -0.250555 | -0.083602 | 256.00 |
| ou_vwap_slow | SPY | 0.003522 | 0.000884 | 0.059310 | -0.029895 | 50.00 |
| ou_vwap_slow | QQQ | 0.082926 | 0.020218 | 0.574562 | -0.020887 | 72.00 |
| ou_vwap_slow | IWM | 0.094692 | 0.022991 | 0.687148 | -0.041863 | 94.00 |
| ou_vwap_slow | equal_weight | 0.060349 | 0.014831 | 0.638571 | -0.019483 | 72.00 |

## Statistical inference

| Configuration | Series | HAC t-stat | Mean bootstrap 95% CI | IC | PSR | DSR |
|---|---|---:|---:|---:|---:|---:|
| ou_vwap_fast | SPY | -1.693029 | [-0.000285, 0.000031] | -0.005678 | 0.139256 | 0.035306 |
| ou_vwap_fast | QQQ | -1.811869 | [-0.000401, 0.000023] | -0.010398 | 0.118754 | 0.017390 |
| ou_vwap_fast | IWM | 0.062774 | [-0.000250, 0.000290] | 0.006182 | 0.523869 | 0.280973 |
| ou_vwap_fast | equal_weight | -1.182821 | [-0.000273, 0.000084] | 0.003025 | 0.212509 | 0.047286 |
| ou_vwap_base | SPY | -2.145633 | [-0.000240, -0.000015] | -0.009298 | 0.033328 | 0.003490 |
| ou_vwap_base | QQQ | -0.121778 | [-0.000172, 0.000170] | -0.009050 | 0.463892 | 0.125836 |
| ou_vwap_base | IWM | 0.123756 | [-0.000163, 0.000215] | 0.000282 | 0.538983 | 0.291860 |
| ou_vwap_base | equal_weight | -0.740629 | [-0.000153, 0.000076] | -0.006515 | 0.316006 | 0.077147 |
| ou_vwap_slow | SPY | 0.116883 | [-0.000059, 0.000077] | -0.000518 | 0.547758 | 0.243915 |
| ou_vwap_slow | QQQ | 1.355994 | [-0.000026, 0.000203] | -0.003047 | 0.917568 | 0.538386 |
| ou_vwap_slow | IWM | 1.309952 | [-0.000043, 0.000232] | 0.016656 | 0.929024 | 0.784196 |
| ou_vwap_slow | equal_weight | 1.421178 | [-0.000015, 0.000149] | 0.012219 | 0.922772 | 0.625790 |

## Cost stress

| Configuration | Series | Cumulative return at 1 bp | Cumulative return at 5 bp |
|---|---|---:|---:|
| ou_vwap_fast | SPY | -0.126555 | -0.282926 |
| ou_vwap_fast | QQQ | -0.183281 | -0.329773 |
| ou_vwap_fast | IWM | -0.001800 | -0.185397 |
| ou_vwap_fast | equal_weight | -0.105102 | -0.266875 |
| ou_vwap_base | SPY | -0.120054 | -0.207634 |
| ou_vwap_base | QQQ | -0.015934 | -0.113873 |
| ou_vwap_base | IWM | 0.004548 | -0.088890 |
| ou_vwap_base | equal_weight | -0.043596 | -0.136727 |
| ou_vwap_slow | SPY | 0.003522 | -0.016353 |
| ou_vwap_slow | QQQ | 0.082926 | 0.052170 |
| ou_vwap_slow | IWM | 0.094692 | 0.054273 |
| ou_vwap_slow | equal_weight | 0.060349 | 0.030220 |

## Execution and leakage controls

- Initial non-flat positions: 0.
- Initial non-zero turnover rows: 0.
- Overnight position violations: 0.
- Training history warms rolling indicators, while execution state resets at every test boundary.
- Undefined statistics remain reported as N/A.

## Interpretation boundary

The evidence tests whether the proposed reversion mechanism is statistically and economically defensible. Profitability is not an acceptance condition. Weak or negative findings are retained and do not trigger calibration replacement.
