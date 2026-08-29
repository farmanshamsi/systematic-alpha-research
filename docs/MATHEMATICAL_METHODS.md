# Mathematical Methods

[Back to README](../README.md) · [Full Mathematical Derivations](MATHEMATICAL_DERIVATIONS.md)

This document develops the mathematical foundations behind the models, statistical tests, portfolio methods, and execution assumptions implemented in Systematic Alpha Research.

## 1. Ornstein-Uhlenbeck Mean Reversion

The mean-reversion component models a transformed intraday residual rather than assuming that the raw ETF price itself is stationary.

### 1.1 Transformed residual

For market price \(P_t\), define a rolling volume-weighted reference \(V_t\).

```math
V_t = \frac{\sum_{i=t-n+1}^{t} VWAP_i \, Volume_i}{\sum_{i=t-n+1}^{t} Volume_i}
```

The modeled state variable is:

```math
X_t = \log\left(\frac{P_t}{V_t}\right)
```

Equivalently:

```math
X_t = \log P_t - \log V_t
```

This transformation interprets the state as a relative displacement from a rolling volume-weighted reference.

### 1.2 Continuous-time OU model

The Ornstein-Uhlenbeck process satisfies:

```math
dX_t = \kappa(\mu-X_t)\,dt + \sigma\,dW_t
```

where \(\mu\) is the equilibrium level, \(\kappa>0\) is the mean-reversion speed, \(\sigma>0\) is the diffusion scale, and \(W_t\) is Brownian motion.

Rearranging:

```math
dX_t + \kappa X_t\,dt = \kappa\mu\,dt + \sigma\,dW_t
```

Multiplying by the integrating factor \(e^{\kappa t}\):

```math
d\left(e^{\kappa t}X_t\right) = \kappa\mu e^{\kappa t}dt + \sigma e^{\kappa t}dW_t
```

Integrating from \(t\) to \(t+\Delta\):

```math
X_{t+\Delta} = \mu + e^{-\kappa\Delta}(X_t-\mu) + \sigma\int_t^{t+\Delta} e^{-\kappa(t+\Delta-s)}\,dW_s
```

### 1.3 Exact transition distribution

The stochastic integral in the OU solution is Gaussian with mean zero. Therefore:

```math
X_{t+\Delta}\mid X_t \sim \mathcal N\left(\mu+e^{-\kappa\Delta}(X_t-\mu), \frac{\sigma^2}{2\kappa}\left(1-e^{-2\kappa\Delta}\right)\right)
```

This exact transition links the continuous-time OU model to the discrete estimator used in the strategy.

### 1.4 AR(1) representation

Write the discrete process as:

```math
X_t = a + \phi X_{t-1} + \eta_t
```

Comparing with the exact OU transition gives:

```math
\phi = e^{-\kappa\Delta}
```

and:

```math
a = \mu(1-\phi)
```

Hence:

```math
\kappa = -\frac{\log\phi}{\Delta}
```

and:

```math
\mu = \frac{a}{1-\phi}
```

For bar-indexed estimation, the implementation takes \(\Delta=1\), so \(\kappa\) is interpreted in inverse-bar units.

### 1.5 Half-life

Mean deviations from equilibrium decay according to:

```math
\mathbb E[X_{t+h}-\mu\mid X_t] = e^{-\kappa h}(X_t-\mu)
```

The half-life \(h_{1/2}\) is defined by:

```math
e^{-\kappa h_{1/2}} = \frac{1}{2}
```

Therefore:

```math
h_{1/2} = \frac{\log 2}{\kappa}
```

Using the AR(1) mapping:

```math
h_{1/2} = -\frac{\Delta\log 2}{\log\phi}
```

With \(\Delta=1\) bar:

```math
h_{1/2}^{(\mathrm{bars})} = -\frac{\log 2}{\log\phi}
```

### 1.6 Innovation variance and stationary variance

Let the AR(1) innovation variance be:

```math
s_\eta^2 = \mathrm{Var}(\eta_t)
```

For the exact discretization of an OU process:

```math
s_\eta^2 = \frac{\sigma^2}{2\kappa}\left(1-e^{-2\kappa\Delta}\right)
```

Using \(\phi=e^{-\kappa\Delta}\):

```math
s_\eta^2 = \frac{\sigma^2}{2\kappa}(1-\phi^2)
```

The continuous-time diffusion coefficient can therefore be recovered as:

```math
\sigma = s_\eta\sqrt{\frac{2\kappa}{1-\phi^2}}
```

The implementation does not require this continuous-time \(\sigma\) directly. It uses the stationary standard deviation of the fitted discrete process.

For a stationary AR(1):

```math
\mathrm{Var}(X_t) = \frac{s_\eta^2}{1-\phi^2}
```

Therefore:

```math
\sigma_X = \frac{s_\eta}{\sqrt{1-\phi^2}}
```

This stationary scale is used to normalize the current residual before constructing the trading state.

### 1.7 OU admissibility conditions

The implementation requires:

```math
0 < \phi < 1
```

This restriction is not arbitrary.

If \(\phi \geq 1\), the fitted process is not a stationary positive-speed mean-reverting OU process.

If \(\phi \leq 0\), the fitted AR(1) has sign-alternating dynamics and cannot be represented as \(\phi=e^{-\kappa\Delta}\) for real \(\kappa>0\).

Accordingly, such estimates are treated as OU-incompatible rather than forced into a mean-reversion interpretation.

The fitted half-life must also lie within a predeclared minimum and maximum range.

Undefined innovation variance, stationary variance, or incomplete rolling estimates also invalidate the regime.

### 1.8 Rolling OLS estimation

Inside each rolling estimation window, the transformed residual is fitted as:

```math
X_t = a + \phi X_{t-1} + \eta_t
```

For \(x_i=X_{i-1}\) and \(y_i=X_i\), the OLS slope is:

```math
\hat{\phi} = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sum_i (x_i-\bar{x})^2}
```

and the intercept is:

```math
\hat{a} = \bar{y} - \hat{\phi}\bar{x}
```

The fitted equilibrium is then:

```math
\hat{\mu} = \frac{\hat{a}}{1-\hat{\phi}}
```

The innovation variance is estimated from the residual sum of squares:

```math
\hat{s}_\eta^2 = \frac{\sum_i \left(y_i-\hat{a}-\hat{\phi}x_i\right)^2}{n-2}
```

and the fitted stationary standard deviation is:

```math
\hat{\sigma}_X = \frac{\hat{s}_\eta}{\sqrt{1-\hat{\phi}^2}}
```

Every quantity at time \(t\) is estimated only from observations available no later than \(t\).

### 1.9 Standardized OU state

The normalized displacement from equilibrium is:

```math
Z_t = \frac{X_t-\hat{\mu}_t}{\hat{\sigma}_{X,t}}
```

The contrarian score is proportional to:

```math
-Z_t
```

A sufficiently negative residual produces a long signal, while a sufficiently positive residual produces a short signal.

### 1.10 Variance-ratio regime filter

For lag \(q\), define:

```math
VR(q)=\frac{\mathrm{Var}(X_t-X_{t-q})}{q\,\mathrm{Var}(X_t-X_{t-1})}
```

A random walk has approximately \(VR(q)\approx1\).

The implementation requires:

```math
VR(q) < VR_{\max}
```

together with valid OU persistence and half-life estimates.

The variance-ratio condition is used as a regime filter, not as an independent source of alpha.

### 1.11 Causal execution timing

A statistically valid state estimate can still produce an invalid backtest if the strategy is allowed to trade on information that was not yet available at the assumed execution time.

Let \(\mathcal F_t\) denote the information set available after bar \(t\) has completed.

The model estimate and trading signal satisfy:

```math
S_t \in \mathcal F_t
```

The implementation then enforces:

```math
Position_t = S_{t-1}
```

so a signal computed from bar \(t\) cannot create exposure on that same observation.

This is implemented explicitly through a one-bar shift before the position series is constructed.

Session-close logic also forces the strategy flat rather than allowing an intraday signal to become unintended overnight exposure.

### 1.12 Signal state machine

The trading rule is stateful rather than a direct pointwise z-score mapping.

For a flat strategy state:

```math
Z_t \leq -z_{\mathrm{entry}} \Rightarrow \text{long signal}
```

and:

```math
Z_t \geq z_{\mathrm{entry}} \Rightarrow \text{short signal}
```

An existing position can be closed when the residual returns sufficiently close to equilibrium, the OU regime becomes invalid, the maximum holding period is reached, or the session boundary requires a reset.

This distinction matters because the strategy path depends on the previous trading state as well as the current residual.

### 1.13 Equation-to-code mapping

The mathematical quantities map directly to `src/systematic_alpha/strategies/ou_vwap_reversion.py`.

| Mathematical object | Implementation |
|---|---|
| \(V_t\) | `volume_weighted_reference` |
| \(X_t=\log(P_t/V_t)\) | `log_price_residual` |
| \(\hat a\) | `ou_intercept` |
| \(\hat\phi\) | `ou_phi` |
| \(\hat\mu=\hat a/(1-\hat\phi)\) | `ou_equilibrium` |
| \(\hat s_\eta\) | `ou_innovation_std` |
| \(\hat\sigma_X\) | `ou_stationary_std` |
| \(h_{1/2}\) | `ou_half_life_bars` |
| \(Z_t\) | `ou_zscore` |
| \(VR(q)\) | `variance_ratio` |
| OU/regime admissibility | `regime_eligible` |
| contrarian score | `signal_score` |
| stateful trading decision | `_target_state(...)` |
| one-bar execution delay | `signal.shift(1)` when constructing `position` |
| trading intensity | `turnover` |
| proportional cost | `transaction_cost` |

The rolling AR(1) calculations are implemented in `_rolling_ou_statistics(...)` rather than hidden behind a black-box regression call.

The code explicitly constructs rolling sums, cross-products, OLS slope and intercept, innovation variance, equilibrium, stationary variance, and half-life.

This makes the estimator auditable and allows its numerical assumptions to be tested directly.

### 1.14 Interpretation boundary

The OU formulation should not be interpreted as proof that the transformed market residual is literally generated by a continuous-time Gaussian OU diffusion.

It is used as a disciplined local approximation:

1. estimate persistence with a causal AR(1);
2. require that the estimate admits a valid OU mapping;
3. require an economically plausible half-life;
4. normalize deviations using fitted stationary variance;
5. reject trend-like regimes through the variance-ratio filter;
6. delay execution so the position does not use future information;
7. evaluate the resulting strategy under costs and statistical inference.

The project therefore keeps model approximation, statistical evidence, and trading profitability as separate claims.

---

## 2. Cointegration and Error Correction

### 2.1 Integrated processes

A price series \(P_t\) is commonly analyzed in logarithms:

```math
Y_t = \log P_t
```

A process is integrated of order one, written \(I(1)\), when its level is non-stationary but its first difference is stationary.

Formally:

```math
Y_t \sim I(1)
```

while:

```math
\Delta Y_t = Y_t-Y_{t-1} \sim I(0)
```

The implementation treats a series as plausibly \(I(1)\) only when the unit-root null is not rejected in levels but is rejected after first differencing.

Daily log prices are used for this long-run integration and cointegration screen.

### 2.2 Long-run equilibrium relation

For two candidate log-price series \(Y_t\) and \(X_t\), the long-run relation is written:

```math
Y_t = \alpha + \beta X_t + \varepsilon_t
```

where \(\alpha\) is the intercept, \(\beta\) is the hedge-ratio coefficient, and \(\varepsilon_t\) is the equilibrium residual.

The fitted residual is:

```math
\hat{\varepsilon}_t = Y_t-\hat{\alpha}-\hat{\beta}X_t
```

Cointegration requires that both component series can be non-stationary while the linear combination is stationary:

```math
Y_t \sim I(1), \qquad X_t \sim I(1)
```

but:

```math
\varepsilon_t \sim I(0)
```

The economic interpretation is that the two levels may wander, but their estimated equilibrium relation does not wander without bound.

### 2.3 OLS estimation of the cointegrating vector

The long-run regression minimizes:

```math
\sum_t \left(Y_t-\alpha-\beta X_t\right)^2
```

with fitted coefficients \(\hat{\alpha}\) and \(\hat{\beta}\).

The project uses one predeclared orientation for each pair rather than searching both directions after observing the results.

That restriction matters because ordinary least-squares cointegrating regressions are not generally symmetric in \(X\) and \(Y\).

### 2.4 Engle-Granger residual test

After estimating the long-run regression, the central null hypothesis is that the residual still contains a unit root.

The residual ADF-style regression can be represented as:

```math
\Delta \hat{\varepsilon}_t = \gamma \hat{\varepsilon}_{t-1} + \sum_{j=1}^{p}\psi_j \Delta\hat{\varepsilon}_{t-j}+u_t
```

The null is:

```math
H_0:\gamma=0
```

corresponding to a unit root in the equilibrium residual.

The alternative is:

```math
H_1:\gamma<0
```

corresponding to residual stationarity and therefore evidence consistent with cointegration.

Because the residual comes from an estimated first-stage regression, ordinary ADF critical values are not the correct Engle-Granger reference distribution. The implementation therefore uses the dedicated Engle-Granger test for the primary cointegration p-value.

### 2.5 Multiple-testing control

Testing several candidate pairs creates a family of hypotheses.

If \(m\) raw p-values are written in ascending order:

```math
p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}
```

the Holm procedure compares them sequentially against increasingly relaxed thresholds.

At family-wise significance level \(\alpha\), the ordered hypothesis at rank \(i\) is compared with:

```math
\frac{\alpha}{m-i+1}
```

The procedure stops when the first hypothesis fails its threshold.

This controls the family-wise error rate more strongly than treating every pairwise p-value as an independent discovery.

The implementation applies Holm correction across the three predeclared candidate ETF pairs.

### 2.6 Hedge-ratio interpretation and stability

Passing a residual stationarity test is not treated as sufficient by itself.

The fitted hedge ratio must also be finite, economically interpretable, and stable through time.

For successive walk-forward estimates, define relative beta drift as:

```math
D_t^{(\beta)} = \frac{|\hat{\beta}_t-\hat{\beta}_{t-1}|}{|\hat{\beta}_{t-1}|}
```

Large drift indicates that the estimated equilibrium relation is changing materially across estimation windows.

The implementation also checks whether the sign of \(\beta\) changes.

A sign reversal is economically more severe than a small magnitude adjustment because it changes the direction of the hedge relation itself.

Only estimates made from information available before the corresponding test fold are allowed to enter the causal stability gate.

An ex-post full-sample beta can be reported descriptively, but it is explicitly excluded from fold eligibility and stability decisions.

### 2.7 Error-correction representation

If \(Y_t\) and \(X_t\) are cointegrated, short-run changes can depend on the previous equilibrium error.

A simple error-correction model can be written:

```math
\Delta Y_t = c + \lambda \hat{\varepsilon}_{t-1} + \sum_{i=1}^{p}\phi_i \Delta Y_{t-i} + \sum_{j=0}^{q}\psi_j \Delta X_{t-j} + \nu_t
```

where:

```math
\hat{\varepsilon}_{t-1}=Y_{t-1}-\hat{\alpha}-\hat{\beta}X_{t-1}
```

The coefficient \(\lambda\) measures how strongly the system responds to disequilibrium.

For a stable error-correction mechanism, the adjustment direction should pull the system back toward equilibrium.

For example, when a positive residual represents \(Y\) being above its long-run relation, a negative adjustment coefficient in the \(Y\) equation implies downward correction.

The ECM interpretation is useful because it separates two ideas:

1. a long-run equilibrium relation in levels;
2. short-run dynamics in changes.

Cointegration therefore does not imply that the two assets move identically at every instant. It implies that deviations from the long-run relation are not allowed to drift indefinitely under the fitted model.

### 2.8 Cointegration residual and OU dynamics

Once a cointegrating relation has passed the statistical gate, its intraday residual can be analyzed as a candidate mean-reverting state.

Define:

```math
\varepsilon_t = Y_t-\alpha-\beta X_t
```

If the residual admits a discrete AR(1) representation:

```math
\varepsilon_t = a+\phi\varepsilon_{t-1}+\eta_t
```

with:

```math
0<\phi<1
```

then the same OU mapping used earlier applies:

```math
\kappa=-\log\phi
```

for one-bar spacing, together with:

```math
\theta=\frac{a}{1-\phi}
```

and:

```math
h_{1/2}=\frac{\log 2}{\kappa}
```

The project uses this OU step only after the prior statistical gates pass.

This ordering is deliberate: a mean-reverting-looking fitted residual is not allowed to rescue a pair that already failed the integration, cointegration, multiplicity, beta, or stability requirements.

### 2.9 Causal walk-forward chronology

Cointegration estimates are path-dependent research objects. A full-sample estimate can accidentally import future information into an earlier decision.

For each walk-forward fold, define a training interval:

```math
[T_{\mathrm{train,start}}, T_{\mathrm{train,end}})
```

and a subsequent test interval:

```math
[T_{\mathrm{test,start}}, T_{\mathrm{test,end}})
```

with the requirement:

```math
T_{\mathrm{train,end}} \leq T_{\mathrm{test,start}}
```

The cointegrating regression for that fold is estimated using training observations only.

The maximum timestamp used in estimation must satisfy:

```math
T_{\max}^{(\mathrm{estimate})} < T_{\mathrm{test,start}}
```

The same training-only restriction applies to:

- integration diagnostics;
- Engle-Granger testing;
- Holm-adjusted pair eligibility;
- residual stationarity;
- hedge-ratio stability;
- OU admissibility.

The implementation explicitly records the maximum information timestamp for each fold so chronology can be audited rather than assumed.

### 2.10 Successive-beta stability

The causal stability comparison uses the previous fold estimate as the reference, not a future full-sample coefficient.

For fold \(k\):

```math
D_k^{(\beta)} = \frac{|\hat{\beta}_k-\hat{\beta}_{k-1}|}{|\hat{\beta}_{k-1}|}
```

The first fold has no previous causal estimate, so no artificial historical reference is invented.

An ex-post coefficient estimated from all available development data is retained only as descriptive evidence and is excluded from eligibility decisions.

This distinction prevents hindsight from making a historically unstable relation appear stable.

### 2.11 Equation-to-code mapping

The main implementation is in `src/systematic_alpha/analysis/cointegration_feasibility.py` and `src/systematic_alpha/analysis/causal_cointegration_chronology.py`.

| Mathematical object | Implementation |
|---|---|
| \(Y_t, X_t\) log prices | synchronized `y_log_price`, `x_log_price` |
| \(Y_t=\alpha+\beta X_t+\varepsilon_t\) | `_fit_long_run(...)` / cointegrating regression estimator |
| \(\hat{\alpha}\) | `alpha` / `alpha_estimate` |
| \(\hat{\beta}\) | `beta` / `beta_estimate` |
| \(\hat{\varepsilon}_t\) | fitted regression residuals |
| level / difference unit-root checks | `_adf_result(...)` and causal frozen ADF path |
| Engle-Granger p-value | `coint(...)` |
| Holm multiplicity correction | `multipletests(..., method="holm")` |
| relative beta drift | `relative_beta_drift` |
| beta sign stability | `beta_sign_change` / stability gate |
| fold residual stationarity | `residual_stationary` |
| training-only OU gate | `_causal_ou_gate(...)` |
| final fold eligibility | `fold_eligibility` / pair summary eligibility |
| ex-post descriptive beta | `ex_post_beta_diagnostics` |

The implementation aligns pair observations by exact shared dates or timestamps and does not forward-fill one asset to manufacture synchronization.

### 2.12 Interpretation boundary

Cointegration is treated as a statistical equilibrium hypothesis, not as proof of an immutable economic law.

A valid pair must therefore survive several distinct questions:

1. are the individual series plausibly \(I(1)\)?
2. is the residual stationary under the predeclared Engle-Granger specification?
3. does the result survive Holm multiplicity control?
4. is the hedge ratio finite and economically interpretable?
5. is the hedge relation sufficiently stable across causal folds?
6. is residual stationarity sufficiently persistent across folds?
7. only then, does the intraday residual admit a valid OU representation?

Failure at an earlier gate is preserved as evidence rather than repaired through specification search.

---

## 3. Causal Trend Models

### 3.1 Price-ratio trend state

The first trend family compares short- and long-horizon averages of price.

For short window \(n_s\):

```math
\bar P_t^{(s)} = \frac{1}{n_s}\sum_{i=0}^{n_s-1}P_{t-i}
```

For long window \(n_l\):

```math
\bar P_t^{(l)} = \frac{1}{n_l}\sum_{i=0}^{n_l-1}P_{t-i}
```

with:

```math
n_s<n_l
```

The implemented ratio is:

```math
R_t = \frac{\bar P_t^{(s)}}{\bar P_t^{(l)}}
```

A neutral band \(\delta\geq0\) defines the trading state.

For the long-short-neutral version:

```math
S_t = \begin{cases} +1, & R_t>1+\delta \\ -1, & R_t<1-\delta \\ 0, & \text{otherwise} \end{cases}
```

For the long-flat comparator:

```math
S_t = \begin{cases} +1, & R_t>1+\delta \\ 0, & \text{otherwise} \end{cases}
```

The two positioning rules are treated as distinct research specifications rather than interchangeable descriptions of the same backtest.

### 3.2 Causal signal-to-position mapping

The signal computed from bar \(t\) uses information in \(\mathcal F_t\).

The saved strategy implementation therefore maps:

```math
Position_t = S_{t-1}
```

through an explicit one-observation shift.

This removes same-row signal look-ahead, but by itself does not prove that a prior close is an executable fill price across a session boundary.

That distinction motivated the later next-bar-open execution convention.

### 3.3 Turnover and proportional transaction cost

For position \(w_t\in\{-1,0,+1\}\), one-way turnover is:

```math
TO_t = |w_t-w_{t-1}|
```

A direct reversal from \(-1\) to \(+1\) therefore has turnover two.

With cost \(c\) basis points per unit turnover:

```math
Cost_t = TO_t\frac{c}{10{,}000}
```

Under return \(r_t\), net strategy return is:

```math
r_t^{net} = w_t r_t - Cost_t
```

### 3.4 Recursive EMA

For smoothing window \(n\), define:

```math
\alpha = \frac{2}{n+1}
```

The exponential moving average evolves recursively as:

```math
EMA_t = \alpha P_t + (1-\alpha)EMA_{t-1}
```

The recursive form assigns exponentially decreasing weight to older observations while preserving a causal state variable.

### 3.5 MACD state

Let \(EMA_t^{(f)}\) and \(EMA_t^{(s)}\) denote fast and slow exponential averages with \(f<s\).

The MACD process is:

```math
M_t = EMA_t^{(f)}-EMA_t^{(s)}
```

The MACD signal line is another EMA:

```math
G_t = EMA^{(m)}(M_t)
```

The histogram is:

```math
H_t = M_t-G_t
```

Its first difference is:

```math
\Delta H_t = H_t-H_{t-1}
```

and its second difference is:

```math
\Delta^2H_t = \Delta H_t-\Delta H_{t-1}
```

These quantities separate level, momentum, and acceleration-like behavior in the MACD state.

As with the price-ratio model, the implementation applies a neutral region before converting the indicator state into a discrete trading signal.

### 3.6 Filtration and feature causality

Every recursive EMA state satisfies:

```math
EMA_t \in \mathcal F_t
```

and therefore:

```math
M_t,\ G_t,\ H_t,\ \Delta H_t,\ \Delta^2H_t \in \mathcal F_t
```

The fact that an indicator is causal does not by itself determine a valid fill price. Signal construction and execution accounting are therefore treated as separate layers.

### 3.7 Next-bar-open execution convention

To avoid using a bar-close signal as though it could be filled at that same close, the final trend methodology separates signal formation from the earliest assumed executable price.

If the signal from completed bar \(t\) is \(S_t\), the earliest next-bar-open target is:

```math
w_{t+1}^{open}=S_t
```

The strategy return is then measured from that next observed open rather than retrospectively assigning the position to the prior close.

For a non-final bar of a session:

```math
r_{t}^{open\rightarrow open} = \frac{O_{t+1}}{O_t}-1
```

so gross strategy return is:

```math
r_t^{gross}=w_t\,r_t^{open\rightarrow open}
```

For the final bar of a session, the position is liquidated using the same-session close:

```math
r_t^{open\rightarrow close} = \frac{C_t}{O_t}-1
```

This prevents an intraday trend position from becoming unintended overnight exposure.

### 3.8 Session-boundary turnover

Let \(w_t\) denote the position held after the current bar open.

At a session open, the prior ending exposure is defined as zero.

The opening turnover is therefore:

```math
TO_t^{open}=|w_t-w_{t-1}^{end}|
```

with:

```math
w_{t-1}^{end}=0
```

at the first bar of a new session.

If bar \(t\) is the session close, forced liquidation adds:

```math
TO_t^{close}=|w_t|
```

Total turnover is:

```math
TO_t=TO_t^{open}+TO_t^{close}
```

and the ending position is:

```math
w_t^{end}=0
```

for every session-close observation.

This makes overnight-flat behavior an accounting invariant rather than a descriptive claim.

### 3.9 Sequential replay

The final methodology independently replays the strategy one event at a time.

For each observation, the replay reconstructs:

- the lagged signal-derived position;
- session-open state;
- session-close state;
- opening turnover;
- forced closing turnover;
- gross return;
- transaction cost;
- net return.

The sequential replay does not reuse vectorized output columns as inputs. It reconstructs the accounting path from the underlying signal and market observations.

### 3.10 Vectorized-sequential parity

Let the batch calculation produce path \(B_t\) and the independent event replay produce \(E_t\).

For discrete positions, the requirement is exact equality:

```math
w_t^{(B)} = w_t^{(E)} \qquad \forall t
```

and likewise for ending positions.

For floating-point accounting quantities, define:

```math
d_x = \max_t |x_t^{(B)}-x_t^{(E)}|
```

for turnover, gross return, transaction cost, and net return.

The implementation requires:

```math
\max_x d_x \leq 10^{-12}
```

This parity test is important because identical headline performance does not prove identical trading chronology. Exact path-level agreement is a stronger validation of the implementation.

### 3.11 Equation-to-code mapping

The trend implementation is distributed across `src/systematic_alpha/strategies/trend_ratio.py`, `src/systematic_alpha/strategies/ema_macd.py`, and `src/systematic_alpha/analysis/trend_methodology_finalization.py`.

| Mathematical object | Implementation |
|---|---|
| \(\bar P_t^{(s)}\) | `short_average` |
| \(\bar P_t^{(l)}\) | `long_average` |
| \(R_t\) | `ma_price_ratio` |
| neutral-band state | `_build_raw_signal(...)` |
| lagged position | grouped `signal.shift(1)` |
| \(TO_t\) | `calculate_turnover(...)` |
| EMA recursion | recursive EMA feature construction |
| MACD \(M_t\) | fast EMA minus slow EMA |
| signal line \(G_t\) | EMA of MACD |
| histogram \(H_t\) | MACD minus signal line |
| next-open / flat-close timing | `apply_next_open_overnight_flat(...)` |
| independent sequential path | `sequential_next_open_replay(...)` |
| parity diagnostics | replay parity table |

### 3.12 Interpretation boundary

The trend models are deliberately simple so that improvements in apparent performance cannot be confused with improvements in methodological validity.

The research separates:

1. indicator construction;
2. signal discretization;
3. execution timing;
4. turnover and costs;
5. walk-forward evaluation;
6. sequential implementation parity.

A strategy can fail economically while still pass these methodological tests. Negative performance therefore does not invalidate the engineering evidence, and methodological correctness does not imply profitable alpha.

---

## 4. Statistical Inference

### 4.1 Why the naive t-statistic is not enough

Let session strategy returns be \(r_1,\ldots,r_T\), with sample mean:

```math
\bar r = \frac{1}{T}\sum_{t=1}^{T}r_t
```

and sample standard deviation \(s_r\).

Under an independence assumption, the usual t-statistic is:

```math
t_{\mathrm{naive}} = \frac{\bar r}{s_r/\sqrt{T}}
```

Strategy returns may exhibit serial dependence because positions persist, signals overlap through time, volatility clusters, and regime filters create state dependence.

If returns are autocorrelated, the iid standard error can understate or overstate uncertainty.

### 4.2 HAC / Newey-West long-run variance

Define demeaned returns:

```math
u_t = r_t-\bar r
```

The lag-\(k\) sample autocovariance used by the implementation is:

```math
\hat{\gamma}_k = \frac{1}{T}\sum_{t=k+1}^{T}u_tu_{t-k}
```

For maximum HAC lag \(L\), the Bartlett weight is:

```math
w_k = 1-\frac{k}{L+1}
```

The estimated long-run variance is:

```math
\hat{\Omega} = \hat{\gamma}_0 + 2\sum_{k=1}^{L}w_k\hat{\gamma}_k
```

The HAC standard error of the mean is:

```math
SE_{\mathrm{HAC}}(\bar r)=\sqrt{\frac{\hat{\Omega}}{T}}
```

and the corresponding statistic is:

```math
t_{\mathrm{HAC}} = \frac{\bar r}{\sqrt{\hat{\Omega}/T}}
```

The primary reversion inference uses a finite predeclared lag length rather than choosing the lag after observing the resulting statistic.

Later robustness analysis varies the HAC lag to test whether the inference is sensitive to that assumption.

### 4.3 Circular block bootstrap

An iid bootstrap would destroy serial dependence by resampling individual returns independently.

The implementation instead resamples contiguous blocks.

For block length \(b\), a sampled block beginning at index \(j\) is:

```math
(r_j,r_{j+1},\ldots,r_{j+b-1})
```

with indexing wrapped circularly at the end of the sample.

Enough blocks are sampled with replacement to construct a bootstrap path of length \(T\).

For replication \(m\), the reconstructed return path is denoted:

```math
r_1^{*(m)},\ldots,r_T^{*(m)}
```

From each bootstrap sample the implementation recomputes the mean return and annualized Sharpe ratio.

The empirical 2.5% and 97.5% quantiles form the reported 95% intervals:

```math
CI_{\mu} = \left[Q_{0.025}(\bar r^*),Q_{0.975}(\bar r^*)\right]
```

and:

```math
CI_{SR} = \left[Q_{0.025}(SR^*),Q_{0.975}(SR^*)\right]
```

The primary inference uses a fixed random seed and fixed block length so that the evidence is reproducible.

Later robustness analysis varies block length while keeping the research claim boundary unchanged.

### 4.4 Sharpe ratio

For per-period returns with mean \(\bar r\) and standard deviation \(s_r\), the annualized Sharpe ratio is:

```math
SR_{\mathrm{ann}} = \frac{\bar r}{s_r}\sqrt{A}
```

where \(A\) is the annualization factor.

For daily or session-level returns, the implementation uses \(A=252\).

The corresponding per-period Sharpe is:

```math
SR = \frac{SR_{\mathrm{ann}}}{\sqrt{A}}
```

### 4.5 Probabilistic Sharpe Ratio

The Probabilistic Sharpe Ratio asks whether an observed Sharpe exceeds a benchmark after accounting for finite sample size, skewness, and kurtosis.

Let \(\hat{SR}\) be the observed per-period Sharpe, \(SR^*\) the benchmark, \(\hat{\gamma}_3\) sample skewness, and \(\hat{\gamma}_4\) non-excess kurtosis.

The implementation forms:

```math
Z = \frac{(\hat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\hat{\gamma}_3\hat{SR}+\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}
```

and evaluates:

```math
PSR = \Phi(Z)
```

where \(\Phi\) is the standard normal cumulative distribution function.

For the ordinary PSR diagnostic, the benchmark is zero.

Unlike a raw Sharpe ratio, this statistic penalizes small samples and non-Gaussian return distributions.

### 4.6 Deflated Sharpe Ratio

When several strategy variants are evaluated, the maximum observed Sharpe can be inflated by selection.

The Deflated Sharpe Ratio adjusts the benchmark upward to reflect the number and dispersion of tried configurations.

Let the per-period Sharpe ratios across \(M\) declared trials be:

```math
SR_1,\ldots,SR_M
```

with cross-trial standard deviation:

```math
\sigma_{SR} = \mathrm{Std}(SR_1,\ldots,SR_M)
```

The implementation approximates the expected maximum Sharpe benchmark as:

```math
SR_{\mathrm{deflated}}^* = \sigma_{SR}\left[(1-\gamma)\Phi^{-1}\left(1-\frac{1}{M}\right)+\gamma\Phi^{-1}\left(1-\frac{1}{Me}\right)\right]
```

where \(\gamma\) is the Euler-Mascheroni constant.

This benchmark is then inserted into the same skewness- and kurtosis-adjusted Sharpe probability calculation:

```math
DSR = \Phi\left(\frac{(\hat{SR}-SR_{\mathrm{deflated}}^*)\sqrt{T-1}}{\sqrt{1-\hat{\gamma}_3\hat{SR}+\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}\right)
```

The declared local DSR scope contains the three OU/VWAP configurations only.

The implementation explicitly records that this is not a globally corrected multiplicity claim across every research decision made in the project.

### 4.7 Multiple-testing interpretation

If enough model variants, thresholds, universes, frequencies, and transformations are explored, an apparently attractive Sharpe can emerge by chance.

Therefore, the relevant null is not simply whether one selected strategy has positive sample Sharpe.

The stronger question is whether the observed Sharpe remains unusual relative to the opportunity for selection across declared trials.

The project handles this conservatively by:

1. freezing trial families before final inference where possible;
2. reporting the number of declared trials;
3. calculating a local DSR diagnostic;
4. explicitly disclosing that broader research multiplicity can exceed the local three-configuration scope.

### 4.8 Information coefficient

For a continuous signal score \(S_t\) and subsequent return \(r_{t+1}\), the information coefficient measures cross-observation linear association:

```math
IC = \mathrm{Corr}(S_t,r_{t+1})
```

The implementation constructs the forward return causally as:

```math
r_{t+1} = \frac{P_{t+1}}{P_t}-1
```

and pairs it with the score known at time \(t\).

For the equal-weight research series, both the signal score and forward return are averaged across the synchronized ETF universe before the correlation is calculated.

The IC is treated as a diagnostic of directional signal information, not as a replacement for executable net strategy performance.

### 4.9 Equation-to-code mapping

The primary implementation is in `src/systematic_alpha/analysis/reversion_inference.py`, with additional sensitivity checks in the later OU robustness analysis.

| Statistical object | Implementation |
|---|---|
| sample mean | `np.mean(values)` |
| naive t-statistic | `_t_statistics(...)` |
| HAC long-run variance | `_t_statistics(...)` weighted autocovariance loop |
| Bartlett weight | `1 - lag / (HAC_LAGS + 1)` |
| annualized Sharpe | `_annualized_sharpe(...)` |
| circular block bootstrap | `_bootstrap_intervals(...)` |
| PSR probability | `_sharpe_probability(...)` with zero benchmark |
| DSR benchmark | `_deflated_benchmark(...)` |
| DSR probability | `_sharpe_probability(...)` with deflated benchmark |
| sample skewness | `scipy.stats.skew(..., bias=False)` |
| sample kurtosis | `scipy.stats.kurtosis(..., fisher=False, bias=False)` |
| information coefficient | correlation of `signal_score` and `forward_return` |
| HAC sensitivity | multiple predeclared lag values in robustness analysis |
| bootstrap sensitivity | multiple block lengths in robustness analysis |

### 4.10 Interpretation boundary

No single inference statistic is treated as decisive.

The research separates several questions:

1. is the average return distinguishable from zero under an iid assumption?
2. does that conclusion survive serial-correlation adjustment?
3. do block-bootstrap intervals exclude economically neutral values?
4. is the observed Sharpe credible after finite-sample skewness and kurtosis adjustment?
5. does the Sharpe remain credible after accounting for declared model trials?
6. does the underlying continuous score contain measurable forward-return information?

Agreement across these diagnostics strengthens evidence.

Disagreement is preserved rather than collapsed into a single favorable statistic.

The project therefore treats statistical inference as evidence calibration, not as a mechanism for manufacturing significance.

---

## 5. Portfolio Mathematics

### 5.1 Return vector and covariance matrix

Let the vector of sleeve returns at time \(t\) be:

```math
\mathbf r_t = (r_{1,t},r_{2,t},\ldots,r_{N,t})^\top
```

For \(N\) sleeves, the covariance matrix is:

```math
\Sigma = \mathrm{Cov}(\mathbf r_t)
```

with element:

```math
\Sigma_{ij} = \mathrm{Cov}(r_i,r_j)
```

The diagonal contains individual sleeve variances:

```math
\Sigma_{ii}=\sigma_i^2
```

while off-diagonal terms measure dependence between sleeves.

For portfolio weights \(\mathbf w\), portfolio variance is:

```math
\sigma_p^2=\mathbf w^\top\Sigma\mathbf w
```

### 5.2 Spectral decomposition

For a symmetric covariance or correlation matrix, spectral decomposition gives:

```math
\Sigma = Q\Lambda Q^\top
```

where \(Q\) contains orthonormal eigenvectors and:

```math
\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_N)
```

contains the eigenvalues.

Large concentration in the first few eigenvalues indicates that apparently different sleeves are dominated by a small number of common risk directions.

The implementation explicitly checks the covariance spectrum and positive-semidefinite consistency rather than assuming the covariance estimate is numerically valid.

### 5.3 Entropy effective rank

Normalize non-negative eigenvalues into spectral probabilities:

```math
p_i=\frac{\lambda_i}{\sum_{j=1}^{N}\lambda_j}
```

The spectral entropy is:

```math
H=-\sum_{i:p_i>0}p_i\log p_i
```

The entropy effective rank is:

```math
r_{\mathrm{eff}}=\exp(H)
```

If one eigenvalue dominates, effective rank approaches one.

If risk is distributed evenly across \(N\) orthogonal directions, effective rank approaches \(N\).

Effective rank therefore measures the dimensionality of diversification rather than merely counting the number of sleeves.

### 5.4 Diversification ratio

For portfolio weights \(\mathbf w\), define the weighted average standalone volatility:

```math
\sum_{i=1}^{N}w_i\sigma_i
```

and portfolio volatility:

```math
\sigma_p=\sqrt{\mathbf w^\top\Sigma\mathbf w}
```

The diversification ratio is:

```math
DR(\mathbf w)=\frac{\sum_{i=1}^{N}w_i\sigma_i}{\sqrt{\mathbf w^\top\Sigma\mathbf w}}
```

For the diversification diagnostic, the implementation uses equal weights across the six frozen sleeves:

```math
w_i=\frac{1}{6}
```

A diversification ratio above one indicates that combining imperfectly correlated sleeves reduces portfolio volatility relative to the weighted average of standalone volatilities.

It does not imply that the portfolio has positive expected return.

### 5.5 Weight concentration

For allocation weights \(w_i\), the Herfindahl concentration measure is:

```math
HHI=\sum_{i=1}^{N}w_i^2
```

The associated effective sleeve count is:

```math
N_{\mathrm{eff}}=\frac{1}{HHI}
```

For equal weights across \(N\) sleeves:

```math
HHI=\frac{1}{N}, \qquad N_{\mathrm{eff}}=N
```

As allocation concentrates in fewer sleeves, \(HHI\) increases and the effective sleeve count falls.

### 5.6 Equal-weight allocation

The simplest allocation rule assigns identical capital to all \(N\) sleeves:

```math
w_i=\frac{1}{N}
```

so:

```math
\mathbf 1^\top\mathbf w=1
```

Equal weighting does not use estimated expected returns, volatility forecasts, or covariance optimization.

It therefore serves as a low-estimation-risk benchmark against which more adaptive allocation rules can be compared.

### 5.7 Inverse-volatility allocation

Let \(\hat{\sigma}_i\) denote the volatility estimate for sleeve \(i\) obtained from the training sample.

Before applying portfolio constraints, inverse-volatility weights are proportional to:

```math
\tilde w_i=\frac{1}{\hat{\sigma}_i}
```

and normalized as:

```math
w_i=\frac{\tilde w_i}{\sum_{j=1}^{N}\tilde w_j}
```

This allocates less capital to individually volatile sleeves and more capital to lower-volatility sleeves.

The implementation additionally imposes a maximum weight constraint.

If a candidate inverse-volatility weight exceeds the cap, that sleeve is fixed at the maximum allowed weight and the remaining portfolio mass is redistributed across the uncapped sleeves.

This water-filling procedure continues until:

```math
\sum_{i=1}^{N}w_i=1
```

with:

```math
0\leq w_i\leq w_{\max}
```

### 5.8 Minimum-variance objective

The third allocation rule minimizes estimated portfolio variance:

```math
\min_{\mathbf w}\quad \mathbf w^\top\hat{\Sigma}\mathbf w
```

subject to:

```math
\mathbf 1^\top\mathbf w=1
```

and:

```math
0\leq w_i\leq w_{\max}
```

The gradient of the quadratic objective is:

```math
\nabla_{\mathbf w}\left(\mathbf w^\top\hat{\Sigma}\mathbf w\right)=2\hat{\Sigma}\mathbf w
```

The implementation supplies this analytical gradient to the constrained optimizer.

### 5.9 Ledoit-Wolf covariance shrinkage

Minimum-variance optimization is highly sensitive to covariance estimation error.

A noisy sample covariance matrix can produce unstable portfolio weights, especially when correlations are high or the estimation sample is limited.

The constrained minimum-variance rule therefore uses a Ledoit-Wolf shrinkage covariance estimator.

In general form, the estimator can be represented as:

```math
\hat{\Sigma}_{LW}=(1-\delta)\hat{\Sigma}_{sample}+\delta F
```

where \(F\) is a structured shrinkage target and \(\delta\in[0,1]\) is the estimated shrinkage intensity.

The purpose of shrinkage is not to increase expected return. It reduces estimation variance by trading some sample specificity for greater covariance stability.

The optimizer therefore solves:

```math
\min_{\mathbf w}\quad \mathbf w^\top\hat{\Sigma}_{LW}\mathbf w
```

under the same full-investment, long-only, and maximum-weight constraints.

### 5.10 Allocation turnover

Let \(\mathbf w_k\) denote the target weight vector for walk-forward fold \(k\).

Allocation turnover at the fold transition is measured as:

```math
TO_k^{alloc}=\sum_{i=1}^{N}|w_{i,k}-w_{i,k-1}|
```

With proportional allocation cost rate \(c_a\):

```math
Cost_k^{alloc}=c_a\,TO_k^{alloc}
```

The allocation layer therefore has its own turnover cost in addition to transaction costs already embedded inside the underlying strategy sleeves.

The implementation explicitly avoids charging the sleeve-level strategy cost twice.

### 5.11 Fixed-holdings weight dynamics

Target allocation weights are established at the beginning of a walk-forward test fold.

Within the fold, the implementation does not mechanically rebalance the portfolio back to those targets after every return observation.

Let the pre-return weight of sleeve \(i\) at time \(t\) be \(w_{i,t}\), and let sleeve return be \(r_{i,t}\).

The gross portfolio return is:

```math
r_{p,t} = \sum_{i=1}^{N}w_{i,t}r_{i,t}
```

Portfolio wealth therefore grows by:

```math
G_t = 1+r_{p,t}
```

After returns are realized, the value of sleeve \(i\) is proportional to:

```math
w_{i,t}(1+r_{i,t})
```

so its new portfolio weight becomes:

```math
w_{i,t}^{post} = \frac{w_{i,t}(1+r_{i,t})}{1+r_{p,t}}
```

The next observation begins with:

```math
w_{i,t+1}=w_{i,t}^{post}
```

Thus, better-performing sleeves naturally gain portfolio weight and weaker sleeves lose weight even when no trade occurs.

This distinction separates fixed holdings from constant-weight daily rebalancing.

The implementation records both pre-return and post-return weights so the entire weight path is auditable.

### 5.12 Fold-transition rebalancing

At the start of a new walk-forward fold, newly estimated target weights may differ from the weights inherited from the previous fold.

If \(\mathbf w_k^{target}\) is the new target and \(\mathbf w_{k-1}^{end}\) is the previous fold ending weight vector, rebalance turnover is:

```math
TO_k^{rebalance} = \sum_{i=1}^{N}\left|w_{i,k}^{target}-w_{i,k-1}^{end}\right|
```

This is more faithful to fixed-holdings accounting than comparing every new target with the previous fold initial target.

The strategy therefore distinguishes allocation decisions from passive weight drift between allocation dates.

### 5.13 Historical Value at Risk

Portfolio risk is evaluated directly from realized out-of-sample portfolio returns rather than assuming a Gaussian return distribution.

Let \(q_{0.05}\) denote the empirical 5th percentile of portfolio returns:

```math
q_{0.05}=Q_{0.05}(r_{p,t})
```

The reported 95% historical Value at Risk is expressed as a non-negative loss magnitude:

```math
VaR_{95}=\max(0,-q_{0.05})
```

This means that approximately five percent of observed portfolio returns lie at or below the corresponding return threshold.

### 5.14 Historical Expected Shortfall

Value at Risk describes a quantile but does not measure the severity of losses beyond that quantile.

Define the historical tail set:

```math
\mathcal T=\{r_{p,t}:r_{p,t}\leq q_{0.05}\}
```

Historical Expected Shortfall is the magnitude of the average return inside this tail:

```math
ES_{95}=\max\left(0,-\frac{1}{|\mathcal T|}\sum_{r\in\mathcal T}r\right)
```

Expected Shortfall therefore captures tail-loss severity that VaR alone can conceal.

### 5.15 Wealth and maximum drawdown

Starting from normalized wealth \(W_0=1\), portfolio wealth evolves as:

```math
W_t=\prod_{j=1}^{t}(1+r_{p,j})
```

The running peak is:

```math
M_t=\max_{0\leq j\leq t}W_j
```

and drawdown is:

```math
D_t=\frac{W_t}{M_t}-1
```

Maximum drawdown is:

```math
MDD=\min_t D_t
```

The sign convention retains drawdown as a non-positive return quantity, while VaR and Expected Shortfall are reported as non-negative loss magnitudes.

### 5.16 Equation-to-code mapping

Portfolio construction and validation are implemented primarily in `src/systematic_alpha/analysis/strategy_diversification.py`, `src/systematic_alpha/analysis/portfolio_allocation_validation.py`, and `src/systematic_alpha/analysis/causal_portfolio_finalization.py`.

| Mathematical object | Implementation |
|---|---|
| covariance matrix \(\Sigma\) | sleeve return covariance diagnostics |
| covariance eigenvalues | `np.linalg.eigvalsh(...)` |
| entropy effective rank | `calculate_entropy_effective_rank(...)` |
| equal-weight diversification ratio | `calculate_equal_weight_diversification_ratio(...)` |
| equal weights | `calculate_equal_weights()` |
| capped inverse-volatility allocation | `calculate_inverse_volatility_weights(...)` |
| Ledoit-Wolf covariance | `LedoitWolf(assume_centered=False)` |
| constrained variance objective | `weights @ covariance @ weights` |
| objective gradient | `2.0 * covariance @ weights` |
| concentration | `herfindahl_concentration` |
| effective sleeve count | `1.0 / herfindahl` |
| fixed-holdings portfolio return | current weights dot sleeve returns |
| drifting weights | `current * (1 + sleeve_return) / growth` |
| historical \(VaR_{95}\) | negative 5% empirical return quantile |
| historical \(ES_{95}\) | negative mean of returns below the 5% quantile |
| maximum drawdown | compounded wealth relative to running maximum |

### 5.17 Interpretation boundary

Portfolio diversification is treated as a risk-combination property, not as a mechanism for converting weak individual strategies into reliable alpha.

The allocation research deliberately avoids expected-return optimization, leverage, short portfolio weights, borrowing, sleeve removal, and ex-post winner selection.

All three predeclared allocation rules are retained as evidence even when one rule appears economically stronger than another.

This keeps covariance estimation, allocation mechanics, and realized profitability as distinct research questions.

---

## 6. Market Microstructure and Event Time

### 6.1 Clock time versus event time

Conventional bars partition the market by elapsed clock time.

For a fixed interval \(\Delta\), a time bar aggregates all trades whose timestamps fall inside the same interval.

Event-time bars instead advance when a specified amount of market activity has accumulated.

This changes the sampling clock from calendar time to trading activity.

The project evaluates three event-time alternatives:

- tick bars, based on trade count;
- volume bars, based on shares traded;
- dollar bars, based on traded notional.

### 6.2 Tick bars

For threshold \(K\), a tick bar closes after approximately \(K\) trades have accumulated.

If \(N_t\) is cumulative trade count, the event clock advances when:

```math
\Delta N_t \geq K
```

Tick bars therefore compress high-trading-activity periods and stretch quiet periods in calendar time.

### 6.3 Volume bars

Let trade size be \(q_j\).

A volume bar closes when accumulated traded shares reach threshold \(V^*\):

```math
\sum_{j\in\mathrm{bar}}q_j \geq V^*
```

Volume time therefore measures market activity in units of shares rather than number of transactions.

### 6.4 Dollar bars

For trade price \(P_j\) and size \(q_j\), trade notional is:

```math
D_j=P_jq_j
```

A dollar bar closes when:

```math
\sum_{j\in\mathrm{bar}}P_jq_j \geq D^*
```

Dollar bars normalize activity by economic value, so identical share volume at very different price levels does not represent identical event size.

### 6.5 Matched-count sampling comparison

To compare sampling methods without deliberately giving one method many more observations, the event-bar diagnostic constructs thresholds targeting approximately the same number of bars.

For target count \(B\) and total trades \(N\):

```math
K=\left\lceil\frac{N}{B}\right\rceil
```

For total volume \(V\):

```math
V^*=\frac{V}{B}
```

and for total traded notional \(D\):

```math
D^*=\frac{D}{B}
```

The corresponding clock-time interval is chosen from total sample duration divided by the target bar count.

These thresholds are engineering comparators, not optimized strategy parameters.

### 6.6 Aggregation conservation laws

A valid resampling procedure should preserve the underlying trading activity.

For trade count:

```math
\sum_{b=1}^{B}N_b=N_{\mathrm{input}}
```

For volume:

```math
\sum_{b=1}^{B}V_b=V_{\mathrm{input}}
```

and for dollar value:

```math
\sum_{b=1}^{B}D_b=D_{\mathrm{input}}
```

The implementation records explicit conservation errors for each sampling method.

A statistically attractive bar construction is not accepted if it loses or duplicates trades, shares, or notional.

### 6.7 VWAP within an aggregated bar

For trades inside one bar, volume-weighted average price is:

```math
VWAP_b=\frac{\sum_{j\in b}P_jq_j}{\sum_{j\in b}q_j}
```

The bar also retains open, high, low, close, volume, trade count, dollar value, and start/end timestamps.

### 6.8 Realized variance

Let intraday log-return pieces for one regular session be:

```math
r_{1},r_{2},\ldots,r_{M}
```

The implementation constructs regular-session realized variance as:

```math
RV=\sum_{i=1}^{M}r_i^2
```

and realized absolute variation as:

```math
AV=\sum_{i=1}^{M}|r_i|
```

Annualized regular-session realized volatility is:

```math
\sigma_{RV,\mathrm{ann}}=\sqrt{252\,RV}
```

Overnight variance is separately measured from the overnight log return:

```math
RV_{\mathrm{overnight}}=r_{\mathrm{overnight}}^2
```

giving total daily realized variance:

```math
RV_{\mathrm{total}}=RV_{\mathrm{regular}}+RV_{\mathrm{overnight}}
```

### 6.9 Bipower variation and jump proxy

Bipower variation is constructed from adjacent absolute intraday returns:

```math
BV=\frac{\pi}{2}\sum_{i=2}^{M}|r_i||r_{i-1}|
```

The implementation defines a non-negative jump-variation proxy as:

```math
JV=\max(0,RV-BV)
```

and jump share as:

```math
JS=\frac{JV}{RV}
```

when realized variance is positive.

This is treated as a descriptive decomposition rather than proof that every excess movement is a structural price jump.

### 6.10 Parkinson range estimator

Using session high \(H_t\) and low \(L_t\), the Parkinson variance estimator is:

```math
\sigma_{P,t}^2=\frac{\left[\log(H_t/L_t)\right]^2}{4\log 2}
```

and its annualized volatility representation is:

```math
\sigma_{P,\mathrm{ann}}=\sqrt{252\,\sigma_{P,t}^2}
```

Range-based and return-based volatility measures are retained as complementary diagnostics rather than assumed to be interchangeable.

### 6.11 Intraday activity seasonality

For bar \(i\) within a session, volume share is:

```math
VS_i=\frac{Volume_i}{\sum_{j=1}^{M}Volume_j}
```

Trade-count share is:

```math
TS_i=\frac{Trades_i}{\sum_{j=1}^{M}Trades_j}
```

and squared-return share is:

```math
RS_i=\frac{r_i^2}{\sum_{j=1}^{M}r_j^2}
```

Aggregating these quantities by bar number and local time reveals systematic intraday concentration in trading activity and volatility.

This matters because equal clock-time intervals are not equal-information intervals.

### 6.12 Sampling-time interpretation

One event-time return and one 15-minute return do not represent the same amount of elapsed time.

Therefore, a one-event-ahead predictive horizon under dollar bars cannot be interpreted as mechanically equivalent to a one-bar-ahead horizon under fixed time bars.

The project keeps this distinction explicit and does not replace the primary 15-minute research frequency based on a small event-bar diagnostic sample.

### 6.13 Equation-to-code mapping

The main implementation is in `src/systematic_alpha/analysis/event_bar_diagnostics.py`, `src/systematic_alpha/analysis/event_time_finalization.py`, and `src/systematic_alpha/analysis/volatility_seasonality.py`.

| Mathematical object | Implementation |
|---|---|
| tick threshold \(K\) | `calculate_event_thresholds(...)` |
| volume threshold \(V^*\) | `calculate_event_thresholds(...)` |
| dollar threshold \(D^*\) | `calculate_event_thresholds(...)` |
| time-bar interval | `infer_time_rule(...)` |
| VWAP | dollar value divided by volume |
| conservation diagnostics | `build_conservation_table(...)` |
| realized variance | `realized_variance_regular` |
| bipower variation | `bipower_variation` |
| jump proxy | `jump_variation_proxy` |
| Parkinson variance | `parkinson_variance` |
| volume seasonality | `volume_share` |
| trade-count seasonality | `trade_count_share` |
| volatility seasonality | `squared_return_share` |

### 6.14 Interpretation boundary

Event-time sampling is treated as a microstructure research tool, not as an automatic source of alpha.

The project distinguishes three separate claims:

1. whether an event-bar implementation conserves the underlying trades correctly;
2. whether event-time sampling changes return and activity distributions;
3. whether those differences ultimately improve executable predictive performance.

Only the first two can be investigated from a limited event-time diagnostic sample without overclaiming strategy evidence.

---

## 7. Execution and Order-State Mathematics

### 7.1 Target position versus executed position

A trading model produces a target position, but a target is not the same object as an executed holding.

Let the signal generated after processing bar \(t\) imply target:

```math
w_t^* \in \{-1,0,+1\}
```

The event-driven replay stores that target as pending state and executes it only on the following observation.

Thus:

```math
w_{t+1}^{exec}=w_t^*
```

The position change is:

```math
\Delta w_{t+1}=w_{t+1}^{exec}-w_t^{exec}
```

and turnover is:

```math
TO_{t+1}=|\Delta w_{t+1}|
```

This explicitly separates forecasting state from execution state.

### 7.2 Event sequence

For each bar, the replay follows a deterministic event order.

The main event types are:

1. `MarketBarEvent`;
2. `FillEvent` when a pending order changes position;
3. `PortfolioSnapshot`;
4. `SignalEvent`;
5. `TargetPositionOrderEvent` when the new target differs from the executed position.

This ordering ensures that the current bar cannot generate a signal and retrospectively receive a fill on the same observation.

An order created at bar \(t\) carries:

```math
execute\_bar\_index=t+1
```

and a pending order is required to produce the corresponding changed-position fill.

### 7.3 Return and equity accounting

Let previous portfolio equity be \(E_{t-1}\), executed position \(w_t\), and asset return \(r_t\).

Gross strategy return is:

```math
r_t^{gross}=w_t r_t
```

With proportional turnover cost \(c_t\):

```math
r_t^{net}=r_t^{gross}-c_t
```

The monetary transaction cost is:

```math
C_t=E_{t-1}c_t
```

and ending equity is:

```math
E_t=E_{t-1}(1+r_t^{net})
```

The replay also decomposes equity into cash and holdings value.

For unit-notional position representation:

```math
Cash_t=E_{t-1}(1-w_t)-C_t
```

and:

```math
Holdings_t=w_tE_{t-1}(1+r_t)
```

which reconciles to:

```math
E_t=Cash_t+Holdings_t
```

The implementation requires every accounting component to remain finite and ending equity to remain strictly positive.

This makes accounting consistency an invariant rather than something inferred from final cumulative return.

### 7.4 Execution shortfall

Paper execution quality is measured relative to the price available when the trading decision was formed.

Let \(P_d\) be decision price and \(P_f\) realized fill price.

Define side sign:

```math
s=\begin{cases}+1,&\text{buy}\\-1,&\text{sell}\end{cases}
```

Total execution shortfall in basis points is:

```math
IS_{total}=s\frac{P_f-P_d}{P_d}\times10{,}000
```

Positive shortfall represents adverse execution relative to the decision price under this sign convention.

### 7.5 Delay, spread, and residual components

Let bid and ask at arrival be \(B\) and \(A\).

Arrival midpoint is:

```math
M=\frac{A+B}{2}
```

and arrival touch is:

```math
T=\begin{cases}A,&\text{buy}\\B,&\text{sell}\end{cases}
```

Delay cost is:

```math
IS_{delay}=s\frac{M-P_d}{P_d}\times10{,}000
```

Spread cost is:

```math
IS_{spread}=s\frac{T-M}{P_d}\times10{,}000
```

Residual execution cost is:

```math
IS_{residual}=s\frac{P_f-T}{P_d}\times10{,}000
```

The decomposition satisfies:

```math
IS_{total}=IS_{delay}+IS_{spread}+IS_{residual}
```

The implementation explicitly verifies this identity numerically rather than assuming the components reconcile.

### 7.6 Round-trip realized P&L

For long quantity \(q\), entry fill \(P_{in}\), and exit fill \(P_{out}\):

```math
PnL_{gross}=q(P_{out}-P_{in})
```

For a short position:

```math
PnL_{gross}=q(P_{in}-P_{out})
```

With total commission \(C\):

```math
PnL_{net}=PnL_{gross}-C
```

Each round trip must contain exactly one entry and one exit, equal quantities, opposite sides, matching symbol and purpose, and chronologically ordered fills.

### 7.7 Execution latency

The execution path records several latency intervals.

Decision-to-submit latency is:

```math
L_{decision} = t_{submit}-t_{decision}
```

Broker acknowledgement latency is:

```math
L_{ack}=t_{broker}-t_{submit}
```

and fill latency is:

```math
L_{fill}=t_{fill}-t_{submit}
```

The timestamps must satisfy chronological ordering, and the quote used at submission must remain within the configured freshness limit.

### 7.8 State invariants

The execution system enforces several hard invariants:

- a pending target executes only on its declared future bar;
- a pending changed-position order must generate one fill;
- turnover equals absolute position change;
- costs are proportional to turnover under the backtest convention;
- cash plus holdings reconcile to ending equity;
- execution timestamps are chronological;
- quotes cannot be crossed or stale at submission;
- round-trip legs must be internally consistent.

These checks convert execution assumptions into testable contracts.

### 7.9 Equation-to-code mapping

The principal implementation is in `src/systematic_alpha/analysis/trend_family_event_replay.py` and `src/systematic_alpha/analysis/execution_performance_validation.py`.

| Mathematical object | Implementation |
|---|---|
| target position \(w_t^*\) | `SignalEvent.target_position` |
| next-observation order | `TargetPositionOrderEvent.execute_bar_index` |
| executed position | replay state `executed_position` |
| position change | `position_change` |
| turnover | `abs(position_change)` |
| fill | `FillEvent` |
| equity state | `PortfolioSnapshot` |
| total shortfall | `total_shortfall_bps` |
| delay component | `delay_bps` |
| spread component | `spread_bps` |
| residual component | `residual_bps` |
| fill latency | `fill_latency_ms` |
| round-trip realized P&L | round-trip P&L reconstruction |

### 7.10 Interpretation boundary

An event-driven replay is more operationally faithful than a vectorized return calculation, but it is still a model of execution.

Observed fills, spread, latency, queue position, partial fills, market impact, and venue-specific microstructure can differ from simplified research assumptions.

The purpose of the execution layer is therefore not to claim perfect realism, but to make every remaining assumption explicit, causal, and auditable.

---

## 8. Research Chronology, Walk-Forward Causality, and Locked Evaluation

### 8.1 Expanding walk-forward structure

For fold \(k\), define a training set:

```math
\mathcal D_k^{train}=\{t:T_{0}\leq t<T_{k}^{train,end}\}
```

and a non-overlapping test set:

```math
\mathcal D_k^{test}=\{t:T_{k}^{test,start}\leq t<T_{k}^{test,end}\}
```

with:

```math
T_{k}^{train,end}\leq T_{k}^{test,start}
```

The training history expands through time while each test interval remains strictly subsequent to the information used for model estimation.

The implementation additionally requires test folds to be chronological and non-overlapping.

### 8.2 Information-set constraint

For any parameter estimate \(\hat{\theta}_k\) used in fold \(k\):

```math
\hat{\theta}_k=f(\mathcal D_k^{train})
```

and therefore:

```math
\hat{\theta}_k\in\mathcal F_{T_k^{test,start}-}
```

No observation from the test interval is permitted to determine that fold parameter estimate.

This applies not only to forecasting coefficients, but also to covariance estimates, hedge ratios, allocation weights, stability gates, and other adaptive state.

### 8.3 Indicator history versus execution state

Historical observations prior to a test boundary may be required to warm up recursive indicators or rolling windows.

This does not imply that a position earned before the test period should be carried into the evaluation interval.

The implementation therefore separates indicator history from execution state.

At a test reset boundary:

```math
w_{0}^{exec}=0
```

and initial test turnover and costs are reconstructed from the reset state.

This prevents a test result from inheriting an economically unobserved pre-test position.

### 8.4 Event-driven walk-forward evaluation

Each walk-forward test fold is replayed through the event-driven engine over complete session boundaries.

The evaluation interval is half-open:

```math
[T_{start},T_{end})
```

and boundaries must coincide with complete trading-session edges.

A test window is rejected if it splits a trading session.

For each strategy-fold combination, the event-driven ledger is independently compared with the vectorized reference implementation.

Discrete states require exact equality, while floating-point accounting quantities must lie within the declared numerical tolerance.

Thus the walk-forward test simultaneously checks:

- chronological train/test separation;
- execution-state reset;
- one-observation delayed execution;
- event sequencing;
- position and turnover accounting;
- gross and net return parity.

### 8.5 Causal model comparison

Performance across folds is aggregated only after each fold has been evaluated independently under its historical information set.

If fold returns are \(r_t^{(k)}\), aggregate out-of-sample evidence is formed from the chronological concatenation:

```math
\{r_t^{OOS}\}=\mathcal D_1^{test}\Vert\mathcal D_2^{test}\Vert\cdots\Vert\mathcal D_K^{test}
```

where \(\Vert\) denotes temporal concatenation rather than re-estimation on the combined test sample.

This preserves the distinction between repeated historical experiments and one retrospectively fitted full-sample backtest.

### 8.6 Locked final evaluation

The final holdout is treated as a one-time evaluation interval rather than another development sample.

Let the development period end before locked interval:

```math
\mathcal D_{dev}\cap\mathcal D_{locked}=\varnothing
```

Before locked data can be evaluated, the implementation requires explicit authorization and verifies the frozen development state.

This includes cryptographic hashes of:

- the canonical development dataset;
- previously generated research artifacts;
- the source files governing the frozen models and evaluation logic.

If a frozen source or required artifact differs from its recorded hash, the final evaluation is rejected.

### 8.7 Development history as warmup only

The locked evaluation may concatenate development history and locked observations for feature construction when a model requires historical state.

However, development data serve only as warmup information.

Performance observations are restricted to:

```math
T_{locked,start}\leq t<T_{locked,end}
```

and execution state is reset at the beginning of the locked period.

This distinction allows causal rolling features to exist at the first holdout observation without allowing development-period P&L or positions to leak into the final result.

### 8.8 One-time holdout principle

Once the holdout is revealed, it cannot remain statistically equivalent to untouched future data if it is repeatedly used for model selection.

The project therefore freezes the model set, execution convention, transaction cost, allocation definition, and reporting rules before the final evaluation.

After access, the protocol prohibits using the locked result to:

- change model parameters;
- select only favorable models;
- alter the market universe;
- alter the cost assumption;
- redefine the evaluation interval;
- suppress negative results.

All frozen model results are retained in the final evidence set.

### 8.9 Reproducibility and immutable evidence

The final-evaluation artifact bundle is written atomically and is not silently overwritten.

For artifact byte sequence \(A_j\), the project records:

```math
h_j=SHA256(A_j)
```

These hashes provide a deterministic fingerprint of the evidence that was generated under the frozen protocol.

The role of hashing is not statistical inference; it is provenance control.

It allows later reviewers to determine whether the dataset, source, or reported evidence changed after the evaluation boundary was crossed.

### 8.10 Equation-to-code mapping

The principal implementations are `src/systematic_alpha/analysis/trend_family_walk_forward.py`, `src/systematic_alpha/analysis/trend_family_event_walk_forward.py`, and `src/systematic_alpha/analysis/locked_final_test.py`.

| Research object | Implementation |
|---|---|
| expanding folds | `build_walk_forward_folds()` |
| chronological train/test partitions | fold construction and validation |
| test-state reset | fold execution reset logic |
| complete-session boundaries | event replay evaluation-window validation |
| event/vectorized parity | walk-forward parity records |
| locked interval | `LOCKED_START`, `LOCKED_END_EXCLUSIVE` |
| authorization gate | `require_authorization(...)` |
| frozen dataset verification | `verify_frozen_development_state(...)` |
| frozen source verification | SHA-256 source inventory comparison |
| holdout execution reset | `execution_state_reset_at_test_start` |
| immutable final artifacts | atomic final-test artifact writer |

### 8.11 Interpretation boundary

Walk-forward testing reduces hindsight bias but does not eliminate every form of research overfitting.

Likewise, a locked holdout is most informative when the research process preceding it is genuinely frozen.

The project therefore treats chronology, code provenance, execution-state isolation, and one-time evaluation as part of the statistical design rather than administrative details.

---
