# Day 30 Causal Cointegration Chronology Specification

Version: `causal_cointegration_chronology_v1_1_svd_ols`

## Purpose and preservation boundary

Day 30 corrects one chronology defect in the frozen Day 14 feasibility method.
The historical fold table estimates each expanding-training hedge ratio and
compares it with a 2020–2025 all-development hedge ratio. For folds before the
end of 2025, that reference was unavailable at the fold origin.

The historical `cointegration_feasibility.py` module, public function, runner,
tests, reports, and artifacts remain unchanged. The corrected chronology is an
independent implementation. It does not overwrite or reinterpret historical
Day 14 evidence.

This phase uses synthetic data only. It creates no runner, empirical artifact,
report, chart, or notebook and authorizes no parameter selection or trading
branch.

## Frozen contract

The corrected path reuses the Day 14:

- ordered pairs and fixed regression orientations: SPY on QQQ, SPY on IWM,
  and QQQ on IWM;
- 2022–2025 expanding folds with a fixed 2020-01-02 training origin;
- log-price representation;
- intercept-plus-slope ordinary least-squares model;
- ADF deterministic-term and AIC lag conventions;
- Engle–Granger `trend="c"`, AIC lag selection, and Holm correction at 5%;
- interpretable-beta interval `[0.10, 10.00]`;
- maximum relative beta drift of `0.25`;
- minimum of three stationary folds;
- conditional OU parameter and half-life gates.

No pair, orientation, statistical threshold, allocation rule, or eligibility
gate is selected from the synthetic results.

## Causal OLS estimation

For pair orientation (Y/X) and fold (f), only synchronized training
observations satisfying

\[
t_f^{\mathrm{start}} \le t < t_f^{\mathrm{train\ end}}
\]

are admitted. The cointegrating regression is

\[
y_t=\alpha_f+\beta_f x_t+u_t.
\]

With

\[
Z_f=
\begin{bmatrix}
1 & x_1\\
\vdots & \vdots\\
1 & x_{n_f}
\end{bmatrix},
\qquad
\widehat\gamma_f=
\begin{bmatrix}
\widehat\alpha_f\\
\widehat\beta_f
\end{bmatrix},
\]

the v1.1 implementation solves the least-squares problem directly by singular
value decomposition:

\[
(\widehat\gamma_f,\mathrm{RSS}_f,r_f,s_f)
=\operatorname{lstsq}(Z_f,y_f;\ rcond=\mathrm{None}),
\]

where (r_f) is numerical rank and
(s_f=(s_{f,\max},s_{f,\min})) contains the singular values in descending
order. For a full-rank design this is algebraically equivalent to

\[
\widehat\gamma_f=(Z_f^\top Z_f)^{-1}Z_f^\top y_f.
\]

The normal-equation calculation `solve(Z.T @ Z, Z.T @ y)` is prohibited for the
cointegrating regression because

\[
\kappa_2(Z_f^\top Z_f)=\kappa_2(Z_f)^2.
\]

Forming (Z_f^\top Z_f) therefore squares the design's two-norm condition number
and unnecessarily amplifies numerical error. `np.linalg.lstsq` is called with
`rcond=None`; numerical rank must equal two. Nonfinite coefficients, residuals,
or singular values, a nonpositive smallest singular value, invalid singular-
value ordering, or exact rank deficiency fails closed.

Every fold reports `design_rank`, `largest_singular_value`,
`smallest_singular_value`, and

\[
\mathrm{design\ condition\ number}
=\frac{s_{f,\max}}{s_{f,\min}}.
\]

The condition number is disclosure, not a gate. This version introduces no
result-selected or tunable condition-number threshold. A full-rank but poorly
conditioned design is retained with its diagnostic rather than silently
rejected.

The maximum timestamp used for the OLS estimate, unit-root checks,
Engle–Granger test, Holm gate, and residual-stationarity test must be strictly
earlier than the fold's exclusive training-end boundary. Equivalently,

\[
t_f^{\mathrm{train\ end,inclusive}}
=t_f^{\mathrm{train\ end,exclusive}}-1\text{ nanosecond}
\]

and the implemented audit is

\[
t_f^{\max\mathrm{info}}
\le t_f^{\mathrm{train\ end,inclusive}}.
\]

Equality at the inclusive training end is legitimate and is not rejected. An
observation equal to the exclusive boundary remains outside training. Test-
period prices do not enter fold estimates or fold eligibility.

## Successive expanding-training stability

The first fold is a baseline:

\[
\beta_{\mathrm{reference},1}=\mathrm{unavailable},
\qquad
D_1=\mathrm{NaN}.
\]

Its status is `baseline`; it contains no fabricated reference beta. For each
later fold,

\[
\beta_{\mathrm{reference},f}=\widehat\beta_{f-1}
\]

and

\[
D_f=\frac{\left|\widehat\beta_f-
\widehat\beta_{f-1}\right|}
{\left|\widehat\beta_{f-1}\right|}.
\]

The comparison passes only when (D_f\le0.25), both estimates are finite and
numerically nonzero, and their signs agree. The numerical zero test uses
`64 * machine epsilon`; it is an admissibility check, not a denominator
stabilizer. No epsilon is added to the drift denominator. A nonfinite or
near-zero reference, a nonfinite or near-zero current beta, or a sign change
fails closed and receives an explicit status.

Pair-level causal stability uses only the three later-fold comparisons. The
overlapping expanding samples make these comparisons dependent; their mean and
maximum are diagnostics, not independent replications.

## Training-only statistical and eligibility chronology

To ensure that future test observations cannot alter same-fold eligibility,
the following are recomputed within each training window:

- level ADF with trend and intercept (`ct`) and first-difference ADF with an
  intercept (`c`), both with AIC lag selection;
- Engle–Granger testing with `trend="c"` and AIC, with Holm correction across
  the three frozen pairs within the same fold;
- residual ADF with no deterministic term (`n`) on the training OLS residual;
- interpretable-beta and successive-stability gates.

The first baseline does not fail merely because no comparison exists. A fold
eligibility row requires the two I(1) preconditions, Holm-adjusted
cointegration, interpretable beta, causal stability, and training-residual
stationarity.

The pair summary applies the frozen minimum of three stationary folds and the
later-fold causal stability gate. When all pre-OU conditions pass, the OU gate
uses intraday training observations and the last causal fold's alpha and beta,
never the all-development beta. `final_pair_eligibility` therefore preserves
the conditional Day 14 gate sequence using information available by the final
fold's training boundary.

## Ex-post statistic isolation

An OLS coefficient using every supplied 2020–2025 observation is exposed only
in `ex_post_beta_diagnostics`. Each row is labelled
`ex_post_descriptive_only` and records:

- `used_as_fold_reference = false`;
- `used_in_stability_gate = false`;
- `used_in_eligibility = false`.

The ex-post coefficient is computed after the causal ledger and is not read by
any fold or pair gate. Appending later observations may change this descriptive
table but cannot revise earlier causal fold rows.

## Fail-closed validation

The implementation rejects:

- pair identifiers or Y/X orientations outside the frozen ordered contract;
- missing, nonnumeric, or nonfinite log prices;
- singular cointegrating regressions;
- nonfinite or invalid SVD coefficients or singular values;
- duplicate or unordered daily or intraday timestamps;
- inconsistent intraday timestamp/session labels;
- train/test overlap, unordered folds, or non-frozen fold boundaries;
- any daily or intraday timestamp on or after 2026-01-01;
- a nonfinite or numerically zero successive-beta reference;
- any fold estimate whose maximum information timestamp exceeds the inclusive
  training end, equivalently reaches or exceeds the exclusive boundary.

Inputs and returned tables are defensively copied. Pair-major, fold-minor order
is deterministic.

## Mathematical disclosure and limitations

OLS in a genuine cointegrating regression is superconsistent: its coefficient
can converge faster than in an ordinary stationary regression. This does not
make the usual stationary-regression OLS t-statistics automatically valid.

Residual stationarity, not a high price-level (R^2), supplies the relevant
cointegration evidence. A visually close or high-(R^2) price relationship is
not sufficient.

Successive expanding estimates are dependent because their samples overlap.
The stability statistic is a descriptive gate and neither proves global
stationarity nor establishes profitability. A structural break can invalidate
a static cointegrating relation even when pre-break diagnostics pass; Day 30
does not add a structural-break test.

The fixed (Y/X) regression orientation is not symmetric. Reversing the pair
changes the regression, residual, and estimated hedge ratio.

The SVD estimator changes numerical implementation, not statistical method.
Synthetic parity tests compare alpha, beta, and residuals with the frozen Day 14
statsmodels OLS calculation using relative and absolute tolerance `5e-12`.
Discrete chronology, stability, stationarity, and eligibility decisions remain
exactly unchanged on the frozen v1 fixture. The tolerance accommodates floating-
point differences between equivalent SVD implementations; it is not a model-
selection threshold.

ADF and Engle–Granger tests have finite-sample power and specification
limitations. Passing them does not prove that a relation will persist.

No Johansen or VECM trading branch may be activated unless the existing
predeclared causal eligibility conditions are satisfied. Day 30 does not add or
activate such a branch.
