# Mathematical Derivations

[Back to README](../README.md) · [Methods Reference](MATHEMATICAL_METHODS.md)

This document develops the principal mathematical results used in Systematic Alpha Research from their underlying assumptions rather than presenting only their final formulas.

The companion `MATHEMATICAL_METHODS.md` documents model definitions, assumptions, implementation contracts, and equation-to-code mappings. This document focuses on derivation: how the key equations arise, what assumptions are required, and how the resulting mathematics connects to the implemented research system.

## 1. Ornstein-Uhlenbeck Dynamics from First Principles

### 1.1 Model definition

Consider the Ornstein-Uhlenbeck stochastic differential equation:

```math
dX_t=\kappa(\mu-X_t)\,dt+\sigma\,dW_t,
\qquad \kappa>0,\quad \sigma>0.
```

The drift term can be expanded as:

```math
\kappa(\mu-X_t)=\kappa\mu-\kappa X_t.
```

Hence:

```math
dX_t+\kappa X_t\,dt=\kappa\mu\,dt+\sigma\,dW_t.
```

This is a linear stochastic differential equation. The deterministic part suggests the integrating factor:

```math
I(t)=e^{\kappa t}.
```

### 1.2 Integrating-factor solution

Apply the product rule to \(e^{\kappa t}X_t\). Because \(e^{\kappa t}\) is deterministic and of finite variation, its quadratic covariation with \(X_t\) is zero. Therefore:

```math
d(e^{\kappa t}X_t)
=e^{\kappa t}\,dX_t+\kappa e^{\kappa t}X_t\,dt.
```

Substituting the OU dynamics:

```math
d(e^{\kappa t}X_t)
=e^{\kappa t}\left[\kappa(\mu-X_t)dt+\sigma dW_t\right]
+\kappa e^{\kappa t}X_tdt.
```

The \(X_t\) drift terms cancel:

```math
d(e^{\kappa t}X_t)
=\kappa\mu e^{\kappa t}dt+\sigma e^{\kappa t}dW_t.
```

Integrating from \(t\) to \(t+\Delta\):

```math
e^{\kappa(t+\Delta)}X_{t+\Delta}-e^{\kappa t}X_t
=
\kappa\mu\int_t^{t+\Delta}e^{\kappa s}ds
+
\sigma\int_t^{t+\Delta}e^{\kappa s}dW_s.
```

The deterministic integral is:

```math
\kappa\mu\int_t^{t+\Delta}e^{\kappa s}ds
=
\mu\left(e^{\kappa(t+\Delta)}-e^{\kappa t}\right).
```

Therefore:

```math
e^{\kappa(t+\Delta)}X_{t+\Delta}
=
e^{\kappa t}X_t
+
\mu\left(e^{\kappa(t+\Delta)}-e^{\kappa t}\right)
+
\sigma\int_t^{t+\Delta}e^{\kappa s}dW_s.
```

Dividing by \(e^{\kappa(t+\Delta)}\):

```math
X_{t+\Delta}
=
e^{-\kappa\Delta}X_t
+
\mu(1-e^{-\kappa\Delta})
+
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s.
```

Rearranging the deterministic terms gives the standard exact OU solution:

```math
X_{t+\Delta}
=
\mu+(X_t-\mu)e^{-\kappa\Delta}
+
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s.
```

This expression is exact; no Euler approximation has been introduced.

### 1.3 Conditional expectation

Conditioning on the information set \(\mathcal F_t\), the Brownian increment after time \(t\) has zero conditional mean:

```math
\mathbb E\left[
\int_t^{t+\Delta}e^{-\kappa(t+\Delta-s)}dW_s
\mid\mathcal F_t
\right]=0.
```

Hence:

```math
\mathbb E[X_{t+\Delta}\mid\mathcal F_t]
=
\mu+(X_t-\mu)e^{-\kappa\Delta}.
```

The conditional deviation from equilibrium therefore satisfies:

```math
\mathbb E[X_{t+\Delta}-\mu\mid\mathcal F_t]
=
e^{-\kappa\Delta}(X_t-\mu).
```

Thus \(\kappa\) controls exponential decay of expected deviations from equilibrium.

### 1.4 Conditional variance from Itô isometry

The stochastic component is:

```math
Z_{t,\Delta}
=
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s.
```

By Itô isometry:

```math
\mathbb E[Z_{t,\Delta}^2\mid\mathcal F_t]
=
\sigma^2\int_t^{t+\Delta}
e^{-2\kappa(t+\Delta-s)}ds.
```

Make the substitution:

```math
u=t+\Delta-s.
```

Then:

```math
\int_t^{t+\Delta}e^{-2\kappa(t+\Delta-s)}ds
=
\int_0^\Delta e^{-2\kappa u}du.
```

Evaluating:

```math
\int_0^\Delta e^{-2\kappa u}du
=
\left[-\frac{1}{2\kappa}e^{-2\kappa u}\right]_0^\Delta
=
\frac{1-e^{-2\kappa\Delta}}{2\kappa}.
```

Therefore:

```math
\operatorname{Var}(X_{t+\Delta}\mid X_t)
=
\frac{\sigma^2}{2\kappa}
\left(1-e^{-2\kappa\Delta}\right).
```

Because the stochastic integral is Gaussian, the exact transition law is:

```math
X_{t+\Delta}\mid X_t
\sim
\mathcal N\left(
\mu+(X_t-\mu)e^{-\kappa\Delta},
\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})
\right).
```

### 1.5 Stationary distribution

From the exact transition derived above:

```math
X_{t+\Delta}\mid X_t
\sim
\mathcal N\left(
\mu+(X_t-\mu)e^{-\kappa\Delta},
\frac{\sigma^2}{2\kappa}\left(1-e^{-2\kappa\Delta}\right)
\right).
```

For \(\kappa>0\):

```math
\lim_{\Delta\to\infty}e^{-\kappa\Delta}=0
```

and:

```math
\lim_{\Delta\to\infty}e^{-2\kappa\Delta}=0.
```

Therefore the limiting conditional mean is:

```math
\lim_{\Delta\to\infty}
\mathbb E[X_{t+\Delta}\mid X_t]
=\mu,
```

while the limiting conditional variance is:

```math
\lim_{\Delta\to\infty}
\operatorname{Var}(X_{t+\Delta}\mid X_t)
=
\frac{\sigma^2}{2\kappa}.
```

Hence the invariant distribution of the OU process is:

```math
X_{\infty}
\sim
\mathcal N\left(
\mu,
\frac{\sigma^2}{2\kappa}
\right).
```

The corresponding stationary standard deviation is:

```math
\sigma_{\infty}
=
\frac{\sigma}{\sqrt{2\kappa}}.
```

The result also exposes the economic interaction between parameters: increasing \(\sigma\) increases equilibrium dispersion, while increasing \(\kappa\) tightens the stationary distribution because deviations are pulled back toward equilibrium more rapidly.

### 1.6 Exact mapping to an AR(1) process

Suppose observations are separated by a fixed interval \(\Delta\).

The exact OU transition is:

```math
X_{t+\Delta}
=
\mu+(X_t-\mu)e^{-\kappa\Delta}
+
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s.
```

Expand the deterministic component:

```math
X_{t+\Delta}
=
\mu(1-e^{-\kappa\Delta})
+
e^{-\kappa\Delta}X_t
+
\eta_{t+\Delta},
```

where:

```math
\eta_{t+\Delta}
=
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s.
```

Define:

```math
\phi=e^{-\kappa\Delta}
```

and:

```math
a=\mu(1-\phi).
```

Then the exact discretization becomes:

```math
X_{n+1}=a+\phi X_n+\eta_{n+1}.
```

This is an AR(1) model, but importantly it is not an Euler approximation. It is the exact discrete-time representation of the continuous OU transition at regularly spaced observations.

The inverse parameter mapping follows directly.

Since:

```math
\phi=e^{-\kappa\Delta},
```

taking logarithms gives:

```math
\log\phi=-\kappa\Delta,
```

and therefore:

```math
\boxed{\kappa=-\frac{\log\phi}{\Delta}}.
```

Similarly, because:

```math
a=\mu(1-\phi),
```

we obtain:

```math
\boxed{\mu=\frac{a}{1-\phi}}.
```

The condition \(\kappa>0\) implies:

```math
0<e^{-\kappa\Delta}<1,
```

so a fitted discrete process is compatible with a positive-speed OU model only when:

```math
0<\phi<1.
```

### 1.7 Innovation variance and diffusion-scale recovery

From the stochastic integral:

```math
\eta_{n+1}
=
\sigma\int_t^{t+\Delta}
e^{-\kappa(t+\Delta-s)}dW_s,
```

the innovation variance derived through Itô isometry is:

```math
\operatorname{Var}(\eta_{n+1})
=
\frac{\sigma^2}{2\kappa}
\left(1-e^{-2\kappa\Delta}\right).
```

Using:

```math
\phi=e^{-\kappa\Delta},
```

we have:

```math
e^{-2\kappa\Delta}=\phi^2.
```

Therefore:

```math
\sigma_{\eta}^{2}
=
\frac{\sigma^2}{2\kappa}(1-\phi^2).
```

Solving for the continuous-time diffusion coefficient gives:

```math
\sigma^2
=
\sigma_{\eta}^{2}
\frac{2\kappa}{1-\phi^2}
```

and hence:

```math
\boxed{
\sigma
=
\sigma_{\eta}
\sqrt{\frac{2\kappa}{1-\phi^2}}
}.
```

There is a useful consistency check between the discrete and continuous stationary variances.

For a stationary AR(1):

```math
\operatorname{Var}(X)
=
\phi^2\operatorname{Var}(X)
+
\sigma_{\eta}^{2}.
```

Move the autoregressive variance term to the left:

```math
(1-\phi^2)\operatorname{Var}(X)
=
\sigma_{\eta}^{2}.
```

Thus:

```math
\operatorname{Var}(X)
=
\frac{\sigma_{\eta}^{2}}{1-\phi^2}.
```

Substituting the exact OU innovation variance:

```math
\operatorname{Var}(X)
=
\frac{1}{1-\phi^2}
\frac{\sigma^2}{2\kappa}(1-\phi^2)
=
\frac{\sigma^2}{2\kappa}.
```

The discrete AR(1) stationary variance therefore reproduces the continuous-time OU stationary variance exactly.

### 1.8 Mean-reversion half-life

From the conditional expectation:

```math
\mathbb E[X_{t+h}-\mu\mid X_t]
=
e^{-\kappa h}(X_t-\mu),
```

the half-life is the horizon at which the expected deviation from equilibrium has fallen to one half of its current magnitude.

Therefore \(h_{1/2}\) satisfies:

```math
e^{-\kappa h_{1/2}}
=
\frac{1}{2}.
```

Taking natural logarithms:

```math
-\kappa h_{1/2}
=
\log\left(\frac{1}{2}\right)
=
-\log 2.
```

Hence:

```math
\boxed{
h_{1/2}=\frac{\log2}{\kappa}
}.
```

Using the discrete mapping:

```math
\kappa=-\frac{\log\phi}{\Delta},
```

gives:

```math
h_{1/2}
=
-\frac{\Delta\log2}{\log\phi}.
```

For the bar-indexed implementation with \(\Delta=1\):

```math
\boxed{
h_{1/2}^{(\mathrm{bars})}
=
-\frac{\log2}{\log\phi}
}.
```

The half-life is therefore not an independently fitted parameter. It is a nonlinear transformation of the estimated autoregressive persistence.

### 1.9 Deriving the AR(1) OLS estimator

Inside one estimation window, write:

```math
y_i=a+\phi x_i+\eta_i,
\qquad
x_i=X_{i-1},\quad y_i=X_i.
```

Ordinary least squares chooses \(a\) and \(\phi\) to minimize:

```math
S(a,\phi)
=
\sum_{i=1}^{n}
(y_i-a-\phi x_i)^2.
```

Set the derivative with respect to \(a\) equal to zero:

```math
\frac{\partial S}{\partial a}
=
-2\sum_{i=1}^{n}
(y_i-a-\phi x_i)
=0.
```

Therefore:

```math
\sum_i y_i-na-\phi\sum_i x_i=0.
```

Dividing by \(n\):

```math
\bar y-a-\phi\bar x=0,
```

so:

```math
\boxed{
a=\bar y-\phi\bar x
}.
```

Now differentiate with respect to \(\phi\):

```math
\frac{\partial S}{\partial\phi}
=
-2\sum_{i=1}^{n}
x_i(y_i-a-\phi x_i)
=0.
```

Substitute \(a=\bar y-\phi\bar x\):

```math
y_i-a-\phi x_i
=
(y_i-\bar y)-\phi(x_i-\bar x).
```

Hence the normal equation becomes:

```math
\sum_i
(x_i-\bar x)
\left[(y_i-\bar y)-\phi(x_i-\bar x)\right]
=0.
```

Expanding:

```math
\sum_i(x_i-\bar x)(y_i-\bar y)
-
\phi\sum_i(x_i-\bar x)^2
=0.
```

Therefore the OLS persistence estimator is:

```math
\boxed{
\hat\phi
=
\frac{
\sum_i(x_i-\bar x)(y_i-\bar y)
}{
\sum_i(x_i-\bar x)^2
}
}.
```

The intercept estimator follows as:

```math
\boxed{
\hat a=\bar y-\hat\phi\bar x
}.

The fitted innovations are:

```math
\hat\eta_i
=
y_i-\hat a-\hat\phi x_i.
```

Because two regression parameters have been estimated, the unbiased residual-variance estimator uses \(n-2\) degrees of freedom:

```math
\hat\sigma_{\eta}^{2}
=
\frac{1}{n-2}
\sum_{i=1}^{n}\hat\eta_i^2.
```

The fitted OU equilibrium is then recovered through the exact mapping:

```math
\boxed{
\hat\mu
=
\frac{\hat a}{1-\hat\phi}
}.
```

The fitted stationary variance is:

```math
\boxed{
\hat\sigma_X^2
=
\frac{\hat\sigma_{\eta}^{2}}{1-\hat\phi^2}
}.
```

and the standardized displacement used by the strategy is:

```math
Z_t
=
\frac{X_t-\hat\mu_t}{\hat\sigma_{X,t}}.
```

This chain is important because every quantity used by the trading state can now be traced back to either the OU SDE or the least-squares objective rather than appearing as an unexplained formula.

### 1.10 OU derivation-to-implementation chain

The complete mathematical chain is:

```text
OU SDE
  -> exact stochastic solution
  -> conditional Gaussian transition
  -> exact AR(1) representation
  -> rolling OLS estimates of a and phi
  -> continuous-time kappa and equilibrium mu
  -> innovation and stationary variance
  -> half-life
  -> standardized residual state
  -> causal trading decision
```

This makes the discrete estimator an explicit consequence of the continuous-time model rather than a loosely associated statistical approximation.

---

## 2. Cointegration and Error-Correction Derivations

### 2.1 Cointegrating regression from the least-squares objective

For two log-price processes \(Y_t\) and \(X_t\), consider the long-run relation:

```math
Y_t=\alpha+\beta X_t+u_t.
```

Stack \(n\) observations into matrix form:

```math
\mathbf y=Z\boldsymbol\gamma+\mathbf u,
```

where:

```math
Z=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix},
\qquad
\boldsymbol\gamma=
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}.
```

Ordinary least squares minimizes:

```math
Q(\boldsymbol\gamma)
=
(\mathbf y-Z\boldsymbol\gamma)^{\mathsf T}
(\mathbf y-Z\boldsymbol\gamma).
```

Expand the quadratic form:

```math
Q
=
\mathbf y^{\mathsf T}\mathbf y
-2\boldsymbol\gamma^{\mathsf T}Z^{\mathsf T}\mathbf y
+
\boldsymbol\gamma^{\mathsf T}Z^{\mathsf T}Z\boldsymbol\gamma.
```

Differentiating with respect to \(\boldsymbol\gamma\):

```math
\frac{\partial Q}{\partial\boldsymbol\gamma}
=
-2Z^{\mathsf T}\mathbf y
+2Z^{\mathsf T}Z\boldsymbol\gamma.
```

Setting the gradient equal to zero gives the normal equations:

```math
Z^{\mathsf T}Z\widehat{\boldsymbol\gamma}
=
Z^{\mathsf T}\mathbf y.
```

If \(Z\) has full column rank:

```math
\widehat{\boldsymbol\gamma}
=
(Z^{\mathsf T}Z)^{-1}Z^{\mathsf T}\mathbf y.
```

Thus the fitted equilibrium residual is:

```math
\widehat u_t
=
Y_t-\widehat\alpha-\widehat\beta X_t.
```

### 2.2 Why the implementation uses SVD rather than an explicit matrix inverse

The normal-equation expression is mathematically useful, but explicitly forming \(Z^{\mathsf T}Z\) can worsen numerical conditioning.

For the matrix 2-norm condition number:

```math
\kappa_2(Z)
=
\frac{s_{\max}(Z)}{s_{\min}(Z)}.
```

The condition number of the normal-equation matrix satisfies:

```math
\kappa_2(Z^{\mathsf T}Z)
=
\kappa_2(Z)^2.
```

Thus a design matrix that is already poorly conditioned becomes substantially more fragile when the normal equations are formed explicitly.

Instead, write the singular-value decomposition:

```math
Z=U\Sigma V^{\mathsf T}.
```

For full column rank:

```math
\Sigma=
\begin{bmatrix}
s_1 & 0\\
0 & s_2
\end{bmatrix},
\qquad s_1,s_2>0.
```

The Moore-Penrose inverse is:

```math
Z^{+}=V\Sigma^{-1}U^{\mathsf T}.
```

Therefore the least-squares estimator can be written as:

```math
\boxed{
\widehat{\boldsymbol\gamma}
=
V\Sigma^{-1}U^{\mathsf T}\mathbf y
}.
```

The implementation requires:

```math
\operatorname{rank}(Z)=2
```

and records:

```math
\kappa_2(Z)=\frac{s_{\max}}{s_{\min}}
```

as a numerical diagnostic.

A stable numerical solution does not establish cointegration. SVD addresses numerical estimation error; it cannot transform an economically unstable relation into a stationary equilibrium.

### 2.3 Integrated processes and cancellation of a common stochastic trend

Suppose \(X_t\) contains a stochastic trend:

```math
X_t=X_{t-1}+\xi_t,
```

where \(\xi_t\) is stationary.

Then \(X_t\) is typically \(I(1)\).

Now suppose another level process shares this stochastic trend:

```math
Y_t=\alpha+\beta X_t+u_t,
```

where \(u_t\) is stationary.

Although both \(X_t\) and \(Y_t\) inherit the non-stationary stochastic trend, form the linear combination:

```math
Y_t-\alpha-\beta X_t.
```

Substituting the relation for \(Y_t\):

```math
Y_t-\alpha-\beta X_t
=
u_t.
```

Therefore:

```math
X_t\sim I(1),
\qquad
Y_t\sim I(1),
```

while:

```math
Y_t-\alpha-\beta X_t\sim I(0).
```

The coefficient \(\beta\) therefore removes the shared stochastic trend if a genuine cointegrating relation exists.

This explains why correlation is insufficient.

Two independent random walks can display high sample correlation and high regression \(R^2\) because both wander persistently through time.

Cointegration instead requires that the estimated equilibrium error itself be stationary.

### 2.4 Deriving an error-correction model from an ARDL representation

Consider the dynamic regression:

```math
Y_t
=
c+\rho Y_{t-1}+\theta_0X_t+\theta_1X_{t-1}+\varepsilon_t.
```

Subtract \(Y_{t-1}\) from both sides:

```math
\Delta Y_t
=
c+(\rho-1)Y_{t-1}
+\theta_0X_t+\theta_1X_{t-1}
+\varepsilon_t.
```

Using:

```math
X_t=X_{t-1}+\Delta X_t,
```

we obtain:

```math
\Delta Y_t
=
c+(\rho-1)Y_{t-1}
+(\theta_0+\theta_1)X_{t-1}
+\theta_0\Delta X_t
+\varepsilon_t.
```

Define the adjustment coefficient:

```math
\lambda=\rho-1.
```

Now define the long-run coefficients:

```math
\alpha=\frac{c}{1-\rho}
```

and:

```math
\beta=\frac{\theta_0+\theta_1}{1-\rho}.
```

Because \(1-\rho=-\lambda\):

```math
c=-\lambda\alpha
```

and:

```math
\theta_0+\theta_1=-\lambda\beta.
```

Substituting these expressions gives:

```math
\Delta Y_t
=
\lambda Y_{t-1}
-\lambda\alpha
-\lambda\beta X_{t-1}
+\theta_0\Delta X_t
+\varepsilon_t.
```

Factor the long-run equilibrium error:

```math
\boxed{
\Delta Y_t
=
\lambda
\left(
Y_{t-1}-\alpha-\beta X_{t-1}
\right)
+
\theta_0\Delta X_t
+
\varepsilon_t
}.
```

If:

```math
u_{t-1}=Y_{t-1}-\alpha-\beta X_{t-1},
```

then:

```math
\boxed{
\Delta Y_t=\lambda u_{t-1}+\theta_0\Delta X_t+\varepsilon_t
}.
```

The error-correction representation is therefore not an unrelated second model. It is a reparameterization of a dynamic levels model that separates long-run disequilibrium from short-run changes.

### 2.5 Stability interpretation of the adjustment coefficient

Suppose the previous equilibrium error is positive:

```math
u_{t-1}>0.
```

This means \(Y_{t-1}\) lies above the fitted long-run relation:

```math
Y_{t-1}>\alpha+\beta X_{t-1}.
```

If:

```math
\lambda<0,
```

then the error-correction contribution satisfies:

```math
\lambda u_{t-1}<0.
```

It therefore pushes \(\Delta Y_t\) downward, reducing the positive equilibrium error.

Likewise, if:

```math
u_{t-1}<0
```

and \(\lambda<0\), then:

```math
\lambda u_{t-1}>0,
```

which pushes \(Y_t\) upward toward the long-run relation.

Thus a negative adjustment coefficient is consistent with restoring equilibrium under this residual orientation.

Because:

```math
\lambda=\rho-1,
```

a stable dynamic coefficient satisfying:

```math
0<\rho<1
```

implies:

```math
-1<\lambda<0.
```

### 2.6 Error correction, AR(1) persistence, and OU speed

Suppose the equilibrium residual itself follows:

```math
u_t=\phi u_{t-1}+\eta_t.
```

Subtract \(u_{t-1}\):

```math
\Delta u_t
=
(\phi-1)u_{t-1}+\eta_t.
```

Therefore the residual error-correction coefficient is:

```math
\lambda=\phi-1.
```

or equivalently:

```math
\phi=1+\lambda.
```

If this residual also admits an OU interpretation:

```math
\phi=e^{-\kappa\Delta},
```

then:

```math
1+\lambda=e^{-\kappa\Delta}.
```

Taking logarithms gives:

```math
\boxed{
\kappa
=
-\frac{\log(1+\lambda)}{\Delta}
}.
```

The corresponding half-life is:

```math
\boxed{
h_{1/2}
=
-\frac{\Delta\log2}{\log(1+\lambda)}
}.
```

This relation makes explicit that the ECM adjustment coefficient, discrete autoregressive persistence, and continuous-time OU mean-reversion speed are different parameterizations of closely related reversion dynamics when the corresponding assumptions hold.

The project does not treat the fitted ECM coefficient itself as a trading signal; the relation is included to clarify the mathematical connection between equilibrium correction and the OU diagnostics used elsewhere.

## 3. Statistical Inference under Dependence and Multiple Testing

### 3.1 Sampling variance of the mean under serial dependence

Let the session-return process be \(r_1,\ldots,r_T\), with mean \(\mu\).

The sample mean is:

```math
\bar r=\frac{1}{T}\sum_{t=1}^{T}r_t.
```

Its variance is:

```math
\operatorname{Var}(\bar r)
=
\operatorname{Var}\left(
\frac{1}{T}\sum_{t=1}^{T}r_t
\right).
```

Pulling out the constant:

```math
\operatorname{Var}(\bar r)
=
\frac{1}{T^2}
\operatorname{Var}\left(
\sum_{t=1}^{T}r_t
\right).
```

For correlated observations:

```math
\operatorname{Var}\left(\sum_{t=1}^{T}r_t\right)
=
\sum_{t=1}^{T}\operatorname{Var}(r_t)
+
2\sum_{1\leq s<t\leq T}\operatorname{Cov}(r_s,r_t).
```

Assume covariance stationarity and define the lag-\(k\) autocovariance:

```math
\gamma_k
=
\operatorname{Cov}(r_t,r_{t-k}).
```

There are \(T-k\) observation pairs separated by lag \(k\). Therefore:

```math
\operatorname{Var}(\bar r)
=
\frac{1}{T^2}
\left[
T\gamma_0
+
2\sum_{k=1}^{T-1}(T-k)\gamma_k
\right].
```

Factor \(T\):

```math
\boxed{
\operatorname{Var}(\bar r)
=
\frac{1}{T}
\left[
\gamma_0
+
2\sum_{k=1}^{T-1}
\left(1-\frac{k}{T}\right)\gamma_k
\right]
}.
```

Under independence, every \(\gamma_k=0\) for \(k>0\), reducing the expression to:

```math
\operatorname{Var}(\bar r)
=
\frac{\gamma_0}{T}.
```

The conventional standard error \(s/\sqrt{T}\) therefore relies on the absence of serial covariance. Positive autocorrelation generally makes the naive standard error too small and the corresponding t-statistic too large.

### 3.2 Long-run variance and HAC estimation

For sufficiently weak dependence, as \(T\) grows the finite-sample weighting term satisfies:

```math
1-\frac{k}{T}\rightarrow1
```

for fixed \(k\).

This motivates the long-run variance:

```math
\boxed{
\Omega
=
\gamma_0
+
2\sum_{k=1}^{\infty}\gamma_k
}.
```

Asymptotically:

```math
\operatorname{Var}(\bar r)
\approx
\frac{\Omega}{T}.
```

The infinite autocovariance sum is unavailable in finite samples, so the implementation truncates it at lag \(L\) and applies Bartlett weights.

For lag \(k\):

```math
w_k
=
1-\frac{k}{L+1}.
```

Let:

```math
\widehat\gamma_k
=
\frac{1}{T}
\sum_{t=k+1}^{T}
(r_t-\bar r)(r_{t-k}-\bar r).
```

The Newey-West long-run variance estimator used by the project is:

```math
\boxed{
\widehat\Omega_{NW}
=
\widehat\gamma_0
+
2\sum_{k=1}^{L}
\left(1-\frac{k}{L+1}\right)
\widehat\gamma_k
}.
```

The standard error of the sample mean becomes:

```math
SE_{HAC}(\bar r)
=
\sqrt{\frac{\widehat\Omega_{NW}}{T}}.
```

Testing the null:

```math
H_0:\mu=0
```

gives the HAC-adjusted statistic:

```math
\boxed{
t_{HAC}
=
\frac{\bar r}
{\sqrt{\widehat\Omega_{NW}/T}}
}.
```

The implemented inference fixes:

```math
L=5.
```

Thus the estimator incorporates session-return covariance through five lags rather than treating every session as an independent observation.

### 3.3 Dependence and effective information content

Under independence:

```math
\operatorname{Var}(\bar r)
=
\frac{\gamma_0}{T}.
```

Under serial dependence:

```math
\operatorname{Var}(\bar r)
\approx
\frac{\Omega}{T}.
```

Define an effective sample size \(T_{\mathrm{eff}}\) by requiring:

```math
\frac{\gamma_0}{T_{\mathrm{eff}}}
=
\frac{\Omega}{T}.
```

Solving:

```math
\boxed{
T_{\mathrm{eff}}
=
T\frac{\gamma_0}{\Omega}
}.
```

When positive serial dependence makes \(\Omega>\gamma_0\):

```math
T_{\mathrm{eff}}<T.
```

Thus a large number of backtest observations does not necessarily imply an equally large amount of independent statistical information.

This is one reason the research reports both naive and HAC-adjusted inference rather than relying only on the observation count.

### 3.4 Circular moving-block bootstrap

An ordinary iid bootstrap samples individual observations independently.

For serially dependent returns, doing so destroys the local dependence structure that the inference procedure is intended to preserve.

Instead, let the observed return sequence be:

```math
\mathcal R=(r_1,r_2,\ldots,r_T).
```

Choose a block length \(\ell\).

For a block beginning at index \(s\), define the circular block:

```math
B_s
=
(r_s,r_{s+1},\ldots,r_{s+\ell-1}),
```

where indices wrap around modulo \(T\).

Equivalently, the \(j\)-th member of the block uses index:

```math
i_j
=
(s+j)\bmod T.
```

To construct one bootstrap sample of length \(T\), draw:

```math
K=\left\lceil\frac{T}{\ell}\right\rceil
```

independent block starting locations:

```math
s_1,\ldots,s_K
\sim
\operatorname{Uniform}\{0,\ldots,T-1\}.
```

Concatenate the corresponding circular blocks and truncate the result after \(T\) observations.

For bootstrap replication \(b\), denote the resulting sequence by:

```math
\mathcal R^{*(b)}.
```

Then compute:

```math
\bar r^{*(b)}
=
\frac{1}{T}\sum_{t=1}^{T}r_t^{*(b)}
```

and:

```math
SR^{*(b)}
=
\frac{\bar r^{*(b)}}
{s^{*(b)}}\sqrt{252}.
```

The empirical 2.5th and 97.5th percentiles of the bootstrap distribution form the reported 95 percent intervals.

The implemented procedure uses:

```math
\ell=5
```

and:

```math
B=2000
```

bootstrap replications.

Circular indexing prevents end-of-sample observations from having systematically fewer opportunities to appear inside a complete block.

### 3.5 Sharpe ratio sampling uncertainty

For per-period returns with sample mean \(\bar r\) and sample standard deviation \(s\), define the per-period estimated Sharpe ratio:

```math
\widehat{SR}
=
\frac{\bar r}{s}.
```

The annualized representation used for reporting is:

```math
\widehat{SR}_{ann}
=
\sqrt{252}\,\widehat{SR}.
```

The inference calculation itself converts the annualized value back to the per-period Sharpe:

```math
\widehat{SR}
=
\frac{\widehat{SR}_{ann}}{\sqrt{252}}.
```

A Sharpe estimate is a random variable because both its numerator and denominator are estimated from a finite sample.

Its sampling uncertainty also depends on higher moments of the return distribution.

Let sample skewness be:

```math
\widehat\gamma_3
```

and ordinary, non-excess kurtosis be:

```math
\widehat\gamma_4.
```

The finite-sample approximation used by the project gives:

```math
\operatorname{Var}(\widehat{SR})
\approx
\frac{
1
-\widehat\gamma_3\widehat{SR}
+
\frac{\widehat\gamma_4-1}{4}\widehat{SR}^{\,2}
}{T-1}.
```

Therefore the approximate standard error is:

```math
SE(\widehat{SR})
=
\sqrt{
\frac{
1-\widehat\gamma_3\widehat{SR}
+\frac{\widehat\gamma_4-1}{4}\widehat{SR}^{\,2}
}{T-1}
}.
```

For a benchmark Sharpe \(SR^*\), standardization gives:

```math
Z_{SR}
=
\frac{\widehat{SR}-SR^*}
{SE(\widehat{SR})}.
```

Substituting the variance expression:

```math
\boxed{
Z_{SR}
=
\frac{
(\widehat{SR}-SR^*)\sqrt{T-1}
}{
\sqrt{
1-\widehat\gamma_3\widehat{SR}
+\frac{\widehat\gamma_4-1}{4}\widehat{SR}^{\,2}
}
}
}.
```

The Probabilistic Sharpe Ratio is then:

```math
\boxed{
PSR(SR^*)
=
\Phi(Z_{SR})
}.
```

where \(\Phi\) is the standard normal cumulative distribution function.

For the unadjusted probabilistic Sharpe calculation in this project:

```math
SR^*=0.
```

PSR therefore asks whether the estimated Sharpe is statistically distinguishable from the benchmark after accounting for finite sample size, skewness, and kurtosis.

### 3.6 Multiple testing and the Deflated Sharpe benchmark

If several strategy configurations are evaluated, the largest observed Sharpe is biased upward even when none possesses genuine alpha.

The reason is an order-statistics effect.

For trial Sharpe estimates:

```math
SR_1,SR_2,\ldots,SR_N,
```

researchers observe not a randomly chosen trial but often the maximum:

```math
SR_{\max}
=
\max_{1\leq i\leq N}SR_i.
```

Even when all trials are centered near the same underlying performance, the expected maximum increases with \(N\).

Let the cross-trial Sharpe dispersion be:

```math
\sigma_{SR}
=
\operatorname{Std}(SR_1,\ldots,SR_N).
```

The project approximates the expected maximum Sharpe under repeated trials using:

```math
SR_{\mathrm{benchmark}}^{DSR}
=
\sigma_{SR}
\left[
(1-\gamma)
\Phi^{-1}\left(1-\frac{1}{N}\right)
+
\gamma
\Phi^{-1}\left(1-\frac{1}{Ne}\right)
\right],
```

where:

```math
\gamma\approx0.5772156649
```

is the Euler-Mascheroni constant and \(e\) is Euler number.

In the OU sensitivity experiment:

```math
N=3,
```

corresponding to the predeclared fast, base, and slow configurations.

The benchmark therefore increases with the dispersion and number of strategy trials rather than remaining fixed at zero.

The Deflated Sharpe probability uses the same finite-sample Sharpe standardization as PSR, but substitutes the multiple-testing benchmark:

```math
\boxed{
DSR
=
\Phi\left(
\frac{
(\widehat{SR}-SR_{\mathrm{benchmark}}^{DSR})\sqrt{T-1}
}{
\sqrt{
1-\widehat\gamma_3\widehat{SR}
+\frac{\widehat\gamma_4-1}{4}\widehat{SR}^{\,2}
}
}
\right)
}.
```

Thus PSR addresses estimation uncertainty for one Sharpe estimate, while DSR raises the hurdle further to account for the fact that several candidate configurations were examined.

### 3.7 Information coefficient

Return-based inference asks whether the strategy PnL is statistically distinguishable from noise.

A separate question is whether the model score itself contains directional information about future returns.

Let \(S_t\) be the model score known at time \(t\), and let:

```math
R_{t+1}
```

be the next-period return.

The information coefficient is the Pearson correlation:

```math
IC
=
\operatorname{Corr}(S_t,R_{t+1}).
```

Expanding:

```math
\boxed{
IC
=
\frac{
\operatorname{Cov}(S_t,R_{t+1})
}{
\sigma_S\sigma_R
}
}.
```

Its sample estimator is:

```math
\widehat{IC}
=
\frac{
\sum_t(S_t-\bar S)(R_{t+1}-\bar R)
}{
\sqrt{\sum_t(S_t-\bar S)^2}
\sqrt{\sum_t(R_{t+1}-\bar R)^2}
}.
```

The forward shift is essential. Correlating \(S_t\) with \(R_t\) could partially measure information already contained in the observation that produced the signal.

The implementation therefore pairs each signal score with the subsequent return rather than the contemporaneous return.

### 3.8 Inference hierarchy used by the research system

The statistical evidence can now be organized as:

```text
session returns
      |
      +--> naive mean t-statistic
      |
      +--> autocovariances
      |       -> HAC long-run variance
      |       -> HAC-adjusted t-statistic
      |
      +--> circular block bootstrap
      |       -> mean-return confidence interval
      |       -> Sharpe confidence interval
      |
      +--> sample Sharpe
              -> skewness and kurtosis correction
              -> Probabilistic Sharpe Ratio
              -> multiple-trial benchmark
              -> Deflated Sharpe Ratio

signal score
      -> next-period return
      -> information coefficient
```

No one statistic is treated as sufficient evidence by itself.

### 3.9 Equation-to-code mapping

The implemented inference is in `src/systematic_alpha/analysis/reversion_inference.py`.

| Mathematical object | Implementation |
|---|---|
| naive t-statistic | `_t_statistics(...)` |
| \(\widehat\gamma_k\) | lagged demeaned dot products |
| Bartlett weight \(1-k/(L+1)\) | `_t_statistics(...)` |
| HAC long-run variance | `long_run_variance` |
| \(t_{HAC}\) | `hac_t_statistic` |
| circular block bootstrap | `_bootstrap_intervals(...)` |
| block length | `BOOTSTRAP_BLOCK_LENGTH = 5` |
| bootstrap replications | `BOOTSTRAP_REPLICATIONS = 2000` |
| annualized Sharpe | `_annualized_sharpe(...)` |
| skewness | `scipy.stats.skew(..., bias=False)` |
| ordinary kurtosis | `scipy.stats.kurtosis(..., fisher=False, bias=False)` |
| PSR probability | `_sharpe_probability(...)` |
| DSR expected-maximum benchmark | `_deflated_benchmark(...)` |
| declared OU trials | three predeclared configurations |
| information coefficient | score / next-return correlation |

The distinction between these procedures is intentional: HAC addresses serial dependence in the sample mean, the block bootstrap provides dependence-aware empirical uncertainty, PSR addresses the sampling distribution of Sharpe, and DSR additionally raises the hurdle for multiple strategy trials.

---

## 4. Portfolio Construction, Dependence, and Fixed-Holdings Accounting

### 4.1 From sleeve returns to portfolio variance

Let the vector of strategy-sleeve returns at time \(t\) be:

```math
\mathbf r_t=
\begin{bmatrix}
r_{1,t}\\
r_{2,t}\\
\vdots\\
r_{N,t}
\end{bmatrix}
```

and let portfolio weights be:

```math
\mathbf w=
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_N
\end{bmatrix}.
```

The portfolio return is the linear combination:

```math
r_{p,t}=\mathbf w^{\mathsf T}\mathbf r_t.
```

Let:

```math
\boldsymbol\mu=\mathbb E[\mathbf r_t]
```

and:

```math
\Sigma=
\mathbb E\left[
(\mathbf r_t-\boldsymbol\mu)
(\mathbf r_t-\boldsymbol\mu)^{\mathsf T}
\right].
```

Then:

```math
\operatorname{Var}(r_{p,t})
=
\mathbb E\left[
(\mathbf w^{\mathsf T}(\mathbf r_t-\boldsymbol\mu))^2
\right].
```

Because a scalar square can be written as a quadratic form:

```math
(\mathbf w^{\mathsf T}\mathbf x)^2
=
\mathbf w^{\mathsf T}\mathbf x\mathbf x^{\mathsf T}\mathbf w,
```

we obtain:

```math
\operatorname{Var}(r_{p,t})
=
\mathbf w^{\mathsf T}
\mathbb E\left[
(\mathbf r_t-\boldsymbol\mu)
(\mathbf r_t-\boldsymbol\mu)^{\mathsf T}
\right]
\mathbf w.
```

Therefore:

```math
\boxed{
\sigma_p^2=\mathbf w^{\mathsf T}\Sigma\mathbf w
}.
```

In scalar form:

```math
\sigma_p^2
=
\sum_iw_i^2\sigma_i^2
+
2\sum_{i<j}
w_iw_j\rho_{ij}\sigma_i\sigma_j.
```

This decomposition shows why diversification depends on covariance among strategy PnL streams, not merely on the number of strategies in the portfolio.

### 4.2 Global minimum-variance portfolio from Lagrange multipliers

Consider the fully invested minimum-variance problem:

```math
\min_{\mathbf w}
\quad
\mathbf w^{\mathsf T}\Sigma\mathbf w
```

subject to:

```math
\mathbf 1^{\mathsf T}\mathbf w=1.
```

Construct the Lagrangian:

```math
\mathcal L(\mathbf w,\lambda)
=
\mathbf w^{\mathsf T}\Sigma\mathbf w
-
\lambda(\mathbf 1^{\mathsf T}\mathbf w-1).
```

Because \(\Sigma\) is symmetric:

```math
\nabla_{\mathbf w}
(\mathbf w^{\mathsf T}\Sigma\mathbf w)
=
2\Sigma\mathbf w.
```

The first-order condition is therefore:

```math
2\Sigma\mathbf w-\lambda\mathbf1=0.
```

Assuming \(\Sigma\) is nonsingular:

```math
\mathbf w
=
\frac{\lambda}{2}\Sigma^{-1}\mathbf1.
```

Impose the budget constraint:

```math
\mathbf1^{\mathsf T}\mathbf w=1.
```

Substitution gives:

```math
\frac{\lambda}{2}
\mathbf1^{\mathsf T}\Sigma^{-1}\mathbf1
=1.
```

Hence:

```math
\lambda
=
\frac{2}
{\mathbf1^{\mathsf T}\Sigma^{-1}\mathbf1}.
```

Substituting back:

```math
\boxed{
\mathbf w_{GMV}
=
\frac{\Sigma^{-1}\mathbf1}
{\mathbf1^{\mathsf T}\Sigma^{-1}\mathbf1}
}.
```

This is the unconstrained analytical benchmark. The implemented portfolio deliberately adds long-only and concentration constraints rather than using this solution without safeguards.

### 4.3 Constrained minimum variance and KKT conditions

The implemented problem is:

```math
\min_{\mathbf w}
\quad
\mathbf w^{\mathsf T}\widehat\Sigma\mathbf w
```

subject to:

```math
\mathbf1^{\mathsf T}\mathbf w=1,
```

```math
w_i\geq0,
```

and:

```math
w_i\leq w_{\max}.
```

The repository fixes:

```math
w_{\max}=0.35.
```

Introduce multipliers \(\lambda\) for the equality constraint, \(\alpha_i\geq0\) for the lower bounds, and \(\beta_i\geq0\) for the upper bounds.

The Lagrangian can be written:

```math
\mathcal L
=
\mathbf w^{\mathsf T}\widehat\Sigma\mathbf w
+
\lambda(\mathbf1^{\mathsf T}\mathbf w-1)
-
\boldsymbol\alpha^{\mathsf T}\mathbf w
+
\boldsymbol\beta^{\mathsf T}
(\mathbf w-w_{\max}\mathbf1).
```

Stationarity requires:

```math
\boxed{
2\widehat\Sigma\mathbf w
+
\lambda\mathbf1
-
\boldsymbol\alpha
+
\boldsymbol\beta
=0
}.
```

Primal feasibility requires:

```math
\mathbf1^{\mathsf T}\mathbf w=1,
\qquad
0\leq w_i\leq w_{\max}.
```

Dual feasibility requires:

```math
\alpha_i\geq0,
\qquad
\beta_i\geq0.
```

Complementary slackness requires:

```math
\alpha_iw_i=0
```

and:

```math
\beta_i(w_i-w_{\max})=0.
```

An interior sleeve therefore has \(\alpha_i=\beta_i=0\), while a sleeve at zero or at the concentration cap can have a non-zero active-constraint multiplier.

The code solves this convex quadratic allocation numerically with SLSQP and independently checks the resulting weight constraints.

### 4.4 Inverse-volatility allocation and capped water filling

The unconstrained inverse-volatility score for sleeve \(i\) is:

```math
q_i=\frac{1}{\sigma_i}.
```

Normalizing these positive scores gives:

```math
\boxed{
w_i
=
\frac{\sigma_i^{-1}}
{\sum_{j=1}^{N}\sigma_j^{-1}}
}.
```

This construction does not use pairwise covariance and therefore is not equivalent to minimum variance.

It allocates less capital to individually volatile sleeves, but two low-volatility sleeves can still contain substantial common risk.

The implementation also imposes:

```math
w_i\leq0.35.
```

If the normalized candidate weight for a sleeve exceeds the cap, the algorithm fixes that sleeve at \(w_{\max}\), removes it from the active set, and redistributes the remaining mass among uncapped sleeves in proportion to their inverse-volatility scores.

Let \(C\) denote the capped set and \(U\) the remaining uncapped set.

The residual allocation mass is:

```math
M
=
1-\sum_{i\in C}w_{\max}.
```

Weights for the active set become:

```math
w_i
=
M
\frac{q_i}{\sum_{j\in U}q_j},
\qquad i\in U.
```

The procedure repeats until no active candidate exceeds the cap.

### 4.5 Covariance shrinkage and spectral structure

A sample covariance matrix can be unstable when correlations are estimated from limited data or when sleeves are strongly dependent.

The constrained minimum-variance implementation therefore uses a Ledoit-Wolf shrinkage covariance rather than the raw sample matrix.

Write the sample covariance as \(S\) and a structured target as \(F\).

The shrinkage estimator has the form:

```math
\boxed{
\widehat\Sigma_{LW}
=
(1-\delta)S+\delta F,
\qquad
0\leq\delta\leq1
}.
```

For a scaled-identity target:

```math
F=\mu I,
\qquad
\mu=\frac{\operatorname{tr}(S)}{N}.
```

If:

```math
S=Q\Lambda Q^{\mathsf T}
```

with eigenvalues \(\lambda_i\), then because \(I\) shares every orthonormal eigenbasis:

```math
\widehat\Sigma_{LW}
=
Q\left[(1-\delta)\Lambda+\delta\mu I\right]Q^{\mathsf T}.
```

Thus each eigenvalue is transformed as:

```math
\boxed{
\lambda_i^{LW}
=
(1-\delta)\lambda_i+\delta\mu
}.
```

Very small eigenvalues are pulled upward while extreme eigenvalues are pulled toward the common target scale. This reduces sensitivity of minimum-variance weights to noisy near-singular covariance directions.

The shrinkage coefficient is estimated from the training data and is not selected by inspecting test-period portfolio performance.

### 4.6 Eigenvalue concentration and effective rank

For a positive-semidefinite covariance matrix:

```math
\Sigma=Q\Lambda Q^{\mathsf T},
```

where:

```math
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N),
\qquad
\lambda_i\geq0.
```

Total variance represented by the covariance matrix is its trace:

```math
\operatorname{tr}(\Sigma)
=
\sum_{i=1}^{N}\lambda_i.
```

Normalize the eigenvalues:

```math
p_i
=
\frac{\lambda_i}{\sum_j\lambda_j}.
```

Then:

```math
\sum_ip_i=1,
```

so the normalized spectrum can be interpreted as a probability distribution over independent covariance directions.

Its Shannon entropy is:

```math
H
=
-\sum_i p_i\log p_i.
```

Exponentiating converts entropy back into an effective dimension:

```math
\boxed{
r_{\mathrm{eff}}
=
\exp\left(-\sum_ip_i\log p_i\right)
}.
```

If one eigenvalue contains essentially all variance, then \(p_1\approx1\) and the remaining \(p_i\approx0\).

Hence:

```math
H\approx0
\quad\Longrightarrow\quad
r_{\mathrm{eff}}\approx1.
```

If variance is evenly distributed across all \(N\) directions:

```math
p_i=\frac{1}{N},
```

then:

```math
H
=
-N\frac1N\log\frac1N
=
\log N,
```

so:

```math
r_{\mathrm{eff}}=N.
```

Therefore:

```math
1\leq r_{\mathrm{eff}}\leq N.
```

The first principal-component variance share is:

```math
PC1
=
\frac{\lambda_{\max}}{\sum_i\lambda_i}.
```

High PC1 share and low effective rank reveal that apparently different sleeves may still be driven by only a small number of common risk directions.

### 4.7 Diversification ratio

Let:

```math
\boldsymbol\sigma=
(\sigma_1,\ldots,\sigma_N)^{\mathsf T}.
```

The weighted average of individual sleeve volatilities is:

```math
\mathbf w^{\mathsf T}\boldsymbol\sigma.
```

Actual portfolio volatility is:

```math
\sigma_p
=
\sqrt{\mathbf w^{\mathsf T}\Sigma\mathbf w}.
```

The diversification ratio is therefore:

```math
\boxed{
DR
=
\frac{\mathbf w^{\mathsf T}\boldsymbol\sigma}
{\sqrt{\mathbf w^{\mathsf T}\Sigma\mathbf w}}
}.
```

When component risks are perfectly positively aligned, combining them produces little reduction in total volatility and the ratio approaches one.

When imperfect dependence lowers portfolio volatility relative to the weighted standalone volatilities, the ratio exceeds one.

The quantity therefore measures risk diversification rather than profitability.

### 4.8 Constant-mix returns versus genuine fixed holdings

A subtle portfolio-accounting distinction arises after target weights have been set.

Suppose a fold begins at time \(\tau\) with target weights:

```math
w_{i,\tau}
```

and total wealth:

```math
W_{\tau}.
```

Initial capital assigned to sleeve \(i\) is:

```math
H_{i,\tau}
=
w_{i,\tau}W_{\tau}.
```

If there is no rebalance during the fold, sleeve holdings evolve recursively:

```math
\boxed{
H_{i,t}
=
H_{i,t-1}(1+r_{i,t})
}.
```

Total portfolio wealth is:

```math
W_t
=
\sum_iH_{i,t}.
```

Immediately before return \(r_{i,t}\) is realized, the effective sleeve weight is:

```math
\widetilde w_{i,t}
=
\frac{H_{i,t-1}}{W_{t-1}}.
```

Therefore the actual portfolio return is:

```math
\boxed{
r_{p,t}
=
\sum_i\widetilde w_{i,t}r_{i,t}
}.
```

After the returns are realized:

```math
H_{i,t}
=
\widetilde w_{i,t}W_{t-1}(1+r_{i,t}).
```

Since:

```math
W_t
=
W_{t-1}(1+r_{p,t}),
```

the post-return weight is:

```math
\boxed{
w_{i,t}^{post}
=
\frac{
\widetilde w_{i,t}(1+r_{i,t})
}{
1+r_{p,t}
}
}.
```

Unless every sleeve realizes the same return, the post-return weights differ from the original targets.

Fixed holdings therefore generate endogenous weight drift.

### 4.9 The hidden rebalancing assumption in constant-weight arithmetic

If instead one computes every session using:

```math
r_{p,t}^{constant}
=
\sum_iw_{i,\tau}r_{i,t}
```

with the original fold weights \(w_{i,\tau}\) on every date, the calculation implicitly resets the portfolio to those target weights before every return observation.

That is a constant-mix portfolio, not fixed holdings.

The two methods coincide on the first period after a rebalance:

```math
\widetilde{\mathbf w}_{\tau+1}
=
\mathbf w_{\tau},
```

but generally diverge afterward because:

```math
\widetilde{\mathbf w}_{t+1}
\neq
\mathbf w_{\tau}.
```

The fixed-holdings calculation therefore preserves the economic consequence of allowing successful sleeves to become larger and unsuccessful sleeves to become smaller between scheduled rebalances.

The project records both pre-return and post-return weight paths so this accounting assumption is directly auditable.

### 4.10 Rebalance turnover after weight drift

Suppose the previous fold ends with drifted weights:

```math
\mathbf w_{old}^{post}
```

and the next fold specifies target weights:

```math
\mathbf w_{new}^{*}.
```

The required weight change is:

```math
\Delta\mathbf w
=
\mathbf w_{new}^{*}
-
\mathbf w_{old}^{post}.
```

The implementation measures allocation turnover from the absolute changes in portfolio weights and applies proportional allocation cost at one basis point per unit of turnover.

Thus portfolio costs depend on the actual drifted state entering a rebalance rather than on an assumed constant target state.

### 4.11 Equation-to-code mapping

The principal implementation is in `src/systematic_alpha/analysis/portfolio_allocation_validation.py` and `src/systematic_alpha/analysis/causal_portfolio_finalization.py`.

| Mathematical object | Implementation |
|---|---|
| equal weight | `calculate_equal_weights()` |
| inverse-volatility score \(1/\sigma_i\) | `calculate_inverse_volatility_weights(...)` |
| 35 percent concentration cap | `MAXIMUM_WEIGHT = 0.35` |
| capped water filling | inverse-volatility allocation loop |
| Ledoit-Wolf covariance | `LedoitWolf(...).fit(...)` |
| \(w^T\Sigma w\) | minimum-variance objective |
| \(2\Sigma w\) | analytical solver gradient |
| full-investment constraint | SLSQP equality constraint |
| long-only / cap constraints | SLSQP bounds |
| fixed-holdings return recursion | `FixedHoldingsPortfolioPath` machinery |
| pre-return weight | `pre_return_weights` |
| post-return drift | `post_return_weights` |
| fold-ending state | `ending_weights` |
| historical VaR / ES | portfolio metric calculation |

The portfolio layer therefore connects covariance geometry, convex optimization, shrinkage estimation, dependence diagnostics, and recursive wealth accounting rather than treating allocation as a static vector of weights.

---

## 5. Causal Execution, Implementation Shortfall, and Risk Identities

### 5.1 Information sets and causal trading decisions

Let \(\mathcal F_t\) denote all information available to the strategy by the completion of observation \(t\).

A causal signal must satisfy:

```math
S_t \in \mathcal F_t.
```

Equivalently, the signal at time \(t\) must be a measurable function only of information known no later than \(t\):

```math
S_t=g(\mathcal F_t).
```

Suppose the model uses the close of bar \(t\) when constructing \(S_t\).

The same close cannot simultaneously be treated as an execution price available before the signal was formed.

A causal implementation therefore delays the position transition:

```math
P_{t+1}=S_t.
```

In the implemented next-open convention, the signal generated on one observed bar becomes the position at the next observed bar open.

Thus the chronological chain is:

```text
information through t
    -> signal S_t
    -> next observed open
    -> position P_{t+1}
    -> subsequent price movement
    -> realized strategy return
```

This ordering prevents a signal from earning a return whose starting price predates the information required to construct that signal.

### 5.2 Next-open return accounting

Let \(O_t\) be the open of observed bar \(t\).

For an ordinary intraday transition, the implemented proxy return is:

```math
R_{t}^{open}
=
\frac{O_{t+1}}{O_t}-1.
```

If \(P_t\) is the position held from the open of bar \(t\), gross strategy return is:

```math
R_{t}^{strategy}
=
P_tR_t^{open}.
```

The position itself is inherited from the previous signal:

```math
P_t=S_{t-1}.
```

Therefore:

```math
\boxed{
R_t^{strategy}
=
S_{t-1}
\left(
\frac{O_{t+1}}{O_t}-1
\right)
}.
```

At the final bar of a session, the implementation substitutes the same-session close for the unavailable next-session open return and liquidates the position.

If \(C_t\) is the session-close price:

```math
R_t^{close}
=
\frac{C_t}{O_t}-1.
```

The ending position is then forced to zero:

```math
P_t^{end}=0.
```

This imposes the overnight-flat contract directly in the return accounting.

### 5.3 Turnover and transaction-cost accounting

Let the position immediately before a rebalance be \(P_{t^-}\) and the desired position be \(P_t\).

Opening turnover is:

```math
TO_t^{open}
=
|P_t-P_{t^-}|.
```

If the session closes while a position is open, forced liquidation contributes:

```math
TO_t^{close}
=
|P_t|.
```

Total turnover is therefore:

```math
TO_t
=
TO_t^{open}+TO_t^{close}.
```

For proportional cost \(c\) basis points per unit of turnover:

```math
Cost_t
=
TO_t\frac{c}{10^4}.
```

Net strategy return is:

```math
\boxed{
R_t^{net}
=
R_t^{gross}
-
TO_t\frac{c}{10^4}
}.
```

This makes turnover an explicit economic state variable rather than a statistic calculated only after the backtest.

### 5.4 Implementation shortfall decomposition

Let:

```math
P_d=\text{decision price},
```

```math
M=\text{arrival midpoint},
```

```math
T=\text{arrival touch price},
```

and:

```math
P_f=\text{fill price}.
```

Define a trade-direction sign:

```math
s=
\begin{cases}
+1,&\text{buy},\\
-1,&\text{sell}.
\end{cases}
```

Total signed implementation shortfall in basis points is:

```math
IS_{total}
=
10^4s
\frac{P_f-P_d}{P_d}.
```

Insert the arrival midpoint and touch price algebraically:

```math
P_f-P_d
=
(M-P_d)
+
(T-M)
+
(P_f-T).
```

Dividing by \(P_d\), multiplying by trade direction and converting to basis points gives:

```math
10^4s\frac{P_f-P_d}{P_d}
=
10^4s\frac{M-P_d}{P_d}
+
10^4s\frac{T-M}{P_d}
+
10^4s\frac{P_f-T}{P_d}.
```

Define:

```math
IS_{delay}
=
10^4s\frac{M-P_d}{P_d},
```

```math
IS_{spread}
=
10^4s\frac{T-M}{P_d},
```

and:

```math
IS_{residual}
=
10^4s\frac{P_f-T}{P_d}.
```

Therefore the exact decomposition identity is:

```math
\boxed{
IS_{total}
=
IS_{delay}
+
IS_{spread}
+
IS_{residual}
}.
```

The implementation explicitly checks that the numerical decomposition error is approximately zero.

### 5.5 Economic interpretation of the shortfall terms

The delay component:

```math
P_d\rightarrow M
```

measures market movement between the strategy decision and order arrival.

The spread component:

```math
M\rightarrow T
```

measures the immediate cost of crossing from the midpoint to the executable side of the quote.

For a buy:

```math
T=Ask,
```

while for a sell:

```math
T=Bid.
```

The residual component:

```math
T\rightarrow P_f
```

captures the part of the fill not explained by decision delay or quoted half-spread.

This may include price movement after arrival, queue effects, partial-fill dynamics, or other execution effects in a real trading environment.

The decomposition is an accounting identity. Whether any component has economic significance depends on the quality and realism of the underlying execution observations.

### 5.6 Round-trip realized PnL

Let quantity be \(Q>0\), entry fill \(P_e\), and exit fill \(P_x\).

For a long trade:

```math
PnL_{gross}^{long}
=
Q(P_x-P_e).
```

For a short trade:

```math
PnL_{gross}^{short}
=
Q(P_e-P_x).
```

Define direction:

```math
d=
\begin{cases}
+1,&\text{long},\\
-1,&\text{short}.
\end{cases}
```

Then both cases can be written:

```math
\boxed{
PnL_{gross}
=
dQ(P_x-P_e)
}.
```

If entry and exit commissions are \(C_e\) and \(C_x\):

```math
\boxed{
PnL_{net}
=
PnL_{gross}-C_e-C_x
}.
```

Execution shortfall can separately be translated from price displacement into currency by multiplying each signed fill-versus-decision difference by executed quantity.

### 5.7 Historical Value at Risk

Let realized portfolio returns be:

```math
r_1,r_2,\ldots,r_T.
```

For a 95 percent historical VaR, define the lower-tail probability:

```math
\alpha=0.05.
```

Let the empirical lower-tail quantile be:

```math
q_{\alpha}
=
F_T^{-1}(\alpha).
```

Because returns are negative in the loss tail while risk is reported as a non-negative loss magnitude, historical VaR is defined as:

```math
\boxed{
VaR_{0.95}
=
\max(0,-q_{0.05})
}.
```

For example, if:

```math
q_{0.05}=-0.02,
```

then:

```math
VaR_{0.95}=0.02.
```

The interpretation is that the historical 5 percent return threshold corresponds to a 2 percent loss magnitude.

No Gaussian distributional assumption is required for this estimator.

### 5.8 Historical Expected Shortfall

VaR identifies a tail threshold but does not describe the severity of outcomes beyond that threshold.

Define the empirical tail set:

```math
\mathcal T_{\alpha}
=
\{r_t:r_t\leq q_{\alpha}\}.
```

The conditional mean return inside the loss tail is:

```math
\bar r_{tail}
=
\frac{1}{|\mathcal T_{\alpha}|}
\sum_{r_t\in\mathcal T_{\alpha}}r_t.
```

Historical Expected Shortfall is reported as the positive magnitude:

```math
\boxed{
ES_{0.95}
=
\max\left(
0,
-\bar r_{tail}
\right)
}.
```

Equivalently, in population notation:

```math
ES_{1-\alpha}
=
-\mathbb E[
R\mid R\leq VaR\ threshold
].
```

For ordinary loss distributions:

```math
ES_{0.95}\geq VaR_{0.95}
```

because Expected Shortfall averages observations that are at least as adverse as the VaR cutoff.

### 5.9 Wealth recursion and drawdown

For simple returns \(r_t>-1\), define normalized starting wealth:

```math
W_0=1.
```

Wealth evolves recursively:

```math
W_t
=
W_{t-1}(1+r_t).
```

Therefore:

```math
\boxed{
W_t
=
\prod_{j=1}^{t}(1+r_j)
}.
```

Define the running wealth peak:

```math
M_t
=
\max_{0\leq j\leq t}W_j.
```

Drawdown is the percentage displacement below that running peak:

```math
\boxed{
DD_t
=
\frac{W_t}{M_t}-1
}.
```

Since \(W_t\leq M_t\):

```math
DD_t\leq0.
```

Maximum drawdown is:

```math
\boxed{
MDD
=
\min_tDD_t
}.
```

This definition preserves the economic path of compounded wealth rather than attempting to infer drawdown from individual return magnitudes.

### 5.10 Equity and realized-return reconciliation

Let \(E_{t-1}\) be equity entering a session and let realized strategy PnL be \(\Pi_t\).

Ending equity is:

```math
\boxed{
E_t=E_{t-1}+\Pi_t
}.
```

The corresponding simple strategy return is:

```math
\boxed{
r_t=\frac{\Pi_t}{E_{t-1}}
}.
```

Substitution gives:

```math
E_t
=
E_{t-1}\left(
1+\frac{\Pi_t}{E_{t-1}}
\right)
=
E_{t-1}(1+r_t).
```

Thus currency PnL accounting and compounded return accounting satisfy the same wealth recursion.

The running equity peak is:

```math
E_t^{peak}
=
\max_{j\leq t}E_j,
```

with equity drawdown:

```math
DD_t
=
\frac{E_t}{E_t^{peak}}-1.
```

### 5.11 Market beta as a covariance projection

Let strategy return be \(R_s\) and benchmark return be \(R_m\).

Consider the linear projection:

```math
R_s=\alpha+\beta R_m+\varepsilon.
```

OLS minimizes:

```math
\mathbb E[(R_s-\alpha-\beta R_m)^2].
```

After demeaning, the first-order condition for \(\beta\) is:

```math
\mathbb E[
R_m(R_s-\beta R_m)
]=0.
```

Therefore:

```math
\operatorname{Cov}(R_s,R_m)
-
\beta\operatorname{Var}(R_m)
=0.
```

and:

```math
\boxed{
\beta
=
\frac{\operatorname{Cov}(R_s,R_m)}
{\operatorname{Var}(R_m)}
}.
```

The execution-risk validation applies this relation against SPY whenever benchmark variance is positive.

### 5.12 Equation-to-code mapping

The causal return implementation is in `src/systematic_alpha/analysis/causal_bar_execution.py`.

Execution shortfall and operational risk calculations are in `src/systematic_alpha/analysis/execution_performance_validation.py`.

Reusable compounded-performance calculations are in `src/systematic_alpha/analysis/strategy_performance.py`.

| Mathematical object | Implementation |
|---|---|
| \(P_t=S_{t-1}\) | one-bar signal shift |
| next-open return | `pnl_proxy_return` |
| overnight liquidation | `ending_position = 0` at session close |
| turnover | open plus forced-close turnover |
| proportional cost | `turnover * cost_bps / 10000` |
| total implementation shortfall | `total_shortfall_bps` |
| decision-delay component | `delay_bps` |
| quoted spread component | `spread_bps` |
| residual fill component | `residual_bps` |
| shortfall identity check | `decomposition_error_bps` |
| round-trip gross PnL | `_round_trip_rows(...)` |
| historical VaR | `_historical_risk(...)` |
| historical Expected Shortfall | `_historical_risk(...)` |
| wealth recursion | compounded simple returns |
| drawdown | wealth divided by running peak minus one |
| market beta | covariance divided by benchmark variance |

The execution layer therefore has a complete mathematical chain from information availability, to delayed position formation, to realized return, to transaction costs, to execution shortfall, to portfolio-level risk measurement.

---

## 6. Mathematical Architecture Summary

The project now connects five mathematical layers:

```text
continuous-time stochastic modelling
        -> OU exact transition
        -> AR(1) estimation
        -> half-life and standardized state

integrated price processes
        -> cointegrating regression
        -> stationary equilibrium residual
        -> error correction
        -> OU-compatible residual dynamics

dependent return inference
        -> HAC long-run variance
        -> block bootstrap
        -> PSR
        -> Deflated Sharpe Ratio

portfolio geometry
        -> covariance quadratic form
        -> constrained minimum variance
        -> shrinkage
        -> eigenstructure and effective rank
        -> fixed-holdings wealth recursion

causal execution
        -> filtration-respecting signal timing
        -> next-open position transition
        -> turnover and cost
        -> implementation shortfall
        -> VaR, Expected Shortfall, beta and drawdown
```

The mathematical objective is not to attach equations to a backtest after the fact. The equations define the statistical assumptions, causal chronology, portfolio state transitions, and validation rules that the implementation is required to satisfy.
