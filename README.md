# Quantitative Alpha Research

## Trend • Cointegration • Ornstein–Uhlenbeck Dynamics • Event-Driven Backtesting • Portfolio Research

Systematic Alpha Research is a quantitative research platform for designing,
testing, and validating systematic trading hypotheses across US equity index
ETFs, with architecture designed to extend into cross-asset research.

The repository emphasizes **mathematical modelling, causal research design,
robust validation, reproducibility, and execution realism** rather than
headline backtest performance.

### Core research areas

- **Trend modelling** — price-ratio and EMA/MACD signal families
- **Statistical arbitrage** — cointegration screening and residual stationarity
- **Mean reversion** — Ornstein–Uhlenbeck dynamics on transformed residuals
- **Walk-forward research** — expanding-history chronological validation
- **Event-driven backtesting** — explicit signal, order, fill, and portfolio states
- **Portfolio structure** — covariance, eigenstructure, effective rank, and diversification
- **Statistical inference** — HAC estimates, moving-block bootstrap, IC, PSR/DSR, and multiple-testing controls
- **Execution research** — transaction costs, shortfall, reconciliation, order-state handling, and Alpaca paper integration

### Research universe

| Area | Current universe |
|---|---|
| Trend | SPY, QQQ, IWM |
| Frequencies | 15m primary; 30m and 60m robustness |
| Pair research | SPY/QQQ, SPY/IWM, QQQ/IWM |
| Mean reversion | SPY, QQQ, IWM intraday OU/VWAP framework |
| Execution | Alpaca paper environment |
| Expansion path | Digital assets / BTC market research |

### Research principles

1. **No look-ahead:** signals, state, and execution timing remain causal.
2. **No result deletion:** negative hypotheses remain part of the research record.
3. **No tuning on locked data:** development and final-test intervals are separated.
4. **Costs are explicit:** turnover and transaction costs enter the research process.
5. **Statistical evidence matters:** profitability alone is not treated as validation.
6. **Research and execution are separated:** backtests cannot silently authorize broker actions.

### Technology

`Python` · `pandas` · `NumPy` · `SciPy` · `statsmodels` · `scikit-learn` · `cvxpy` · `Alpaca API` · `pytest` · `Parquet`

---

## Mathematical Research Highlights

The repository contains full first-principles derivations, but the landing page intentionally shows only the mathematical structures most relevant to the research and execution architecture.

**[Full Mathematical Derivations](docs/MATHEMATICAL_DERIVATIONS.md)** · **[Methods and Equation-to-Code Mapping](docs/MATHEMATICAL_METHODS.md)**

### Statistical Arbitrage and Mean Reversion

A candidate equilibrium relation is expressed through a stationary cointegrating residual:

```math
u_t = Y_t-\alpha-\beta X_t,\qquad u_t\sim I(0).
```

When the residual dynamics admit an Ornstein-Uhlenbeck representation:

```math
dX_t=\kappa(\mu-X_t)dt+\sigma dW_t.
```

Using the integrating factor $e^{\kappa t}$ gives the exact transition:

```math
X_{t+\Delta}
=
\mu+(X_t-\mu)e^{-\kappa\Delta}
+
\sigma\int_t^{t+\Delta}e^{-\kappa(t+\Delta-s)}dW_s.
```

Sampling every $\Delta$ units produces the exact AR(1) representation:

```math
X_{n+1}=a+\phi X_n+\eta_{n+1},
\qquad \phi=e^{-\kappa\Delta},
\qquad a=\mu(1-\phi).
```

Therefore:

```math
\kappa=-\frac{\log\phi}{\Delta},
\qquad
t_{1/2}=\frac{\log 2}{\kappa}.
```

This links the continuous-time mean-reversion model directly to the discrete estimator used by the research system.

### Dependence-Aware Statistical Inference

For serially dependent returns, the variance of the sample mean contains autocovariance terms that an iid t-test ignores:

```math
\mathrm{Var}(\bar r)
=
\frac{1}{T}\left[\gamma_0+2\sum_{k=1}^{T-1}\left(1-\frac{k}{T}\right)\gamma_k\right].
```

This motivates the long-run variance, estimated with Bartlett-weighted Newey-West autocovariances:

```math
\widehat{\Omega}_{NW}
=
\widehat{\gamma}_0
+
2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\widehat{\gamma}_k.
```

The corresponding HAC t-statistic is:

```math
t_{HAC}
=
\frac{\bar r}{\sqrt{\widehat{\Omega}_{NW}/T}}.
```

The research complements this with circular block-bootstrap intervals, information coefficients, Probabilistic Sharpe Ratio, and Deflated Sharpe Ratio.

### Portfolio Optimization and Risk Geometry

Portfolio variance is the covariance quadratic form:

```math
\sigma_p^2=\mathbf w^{\mathsf T}\Sigma\mathbf w.
```

The fully invested minimum-variance benchmark solves:

```math
\min_{\mathbf w}\;\mathbf w^{\mathsf T}\Sigma\mathbf w
\qquad
\text{s.t.}\quad \mathbf{1}^{\mathsf T}\mathbf w=1.
```

Using the Lagrangian:

```math
\mathcal L=\mathbf w^{\mathsf T}\Sigma\mathbf w-\lambda(\mathbf{1}^{\mathsf T}\mathbf w-1),
```

the first-order condition is:

```math
2\Sigma\mathbf w-\lambda\mathbf{1}=0,
```

which gives:

```math
\mathbf w_{GMV}
=
\frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^{\mathsf T}\Sigma^{-1}\mathbf{1}}.
```

The implemented allocator adds Ledoit-Wolf covariance shrinkage, long-only constraints, a 35 percent sleeve cap, effective-rank diagnostics, and fixed-holdings weight drift.

### Causal Execution and Transaction Costs

Signals are separated from execution so information observed at \(t\) cannot earn a return beginning before it existed:

```math
P_t=S_{t-1}.
```

Turnover enters the return process explicitly:

```math
R_t^{net}
=
R_t^{gross}
-
TO_t\frac{c}{10^4}.
```

The research therefore treats execution timing and trading costs as part of the model contract rather than as post-backtest adjustments.

### Implementation Shortfall

Execution quality is decomposed from decision price \(P_d\), through arrival midpoint \(M\) and executable touch \(T\), to fill price \(P_f\):

```math
P_f-P_d=(M-P_d)+(T-M)+(P_f-T).
```

After scaling by trade direction \(s\) and decision price:

```math
IS_{total}
=
10^4s\frac{P_f-P_d}{P_d}
=
IS_{delay}+IS_{spread}+IS_{residual}.
```

The terms measure decision-to-arrival movement, quoted-spread cost, and residual fill cost. The implementation verifies that their numerical sum reconciles to total shortfall.

The detailed derivation document develops these results from their assumptions and links the resulting equations back to the implemented research code.

---

## Market-Data Engineering

The current implementation includes a reusable historical market-data pipeline with:

- Alpaca OHLCV acquisition;
- historical Level-1 quote acquisition;
- historical trade acquisition;
- UTC timestamp normalization;
- normalized bar, quote, and trade schemas;
- OHLC consistency checks;
- duplicate detection;
- missing-bar detection;
- immutable Parquet storage;
- JSON provenance manifests;
- SHA-256 dataset hashes;
- one-session 1-to-15/30/60-minute resampling validation and 15-to-30/60-minute
  robustness resampling;
- volume-weighted VWAP aggregation;
- secure credential loading;
- mocked unit tests for external API behaviour.

The canonical 2020-2025 research panel contains 117,192 provider-native
15-minute Alpaca SIP rows: 39,064 observations and 1,508 sessions for each of
SPY, QQQ, and IWM. It was not reconstructed from six years of one-minute bars.

### Acquisition and Resampling Validation

A complete SPY regular session for 15 December 2025 was acquired from the Alpaca IEX feed.

| Validation item | Result |
|---|---:|
| Raw one-minute bars | 390 |
| Missing internal bars | 0 |
| Duplicate symbol/timestamp rows | 0 |
| Missing values | 0 |
| Fifteen-minute bars | 26 |
| Thirty-minute bars | 13 |
| Sixty-minute bars | 7 |

The 390 one-minute observations correspond to the complete 6.5-hour regular US equity session.

Raw and processed market data are excluded from Git. Each local dataset is associated with provenance metadata containing the provider, feed, request interval, schema, timestamp range, row count, SHA-256 hash, and transformation details.

---

## Market Microstructure Layer

Historical Level-1 quotes and trades support execution diagnostics and
market-quality features.

### Quoted Spread

```math
Spread_t
=
Ask_t
-
Bid_t
```

### Relative Spread

```math
RelativeSpread_t
=
\frac{Ask_t-Bid_t}
{\frac{Ask_t+Bid_t}{2}}
```

### Quote Imbalance

```math
QI_t
=
\frac{BidSize_t-AskSize_t}
{BidSize_t+AskSize_t}
```

### Microprice

```math
Microprice_t
=
\frac{
Ask_t \cdot BidSize_t
+
Bid_t \cdot AskSize_t
}{
BidSize_t+AskSize_t
}
```

The microstructure layer is retained primarily for:

- a spread filter;
- an execution-quality filter;
- an entry-confirmation layer;
- a slippage diagnostic;
- a paper-trading monitoring tool.

It will not be treated as independent alpha unless controlled ablation demonstrates robust incremental value.

---

## Research Validation

The project separates development, walk-forward analysis, and the locked final-test interval.

| Period | Purpose |
|---|---|
| 2020-01-02 to 2025-12-31 | Development and walk-forward research |
| 2026-01-02 to 2026-06-30 | Locked final test |

The final-test period must not be used for indicator selection, pair selection, threshold optimization, cost calibration, or model redesign.

Implemented or explicitly reported statistical evaluation includes:

- annualized return and volatility;
- Sharpe and Sortino ratios;
- maximum drawdown and Calmar ratio;
- turnover and cost attribution;
- hit rate and payoff ratio;
- long/short attribution;
- rolling beta and factor exposure;
- Value at Risk and Expected Shortfall;
- information coefficient and information ratio;
- HAC/Newey-West-adjusted inference;
- AIC and BIC where appropriate;
- Deflated Sharpe Ratio;
- block-bootstrap confidence intervals;
- parameter-sensitivity surfaces;
- multiple-testing awareness;
- walk-forward and final out-of-sample performance.

A high in-sample Sharpe ratio will not be treated as sufficient evidence of a valid strategy.

---

## Transaction-Cost and Execution Model

Transaction costs are incorporated directly into strategy accounting rather than applied only to headline results.

The execution layer models:

- turnover-dependent transaction costs;
- quoted spread and executable touch;
- execution delay and implementation shortfall;
- slippage and cost-stress scenarios;
- partial fills, rejected orders, cancellations, and stale orders;
- pair-trade legging risk;
- paper-fill versus market-price reconciliation.

Detailed causal timing, turnover equations, and the decision-to-fill shortfall decomposition are developed in **[Mathematical Derivations](docs/MATHEMATICAL_DERIVATIONS.md)**.

---

## Portfolio and Risk Layer

Six systematic strategy sleeves are evaluated as a portfolio rather than as isolated backtests.

The implemented allocation framework includes:

- equal-weight allocation;
- capped inverse-volatility allocation;
- Ledoit-Wolf covariance estimation;
- constrained minimum-variance optimization;
- long-only and concentration constraints;
- covariance eigenstructure and effective-rank diagnostics;
- diversification-ratio and concentration analysis;
- fixed-holdings weight drift between scheduled rebalances;
- historical VaR, Expected Shortfall, drawdown, and turnover attribution.

Allocation rules are predeclared and retained regardless of realized performance. Portfolio construction is treated as a risk-allocation problem rather than a mechanism for selecting whichever historical weighting rule performed best.

The optimization, covariance, effective-rank, and fixed-holdings derivations are developed in **[Mathematical Derivations](docs/MATHEMATICAL_DERIVATIONS.md)**.

---

## Execution Architecture

```text
Historical and live market data
        ↓
Validation and normalization
        ↓
Immutable local storage
        ↓
Time bars and event bars
        ↓
Features and indicators
        ↓
Strategy signals
        ↓
Target positions
        ↓
Portfolio risk and allocation
        ↓
Execution adapter
        ↓
Order-state management
        ↓
Broker reconciliation
        ↓
Monitoring and reporting
```

Implemented execution controls include:

- Alpaca REST and WebSocket connectivity;
- paper-only safeguards;
- stale-feed detection;
- reconnect and resubscribe logic;
- idempotent order handling;
- duplicate update protection;
- partial-fill handling;
- rejected and cancelled orders;
- account, position, and order reconciliation;
- exposure limits;
- kill switches;
- scheduled operation;
- Docker-based reproducibility.

---

## Implementation Status

### Research capabilities

- Immutable development datasets with provenance manifests and reproducible local storage
- Two causal trend families with sensitivity analysis, cross-market robustness, and chronological walk-forward validation
- Vectorized and event-driven accounting consistency checks
- Cointegration feasibility testing with multiple-testing control and no forced strategy promotion
- Ornstein-Uhlenbeck / VWAP transformed-residual mean-reversion research
- Transaction-cost, turnover, HAC, bootstrap, IC, PSR, and DSR diagnostics
- Portfolio dependence analysis using covariance structure, eigenvalues, effective rank, and diversification measures
- Causal execution-timing validation and representative event-time market-data studies

### Execution and risk infrastructure

- Alpaca paper-broker integration with strict paper-endpoint enforcement
- Read-only broker preflight and environment-backed credential handling
- Explicit order-state transitions, partial fills, timeouts, reconciliation, and monitoring
- Exposure limits, circuit breakers, fail-closed controls, and a latched kill switch
- Execution-shortfall, round-trip accounting, drawdown, VaR/ES, and beta diagnostics
- Synthetic known-answer fixtures for execution and reconciliation testing

### Research engineering

- Deterministic evidence bundles and SHA-256 verification
- Exact dependency locking
- Unit, integration, chronology, safety, and regression testing
- Continuous integration
- Container and Compose definitions
- Offline runtime-health validation
- Reproducible research runners and operator workflows

### Ongoing extensions

- Broader empirical paper-execution validation
- Expanded event-time and market-microstructure research
- Cross-asset extension beyond US equity index ETFs
- Digital-asset / BTC research integration
- Further portfolio-construction and risk-model research

---

## Repository Structure

```text
systematic-alpha-research/
├── src/systematic_alpha/
│   ├── analysis/        # research diagnostics, validation, inference, walk-forward studies
│   ├── strategies/      # trend and OU/VWAP strategy implementations
│   ├── data/            # acquisition, schemas, validation, resampling, local storage
│   ├── broker/          # paper-execution boundaries, order state, reconciliation, monitoring
│   └── operations/      # runtime validation and reproducible operational workflows
├── scripts/             # reproducible research and execution runners
├── tests/               # unit, integration, chronology, safety, and regression tests
├── config/              # research and operational configuration
├── docs/                # specifications, methodology, audits, and decision records
├── artifacts/           # generated research evidence and experiment outputs
└── pyproject.toml       # package metadata and dependencies
```

The package is intentionally separated into research, data, strategy,
execution, and operational layers so that model development does not silently
cross into broker mutation or live execution.

## Installation

Python 3.11 is used for development.

```bash
git clone https://github.com/farmanshamsi/systematic-alpha-research.git
cd systematic-alpha-research

python -m venv .venv
source .venv/bin/activate

python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

Broker credentials are supplied through the local environment only and are not
stored in source, reports, artifacts, or configuration.

Run the full test suite:

```bash
python -m pytest
```

Research runners under `scripts/` provide reproducible workflows for data
validation, model diagnostics, walk-forward analysis, event-driven replay,
portfolio research, reporting, and operational checks.

---

## Reproducibility Verification

The research framework uses:

- immutable source manifests and provenance checks;
- deterministic research and report artifacts;
- SHA-256 integrity verification;
- chronological and regression tests;
- vectorized/event-driven parity checks;
- exact dependency locking;
- `git diff --check` before publication;
- offline validation paths that require no broker credentials;
- explicit separation between research code and broker mutation.

Generated evidence and historical experiment records remain available within
the repository for auditability, while the README focuses on methodology and
system architecture rather than engine performance.

---

## Safety

This repository is configured for **paper trading only**.

Current safeguards include:

- live trading disabled;
- paper mode required;
- manual order confirmation required;
- kill-switch support;
- credentials stored outside source control;
- raw market data excluded from Git;
- locked final-test period;
- immutable raw-data storage.

No component should be considered ready for live capital without further validation, operational testing, independent review, and explicit removal of paper-only restrictions.

---

## Research Philosophy

The project follows five principles:

1. **Statistical evidence before trading logic**
2. **Out-of-sample performance before optimization claims**
3. **Transaction costs before headline returns**
4. **Reproducibility before complexity**
5. **Execution realism before live deployment**

The intended result is not simply a profitable backtest. It is a transparent quantitative research and execution system whose assumptions, data, mathematics, software behaviour, and limitations can be inspected and challenged.
