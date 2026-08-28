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

## Mathematical Research Framework


### 1. Price-Ratio Trend Following

The first trend model compares short- and long-horizon estimates of price.

Short-horizon average:

```math
\bar{P}^{(s)}_t
=
\frac{1}{n_s}
\sum_{i=0}^{n_s-1} P_{t-i}
```

Long-horizon average:

```math
\bar{P}^{(l)}_t
=
\frac{1}{n_l}
\sum_{i=0}^{n_l-1} P_{t-i},
\qquad
n_s < n_l
```

Relative trend state:

```math
R_t
=
\frac{\bar{P}^{(s)}_t}
{\bar{P}^{(l)}_t}
-1
```

Volatility-normalized trend state:

```math
Z^{\mathrm{trend}}_t
=
\frac{R_t}{\hat{\sigma}_t}
```

Implemented coverage is:

- fixed versus volatility-scaled thresholds;
- a long-short-neutral historical baseline and a separately identified
  long-flat comparator;
- 15-, 30-, and 60-minute time bars;
- a representative five-session time-bar versus dollar-bar indicator study;
- turnover and slippage sensitivity;
- parameter-surface stability;
- out-of-sample persistence across SPY, QQQ, and IWM; and
- next-bar-open/overnight-flat execution with exact sequential replay parity.

---

### 2. EMA and MACD Trend Following

The second trend strategy is deliberately distinct from the price-ratio model.

An exponential moving average evolves recursively as:

```math
EMA_t
=
\alpha P_t
+
(1-\alpha)EMA_{t-1}
```

The smoothing coefficient is:

```math
\alpha
=
\frac{2}{n+1}
```

The MACD state is:

```math
MACD_t
=
EMA^{(f)}_t
-
EMA^{(s)}_t
```

where \(f\) and \(s\) denote the fast and slow horizons.

The signal line is:

```math
Signal_t
=
EMA^{(m)}(MACD_t)
```

The MACD histogram is:

```math
H_t
=
MACD_t
-
Signal_t
```

First difference of the histogram:

```math
\Delta H_t
=
H_t
-
H_{t-1}
```

Second difference of the histogram:

```math
\Delta^2 H_t
=
\Delta H_t
-
\Delta H_{t-1}
```

Candidate confirmation filters include:

- realized-volatility regimes;
- volume participation;
- higher-timeframe agreement;
- ADX-based directional strength;
- Level-1 spread and quote conditions.

Each additional filter will be evaluated through ablation rather than assumed to add value.

---

### 3a. Cointegration Feasibility Framework

The initial mean-reversion route went beyond a simple Bollinger Band or rolling
Z-score by requiring economically plausible series, integration diagnostics,
Holm-controlled residual stationarity, fold stability, and OU feasibility
before any trading backtest.

For two candidate price series \(X_t\) and \(Y_t\), the long-run relationship is estimated as:

```math
Y_t
=
\alpha
+
\beta X_t
+
\varepsilon_t
```

where:

- \(\alpha\) is the intercept;
- \(\beta\) is the hedge ratio;
- \(\varepsilon_t\) is the equilibrium residual.

The central hypothesis is that the two price series may each be non-stationary:

```math
X_t \sim I(1)
```

```math
Y_t \sim I(1)
```

while a linear combination is stationary:

```math
\varepsilon_t
=
Y_t
-
\alpha
-
\beta X_t
\sim I(0)
```

The residual was evaluated using an Engle-Granger framework with appropriate
residual-based inference.

Initial statistical eligibility requires:

- economically defensible linkage;
- both individual log-price series behaving plausibly as \(I(1)\);
- a stationary equilibrium residual under predeclared inference;
- a stable and interpretable static hedge ratio;
- acceptable structural stability;
- a valid OU representation when supported by the residual dynamics.

Spread crossings, transaction costs, trading thresholds and out-of-sample
performance are evaluated only in later strategy-development stages.

#### Error-Correction Model

Short-run changes can be connected to the previous equilibrium deviation through:

```math
\Delta Y_t
=
c
+
\lambda \varepsilon_{t-1}
+
\sum_i \phi_i \Delta Y_{t-i}
+
\sum_j \psi_j \Delta X_{t-j}
+
u_t
```

The coefficient \(\lambda\) measures the speed and direction of adjustment toward the long-run equilibrium.

#### Ornstein-Uhlenbeck Representation

When supported by the residual dynamics, the spread will also be modelled as an Ornstein-Uhlenbeck process:

```math
d\varepsilon_t
=
\kappa(\mu-\varepsilon_t)\,dt
+
\sigma\,dW_t
```

where:

- \(\mu\) is the long-run spread mean;
- \(\kappa\) is the mean-reversion speed;
- \(\sigma\) is the diffusion volatility;
- \(W_t\) is Brownian motion.

The theoretical half-life is:

```math
t_{1/2}
=
\frac{\ln 2}{\kappa}
```

The equilibrium standard deviation is:

```math
\sigma_{\mathrm{eq}}
=
\frac{\sigma}
{\sqrt{2\kappa}}
```

A normalized spread state can then be written as:

```math
Z_t
=
\frac{\varepsilon_t-\mu}
{\sigma_{\mathrm{eq}}}
```

The frozen feasibility study used:

- one predeclared regression orientation for each candidate pair;
- a static hedge ratio;
- fixed deterministic terms;
- Engle-Granger residual-based inference;
- predeclared stability and OU diagnostics;
- no profitability, ranking or trading gate.

All three SPY/QQQ/IWM pairs failed the Holm-controlled cointegration gate, so no
pair was ranked, selected, or converted into a profitability backtest. Reverse
orientations, rolling hedge ratios, and alternative deterministic terms were
not searched after seeing this rejection.

### 3b. Implemented OU/VWAP Transformed-Residual Reversion

The implemented reversion family subtracts a rolling volume-weighted reference
from log price, then estimates a causal AR(1)-style OU half-life on the
transformed residual. A variance-ratio gate rejects trend-like residuals. Entry,
exit, maximum holding, one-bar delay, overnight-flat behavior, and turnover
costs are explicit state transitions rather than an unconstrained price
z-score.

Three fast/base/slow calibrations were predeclared as sensitivity cases. At one
basis point per turnover, the fast and base equal-weight variants returned
-10.51% and -4.36%; the slow variant returned +6.03%. The slow variant stayed
positive at five basis points (+3.02%), but HAC and block-bootstrap inference
remained inconclusive. It is evidence worth preserving, not a promoted winner.

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

A simple initial slippage model is:

```math
c_t
=
\frac{\text{slippage bps}}{10{,}000}
\left|
\Delta w_t
\right|
```

where \(\Delta w_t\) represents the portfolio weight traded.

Implemented execution analysis includes:

- quoted half-spread;
- slippage scenarios;
- turnover sensitivity;
- borrow costs;
- legging risk in pair trades;
- partial fills;
- rejected and cancelled orders;
- stale orders;
- execution delay;
- paper-fill versus market-price comparison.

---

## Portfolio and Risk Layer

The six frozen trend sleeves were compared using:

- equal allocation;
- inverse-volatility allocation;
- minimum-variance allocation;
- constrained cost-aware optimization.

Full-investment constraint:

```math
\sum_i w_i
=
1
```

Single-position concentration constraint:

```math
|w_i|
\leq
w_{\max}
```

Gross-exposure constraint:

```math
\sum_i |w_i|
\leq
G_{\max}
```

All three long-only allocation rules lost money over the common 1,003-session
walk-forward panel. Allocation is retained as valid research evidence, but it is not
being expanded to disguise weak standalone sleeves.

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
