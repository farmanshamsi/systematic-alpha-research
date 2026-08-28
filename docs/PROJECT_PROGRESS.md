# Axiom Algorithmic Trading Project - Cumulative Progress Report

- Last updated: 2026-08-02
- Completed research day: Day 18
- Current branch: `day18-alpaca-paper-boundary`
- Day 18 freeze commit: `a01feddea1e135e0ddd2ad8a4682a7ae637dd850`
- Remote status: Day 18 branch pushed to GitHub
- Main-branch status: `main` remains at the Day 15 merge, `56903d7`
- Canonical development data: 2020-01-02 through 2025-12-31
- Locked final-test data: 2026-01-02 through 2026-06-30, not accessed
- CQF submission deadline: Tuesday 18 August 2026, 23:59 BST

## 1. Purpose and status convention

This is the living progress record for the Axiom Alpha Engine. It replaces the
older static planning snapshot as the current source of truth for what has been
implemented, what the evidence says, what remains incomplete, and what should
be done next. It must be updated after every completed project day.

Status terms in this file are deliberately distinct:

- **Complete**: implementation, tests, and required artifacts exist locally.
- **Frozen**: the day's specification and outputs are closed to silent changes.
- **Pushed**: the commit exists on a GitHub branch.
- **Merged**: the work exists on `main`.
- **Promoted**: a strategy has passed a separately frozen economic and
  operational promotion protocol. No strategy has been promoted yet.

## 2. Technical summary

Days 1-18 are complete. The project now has a strong research-engineering
foundation: immutable market data, causal signal timing, two trend strategies,
parameter sensitivity, cross-symbol and cross-frequency robustness,
chronological walk-forward testing, an event-driven replay engine, pair-trading
feasibility analysis, diversification and allocation analysis, and a non-trivial
reversion strategy with formal statistical inference. The live operational
boundary now begins with a tested read-only Alpaca paper adapter, exact paper
endpoint enforcement, credential redaction, and account/clock/asset preflight.

The current evidence does **not** establish a robustly profitable deployable
strategy. Both trend baselines were negative after costs in aggregate
out-of-sample testing. Pair reversion was correctly rejected because none of
the three candidate pairs passed the Holm-adjusted cointegration gate. The Day
17 slow OU/VWAP residual configuration was positive after costs, including at
the five-basis-point stress, but its confidence interval crossed zero and its
Deflated Sharpe evidence was not conclusive. It is a research candidate, not a
paper-trading selection.

The project can still reach a high professor-facing standard, but it is not a
9/10 submission yet. The main remaining weaknesses are now clear: order-state
and reconciliation controls, controlled paper-trading evidence, execution
benchmarking, reproducible deployment, and the final mathematical report. Those
items are more important than adding further allocation models or searching a
larger parameter grid.

## 3. What changed after rereading the Module 04 material

The January 2026 project brief is the governing brief. The earlier June 2025
brief and the July planning snapshots remain historical references only.

| Reread requirement or lesson | Effect on the project |
|---|---|
| The official topic requires two trend strategies and one non-trivial reversion strategy. | The two trend families were retained. After Day 14 rejected all candidate pairs, Day 17 implemented a causal OU/VWAP residual strategy rather than forcing a statistically invalid pair trade. |
| Reversion must go beyond a simple price z-score; transformed features and regime diagnostics should be defensible. | Day 17 uses the log close-to-volume-weighted-reference residual, rolling OU diagnostics, a variance-ratio gate, one-bar execution delay, and overnight-flat positions. |
| Strategy testing should include formal inference, Deflated Sharpe, information coefficient, t-statistics, and cost/slippage analysis. | Day 17 added naive and Newey-West t-statistics, moving-block bootstrap intervals, IC, PSR, DSR, and 0/1/2/5-basis-point cost stresses. |
| Live testing through a broker API is required, and code must handle exceptions and incorrect or inconsistent broker responses. | Day 18 completed the paper-only, read-only broker boundary and preflight. Days 19-21 now address order state, reconciliation, and controlled paper execution. |
| Operational discipline includes partial fills, stale streams, position verification, reconciliation, scheduling, and containerization. | Days 18-23 are refocused around a broker adapter, order-state machine, reconciliation, monitoring, safety controls, paper evidence, and deployment reproducibility. |
| The report should include execution benchmarking, dynamic VaR/drawdown, Beta-to-SPY, and saved fills/P&L. | Day 16 supplies initial VaR/ES and drawdown evidence, but dynamic risk reporting, beta, realized slippage, and paper fills/P&L remain incomplete. |
| Systematic portfolio reallocation is of less utility to this algorithmic-trading topic. | Day 16 remains useful CQF evidence, but no more time should be spent expanding allocation rules unless the core broker/report obligations are already complete. |
| The submission must be an analytical report, not a notebook dump, with mathematical models, numerical methods, stress tests, limitations, and working code. | Days 24-25 are reserved for the report, reproducibility audit, final freeze, submission package, and defense preparation. |
| The submission requires one correctly named report file, one correctly named code zip, working code, and a hand-signed declaration. | Packaging and declaration checks are explicit Day 25 acceptance criteria. |

### Relevant Module 04 reference set

- `Books/CQF Final Project Brief - Jan 26 v.4.pdf` - governing topic,
  operational, reporting, and submission requirements.
- `Portfolio Management/JA262.1 Notes.pdf` - portfolio-risk and allocation
  context used to interpret Days 15-16.
- `Statistical Essentials for VaR & ES/.../TUTORIAL Statistical VaR ES ppt.pdf`
  - VaR/ES definitions and backtesting context.
- `FP TUTORIAL III/TUTORIAL_TS RELEASE/FP_PairsCoint_RELEASE AN.pdf` and the
  related VECM material - cointegration and error-correction context for Day 14.
- `AL_Final_Project_V1_1_UPDATED_MEMORY_SNAPSHOT_2026-07-23.md` - historical
  plan only; this report now supersedes its progress section.

## 4. Frozen scope, data, and research rules

### Project scope

The Axiom project covers:

1. two intraday trend-following strategy families;
2. one non-trivial intraday reversion family;
3. reproducible historical and chronological out-of-sample evaluation;
4. economic and statistical validation after realistic costs;
5. Alpaca paper-broker integration and operational controls;
6. execution and risk reporting; and
7. a mathematical CQF report with working code and honest limitations.

Unrelated option-pricing, local-volatility, DeepPDE, DeFi, large factor-zoo,
and unrestricted machine-learning work are excluded from the Axiom core.

### Data contract

- Primary symbol: SPY.
- Robustness symbols: QQQ and IWM.
- Primary frequency: 15-minute regular-trading-hours bars.
- Robustness frequencies: 30 and 60 minutes where specified.
- Time handling: source timestamps normalized to UTC with New York session
  logic.
- Canonical development panel: 117,192 rows, 39,064 per symbol, 1,508 sessions
  per symbol, 2020-01-02 through 2025-12-31.
- Locked 2026 data must be rejected by development runners and must not be read
  until the complete protocol is frozen and the user explicitly authorizes the
  one-time final test.

### Evaluation principles

- Signals calculated on bar `t` may first affect positions on bar `t+1`.
- Training always precedes testing; future rows cannot set current parameters.
- Transaction costs, turnover, drawdown, and failure modes are part of the
  result, not optional presentation details.
- Profitability is a primary economic objective, but it is not a software-test
  gate. Negative results must remain visible and must not be rewritten through
  silent parameter selection.
- No ranking, winner declaration, or deployment recommendation is allowed
  unless a promotion protocol is frozen before inspecting the relevant result.

## 5. Day-by-day completion ledger

| Day | Status | Completed work and evidence | Research conclusion / implication |
|---:|---|---|---|
| 1 | Complete | Initialized the repository, project scope, paper-only safety boundary, configuration, and decision log. Froze SPY as primary, QQQ/IWM for robustness, 15-minute bars as primary, and development/locked dates. | The research question and safety boundary are explicit. No real-money trading is authorized. |
| 2 | Complete | Built the package and test scaffolding plus a validated market-data pipeline with provider access, schemas, validators, local storage, resampling, and microstructure checks. | The project became executable and testable rather than notebook-only. |
| 3 | Complete | Hardened data architecture and mathematical/technical documentation: immutable provenance, time-zone/session rules, resampling conventions, and the core framework in the README and decisions log. | Research assumptions became auditable and reproducible. |
| 4 | Complete | Added immutable raw and canonical bars, quote/trade ingestion, event-bar construction, and external daily OHLCV reconciliation. | Multiple market-data representations can be validated without changing the canonical research series. |
| 5 | Complete | Built the full development dataset and EDA/stylized-facts bundle. Verified coverage, missingness, session integrity, return moments, tails, volatility clustering, intraday seasonality, dependence, and event-bar conservation. | Daily returns are heavy-tailed and negatively skewed; volatility clusters; dependence changes by regime. The available event-bar sample was too short for strong inference. |
| 6 | Complete | Implemented the frozen price-to-long-average trend-ratio baseline with one-bar delay, turnover, costs, diagnostics, plots, and findings. | Gross cumulative return was +26.14%, but the one-basis-point net result was -4.12%; turnover of 2,743 dominated the gross edge. |
| 7 | Complete | Ran predeclared trend-ratio sensitivity, regime, neighborhood-stability, holding, turnover, and break-even-cost analysis. | 14/36 configurations were net positive and 31/36 had positive break-even cost, but the exercise deliberately did not select a winner. |
| 8 | Complete | Implemented the second trend family using causal EMA/MACD logic with the same cost and timing discipline. | Gross return was -4.24%, net return -31.28%, net Sharpe -0.318, and turnover 3,318; predictive evidence was weak. |
| 9 | Complete | Ran the EMA/MACD sensitivity and stability grid with annual, regime, cost, turnover, and neighborhood diagnostics. | Only 3/108 configurations were net positive and 23/108 had positive break-even cost. No configuration was promoted. |
| 10 | Complete | Tested both trend families across SPY, QQQ, IWM and 15/30/60-minute bars, with deterministic report artifacts and non-degeneracy checks. | Results showed material symbol/frequency dependence and sign reversals. Thirty-minute cases were relatively better in places, but the evidence did not justify retrospective selection. |
| 11 | Complete | Added four expanding-history walk-forward folds with frozen baselines and strict train/test chronology for 2022-2025. | Aggregate OOS trend-ratio net return was -13.11% with Sharpe -0.140 and max drawdown -29.16%; EMA/MACD was -38.39% with Sharpe -0.915 and max drawdown -45.99%. |
| 12 | Complete | Built a deterministic event-driven replay engine and verified positions, P&L, turnover, event counts, and vectorized parity. | The event-driven implementation reproduces the vectorized research logic and exposes execution-state transitions explicitly. |
| 13 | Complete | Ran the event-driven engine over all walk-forward folds and produced deterministic fold, aggregate, position, event-count, and parity reports. | Exact parity held across all required comparisons while fold state reset flat, confirming that Day 11 results were not an artifact of vectorized implementation. |
| 14 | Complete | Performed development-only integration, cointegration, multiple-testing, fold-stability, and OU-eligibility diagnostics on SPY/QQQ/IWM pairs. | All three series were plausibly I(1), but 0/3 pairs passed the Holm-adjusted cointegration gate. No pair trade was manufactured. |
| 15 | Complete | Built a six-sleeve daily return panel, correlation and covariance diagnostics, PCA concentration, effective rank, diversification ratio, and ensemble-feasibility gates. | Across 1,508 aligned sessions, max absolute correlation was 0.818743, PC1 share 0.424795, effective rank 3.758647, and diversification ratio exceeded one. Combining sleeves was mechanically feasible, not necessarily profitable. |
| 16 | Complete, frozen, pushed on Day 17 branch | Evaluated equal weight, inverse volatility, and constrained minimum variance across the six frozen sleeves using train-only weights, four chronological folds, costs, concentration, drawdown, historical VaR, and ES. | Aggregate net returns were -29.63%, -29.48%, and -28.61%, respectively. The work is valid economic evidence, but no rule was ranked or selected. Further allocation expansion is deprioritized. |
| 17 | Complete, frozen, pushed | Implemented a non-trivial OU/VWAP residual reversion strategy with a variance-ratio gate, three predeclared calibrations, chronological folds, one-bar delay, overnight-flat logic, four cost levels, HAC inference, moving-block bootstrap, IC, PSR, and DSR. Produced an eight-file deterministic bundle. | Fast and base equal-weight cases were negative at one basis point. Slow was +6.03% at one basis point and +3.02% at five basis points, with annualized Sharpe 0.639 and max drawdown -1.95% at one basis point. HAC t-stat was 1.421, the mean CI crossed zero, and DSR was 0.626; promising, but statistically inconclusive and not promoted. |
| 18 | Complete, frozen, pushed | Implemented a broker-neutral read-only Alpaca paper adapter with exact endpoint enforcement, environment-backed redacted credentials, account/clock/SPY-QQQ-IWM preflight, capability mapping, fail-closed error taxonomy, deterministic five-file artifacts, and 62 synthetic safety tests. The approved live preflight passed on 2026-08-02. | Account status was active and unblocked; SPY, QQQ, and IWM were active, tradable, shortable, easy to borrow, and fractionable. All gates passed, no credential or financial identifier was persisted, locked data remained untouched, and zero orders were submitted. This proves connectivity and safety boundaries, not execution readiness or profitability. |

## 6. Current evidence dashboard

### Complete

- Reproducible development data and integrity checks.
- Two required trend-following families.
- Sensitivity, regime, cost, turnover, and cross-market robustness analysis.
- Chronological walk-forward evaluation.
- Vectorized/event-driven equivalence.
- Cointegration feasibility gate with an honest no-pair result.
- Multi-sleeve dependence and allocation analysis.
- Non-trivial reversion family with formal uncertainty estimates.
- Deterministic artifacts with SHA-256 manifests for Days 10-18.
- Read-only Alpaca paper endpoint, account, clock, and core-symbol preflight.
- Day 18 focused tests: 62 passed.
- Full repository suite after Day 18: 795 passed.
- Day 18 artifact hash audit and fixed-input byte-for-byte replay: passed.

### Partial

- Transaction costs are stress assumptions, not yet calibrated from observed
  paper fills.
- VaR/ES and drawdown exist for Day 16, but dynamic strategy-level risk and
  Beta-to-SPY reporting are not complete.
- Event bars and quote/trade infrastructure exist, but the captured sample is
  not yet sufficient for broad empirical claims.
- Alpaca paper account and asset eligibility were revalidated on 2026-08-02,
  but order, stream, fill, and reconciliation behaviour remain untested.
- The README and final narrative do not yet summarize all results through Day
  18.

### Not complete

- Order-state machine with rejected, canceled, partial, duplicate, stale, and
  inconsistent responses.
- Position, cash, order, and fill reconciliation.
- WebSocket staleness detection, reconnect/backoff, circuit breaker, and kill
  switch.
- Controlled Alpaca paper-trading run with saved orders, fills, P&L, and logs.
- Realized-slippage decomposition and backtest-versus-paper comparison.
- Docker image, scheduling/runbook, and deployment verification.
- Final mathematical report, numerical-methods table, limitations, conclusion,
  and defense material.
- One-time locked 2026 test, final named report, code zip, and hand-signed
  declaration.

## 7. Profitability and promotion status

Profitability remains a priority. The correct interpretation of the project so
far is:

1. the baseline trend evidence is economically weak after costs;
2. the pair-reversion route was infeasible under its frozen statistical gate;
3. the Day 17 slow reversion calibration is the first positive net candidate,
   but the result is not statistically conclusive; and
4. no configuration has yet earned promotion to paper trading as a claimed
   alpha strategy.

The next profitability decision must not be made by choosing the best-looking
development row after the fact. Before any candidate is labeled "promoted," a
separate protocol should freeze the required cost robustness, chronological
fold stability, trade count, drawdown, inference, operational safety, and paper
execution evidence. The locked 2026 interval must remain untouched until that
protocol and all code are frozen.

## 8. Revised plan for Days 18-25

The original schedule expected broker work to start around Day 16. Days 16-17
were instead used to close two important research gaps: portfolio/economic
validation and the required non-trivial reversion strategy with inference. That
was defensible, but the project should now stop expanding historical research
and move to the mandatory operational and reporting work.

| Day | Planned deliverable | Completion gate |
|---:|---|---|
| 18 | **Complete - Alpaca paper-broker boundary and preflight.** Implemented a broker interface, environment-backed credential loading, paper endpoint enforcement, read-only clock/account/asset checks, symbol eligibility checks, order-type/TIF capability mapping, and fail-closed error handling. | Passed: no secret in repository or artifacts; paper endpoint cannot be bypassed; 62 focused tests and 795 full-suite tests passed; the live read-only preflight passed; zero orders were submitted. |
| 19 | **Order-state machine.** Model submit/acknowledge/partial-fill/fill/cancel/reject/expire transitions, client-order idempotency, duplicate/out-of-order events, timeouts, and contradictory broker responses. | Known-answer tests cover normal and failure paths; every transition is auditable and illegal transitions fail closed. |
| 20 | **Reconciliation and monitoring.** Reconcile intended orders against broker orders, fills, positions, cash, and local state; add stale-stream detection, reconnect/backoff, circuit breaker, exposure caps, and kill switch. | Restart-safe reconciliation and simulated stale/partial/incorrect-response scenarios pass without opening unintended exposure. |
| 21 | **Controlled paper execution.** Connect a predeclared strategy candidate to the paper-only loop, use deliberately bounded size and safe order parameters, persist signals/orders/fills/positions/P&L, and test recovery. | Explicit user authorization before order submission; paper account only; complete audit trail; no unresolved position or order mismatch at shutdown. |
| 22 | **Execution and live-performance validation.** Decompose decision-price, arrival-price, spread, delay, and fill slippage; compare expected versus paper outcomes; add rolling drawdown, VaR/ES, Beta-to-SPY, exposure, turnover, and P&L reporting. | Reconciled fills are the source of realized execution metrics; assumptions and small-sample limitations are explicit. |
| 23 | **Reproducible operations.** Add dependency lock, container, scheduled-run entry points, configuration validation, health checks, runbook, persistence/backup policy, and CI smoke tests. | A clean environment can reproduce a safe paper-mode startup and shutdown from documented instructions. |
| 24 | **Final analytical report.** Write the mathematical models, numerical/statistical methods table, data and leakage controls, strategy experiments, stress tests, execution evidence, limitations, conclusions, and further work. Update the README to match the final repository. | Every important claim traces to a table, figure, artifact, or test; no notebook dump; negative and inconclusive findings remain visible. |
| 25 | **Final freeze and submission package.** Freeze code/config/report, run the full reproducibility audit, decide whether the predeclared locked-test gate is ready, run the locked test once only with explicit authorization, package the correctly named report and code zip, add the hand-signed declaration, and prepare viva questions. | Clean clone works; full tests pass; manifests verify; report and zip naming is correct; declaration is present; locked result is reported without retuning. |

## 9. Immediate next action: Day 19

Day 19 should begin with a frozen synthetic order-state specification before
implementation. It should define:

- exact internal order identifiers and client-order idempotency rules;
- legal state transitions for pending, accepted, new, partially filled, filled,
  canceled, rejected, expired, replaced, and done-for-day messages;
- cumulative versus incremental fill accounting;
- duplicate, missing, stale, contradictory, and out-of-order message handling;
- timeout semantics and fail-closed recovery state;
- append-only transition evidence and deterministic replay; and
- synthetic tests only, with no order submission.

Day 19 does not require broker credentials or network access. The Alpaca paper
credentials remain local and must never be copied into code, fixtures,
artifacts, Markdown, or chat output.

## 10. Risks and open decisions

| Risk / decision | Current handling | Required resolution |
|---|---|---|
| Development overfitting | Predeclared grids, locked 2026 boundary, no silent ranking | Freeze candidate-promotion gates before further selection. |
| Weak historical profitability | Reported without suppression | Use paper execution to estimate implementability; do not retune on locked data. |
| Broker/account drift | Read-only preflight passed on 2026-08-02 | Recheck before every later paper-order session; never treat Day 18 as permanent eligibility. |
| Operational failure | Not yet fully implemented | Complete Days 19-21 failure-state and reconciliation gates. |
| Unrealistic costs | Cost stresses are assumptions | Calibrate and decompose against paper fills on Day 22. |
| Schedule pressure | Core research took two extra days before broker work | Do not expand allocation, ML, or unrelated topics before required broker and report work. |
| Submission compliance | Requirements identified but package not built | Complete Day 25 naming, declaration, working-code, and archive checks. |
| Academic integrity and defense | Code and evidence are reproducible, but final explanation is unwritten | Write the report in the user's own defensible voice and prepare line-by-line mathematical and implementation explanations. |

## 11. Repository evidence index

- Decision log: [DECISIONS.md](DECISIONS.md)
- Day 5 EDA findings: [DAY05_FINDINGS.md](../artifacts/day05/day05_report_v1/DAY05_FINDINGS.md)
- Day 6 trend-ratio report: [findings.md](../artifacts/day06/trend_ratio_baseline_v1/findings.md)
- Day 7 trend-ratio sensitivity: [findings.md](../artifacts/day07/trend_ratio_sensitivity_v1/findings.md)
- Day 8 EMA/MACD report: [findings.md](../artifacts/day08/ema_macd_baseline_v1/findings.md)
- Day 9 EMA/MACD sensitivity: [findings.md](../artifacts/day09/ema_macd_sensitivity_v1/findings.md)
- Day 10 cross-market robustness: [report.md](../artifacts/day10/report.md)
- Day 11 walk-forward results: [report.md](../artifacts/day11/report.md)
- Day 12 event-driven parity: [report.md](../artifacts/day12/report.md)
- Day 13 event-driven walk-forward: [report.md](../artifacts/day13/report.md)
- Day 14 cointegration feasibility: [report.md](../artifacts/day14/report.md)
- Day 15 diversification: [report.md](../artifacts/day15/report.md)
- Day 16 specification: [DAY16_PORTFOLIO_VALIDATION_SPECIFICATION.md](DAY16_PORTFOLIO_VALIDATION_SPECIFICATION.md)
- Day 16 portfolio/economic results: [report.md](../artifacts/day16/report.md)
- Day 17 specification: [DAY17_REVERSION_INFERENCE_SPECIFICATION.md](DAY17_REVERSION_INFERENCE_SPECIFICATION.md)
- Day 17 reversion/inference results: [report.md](../artifacts/day17/report.md)
- Day 18 specification: [DAY18_ALPACA_PAPER_BOUNDARY_SPECIFICATION.md](DAY18_ALPACA_PAPER_BOUNDARY_SPECIFICATION.md)
- Day 18 paper preflight: [report.md](../artifacts/day18/report.md)

## 12. Required update after every completed day

After each day, update this file with:

1. completion date, branch, commit, push, and merge state;
2. implemented scope and any change from the planned scope;
3. focused and full test counts;
4. artifact path, manifest/hash audit, and replay result where applicable;
5. headline economic and statistical findings, including negative results;
6. data interval used and confirmation that locked data remained untouched;
7. limitations, new risks, and decisions that remain open; and
8. the exact next-day deliverable and its completion gate.

This update protocol is part of project governance. A day is not considered
fully handed off until the code, evidence, and this progress record agree.
