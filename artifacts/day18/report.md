# Day 18 Alpaca Paper-Broker Preflight

## Outcome

- Preflight: **PASS**
- Endpoint: `https://paper-api.alpaca.markets`
- Broker SDK: `alpaca-py 0.43.5`
- Broker timestamp: `2026-08-02T18:02:17.211412Z`
- Market open at snapshot: `false`
- Credentials loaded: `true`
- Credential values persisted: `false`
- Order submission enabled: `false`
- Order submission occurred: `false`

## Mechanical gates

- Account gate: `true`
- Clock gate: `true`
- Asset gate: `true`

## Core-symbol eligibility

- SPY: long=true, short=true, fractional=true
- QQQ: long=true, short=true, fractional=true
- IWM: long=true, short=true, fractional=true

## Interpretation

This snapshot verifies a read-only connection to the frozen Alpaca paper
endpoint and checks the current account, clock, and core-symbol state. It does
not authorize an order, validate fill handling, or demonstrate profitable
paper execution. All provider order capabilities remain unauthorized on Day
18. Day 19 is limited to a synthetic order-state machine unless a later frozen
contract explicitly changes that boundary.

## Redaction and limitations

No credential value, account identifier, account number, balance, cash, buying
power, equity, or portfolio value is included. Broker state and asset
eligibility can change after this snapshot and must be rechecked before any
future paper-order session.
