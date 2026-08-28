"""Execution shortfall, realized P&L, and dynamic risk validation for Day 22."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import exchange_calendars as xcals
import numpy as np
import pandas as pd


BASIS_POINTS: Final[Decimal] = Decimal("10000")
STARTING_EQUITY: Final[Decimal] = Decimal("100000")
MAX_QUOTE_AGE: Final[timedelta] = timedelta(seconds=2)
RISK_MINIMUM_OBSERVATIONS: Final[int] = 20
ANNUALIZATION_FACTOR: Final[float] = 252.0
PURPOSE_ORDER: Final[tuple[str, ...]] = (
    "strategy_signal",
    "calibration_probe",
)
SCHEDULE_TIMES: Final[tuple[str, ...]] = ("10:15", "14:15")

EXECUTION_SHORTFALL_COLUMNS: Final[tuple[str, ...]] = (
    "execution_id",
    "round_trip_id",
    "purpose",
    "leg",
    "symbol",
    "side",
    "quantity",
    "decision_at_utc",
    "decision_price",
    "quote_at_utc",
    "bid_price",
    "ask_price",
    "arrival_mid",
    "arrival_touch",
    "submitted_at_utc",
    "broker_submitted_at_utc",
    "filled_at_utc",
    "fill_price",
    "commission",
    "total_shortfall_bps",
    "delay_bps",
    "spread_bps",
    "residual_bps",
    "decomposition_error_bps",
    "decision_to_submit_ms",
    "broker_ack_ms",
    "fill_latency_ms",
)
ROUND_TRIP_COLUMNS: Final[tuple[str, ...]] = (
    "round_trip_id",
    "purpose",
    "symbol",
    "direction",
    "quantity",
    "entry_execution_id",
    "exit_execution_id",
    "entry_fill_price",
    "exit_fill_price",
    "entry_filled_at_utc",
    "exit_filled_at_utc",
    "holding_seconds",
    "gross_pnl",
    "commission",
    "net_pnl",
    "execution_shortfall_usd",
)
DAILY_PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "strategy_net_pnl",
    "prior_equity",
    "ending_equity",
    "strategy_return",
    "spy_return",
    "running_peak_equity",
    "drawdown",
    "gross_exposure",
    "net_exposure",
    "turnover_notional",
    "rolling_20d_volatility",
    "rolling_20d_var_95",
    "rolling_20d_es_95",
)
RISK_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "purpose",
    "observations",
    "risk_metrics_available",
    "availability_reason",
    "starting_equity",
    "ending_equity",
    "cumulative_pnl",
    "cumulative_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "historical_var_95",
    "historical_es_95",
    "beta_to_spy",
    "total_turnover_notional",
    "maximum_gross_exposure",
    "maximum_absolute_net_exposure",
    "strategy_round_trips",
    "hit_rate",
    "profit_factor",
)
CAMPAIGN_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "schedule_order",
    "campaign_id",
    "session_date",
    "scheduled_at_utc",
    "scheduled_at_new_york",
    "entry_side",
    "quantity",
    "maximum_notional_usd",
    "purpose",
    "authorization_granted",
    "status",
)
CAMPAIGN_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "purpose",
    "alpha_eligible",
    "executions",
    "round_trips",
    "filled_quantity",
    "mean_total_shortfall_bps",
    "median_fill_latency_ms",
    "gross_pnl",
    "commission",
    "net_pnl",
)


class ExecutionPerformanceValidationError(ValueError):
    """Raised when Day 22 evidence is malformed or contradictory."""


def _decimal(value: object, *, name: str, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ExecutionPerformanceValidationError(f"{name} must be decimal-like.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExecutionPerformanceValidationError(
            f"{name} must be decimal-like."
        ) from exc
    if not number.is_finite() or number < 0 or (not allow_zero and number == 0):
        raise ExecutionPerformanceValidationError(f"{name} is outside its range.")
    return number


def _signed_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ExecutionPerformanceValidationError(f"{name} must be decimal-like.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExecutionPerformanceValidationError(
            f"{name} must be decimal-like."
        ) from exc
    if not number.is_finite():
        raise ExecutionPerformanceValidationError(f"{name} must be finite.")
    return number


def _utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ExecutionPerformanceValidationError(
            f"{name} must be a timezone-aware datetime."
        )
    return value.astimezone(timezone.utc)


def _identifier(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ExecutionPerformanceValidationError(
            f"{name} must be a non-empty identifier without whitespace."
        )
    return value


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    execution_id: str
    round_trip_id: str
    purpose: str
    leg: str
    symbol: str
    side: str
    quantity: Decimal
    decision_at: datetime
    decision_price: Decimal
    quote_at: datetime
    bid_price: Decimal
    ask_price: Decimal
    submitted_at: datetime
    broker_submitted_at: datetime
    filled_at: datetime
    fill_price: Decimal
    commission: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _identifier(self.execution_id, name="execution_id")
        )
        object.__setattr__(
            self,
            "round_trip_id",
            _identifier(self.round_trip_id, name="round_trip_id"),
        )
        if self.purpose not in PURPOSE_ORDER:
            raise ExecutionPerformanceValidationError("purpose is not frozen.")
        if self.leg not in {"entry", "exit"}:
            raise ExecutionPerformanceValidationError("leg must be entry or exit.")
        if self.symbol != "SPY":
            raise ExecutionPerformanceValidationError("symbol must be exactly SPY.")
        if self.side not in {"buy", "sell"}:
            raise ExecutionPerformanceValidationError("side must be buy or sell.")
        for name in (
            "quantity",
            "decision_price",
            "bid_price",
            "ask_price",
            "fill_price",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "commission",
            _decimal(self.commission, name="commission", allow_zero=True),
        )
        for name in (
            "decision_at",
            "quote_at",
            "submitted_at",
            "broker_submitted_at",
            "filled_at",
        ):
            object.__setattr__(self, name, _utc(getattr(self, name), name=name))
        if self.bid_price > self.ask_price:
            raise ExecutionPerformanceValidationError("quote is crossed.")
        times = (
            self.decision_at,
            self.quote_at,
            self.submitted_at,
            self.broker_submitted_at,
            self.filled_at,
        )
        if any(left > right for left, right in zip(times, times[1:])):
            raise ExecutionPerformanceValidationError(
                "execution timestamps are not chronological."
            )
        if self.submitted_at - self.quote_at > MAX_QUOTE_AGE:
            raise ExecutionPerformanceValidationError(
                "quote is stale at order submission."
            )


@dataclass(frozen=True, slots=True)
class DailyPortfolioSnapshot:
    session_date: date
    strategy_net_pnl: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    turnover_notional: Decimal
    spy_return: float

    def __post_init__(self) -> None:
        if not isinstance(self.session_date, date) or isinstance(
            self.session_date, datetime
        ):
            raise ExecutionPerformanceValidationError(
                "session_date must be a date."
            )
        object.__setattr__(
            self,
            "strategy_net_pnl",
            _signed_decimal(self.strategy_net_pnl, name="strategy_net_pnl"),
        )
        for name in ("gross_exposure", "turnover_notional"):
            object.__setattr__(
                self,
                name,
                _decimal(getattr(self, name), name=name, allow_zero=True),
            )
        object.__setattr__(
            self,
            "net_exposure",
            _signed_decimal(self.net_exposure, name="net_exposure"),
        )
        if isinstance(self.spy_return, bool) or not math.isfinite(float(self.spy_return)):
            raise ExecutionPerformanceValidationError("spy_return must be finite.")
        if float(self.spy_return) <= -1.0:
            raise ExecutionPerformanceValidationError(
                "spy_return must be greater than -1."
            )


@dataclass(frozen=True, slots=True)
class Day22AnalysisResults:
    execution_shortfall: tuple[Mapping[str, object], ...]
    round_trip_pnl: tuple[Mapping[str, object], ...]
    daily_performance: tuple[Mapping[str, object], ...]
    risk_summary: tuple[Mapping[str, object], ...]
    campaign_schedule: tuple[Mapping[str, object], ...]
    campaign_summary: tuple[Mapping[str, object], ...]
    evaluation_complete: bool

    def __post_init__(self) -> None:
        for name in (
            "execution_shortfall",
            "round_trip_pnl",
            "daily_performance",
            "risk_summary",
            "campaign_schedule",
            "campaign_summary",
        ):
            rows = getattr(self, name)
            object.__setattr__(
                self,
                name,
                tuple(MappingProxyType(dict(row)) for row in rows),
            )


def _shortfall_row(record: ExecutionRecord) -> dict[str, object]:
    sign = Decimal("1") if record.side == "buy" else Decimal("-1")
    mid = (record.bid_price + record.ask_price) / Decimal("2")
    touch = record.ask_price if record.side == "buy" else record.bid_price
    total = sign * (record.fill_price - record.decision_price) / record.decision_price * BASIS_POINTS
    delay = sign * (mid - record.decision_price) / record.decision_price * BASIS_POINTS
    spread = sign * (touch - mid) / record.decision_price * BASIS_POINTS
    residual = sign * (record.fill_price - touch) / record.decision_price * BASIS_POINTS
    error = total - delay - spread - residual
    if abs(error) > Decimal("1e-9"):
        raise RuntimeError("Day 22 shortfall decomposition failed.")
    return {
        "execution_id": record.execution_id,
        "round_trip_id": record.round_trip_id,
        "purpose": record.purpose,
        "leg": record.leg,
        "symbol": record.symbol,
        "side": record.side,
        "quantity": record.quantity,
        "decision_at_utc": record.decision_at.isoformat(),
        "decision_price": record.decision_price,
        "quote_at_utc": record.quote_at.isoformat(),
        "bid_price": record.bid_price,
        "ask_price": record.ask_price,
        "arrival_mid": mid,
        "arrival_touch": touch,
        "submitted_at_utc": record.submitted_at.isoformat(),
        "broker_submitted_at_utc": record.broker_submitted_at.isoformat(),
        "filled_at_utc": record.filled_at.isoformat(),
        "fill_price": record.fill_price,
        "commission": record.commission,
        "total_shortfall_bps": float(total),
        "delay_bps": float(delay),
        "spread_bps": float(spread),
        "residual_bps": float(residual),
        "decomposition_error_bps": float(error),
        "decision_to_submit_ms": (
            (record.submitted_at - record.decision_at).total_seconds() * 1000.0
        ),
        "broker_ack_ms": (
            (record.broker_submitted_at - record.submitted_at).total_seconds()
            * 1000.0
        ),
        "fill_latency_ms": (
            (record.filled_at - record.submitted_at).total_seconds() * 1000.0
        ),
    }


def _round_trip_rows(
    records: tuple[ExecutionRecord, ...],
) -> tuple[dict[str, object], ...]:
    groups: dict[str, list[ExecutionRecord]] = {}
    for record in records:
        groups.setdefault(record.round_trip_id, []).append(record)
    rows: list[dict[str, object]] = []
    for round_trip_id in sorted(groups):
        group = groups[round_trip_id]
        if len(group) != 2:
            raise ExecutionPerformanceValidationError(
                "Each round trip must contain exactly two executions."
            )
        entries = tuple(item for item in group if item.leg == "entry")
        exits = tuple(item for item in group if item.leg == "exit")
        if len(entries) != 1 or len(exits) != 1:
            raise ExecutionPerformanceValidationError(
                "Round trip requires one entry and one exit."
            )
        entry, exit_record = entries[0], exits[0]
        if (
            entry.purpose != exit_record.purpose
            or entry.symbol != exit_record.symbol
            or entry.quantity != exit_record.quantity
            or entry.side == exit_record.side
            or entry.filled_at >= exit_record.filled_at
        ):
            raise ExecutionPerformanceValidationError(
                "Round-trip legs are contradictory."
            )
        direction = "long" if entry.side == "buy" else "short"
        gross = (
            entry.quantity * (exit_record.fill_price - entry.fill_price)
            if direction == "long"
            else entry.quantity * (entry.fill_price - exit_record.fill_price)
        )
        commission = entry.commission + exit_record.commission
        net = gross - commission
        entry_sign = Decimal("1") if entry.side == "buy" else Decimal("-1")
        exit_sign = Decimal("1") if exit_record.side == "buy" else Decimal("-1")
        shortfall_usd = (
            entry_sign * (entry.fill_price - entry.decision_price) * entry.quantity
            + exit_sign
            * (exit_record.fill_price - exit_record.decision_price)
            * exit_record.quantity
        )
        rows.append(
            {
                "round_trip_id": round_trip_id,
                "purpose": entry.purpose,
                "symbol": entry.symbol,
                "direction": direction,
                "quantity": entry.quantity,
                "entry_execution_id": entry.execution_id,
                "exit_execution_id": exit_record.execution_id,
                "entry_fill_price": entry.fill_price,
                "exit_fill_price": exit_record.fill_price,
                "entry_filled_at_utc": entry.filled_at.isoformat(),
                "exit_filled_at_utc": exit_record.filled_at.isoformat(),
                "holding_seconds": (
                    exit_record.filled_at - entry.filled_at
                ).total_seconds(),
                "gross_pnl": gross,
                "commission": commission,
                "net_pnl": net,
                "execution_shortfall_usd": shortfall_usd,
            }
        )
    return tuple(rows)


def _historical_risk(values: np.ndarray) -> tuple[float, float]:
    quantile = float(np.quantile(values, 0.05, method="linear"))
    tail = values[values <= quantile]
    return max(0.0, -quantile), max(0.0, -float(np.mean(tail)))


def _daily_rows(
    snapshots: tuple[DailyPortfolioSnapshot, ...],
    *,
    starting_equity: Decimal,
) -> tuple[dict[str, object], ...]:
    if len({item.session_date for item in snapshots}) != len(snapshots):
        raise ExecutionPerformanceValidationError("Daily session dates must be unique.")
    ordered = tuple(sorted(snapshots, key=lambda item: item.session_date))
    equity = starting_equity
    peak = starting_equity
    returns: list[float] = []
    rows: list[dict[str, object]] = []
    for item in ordered:
        prior = equity
        equity = prior + item.strategy_net_pnl
        if equity <= 0:
            raise ExecutionPerformanceValidationError("Paper equity became non-positive.")
        strategy_return = float(item.strategy_net_pnl / prior)
        returns.append(strategy_return)
        peak = max(peak, equity)
        drawdown = float(equity / peak - Decimal("1"))
        rolling_vol: float | None = None
        rolling_var: float | None = None
        rolling_es: float | None = None
        if len(returns) >= RISK_MINIMUM_OBSERVATIONS:
            window = np.asarray(returns[-RISK_MINIMUM_OBSERVATIONS:], dtype="float64")
            rolling_vol = float(np.std(window, ddof=1) * math.sqrt(ANNUALIZATION_FACTOR))
            rolling_var, rolling_es = _historical_risk(window)
        rows.append(
            {
                "session_date": item.session_date.isoformat(),
                "strategy_net_pnl": item.strategy_net_pnl,
                "prior_equity": prior,
                "ending_equity": equity,
                "strategy_return": strategy_return,
                "spy_return": float(item.spy_return),
                "running_peak_equity": peak,
                "drawdown": drawdown,
                "gross_exposure": item.gross_exposure,
                "net_exposure": item.net_exposure,
                "turnover_notional": item.turnover_notional,
                "rolling_20d_volatility": rolling_vol,
                "rolling_20d_var_95": rolling_var,
                "rolling_20d_es_95": rolling_es,
            }
        )
    return tuple(rows)


def _risk_row(
    daily_rows: tuple[dict[str, object], ...],
    round_trips: tuple[dict[str, object], ...],
    *,
    starting_equity: Decimal,
) -> dict[str, object]:
    returns = np.asarray([row["strategy_return"] for row in daily_rows], dtype="float64")
    spy = np.asarray([row["spy_return"] for row in daily_rows], dtype="float64")
    available = len(returns) >= RISK_MINIMUM_OBSERVATIONS
    volatility: float | None = None
    sharpe: float | None = None
    var: float | None = None
    es: float | None = None
    beta: float | None = None
    reason = "available" if available else "insufficient_daily_observations"
    if available:
        sample_std = float(np.std(returns, ddof=1))
        volatility = sample_std * math.sqrt(ANNUALIZATION_FACTOR)
        sharpe = (
            None
            if sample_std == 0.0
            else float(np.mean(returns) / sample_std * math.sqrt(ANNUALIZATION_FACTOR))
        )
        var, es = _historical_risk(returns)
        spy_variance = float(np.var(spy, ddof=1))
        if math.isfinite(spy_variance) and spy_variance > 0.0:
            beta = float(np.cov(returns, spy, ddof=1)[0, 1] / spy_variance)
        else:
            available = False
            reason = "spy_variance_not_positive"
    strategy_trips = tuple(
        row for row in round_trips if row["purpose"] == "strategy_signal"
    )
    net_pnl = np.asarray([float(row["net_pnl"]) for row in strategy_trips])
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl < 0]
    hit_rate = None if len(net_pnl) == 0 else float(len(wins) / len(net_pnl))
    profit_factor = (
        None
        if len(losses) == 0
        else float(wins.sum() / abs(losses.sum()))
    )
    ending_equity = (
        starting_equity if not daily_rows else daily_rows[-1]["ending_equity"]
    )
    return {
        "purpose": "strategy_signal",
        "observations": len(daily_rows),
        "risk_metrics_available": available,
        "availability_reason": reason,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "cumulative_pnl": Decimal(str(ending_equity)) - starting_equity,
        "cumulative_return": float(Decimal(str(ending_equity)) / starting_equity - Decimal("1")),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": (
            None if not daily_rows else min(float(row["drawdown"]) for row in daily_rows)
        ),
        "historical_var_95": var,
        "historical_es_95": es,
        "beta_to_spy": beta,
        "total_turnover_notional": sum(
            (Decimal(str(row["turnover_notional"])) for row in daily_rows),
            Decimal("0"),
        ),
        "maximum_gross_exposure": max(
            (Decimal(str(row["gross_exposure"])) for row in daily_rows),
            default=Decimal("0"),
        ),
        "maximum_absolute_net_exposure": max(
            (abs(Decimal(str(row["net_exposure"]))) for row in daily_rows),
            default=Decimal("0"),
        ),
        "strategy_round_trips": len(strategy_trips),
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
    }


def build_campaign_schedule(
    activation_date: date,
    *,
    campaign_id: str = "day22_calibration_v1",
) -> tuple[dict[str, object], ...]:
    """Build ten fixed, unauthorized calibration slots across five XNYS sessions."""

    if not isinstance(activation_date, date) or isinstance(activation_date, datetime):
        raise ExecutionPerformanceValidationError("activation_date must be a date.")
    _identifier(campaign_id, name="campaign_id")
    calendar = xcals.get_calendar("XNYS")
    start = pd.Timestamp(activation_date)
    sessions = calendar.sessions_in_range(start, start + pd.Timedelta(days=20))[:5]
    if len(sessions) != 5:
        raise ExecutionPerformanceValidationError("Five XNYS sessions are unavailable.")
    rows: list[dict[str, object]] = []
    order = 0
    for session in sessions:
        session_text = pd.Timestamp(session).strftime("%Y-%m-%d")
        for local_time in SCHEDULE_TIMES:
            order += 1
            local = pd.Timestamp(
                f"{session_text} {local_time}", tz="America/New_York"
            )
            rows.append(
                {
                    "schedule_order": order,
                    "campaign_id": campaign_id,
                    "session_date": session_text,
                    "scheduled_at_utc": local.tz_convert("UTC").isoformat(),
                    "scheduled_at_new_york": local.isoformat(),
                    "entry_side": "buy" if order % 2 == 1 else "sell",
                    "quantity": Decimal("0.01"),
                    "maximum_notional_usd": Decimal("10.00"),
                    "purpose": "calibration_probe",
                    "authorization_granted": False,
                    "status": "planned_not_authorized",
                }
            )
    return tuple(rows)


def _campaign_summary(
    shortfall: tuple[dict[str, object], ...],
    round_trips: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for purpose in PURPOSE_ORDER:
        executions = tuple(row for row in shortfall if row["purpose"] == purpose)
        trips = tuple(row for row in round_trips if row["purpose"] == purpose)
        rows.append(
            {
                "purpose": purpose,
                "alpha_eligible": purpose == "strategy_signal",
                "executions": len(executions),
                "round_trips": len(trips),
                "filled_quantity": sum(
                    (Decimal(str(row["quantity"])) for row in executions),
                    Decimal("0"),
                ),
                "mean_total_shortfall_bps": (
                    None
                    if not executions
                    else float(np.mean([row["total_shortfall_bps"] for row in executions]))
                ),
                "median_fill_latency_ms": (
                    None
                    if not executions
                    else float(np.median([row["fill_latency_ms"] for row in executions]))
                ),
                "gross_pnl": sum(
                    (Decimal(str(row["gross_pnl"])) for row in trips), Decimal("0")
                ),
                "commission": sum(
                    (Decimal(str(row["commission"])) for row in trips), Decimal("0")
                ),
                "net_pnl": sum(
                    (Decimal(str(row["net_pnl"])) for row in trips), Decimal("0")
                ),
            }
        )
    return tuple(rows)


def analyze_execution_performance(
    executions: Sequence[ExecutionRecord],
    daily_snapshots: Sequence[DailyPortfolioSnapshot],
    *,
    activation_date: date,
    starting_equity: Decimal = STARTING_EQUITY,
) -> Day22AnalysisResults:
    """Validate evidence and calculate the complete frozen Day 22 tables."""

    records = tuple(executions)
    snapshots = tuple(daily_snapshots)
    if not records:
        raise ExecutionPerformanceValidationError("Execution records cannot be empty.")
    if not snapshots:
        raise ExecutionPerformanceValidationError("Daily snapshots cannot be empty.")
    if not all(isinstance(item, ExecutionRecord) for item in records):
        raise TypeError("executions must contain ExecutionRecord objects.")
    if not all(isinstance(item, DailyPortfolioSnapshot) for item in snapshots):
        raise TypeError("daily_snapshots must contain DailyPortfolioSnapshot objects.")
    initial_equity = _decimal(starting_equity, name="starting_equity")
    ids = tuple(item.execution_id for item in records)
    if len(ids) != len(set(ids)):
        raise ExecutionPerformanceValidationError("execution_id values must be unique.")
    ordered = tuple(sorted(records, key=lambda item: (item.decision_at, item.execution_id)))
    shortfall = tuple(_shortfall_row(item) for item in ordered)
    round_trips = _round_trip_rows(ordered)
    strategy_trade_pnl = sum(
        (
            Decimal(str(row["net_pnl"]))
            for row in round_trips
            if row["purpose"] == "strategy_signal"
        ),
        Decimal("0"),
    )
    strategy_daily_pnl = sum(
        (item.strategy_net_pnl for item in snapshots), Decimal("0")
    )
    if strategy_trade_pnl != strategy_daily_pnl:
        raise ExecutionPerformanceValidationError(
            "Strategy daily P&L does not reconcile to round trips."
        )
    daily = _daily_rows(snapshots, starting_equity=initial_equity)
    risk = (_risk_row(daily, round_trips, starting_equity=initial_equity),)
    schedule = build_campaign_schedule(activation_date)
    campaign = _campaign_summary(shortfall, round_trips)
    for rows, columns, name in (
        (shortfall, EXECUTION_SHORTFALL_COLUMNS, "execution_shortfall"),
        (round_trips, ROUND_TRIP_COLUMNS, "round_trip_pnl"),
        (daily, DAILY_PERFORMANCE_COLUMNS, "daily_performance"),
        (risk, RISK_SUMMARY_COLUMNS, "risk_summary"),
        (schedule, CAMPAIGN_SCHEDULE_COLUMNS, "campaign_schedule"),
        (campaign, CAMPAIGN_SUMMARY_COLUMNS, "campaign_summary"),
    ):
        if any(tuple(row) != columns for row in rows):
            raise RuntimeError(f"Day 22 {name} schema changed.")
    return Day22AnalysisResults(
        execution_shortfall=shortfall,
        round_trip_pnl=round_trips,
        daily_performance=daily,
        risk_summary=risk,
        campaign_schedule=schedule,
        campaign_summary=campaign,
        evaluation_complete=True,
    )
