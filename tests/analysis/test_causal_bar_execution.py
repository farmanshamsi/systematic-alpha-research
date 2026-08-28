"""Known-answer tests for model-independent causal bar accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_alpha.analysis.causal_bar_execution import (
    CausalBarExecutionError,
    apply_causal_next_open_overnight_flat,
)


def timing_input() -> pd.DataFrame:
    """Return two sessions covering every turnover transition."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 14:30Z",
                    "2025-01-02 14:45Z",
                    "2025-01-02 15:00Z",
                    "2025-01-02 15:15Z",
                    "2025-01-02 15:30Z",
                    "2025-01-02 15:45Z",
                    "2025-01-03 14:30Z",
                    "2025-01-03 14:45Z",
                    "2025-01-03 15:00Z",
                ],
                utc=True,
            ),
            "symbol": ["SPY"] * 9,
            "session_date": ["2025-01-02"] * 6
            + ["2025-01-03"] * 3,
            "open": [
                100.0,
                110.0,
                121.0,
                133.1,
                146.41,
                161.051,
                200.0,
                180.0,
                198.0,
            ],
            "close": [
                101.0,
                111.0,
                122.0,
                134.0,
                147.0,
                177.1561,
                201.0,
                181.0,
                217.8,
            ],
            "signal": [1, -1, 0, 1, 1, -1, 1, 1, 0],
            "signal_available": [True] * 9,
            "raw_feature": np.arange(9, dtype="float64"),
        }
    )


def sequential_oracle(
    source: pd.DataFrame,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    """Calculate the timing contract with a deliberately simple loop."""

    rows: list[dict[str, float | int]] = []
    ordered = source.sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    for _, group in ordered.groupby("symbol", observed=True, sort=True):
        part = group.reset_index(drop=True)
        signals = part["signal"].to_numpy(dtype="int8")
        opens = part["open"].to_numpy(dtype="float64")
        closes = part["close"].to_numpy(dtype="float64")
        sessions = part["session_date"].astype(str).to_numpy()
        previous_position = 0
        for index in range(len(part)):
            session_open = index == 0 or sessions[index - 1] != sessions[index]
            session_close = (
                index == len(part) - 1
                or sessions[index + 1] != sessions[index]
            )
            position = 0 if index == 0 else int(signals[index - 1])
            previous_end = 0 if session_open else previous_position
            open_turnover = abs(position - previous_end)
            close_turnover = abs(position) if session_close else 0
            proxy_return = (
                closes[index] / opens[index] - 1.0
                if session_close
                else opens[index + 1] / opens[index] - 1.0
            )
            turnover = float(open_turnover + close_turnover)
            gross = float(position) * proxy_return
            cost = turnover * cost_bps_per_turnover / 10_000.0
            rows.append(
                {
                    "position": position,
                    "ending_position": 0 if session_close else position,
                    "open_turnover": float(open_turnover),
                    "close_turnover": float(close_turnover),
                    "turnover": turnover,
                    "pnl_proxy_return": proxy_return,
                    "gross_strategy_return": gross,
                    "transaction_cost": cost,
                    "net_strategy_return": gross - cost,
                }
            )
            previous_position = position
    return pd.DataFrame.from_records(rows)


def test_signal_waits_until_next_open_and_nonclose_return_is_exact() -> None:
    result = apply_causal_next_open_overnight_flat(
        timing_input(), cost_bps_per_turnover=0.0
    )

    assert result.loc[0, "signal"] == 1
    assert result.loc[0, "position"] == 0
    assert result.loc[0, "gross_strategy_return"] == 0.0
    assert result.loc[1, "position"] == 1
    assert result.loc[1, "pnl_proxy_return"] == pytest.approx(
        121.0 / 110.0 - 1.0
    )
    assert result.loc[1, "gross_strategy_return"] == pytest.approx(
        121.0 / 110.0 - 1.0
    )


def test_session_close_uses_open_to_close_and_charges_liquidation() -> None:
    result = apply_causal_next_open_overnight_flat(
        timing_input(), cost_bps_per_turnover=5.0
    )

    assert bool(result.loc[5, "is_session_close"])
    assert result.loc[5, "position"] == 1
    assert result.loc[5, "pnl_proxy_return"] == pytest.approx(
        177.1561 / 161.051 - 1.0
    )
    assert result.loc[5, "close_turnover"] == 1.0
    assert result.loc[5, "transaction_cost"] == pytest.approx(0.0005)
    assert result.loc[5, "ending_position"] == 0

    assert bool(result.loc[8, "is_session_close"])
    assert result.loc[8, "position"] == 1
    assert result.loc[8, "close_turnover"] == 1.0
    assert result.loc[8, "transaction_cost"] == pytest.approx(0.0005)
    assert result.loc[8, "ending_position"] == 0


def test_entry_reversal_exit_and_terminal_turnover_are_exact() -> None:
    result = apply_causal_next_open_overnight_flat(
        timing_input(), cost_bps_per_turnover=1.0
    )

    assert result["position"].tolist() == [0, 1, -1, 0, 1, 1, -1, 1, 1]
    assert result["open_turnover"].tolist() == [
        0.0,
        1.0,
        2.0,
        1.0,
        1.0,
        0.0,
        1.0,
        2.0,
        0.0,
    ]
    assert result["close_turnover"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["turnover"].tolist() == [
        0.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
    ]


def test_session_reset_prevents_overnight_carry_but_charges_reentry() -> None:
    result = apply_causal_next_open_overnight_flat(
        timing_input(), cost_bps_per_turnover=0.0
    )

    closes = result["is_session_close"].astype(bool)
    assert result.loc[closes, "ending_position"].eq(0).all()
    assert bool(result.loc[6, "is_session_open"])
    assert result.loc[5, "ending_position"] == 0
    assert result.loc[6, "position"] == -1
    assert result.loc[6, "open_turnover"] == 1.0


def test_future_session_mutation_cannot_change_completed_prior_session() -> None:
    source = timing_input()
    mutated = source.copy(deep=True)
    mutated.loc[6:, "open"] = [400.0, 360.0, 396.0]
    mutated.loc[6:, "close"] = [402.0, 362.0, 435.6]

    before = apply_causal_next_open_overnight_flat(
        source, cost_bps_per_turnover=2.0
    )
    after = apply_causal_next_open_overnight_flat(
        mutated, cost_bps_per_turnover=2.0
    )
    columns = (
        "position",
        "ending_position",
        "open_turnover",
        "close_turnover",
        "turnover",
        "pnl_proxy_return",
        "gross_strategy_return",
        "transaction_cost",
        "net_strategy_return",
    )
    pd.testing.assert_frame_equal(
        before.loc[:5, columns],
        after.loc[:5, columns],
        check_exact=True,
    )


def test_cost_stress_changes_only_cost_and_net_return() -> None:
    source = timing_input()
    zero = apply_causal_next_open_overnight_flat(
        source, cost_bps_per_turnover=0.0
    )
    stressed = apply_causal_next_open_overnight_flat(
        source, cost_bps_per_turnover=7.5
    )

    for column in (
        "signal",
        "position",
        "ending_position",
        "open_turnover",
        "close_turnover",
        "turnover",
        "pnl_proxy_return",
        "gross_strategy_return",
    ):
        pd.testing.assert_series_equal(zero[column], stressed[column])
    assert zero["transaction_cost"].eq(0.0).all()
    assert stressed.loc[stressed["turnover"].gt(0.0), "net_strategy_return"].lt(
        stressed.loc[
            stressed["turnover"].gt(0.0), "gross_strategy_return"
        ]
    ).all()


def test_batch_output_matches_simple_sequential_oracle() -> None:
    source = timing_input().sample(frac=1.0, random_state=27).reset_index(
        drop=True
    )
    batch = apply_causal_next_open_overnight_flat(
        source, cost_bps_per_turnover=2.5
    )
    oracle = sequential_oracle(source, cost_bps_per_turnover=2.5)

    for column in oracle.columns:
        if column in {"position", "ending_position"}:
            np.testing.assert_array_equal(batch[column], oracle[column])
        else:
            np.testing.assert_allclose(
                batch[column], oracle[column], rtol=0.0, atol=1.0e-15
            )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda frame: frame.drop(columns="close"),
        lambda frame: frame.assign(open=np.inf),
        lambda frame: frame.assign(
            open=[1.0e-308, 1.0e308, *frame["open"].iloc[2:]]
        ),
    ),
)
def test_invalid_timing_inputs_fail_closed(mutation) -> None:
    with pytest.raises(CausalBarExecutionError):
        apply_causal_next_open_overnight_flat(
            mutation(timing_input()), cost_bps_per_turnover=1.0
        )
