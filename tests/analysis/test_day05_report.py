"""Tests for the consolidated Day 05 report."""

from __future__ import annotations

import pandas as pd

from systematic_alpha.analysis.day05_report import (
    build_findings_markdown,
    markdown_table,
)


def make_tables() -> dict[str, pd.DataFrame]:
    """Create minimal valid report tables."""

    symbols = ["IWM", "QQQ", "SPY"]

    moments = pd.DataFrame(
        {
            "symbol": symbols,
            "return_type": (
                ["close_to_close_log_return"] * 3
            ),
            "observations": [100, 100, 100],
            "mean": [0.001, 0.002, 0.0015],
            "standard_deviation": [
                0.02,
                0.018,
                0.015,
            ],
            "skewness": [-0.5, -0.4, -0.6],
            "excess_kurtosis": [
                7.0,
                6.0,
                10.0,
            ],
            "jarque_bera_pvalue": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    tail_rates = pd.DataFrame(
        {
            "symbol": symbols,
            "return_type": (
                ["close_to_close_log_return"] * 3
            ),
            "threshold_sigma": [3.0] * 3,
            "two_sided_count": [2, 3, 4],
            "empirical_two_sided_rate": [
                0.02,
                0.03,
                0.04,
            ],
            "normal_two_sided_rate": [
                0.0027
            ] * 3,
            "empirical_to_normal_ratio": [
                7.4,
                11.1,
                14.8,
            ],
        }
    )

    intraday_acf = pd.DataFrame(
        {
            "symbol": ["SPY"] * 3,
            "transformation": [
                "raw",
                "absolute",
                "squared",
            ],
            "lag": [1, 1, 1],
            "pair_count": [100, 100, 100],
            "autocorrelation": [
                0.01,
                0.30,
                0.25,
            ],
        }
    )

    daily_ljung_box = pd.DataFrame(
        {
            "symbol": ["SPY"] * 3,
            "transformation": [
                "raw",
                "absolute",
                "squared",
            ],
            "lag": [20, 20, 20],
            "ljung_box_statistic": [
                20.0,
                100.0,
                90.0,
            ],
            "ljung_box_pvalue": [
                0.01,
                0.0,
                0.0,
            ],
            "reject_at_5pct": [
                True,
                True,
                True,
            ],
        }
    )

    volatility_summary = pd.DataFrame(
        {
            "symbol": symbols,
            "sessions": [100] * 3,
            "median_annualized_total_rv": [
                0.20,
                0.18,
                0.15,
            ],
        }
    )

    dependence = pd.DataFrame(
        {
            "symbol_a": [
                "IWM",
                "IWM",
                "QQQ",
            ],
            "symbol_b": [
                "QQQ",
                "SPY",
                "SPY",
            ],
            "observations": [100] * 3,
            "pearson_correlation": [
                0.70,
                0.75,
                0.90,
            ],
            "spearman_correlation": [
                0.68,
                0.72,
                0.88,
            ],
            "kendall_tau": [
                0.50,
                0.55,
                0.70,
            ],
        }
    )

    tail_dependence = pd.DataFrame(
        {
            "symbol_a": [
                "IWM",
                "IWM",
                "QQQ",
            ],
            "symbol_b": [
                "QQQ",
                "SPY",
                "SPY",
            ],
            "tail_probability": [0.05] * 3,
            "lower_joint_rate": [
                0.03,
                0.03,
                0.04,
            ],
            "upper_joint_rate": [
                0.02,
                0.02,
                0.03,
            ],
            "lower_coexceedance_ratio": [
                0.6,
                0.6,
                0.8,
            ],
            "upper_coexceedance_ratio": [
                0.4,
                0.4,
                0.6,
            ],
        }
    )

    event_decision = pd.DataFrame(
        {
            "sampling_method": [
                "15-minute time bars",
                "dollar bars",
            ],
            "project_role": [
                "primary",
                "future candidate",
            ],
            "decision": [
                "retain",
                "defer",
            ],
            "reason": [
                "complete history",
                "insufficient trade sample",
            ],
        }
    )

    return {
        "moments": moments,
        "tail_rates": tail_rates,
        "intraday_acf": intraday_acf,
        "daily_ljung_box": (
            daily_ljung_box
        ),
        "volatility_summary": (
            volatility_summary
        ),
        "pairwise_dependence": dependence,
        "tail_dependence": tail_dependence,
        "event_decision": event_decision,
    }


def test_markdown_table_has_headers() -> None:
    table = markdown_table(
        pd.DataFrame(
            {
                "name": ["SPY"],
                "value": [0.5],
            }
        )
    )

    assert "| name | value |" in table
    assert "| SPY | 0.5 |" in table


def test_report_contains_required_sections() -> None:
    report = build_findings_markdown(
        make_tables(),
        ["example.png"],
    )

    assert "# Day 05" in report
    assert "Daily return distributions" in report
    assert "Realized volatility" in report
    assert "Cross-asset dependence" in report
    assert "Time bars versus event bars" in report


def test_report_preserves_sampling_decision() -> None:
    report = build_findings_markdown(
        make_tables(),
        ["example.png"],
    )

    assert "15-minute time bars" in report
    assert "Dollar bars" in report
    assert "engineering smoke test" in report


def test_report_states_locked_test_not_accessed() -> None:
    report = build_findings_markdown(
        make_tables(),
        ["example.png"],
    )

    assert "**Locked final test:** Not accessed" in report
    assert "does not establish trading profitability" in report
