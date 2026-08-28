"""Build the consolidated Day 05 analytical report and figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class Day05ReportError(ValueError):
    """Raised when Day 05 report artifacts are incomplete."""


@dataclass(frozen=True)
class Day05ArtifactPaths:
    """Canonical paths to completed Day 05 evidence."""

    root: Path

    @property
    def sources(self) -> dict[str, Path]:
        """Return the canonical Day 05 artifact map."""

        return {
            "session_returns": (
                self.root
                / "eda_v1"
                / "data"
                / "session_return_features.parquet"
            ),
            "moments": (
                self.root
                / "stylized_facts_v1"
                / "tables"
                / "return_moments.csv"
            ),
            "tail_rates": (
                self.root
                / "stylized_facts_v1"
                / "tables"
                / "tail_rates.csv"
            ),
            "intraday_acf": (
                self.root
                / "stylized_facts_v1"
                / "tables"
                / "intraday_acf.csv"
            ),
            "daily_ljung_box": (
                self.root
                / "stylized_facts_v1"
                / "tables"
                / "daily_ljung_box.csv"
            ),
            "daily_volatility": (
                self.root
                / "volatility_seasonality_v1"
                / "data"
                / "daily_volatility.parquet"
            ),
            "volatility_summary": (
                self.root
                / "volatility_seasonality_v1"
                / "tables"
                / "volatility_summary.csv"
            ),
            "seasonality": (
                self.root
                / "volatility_seasonality_v1"
                / "tables"
                / "intraday_seasonality.csv"
            ),
            "pairwise_dependence": (
                self.root
                / "dependence_v1"
                / "tables"
                / "pairwise_dependence.csv"
            ),
            "tail_dependence": (
                self.root
                / "dependence_v1"
                / "tables"
                / "tail_dependence.csv"
            ),
            "rolling_dependence": (
                self.root
                / "dependence_v1"
                / "tables"
                / "rolling_dependence.csv"
            ),
            "regime_dependence": (
                self.root
                / "dependence_v1"
                / "tables"
                / "regime_dependence.csv"
            ),
            "event_comparison": (
                self.root
                / "event_bar_diagnostics_v1"
                / "tables"
                / "sampling_comparison.csv"
            ),
            "event_conservation": (
                self.root
                / "event_bar_diagnostics_v1"
                / "tables"
                / "conservation.csv"
            ),
            "event_decision": (
                self.root
                / "event_bar_diagnostics_v1"
                / "tables"
                / "sampling_decision.csv"
            ),
        }


def load_day05_artifacts(
    paths: Day05ArtifactPaths,
) -> dict[str, pd.DataFrame]:
    """Load completed Day 05 outputs without recalculation."""

    missing = [
        path
        for path in paths.sources.values()
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"- {path}"
            for path in missing
        )

        raise Day05ReportError(
            "Day 05 report inputs are missing:\n"
            f"{formatted}"
        )

    tables: dict[str, pd.DataFrame] = {}

    for name, path in paths.sources.items():
        if path.suffix == ".parquet":
            tables[name] = pd.read_parquet(
                path
            )
        else:
            tables[name] = pd.read_csv(
                path
            )

    return tables


def _format_cell(value: object) -> str:
    """Format one Markdown-table cell."""

    if pd.isna(value):
        return ""

    if isinstance(value, float):
        return f"{value:.6g}"

    return (
        str(value)
        .replace("|", r"\|")
        .replace("\n", " ")
    )


def markdown_table(
    frame: pd.DataFrame,
) -> str:
    """Render a DataFrame without requiring tabulate."""

    if frame.empty:
        return "_No observations available._"

    formatted = frame.map(
        _format_cell
    )

    headers = [
        str(column)
        for column in formatted.columns
    ]

    header_line = (
        "| "
        + " | ".join(headers)
        + " |"
    )

    separator_line = (
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |"
    )

    rows = [
        "| "
        + " | ".join(
            str(value)
            for value in row
        )
        + " |"
        for row in formatted.itertuples(
            index=False,
            name=None,
        )
    ]

    return "\n".join(
        [
            header_line,
            separator_line,
            *rows,
        ]
    )


def _save_figure(
    figure: plt.Figure,
    path: Path,
) -> None:
    """Save and close one figure."""

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_daily_return_figure(
    sessions: pd.DataFrame,
    path: Path,
) -> None:
    """Plot daily close-to-close return distributions."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for symbol, group in sessions.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        values = pd.to_numeric(
            group[
                "close_to_close_log_return"
            ],
            errors="coerce",
        ).dropna()

        axis.hist(
            values,
            bins=100,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=str(symbol),
        )

    axis.set_title(
        "Daily Close-to-Close Log-Return Distributions"
    )

    axis.set_xlabel("Log return")
    axis.set_ylabel("Density")
    axis.legend()

    _save_figure(figure, path)


def create_rolling_volatility_figure(
    daily_volatility: pd.DataFrame,
    path: Path,
) -> None:
    """Plot the 21-day rolling realized-volatility series."""

    frame = daily_volatility.copy()

    frame["session_date"] = pd.to_datetime(
        frame["session_date"],
        errors="raise",
    )

    column = (
        "rolling_total_realized_"
        "volatility_21d"
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for symbol, group in frame.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        group = group.sort_values(
            "session_date"
        )

        axis.plot(
            group["session_date"],
            group[column],
            label=str(symbol),
        )

    axis.set_title(
        "21-Day Rolling Annualized Realized Volatility"
    )

    axis.set_xlabel("Session")
    axis.set_ylabel(
        "Annualized volatility"
    )

    axis.legend()

    _save_figure(figure, path)


def create_intraday_seasonality_figure(
    seasonality: pd.DataFrame,
    path: Path,
) -> None:
    """Plot intraday variance-share seasonality."""

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for symbol, group in seasonality.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        group = group.sort_values(
            "bar_number"
        )

        axis.plot(
            group["bar_number"],
            group[
                "mean_squared_return_share"
            ],
            marker="o",
            markersize=3,
            label=str(symbol),
        )

    axis.set_title(
        "Intraday Share of Regular-Session Variance"
    )

    axis.set_xlabel(
        "15-minute bar number"
    )

    axis.set_ylabel(
        "Mean share of session squared returns"
    )

    axis.legend()

    _save_figure(figure, path)


def create_rolling_dependence_figure(
    rolling: pd.DataFrame,
    path: Path,
) -> None:
    """Plot 63-session rolling Pearson correlations."""

    frame = rolling.loc[
        rolling["window"].eq(63)
    ].copy()

    frame["session_date"] = pd.to_datetime(
        frame["session_date"],
        errors="raise",
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for (
        symbol_a,
        symbol_b,
    ), group in frame.groupby(
        ["symbol_a", "symbol_b"],
        observed=True,
        sort=True,
    ):
        group = group.sort_values(
            "session_date"
        )

        axis.plot(
            group["session_date"],
            group[
                "pearson_correlation"
            ],
            label=(
                f"{symbol_a}/{symbol_b}"
            ),
        )

    axis.set_title(
        "63-Session Rolling Return Correlation"
    )

    axis.set_xlabel("Session")
    axis.set_ylabel(
        "Pearson correlation"
    )

    axis.set_ylim(-1.0, 1.0)
    axis.legend()

    _save_figure(figure, path)


def create_tail_dependence_figure(
    tails: pd.DataFrame,
    path: Path,
) -> None:
    """Plot five-percent joint-tail co-exceedance ratios."""

    frame = tails.loc[
        tails[
            "tail_probability"
        ].eq(0.05)
    ].copy()

    frame["pair"] = (
        frame["symbol_a"]
        + "/"
        + frame["symbol_b"]
    )

    plot_frame = frame.set_index(
        "pair"
    )[
        [
            "lower_coexceedance_ratio",
            "upper_coexceedance_ratio",
        ]
    ]

    axis = plot_frame.plot.bar(
        figsize=(9, 6)
    )

    axis.set_title(
        "Five-Percent Joint-Tail Co-Exceedance"
    )

    axis.set_xlabel("ETF pair")
    axis.set_ylabel(
        "Conditional co-exceedance ratio"
    )

    axis.legend(
        [
            "Lower tail",
            "Upper tail",
        ]
    )

    _save_figure(
        axis.figure,
        path,
    )


def create_figures(
    tables: Mapping[str, pd.DataFrame],
    figure_dir: Path,
) -> list[str]:
    """Create all Day 05 report figures."""

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures = {
        "daily_return_distributions.png": (
            create_daily_return_figure,
            tables["session_returns"],
        ),
        "rolling_realized_volatility.png": (
            create_rolling_volatility_figure,
            tables["daily_volatility"],
        ),
        "intraday_variance_seasonality.png": (
            create_intraday_seasonality_figure,
            tables["seasonality"],
        ),
        "rolling_dependence_63d.png": (
            create_rolling_dependence_figure,
            tables["rolling_dependence"],
        ),
        "tail_coexceedance_5pct.png": (
            create_tail_dependence_figure,
            tables["tail_dependence"],
        ),
    }

    for filename, (
        builder,
        frame,
    ) in figures.items():
        builder(
            frame,
            figure_dir / filename,
        )

    return list(figures)


def build_findings_markdown(
    tables: Mapping[str, pd.DataFrame],
    figure_names: list[str],
) -> str:
    """Build the consolidated Day 05 findings report."""

    daily_moments = tables[
        "moments"
    ].loc[
        tables["moments"][
            "return_type"
        ].eq(
            "close_to_close_log_return"
        ),
        [
            "symbol",
            "observations",
            "mean",
            "standard_deviation",
            "skewness",
            "excess_kurtosis",
            "jarque_bera_pvalue",
        ],
    ].sort_values("symbol")

    three_sigma = tables[
        "tail_rates"
    ].loc[
        (
            tables["tail_rates"][
                "return_type"
            ].eq(
                "close_to_close_log_return"
            )
        )
        & (
            tables["tail_rates"][
                "threshold_sigma"
            ].eq(3.0)
        ),
        [
            "symbol",
            "two_sided_count",
            "empirical_two_sided_rate",
            "normal_two_sided_rate",
            "empirical_to_normal_ratio",
        ],
    ].sort_values("symbol")

    intraday_lag_one = tables[
        "intraday_acf"
    ].loc[
        tables["intraday_acf"][
            "lag"
        ].eq(1),
        [
            "symbol",
            "transformation",
            "pair_count",
            "autocorrelation",
        ],
    ].sort_values(
        [
            "symbol",
            "transformation",
        ]
    )

    daily_ljung_box = tables[
        "daily_ljung_box"
    ].loc[
        tables["daily_ljung_box"][
            "lag"
        ].eq(20),
        [
            "symbol",
            "transformation",
            "ljung_box_statistic",
            "ljung_box_pvalue",
            "reject_at_5pct",
        ],
    ].sort_values(
        [
            "symbol",
            "transformation",
        ]
    )

    volatility = tables[
        "volatility_summary"
    ].sort_values("symbol")

    dependence = tables[
        "pairwise_dependence"
    ][
        [
            "symbol_a",
            "symbol_b",
            "observations",
            "pearson_correlation",
            "spearman_correlation",
            "kendall_tau",
        ]
    ].sort_values(
        ["symbol_a", "symbol_b"]
    )

    five_percent_tails = tables[
        "tail_dependence"
    ].loc[
        tables["tail_dependence"][
            "tail_probability"
        ].eq(0.05),
        [
            "symbol_a",
            "symbol_b",
            "lower_joint_rate",
            "upper_joint_rate",
            "lower_coexceedance_ratio",
            "upper_coexceedance_ratio",
        ],
    ].sort_values(
        ["symbol_a", "symbol_b"]
    )

    event_decision = tables[
        "event_decision"
    ]

    most_volatile = daily_moments.sort_values(
        "standard_deviation",
        ascending=False,
    ).iloc[0]

    highest_kurtosis = daily_moments.sort_values(
        "excess_kurtosis",
        ascending=False,
    ).iloc[0]

    highest_tail_multiple = (
        three_sigma.sort_values(
            "empirical_to_normal_ratio",
            ascending=False,
        ).iloc[0]
    )

    strongest_dependence = (
        dependence.sort_values(
            "pearson_correlation",
            ascending=False,
        ).iloc[0]
    )

    figure_links = "\n".join(
        f"- [{name}](figures/{name})"
        for name in figure_names
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    return f"""# Day 05 — Data, Returns and Stylized-Fact Findings

- **Generated:** {generated_at}
- **Development period:** 2020-01-02 through 2025-12-31
- **Primary universe:** SPY, QQQ and IWM
- **Primary frequency:** 15-minute regular-session SIP bars
- **Locked final test:** Not accessed

## 1. Executive findings

The Day 05 evidence rejects a simple independent Gaussian-return model.

- **{most_volatile["symbol"]}** had the highest daily standard deviation at approximately **{most_volatile["standard_deviation"]:.4%}**.
- All three ETFs displayed negative daily skewness.
- **{highest_kurtosis["symbol"]}** had the largest daily excess kurtosis at approximately **{highest_kurtosis["excess_kurtosis"]:.2f}**.
- The largest observed three-standard-deviation frequency multiple was **{highest_tail_multiple["empirical_to_normal_ratio"]:.2f}×** the Gaussian benchmark for **{highest_tail_multiple["symbol"]}**.
- Raw 15-minute return autocorrelation was economically small, while absolute and squared-return autocorrelation was substantially larger.
- Daily Ljung–Box tests rejected serial independence, especially for absolute and squared returns.
- Return dependence varied through time and increased during stressed volatility conditions.
- Event-bar conservation was verified, but the available one-minute trade sample was not sufficient for statistical comparison.

These results support volatility-aware risk management and regime analysis. They do not by themselves demonstrate a profitable trading signal.

## 2. Return definitions

Simple return:

\\[
R_t = \\frac{{P_t}}{{P_{{t-1}}}}-1.
\\]

Log return:

\\[
r_t = \\log P_t-\\log P_{{t-1}}.
\\]

Intraday bar returns exclude overnight transitions. Overnight returns were calculated separately as:

\\[
r_d^{{ON}}
=
\\log O_d-\\log C_{{d-1}}.
\\]

The daily decomposition was numerically reconciled:

\\[
r_d^{{CC}}
=
r_d^{{ON}}
+
r_d^{{RS}}.
\\]

## 3. Daily return distributions

{markdown_table(daily_moments)}

Jarque–Bera p-values were effectively zero for the development sample. Because the sample is large, the economic interpretation relies on skewness, excess kurtosis, empirical quantiles and tail frequencies rather than the p-value alone.

![Daily return distributions](figures/daily_return_distributions.png)

## 4. Empirical tail behaviour

{markdown_table(three_sigma)}

The Gaussian two-sided probability beyond three standard deviations is approximately 0.27%. The observed frequencies were several times larger.

This finding discourages reliance on an unadjusted Gaussian VaR model and supports later Historical Simulation, Filtered Historical Simulation and conditional-volatility comparisons.

## 5. Serial dependence and volatility clustering

### Intraday lag-one autocorrelation

{markdown_table(intraday_lag_one)}

Raw return dependence was small. The larger dependence in absolute and squared returns provides direct evidence of volatility clustering.

### Daily Ljung–Box evidence at lag 20

{markdown_table(daily_ljung_box)}

Rejection in raw returns must not automatically be interpreted as exploitable directional predictability. Crisis observations, volatility clustering and regime shifts can also affect these tests.

## 6. Realized volatility

\\[
RV_d
=
\\sum_{{k=1}}^{{N_d}}
r_{{d,k}}^2.
\\]

Annualized daily realized volatility was calculated as:

\\[
\\sigma_{{d,ann}}
=
\\sqrt{{252RV_d}}.
\\]

{markdown_table(volatility)}

![Rolling realized volatility](figures/rolling_realized_volatility.png)

The results justify later EWMA, GARCH and asymmetric GJR-GARCH comparisons.

## 7. Intraday seasonality

![Intraday variance seasonality](figures/intraday_variance_seasonality.png)

Variance, volume and trade activity were not uniform across the session. This affects:

- volatility scaling;
- execution-cost assumptions;
- signal timing;
- stop and threshold calibration;
- interpretation of opening and closing bars.

The project will not assume that all 15-minute intervals have identical risk.

## 8. Cross-asset dependence

{markdown_table(dependence)}

The strongest full-sample Pearson dependence was observed for **{strongest_dependence["symbol_a"]}/{strongest_dependence["symbol_b"]}** at approximately **{strongest_dependence["pearson_correlation"]:.3f}**.

Pearson, Spearman and Kendall measures were reported separately because linear correlation does not fully describe dependence.

![Rolling dependence](figures/rolling_dependence_63d.png)

Dependence was time-varying. Static full-sample correlation must therefore be treated as a summary rather than an invariant parameter.

## 9. Finite-threshold joint-tail dependence

{markdown_table(five_percent_tails)}

These are finite-threshold co-exceedance measures and are not presented as asymptotic copula tail-dependence coefficients.

![Tail co-exceedance](figures/tail_coexceedance_5pct.png)

Correlation and joint-tail dependence may help generate pair candidates, but neither establishes cointegration or profitable mean reversion.

## 10. Time bars versus event bars

{markdown_table(event_decision)}

The event-bar engine preserved:

- every trade;
- every share;
- total dollar notional;
- whole-trade threshold crossings;
- the final residual bar.

The current trade sample contained only one minute of SPY IEX observations. It was therefore classified as an engineering smoke test, not statistical evidence.

The controlled decision is:

- retain 15-minute SIP time bars as the primary project frequency;
- use 30-minute and 60-minute time bars for robustness;
- retain dollar bars as the preferred future event-bar candidate;
- defer event-bar acceptance until representative multi-session trade data is available.

## 11. Implications for Days 06–10

1. Trend models should be tested on 15-minute bars first.
2. Parameters should not be justified by Gaussian-return assumptions.
3. Volatility-scaled thresholds and regime diagnostics are empirically justified.
4. Candidate pairs require economic justification and cointegration testing; correlation alone is insufficient.
5. Opening and closing periods may require separate execution and risk treatment.
6. Dollar bars remain optional and must not delay the CQF-core strategy work.
7. All estimation and model selection remain confined to the development sample.

## 12. Limitations

- Day 05 is descriptive and does not establish trading profitability.
- Formal tests can be dominated by large samples and crisis observations.
- Static dependence summaries may conceal regime changes.
- The event-bar sample is too short for statistical inference.
- Bid–ask spread and Level-1 quote effects are analysed later.
- The locked 2026 final-test period was not accessed.

## 13. Generated figures

{figure_links}
"""


def write_day05_report(
    *,
    artifact_root: Path,
    output_dir: Path,
) -> Path:
    """Load evidence, create figures and write the report."""

    paths = Day05ArtifactPaths(
        root=artifact_root
    )

    tables = load_day05_artifacts(
        paths
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_names = create_figures(
        tables,
        output_dir / "figures",
    )

    report = build_findings_markdown(
        tables,
        figure_names,
    )

    report_path = (
        output_dir
        / "DAY05_FINDINGS.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    metadata = {
        "analysis": (
            "day05_consolidated_report_v1"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifact_root": str(
            artifact_root
        ),
        "report_path": str(
            report_path
        ),
        "figure_files": figure_names,
        "source_files": {
            name: str(path)
            for name, path in (
                paths.sources.items()
            )
        },
        "locked_test_accessed": False,
        "calculation_policy": (
            "Presentation layer only. "
            "Existing Day 05 tables are the "
            "source of truth."
        ),
    }

    (
        output_dir
        / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the consolidated Day 05 "
            "findings report."
        )
    )

    parser.add_argument(
        "--artifact-root",
        default="artifacts/day05",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/day05/"
            "day05_report_v1"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Build the final Day 05 report."""

    args = parse_args()

    report_path = write_day05_report(
        artifact_root=Path(
            args.artifact_root
        ),
        output_dir=Path(
            args.output_dir
        ),
    )

    print(
        "===== D05-T04 CONSOLIDATED REPORT ====="
    )

    print("Report:", report_path.resolve())

    print("\nD05-T04 PASSED")


if __name__ == "__main__":
    main()
