"""Run the full analysis: load, clean, summarise, model, save figures and results.

    python -m src.analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import clean, features
from src.modeling import LABELS, PREDICTORS, fit_model

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "owid-covid-data.csv"
FIG = ROOT / "reports" / "figures"

def plot_global_timeline(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(monthly["month"], monthly["new_cases"] / 1e6, color="#1f3a5f")
    axes[0].set_ylabel("New cases (millions)")
    axes[1].plot(monthly["month"], monthly["new_deaths"] / 1e3, color="#b03a2e")
    axes[1].set_ylabel("New deaths (thousands)")
    axes[0].set_title("Monthly new COVID-19 cases and deaths, sum of reporting countries")
    fig.tight_layout()
    fig.savefig(FIG / "global_timeline.png", dpi=150)
    plt.close(fig)


def plot_age_scatter(sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for continent, grp in sample.groupby("continent"):
        ax.scatter(grp["median_age"], grp["deaths_per_million"], s=28, alpha=0.8, label=continent)
    ax.set_yscale("log")
    ax.set_xlabel("Median age")
    ax.set_ylabel("Deaths per million (log scale)")
    ax.set_title("Older countries reported more COVID-19 deaths per million")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "age_vs_deaths.png", dpi=150)
    plt.close(fig)


def plot_continent_bars(sample: pd.DataFrame) -> None:
    med = sample.groupby("continent")["deaths_per_million"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(med.index, med.values, color="#1f3a5f")
    ax.set_xlabel("Median deaths per million")
    ax.set_title("Median COVID-19 deaths per million by continent")
    fig.tight_layout()
    fig.savefig(FIG / "continent_median_deaths.png", dpi=150)
    plt.close(fig)


def plot_coefficients(model) -> None:
    coefs = model.params.drop("const")
    errs = model.bse.drop("const") * 1.96
    order = coefs.sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.errorbar(coefs[order], range(len(order)), xerr=errs[order], fmt="o", color="#1f3a5f", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[c] for c in order])
    ax.set_xlabel("Effect on log10(deaths per million), per 1 standard deviation")
    ax.set_title("Regression coefficients with 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(FIG / "regression_coefficients.png", dpi=150)
    plt.close(fig)


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = clean.load_raw(RAW)
    df = clean.clean(raw)

    summary = features.country_summary(df)
    sample = features.analysis_sample(summary)
    monthly = features.monthly_global(df)

    spearman = (
        sample[PREDICTORS + ["deaths_per_million"]]
        .corr(method="spearman")["deaths_per_million"]
        .drop("deaths_per_million")
    )
    model, n_model = fit_model(sample)

    by_continent = sample.groupby("continent")["deaths_per_million"].median().sort_values(ascending=False)

    plot_global_timeline(monthly)
    plot_age_scatter(sample)
    plot_continent_bars(sample)
    plot_coefficients(model)

    peak_cases_month = monthly.loc[monthly["new_cases"].idxmax()]
    peak_deaths_month = monthly.loc[monthly["new_deaths"].idxmax()]

    results = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "countries_total": int(summary["iso_code"].nunique()),
        "countries_in_sample": int(len(sample)),
        "countries_in_model": int(n_model),
        "data_start": str(df["date"].min().date()),
        "data_end": str(df["date"].max().date()),
        "negative_daily_values_fixed": int(
            ((raw["new_cases"] < 0) | (raw["new_deaths"] < 0)).sum()
        ),
        "spearman_with_deaths_per_million": {k: round(float(v), 3) for k, v in spearman.items()},
        "r_squared": round(float(model.rsquared), 3),
        "adj_r_squared": round(float(model.rsquared_adj), 3),
        "coefficients": {
            k: {"coef": round(float(model.params[k]), 3), "p_value": round(float(model.pvalues[k]), 4)}
            for k in PREDICTORS
        },
        "median_deaths_per_million_by_continent": {k: round(float(v), 1) for k, v in by_continent.items()},
        "peak_cases_month": str(peak_cases_month["month"].date()),
        "peak_cases_month_value": float(peak_cases_month["new_cases"]),
        "peak_deaths_month": str(peak_deaths_month["month"].date()),
        "peak_deaths_month_value": float(peak_deaths_month["new_deaths"]),
    }
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
