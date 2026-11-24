"""Build the country-level table used for the analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_COLS = [
    "continent", "location", "population", "median_age", "aged_65_older",
    "gdp_per_capita", "hospital_beds_per_thousand", "diabetes_prevalence",
    "life_expectancy", "human_development_index", "population_density",
]


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country with totals, rates and background indicators.

    Expects cleaned data (see ``clean.clean``). Totals use the final
    cumulative value, rates are recomputed from those totals and population.
    """
    ordered = df.sort_values("date")
    grouped = ordered.groupby("iso_code")

    summary = grouped[STATIC_COLS].last()  # last non-null value per column
    summary["total_cases"] = grouped["total_cases"].max()
    summary["total_deaths"] = grouped["total_deaths"].max()
    summary["peak_fully_vaccinated_pct"] = grouped["people_fully_vaccinated_per_hundred"].max()
    summary["mean_stringency"] = grouped["stringency_index"].mean()

    pop = summary["population"].where(summary["population"] > 0)
    summary["cases_per_million"] = summary["total_cases"] / pop * 1e6
    summary["deaths_per_million"] = summary["total_deaths"] / pop * 1e6
    summary["cfr_pct"] = (
        summary["total_deaths"] / summary["total_cases"].where(summary["total_cases"] > 0) * 100
    )
    summary["log_gdp_per_capita"] = np.log10(summary["gdp_per_capita"].where(summary["gdp_per_capita"] > 0))
    summary["log_deaths_per_million"] = np.log10(
        summary["deaths_per_million"].where(summary["deaths_per_million"] > 0)
    )
    return summary.reset_index()


def monthly_global(df: pd.DataFrame) -> pd.DataFrame:
    """Sum new cases and deaths across countries for each calendar month."""
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")[["new_cases", "new_deaths"]]
        .sum(min_count=1)
        .reset_index()
    )
    return monthly


def analysis_sample(summary: pd.DataFrame, min_population: float = 1_000_000) -> pd.DataFrame:
    """Countries with enough population and reported deaths to compare fairly."""
    keep = (
        (summary["population"] >= min_population)
        & (summary["total_cases"] > 0)
        & (summary["deaths_per_million"] > 0)
    )
    return summary.loc[keep].copy()
