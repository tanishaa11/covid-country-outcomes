"""Cleaning helpers for the Our World in Data (OWID) COVID-19 dataset.

Each function takes a DataFrame and returns a new one, so they are easy to test
and can be chained in any order through :func:`clean`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USE_COLS = [
    "iso_code", "continent", "location", "date",
    "total_cases", "new_cases", "total_deaths", "new_deaths",
    "people_fully_vaccinated_per_hundred", "stringency_index",
    "population", "population_density", "median_age", "aged_65_older",
    "gdp_per_capita", "diabetes_prevalence", "hospital_beds_per_thousand",
    "life_expectancy", "human_development_index",
]


def load_raw(path) -> pd.DataFrame:
    """Read only the columns this project uses, with dates parsed."""
    return pd.read_csv(path, usecols=USE_COLS, parse_dates=["date"])


def drop_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are not countries (World, continents, income groups).

    OWID leaves ``continent`` empty for aggregate rows. Some real countries
    (for example Kosovo) carry an ``OWID_`` iso code, so the continent check is
    safer than filtering on the code prefix.
    """
    return df.loc[df["continent"].notna()].copy()


def negative_to_nan(df: pd.DataFrame, cols=("new_cases", "new_deaths")) -> pd.DataFrame:
    """Turn negative daily counts into NaN.

    Negative values come from countries revising earlier totals downward; they
    are data corrections, not real negative cases.
    """
    out = df.copy()
    for col in cols:
        out.loc[out[col] < 0, col] = np.nan
    return out


def enforce_cumulative_monotonic(
    df: pd.DataFrame, cols=("total_cases", "total_deaths")
) -> pd.DataFrame:
    """Make cumulative totals non-decreasing within each country."""
    out = df.sort_values(["iso_code", "date"]).copy()
    for col in cols:
        out[col] = out.groupby("iso_code")[col].cummax()
    return out


def drop_duplicate_days(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per country per day (the last one if repeated)."""
    return df.drop_duplicates(subset=["iso_code", "date"], keep="last").copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    out = drop_aggregates(df)
    out = drop_duplicate_days(out)
    out = negative_to_nan(out)
    out = enforce_cumulative_monotonic(out)
    return out.reset_index(drop=True)
