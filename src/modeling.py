"""Regression model used to compare country-level factors."""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

PREDICTORS = [
    "median_age", "log_gdp_per_capita", "hospital_beds_per_thousand",
    "diabetes_prevalence", "peak_fully_vaccinated_pct", "mean_stringency",
]
LABELS = {
    "median_age": "Median age",
    "log_gdp_per_capita": "GDP per capita (log)",
    "hospital_beds_per_thousand": "Hospital beds per 1,000",
    "diabetes_prevalence": "Diabetes prevalence",
    "peak_fully_vaccinated_pct": "Peak fully vaccinated %",
    "mean_stringency": "Mean stringency index",
}


def standardise(data: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column so coefficients are per 1 standard deviation."""
    return (data - data.mean()) / data.std()


def fit_model(sample: pd.DataFrame):
    """OLS of log10(deaths per million) on standardised predictors.

    Rows with any missing predictor are dropped; the number of rows actually
    used is returned alongside the fitted model.
    """
    data = sample[PREDICTORS + ["log_deaths_per_million"]].dropna()
    x = standardise(data[PREDICTORS])
    model = sm.OLS(data["log_deaths_per_million"], sm.add_constant(x)).fit()
    return model, len(data)
