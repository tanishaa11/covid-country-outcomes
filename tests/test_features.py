import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features  # noqa: E402


def make_clean_frame():
    rows = []
    for iso, continent, pop, deaths, cases in [
        ("AAA", "Europe", 2_000_000, 400, 20_000),
        ("BBB", "Asia", 5_000_000, 50, 10_000),
        ("CCC", "Africa", 500_000, 10, 2_000),   # below population cut-off
        ("DDD", "Asia", 3_000_000, 0, 0),         # no cases at all
    ]:
        for day, frac in [("2020-01-01", 0.5), ("2020-01-02", 1.0)]:
            rows.append({
                "iso_code": iso, "continent": continent, "location": iso,
                "date": pd.Timestamp(day), "population": pop,
                "median_age": 40.0, "aged_65_older": 15.0,
                "gdp_per_capita": 20000.0, "hospital_beds_per_thousand": 3.0,
                "diabetes_prevalence": 7.0, "life_expectancy": 78.0,
                "human_development_index": 0.8, "population_density": 100.0,
                "total_cases": cases * frac, "new_cases": 1.0,
                "total_deaths": deaths * frac, "new_deaths": 1.0,
                "people_fully_vaccinated_per_hundred": 50.0 * frac,
                "stringency_index": 60.0,
            })
    return pd.DataFrame(rows)


class TestCountrySummary(unittest.TestCase):
    def setUp(self):
        self.summary = features.country_summary(make_clean_frame()).set_index("iso_code")

    def test_one_row_per_country(self):
        self.assertEqual(len(self.summary), 4)

    def test_deaths_per_million_matches_hand_calculation(self):
        # 400 deaths in 2,000,000 people = 200 per million
        self.assertAlmostEqual(self.summary.loc["AAA", "deaths_per_million"], 200.0)

    def test_uses_final_total_not_first_day(self):
        self.assertEqual(self.summary.loc["BBB", "total_deaths"], 50)

    def test_cfr_is_nan_when_no_cases(self):
        self.assertTrue(np.isnan(self.summary.loc["DDD", "cfr_pct"]))

    def test_cfr_value(self):
        self.assertAlmostEqual(self.summary.loc["AAA", "cfr_pct"], 2.0)

    def test_peak_vaccination_is_maximum(self):
        self.assertEqual(self.summary.loc["AAA", "peak_fully_vaccinated_pct"], 50.0)

    def test_log_columns_are_log10(self):
        self.assertAlmostEqual(self.summary.loc["AAA", "log_gdp_per_capita"], np.log10(20000.0))

    def test_zero_deaths_gives_nan_log(self):
        self.assertTrue(np.isnan(self.summary.loc["DDD", "log_deaths_per_million"]))


class TestAnalysisSample(unittest.TestCase):
    def test_filters_small_countries_and_zero_death_countries(self):
        summary = features.country_summary(make_clean_frame())
        sample = features.analysis_sample(summary)
        self.assertEqual(set(sample["iso_code"]), {"AAA", "BBB"})

    def test_custom_population_cutoff(self):
        summary = features.country_summary(make_clean_frame())
        sample = features.analysis_sample(summary, min_population=100_000)
        self.assertIn("CCC", set(sample["iso_code"]))


class TestMonthlyGlobal(unittest.TestCase):
    def test_sums_across_countries_per_month(self):
        df = make_clean_frame()
        monthly = features.monthly_global(df)
        self.assertEqual(len(monthly), 1)
        # 4 countries * 2 days * 1 new case per day
        self.assertEqual(monthly.loc[0, "new_cases"], 8.0)


if __name__ == "__main__":
    unittest.main()
