import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clean  # noqa: E402


def make_frame(rows):
    cols = ["iso_code", "continent", "date", "total_cases", "new_cases", "total_deaths", "new_deaths"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestDropAggregates(unittest.TestCase):
    def test_removes_world_and_continents(self):
        df = make_frame([
            ["OWID_WRL", None, "2020-01-01", 10, 1, 0, 0],
            ["IND", "Asia", "2020-01-01", 5, 1, 0, 0],
        ])
        out = clean.drop_aggregates(df)
        self.assertEqual(list(out["iso_code"]), ["IND"])

    def test_keeps_kosovo_even_with_owid_prefix(self):
        df = make_frame([["OWID_KOS", "Europe", "2020-01-01", 5, 1, 0, 0]])
        self.assertEqual(len(clean.drop_aggregates(df)), 1)

    def test_empty_input_returns_empty(self):
        df = make_frame([])
        self.assertTrue(clean.drop_aggregates(df).empty)


class TestNegativeToNan(unittest.TestCase):
    def test_negative_becomes_nan_and_positive_kept(self):
        df = make_frame([
            ["IND", "Asia", "2020-01-01", 5, -3, 0, -1],
            ["IND", "Asia", "2020-01-02", 8, 3, 1, 1],
        ])
        out = clean.negative_to_nan(df)
        self.assertTrue(np.isnan(out.loc[0, "new_cases"]))
        self.assertTrue(np.isnan(out.loc[0, "new_deaths"]))
        self.assertEqual(out.loc[1, "new_cases"], 3)

    def test_zero_is_not_treated_as_negative(self):
        df = make_frame([["IND", "Asia", "2020-01-01", 5, 0, 0, 0]])
        self.assertEqual(clean.negative_to_nan(df).loc[0, "new_cases"], 0)

    def test_does_not_modify_input(self):
        df = make_frame([["IND", "Asia", "2020-01-01", 5, -3, 0, 0]])
        clean.negative_to_nan(df)
        self.assertEqual(df.loc[0, "new_cases"], -3)


class TestMonotonic(unittest.TestCase):
    def test_downward_revision_is_flattened(self):
        df = make_frame([
            ["IND", "Asia", "2020-01-01", 100, 0, 5, 0],
            ["IND", "Asia", "2020-01-02", 90, 0, 4, 0],
            ["IND", "Asia", "2020-01-03", 120, 0, 6, 0],
        ])
        out = clean.enforce_cumulative_monotonic(df)
        self.assertEqual(list(out["total_cases"]), [100, 100, 120])
        self.assertEqual(list(out["total_deaths"]), [5, 5, 6])

    def test_countries_do_not_affect_each_other(self):
        df = make_frame([
            ["IND", "Asia", "2020-01-01", 1000, 0, 0, 0],
            ["NPL", "Asia", "2020-01-01", 10, 0, 0, 0],
            ["NPL", "Asia", "2020-01-02", 12, 0, 0, 0],
        ])
        out = clean.enforce_cumulative_monotonic(df)
        nepal = out[out["iso_code"] == "NPL"]["total_cases"].tolist()
        self.assertEqual(nepal, [10, 12])

    def test_unsorted_input_is_sorted_by_date(self):
        df = make_frame([
            ["IND", "Asia", "2020-01-03", 120, 0, 0, 0],
            ["IND", "Asia", "2020-01-01", 100, 0, 0, 0],
        ])
        out = clean.enforce_cumulative_monotonic(df)
        self.assertEqual(list(out["total_cases"]), [100, 120])


class TestDuplicates(unittest.TestCase):
    def test_repeated_day_keeps_last_row(self):
        df = make_frame([
            ["IND", "Asia", "2020-01-01", 5, 1, 0, 0],
            ["IND", "Asia", "2020-01-01", 7, 2, 0, 0],
        ])
        out = clean.drop_duplicate_days(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["total_cases"], 7)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = make_frame([
            ["OWID_WRL", None, "2020-01-01", 999, 9, 9, 9],
            ["IND", "Asia", "2020-01-01", 100, 5, 1, 0],
            ["IND", "Asia", "2020-01-02", 90, -10, 1, 0],
            ["IND", "Asia", "2020-01-03", 130, 40, 2, 1],
        ])

    def test_full_pipeline(self):
        out = clean.clean(self.df)
        self.assertNotIn("OWID_WRL", set(out["iso_code"]))
        self.assertEqual(list(out["total_cases"]), [100, 100, 130])
        self.assertTrue(out["new_cases"].dropna().ge(0).all())

    def test_pipeline_is_idempotent(self):
        once = clean.clean(self.df)
        twice = clean.clean(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_no_duplicate_country_days(self):
        out = clean.clean(self.df)
        self.assertFalse(out.duplicated(subset=["iso_code", "date"]).any())


if __name__ == "__main__":
    unittest.main()
