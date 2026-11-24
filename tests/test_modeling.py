import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import modeling  # noqa: E402


def synthetic_sample(n=60, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({c: rng.normal(size=n) for c in modeling.PREDICTORS})
    # outcome depends only on median_age, plus a little noise
    data["log_deaths_per_million"] = 2.0 * data["median_age"] + rng.normal(scale=0.1, size=n)
    return data


class TestStandardise(unittest.TestCase):
    def test_columns_have_zero_mean_and_unit_std(self):
        out = modeling.standardise(synthetic_sample()[modeling.PREDICTORS])
        self.assertTrue(np.allclose(out.mean(), 0, atol=1e-9))
        self.assertTrue(np.allclose(out.std(), 1, atol=1e-9))


class TestFitModel(unittest.TestCase):
    def test_recovers_the_driving_predictor(self):
        model, n = modeling.fit_model(synthetic_sample())
        self.assertEqual(n, 60)
        self.assertGreater(model.params["median_age"], 1.0)
        self.assertLess(model.pvalues["median_age"], 0.001)
        self.assertGreater(model.rsquared, 0.9)

    def test_irrelevant_predictors_are_not_significant(self):
        model, _ = modeling.fit_model(synthetic_sample())
        self.assertGreater(model.pvalues["hospital_beds_per_thousand"], 0.01)

    def test_rows_with_missing_values_are_dropped(self):
        data = synthetic_sample()
        data.loc[0:4, "median_age"] = np.nan
        _, n = modeling.fit_model(data)
        self.assertEqual(n, 55)

    def test_predictor_labels_cover_every_predictor(self):
        self.assertEqual(set(modeling.PREDICTORS), set(modeling.LABELS))


if __name__ == "__main__":
    unittest.main()
