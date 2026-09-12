"""Small offline regressions for cross-split donors and missing outcomes."""
import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from model.preprocessing import (forward_fill_predictors, validate_export_keys,
                                 validate_outcomes, validate_shards)


class PreprocessingTests(unittest.TestCase):
    def panel(self, values):
        return pd.DataFrame({"county_fips": ["00001"] * len(values),
                             "date": pd.date_range("2018-04-01", periods=len(values), freq="MS"),
                             "x": values})

    def test_first_test_value_cannot_fill_training(self):
        with self.assertRaisesRegex(ValueError, "backward fill is forbidden"):
            forward_fill_predictors(self.panel([np.nan, np.nan, 7.0]), ["x"])

    def test_heldout_values_cannot_change_training(self):
        a = forward_fill_predictors(self.panel([1.0, np.nan, 7.0, 9.0]), ["x"])
        b = forward_fill_predictors(self.panel([1.0, np.nan, -700.0, 900.0]), ["x"])
        pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])
        self.assertEqual(a.x.tolist(), [1.0, 1.0, 7.0, 9.0])

    def test_missing_outcome_fails(self):
        with self.assertRaisesRegex(ValueError, "response imputation is forbidden"):
            validate_outcomes(pd.DataFrame({"burned_fraction": [0.0, np.nan], "fire_occurred": [0, 0]}))

    def test_invalid_numeric_outcome_fails(self):
        with self.assertRaises(ValueError):
            validate_outcomes(pd.DataFrame({"burned_fraction": ["bad"], "fire_occurred": [0]}))

    def test_occurrence_matches_observed_positive(self):
        validate_outcomes(pd.DataFrame({"burned_fraction": [0, 1e-8, .2], "fire_occurred": [0, 1, 1]}))
        with self.assertRaisesRegex(ValueError, "indicator"):
            validate_outcomes(pd.DataFrame({"burned_fraction": [.2], "fire_occurred": [0]}))

    def test_missing_year_fails(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_shards(["combined_2018.csv"], [2018, 2019])

    def test_missing_key_fails_before_spine_join(self):
        raw = pd.DataFrame({"county_fips": ["00001"] * 11, "year": [2018] * 11,
                            "month": range(1, 12)})
        with self.assertRaisesRegex(ValueError, "1 missing"):
            validate_export_keys(raw, ["00001"], [2018])

    def test_duplicate_key_fails(self):
        raw = pd.DataFrame({"county_fips": ["00001"] * 2, "year": [2018] * 2, "month": [1, 1]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_export_keys(raw, ["00001"], [2018])

    def test_incomplete_cached_spine_is_rejected(self):
        # Exercise the real assembly script: checking raw keys alone is insufficient
        # if a cached spine silently drops one of those otherwise valid records.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "data").mkdir()
            raw = pd.DataFrame({"county_fips": ["00001"] * 12, "year": [2018] * 12,
                                "month": range(1, 13), "burned_m2": [0] * 12,
                                "mapped_m2": [0] * 12, "county_area_km2": [100] * 12})
            raw.to_csv(root / "raw/combined_2018.csv", index=False)
            raw.loc[raw.month > 1, ["county_fips", "year", "month"]].to_parquet(
                root / "data/spine.parquet", index=False)
            (root / "data/node_index.json").write_text(json.dumps({"00001": 0}))
            config = types.ModuleType("config")
            config.OUTPUT_DIR = str(root)
            config.START_YEAR = config.END_YEAR = 2018
            config.FIRE_FLOOR_KM2 = 0
            config.IGBP_GROUPS = {}
            script = Path(__file__).resolve().parents[1] / "step10_join.py"
            with patch.dict("sys.modules", {"config": config}), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "1 missing"):
                    runpy.run_path(str(script))
            self.assertFalse((root / "data/master.parquet").exists())

    def test_imputer_cannot_be_used_for_responses(self):
        frame = self.panel([1, 2]).rename(columns={"x": "burned_fraction"})
        with self.assertRaisesRegex(ValueError, "Response columns"):
            forward_fill_predictors(frame, ["burned_fraction"])


if __name__ == "__main__":
    unittest.main()
