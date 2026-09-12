"""Small adversarial panels exercise the complete forecast design, not just lag helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import features as F


class ForecastSafeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.dates = pd.date_range("2003-01-01", periods=120, freq="MS")
        self.panel = pd.DataFrame({"date": np.repeat(self.dates, 3),
                                   "node_id": np.tile(np.arange(3), 120)})
        t = np.repeat(np.arange(120), 3)
        self.panel[F.TARGET_COL] = np.where((t + self.panel.node_id) % 4 == 0,
                                           (t + 1) / 1000, 0).astype(np.float32)
        self.panel[F.GATE_COL] = (self.panel[F.TARGET_COL] > 0).astype(np.float32)
        for i, c in enumerate(F.FORECAST_COLS):
            self.panel[c] = (self.panel.node_id + i).astype(np.float32)
        for c in ("lc_dominant", "evc", "pct_forest", "pop_density", "built_frac", "tmmx"):
            self.panel[c] = t.astype(np.float32)
        self.ranges = {"train": (self.dates[0], self.dates[71]),
                       "test": (self.dates[72], self.dates[95]),
                       "validation": (self.dates[96], self.dates[119])}
        (self.data_dir / "node_index.json").write_text(json.dumps({"00001": 0, "00002": 1,
                                                                  "00003": 2}))
        np.savez(self.data_dir / "county_graph.npz",
                 edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
        self.spec = dict(forecast_safe=True, lookback=48, horizon=12,
                         max_spatial=3, max_temporal=47, n_harmonics=6)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, panel=None, **overrides):
        panel = self.panel if panel is None else panel

        def load(columns=None):
            return panel.copy(deep=True), self.ranges.copy()

        with patch.object(F, "DATA_DIR", self.data_dir), patch.object(F, "_load_full_panel", load):
            return F.build_dataset(**{**self.spec, **overrides})

    def perturb_response(self, month, onwards=False):
        panel = self.panel.copy()
        mask = panel.date >= self.dates[month] if onwards else panel.date == self.dates[month]
        panel.loc[mask, F.TARGET_COL] = 0.7
        panel.loc[mask, F.GATE_COL] = 1
        return panel

    def test_no_response_after_origin_reaches_any_feature(self):
        clean = self.build()
        for month in (59, 72, 96):  # own training targets, test outcomes, validation outcomes
            with self.subTest(month=month):
                changed = self.build(self.perturb_response(month, onwards=True))
                changed_targets = 0
                for split, d in clean.splits.items():
                    earlier = d["meta"].origin_date < self.dates[month]
                    pd.testing.assert_frame_equal(d["X"].loc[earlier],
                                                  changed.splits[split]["X"].loc[earlier])
                    changed_targets += np.count_nonzero(d["y"][earlier] !=
                                                         changed.splits[split]["y"][earlier])
                self.assertGreater(changed_targets, 0)

    def test_county_summaries_strictly_precede_origin(self):
        clean = self.build()
        changed = self.build(self.perturb_response(71))
        cols = ["county_mean_y", "county_occ_rate"]
        d, after = clean.splits["test"], changed.splits["test"]
        origin = d["meta"].origin_date == self.dates[71]
        pd.testing.assert_frame_equal(d["X"].loc[origin, cols], after["X"].loc[origin, cols])
        self.assertFalse(d["X"].loc[origin, "y_o"].equals(after["X"].loc[origin, "y_o"]))
        historical = self.panel.loc[self.panel.date < self.dates[71]].groupby("node_id")
        np.testing.assert_allclose(d["X"].loc[origin, "county_mean_y"].iloc[:3],
                                   historical[F.TARGET_COL].mean(), rtol=1e-6)

    def test_forbidden_covariates_and_heldout_terrain_have_no_effect(self):
        clean = self.build()
        changed = self.panel.copy()
        for c in ("lc_dominant", "evc", "pct_forest", "pop_density", "built_frac", "tmmx"):
            changed[c] = np.nan
        changed.loc[changed.date > self.dates[0], list(F.FORECAST_COLS)] = 10000
        after = self.build(changed)
        for split, d in clean.splits.items():
            pd.testing.assert_frame_equal(d["X"], after.splits[split]["X"])
            self.assertEqual(list(d["X"][F.CAT_COL].cat.categories), [0])
        self.assertEqual(set(clean.raw_cov_names), set(F.FORECAST_COLS))

    def test_invalid_protocol_combinations_are_rejected(self):
        for kw in ({"climatology": True}, {"clim_oof": True}, {"static_only": True},
                   {"n_nbr_pcs": 1}, {"lookback": 12}, {"horizon": 13},
                   {"max_spatial": 4}, {"max_temporal": 48}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.build(**kw)

    def test_selected_columns_preserve_protocol(self):
        ds = F.select_columns(self.build(), n_spatial=1, n_temporal=3, n_harmonics=2)
        self.assertTrue(ds.forecast_safe)
        self.assertNotIn("st_y_s3_t47", ds.feature_names)
        self.assertNotIn("month_sin6", ds.feature_names)

    def test_best_params_exclude_pruned_and_nonfinite_trials(self):
        import optuna
        from .tune import save_progress

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("arm_spec", {"forecast_safe": True})
        for state, value in ((optuna.trial.TrialState.PRUNED, -100),
                             (optuna.trial.TrialState.COMPLETE, float("inf")),
                             (optuna.trial.TrialState.COMPLETE, -0.2)):
            study.add_trial(optuna.trial.create_trial(state=state, value=value))
        save_progress(study, self.data_dir)
        board = json.loads((self.data_dir / "leaderboard.json").read_text())
        self.assertEqual([r["trial"] for r in board], [2])
        params = json.loads((self.data_dir / "best_params.json").read_text())
        self.assertTrue(params["forecast_safe"])


if __name__ == "__main__":
    unittest.main()
