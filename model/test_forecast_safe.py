"""Perturbation tests for the strict forecast information boundary.

Run: python -m unittest model.test_forecast_safe
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import torch

from . import data
from .config import Config
from .model import SpatioTemporalZIB, build_norm_adj
from .train import train_model
from .tune import completed_trials


class ForecastBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        dates = pd.date_range("2000-01-01", periods=90, freq="MS")
        rows = []
        for t, date in enumerate(dates):
            for n in range(3):
                rows.append({"date": date, "month": date.month, "node_id": n,
                             "split": "train" if t < 60 else "test" if t < 75 else "validation",
                             "burned_fraction": (0.005 * (n + 1) if t % 3 == 0 else 0.0),
                             "fire_occurred": float(t % 3 == 0),
                             "lc_dominant": (n + t // 12) % 5,
                             "elev": float(n), "slope": float(n + 1),
                             "aspect_cos": float(n) / 3, "aspect_sin": -float(n) / 3,
                             "pop_density": float(t), "built_frac": float(t) / 100,
                             "evc_mean": float(t), "weather": float(t)})
        self.full = pd.DataFrame(rows)
        self.ranges = {s: (g.date.min(), g.date.max())
                       for s, g in self.full.groupby("split", sort=False)}

    def build(self, full=None):
        frame = self.full if full is None else full
        with patch.object(data, "_load_full_panel", return_value=(frame.copy(), self.ranges)), \
             patch.object(data.np, "load", return_value={"edge_index": np.array([[0, 1], [1, 2]])}), \
             patch.object(Path, "read_text", return_value=json.dumps({"00001": 0, "00002": 1, "00003": 2})):
            return data.build_panel(12, 3, n_harmonics=2, forecast_safe=True)

    def batch(self, panel, origin):
        row = data.WindowDataset(panel, [origin], 12, 3)[0]
        return {k: v.unsqueeze(0) for k, v in row.items()}

    def config(self):
        return Config(forecast_safe=True, lookback=12, horizon=3, n_harmonics=2,
                      lc_embed_dim=0, gcn_hidden=8, gcn_layers=1, lstm_hidden=8,
                      county_embed_dim=2, max_epochs=1, batch_size=2, device="cpu",
                      n_samples=0, drop_features=[])

    def test_heldout_perturbations_do_not_change_training_inputs_or_fit(self):
        altered = self.full.copy()
        heldout = altered.split != "train"
        for column in altered.select_dtypes(include="number").columns:
            if column not in {"month", "node_id"}:
                altered[column] = altered[column].astype(float)
                altered.loc[heldout, column] = 0.7
        panels = [self.build(), self.build(altered)]
        for origin in panels[0].split_origins["train"]:
            for key, value in self.batch(panels[0], origin).items():
                torch.testing.assert_close(value, self.batch(panels[1], origin)[key], rtol=0, atol=0)
        # One fixed epoch deliberately removes checkpoint selection as a confounder:
        # test is authorized for tuning, but its labels must never enter backpropagation.
        states = []
        for panel in panels:
            panel.split_origins = {s: origins[:2] for s, origins in panel.split_origins.items()}
            with tempfile.TemporaryDirectory() as directory:
                _, _, val, model = train_model(self.config(), panel, Path(directory),
                                               eval_validation=False, verbose=False)
                self.assertIsNone(val)
                report = json.loads((Path(directory) / "metrics.json").read_text())
                self.assertNotIn("validation", report)
                states.append({k: v.clone() for k, v in model.state_dict().items()})
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_future_outcomes_do_not_change_origin_inputs_or_predictions(self):
        original = self.build()
        origin = original.split_origins["test"][0]
        altered = self.full.copy()
        altered.loc[altered.date > original.dates[origin], "burned_fraction"] = 0.6
        changed = self.build(altered)
        left, right = self.batch(original, origin), self.batch(changed, origin)
        self.assertFalse(torch.equal(left["y"], right["y"]))
        for key in left.keys() - {"y"}:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        model = SpatioTemporalZIB(self.config(), original.cov.shape[-1], 1, n_nodes=3).eval()
        adjacency = build_norm_adj(original.edge_index, 3)
        for key, value in model(left, adjacency).items():
            torch.testing.assert_close(value, model(right, adjacency)[key], rtol=0, atol=0)
        at_origin = self.full.copy()
        at_origin.loc[at_origin.date == original.dates[origin], "burned_fraction"] = 0.7
        self.assertFalse(torch.equal(left["enc_ar"], self.batch(self.build(at_origin), origin)["enc_ar"]))

    def test_forbidden_covariates_and_category_vocab_have_no_effect(self):
        changed = self.full.copy()
        for column in ["lc_dominant", "pop_density", "built_frac", "evc_mean", "weather"]:
            changed[column] = np.arange(len(changed)) * 1000.0
        panels = [self.build(), self.build(changed)]
        self.assertEqual(panels[0].cov_names[:4], list(data.FORECAST_COLS))
        for panel in panels:
            self.assertEqual(panel.n_lc_classes, 1)
            self.assertEqual(int(panel.cat.sum()), 0)
        torch.testing.assert_close(panels[0].cov, panels[1].cov, rtol=0, atol=0)
        origin = panels[0].split_origins["validation"][0]
        model = SpatioTemporalZIB(self.config(), panels[0].cov.shape[-1], 1, n_nodes=3).eval()
        adjacency = build_norm_adj(panels[0].edge_index, 3)
        outputs = [model(self.batch(panel, origin), adjacency) for panel in panels]
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)

    def test_window_bounds_and_whole_split_targets(self):
        panel = self.build()
        for origin in [-1, 0, 10, len(panel.dates) - 3, len(panel.dates)]:
            with self.assertRaises(ValueError):
                data.WindowDataset(panel, [origin], 12, 3)
        targets = {}
        for split, origins in panel.split_origins.items():
            targets[split] = {t for origin in origins for t in range(origin + 1, origin + 4)}
            start, end = self.ranges[split]
            self.assertTrue(all(start <= panel.dates[t] <= end for t in targets[split]))
        self.assertFalse(targets["train"] & targets["test"])
        self.assertFalse(targets["test"] & targets["validation"])
        self.assertFalse(targets["train"] & targets["validation"])
        overlap = dict(self.ranges, test=self.ranges["train"])
        with self.assertRaises(ValueError):
            data.rebuild_origins(panel.dates, overlap, 12, 3)

    def test_safe_mode_rejects_pca_bypass_teacher_forcing_and_landcover(self):
        for field, value in [("n_pca", 2), ("static_bypass", True),
                             ("teacher_forcing", True), ("lc_embed_dim", 1)]:
            with self.assertRaises(ValueError):
                dataclasses.replace(self.config(), **{field: value}).validate_forecast_safe()
        with self.assertRaises(ValueError):
            data.fit_pca(self.build())

    def test_pruned_intermediate_score_cannot_be_a_finalist(self):
        study = optuna.create_study(direction="minimize")
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, value=-99.0))
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, value=-1.0))
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, value=float("inf")))
        self.assertEqual([trial.number for trial in completed_trials(study)], [1])


if __name__ == "__main__":
    unittest.main()
