"""Small independent checks for descriptive STACF and occurrence-curve reporting."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import prepare_forecast_results as report

from prepare_forecast_results import (discrimination, observed_stacf, sample_plot_curve,
                                     stacf_operators, stacf_permutations, stacf_nulls,
                                     stacf_pvalues, _gram, _stacf, _perm_p)


class ForecastReportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.array([[0, 1, 2], [1, 2, 3]])  # chain plus isolated node 4
        self.operators = stacf_operators(self.edges, 5, 3)
        self.cube = np.random.default_rng(71).normal(size=(2, 6, 5))
        self.cube[0, 5] = np.nan
        self.cube[1, 0] = np.nan
        rows, residuals = [], []
        for h in range(2):
            for t in range(6):
                for county in range(5):
                    if np.isfinite(self.cube[h, t, county]):
                        rows.append({"horizon": h + 1,
                                     "target_date": pd.Timestamp("2000-01-01") + pd.DateOffset(months=t),
                                     "node_id": county})
                        residuals.append(self.cube[h, t, county])
        self.frame = pd.DataFrame(rows)
        self.residuals = np.array(residuals)

    def direct(self, field, valid, max_lag):
        """Scalar-loop definition; no Gram matrices or vectorized lag indexing."""
        H, M, N = field.shape
        weighted = np.zeros((len(self.operators), H, M, N))
        energies = []
        for l, operator in enumerate(self.operators):
            w = operator.toarray()
            square_sum, slices = 0.0, 0
            for h in range(H):
                for t in range(M):
                    if not valid[h, t]:
                        continue
                    slices += 1
                    for i in range(N):
                        weighted[l, h, t, i] = sum(w[i, j] * field[h, t, j] for j in range(N))
                        square_sum += weighted[l, h, t, i] ** 2
            energies.append(square_sum / slices / N)
        expected = np.zeros((len(self.operators), max_lag + 1))
        counts = []
        for k in range(max_lag + 1):
            pairs = [(h, t) for h in range(H) for t in range(M - k)
                     if valid[h, t] and valid[h, t + k]]
            counts.append(len(pairs))
            for l in range(len(self.operators)):
                product = sum(weighted[l, h, t, i] * field[h, t + k, i]
                              for h, t in pairs for i in range(N))
                expected[l, k] = product / len(pairs) / N / np.sqrt(energies[l] * energies[0])
        return expected, counts

    def test_gram_matches_direct_horizon_preserving_formula(self):
        result = observed_stacf(self.frame, self.residuals, self.operators, 4)
        present = np.isfinite(self.cube)
        raw = np.where(present, self.cube - np.nanmean(self.cube), 0.0)
        for name, field in (("raw", raw), ("county_anomaly", raw - raw.mean(axis=2, keepdims=True))):
            expected, counts = self.direct(field, present.all(axis=2), 4)
            np.testing.assert_allclose(result[name]["rho"], expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(result["n_horizon_month_pairs"], counts)
            self.assertAlmostEqual(result[name]["rho"][0][0], 1)

    def test_graph_orders_are_exclusive_and_isolate_has_zero_rows(self):
        dense = [w.toarray() for w in self.operators]
        self.assertEqual(dense[1][0, 1], 1)
        self.assertEqual(dense[2][0, 2], 1)
        self.assertEqual(dense[3][0, 3], 1)
        for order in range(1, 4):
            self.assertEqual(dense[order][4].sum(), 0)
            self.assertEqual(np.trace(dense[order]), 0)

    def test_lag_order_and_county_bounds_fail_closed(self):
        for lag in (-1, 6, 1.5):
            with self.assertRaisesRegex(ValueError, "temporal lag"):
                observed_stacf(self.frame, self.residuals, self.operators, lag)
        for order in (-1, 5, 1.5):
            with self.assertRaisesRegex(ValueError, "spatial order"):
                stacf_operators(self.edges, 5, order)
        invalid = self.frame.copy()
        invalid.loc[0, "node_id"] = 5
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            observed_stacf(invalid, self.residuals, self.operators, 4)
        with self.assertRaisesRegex(ValueError, "outside county index"):
            stacf_operators(np.array([[-1], [0]]), 5, 3)

    def test_partial_slices_and_calendar_gaps_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "ragged"):
            observed_stacf(self.frame.iloc[1:], self.residuals[1:], self.operators, 4)
        keep = self.frame.target_date != pd.Timestamp("2000-03-01")
        with self.assertRaisesRegex(ValueError, "consecutive"):
            observed_stacf(self.frame.loc[keep], self.residuals[keep], self.operators, 3)

    def test_zero_variance_is_undefined(self):
        result = observed_stacf(self.frame, np.ones(len(self.frame)), self.operators, 4)
        for field in ("raw", "county_anomaly"):
            self.assertTrue(all(value is None for row in result[field]["rho"] for value in row))

    def test_ties_distinguish_pr_trapezoid_from_average_precision(self):
        result = discrimination(np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5]), True)
        self.assertEqual(result["gate_auc"], .5)
        self.assertEqual(result["gate_average_precision"], .5)
        self.assertEqual(result["gate_pr_auc"], .75)
        self.assertEqual(result["discrimination_curves"]["roc"]["fpr"], [0, 1])
        self.assertEqual(result["discrimination_curves"]["precision_recall"]["recall"], [1, 0])

    def test_plot_sampling_preserves_endpoints_and_bin_extrema(self):
        x = np.linspace(0, 1, 10001)
        y = np.random.default_rng(7).random(len(x))
        sx, sy = sample_plot_curve(x, y, bins=8)
        self.assertEqual((sx[0], sy[0], sx[-1], sy[-1]), (x[0], y[0], x[-1], y[-1]))
        self.assertLessEqual(len(sx), 4 * 9 + 2)
        groups = np.floor(x * 8).astype(int)
        selected = set(zip(sx, sy))
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            for i in (indices[np.argmin(y[indices])], indices[np.argmax(y[indices])]):
                self.assertIn((x[i], y[i]), selected)

    def test_permutation_nulls_match_original_direct_recomputation(self):
        present = np.isfinite(self.cube)
        valid = present.all(axis=2)
        raw = np.where(present, self.cube - np.nanmean(self.cube), 0.0)
        month_orders, county_orders = stacf_permutations(6, 5, 19, 271828)
        for field in (raw, raw - raw.mean(axis=2, keepdims=True)):
            rho, pairs, month_null, county_null = stacf_nulls(
                field, self.operators, valid, 4, month_orders, county_orders)
            expected_month = np.stack([_stacf(*_gram(field[:, order, :], self.operators),
                                              valid[:, order], 4)[0] for order in month_orders])
            expected_county = np.stack([_stacf(*_gram(field[:, :, order], self.operators),
                                               valid, 4)[0] for order in county_orders])
            np.testing.assert_allclose(month_null, expected_month, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(county_null, expected_county, atol=1e-12, rtol=1e-12)
            actual = stacf_pvalues(rho, month_null, county_null)
            for name, null in (("month", expected_month), ("county", expected_county)):
                original = _perm_p(rho, null)
                original[0, 0] = np.nan
                np.testing.assert_allclose(np.array(actual[f"p_{name}"], dtype=float), original,
                                           atol=1e-12, rtol=1e-12, equal_nan=True)

    def test_centered_pvalues_count_ties_route_cells_and_mask_undefined(self):
        rho = np.array([[1., 2., np.nan], [1.5, -1., 0.]])
        month = np.array([[[1., x, 0.], [y, -x, y]]
                          for x, y in ((0., 0.), (1., 1.), (2., 2.), (3., 3.))])
        county = month[::-1] + np.array([[0., 1., 0.], [1., 2., 1.]])
        result = stacf_pvalues(rho, month, county)
        for name, null in (("month", month), ("county", county)):
            for l, k in ((0, 1), (1, 0), (1, 1), (1, 2)):
                center = sum(null[:, l, k]) / len(null)
                hits = sum(abs(value - center) >= abs(rho[l, k] - center) for value in null[:, l, k])
                self.assertEqual(result[f"p_{name}"][l][k], (1 + hits) / (len(null) + 1))
            self.assertIsNone(result[f"p_{name}"][0][0])
            self.assertIsNone(result[f"p_{name}"][0][2])
        self.assertEqual(result["p_display"][0][1], result["p_month"][0][1])
        self.assertEqual(result["p_display"][1][0], result["p_county"][1][0])
        self.assertEqual(result["p_display"][1][1], max(result["p_month"][1][1], result["p_county"][1][1]))

    def test_permutations_are_reproducible_and_zero_variance_has_no_pvalue(self):
        a = observed_stacf(self.frame, self.residuals, self.operators, 4, n_perm=19)
        b = observed_stacf(self.frame, self.residuals * 2, self.operators, 4, n_perm=19)
        self.assertEqual(a["permutation_sha256"], b["permutation_sha256"])
        self.assertEqual(a["p_value_resolution"], .05)
        zero = observed_stacf(self.frame, np.ones(len(self.frame)), self.operators, 4, n_perm=19)
        for field in ("raw", "county_anomaly"):
            self.assertTrue(all(value is None for row in zero[field]["p_display"] for value in row))

    def test_comparison_and_validation_blocks_read_only_their_assigned_splits(self):
        row = {"rows": 120, "mean_nll": -.2, "mean_crps": .001,
               "mae_expected_fraction": .002, "brier": .05, "residual_mean": 0.,
               "residual_sd": 1., "moran_i_county_mean_residual": .1, "lag1_monthly_mean_residual": .2}
        development = {"results": {model: {"train": row, "test": row} for model in report.MODELS}}
        later = {"results": {model: {"validation": row} for model in report.MODELS}}
        comparison = report.comparison_block(development)
        validation = report.validation_block(later)
        self.assertNotIn("| Validation |", comparison)
        self.assertNotIn("validation-diagnostics.png", comparison)
        self.assertNotIn("| Train |", validation)
        self.assertNotIn("| Test |", validation)
        for block in (comparison, validation):
            self.assertIn("Expected-fraction MAE", block)
            self.assertIn("Occurrence Brier", block)
            self.assertIn("Forecast requests", block)

    def test_model_blocks_need_selection_metadata_but_no_split_scores(self):
        for model, spec in report.MODELS.items():
            summary = {"model_specs": {model: {
                "selection": {"trials": [{"trial": 1}], "complete_trials": 1, "selected_trial": 1},
                "checkpoint": {"path": spec["checkpoint"]},
                "config": {"best_iteration": 9, "max_spatial": 1, "max_temporal": 3,
                           "n_harmonics": 2, "gcn_layers": 2, "gcn_hidden": 64,
                           "lstm_layers": 1, "lstm_hidden": 64},
            }}}
            block = report.model_block(model, summary)
            self.assertIn("Trial 1", block)
            self.assertNotIn("| Split |", block)
            self.assertNotIn("Validation", block)

    def test_all_diagnostic_panels_use_one_vertical_column(self):
        rho = {"rho": [[1, .1, 0], [.1, 0, -.1]],
               "p_display": [[None, .1, .5], [.01, .2, .4]]}
        stacf = {"raw": rho, "county_anomaly": rho, "max_lag": 2, "max_order": 1,
                 "n_perm": 199, "permutation_seed": 271828, "p_value_resolution": .005}
        row = {"qq_theoretical": [-1, 0, 1], "qq_observed": [-1, 0, 1],
               "gate_calibration": {"predicted": [.1, .2], "observed": [.1, .2]},
               "monthly": {"date": ["2000-01-01", "2000-02-01", "2000-03-01"], "residual": [0, .01, 0]},
               "gate_auc": .7, "gate_pr_auc": .4, "gate_average_precision": .41,
               "occurrence_rate": .1, "stacf": stacf,
               "discrimination_curves": {"roc": {"fpr": [0, .5, 1], "tpr": [0, .8, 1]},
                                         "precision_recall": {"recall": [1, .5, 0], "precision": [.1, .5, 1]}}}
        results = {model: {split: row for split in report.SPLITS} for model in report.MODELS}
        captured = []
        original_subplots = report.plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append(np.asarray(axes).ravel())
            return figure, axes

        with patch.object(report.plt, "subplots", side_effect=capture), patch("matplotlib.figure.Figure.savefig"):
            report.draw_figure(results, ("train", "test"), Path("unused.png"))
            report.draw_train_stacf(results[next(iter(report.MODELS))], "Fixture", .1, Path("unused.png"))
            report.draw_discrimination(results, "test", Path("unused.png"))
        self.assertEqual([len(axes) for axes in captured], [6, 2, 2])
        for axes in captured:
            self.assertTrue(all(ax.get_subplotspec().get_gridspec().ncols == 1 for ax in axes))
            self.assertTrue(np.all(np.diff([ax.get_position().y0 for ax in axes]) < 0))


if __name__ == "__main__":
    unittest.main()
