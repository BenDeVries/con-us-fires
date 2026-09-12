import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from conditional.data import Inputs, harmonic_block, load_frames, fit_transform, apply_transform
from conditional.spec import LOOKBACKS, initialization, specification
from conditional.state import ROOT, Session, Paused, read_json
from conditional import neural
from model.data import WindowDataset


class ConditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.spec = specification(smoke=True)
        cls.inputs = Inputs(cls.directory, cls.spec, fit=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def params(self):
        p = initialization('gnn')[0]
        p.update(representation='pca', n_pca=4, n_harmonics=2, gcn_hidden=8,
                 gcn_layers=2, lstm_hidden=64, lstm_layers=1, head_hidden=0,
                 lookback=15, teacher_forcing=True, tf_ratio_start=.8, tf_ratio_end=.1)
        return p

    def test_balanced_initialization(self):
        for family in ('gnn','xgb'):
            rows = initialization(family)
            self.assertEqual(rows, initialization(family))
            self.assertEqual(len(rows), 28)
            self.assertEqual(set(Counter(p['n_harmonics'] for p in rows).values()), {4})
            self.assertEqual(set(Counter(p.get('n_pca','raw') for p in rows).values()), {4})
        counts = Counter(p['lookback'] for p in initialization('gnn'))
        self.assertEqual(sorted(counts), LOOKBACKS)
        self.assertGreaterEqual(min(counts.values()), 2)

    def test_harmonic_rank_and_boundaries(self):
        names, x = harmonic_block(np.arange(1,13), 6)
        self.assertEqual(x.shape, (12,11)); self.assertNotIn('month_sin6', names)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([np.ones(12), x])), 12)
        self.assertEqual(harmonic_block(np.arange(1,13), 0)[1].shape, (12,0))
        with self.assertRaises(ValueError):
            harmonic_block(np.arange(1,13), 7)

    def test_all_lookbacks_have_common_requests(self):
        p = self.params(); panel = self.inputs.panel(p)
        origins = panel.split_origins['test']
        for length in LOOKBACKS:
            dataset = WindowDataset(panel, origins, length, 12)
            for i, origin in enumerate(origins):
                row = dataset[i]
                self.assertEqual(row['enc_cov'].shape[0], length)
                torch.testing.assert_close(row['enc_ar'][-1], panel.ar[origin])
                torch.testing.assert_close(row['y'], panel.y[origin+1:origin+13])
                torch.testing.assert_close(row['enc_cov'][0], panel.cov[origin-length+1])

    def test_zero_full_raw_and_saved_transform(self):
        p = self.params(); p.update(n_harmonics=0)
        for k in (0,1,39):
            p['n_pca'] = k
            panel = self.inputs.panel(p)
            self.assertEqual(panel.cov.shape[-1], k+12)
            np.testing.assert_array_equal(panel.cov[...,k:].numpy(), self.inputs.raw[...,39:])
        p['representation'] = 'raw'
        self.assertEqual(self.inputs.panel(p).cov.shape[-1], 51)
        reloaded = Inputs(self.directory, self.spec)
        np.testing.assert_array_equal(reloaded.pcs, self.inputs.pcs)

    def test_heldout_isolation_and_unknown_categories(self):
        full, dates, n = load_frames(self.spec)
        modified = full.copy()
        held = modified.split.ne('train')
        modified.loc[held, self.inputs.meta['names']] += 100
        modified.loc[held, 'lc_dominant'] = 999
        args = (self.inputs.meta['names'], self.inputs.meta['static_names'], self.inputs.meta['source_scaler'])
        a, meta_a = fit_transform(full, *args)
        b, meta_b = fit_transform(modified, *args)
        self.assertEqual(meta_a, meta_b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        xa = apply_transform(full, meta_a, a, 4)
        xb = apply_transform(modified, meta_b, b, 4)
        np.testing.assert_array_equal(xa[~held], xb[~held])
        self.assertFalse(np.array_equal(xa[held], xb[held]))
        self.assertNotIn(999, meta_b['categories'])

    def test_future_response_cannot_change_earlier_history(self):
        inputs = copy.copy(self.inputs)
        inputs.y = self.inputs.y.copy(); inputs.occ = self.inputs.occ.copy()
        p = self.params(); p.update(n_spatial=3, n_temporal=47)
        # Force one origin so the perturbation represents future data for every checked row.
        inputs.origins = dict(self.inputs.origins)
        o = inputs.origins['test'][0]; inputs.origins['test'] = [o]
        before, _ = inputs.tree_design(p, 'test')
        panel = inputs.panel(p)
        old = {k: v.clone() for k,v in WindowDataset(panel, [o], 48, 12)[0].items()}
        inputs.y[o+1:] = .5; inputs.occ[o+1:] = 1
        after, _ = inputs.tree_design(p, 'test')
        # Direct history reads must remain fixed; cached causal summaries are checked separately.
        np.testing.assert_array_equal(before.drop(columns='lc_dominant'), after.drop(columns='lc_dominant'))
        new = WindowDataset(inputs.panel(p), [o], 48, 12)[0]
        torch.testing.assert_close(old['enc_ar'], new['enc_ar'])
        self.assertFalse(torch.equal(old['y'], new['y']))
        np.testing.assert_allclose(inputs.county_y[o], self.inputs.y[:o].mean(axis=0), rtol=1e-6)

    def test_neural_resume_matches_uninterrupted(self):
        p = self.params(); spec = copy.deepcopy(self.spec)
        spec['limits']['gnn'] = 3
        uninterrupted = self.directory / 'nn_full'
        resumed = self.directory / 'nn_resume'
        score = neural.fit(self.inputs, p, spec, uninterrupted, 0, Session(), device='cpu')
        with self.assertRaises(Paused):
            neural.fit(self.inputs, p, spec, resumed, 0, Session(boundary_limit=1), device='cpu')
        self.assertFalse(read_json(resumed / 'progress.json')['done'])
        other = neural.fit(self.inputs, p, spec, resumed, 0, Session(), device='cpu')
        self.assertEqual(score, other)
        full = neural.load_state(uninterrupted / 'last.pt'); part = neural.load_state(resumed / 'last.pt')
        for key in full['model']:
            torch.testing.assert_close(full['model'][key], part['model'][key], atol=0, rtol=0)
        pred = neural.predict(self.inputs, resumed, 'test', device='cpu')
        from conditional.metrics import row_nll
        self.assertEqual(float(row_nll(self.inputs.keys('test').y_true, **{'p':pred['p_occ'],
                         'mu':pred['mu'], 'phi':pred['phi']}).mean()), other)


if __name__ == '__main__':
    unittest.main()
