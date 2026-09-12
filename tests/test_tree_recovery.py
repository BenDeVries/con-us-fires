from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from conditional.data import Inputs
from conditional.spec import specification, initialization
from conditional.state import Session, Paused, load_pickle
from conditional import trees


class TreeRecoveryTests(unittest.TestCase):
    def test_global_round_rng_and_best_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); spec = specification(smoke=True)
            spec['limits']['xgb'] = 6
            inputs = Inputs(directory, spec, fit=True)
            p = initialization('xgb')[0]
            p.update(representation='pca', n_pca=0, n_harmonics=0, max_depth=3,
                     n_spatial=1, n_temporal=3, subsample=.65, colsample_bytree=.7,
                     min_child_weight=5., eta=.1)
            full = directory / 'full'; resumed = directory / 'resume'
            score = trees.fit(inputs, p, spec, full, 0, Session())
            with self.assertRaises(Paused):
                trees.fit(inputs, p, spec, resumed, 0, Session(boundary_limit=2))
            other = trees.fit(inputs, p, spec, resumed, 0, Session())
            self.assertEqual(score, other)
            self.assertEqual([r['test_nll'] for r in load_pickle(full/'last.pkl')['history']],
                             [r['test_nll'] for r in load_pickle(resumed/'last.pkl')['history']])
            for key, values in trees.predict(inputs, full, 'test').items():
                np.testing.assert_array_equal(values, trees.predict(inputs, resumed, 'test')[key])
            from conditional.metrics import row_nll
            pred = trees.predict(inputs, full, 'test')
            self.assertEqual(float(row_nll(inputs.keys('test').y_true, pred['p_occ'], pred['mu'], pred['phi']).mean()), score)


if __name__ == '__main__':
    unittest.main()
