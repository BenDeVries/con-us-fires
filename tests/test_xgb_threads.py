from pathlib import Path
import tempfile
import unittest

import torch

from conditional import trees
from conditional.data import Inputs
from conditional.spec import specification, initialization
from conditional.state import Session, Paused, load_pickle, read_json
from tools.run_xgb import thread_override


class ThreadOverrideTests(unittest.TestCase):
    def test_16_thread_restart_preserves_checkpoint_and_history(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = specification(smoke=True)
            inputs = Inputs(root, spec, fit=True)
            p = initialization('xgb')[0]
            p.update(representation='pca', n_pca=0, n_harmonics=0, max_depth=3,
                     n_spatial=1, n_temporal=3, min_child_weight=5.)
            target = root / 'fit'
            with self.assertRaises(Paused):
                trees.fit(inputs, p, spec, target, 0, Session(boundary_limit=2))
            before = load_pickle(target / 'last.pkl')
            with thread_override(16):
                trees.fit(inputs, p, spec, target, 0, Session())
                self.assertEqual(torch.get_num_threads(), 16)
            after = load_pickle(target / 'last.pkl')
            self.assertEqual(before['identity'], after['identity'])
            self.assertEqual(before['history'], after['history'][:2])
            self.assertEqual(after['iteration'], spec['limits']['xgb'])
            self.assertEqual(read_json(target / 'config.json')['booster_params']['nthread'], 16)
            self.assertEqual(read_json(target / 'resource_history.json')[0]['after_iteration'], 2)
            self.assertEqual(spec['xgb_nthread'], 8)
            self.assertTrue(after['done'])


if __name__ == '__main__':
    unittest.main()
