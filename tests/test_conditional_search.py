from pathlib import Path
import tempfile
import unittest

import optuna
from optuna.trial import TrialState

from conditional.search import make_study, next_trial
from conditional.spec import initialization, specification
from conditional.state import save_pickle, load_pickle


class SearchRecoveryTests(unittest.TestCase):
    def test_tpe_state_recovers_same_next_proposal(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        spec = specification(smoke=True)
        spec['initialization'] = 28; spec['adaptive_trials'] = 2
        rows = initialization('gnn')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            study, journal, path = make_study(directory, 'gnn', spec)
            for i in range(28):
                active = next_trial(study, journal, rows, spec, 'gnn')
                self.assertTrue(active['initialization'])
                study.tell(active['number'], float(i % 6))
            journal['sampler'] = study.sampler
            save_pickle(path, journal)
            # Preserve the exact completed study and RNG boundary before the next ask.
            import shutil
            alternate = directory / 'alternate'; alternate.mkdir()
            shutil.copy2(directory / 'study.db', alternate / 'study.db')
            shutil.copy2(path, alternate / 'journal.pkl')
            a = next_trial(study, journal, rows, spec, 'gnn')
            loaded, other, _ = make_study(alternate, 'gnn', spec)
            b = next_trial(loaded, other, rows, spec, 'gnn')
            self.assertEqual(a['parameters'], b['parameters'])
            self.assertEqual(a['number'], b['number'])
            self.assertFalse(a['initialization'])

    def test_failed_initial_slot_is_replaced_before_adaptive_search(self):
        spec = specification(smoke=True)
        rows = initialization('gnn')[:1]
        with tempfile.TemporaryDirectory() as tmp:
            study, journal, _ = make_study(Path(tmp), 'gnn', spec)
            first = next_trial(study, journal, rows, spec, 'gnn')
            study.tell(first['number'], state=TrialState.FAIL)
            retry = next_trial(study, journal, rows, spec, 'gnn')
            for key in ('representation', 'n_pca', 'n_harmonics', 'lookback'):
                self.assertEqual(first['parameters'].get(key), retry['parameters'].get(key))
            self.assertTrue(retry['initialization'])


if __name__ == '__main__':
    unittest.main()
