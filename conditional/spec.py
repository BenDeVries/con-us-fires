"""Frozen search domain and balanced, reproducible fresh initialization."""
from __future__ import annotations

import importlib.metadata
import platform
import numpy as np

from .state import ROOT, digest, sha256

LOOKBACKS = list(range(12, 49, 3))
REPRESENTATIONS = [0, 1, 4, 10, 20, 39, 'raw']
SPACE = {
    'shared': {'representation': ['pca', 'raw'], 'n_pca': [0, 39], 'n_harmonics': [0, 6]},
    'gnn': {
        'lookback': LOOKBACKS, 'gcn_hidden': [8, 16, 32, 64, 128], 'gcn_layers': [1, 3],
        'gcn_dropout': [0., .3], 'lstm_hidden': [64, 128, 256, 384], 'lstm_layers': [1, 2],
        'lc_embed_dim': [4, 8, 16], 'county_embed_dim': [0, 32],
        'head_hidden': [0, 32, 64, 128], 'lr': [3e-4, 6e-3], 'weight_decay': [1e-6, 1e-3],
        'teacher_forcing': [False, True], 'tf_ratio_start': [.3, 1.], 'tf_ratio_end': [0., .3]},
    'xgb': {'eta': [.01, .15], 'max_depth': [3, 9], 'subsample': [.5, 1.],
            'colsample_bytree': [.4, 1.], 'min_child_weight': [5., 400.],
            'reg_lambda': [.1, 50.], 'max_delta_step': [.25, 4.],
            'n_spatial': [0, 3], 'n_temporal': [0, 47]},
}
CATEGORICAL = {'gcn_hidden', 'lstm_hidden', 'lc_embed_dim', 'head_hidden', 'teacher_forcing'}
LOG = {'lr', 'weight_decay', 'eta', 'min_child_weight', 'reg_lambda', 'max_delta_step'}
INTEGERS = {'gcn_layers', 'lstm_layers', 'county_embed_dim', 'max_depth', 'n_spatial', 'n_temporal'}


def suggest(trial, family):
    p = {'representation': trial.suggest_categorical('representation', ['pca', 'raw'])}
    if p['representation'] == 'pca':
        p['n_pca'] = trial.suggest_int('n_pca', 0, 39)
    p['n_harmonics'] = trial.suggest_int('n_harmonics', 0, 6)
    for key, bounds in SPACE[family].items():
        if key.startswith('tf_ratio') and not p['teacher_forcing']:
            continue
        if key == 'lookback':
            p[key] = trial.suggest_int(key, 12, 48, step=3)
        elif key in CATEGORICAL:
            p[key] = trial.suggest_categorical(key, bounds)
        elif key in INTEGERS:
            p[key] = trial.suggest_int(key, *bounds)
        else:
            p[key] = trial.suggest_float(key, *bounds, log=key in LOG)
    return p


def initialization(family, seed=20260908):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(28):
        r, block = i % 7, i // 7
        rep = REPRESENTATIONS[r]
        row = {'representation': 'raw' if rep == 'raw' else 'pca',
               'n_harmonics': (r + 2 * block) % 7}
        if rep != 'raw':
            row['n_pca'] = rep
        rows.append(row)
    for key, bounds in SPACE[family].items():
        if key == 'lookback' or key in CATEGORICAL:
            values = np.resize(np.asarray(bounds), 28)
            rng.shuffle(values)
        else:
            u = (rng.permutation(28) + rng.random(28)) / 28
            lo, hi = bounds
            values = (np.exp(np.log(lo) + u * np.log(hi / lo)) if key in LOG else
                      np.floor(lo + u * (hi - lo + 1)).astype(int) if key in INTEGERS else
                      lo + u * (hi - lo))
        for row, value in zip(rows, values):
            row[key] = value.item() if hasattr(value, 'item') else value
    for row in rows:
        if family == 'gnn' and not row['teacher_forcing']:
            row.pop('tf_ratio_start'); row.pop('tf_ratio_end')
    # Shuffle the balanced schedule, so an interrupted first session is not a dimension ladder.
    rng.shuffle(rows)
    return rows


def environment():
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'torch', 'xgboost', 'xgboostlss', 'optuna', 'pyarrow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'packages': versions}


def input_hashes():
    names = ['train.parquet', 'test.parquet', 'validation.parquet', 'feature_metadata.json',
             'feature_scaler.json', 'county_graph.npz', 'node_index.json']
    return {f'output/data/{name}': sha256(ROOT / 'output/data' / name) for name in names}


def code_hashes():
    files = list((ROOT / 'conditional').glob('*.py'))
    files += list((ROOT / 'reporting').glob('*.py'))
    files += [ROOT / p for p in ('model/config.py', 'model/model.py', 'model/data.py',
                                 'model/zib.py', 'model/spatial.py', 'model/xgb/features.py',
                                 'model/xgb/zabeta_hurdle.py', 'model/xgb/common.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def specification(smoke=False):
    value = {
        'schema_version': 1, 'feature_layout': 'conditional_dynamic_pca_v1',
        'information_set': 'observed target-period covariates; origin-available fire history',
        'horizon': 12, 'origin_lookback': 48, 'lookback_step': 3,
        'splits': {'train': ['2003-01-01', '2018-05-01'], 'test': ['2018-06-01', '2020-07-01'],
                   'validation': ['2020-08-01', '2024-12-01']},
        'space': SPACE, 'sampler_seed': 20260908, 'fit_seed': 0,
        'initialization': 28, 'adaptive_trials': 52,
        'limits': {'gnn': 80, 'xgb': 1500}, 'patience': {'gnn': 10, 'xgb': 50},
        'warmup': {'gnn': 15, 'xgb': 100}, 'batch_size': 4, 'xgb_nthread': 8,
        'links': {'gnn': 'logit; precision softplus + 1',
                  'xgb': 'sigmoid zero gate; softplus alpha; sigmoid beta capped at 3000'},
        'objective': 'mean test true-hurdle NLL', 'n_samples': 0,
        'tree_recovery': 'reset transient updater each round; global iteration seeds',
        'xgb_gate_bounds': [0.001, 0.999], 'xgb_beta_cap': 3000.,
        'finalists': 3, 'finalist_seeds': [0, 1, 2], 'published_seed': 0,
        'inputs': input_hashes(), 'code': code_hashes(), 'smoke': smoke,
    }
    if smoke:
        value.update(initialization=2, adaptive_trials=0, finalists=1, finalist_seeds=[0])
        value['limits'] = {'gnn': 2, 'xgb': 4}
        value['smoke_nodes'] = 16
        value['smoke_origins'] = 2
    value['sha256'] = digest(value)
    return value
