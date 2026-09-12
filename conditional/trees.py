"""Round-boundary recovery using global iteration seeds and native booster state."""
from __future__ import annotations

import time
import numpy as np
import torch
import xgboost as xgb
from xgboostlss.model import XGBoostLSS
from model.xgb.zabeta_hurdle import ZABetaHurdle

from .metrics import row_nll
from .state import digest, load_pickle, save_pickle, read_json, write_json


def parameters(margin, dist):
    margin = np.asarray(margin, dtype='float64').reshape(-1, 3)
    if not np.isfinite(margin).all():
        raise FloatingPointError('Nonfinite boosting margin')
    with torch.no_grad():
        a, b, gate = [fn(torch.from_numpy(margin[:, i])).numpy()
                      for i, fn in enumerate(dist.param_dict.values())]
    return {'p_occ': 1-gate, 'mu': a/(a+b), 'phi': a+b}


def matrix(inputs, p, split, nthread):
    frame, y = inputs.tree_design(p, split)
    return xgb.DMatrix(frame, label=y, enable_categorical=True, nthread=nthread)


def fit(inputs, p, spec, directory, seed, session, report=None):
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest({'spec': spec['sha256'], 'params': p, 'seed': seed})
    saved = directory / 'last.pkl'
    state = load_pickle(saved) if saved.exists() else None
    if state and state['identity'] != identity:
        raise ValueError('Tree checkpoint does not match experiment/configuration')
    if state and state['done']:
        save_pickle(directory / 'best.pkl', {'booster': state['best_booster'],
                    'start_values': state['start_values']})
        return state['best_nll']
    dtrain = matrix(inputs, p, 'train', spec['xgb_nthread'])
    dtest = matrix(inputs, p, 'test', spec['xgb_nthread'])
    lss = XGBoostLSS(ZABetaHurdle())
    if state:
        lss.start_values = state['start_values']
    lss.set_base_margin(dtrain); lss.set_base_margin(dtest)
    params = {k: v for k, v in p.items() if k in spec['space']['xgb'] and not k.startswith('n_')}
    params['lambda'] = params.pop('reg_lambda')
    params.update(tree_method='hist', device='cpu', nthread=spec['xgb_nthread'],
                  seed=seed, seed_per_iteration=True)
    lss.set_params_adj(params)
    params['base_score'] = '[0,0,0]'
    booster = state['snapshot'] if state else xgb.Booster(params, [dtrain, dtest])
    # load_model restores serialized settings; explicitly reapply the frozen settings.
    booster.set_param(params)
    if state is None:
        state = {'identity': identity, 'iteration': 0, 'best_nll': None, 'bad': 0,
                 'best_iteration': None, 'history': [], 'start_values': lss.start_values,
                 'done': False}
    write_json(directory / 'config.json', {'parameters': p, 'booster_params': params,
        'seed': seed, 'identity': identity, 'feature_names': dtrain.feature_names,
        'feature_layout': spec['feature_layout']})
    limit = spec['limits']['xgb']
    for iteration in range(state['iteration'], limit):
        t0 = time.monotonic()
        # Reset transient native updater/sampler state at every round in both fresh
        # and resumed fits. XGBoost does not serialize its column sampler RNG.
        # Global iteration seeds then determine the same draws after any restart.
        booster.reset()
        booster.update(dtrain, iteration, fobj=lss.dist.objective_fn)
        pred = parameters(booster.predict(dtest, output_margin=True), lss.dist)
        score = float(row_nll(dtest.get_label(), pred['p_occ'], pred['mu'], pred['phi']).mean())
        state['iteration'] = iteration + 1
        state['history'].append({'iteration': iteration+1, 'test_nll': score,
                                 'seconds': time.monotonic()-t0})
        raw = booster.save_raw(raw_format='ubj')
        if state['best_nll'] is None or score < state['best_nll']:
            state.update(best_nll=score, bad=0, best_iteration=iteration+1, best_booster=raw)
        else:
            state['bad'] += 1
        state.update(booster=raw, snapshot=booster, done=state['bad'] >= spec['patience']['xgb'] or iteration+1 == limit)
        save_pickle(saved, state)
        # Inference artifact has no post-optimum trees or training callback state.
        save_pickle(directory / 'best.pkl', {'booster': state['best_booster'],
                    'start_values': state['start_values']})
        write_json(directory / 'progress.json', {k: state[k] for k in
                   ('iteration', 'best_iteration', 'best_nll', 'bad', 'done', 'history')})
        if (iteration+1) % 10 == 0 or state['done'] or iteration == 0:
            print(f'xgb {directory.name} round={iteration+1} test_nll={score:.7f} best={state["best_nll"]:.7f}', flush=True)
        if report is not None:
            report(iteration+1, score, state['done'])
        session.boundary()
        if state['done']:
            break
    return state['best_nll']


def predict(inputs, directory, split):
    record = read_json(directory / 'config.json'); state = load_pickle(directory / 'best.pkl')
    dm = matrix(inputs, record['parameters'], split, record['booster_params']['nthread'])
    if dm.feature_names != record['feature_names']:
        raise ValueError('Prediction features differ from fitted features')
    dm.set_base_margin(np.tile(state['start_values'], (dm.num_row(), 1)).ravel())
    booster = xgb.Booster(model_file=state['booster'])
    return parameters(booster.predict(dm, output_margin=True), ZABetaHurdle())
