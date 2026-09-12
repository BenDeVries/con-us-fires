"""Epoch-boundary recovery including optimizer, RNG and early stopping."""
from __future__ import annotations

import dataclasses
from io import BytesIO
import random
import time

import numpy as np
from scipy.special import expit
import torch
from torch.utils.data import DataLoader

from model.config import Config
from model.data import WindowDataset
from model.model import SpatioTemporalZIB, build_norm_adj
from model.zib import zib_nll
from .metrics import row_nll
from .state import atomic_bytes, digest, read_json, write_json


def configuration(p, spec, seed, device):
    cfg = Config(drop_features=[], n_samples=0, link='logit', static_bypass=True,
                 horizon=12, seed=seed, device=device, batch_size=4,
                 max_epochs=spec['limits']['gnn'], patience=spec['patience']['gnn'])
    for key, value in p.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.n_pca = p.get('n_pca') if p['representation'] == 'pca' else None
    return cfg


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_state(path, state):
    out = BytesIO(); torch.save(state, out); atomic_bytes(path, out.getvalue())


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def batch_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def tf_ratio(cfg, epoch):
    if not cfg.teacher_forcing:
        return 0.
    return cfg.tf_ratio_start + (cfg.tf_ratio_end - cfg.tf_ratio_start) * min(
        (epoch - 1) / max(cfg.max_epochs - 1, 1), 1.)


@torch.no_grad()
def predict_model(model, panel, split, device):
    model.eval()
    dataset = WindowDataset(panel, panel.split_origins[split], model.cfg.lookback, 12)
    adjacency = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    result = {k: [] for k in ('p_occ', 'mu', 'phi')}
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        out = model(batch_device(batch, device), adjacency)
        result['p_occ'].append(expit(-out['pi_logit'].cpu().numpy().astype('float64')).ravel())
        for key in ('mu', 'phi'):
            result[key].append(out[key].cpu().numpy().astype('float64').ravel())
    return {k: np.concatenate(v) for k, v in result.items()}


def fit(inputs, p, spec, directory, seed, session, report=None, device='cuda'):
    directory.mkdir(parents=True, exist_ok=True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cfg = configuration(p, spec, seed, device)
    identity = digest({'spec': spec['sha256'], 'params': p, 'seed': seed, 'device': device})
    state_path = directory / 'last.pt'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    panel = inputs.panel(p)
    model = SpatioTemporalZIB(cfg, len(panel.cov_names), panel.n_lc_classes, panel.n_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    adjacency = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    train = WindowDataset(panel, panel.split_origins['train'], cfg.lookback, 12)
    truth = inputs.keys('test').y_true.to_numpy()
    state = {'identity': identity, 'epoch': 0, 'best_nll': None, 'bad': 0, 'best_epoch': None,
             'best_model': None, 'history': [], 'microbatch': 1, 'done': False}
    if state_path.exists():
        state = load_state(state_path)
        if state['identity'] != identity:
            raise ValueError('Neural checkpoint does not match experiment/configuration/device')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
    write_json(directory / 'config.json', {'parameters': p, 'config': dataclasses.asdict(cfg),
        'identity': identity, 'feature_names': panel.cov_names, 'feature_layout': spec['feature_layout']})
    if state['done']:
        save_state(directory / 'best.pt', state['best_model'])
        return state['best_nll']
    for epoch in range(state['epoch'] + 1, cfg.max_epochs + 1):
        t0 = time.monotonic(); model.train()
        # Shuffle effective batches once; gradient accumulation includes the final partial batch.
        loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0)
        total, count = 0., 0
        for batch in loader:
            size = len(batch['y']); opt.zero_grad(set_to_none=True)
            for start in range(0, size, state['microbatch']):
                micro = {k: v[start:start + state['microbatch']] for k, v in batch.items()}
                micro = batch_device(micro, device)
                out = model(micro, adjacency, tf_ratio=tf_ratio(cfg, epoch))
                loss = zib_nll(out['pi_logit'], out['mu'], out['phi'], micro['y'], link='logit')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite neural loss')
                weight = len(micro['y']) / size
                (loss * weight).backward()
                total += float(loss.detach()) * len(micro['y']); count += len(micro['y'])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite neural gradient')
            opt.step()
        pred = predict_model(model, panel, 'test', device)
        score = float(row_nll(truth, pred['p_occ'], pred['mu'], pred['phi']).mean())
        state['epoch'] = epoch
        state['history'].append({'epoch': epoch, 'train_nll': total/count,
                                 'test_nll': score, 'seconds': time.monotonic()-t0})
        if state['best_nll'] is None or score < state['best_nll']:
            state.update(best_nll=score, bad=0, best_epoch=epoch,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            state['bad'] += 1
        state['done'] = state['bad'] >= cfg.patience or epoch == cfg.max_epochs
        state.update(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     optimizer=opt.state_dict(), rng=rng_state())
        save_state(state_path, state)
        save_state(directory / 'best.pt', state['best_model'])
        write_json(directory / 'progress.json', {k: state[k] for k in
                   ('epoch', 'best_epoch', 'best_nll', 'bad', 'done', 'history', 'microbatch')})
        print(f'gnn {directory.name} epoch={epoch} test_nll={score:.7f} best={state["best_nll"]:.7f}', flush=True)
        if report is not None:
            report(epoch, score, state['done'])
        session.boundary()
        if state['done']:
            break
    return state['best_nll']


def predict(inputs, directory, split, device='cpu'):
    record = read_json(directory / 'config.json')
    p = record['parameters']; panel = inputs.panel(p)
    cfg = Config(**record['config'])
    model = SpatioTemporalZIB(cfg, len(panel.cov_names), panel.n_lc_classes, panel.n_nodes).to(device)
    if panel.cov_names != record['feature_names']:
        raise ValueError('Prediction features differ from fitted features')
    model.load_state_dict(load_state(directory / 'best.pt'))
    return predict_model(model, panel, split, device)
