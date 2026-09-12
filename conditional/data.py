"""Shared, train-fitted conditional inputs and origin-anchored tree features."""
from __future__ import annotations

import dataclasses
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from model.data import PanelTensors, STATIC_COLS, TAIL_TRANSFORMS, rebuild_origins, WindowDataset
from model.spatial import exclusive_orders
from model.xgb.features import (_rolling_mean, _months_since_fire, _st_cells,
                                 AR_LAGS, ROLL_WINDOWS)
from .state import ROOT, atomic_bytes, read_json, write_json, sha256

SPLITS = ('train', 'test', 'validation')
KEYS = ['origin_date', 'horizon', 'target_date', 'county_fips', 'node_id']


def harmonic_block(month, order):
    if not isinstance(order, int) or not 0 <= order <= 6:
        raise ValueError('Harmonic order must be an integer in 0..6')
    names, values = [], []
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * (np.asarray(month) - 1) / 12
        if k != 6:
            names.append(f'month_sin{k}'); values.append(np.sin(angle))
        names.append(f'month_cos{k}'); values.append(np.cos(angle))
    return names, (np.stack(values, axis=-1).astype('float32') if values else
                   np.empty((len(month), 0), dtype='float32'))


def load_frames(spec):
    frames = []
    for split in SPLITS:
        df = pd.read_parquet(ROOT / 'output/data' / f'{split}.parquet')
        df['date'] = pd.to_datetime(df.date)
        if spec.get('smoke'):
            df = df[df.node_id < spec['smoke_nodes']].copy()
        if [str(df.date.min().date()), str(df.date.max().date())] != spec['splits'][split]:
            raise ValueError(f'Unexpected {split} dates')
        df['split'] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).sort_values(['date', 'node_id']).reset_index(drop=True)
    if full.duplicated(['date', 'node_id']).any():
        raise ValueError('Duplicate county-month')
    node_index = read_json(ROOT / 'output/data/node_index.json')
    fips = full.county_fips.astype(str).str.zfill(5)
    if not np.array_equal(fips.map(node_index).to_numpy(), full.node_id.to_numpy()):
        raise ValueError('County FIPS do not match the frozen graph node index')
    dates = np.sort(full.date.unique())
    n = full.node_id.nunique()
    if not np.array_equal(full.node_id.unique(), np.arange(n)) or len(full) != n * len(dates):
        raise ValueError('Panel must be rectangular with contiguous node IDs')
    expected = pd.date_range(dates[0], dates[-1], freq='MS').to_numpy()
    if not np.array_equal(dates, expected):
        raise ValueError('Panel has missing months')
    y = full.burned_fraction.to_numpy()
    if not np.isfinite(y).all() or not ((y >= 0) & (y < 1)).all():
        raise ValueError('Missing/invalid outcome: no response imputation or clipping')
    if not np.array_equal(full.fire_occurred.to_numpy(), y > 0):
        raise ValueError('Occurrence does not match observed burned fraction')
    return full, dates, n


def fit_transform(full, names, static, scaler):
    train = full.split.eq('train').to_numpy()
    x = full[names].to_numpy(dtype='float64', copy=True)
    if not np.isfinite(x).all():
        raise ValueError('Prepared predictors must be finite')
    tail = {}
    for j, name in enumerate(names):
        if name not in static or name not in TAIL_TRANSFORMS:
            continue
        raw = np.maximum(x[:, j] * scaler[name]['std'] + scaler[name]['mean'], 0)
        value = np.log1p(raw) if TAIL_TRANSFORMS[name] == 'log1p' else np.sqrt(raw)
        mean, sd = float(value[train].mean()), float(value[train].std())
        tail[name] = [mean, sd or 1.]
    d = len(names) - len(static)
    x_train = x[train, :d]
    mean = x_train.mean(axis=0)
    covariance = (x_train - mean).T @ (x_train - mean) / (len(x_train) - 1)
    eigen, rotation = np.linalg.eigh(covariance)
    eigen, rotation = eigen[::-1], rotation[:, ::-1]
    anchor = np.argmax(np.abs(rotation), axis=0)
    rotation *= np.where(rotation[anchor, np.arange(d)] < 0, -1., 1.)
    return {'mean': mean, 'rotation': rotation, 'eigenvalues': eigen}, {
        'layout': 'conditional_dynamic_pca_v1', 'names': names, 'static_names': static,
        'dynamic_names': names[:d], 'tail': tail, 'source_scaler': scaler,
        'categories': sorted(int(v) for v in full.loc[train, 'lc_dominant'].unique()),
        'unknown_category': -1, 'pca_whiten': False,
    }


def apply_transform(full, meta, arrays, pca):
    names = meta['names']
    x = full[names].to_numpy(dtype='float64', copy=True)
    for name, (mean, sd) in meta['tail'].items():
        j = names.index(name); scaler = meta['source_scaler'][name]
        raw = np.maximum(x[:, j] * scaler['std'] + scaler['mean'], 0)
        value = np.log1p(raw) if TAIL_TRANSFORMS[name] == 'log1p' else np.sqrt(raw)
        x[:, j] = (value - mean) / sd
    if pca is not None:
        if not isinstance(pca, int) or not 0 <= pca <= len(meta['dynamic_names']):
            raise ValueError('PC count outside fitted dynamic feature domain')
        d = len(meta['dynamic_names'])
        x = np.concatenate(((x[:, :d] - arrays['mean']) @ arrays['rotation'][:, :pca],
                            x[:, d:]), axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed input')
    return x.astype('float32')


class Inputs:
    def __init__(self, experiment, spec, fit=False):
        self.spec = spec
        full, dates, n = load_frames(spec)
        self.full, self.dates, self.n = full, dates, n
        directory = Path(experiment) / 'preprocessing'
        if fit:
            metadata = read_json(ROOT / 'output/data/feature_metadata.json')
            names = sorted(k for k, v in metadata.items() if v.get('role') == 'predictor'
                           and k != 'lc_dominant')
            static = [c for c in names if c in STATIC_COLS]
            names = [c for c in names if c not in static] + static
            if len(names) != 51 or len(static) != 12:
                raise ValueError('Expected 39 dynamic and 12 bypass predictors')
            scaler = {row['feature']: row for row in read_json(ROOT / 'output/data/feature_scaler.json')}
            arrays, meta = fit_transform(full, names, static, scaler)
            stream = BytesIO(); np.savez(stream, **arrays)
            atomic_bytes(directory / 'transform.npz', stream.getvalue())
            meta['transform_sha256'] = sha256(directory / 'transform.npz')
            write_json(directory / 'transform.json', meta)
        self.meta = read_json(directory / 'transform.json')
        if sha256(directory / 'transform.npz') != self.meta['transform_sha256']:
            raise ValueError('Preprocessing artifact hash mismatch')
        with np.load(directory / 'transform.npz') as z:
            self.arrays = dict(z)
        self.raw = apply_transform(full, self.meta, self.arrays, None).reshape(len(dates), n, -1)
        self.pcs = ((self.raw[..., :39].astype('float64') - self.arrays['mean']) @
                    self.arrays['rotation']).astype('float32')
        self.y = full.burned_fraction.to_numpy(dtype='float32', copy=True).reshape(-1, n)
        self.occ = (self.y > 0).astype('float32')
        mapping = {v: i + 1 for i, v in enumerate(self.meta['categories'])}
        self.cat = full.lc_dominant.map(mapping).fillna(0).to_numpy(dtype='int64', copy=True).reshape(-1, n)
        self.fips = full.iloc[:n].county_fips.astype(str).str.zfill(5).to_numpy()
        edge = np.load(ROOT / 'output/data/county_graph.npz')['edge_index']
        edge = edge[:, (edge < n).all(axis=0)]
        self.edge = edge
        self.origins = rebuild_origins(dates, spec['splits'], 48, 12)
        if spec.get('smoke'):
            self.origins = {s: v[:spec['smoke_origins']] for s, v in self.origins.items()}
        # Distinct history grids are tiny relative to assembled tree designs.
        adj = sp.coo_matrix((np.ones(edge.shape[1]), tuple(edge)), shape=(n, n)).tocsr()
        adj = adj.maximum(adj.T); adj.setdiag(0); adj.eliminate_zeros()
        self.adj = sp.diags(1 / np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)) @ adj
        self.roll = {w: _rolling_mean(self.y, w) for w in ROLL_WINDOWS}
        self.occ12 = _rolling_mean(self.occ, 12)
        self.msf = _months_since_fire(self.occ)
        self.nb_roll = (self.adj @ self.roll[12].T).T
        self.nb_occ = (self.adj @ self.occ12.T).T
        self.nb_y = (self.adj @ self.y.T).T
        denom = np.maximum(np.arange(len(dates)), 1)[:, None]
        self.county_y = np.vstack([np.zeros((1, n)), np.cumsum(self.y[:-1], axis=0, dtype='float64')]) / denom
        self.county_occ = np.vstack([np.zeros((1, n)), np.cumsum(self.occ[:-1], axis=0, dtype='float64')]) / denom
        shells = exclusive_orders(torch.from_numpy(edge), n, 3, sparse=True)
        self.st = {(c, s): (values if s == 0 else (shells[s] @ values.T).T).astype('float32')
                   for c, values in [('y', self.y), ('occ', self.occ)] for s in range(4)}
        del self.full

    def covariates(self, p):
        k = p.get('n_pca') if p['representation'] == 'pca' else None
        if k is not None and (not isinstance(k, int) or not 0 <= k <= 39):
            raise ValueError('Invalid n_pca')
        x = self.raw if k is None else np.concatenate([self.pcs[..., :k], self.raw[..., 39:]], axis=-1)
        names = self.meta['names'] if k is None else [f'PC{i+1}' for i in range(k)] + self.meta['static_names']
        hn, h = harmonic_block(pd.DatetimeIndex(self.dates).month, p['n_harmonics'])
        return np.concatenate([x, np.broadcast_to(h[:, None, :], (len(h), self.n, len(hn)))], axis=-1), names + hn

    def panel(self, p):
        cov, names = self.covariates(p)
        return PanelTensors(cov=torch.from_numpy(cov), cat=torch.from_numpy(self.cat),
            ar=torch.from_numpy(np.stack([self.y, self.occ], axis=-1)), y=torch.from_numpy(self.y),
            dates=self.dates, edge_index=torch.from_numpy(self.edge), n_nodes=self.n,
            cov_names=names, n_lc_classes=len(self.meta['categories']) + 1,
            split_origins=self.origins, node_fips=self.fips, split_date_ranges=self.spec['splits'])

    def keys(self, split):
        rows = []
        for o in self.origins[split]:
            for h in range(1, 13):
                rows.append(pd.DataFrame({'origin_date': self.dates[o], 'horizon': h,
                    'target_date': self.dates[o+h], 'county_fips': self.fips,
                    'node_id': np.arange(self.n), 'y_true': self.y[o+h]}))
        return pd.concat(rows, ignore_index=True)

    def tree_design(self, p, split):
        cov, names = self.covariates(p)
        s, t = p['n_spatial'], p['n_temporal']
        if not 0 <= s <= 3 or not 0 <= t <= 47:
            raise ValueError('History exceeds common origin allowance')
        cells = list(_st_cells(s, t)) if s or t else []
        hist_names = ['y_o', 'occ_o'] + [f'y_lag{k}' for k in AR_LAGS] + [f'y_roll{w}' for w in ROLL_WINDOWS]
        hist_names += ['occ_rate12', 'months_since_fire', 'y_seasonal', 'nb_y_roll12',
                       'nb_occ_rate12', 'nb_y_seasonal', 'county_mean_y', 'county_occ_rate']
        names = names + hist_names + [f'st_{c}_s{s}_t{t}' for c, s, t in cells] + ['horizon']
        x = np.empty((len(self.origins[split]) * 12 * self.n, len(names)), dtype='float32')
        cat = np.empty(len(x), dtype='int64'); labels = np.empty(len(x), dtype='float32')
        r = 0
        for o in self.origins[split]:
            for h in range(1, 13):
                m = o+h
                history = [self.y[o], self.occ[o], *[self.y[o-k] for k in AR_LAGS],
                    *[self.roll[w][o] for w in ROLL_WINDOWS], self.occ12[o], self.msf[o],
                    self.y[m-12], self.nb_roll[o], self.nb_occ[o], self.nb_y[m-12],
                    self.county_y[o], self.county_occ[o],
                    *[self.st[c, shell][o-lag] for c, shell, lag in cells], np.full(self.n, h)]
                x[r:r+self.n] = np.column_stack([cov[m], *history])
                cat[r:r+self.n] = self.cat[m]; labels[r:r+self.n] = self.y[m]
                r += self.n
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite tree features')
        frame = pd.DataFrame(x, columns=names)
        frame['lc_dominant'] = pd.Categorical(cat, categories=range(len(self.meta['categories']) + 1))
        return frame, labels
