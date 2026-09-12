"""CLI: prepare, run/resume, status, select, predict, report, package, refit."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .state import ROOT, Session, read_json, write_json, sha256, digest


def selections(experiment):
    spec = read_json(experiment / 'spec.json')
    result = {}
    for family in ('xgb', 'gnn'):
        record = read_json(experiment / family / 'selection.json')
        if record['experiment_sha256'] != spec['sha256']:
            raise ValueError('Selection belongs to a different experiment')
        for name, expected in record['files'].items():
            if sha256(experiment / family / 'selected' / name) != expected:
                raise ValueError(f'Selected {family} artifact changed: {name}')
        result[family] = record
    return result


def freeze(experiment):
    records = selections(experiment)
    path = experiment / 'assessment.json'
    frozen = {'selections': {k: digest(v) for k, v in records.items()},
              'rule': 'Both representatives frozen before validation; assessment cannot select replacements.'}
    if path.exists() and read_json(path) != frozen:
        raise ValueError('Representatives changed after assessment was frozen')
    if not path.exists():
        write_json(path, frozen)
    return frozen


def predict(experiment, family, splits, device):
    import numpy as np
    import torch
    from .data import Inputs
    from .search import verify
    from .metrics import row_nll
    spec = verify(experiment)
    freeze(experiment)
    inputs = Inputs(experiment, spec)
    torch.set_num_threads(2 if family == 'gnn' else spec['xgb_nthread'])
    directory = experiment / family / 'selected'
    trainer = __import__(f'conditional.{"neural" if family == "gnn" else "trees"}', fromlist=['predict'])
    for split in splits:
        kwargs = {'device': device} if family == 'gnn' else {}
        pred = trainer.predict(inputs, directory, split, **kwargs)
        frame = inputs.keys(split)
        for key in ('p_occ', 'mu', 'phi'):
            frame[key] = pred[key]
        frame['e_y'] = frame.p_occ * frame.mu
        nll = float(row_nll(frame.y_true, frame.p_occ, frame.mu, frame.phi).mean())
        if split == 'test':
            expected = read_json(experiment / family / 'selection.json')['seed0_test_nll']
            if not np.isclose(nll, expected, rtol=0, atol=1e-6):
                raise ValueError(f'Prediction NLL {nll} differs from selected fit {expected}')
        path = directory / f'predictions_{split}.parquet'
        tmp = path.with_suffix('.partial.parquet'); frame.to_parquet(tmp, index=False); tmp.replace(path)
        write_json(directory / f'predictions_{split}.json', {'sha256': sha256(path), 'rows': len(frame),
            'mean_nll': nll, 'selection_sha256': digest(read_json(experiment / family / 'selection.json'))})
        print(f'{family}/{split}: {len(frame):,} rows, NLL={nll:.7f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='conditional_v1', help='directory name below output/')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--smoke', action='store_true')
    for name in ('run', 'select', 'refit'):
        p = sub.add_parser(name); p.add_argument('--model', choices=['gnn', 'xgb'], required=True)
        p.add_argument('--hours', type=float, default=12); p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
        if name == 'run':
            p.add_argument('--pause-after', type=int, help='test recovery after this many epoch/round boundaries')
    sub.add_parser('status')
    pred = sub.add_parser('predict'); pred.add_argument('--model', choices=['gnn', 'xgb'], required=True)
    pred.add_argument('--split', choices=['train', 'test', 'validation', 'all'], default='all')
    pred.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    report = sub.add_parser('report'); report.add_argument('--publish', action='store_true')
    report.add_argument('--reuse-stacf', action='store_true')
    package = sub.add_parser('package'); package.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in ('.', '..'):
        parser.error('experiment must be a single directory name')
    experiment = ROOT / 'output' / args.experiment
    if args.command == 'prepare':
        from .spec import specification
        from .data import Inputs
        if (experiment / 'spec.json').exists():
            from .search import verify
            spec = verify(experiment)
            if spec['smoke'] != args.smoke:
                raise ValueError('Smoke and full studies cannot share an experiment')
            print('Existing compatible experiment; use run to continue.'); return
        spec = specification(args.smoke)
        inputs = Inputs(experiment, spec, fit=True)
        write_json(experiment / 'preprocessing/manifest.json', {
            name: sha256(experiment / 'preprocessing' / name) for name in ('transform.json', 'transform.npz')})
        write_json(experiment / 'spec.json', spec)
        from .spec import initialization
        for family in ('gnn', 'xgb'):
            write_json(experiment / family / 'initialization.json', initialization(family, spec['sampler_seed']))
        print(f'Prepared {experiment}: {inputs.n} counties; origins { {k: len(v) for k,v in inputs.origins.items()} }')
    elif args.command == 'status':
        for family in ('gnn', 'xgb'):
            directory = experiment / family
            path = directory / 'status.json'
            if not path.exists():
                print(f'{family}: no session boundary recorded yet')
            else:
                import json
                print(f'{family}: {json.dumps(read_json(path), indent=2)}')
            # Live checkpoint progress remains available while a session is running.
            for path in sorted(directory.glob('trial_*/progress.json')):
                info = read_json(path)
                if not info['done']:
                    print(path.parent.name, {k: v for k, v in info.items() if k != 'history'})
    elif args.command in ('run', 'select'):
        from . import search
        kwargs = {'boundary_limit': args.pause_after} if args.command == 'run' else {}
        getattr(search, args.command)(experiment, args.model, args.hours, args.device, **kwargs)
    elif args.command == 'predict':
        predict(experiment, args.model, ('train', 'test', 'validation') if args.split == 'all' else (args.split,), args.device)
    elif args.command == 'report':
        from .report import build
        build(experiment, publish=args.publish, reuse_stacf=args.reuse_stacf)
    elif args.command == 'package':
        from .release import package
        package(experiment, args.destination)
    elif args.command == 'refit':
        from .data import Inputs
        from .search import verify
        spec = verify(experiment, args.model)
        record = read_json(experiment / args.model / 'selection.json')
        inputs = Inputs(experiment, spec)
        trainer = __import__(f'conditional.{"neural" if args.model == "gnn" else "trees"}', fromlist=['fit'])
        session = Session(args.hours); session.install_signals()
        kwargs = {'device': args.device} if args.model == 'gnn' else {}
        from .state import Paused
        try:
            trainer.fit(inputs, record['parameters'], spec, experiment / 'refits' / args.model,
                        0, session, **kwargs)
        except Paused as e:
            print(e)


if __name__ == '__main__':
    main()
