"""Single-worker ask/tell studies with a durable active-trial journal."""
from __future__ import annotations

import fcntl
import gc
import time
import traceback

import optuna
from optuna.trial import TrialState
import torch

from .data import Inputs
from .spec import initialization, suggest, environment, code_hashes, input_hashes
from .state import read_json, write_json, save_pickle, load_pickle, Session, Paused


def verify(experiment, family=None):
    spec = read_json(experiment / 'spec.json')
    from .state import digest, sha256
    recorded = spec.pop('sha256')
    if digest(spec) != recorded:
        raise ValueError('Experiment specification changed')
    spec['sha256'] = recorded
    if spec['inputs'] != input_hashes() or spec['code'] != code_hashes():
        raise ValueError('Inputs or training source changed: create a new experiment')
    prep = read_json(experiment / 'preprocessing/manifest.json')
    for name, value in prep.items():
        if sha256(experiment / 'preprocessing' / name) != value:
            raise ValueError('Preprocessing changed since experiment preparation')
    if family:
        path = experiment / family / 'environment.json'
        current = environment()
        if path.exists() and read_json(path) != current:
            raise ValueError('Training environment changed: resume in the recorded environment')
        if not path.exists():
            write_json(path, current)
    return spec


def make_study(directory, family, spec):
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / 'journal.pkl'
    journal = load_pickle(journal_path) if journal_path.exists() else {
        'active': None, 'elapsed_seconds': 0., 'adaptive_started': 0,
        'sampler': optuna.samplers.TPESampler(seed=spec['sampler_seed'], n_startup_trials=0,
            multivariate=True, group=True), 'finished': False}
    study = optuna.create_study(storage=f'sqlite:///{directory / "study.db"}',
        study_name=f'conditional_{family}', direction='minimize', load_if_exists=True,
        sampler=journal['sampler'], pruner=optuna.pruners.MedianPruner(
            n_startup_trials=spec['initialization'], n_warmup_steps=spec['warmup'][family],
            n_min_trials=5))
    return study, journal, journal_path


def initial_rows(family, spec):
    rows = initialization(family, spec['sampler_seed'])
    if spec['smoke']:
        rows = rows[:2]
        rows[0].update(representation='pca', n_pca=0, n_harmonics=0)
        rows[1].update(representation='raw', n_harmonics=6)
        rows[1].pop('n_pca', None)
        for row in rows:
            if family == 'gnn':
                row.update(gcn_hidden=8, lstm_hidden=64, gcn_layers=1, lstm_layers=1,
                           head_hidden=0, lookback=15)
            else:
                row.update(max_depth=3, n_spatial=1, n_temporal=3)
    return rows


def next_trial(study, journal, rows, spec, family):
    completed = {t.user_attrs['initial_slot'] for t in study.trials if t.state == TrialState.COMPLETE
                 and 'initial_slot' in t.user_attrs}
    missing = [i for i in range(len(rows)) if i not in completed]
    if missing:
        slot = missing[0]
        attempts = [t for t in study.trials if t.user_attrs.get('initial_slot') == slot]
        row = dict(rows[slot])
        # Failed structures retain their mandated representation/harmonic/lookback coverage.
        # A second draw uses smaller compute dimensions, not a different feature stratum.
        if attempts and family == 'gnn':
            row.update(gcn_hidden=16, lstm_hidden=64, gcn_layers=1, lstm_layers=1, head_hidden=0)
        elif attempts:
            row.update(max_depth=3, eta=.01, max_delta_step=.25)
        if len(attempts) >= 3:
            raise RuntimeError(f'Initialization slot {slot} failed three times; inspect failures before continuing')
        # Reuse a WAITING proposal if interruption occurred between enqueue and ask.
        if not any(t.state == TrialState.WAITING for t in study.trials):
            study.enqueue_trial(row, user_attrs={'initial_slot': slot})
    elif journal['adaptive_started'] >= spec['adaptive_trials']:
        return None
    trial = study.ask()
    if not missing:
        trial.set_user_attr('adaptive', True)
        journal['adaptive_started'] += 1
    p = suggest(trial, family)
    return {'trial_id': trial._trial_id, 'number': trial.number, 'parameters': p,
            'initialization': bool(missing)}


def leaderboard(study, directory):
    rows = [{'trial': t.number, 'state': t.state.name, 'test_nll': t.value,
             'parameters': t.params, 'attributes': t.user_attrs,
             'seconds': t.duration.total_seconds() if t.duration else None}
            for t in study.trials]
    write_json(directory / 'leaderboard.json', rows)


def run(experiment, family, hours=12, device='cuda', boundary_limit=None):
    directory = experiment / family
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'{family} already has a worker')
        spec = verify(experiment, family)
        torch.set_num_threads(2 if family == 'gnn' else spec['xgb_nthread'])
        session = Session(hours, boundary_limit); session.install_signals()
        study, journal, journal_path = make_study(directory, family, spec)
        rows = initial_rows(family, spec)
        inputs = Inputs(experiment, spec)
        trainer = __import__(f'conditional.{"neural" if family == "gnn" else "trees"}', fromlist=['fit'])
        # Recover an ask interrupted before the journal commit. Already assigned parameter
        # values live in SQLite; complete its proposal before scheduling any other fit.
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        if journal['active'] is None and running:
            if len(running) != 1:
                raise RuntimeError('Ambiguous running trials; expected one worker')
            trial = optuna.trial.Trial(study, running[0]._trial_id)
            journal['active'] = {'trial_id': trial._trial_id, 'number': trial.number,
                'parameters': suggest(trial, family), 'initialization': 'initial_slot' in trial.user_attrs}
        journal['adaptive_started'] = sum(t.user_attrs.get('adaptive', False) for t in study.trials)
        try:
            while not session.expired():
                if journal['active'] is None:
                    journal['active'] = next_trial(study, journal, rows, spec, family)
                    journal['sampler'] = study.sampler
                    save_pickle(journal_path, journal)
                active = journal['active']
                if active is None:
                    journal['finished'] = True
                    break
                trial = optuna.trial.Trial(study, active['trial_id'])
                frozen = study._storage.get_trial(active['trial_id'])
                if frozen.state.is_finished():
                    journal['active'] = None
                    continue
                path = directory / f'trial_{active["number"]:04d}'
                def report(step, score, done):
                    if step not in trial._get_latest_trial().intermediate_values:
                        trial.report(score, step)
                    if not active['initialization'] and not done and trial.should_prune():
                        raise optuna.TrialPruned(f'Median test NLL at step {step}')
                print(f'{family} trial={trial.number} initialization={active["initialization"]} '
                      f'params={active["parameters"]}', flush=True)
                try:
                    kwargs = {'device': device} if family == 'gnn' else {}
                    score = trainer.fit(inputs, active['parameters'], spec, path, 0,
                                        session, report=report, **kwargs)
                    study.tell(trial, score)
                except optuna.TrialPruned as e:
                    trial.set_user_attr('reason', str(e)); study.tell(trial, state=TrialState.PRUNED)
                except (torch.cuda.OutOfMemoryError, FloatingPointError) as e:
                    trial.set_user_attr('reason', type(e).__name__ + ': ' + str(e))
                    write_json(path / 'failure.json', {'type': type(e).__name__, 'message': str(e)})
                    study.tell(trial, state=TrialState.FAIL)
                journal['active'] = None
                journal['sampler'] = study.sampler
                save_pickle(journal_path, journal); leaderboard(study, directory)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Paused as e:
            print(str(e), flush=True)
        finally:
            journal['elapsed_seconds'] += time.monotonic() - session.started
            journal['sampler'] = study.sampler
            save_pickle(journal_path, journal); leaderboard(study, directory)
            write_json(directory / 'status.json', {
                'finished': journal['finished'], 'active_trial': journal['active'],
                'elapsed_seconds': journal['elapsed_seconds'],
                'completed': sum(t.state == TrialState.COMPLETE for t in study.trials),
                'initial_completed': len({t.user_attrs['initial_slot'] for t in study.trials
                    if t.state == TrialState.COMPLETE and 'initial_slot' in t.user_attrs}),
                'initial_required': len(rows), 'adaptive_started': journal['adaptive_started'],
                'adaptive_required': spec['adaptive_trials']})


def select(experiment, family, hours=12, device='cuda'):
    import shutil
    import numpy as np
    directory = experiment / family
    with (directory / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = verify(experiment, family)
        if (experiment / 'assessment.json').exists():
            from .__main__ import freeze
            freeze(experiment)
            print(f'{family}: existing representative is already frozen for assessment', flush=True)
            return
        if not read_json(directory / 'status.json')['finished']:
            raise ValueError('Complete initialization and the adaptive search before selection')
        table = read_json(directory / 'leaderboard.json')
        candidates, seen = [], set()
        from .state import digest, sha256
        for row in sorted((r for r in table if r['state'] == 'COMPLETE'), key=lambda r: r['test_nll']):
            identity = digest(row['parameters'])
            if identity not in seen:
                candidates.append(row); seen.add(identity)
            if len(candidates) == spec['finalists']:
                break
        session = Session(hours); session.install_signals()
        torch.set_num_threads(2 if family == 'gnn' else spec['xgb_nthread'])
        inputs = Inputs(experiment, spec)
        trainer = __import__(f'conditional.{"neural" if family == "gnn" else "trees"}', fromlist=['fit'])
        evidence = []
        try:
            for row in candidates:
                values = []
                for seed in spec['finalist_seeds']:
                    if seed == 0:
                        score = row['test_nll']
                    else:
                        path = directory / 'finalists' / f'trial_{row["trial"]:04d}_seed_{seed}'
                        kwargs = {'device': device} if family == 'gnn' else {}
                        score = trainer.fit(inputs, row['parameters'], spec, path, seed, session, **kwargs)
                    values.append(score)
                evidence.append({'trial': row['trial'], 'parameters': row['parameters'],
                    'seeds': spec['finalist_seeds'], 'test_nll': values,
                    'mean_test_nll': float(np.mean(values)),
                    'sd_test_nll': float(np.std(values, ddof=1)) if len(values) > 1 else None})
                write_json(directory / 'finalist_evidence.json', evidence)
        except Paused as e:
            print(e, flush=True); return
        best = min(evidence, key=lambda r: (r['mean_test_nll'], r['trial']))
        source = directory / f'trial_{best["trial"]:04d}'
        destination = directory / 'selected'; destination.mkdir(exist_ok=True)
        checkpoint = 'best.pt' if family == 'gnn' else 'best.pkl'
        for name in ('config.json', 'progress.json', checkpoint):
            shutil.copy2(source / name, destination / name)
        record = {'experiment_sha256': spec['sha256'], 'family': family, 'trial': best['trial'],
            'parameters': best['parameters'], 'seed': 0, 'selection': 'minimum mean test NLL across finalist seeds',
            'mean_test_nll': best['mean_test_nll'], 'seed0_test_nll': best['test_nll'][0],
            'files': {name: sha256(destination / name) for name in ('config.json', 'progress.json', checkpoint)}}
        write_json(directory / 'selection.json', record)
        print(f'{family} selected trial {best["trial"]}: mean test NLL {best["mean_test_nll"]:.7f}', flush=True)
