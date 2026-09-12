"""Resume XGBoost with an explicit, recorded CPU-resource override.

The conditional experiment's original source/data checks still run unchanged.
Only the fitting call's CPU thread count is overlaid; checkpoint and Optuna
identities, feature choices, seeds, limits, and selection rules stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def thread_override(threads):
    """Adapt the existing trainer at its supported spec argument, after verification."""
    if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError('Thread count must be within the available logical CPU count')
    import torch
    from conditional import trees
    from conditional.state import read_json, write_json
    original = trees.fit

    def fit(inputs, p, spec, directory, seed, session, report=None):
        effective = dict(spec, xgb_nthread=threads)
        torch.set_num_threads(threads)
        directory.mkdir(parents=True, exist_ok=True)
        progress = directory / 'progress.json'
        history_path = directory / 'resource_history.json'
        history = read_json(history_path) if history_path.exists() else []
        config_path = directory / 'config.json'
        previous = (read_json(config_path)['booster_params']['nthread'] if config_path.exists()
                    else spec['xgb_nthread'])
        if not history or history[-1]['threads'] != threads:
            history.append({'time_utc': datetime.now(timezone.utc).isoformat(),
                'threads': threads, 'previous_threads': previous,
                'after_iteration': read_json(progress)['iteration'] if progress.exists() else 0,
                'reason': 'User-requested CPU resource setting; original experiment identity retained.'})
            write_json(history_path, history)
        print(f'XGBoost CPU override: native/gradient threads={threads}; '
              f'resume after round {read_json(progress)["iteration"] if progress.exists() else 0}', flush=True)
        return original(inputs, p, effective, directory, seed, session, report=report)

    trees.fit = fit
    try:
        yield
    finally:
        trees.fit = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='conditional_v1')
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--original-supervisor', type=int)
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.hours <= 0:
        parser.error('Use a directory name and a positive session duration')
    # Set pools before importing numpy/torch. OpenBLAS remains small for panel preparation.
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    from conditional import search
    from conditional.state import read_json, write_json, sha256
    directory = ROOT / 'output' / args.experiment
    deadline = time.monotonic() + args.hours * 3600
    write_json(directory / 'xgb/runtime_resources.json', {
        'threads': args.threads, 'pid': os.getpid(), 'hours': args.hours,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'entrypoint': 'tools/run_xgb.py', 'entrypoint_sha256': sha256(__file__),
        'original_supervisor': args.original_supervisor})
    with thread_override(args.threads):
        search.run(directory, 'xgb', hours=args.hours)
        if read_json(directory / 'xgb/status.json')['finished'] and time.monotonic() < deadline:
            search.select(directory, 'xgb', hours=(deadline-time.monotonic())/3600)
    # Keep the resource history with a representative selected by this worker.
    selected = directory / 'xgb/selection.json'
    if selected.exists():
        import shutil
        trial = read_json(selected)['trial']
        source = directory / 'xgb' / f'trial_{trial:04d}' / 'resource_history.json'
        if source.exists():
            shutil.copy2(source, directory / 'xgb/selected/resource_history.json')
    # The old coordinator continues to own the neural worker. If it finishes first,
    # complete the originally authorized assessment once both representatives exist.
    if not selected.exists() or args.original_supervisor is None:
        return
    while time.monotonic() < deadline:
        if (directory / 'session-result.json').exists():
            break
        time.sleep(5)
    if (directory / 'gnn/selection.json').exists() and time.monotonic() < deadline:
        if not (directory / 'report/summary.json').exists():
            import subprocess
            from conditional.__main__ import predict
            predict(directory, 'xgb', ('train','test','validation'), 'cpu')
            info = read_json(directory / 'session.json')
            neural_python = info.get('nn_python')
            if neural_python is None:
                # Resolve from the existing installed environment beside this interpreter.
                neural_python = str(Path(sys.executable).resolve().parents[2] / 'fire-nn/bin/python')
            for command in (['predict','--model','gnn','--device','cuda'], ['report','--publish']):
                subprocess.run([neural_python, '-m','conditional','--experiment',args.experiment,*command],
                               cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
