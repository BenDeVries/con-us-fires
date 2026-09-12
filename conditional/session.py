"""Run two independent local model workers for one shared wall-clock session."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .state import ROOT, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', default='conditional_v1')
    p.add_argument('--hours', type=float, default=12)
    p.add_argument('--nn-python', type=Path, required=True)
    p.add_argument('--xgb-python', type=Path, required=True)
    args = p.parse_args()
    if Path(args.experiment).name != args.experiment:
        p.error('experiment must be a directory name')
    root = ROOT / 'output' / args.experiment
    interpreters = {'gnn': args.nn_python.resolve(), 'xgb': args.xgb_python.resolve()}
    deadline = time.monotonic() + args.hours * 3600
    workers, stages, logs = {}, {}, {}
    env = dict(os.environ, PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def spawn(family, command):
        remaining = max(0., (deadline-time.monotonic()) / 3600)
        argv = [str(interpreters[family]), '-m', 'conditional', '--experiment', args.experiment, command,
                '--model', family]
        if command in ('run','select'):
            argv += ['--hours', str(remaining)]
        if family == 'gnn':
            argv += ['--device','cuda']
        stages[family] = command
        workers[family] = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=logs[family], stderr=subprocess.STDOUT)
    def pause(signum, frame):
        nonlocal deadline
        deadline = time.monotonic()
        for worker in workers.values():
            if worker.poll() is None:
                worker.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGTERM, pause); signal.signal(signal.SIGINT, pause)
    for family in interpreters:
        logs[family] = (root / family / 'session.log').open('a', buffering=1)
        spawn(family, 'run')
    write_json(root / 'session.json', {'pid': os.getpid(), 'workers': {k: v.pid for k,v in workers.items()},
        'hours': args.hours, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    failed = {}
    while workers:
        for family, worker in list(workers.items()):
            code = worker.poll()
            if code is None:
                continue
            del workers[family]
            if code:
                failed[family] = code
            elif time.monotonic() < deadline and stages[family] == 'run':
                if read_json(root / family / 'status.json')['finished']:
                    spawn(family, 'select')
        if workers:
            time.sleep(5)
    if not failed and time.monotonic() < deadline and all((root / f / 'selection.json').exists() for f in interpreters):
        for family, python in interpreters.items():
            argv = [str(python), '-m','conditional','--experiment',args.experiment,'predict','--model',family]
            if family == 'gnn':
                argv += ['--device','cuda']
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=logs[family], stderr=subprocess.STDOUT)
            if result.returncode:
                failed[family] = result.returncode
        if not failed:
            result = subprocess.run([str(interpreters['gnn']),'-m','conditional','--experiment',args.experiment,
                                     'report','--publish'], cwd=ROOT, env=env,
                                    stdout=logs['gnn'], stderr=subprocess.STDOUT)
            if result.returncode:
                failed['report'] = result.returncode
    write_json(root / 'session-result.json', {'failed': failed, 'stages': stages,
        'ended_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    for log in logs.values():
        log.close()
    sys.exit(bool(failed))


if __name__ == '__main__':
    main()
