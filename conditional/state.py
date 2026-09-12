"""Atomic local experiment state and cooperative pause boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def save_pickle(path, value):
    atomic_bytes(path, pickle.dumps(value, protocol=5))


def load_pickle(path):
    # Only locally generated, hash-checked experiment files are accepted by the CLI.
    with Path(path).open('rb') as f:
        return pickle.load(f)


class Paused(Exception):
    pass


class Session:
    def __init__(self, hours=12, boundary_limit=None):
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600
        self.requested = False
        self.boundaries = 0
        self.boundary_limit = boundary_limit

    def install_signals(self):
        def pause(signum, frame):
            self.requested = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, pause)

    def expired(self):
        return self.requested or time.monotonic() >= self.deadline

    def boundary(self):
        self.boundaries += 1
        if self.expired() or (self.boundary_limit is not None and
                              self.boundaries >= self.boundary_limit):
            raise Paused('Checkpoint saved; continue with the same run command.')
