"""Freeze the completed test-selected GNN before exporting any validation predictions.

The destination must be new. Prediction is a separate, subsequent command, so the
configuration, checkpoint and protocol hashes are recorded before validation is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import optuna

from .predict import load_cfg
from .tune import TUNE_DIR, completed_trials


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(study_name: str, destination: Path, protocol: Path):
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///{TUNE_DIR / 'study.db'}")
    if any(t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING)
           for t in study.trials):
        raise ValueError("finish all running/queued trials before freezing the representative")
    complete = completed_trials(study)
    if not complete:
        raise ValueError("no finite COMPLETE trial can be selected")
    best = min(complete, key=lambda trial: trial.value)
    source = TUNE_DIR / study_name / f"trial_{best.number:03d}"
    cfg = load_cfg(source)
    cfg.validate_forecast_safe()
    rules = json.loads(protocol.read_text())
    if not cfg.forecast_safe or cfg.n_samples != 0:
        raise ValueError("representative must be forecast_safe with sampling disabled")
    if (cfg.lookback, cfg.horizon) != (rules["lookback"], rules["horizon"]):
        raise ValueError("representative windows disagree with the frozen protocol")
    metrics = json.loads((source / "metrics.json").read_text())
    if "validation" in metrics or metrics["best_test_nll"] != best.value:
        raise ValueError("checkpoint must be test-selected without validation evaluation")
    history = json.loads((source / "training_history.json").read_text())
    if min(row["test_nll"] for row in history) != best.value:
        raise ValueError("checkpoint is not the minimum recorded test NLL")
    checkpoint_hash, config_hash = sha256(source / "best.pt"), sha256(source / "config.json")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("best.pt", "config.json", "metrics.json", "training_history.json"):
        shutil.copy2(source / name, destination / name)
    states = Counter(t.state.name for t in study.trials)
    manifest = {
        "study": study_name, "trial": best.number, "selection_split": "test",
        "test_nll": best.value, "best_epoch": metrics["best_epoch"],
        "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
        "source_checkpoint": str(source / "best.pt"),
        "source_config": str(source / "config.json"),
        "protocol_path": str(protocol), "protocol_sha256": sha256(protocol),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_opened_during_tuning": False,
        "trial_states": dict(states), "finite_complete_trials": len(complete),
        "runtime_notes": [
            "Trial 0 failed because CUDA does not support zero-width embedding backward; the module was removed.",
            "Trials 1,2,3,5,6,7,8 exhausted GPU memory at batch size 4; these are not model-quality results.",
            "The batch-size-1 extension records each configuration and any retry/pruning reason in trial attributes.",
            "Trial 16 was interrupted to bound the extension; the final larger-network retry used test early stopping without median pruning.",
        ] if study_name == "forecast_safe_20260907_gnn" else [],
        "prediction_intervals": False,
        "sources_sha256": {str(path): sha256(path) for path in (
            Path("model/config.py"), Path("model/data.py"), Path("model/model.py"),
            Path("model/train.py"), Path("model/tune.py"), Path("model/predict.py"),
            Path("model/zib.py"), Path("model/freeze_forecast.py"),
            Path("output/forecast_safe_20260907/input_audit.json"),
            TUNE_DIR / study_name / "leaderboard.json",
            TUNE_DIR / study_name / "best_config.json",
        )},
    }
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path("output/forecast_safe_20260907/protocol.json"))
    args = parser.parse_args()
    freeze(args.study, args.destination, args.protocol)


if __name__ == "__main__":
    main()
