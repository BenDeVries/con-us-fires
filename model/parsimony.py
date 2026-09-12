"""Capacity (parsimony) selection: how much of the tuned model is load-bearing?

A 12-cell factorial settled the axes Optuna cannot search (archived in `docs/archive/ABLATION.md`).
This driver settles the one Optuna *can* search but has no incentive to: size. A TPE study
minimizing test NLL will
happily spend 3.9M parameters to buy 0.0004 nats, which is a fifth of the 0.00195-nat seed
SD -- i.e. nothing. The study's own leaderboard cannot settle the question either, because
every trial is a single seed and single-seed test NLL cannot resolve below ~0.004 nats.

So the reductions are re-measured here as a designed experiment: a one-at-a-time ladder of
capacity cuts from the winning trial, each replicated over every seed, with seed as a
blocking factor. Structural cost is the exact trainable parameter count, read off the built
model rather than derived, and the selection rule is the factorial's -- smallest model whose
blocked mean NLL is within a threshold of the best.

Unlike that factorial, seeds do NOT give matched initializations here: every cut changes a
tensor shape, which shifts the RNG stream. Blocking still removes the shared batch-order and
data-shuffling variance, but the paired deltas are weaker than the ~9x the factorial got.
That is why this runs 5 seeds where the factorial ran 3.

Validation-blind by construction (`eval_validation=False`), same guarantee `model.tune` gives.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run --live-stream -n fire-nn python -u -m model.parsimony --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time

import torch

from .config import Config, CKPT_DIR
from .train import train_model, load_panel, apply_params

PARSIMONY_DIR = CKPT_DIR / "parsimony"

# One-at-a-time cuts from the study's best trial. Each entry is the set of Config overrides
# applied to that baseline; "base" is the baseline itself. The cuts target the four places
# the parameter budget actually sits (LSTM width, head MLP, graph conv, PCA input width).
CANDIDATES: dict[str, dict] = {
    "base":    {},
    "lstm64":  {"lstm_hidden": 64},
    "head0":   {"head_hidden": 0},
    "gcn32x1": {"gcn_hidden": 32, "gcn_layers": 1},
    "pca12":   {"n_pca": 12},
}
BASE = "base"

# Unpaired seed SD of test NLL for this model (2026-08-08, 3 seeds of one fixed config).
SEED_SD = 0.00195


def run_cell(base_cfg, panel, name, overrides, seed):
    cfg = dataclasses.replace(base_cfg, seed=seed, **overrides)
    ckpt = PARSIMONY_DIR / f"{name}_s{seed}"
    ckpt.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(ckpt / "run.log", "w", buffering=1) as log_file:
        best_nll, test_m, _, model = train_model(
            cfg, panel, ckpt, verbose=False, log_file=log_file, eval_validation=False)
    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "config": name, "seed": seed, "overrides": overrides,
        "n_params": int(n_params),
        "test_nll": best_nll,
        "gate_auc": test_m.get("gate_auc"),
        "gate_ap": test_m.get("gate_ap"),
        "balanced_accuracy": test_m.get("balanced_accuracy"),
        "mae_pos": test_m.get("mae_pos"),
        "mae_full": test_m.get("mae_full"),
        "minutes": round(elapsed / 60, 2),
    }


def analyze(results_path):
    import numpy as np

    records = json.loads(results_path.read_text())
    names = sorted({r["config"] for r in records},
                   key=lambda n: min(r["n_params"] for r in records if r["config"] == n))
    seeds = sorted({r["seed"] for r in records})
    y = np.array([r["test_nll"] for r in records])
    print(f"\n{len(records)} runs | {len(names)} configs x {len(seeds)} seeds "
          f"(balanced: {len(records) == len(names) * len(seeds)})\n")

    # Two-way additive model: config effect + seed block. No interaction term -- with one
    # run per (config, seed) an interaction would be perfectly confounded with the residual.
    def effects(key, levels):
        base = levels[-1]
        return [np.array([1.0 if r[key] == lv else (-1.0 if r[key] == base else 0.0)
                          for r in records]) for lv in levels[:-1]]

    cols = [np.ones(len(records))] + effects("config", names)
    if len(seeds) > 1:
        cols += effects("seed", seeds)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    if dof <= 0:
        print(f"only {len(y)} runs for a rank-{np.linalg.matrix_rank(X)} design")
        return
    s2 = float(resid @ resid / dof)
    print(f"residual SD {np.sqrt(s2):.5f} nats on {dof} df | "
          f"raw SD of all {len(y)} test NLLs: {y.std(ddof=1):.5f}")

    summary = []
    for n in names:
        sub = [r for r in records if r["config"] == n]
        summary.append({
            "name": n,
            "params": max(r["n_params"] for r in sub),
            "mean": float(np.mean([r["test_nll"] for r in sub])),
            "sd": float(np.std([r["test_nll"] for r in sub], ddof=1)) if len(sub) > 1 else float("nan"),
            "auc": float(np.mean([r["gate_auc"] for r in sub])),
            "mae_pos": float(np.mean([r["mae_pos"] for r in sub])),
            "n": len(sub),
        })
    by_name = {d["name"]: d for d in summary}

    # Paired delta vs the baseline over the seeds both ran. Seeds do not share an init
    # stream across configs here, but they do share batch order, so the pairing still
    # removes part of the run-to-run variance -- report it alongside the blocked SE.
    base_by_seed = {r["seed"]: r["test_nll"] for r in records if r["config"] == BASE}
    print(f"\n{'config':<10}{'params':>10}{'mean nll':>11}{'sd':>9}{'d vs base':>11}"
          f"{'sd(d)':>9}{'t':>7}{'auc':>8}{'mae_pos':>9}")
    for d in sorted(summary, key=lambda d: d["params"]):
        sub = [r for r in records if r["config"] == d["name"]]
        pairs = [r["test_nll"] - base_by_seed[r["seed"]] for r in sub if r["seed"] in base_by_seed]
        if d["name"] != BASE and len(pairs) > 1:
            dm, ds = float(np.mean(pairs)), float(np.std(pairs, ddof=1))
            t = dm / (ds / np.sqrt(len(pairs))) if ds > 0 else float("nan")
            extra = f"{dm:>+11.5f}{ds:>9.5f}{t:>7.2f}"
        else:
            extra = f"{'-':>11}{'-':>9}{'-':>7}"
        print(f"{d['name']:<10}{d['params']:>10,}{d['mean']:>+11.5f}{d['sd']:>9.5f}"
              f"{extra}{d['auc']:>8.4f}{d['mae_pos']:>9.5f}")

    # Same two bars as model.ablate, with cost = parameter count instead of variance
    # components: statistical (can we resolve it) and practical (does it beat reseeding).
    se_cell = float(np.sqrt(s2 / len(seeds)))
    best = min(summary, key=lambda d: d["mean"])
    print(f"\nbest config {best['name']} at {best['mean']:+.5f} ({best['params']:,} params)")
    for label, bar in (("statistical (1 SE of a blocked config mean)", se_cell),
                       ("practical (1 seed SD, the deploy-time noise floor)", SEED_SD)):
        thresh = best["mean"] + bar
        within = [d for d in summary if d["mean"] <= thresh]
        pick = min(within, key=lambda d: (d["params"], d["mean"]))
        print(f"\n{label}: bar {bar:.5f} -> nll <= {thresh:+.5f}, "
              f"{len(within)}/{len(summary)} configs qualify")
        for d in sorted(within, key=lambda d: d["params"]):
            print(f"   {d['name']:<10}{d['mean']:>+10.5f}{d['params']:>12,} params")
        print(f"   PICK: {pick['name']} ({pick['params']:,} params, "
              f"{pick['mean']:+.5f}, {pick['mean'] - best['mean']:+.5f} vs best, "
              f"{by_name[BASE]['params'] / pick['params']:.2f}x smaller than {BASE})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="read results.json and print the blocked report (no GPU needed)")
    ap.add_argument("--params", type=str, default=str(CKPT_DIR / "tune" / "best_params.json"),
                    help="baseline hyperparameters; every candidate is a cut from these")
    ap.add_argument("--n-harmonics", type=int, default=6,
                    help="not stored in best_params.json; PCA studies use 6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="blocking factor: each seed is one complete replicate of the ladder")
    ap.add_argument("--configs", type=str, nargs="+", default=list(CANDIDATES),
                    help=f"which candidates to run (default all of {list(CANDIDATES)})")
    ap.add_argument("--combine", type=str, nargs="+", default=None,
                    help="run one extra candidate whose overrides are the union of the named "
                         "cuts, e.g. --combine lstm64 head0 (stage 2)")
    ap.add_argument("--link", type=str, default="logit")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10,
                    help="match the study that produced the baseline (tune default)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(PARSIMONY_DIR / "results.json")
        return

    plan_cfgs = {k: CANDIDATES[k] for k in args.configs}
    if args.combine:
        combo = {}
        for k in args.combine:
            combo.update(CANDIDATES[k])
        plan_cfgs["+".join(args.combine)] = combo

    base_cfg = Config()
    apply_params(base_cfg, args.params)
    base_cfg.n_harmonics = args.n_harmonics
    base_cfg.link = args.link
    base_cfg.max_epochs, base_cfg.patience = args.epochs, args.patience
    if args.device is not None:
        base_cfg.device = args.device

    PARSIMONY_DIR.mkdir(parents=True, exist_ok=True)
    results_path = PARSIMONY_DIR / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["config"], r["seed"]) for r in results}

    panel = load_panel(base_cfg)      # lookback is a baseline invariant, so one build serves all

    # Seed-major: a complete replicate lands after every len(plan_cfgs) runs, so an
    # interrupted sweep still analyzes as a balanced (if lower-power) blocked design.
    todo = [(s, n) for s in args.seeds for n in plan_cfgs if (n, s) not in done]
    total = len(args.seeds) * len(plan_cfgs)
    print(f"\n{len(plan_cfgs)} configs x {len(args.seeds)} seeds = {total} runs "
          f"({total - len(todo)} already done, {len(todo)} to go)\n", flush=True)

    for i, (seed, name) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name}_s{seed} ...", flush=True)
        rec = run_cell(base_cfg, panel, name, plan_cfgs[name], seed)
        results.append(rec)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[{i}/{len(todo)}] {name}_s{seed}  nll {rec['test_nll']:+.5f}  "
              f"auc {rec['gate_auc']:.4f}  params {rec['n_params']:,}  "
              f"({rec['minutes']:.1f} min)\n", flush=True)

    print(f"wrote {results_path} ({len(results)} runs)")


if __name__ == "__main__":
    main()
