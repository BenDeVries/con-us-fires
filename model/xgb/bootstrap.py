"""Parameter uncertainty for a fitted xgb arm, by refitting on resampled training data.

Refits the shipped configuration on B month-block resamples of the *training* design and
scores each refit on validation. The spread across refits is uncertainty in the fitted
booster. It answers "how much of this arm's validation score is an accident of the
training sample?" and is a different quantity from `diagnostics.month_block_boot`, which
resamples the *evaluation* rows and holds the fit fixed.

Three choices that keep the estimate honest, all of which change the answer if reversed:

* **The resampling unit is a whole target month, in moving blocks of 3.** Every horizon
  and every county of a sampled month moves together. Resampling rows independently would
  treat the twelve horizons that share a target month as twelve independent draws and
  shrink the interval by roughly sqrt(12).
* **The round count is fixed at the shipped fit's `best_iteration`, with early stopping
  off.** Early stopping inside a replicate would let the test split choose each
  replicate's stopping point, folding stopping-point variance into what is meant to be
  parameter variance -- and spending a test look per replicate.
* **The booster seed is held at the shipped value**, so the resample is the only thing
  that varies between replicates.

Resumable: replicates already in replicates.json are skipped, so rerunning the identical
command continues an interrupted run.

    conda run -n fire-xgb python -m model.xgb.bootstrap --model-dir output/xgb_spacetime
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from xgboostlss.model import XGBoostLSS

from ..zib import compute_metrics
from . import reduction as red_mod
from .common import OUT_DIR, make_dmatrix, params_from_predt
from .features import apply_reduction, build_dataset
from .zabeta_hurdle import C0_CAP, ZABetaHurdle

# XGBoostLSS writes these into the params dict itself at train time; passing them back in
# fights with it.
_LSS_OWNED = ("objective", "base_score", "num_target", "disable_default_eval_metric")


def month_block_indices(target_date: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one moving-block resample over whole target months.

    Mirrors `diagnostics.month_block_boot`'s block construction so a parameter interval and
    a score interval are built on the same dependence assumption.
    """
    months, inv = np.unique(target_date, return_inverse=True)
    T = len(months)
    rows_by_month = [np.flatnonzero(inv == m) for m in range(T)]
    nblk = int(np.ceil(T / block))
    if block == 1:
        idx = rng.integers(0, T, size=T)
    else:
        starts = rng.integers(0, max(T - block + 1, 1), size=nblk)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T] % T
    return np.concatenate([rows_by_month[m] for m in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="directory holding model.pkl + config.json (defaults to output/xgb)")
    ap.add_argument("--replicates", type=int, default=25)
    ap.add_argument("--block", type=int, default=3,
                    help="moving-block length in months; matches the block-3 score intervals")
    ap.add_argument("--seed", type=int, default=0, help="seed for the resampling stream")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else OUT_DIR
    cfg = json.loads((model_dir / "config.json").read_text())
    out_dir = model_dir / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "replicates.json"
    done = json.loads(rec_path.read_text()) if rec_path.exists() else []
    have = {r["replicate"] for r in done}

    n_rounds = cfg["best_iteration"] + 1
    params = {k: v for k, v in cfg["params"].items() if k not in _LSS_OWNED}
    print(f"{model_dir.name}: {args.replicates} replicates, block={args.block} months, "
          f"{n_rounds} rounds fixed, seed {params.get('seed')}")

    print("building dataset...")
    t0 = time.time()
    ds = build_dataset(cfg["lookback"], cfg["horizon"],
                       climatology=cfg.get("climatology", False),
                       clim_oof=cfg.get("clim_oof", False),
                       max_spatial=cfg.get("max_spatial", 0),
                       max_temporal=cfg.get("max_temporal", 0),
                       n_nbr_pcs=cfg.get("n_nbr_pcs", 0),
                       static_only=cfg.get("static_only", False),
                       n_harmonics=cfg.get("n_harmonics", 1),
                       want_splits=("train", "validation"))
    if cfg.get("reduction", "none") != "none":
        ds = apply_reduction(ds, red_mod.load(model_dir))
    tr, va = ds.splits["train"], ds.splits["validation"]
    assert list(tr["X"].columns) == cfg["feature_names"], "feature columns differ from training"
    print(f"  train {len(tr['y']):,} rows | validation {len(va['y']):,} rows "
          f"| {len(ds.feature_names)} features | {time.time()-t0:.1f}s")

    tdate = tr["meta"]["target_date"].to_numpy()
    dval = make_dmatrix(va["X"], va["y"])
    y_val = torch.from_numpy(np.asarray(va["y"], dtype=np.float64))

    for b in range(args.replicates):
        if b in have:
            continue
        rng = np.random.default_rng(args.seed + b)
        idx = month_block_indices(tdate, args.block, rng)
        t0 = time.time()
        dtrain = make_dmatrix(tr["X"].take(idx), tr["y"][idx])
        # XGBoostLSS indexes evals[1] unconditionally and re-expands that matrix's label under
        # a flag separate from the one guarding dtrain, so handing it dtrain twice would expand
        # the training labels a second time and corrupt the fit. Early stopping is off, so this
        # second set only feeds logging; slicing the resample keeps test and validation unseen.
        stub = idx[:2048]
        deval = make_dmatrix(tr["X"].take(stub), tr["y"][stub])
        xgblss = XGBoostLSS(ZABetaHurdle(stabilization=cfg.get("stabilization", "L2"),
                                         loss_fn="nll", c0_cap=C0_CAP))
        xgblss.train(params, dtrain, num_boost_round=n_rounds,
                     evals=[(dtrain, "train"), (deval, "resample_slice")],
                     early_stopping_rounds=None, verbose_eval=False)
        del dtrain, deval
        p = params_from_predt(xgblss.predict(dval, pred_type="parameters", n_samples=1))
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
        m = compute_metrics(t(p["pi_logit"]), t(p["mu"]), t(p["phi"]), y_val, link="logit")
        del xgblss
        m.update({"replicate": b, "n_rows": int(len(idx)), "seconds": round(time.time() - t0, 1)})
        done.append(m)
        rec_path.write_text(json.dumps(sorted(done, key=lambda r: r["replicate"]), indent=2))
        print(f"  b{b:02d} nll {m['nll']:+.6f}  gate_ap {m.get('gate_ap', float('nan')):.4f}  "
              f"{m['seconds']:.0f}s")

    keys = [k for k in done[0] if k not in ("replicate", "n_rows", "seconds")]
    summary = {"model_dir": str(model_dir), "replicates": len(done), "block_months": args.block,
               "num_boost_round": n_rounds, "split": "validation",
               "point": json.loads((model_dir / "metrics.json").read_text()).get("validation"),
               "spread": {}}
    for k in keys:
        v = np.array([r[k] for r in done], dtype=float)
        summary["spread"][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                "lo2.5": float(np.quantile(v, 0.025)),
                                "hi97.5": float(np.quantile(v, 0.975))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'summary.json'}")
    for k in ("nll", "gate_ap", "mae_full"):
        if k in summary["spread"]:
            s = summary["spread"][k]
            print(f"  {k:10s} {s['mean']:+.6f} +/- {s['sd']:.6f}  "
                  f"[{s['lo2.5']:+.6f}, {s['hi97.5']:+.6f}]")


if __name__ == "__main__":
    main()
