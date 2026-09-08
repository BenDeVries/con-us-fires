"""Select the reduction dimension for the boosted baseline on a ladder of rungs.

Ports the rule of `inla_glm_comparison.qmd` §6.5: score every candidate representation
under *fixed* hyperparameters, then take the smallest fixed-effect dimension whose
paired month-block bootstrap interval for the score difference against the best rung
still contains zero. Holding hyperparameters fixed is what makes the ladder a
measurement of the representation rather than of the tuner; the per-arm re-tuning in
`model/xgb/tune.py` comes afterwards.

    conda run -n fire-xgb python -m model.xgb.select_reduction
    conda run -n fire-xgb python -m model.xgb.select_reduction --rungs raw G31 R0.90

Resumable: one record per rung in output/xgb/reduction_ladder/results.json, so an
interrupted sweep continues where it stopped. Validation-blind, matching the HPO
protocol in CLAUDE.md -- only the test split is ever scored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import reduction as red_mod
from .common import OUT_DIR, make_dmatrix, params_from_predt
from .features import apply_reduction, build_dataset
from .train import DEFAULT_PARAMS, fit

LADDER_DIR = OUT_DIR / "reduction_ladder"

# raw is the reference rung. Arm G takes the top k of one global rotation; k = 31 lines
# up with the GNN's tuned n_pca and k = 51 is a full-rank rotation, which isolates
# rotation-without-truncation from truncation. Arm R varies per-block variance retention.
GLOBAL_K = [4, 8, 12, 17, 22, 31, 39, 51]
BLOCK_VAR = [0.80, 0.90, 0.95, 0.99]
# Static-bypass arms: the rotation covers the 39 dynamic predictors and the 12 of
# model.data.STATIC_COLS are appended raw. B39 rotates the dynamic block at full rank, so
# it is 71 features wide exactly like raw and G51 -- the control that separates "rotation
# destroys axis alignment" from "truncation discards variance".
BYPASS_K = [4, 8, 12, 17, 22, 28, 34, 39]


def rung_specs() -> dict[str, dict]:
    specs = {"raw": {"arm": "none"}}
    for k in GLOBAL_K:
        specs[f"G{k}"] = {"arm": "global", "n_pca": k}
    for v in BLOCK_VAR:
        specs[f"R{v:.2f}"] = {"arm": "block", "var_frac": v}
    for k in BYPASS_K:
        specs[f"B{k}"] = {"arm": "global", "n_pca": k, "static_bypass": True}
    for v in BLOCK_VAR:
        specs[f"BR{v:.2f}"] = {"arm": "block", "var_frac": v, "static_bypass": True}
    return specs


def row_nll(p, y) -> np.ndarray:
    """Per-row negative ZIB log score from the predicted parameter frame."""
    from scipy.stats import beta as sbeta
    p_occ = np.clip(p["p_occ"], 1e-9, 1 - 1e-9)
    mu = np.clip(p["mu"], 1e-9, 1 - 1e-9)
    phi = np.asarray(p["phi"], dtype=np.float64)
    pos = y > 0
    ls = np.where(pos,
                  np.log(p_occ) + sbeta.logpdf(np.where(pos, y, 0.5),
                                               mu * phi, (1 - mu) * phi),
                  np.log1p(-p_occ))
    return -ls


def month_block_boot(sums: np.ndarray, counts: np.ndarray, block: int = 3,
                     B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap resampling whole target months, so all 12 horizons move together."""
    T = len(sums)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    nblk = int(np.ceil(T / block))
    for i in range(B):
        if block == 1:
            idx = rng.integers(0, T, size=T)
        else:
            starts = rng.integers(0, max(T - block + 1, 1), size=nblk)
            idx = (starts[:, None] + np.arange(block)).ravel()[:T] % T
        out[i] = sums[idx].sum() / counts[idx].sum()
    return (float(sums.sum() / counts.sum()),
            float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2)))


def run_rung(ds_raw, name: str, spec: dict, params: dict, rounds: int, esr: int,
             eig_floor: float) -> dict:
    if spec["arm"] == "none":
        ds, red = ds_raw, None
    else:
        red = red_mod.fit_from_panel(ds_raw.raw_cov_names, spec["arm"],
                                     n_pca=spec.get("n_pca"),
                                     var_frac=spec.get("var_frac", 0.90),
                                     eig_floor=eig_floor,
                                     static_bypass=spec.get("static_bypass", False))
        ds = apply_reduction(ds_raw, red)

    xgblss, best_it, secs = fit(ds, params, num_boost_round=rounds,
                                early_stopping_rounds=esr, verbose_eval=False)
    dtest = make_dmatrix(ds.splits["test"]["X"])
    p = params_from_predt(xgblss.predict(dtest, pred_type="parameters", n_samples=1))
    y = np.asarray(ds.splits["test"]["y"], dtype=np.float64)
    nll = row_nll(p, y)

    months = pd.to_datetime(ds.splits["test"]["meta"]["target_date"])
    g = pd.DataFrame({"m": months.to_numpy(), "nll": nll}).groupby("m")["nll"]
    per_month = g.agg(["sum", "count"]).sort_index()

    return {
        "rung": name, "arm": spec["arm"],
        "static_bypass": spec.get("static_bypass", False),
        "n_components": (red.params.get("n_components") if red is not None else None),
        "n_pca": spec.get("n_pca"), "var_frac": spec.get("var_frac"),
        "k": (red.k if red is not None else len(ds_raw.raw_cov_names)),
        "n_features": len(ds.num_features),
        "test_nll": float(nll.mean()),
        "best_iteration": best_it, "seconds": round(secs, 1),
        "month_sum": per_month["sum"].tolist(),
        "month_count": per_month["count"].astype(int).tolist(),
        "months": [str(m.date()) for m in per_month.index],
        "cum_var": (float(red.eigen["global"]["cum_var"]) if red is not None
                    and red.arm == "global" else None),
    }


def analyze(results: list[dict], block: int = 3, seed: int = 0) -> dict:
    """Smallest-k rung whose bootstrap interval against the best rung contains zero."""
    by = {r["rung"]: r for r in results}
    best = min(results, key=lambda r: r["test_nll"])
    ref_sum = np.array(best["month_sum"])
    ref_cnt = np.array(best["month_count"])

    rows = []
    for r in results:
        s, c = np.array(r["month_sum"]), np.array(r["month_count"])
        if len(s) != len(ref_sum):
            continue
        mean, lo, hi = month_block_boot(s - ref_sum, c, block=block, seed=seed)
        rows.append({"rung": r["rung"], "k": r["k"], "n_features": r["n_features"],
                     "test_nll": r["test_nll"], "delta": mean, "lo": lo, "hi": hi,
                     "tied_with_best": bool(lo <= 0 <= hi)})
    rows.sort(key=lambda d: (d["k"], d["test_nll"]))
    tied = [d for d in rows if d["tied_with_best"]]
    chosen = min(tied, key=lambda d: (d["k"], d["test_nll"])) if tied else rows[0]

    print(f"\n{'rung':>8} {'k':>4} {'feat':>5} {'test NLL':>11} {'Δ vs best':>11} "
          f"{'95% CI':>22}  tie")
    for d in sorted(rows, key=lambda d: d["test_nll"]):
        mark = "*" if d["rung"] == chosen["rung"] else " "
        print(f"{mark}{d['rung']:>7} {d['k']:>4} {d['n_features']:>5} "
              f"{d['test_nll']:>11.5f} {d['delta']:>+11.5f} "
              f"[{d['lo']:>+9.5f},{d['hi']:>+9.5f}]  {'yes' if d['tied_with_best'] else 'no'}")
    print(f"\nbest by NLL: {best['rung']}   selected (smallest tied k): {chosen['rung']}")
    return {"best_rung": best["rung"], "selected": chosen, "table": rows,
            "block": block, "reference": best["rung"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=None, help="subset of rung names")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--eig-floor", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block", type=int, default=3, help="bootstrap block length in months")
    ap.add_argument("--analyze", action="store_true",
                    help="re-read results.json and print the selection table only")
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--ladder-dir", default=None,
                    help="where results.json/selection.json live (default "
                         "output/xgb/reduction_ladder). Rungs resume from whatever is "
                         "already in that file, so a ladder run under a changed "
                         "likelihood needs its own directory or it will silently "
                         "compare across two different scores")
    args = ap.parse_args()

    ladder_dir = Path(args.ladder_dir) if args.ladder_dir else LADDER_DIR
    ladder_dir.mkdir(parents=True, exist_ok=True)
    results_path = ladder_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []

    if args.analyze:
        if not results:
            raise SystemExit(f"no results at {results_path}")
        sel = analyze(results, block=args.block)
        (ladder_dir / "selection.json").write_text(json.dumps(sel, indent=2))
        return

    specs = rung_specs()
    names = args.rungs or list(specs)
    unknown = set(names) - set(specs)
    if unknown:
        raise SystemExit(f"unknown rungs {sorted(unknown)}; available {list(specs)}")

    params = dict(DEFAULT_PARAMS, device=args.device)
    done = {r["rung"] for r in results}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} rungs ({len(names)-len(todo)} already done, {len(todo)} to go)")
    if not todo:
        analyze(results, block=args.block)
        return

    t0 = time.time()
    # One design assembly serves every rung: the rotation is a matmul on the covariate
    # block, so rebuilding a 5.1M-row panel per rung would dominate the run time.
    ds_raw = build_dataset(args.lookback, args.horizon, max_origins=args.max_origins)
    print(f"design: {len(ds_raw.feature_names)} features, "
          f"{len(ds_raw.splits['train']['y']):,} train rows, built in {time.time()-t0:.0f}s",
          flush=True)

    for i, name in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} ...", flush=True)
        rec = run_rung(ds_raw, name, specs[name], params,
                       args.num_boost_round, args.early_stopping_rounds, args.eig_floor)
        results.append(rec)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[{i}/{len(todo)}] {name}  k={rec['k']}  test NLL {rec['test_nll']:+.5f}  "
              f"({rec['seconds']/60:.1f} min)\n", flush=True)

    sel = analyze(results, block=args.block)
    (ladder_dir / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"wrote {results_path} and selection.json")


if __name__ == "__main__":
    main()
