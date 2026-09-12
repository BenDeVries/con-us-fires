"""Train the XGBoostLSS zero-adjusted Beta (ZIBeta) wildfire forecast baseline.

A direct multi-horizon gradient-boosted counterpart to the GCN->LSTM net. One
booster jointly predicts the hurdle's three parameters (gate, Beta alpha/beta) per
(origin, horizon, county) row. Outputs go to output/xgb/ (separate from the NN's
output/model/).

    conda run -n fire-xgb python -m model.xgb.train --smoke   # tiny plumbing check
    conda run -n fire-xgb python -m model.xgb.train           # full training
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from xgboostlss.model import XGBoostLSS

from ..data import STATIC_COLS
from ..zib import compute_metrics
from . import reduction as red_mod
from .common import OUT_DIR, make_dmatrix, params_from_predt
from .features import apply_reduction, build_dataset
from .zabeta_hurdle import C0_CAP, ZABetaHurdle, spec as hurdle_spec

DEFAULT_PARAMS = {
    "eta": 0.03, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.7,
    "min_child_weight": 50.0, "lambda": 1.0, "max_delta_step": 1.0,
    "tree_method": "hist", "device": "cpu", "seed": 0,
}


def _metrics(xgblss, dmat, y) -> dict:
    p = params_from_predt(xgblss.predict(dmat, pred_type="parameters", n_samples=1))
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
    return compute_metrics(t(p["pi_logit"]), t(p["mu"]), t(p["phi"]), t(y), link="logit")


def fit(ds, params: dict, num_boost_round: int = 500, early_stopping_rounds: int | None = 30,
        stabilization: str = "L2", verbose_eval: int | bool = 10, c0_cap: float = C0_CAP):
    """Train one booster on an already-assembled dataset. Returns (xgblss, best_it, seconds).

    Factored out of `main` so the reduction ladder and the tuner can reuse a single
    in-process dataset across many fits instead of rebuilding a 5.1M-row design each time.
    """
    dtrain = make_dmatrix(ds.splits["train"]["X"], ds.splits["train"]["y"])
    dtest = make_dmatrix(ds.splits["test"]["X"], ds.splits["test"]["y"])
    xgblss = XGBoostLSS(ZABetaHurdle(stabilization=stabilization, loss_fn="nll",
                                     c0_cap=c0_cap))
    t0 = time.time()
    xgblss.train(params, dtrain, num_boost_round=num_boost_round,
                 evals=[(dtrain, "train"), (dtest, "test")],
                 early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    elapsed = time.time() - t0
    best_it = getattr(xgblss.booster, "best_iteration", None)
    # predict_dist uses every tree in the booster, so drop the post-optimum rounds that
    # early stopping kept past best_iteration; otherwise predictions include them.
    if best_it is not None and early_stopping_rounds is not None:
        xgblss.booster = xgblss.booster[: best_it + 1]
    return xgblss, best_it, elapsed


def build_reduced(lookback: int, horizon: int, arm: str, n_pca: int | None = None,
                  var_frac: float = 0.90, eig_floor: float = 0.05,
                  max_origins: int | None = None, ds=None, climatology: bool = False,
                  clim_oof: bool = False, max_spatial: int = 0, max_temporal: int = 0,
                  n_nbr_pcs: int = 0, static_only: bool = False, n_harmonics: int = 1,
                  forecast_safe: bool = False):
    """Assemble the design and apply the requested reduction. Returns (ds, Reduction|None)."""
    if forecast_safe and arm != "none":
        raise ValueError("forecast_safe requires reduction='none'")
    if ds is not None and ds.forecast_safe != forecast_safe:
        raise ValueError("dataset forecast_safe does not match the requested protocol")
    if static_only and arm != "none":
        raise SystemExit(f"--reduction {arm} rotates the target-month covariate block, but "
                         f"static_only leaves only {len(STATIC_COLS)} near-constant columns "
                         f"there; that rotation is degenerate. Use --reduction none.")
    if ds is None:
        ds = build_dataset(lookback, horizon, max_origins=max_origins,
                           climatology=climatology, clim_oof=clim_oof,
                           max_spatial=max_spatial, max_temporal=max_temporal,
                           n_nbr_pcs=n_nbr_pcs, static_only=static_only,
                           n_harmonics=n_harmonics, forecast_safe=forecast_safe)
    if arm == "none":
        return ds, None
    red = red_mod.fit_from_panel(ds.raw_cov_names, arm, n_pca=n_pca,
                                 var_frac=var_frac, eig_floor=eig_floor)
    return apply_reduction(ds, red), red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="few origins + few rounds")
    ap.add_argument("--max-origins", type=int, default=None,
                    help="cap origins per split (debug/probe); --smoke implies 2")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample-bytree", type=float, default=0.7)
    ap.add_argument("--min-child-weight", type=float, default=50.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--max-delta-step", type=float, default=1.0,
                    help="cap on per-round leaf deltas; 0 disables. Without it a "
                         "single huge concentration0 leaf can saturate the C0_CAP "
                         "sigmoid and invert the gate (see zabeta_hurdle.py)")
    ap.add_argument("--stabilization", default="L2", choices=["None", "MAD", "L2"])
    ap.add_argument("--c0-cap", type=float, default=C0_CAP,
                    help="ceiling on the Beta's concentration0; check the fitted beta "
                         "histogram against it, since the response fn is a sigmoid and "
                         "saturates well before the cap (see zabeta_hurdle.py)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (xgboost tree device)")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--forecast-safe", action="store_true",
                    help="origin-safe terrain-only forecasting with expanding county history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reduction", default="none", choices=["none", "global", "block"],
                    help="principal-component reduction of the 51 target-month covariates")
    ap.add_argument("--n-pca", type=int, default=None, help="components kept by --reduction global")
    ap.add_argument("--var-frac", type=float, default=0.90,
                    help="per-block variance retained by --reduction block")
    ap.add_argument("--eig-floor", type=float, default=0.05,
                    help="drop block components with lambda_j < eig_floor * lambda_1")
    ap.add_argument("--params", default=None,
                    help="JSON of tuned booster params; overrides the individual flags. "
                         "A nested 'design' key (written by model.xgb.tune --tune-lags) "
                         "sets the three space-time counts below")
    ap.add_argument("--max-spatial", type=int, default=0,
                    help="exclusive neighbour shells for the space-time fire-history "
                         "block (st_{y,occ}_s{l}_t{k}); 0 omits the block")
    ap.add_argument("--max-temporal", type=int, default=0,
                    help="origin-anchored lags for the space-time block; needs "
                         "--lookback greater than this")
    ap.add_argument("--n-nbr-pcs", type=int, default=0,
                    help="neighbour-averaged target-month covariate PCs (nbpc_s{l}_PC{j}); "
                         "needs --max-spatial > 0")
    ap.add_argument("--n-harmonics", type=int, default=1,
                    help="seasonal order: sin/cos pairs for k=1..n at period 12/k, the same "
                         "basis as the GNN's n_harmonics. Overridden by a 'design' block in "
                         "--params. 6 is Nyquist for monthly data")
    ap.add_argument("--static-only", action="store_true",
                    help="rebuild the legacy near-static subset in data.STATIC_COLS. "
                         "It includes annual/epoch fields and is not origin-safe; use "
                         "--forecast-safe for new forecasts")
    ap.add_argument("--climatology", action="store_true",
                    help="append the train-only county x calendar-month ZIB triple "
                         "(clim_p_occ, clim_mu, clim_phi) from model.climatology")
    ap.add_argument("--clim-oof", action="store_true",
                    help="with --climatology, give train rows the leave-one-year-out fold "
                         "that excludes their own year, so the encoding is not fit to the "
                         "label it predicts; test/validation keep the full-train table")
    ap.add_argument("--no-eval-validation", dest="eval_validation", action="store_false",
                    help="skip validation scoring; use for exploratory arms so the "
                         "rationed validation look is not spent (see CLAUDE.md)")
    args = ap.parse_args()
    torch.set_num_threads(args.nthread)

    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / "smoke" if args.smoke else OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned = json.loads(Path(args.params).read_text()) if args.params else {}
    saved_safe = tuned.pop("forecast_safe", False)
    if args.params and saved_safe != args.forecast_safe:
        raise ValueError("tuned params forecast_safe does not match --forecast-safe")
    # `design` is not a booster param: the tuner nests the searched space-time counts
    # there so they reach build_dataset rather than xgboost.
    design = tuned.pop("design", None) or {
        "n_spatial": args.max_spatial, "n_temporal": args.max_temporal,
        "n_nbr_pcs": args.n_nbr_pcs}
    design.setdefault("n_harmonics", args.n_harmonics)

    print("building dataset...")
    t0 = time.time()
    max_origins = 2 if args.smoke else args.max_origins
    ds, red = build_reduced(args.lookback, args.horizon, args.reduction, n_pca=args.n_pca,
                            var_frac=args.var_frac, eig_floor=args.eig_floor,
                            max_origins=max_origins, climatology=args.climatology,
                            clim_oof=args.clim_oof, max_spatial=design["n_spatial"],
                            max_temporal=design["n_temporal"],
                            n_nbr_pcs=design["n_nbr_pcs"],
                            static_only=args.static_only,
                            n_harmonics=design["n_harmonics"], forecast_safe=args.forecast_safe)
    for s, d in ds.splits.items():
        print(f"  {s}: rows={len(d['y']):,} pos_frac={(d['y']>0).mean():.4f}")
    print(f"  features={len(ds.feature_names)} built in {time.time()-t0:.1f}s")
    if red is not None:
        print(f"  reduction={red.arm} k={red.k} ({len(red.cov_names)} covariates -> {red.k})")

    params = dict(DEFAULT_PARAMS)
    params.update({
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "max_delta_step": args.max_delta_step,
        "device": args.device,
        "seed": args.seed,
        "nthread": args.nthread,
    })
    params.update(tuned)
    n_rounds = 10 if args.smoke else args.num_boost_round
    esr = None if args.smoke else args.early_stopping_rounds

    print(f"training: rounds={n_rounds} early_stop={esr} device={params['device']}")
    xgblss, best_it, train_s = fit(ds, params, num_boost_round=n_rounds,
                                   early_stopping_rounds=esr,
                                   stabilization=args.stabilization, c0_cap=args.c0_cap)
    print(f"trained in {train_s:.1f}s  best_iteration={best_it}")

    xgblss.save_model(str(out_dir / "model.pkl"))
    if red is not None:
        red_mod.save(red, out_dir)

    dtest = make_dmatrix(ds.splits["test"]["X"], ds.splits["test"]["y"])
    report = {"test": _metrics(xgblss, dtest, ds.splits["test"]["y"])}
    if "validation" in ds.splits and args.eval_validation:
        dval = make_dmatrix(ds.splits["validation"]["X"], ds.splits["validation"]["y"])
        report["validation"] = _metrics(xgblss, dval, ds.splits["validation"]["y"])
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    config = {
        "lookback": args.lookback, "horizon": args.horizon,
        "num_boost_round": n_rounds, "best_iteration": best_it,
        "early_stopping_rounds": esr, "params": params,
        "stabilization": args.stabilization, "distribution": hurdle_spec(args.c0_cap),
        "c0_cap": args.c0_cap,
        "feature_names": ds.feature_names, "num_features": ds.num_features,
        "cat_feature": ds.cat_feature, "lc_categories": ds.lc_categories,
        "reduction": args.reduction, "raw_cov_names": ds.raw_cov_names,
        "climatology": args.climatology, "clim_oof": args.clim_oof,
        "max_spatial": design["n_spatial"], "max_temporal": design["n_temporal"],
        "n_nbr_pcs": design["n_nbr_pcs"], "static_only": args.static_only,
        "n_harmonics": design["n_harmonics"],
        "forecast_safe": args.forecast_safe,
        "reduction_k": (red.k if red is not None else None),
        "reduction_params": (red.params if red is not None else None),
        "train_seconds": round(train_s, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    for split in ("test", "validation"):
        if split in report:
            m = {k: round(v, 4) for k, v in report[split].items() if isinstance(v, float)}
            print(f"FINAL {split:>10}: {m}")
    print(f"saved -> {out_dir}/model.pkl, metrics.json, config.json")


if __name__ == "__main__":
    main()
