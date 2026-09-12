"""Parameter uncertainty for a fitted xgb arm, by refitting at K seeds on the unaltered design.

The companion to `model.ensemble` on the boosted side, and the alternative to
`model.xgb.bootstrap` for anyone unwilling to perturb the data. The shipped configuration
draws `subsample=0.740` of the rows and `colsample_bytree=0.412` of the columns at every
tree, so the booster seed alone moves the fit substantially while every training row stays
exactly where it is.

What this measures and what it does not:

* It **is** the variance of the fitted booster conditional on this training sample -- the
  part of a prediction that is an accident of which rows and columns the sampler happened
  to draw. That is the component you need for a predictive interval that is honest about
  not knowing the parameters.
* It is **not** the sampling variance of the training panel. Only a resampling scheme can
  reach that, which is exactly why `bootstrap.py` alters the data. The two numbers are
  complementary, and the seed spread is the smaller of the two by construction.

Members write `predictions_validation.parquet` in `predict.py`'s schema, so the density
mixture p(y) = mean_k p_k(y) is formed by `model.ensemble.summarize` unchanged -- xgb's
`params_from_predt` sets `pi_logit = log(gate/(1-gate))` and `p_occ = 1-gate`, which is the
exact inverse of the `_logits` round-trip that function applies.

The round count is fixed at the shipped fit's `best_iteration` with early stopping off, so
no member spends a look at the test split and the seed is the only thing that varies.

Resumable: a member whose predictions exist is skipped.

    conda run -n fire-xgb python -m model.xgb.seed_ensemble --model-dir output/xgb_spacetime
    conda run -n pytorch  python -m model.xgb.seed_ensemble --model-dir output/xgb_spacetime --summarize
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# `.common` imports xgboost, which the CPU env used for --summarize does not have, so it is
# imported inside the training branch instead of at module scope.

# XGBoostLSS writes these into the params dict itself at train time; passing them back in
# fights with it.
_LSS_OWNED = ("objective", "base_score", "num_target", "disable_default_eval_metric")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="output/xgb",
                    help="directory holding model.pkl + config.json")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first member seed; members use seed0..seed0+K-1")
    ap.add_argument("--summarize", action="store_true",
                    help="score existing members instead of training (CPU only, pytorch env)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    root = model_dir / "seed_ensemble"

    if args.summarize:
        from ..ensemble import summarize
        out = summarize(root, "logit")
        (root / "summary.json").write_text(json.dumps(out, indent=2))
        s, e = out["spread"]["nll"], out["ensemble"]
        print(f"{len(out['members'])} members")
        print(f"  member NLL   {s['mean']:+.6f} +/- {s['sd']:.6f}  [{s['min']:+.6f}, {s['max']:+.6f}]")
        print(f"  ensemble NLL {e['nll']:+.6f}   (gain over the mean member "
              f"{e['nll'] - s['mean']:+.6f})")
        print(f"wrote {root/'summary.json'}")
        return

    import numpy as np
    from xgboostlss.model import XGBoostLSS

    from . import reduction as red_mod
    from .common import make_dmatrix, params_from_predt
    from .features import apply_reduction, build_dataset
    from .zabeta_hurdle import C0_CAP, ZABetaHurdle

    cfg = json.loads((model_dir / "config.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    n_rounds = cfg["best_iteration"] + 1
    params = {k: v for k, v in cfg["params"].items() if k not in _LSS_OWNED}
    print(f"{model_dir.name}: {args.members} members, {n_rounds} rounds fixed, "
          f"subsample={params.get('subsample'):.4f} colsample={params.get('colsample_bytree'):.4f}")

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

    dtrain = make_dmatrix(tr["X"], tr["y"])
    # XGBoostLSS indexes evals[1] unconditionally and re-expands that matrix's label under a
    # flag separate from the one guarding dtrain, so handing it dtrain twice would expand the
    # training labels a second time and corrupt the fit. Early stopping is off, so this second
    # set only feeds logging; a training slice keeps test and validation unseen.
    deval = make_dmatrix(tr["X"].iloc[:2048], tr["y"][:2048])
    dval = make_dmatrix(va["X"])

    for i in range(args.members):
        seed = args.seed0 + i
        mdir = root / f"seed_{seed}"
        pred = mdir / "predictions_validation.parquet"
        if pred.exists():
            print(f"seed {seed}: already done, skipping")
            continue
        mdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        mparams = dict(params, seed=seed)
        xgblss = XGBoostLSS(ZABetaHurdle(stabilization=cfg.get("stabilization", "L2"),
                                         loss_fn="nll", c0_cap=C0_CAP))
        xgblss.train(mparams, dtrain, num_boost_round=n_rounds,
                     evals=[(dtrain, "train"), (deval, "train_slice")],
                     early_stopping_rounds=None, verbose_eval=False)
        p = params_from_predt(xgblss.predict(dval, pred_type="parameters", n_samples=1))
        del xgblss

        out = va["meta"].copy()
        out["y_true"] = va["y"]
        for k in ("p_occ", "mu", "phi", "e_y"):
            out[k] = p[k]
        out.to_parquet(pred, index=False)
        print(f"  seed {seed}: mean p_occ {np.mean(p['p_occ']):.4f} | "
              f"mean e_y {np.mean(p['e_y']):.5f} | {time.time()-t0:.0f}s")

    print(f"\nnow score them:  conda run -n pytorch python -m model.xgb.seed_ensemble "
          f"--model-dir {model_dir} --summarize")


if __name__ == "__main__":
    main()
