"""Export tidy predictions from the trained ZABeta baseline.

Schema matches the NN's model/predict.py (one row per origin x horizon x county)
so the two models' predictions are directly comparable:
    origin_date, horizon, target_date, county_fips, node_id, y_true, p_occ, mu, e_y
plus phi (Beta precision).

    conda run -n fire-xgb python -m model.xgb.predict --split validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from xgboostlss.model import XGBoostLSS

from . import reduction as red_mod
from .common import OUT_DIR, make_dmatrix, params_from_predt
from .features import apply_reduction, build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    choices=["train", "test", "validation"])
    ap.add_argument("--model-dir", default=None,
                    help="directory holding model.pkl + config.json; predictions written here too "
                         "(defaults to output/xgb)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else OUT_DIR
    cfg = json.loads((model_dir / "config.json").read_text())
    xgblss = XGBoostLSS.load_model(str(model_dir / "model.pkl"))

    ds = build_dataset(cfg["lookback"], cfg["horizon"],
                       climatology=cfg.get("climatology", False),
                       clim_oof=cfg.get("clim_oof", False),
                       max_spatial=cfg.get("max_spatial", 0),
                       max_temporal=cfg.get("max_temporal", 0),
                       n_nbr_pcs=cfg.get("n_nbr_pcs", 0),
                       static_only=cfg.get("static_only", False),
                       n_harmonics=cfg.get("n_harmonics", 1),
                       forecast_safe=cfg.get("forecast_safe", False),
                       want_splits=(args.split,))
    if cfg.get("reduction", "none") != "none":
        ds = apply_reduction(ds, red_mod.load(model_dir))
    if args.split not in ds.splits:
        raise SystemExit(f"split '{args.split}' has no forecast windows")
    d = ds.splits[args.split]
    # guard against feature drift between train and predict
    assert list(d["X"].columns) == cfg["feature_names"], "feature columns differ from training"

    dmat = make_dmatrix(d["X"])
    # n_samples=1: predict_dist always draws samples before returning parameters;
    # the default 1000/row OOMs on the 5.1M-row train split. Params are deterministic.
    p = params_from_predt(xgblss.predict(dmat, pred_type="parameters", n_samples=1))

    out = d["meta"].copy()
    out["y_true"] = d["y"]
    out["p_occ"] = p["p_occ"]
    out["mu"] = p["mu"]
    out["phi"] = p["phi"]
    out["e_y"] = p["e_y"]

    out_path = model_dir / f"predictions_{args.split}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  rows={len(out):,}")
    print(f"  mean p_occ {out.p_occ.mean():.4f} | mean e_y {out.e_y.mean():.5f} "
          f"| obs frac_pos {(out.y_true > 0).mean():.4f}")


if __name__ == "__main__":
    main()
