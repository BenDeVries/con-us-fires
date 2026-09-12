"""Run the trained model and export a tidy predictions parquet.

Each row is one (forecast origin, horizon month, county) with the observed target
and the zero-inflated Beta outputs:
  p_occ = 1 - pi   (predicted P(fire occurs))
  mu               (predicted burned fraction | fire occurs)
  e_y = p_occ * mu (unconditional expected burned fraction)

Run:  conda run -n fire-nn python -m model.predict --split validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, CKPT_DIR
from .data import build_panel, WindowDataset, load_pca, apply_saved_pca
from .model import SpatioTemporalZIB, build_norm_adj
from .zib import gate_fire_prob


def load_cfg(ckpt_dir: Path) -> Config:
    cfg = Config()
    p = ckpt_dir / "config.json"
    if p.exists():
        for k, v in json.loads(p.read_text()).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    choices=["train", "test", "validation"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="ancestral ZIB sample trajectories for the predictive ensemble "
                         "(default cfg.n_samples; 0 disables sampling -> deterministic columns only)")
    ap.add_argument("--sample-chunk", type=int, default=None,
                    help="trajectories decoded per chunk (default cfg.sample_chunk); "
                         "lower it to fit a smaller GPU — memory scales with this, not n-samples")
    ap.add_argument("--ckpt-dir", default=None,
                    help="directory holding config.json + best.pt to load "
                         "(e.g. output/model/tune/trial_022); predictions are written here too. "
                         "defaults to output/model")
    ap.add_argument("--out-dir", default=None,
                    help="prediction destination (defaults to the checkpoint directory)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    cfg = load_cfg(ckpt_dir)
    cfg.validate_forecast_safe()
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        if cfg.forecast_safe:
            raise RuntimeError("forecast_safe GPU prediction requested but CUDA is unavailable")
        device = "cpu"

    drop = [] if cfg.n_pca is not None else cfg.drop_features
    n_harm = getattr(cfg, "n_harmonics", 1)
    panel = build_panel(cfg.lookback, cfg.horizon, drop, n_harmonics=n_harm,
                        static_bypass=getattr(cfg, "static_bypass", False),
                        forecast_safe=cfg.forecast_safe)
    if cfg.n_pca is not None:
        pca = load_pca(ckpt_dir / "pca_transform.npz")
        panel = apply_saved_pca(panel, pca, cfg.n_pca)
    A = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    model = SpatioTemporalZIB(cfg, panel.cov.shape[-1], panel.n_lc_classes,
                              n_nodes=panel.n_nodes).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    model.eval()

    origins = panel.split_origins[args.split]
    if not origins:
        raise SystemExit(f"split '{args.split}' has no forecast windows")
    ds = WindowDataset(panel, origins, cfg.lookback, cfg.horizon)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    n_samples = cfg.n_samples if args.n_samples is None else args.n_samples
    sample_chunk = cfg.sample_chunk if args.sample_chunk is None else args.sample_chunk
    if n_samples > 0:                  # seed once so the ancestral rollout is reproducible
        torch.manual_seed(cfg.sample_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(cfg.sample_seed)
    q_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    q_names = ["q05", "q25", "q50", "q75", "q95"]

    H, N = cfg.horizon, panel.n_nodes
    P, MU, PHI, Y, OV = [], [], [], [], []
    EYM, POE, QS = [], [], []          # ensemble mean, P(occ), quantiles (only if sampling)
    for batch in dl:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b, A)
        P.append(gate_fire_prob(out["pi_logit"], cfg.link).cpu().numpy())
        MU.append(out["mu"].cpu().numpy())
        PHI.append(out["phi"].cpu().numpy())
        Y.append(batch["y"].numpy())
        OV.append(batch["origin"].numpy())
        if n_samples > 0:
            ys = model.sample(b, A, n_samples, chunk=sample_chunk).numpy()  # [B,S,H,N]
            EYM.append(ys.mean(axis=1))                        # [B,H,N] origin-conditioned mean
            POE.append((ys > 0).mean(axis=1))                  # [B,H,N] P(fire) over trajectories
            QS.append(np.quantile(ys, q_levels, axis=1))       # [Q,B,H,N]
    p_occ = np.concatenate(P)          # [G, H, N]
    mu = np.concatenate(MU)
    phi = np.concatenate(PHI)
    y = np.concatenate(Y)
    ov = np.concatenate(OV)            # [G] origin month indices
    e_y = p_occ * mu
    G = len(ov)

    dates = panel.dates
    hz = np.arange(1, H + 1)
    target_date = dates[ov[:, None] + hz[None, :]]    # [G, H]
    origin_date = dates[ov]                            # [G]
    node_id = np.arange(N)

    df = pd.DataFrame({
        "origin_date": np.repeat(origin_date, H * N),
        "horizon": np.tile(np.repeat(hz, N), G),
        "target_date": np.repeat(target_date.reshape(G * H), N),
        "county_fips": np.tile(panel.node_fips, G * H),
        "node_id": np.tile(node_id, G * H),
        "y_true": y.reshape(-1),
        "p_occ": p_occ.reshape(-1),
        "mu": mu.reshape(-1),
        "phi": phi.reshape(-1),
        "e_y": e_y.reshape(-1),
    })

    if n_samples > 0:                  # additive ensemble columns from the ancestral rollout
        e_y_mean = np.concatenate(EYM)                 # [G,H,N]
        p_occ_ens = np.concatenate(POE)                # [G,H,N]
        quant = np.concatenate(QS, axis=1)             # [Q,G,H,N]
        df["e_y_mean"] = e_y_mean.reshape(-1)
        df["p_occ_ens"] = p_occ_ens.reshape(-1)
        for i, name in enumerate(q_names):
            df[name] = quant[i].reshape(-1)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{args.split}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  rows={len(df):,}  origins={G}  "
          f"origin_dates {origin_date.min()}..{origin_date.max()}")
    print(f"  mean p_occ {df.p_occ.mean():.4f} | mean e_y {df.e_y.mean():.5f} "
          f"| obs frac_pos {(df.y_true > 0).mean():.4f}")
    if n_samples > 0:
        print(f"  sampled S={n_samples} | mean e_y_mean {df.e_y_mean.mean():.5f} "
              f"| mean p_occ_ens {df.p_occ_ens.mean():.4f} | mean q50 {df.q50.mean():.5f}")


if __name__ == "__main__":
    main()
