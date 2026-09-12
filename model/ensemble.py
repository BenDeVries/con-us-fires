"""Deep ensemble for the GCN->LSTM, scored on validation.

Refits the shipped configuration under K different seeds and keeps every member. Two
things come out of it:

* **Spread across members** -- the uncertainty in the fitted network. A single GNN fit is
  one draw from initialisation + batch order + GPU non-determinism, and CLAUDE.md's
  seed noise floor (~0.00195 nats) says that draw is not small next to the gaps this
  project reports. This measures it directly on validation instead of inferring it.
* **The mixture predictive** -- p(y) = mean_k p_k(y), which is a strictly better forecast
  than any member and is the thing to quote if the ensemble is shipped as the model.

The mixture is formed on the **density**, not the parameters. Averaging (pi, mu, phi)
across members would produce a single ZIB that is not the mixture of the members' ZIBs
and would understate the spread; averaging the NLLs would compute the mean member score,
which is a different and always-worse quantity than the ensemble score.

Members train against the test split exactly as the shipped run did, so validation stays
untouched until `--summarize`.

    conda run -n fire-nn python -m model.ensemble --ckpt-dir output/model/v2_trial023
    conda run -n pytorch python -m model.ensemble --ckpt-dir output/model/v2_trial023 --summarize
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import CKPT_DIR
from .data import WindowDataset, truncate_harmonics, truncate_pca
from .model import build_norm_adj
from .predict import load_cfg
from .train import load_panel, train_model
from .zib import compute_metrics, gate_fire_prob, zib_nll


@torch.no_grad()
def predict_split(model, panel, cfg, split: str, device: str) -> pd.DataFrame:
    """Deterministic (pi, mu, phi) for one split. No ancestral sampling: the ensemble's
    predictive spread comes from the members, not from within-member trajectories."""
    A = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    dl = DataLoader(WindowDataset(panel, panel.split_origins[split], cfg.lookback, cfg.horizon),
                    batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    P, MU, PHI, Y, OV = [], [], [], [], []
    for batch in dl:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b, A)
        P.append(gate_fire_prob(out["pi_logit"], cfg.link).cpu().numpy())
        MU.append(out["mu"].cpu().numpy())
        PHI.append(out["phi"].cpu().numpy())
        Y.append(batch["y"].numpy())
        OV.append(batch["origin"].numpy())
    p_occ, mu, phi = np.concatenate(P), np.concatenate(MU), np.concatenate(PHI)
    y, ov = np.concatenate(Y), np.concatenate(OV)
    G, H, N = len(ov), cfg.horizon, panel.n_nodes
    hz = np.arange(1, H + 1)
    return pd.DataFrame({
        "origin_date": np.repeat(panel.dates[ov], H * N),
        "horizon": np.tile(np.repeat(hz, N), G),
        "target_date": np.repeat(panel.dates[ov[:, None] + hz[None, :]].reshape(G * H), N),
        "county_fips": np.tile(panel.node_fips, G * H),
        "node_id": np.tile(np.arange(N), G * H),
        "y_true": y.reshape(-1), "p_occ": p_occ.reshape(-1),
        "mu": mu.reshape(-1), "phi": phi.reshape(-1),
        "e_y": (p_occ * mu).reshape(-1),
    })


def _logits(p_occ: np.ndarray) -> torch.Tensor:
    """pi_logit from the stored P(fire). p_occ = 1 - pi, so pi_logit = log((1-p)/p)."""
    p = np.clip(p_occ.astype(np.float64), 1e-12, 1 - 1e-12)
    return torch.from_numpy(np.log((1 - p) / p))


def summarize(root: Path, link: str) -> dict:
    members = sorted(root.glob("seed_*/predictions_validation.parquet"))
    if not members:
        raise SystemExit(f"no member predictions under {root}")
    base = pd.read_parquet(members[0])
    y = torch.from_numpy(base.y_true.to_numpy(np.float64))
    keys = base[["origin_date", "horizon", "county_fips"]]

    per_member, logp, e_y_sum, p_occ_sum = [], [], None, None
    for m in members:
        d = pd.read_parquet(m)
        assert d[["origin_date", "horizon", "county_fips"]].equals(keys), f"{m} row set differs"
        pi_logit = _logits(d.p_occ.to_numpy())
        mu = torch.from_numpy(d.mu.to_numpy(np.float64))
        phi = torch.from_numpy(d.phi.to_numpy(np.float64))
        met = compute_metrics(pi_logit, mu, phi, y, link=link)
        met["member"] = m.parent.name
        per_member.append(met)
        logp.append(-zib_nll(pi_logit, mu, phi, y, reduction="none", link=link).numpy())
        ey, po = d.e_y.to_numpy(np.float64), d.p_occ.to_numpy(np.float64)
        e_y_sum = ey if e_y_sum is None else e_y_sum + ey
        p_occ_sum = po if p_occ_sum is None else p_occ_sum + po

    K = len(members)
    L = np.stack(logp)                                     # [K, rows]
    mix_logp = torch.logsumexp(torch.from_numpy(L), dim=0).numpy() - np.log(K)
    ens = {"nll": float(-mix_logp.mean()),
           "mae_full": float(np.abs(e_y_sum / K - base.y_true.to_numpy()).mean()),
           "members": K}
    label = (base.y_true.to_numpy() > 0).astype("int8")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        ens["gate_auc"] = float(roc_auc_score(label, p_occ_sum / K))
        ens["gate_ap"] = float(average_precision_score(label, p_occ_sum / K))
    except ImportError:
        pass

    keys_num = [k for k in per_member[0] if isinstance(per_member[0][k], float)]
    spread = {}
    for k in keys_num:
        v = np.array([m[k] for m in per_member], dtype=float)
        spread[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "min": float(v.min()), "max": float(v.max())}
    return {"split": "validation", "members": [m["member"] for m in per_member],
            "per_member": per_member, "spread": spread, "ensemble": ens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None,
                    help="the shipped run whose config is replicated (defaults to output/model)")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0, help="first member seed; members use seed0..seed0+K-1")
    ap.add_argument("--summarize", action="store_true",
                    help="score existing members instead of training (CPU only)")
    ap.add_argument("--pin-epochs", type=int, default=None,
                    help="train every member exactly N epochs and keep the final weights, "
                         "spending no test look; N is the shipped run's best epoch. This is what "
                         "makes the ensemble protocol-comparable with model.xgb.seed_ensemble, "
                         "which pins rounds at best_iteration the same way.")
    ap.add_argument("--out-name", default="ensemble",
                    help="subdirectory of --ckpt-dir holding the members")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    root = ckpt_dir / args.out_name
    cfg = load_cfg(ckpt_dir)

    if args.summarize:
        out = summarize(root, cfg.link)
        (root / "summary.json").write_text(json.dumps(out, indent=2))
        s, e = out["spread"]["nll"], out["ensemble"]
        print(f"{len(out['members'])} members")
        print(f"  member NLL  {s['mean']:+.6f} +/- {s['sd']:.6f}  [{s['min']:+.6f}, {s['max']:+.6f}]")
        print(f"  ensemble NLL {e['nll']:+.6f}   (gain over the mean member "
              f"{e['nll'] - s['mean']:+.6f})")
        print(f"wrote {root/'summary.json'}")
        return

    root.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    # The rotation is a train-split eigendecomposition and does not read cfg.seed, so every
    # member shares it. Slicing the shared panel once reproduces what each member trains on
    # (train_model slices its own copy the same way) without rebuilding the panel K times.
    sliced = (truncate_pca(panel, cfg.n_pca)
              if cfg.n_pca is not None and panel.pca_mean is not None else panel)
    ppanel = truncate_harmonics(sliced, cfg.n_harmonics)

    for i in range(args.members):
        seed = args.seed0 + i
        mdir = root / f"seed_{seed}"
        pred = mdir / "predictions_validation.parquet"
        if pred.exists():
            print(f"seed {seed}: already done, skipping")
            continue
        print(f"\n=== member seed {seed} ===")
        t0 = time.time()
        mcfg = copy.deepcopy(cfg)
        mcfg.seed = seed
        if args.pin_epochs:
            mcfg.max_epochs, mcfg.patience = args.pin_epochs, 10 ** 9
        best, test_met, _, model = train_model(mcfg, panel, mdir, verbose=True,
                                               eval_validation=False,
                                               restore_best=not args.pin_epochs)
        if args.pin_epochs:
            # the weights in hand are the final epoch's; best.pt is the test argmin and is
            # deliberately not loaded, so save what was actually scored under its own name
            torch.save(model.state_dict(), mdir / "final.pt")
        else:
            model.load_state_dict(torch.load(mdir / "best.pt", map_location=device))
        predict_split(model, ppanel, mcfg, "validation", device).to_parquet(pred, index=False)
        print(f"  test_nll {best:+.6f}  wrote {pred.name}  {time.time()-t0:.0f}s")

    print("\nnow score them:  python -m model.ensemble "
          f"--ckpt-dir {ckpt_dir} --out-name {args.out_name} --summarize")


if __name__ == "__main__":
    main()
