"""Train the GCN->LSTM zero-inflated Beta wildfire forecaster.

Run:  conda run -n fire-nn python -m model.train            # full training
      conda run -n fire-nn python -m model.train --smoke    # 2-epoch sanity check
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataclasses

from .config import Config, CKPT_DIR
from .data import (build_panel, WindowDataset, fit_pca, save_pca, truncate_pca,
                   truncate_harmonics)
from .model import SpatioTemporalZIB, build_norm_adj
from .zib import zib_nll, compute_metrics, per_horizon_nll


def _loaders(panel, cfg):
    def mk(split, shuffle):
        ds = WindowDataset(panel, panel.split_origins[split], cfg.lookback, cfg.horizon)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)
    return mk("train", True), mk("test", False), mk("validation", False)


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _tf_ratio(cfg, epoch: int) -> float:
    """Teacher-forcing ratio for `epoch` (1-indexed): linear anneal tf_ratio_start ->
    tf_ratio_end over tf_anneal_epochs (0 -> max_epochs). 0.0 when teacher forcing is off."""
    if not cfg.teacher_forcing:
        return 0.0
    span = cfg.tf_anneal_epochs or cfg.max_epochs
    if span <= 1:
        return cfg.tf_ratio_end
    frac = min(max((epoch - 1) / (span - 1), 0.0), 1.0)
    return cfg.tf_ratio_start + (cfg.tf_ratio_end - cfg.tf_ratio_start) * frac


@torch.no_grad()
def evaluate(model, loader, A, device):
    model.eval()
    pis, mus, phis, ys = [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch, A)
        pis.append(out["pi_logit"]); mus.append(out["mu"])
        phis.append(out["phi"]); ys.append(batch["y"])
    pi = torch.cat(pis); mu = torch.cat(mus); phi = torch.cat(phis); y = torch.cat(ys)
    link = model.cfg.link
    metrics = compute_metrics(pi, mu, phi, y, link=link)
    metrics["per_horizon_nll"] = per_horizon_nll(pi, mu, phi, y, link=link)
    return metrics


def train_model(cfg, panel, ckpt_dir, report_cb=None, verbose=True, monitor_split="test",
                log_file=None, eval_validation=True, restore_best=True):
    """Train one model to early stopping. Returns (best_nll, test_metrics, val_metrics, model).

    panel is prebuilt and reused across calls (the tuner builds it once). report_cb, if
    given, is called as report_cb(epoch, test_nll) after every epoch -- it may raise to
    abort the run early (used for Optuna pruning). Writes best.pt/metrics.json/config.json
    into ckpt_dir.

    eval_validation=False leaves the held-out validation split completely untouched and
    returns val_metrics=None. The tuner sets this so that repeated HPO runs cannot erode
    validation as a final estimate; it is ignored when validation *is* the monitored split."""
    cfg.validate_forecast_safe()
    if cfg.forecast_safe:
        if not panel.forecast_safe or monitor_split != "test":
            raise ValueError("forecast_safe requires a safe panel and test-only selection")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("forecast_safe GPU training requested but CUDA is unavailable")
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    if cfg.n_pca is not None and panel.pca_mean is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_pca(panel, ckpt_dir / "pca_transform.npz")
        panel = truncate_pca(panel, cfg.n_pca)
    # No-op unless the caller passed a panel built at a higher order than this config asks
    # for -- which is exactly what the tuner does, to search n_harmonics without rebuilding.
    panel = truncate_harmonics(panel, cfg.n_harmonics)
    n_cov = panel.cov.shape[-1]

    train_dl, test_dl, val_dl = _loaders(panel, cfg)
    monitor_dl = val_dl if monitor_split == "validation" else test_dl
    monitor_label = "val_nll " if monitor_split == "validation" else "test_nll"
    A = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    model = SpatioTemporalZIB(cfg, n_cov, panel.n_lc_classes,
                              n_nodes=panel.n_nodes).to(device)
    # cfg.county_embed_dim's embedding is in the decayed group with everything else: that
    # shrinkage is what separates it from 3108*d free intercepts, and weight_decay is tuned
    # per trial. It is shared with the GCN/LSTM weights, so an embedding-specific prior scale
    # is not separately identified — noted, not fixed.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if verbose:
        print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")
    best_nll, best_state, bad = float("inf"), None, 0
    best_epoch, history = None, []

    use_cuda = device == "cuda"
    interrupted = False
    for epoch in range(1, cfg.max_epochs + 1):
        try:
            model.train(); t0 = time.time(); tot, n = 0.0, 0
            tf = _tf_ratio(cfg, epoch)
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            gnorm = 0.0
            loader = tqdm(train_dl, desc=f"epoch {epoch:3d}/{cfg.max_epochs}",
                          leave=False, dynamic_ncols=True, unit="batch") if verbose else train_dl
            for batch in loader:
                batch = _to_device(batch, device)
                opt.zero_grad()
                out = model(batch, A, tf_ratio=tf)
                loss = zib_nll(out["pi_logit"], out["mu"], out["phi"], batch["y"], link=cfg.link)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
                opt.step()
                bs = batch["y"].shape[0]
                tot += loss.item() * bs; n += bs
                if verbose:
                    gpu = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
                    loader.set_postfix(nll=f"{tot/n:+.4f}", grad=f"{gnorm:.2f}",
                                       gpu=f"{gpu:.1f}G", refresh=False)
            if verbose:
                loader.close()
            monitor_m = evaluate(model, monitor_dl, A, device)
            if not np.isfinite(monitor_m["nll"]):
                raise FloatingPointError("non-finite monitored NLL")
            history.append({"epoch": epoch, "train_nll": tot / n,
                            "test_nll": monitor_m["nll"],
                            "seconds": time.time() - t0})
            (ckpt_dir / "training_history.json").write_text(json.dumps(history, indent=2))
            is_best = monitor_m["nll"] < best_nll - (0.0 if cfg.forecast_safe else 1e-5)
            if verbose:
                gpu = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
                tf_str = f"| tf {tf:.2f} " if cfg.teacher_forcing else ""
                line = (
                    f"epoch {epoch:3d}/{cfg.max_epochs} | train_nll {tot/n:+.4f} "
                    f"{tf_str}"
                    f"| {monitor_label} {monitor_m['nll']:+.4f}{' *' if is_best else '  '} "
                    f"| bal_acc {monitor_m.get('balanced_accuracy', float('nan')):.3f} "
                    f"| auc {monitor_m.get('gate_auc', float('nan')):.3f} "
                    f"| ap {monitor_m.get('gate_ap', float('nan')):.3f} "
                    f"| mae_pos {monitor_m.get('mae_pos', float('nan')):.4f} "
                    f"| grad {gnorm:.2f} | {gpu:.1f}G | {time.time()-t0:.1f}s "
                    f"| bad {bad}/{cfg.patience}"
                )
                tqdm.write(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()

            if is_best:
                best_epoch = epoch
                best_nll, best_state, bad = monitor_m["nll"], {k: v.detach().cpu().clone()
                                                                for k, v in model.state_dict().items()}, 0
                torch.save(best_state, ckpt_dir / "best.pt")
            else:
                bad += 1
                if bad >= cfg.patience:
                    if verbose:
                        msg = f"early stop at epoch {epoch} (best {monitor_label} {best_nll:+.4f})"
                        tqdm.write(msg)
                        if log_file is not None:
                            log_file.write(msg + "\n"); log_file.flush()
                    break

            if report_cb is not None:
                report_cb(epoch, monitor_m)      # may raise to abort (pruning)

        except KeyboardInterrupt:
            if verbose:
                tqdm.write(f"\nInterrupted at epoch {epoch} — loading best checkpoint...")
            interrupted = True
            break

    if interrupted and verbose:
        tqdm.write(f"Best {monitor_label} seen: {best_nll:+.4f}")
    if cfg.forecast_safe and interrupted:
        raise KeyboardInterrupt("interrupted forecast-safe trial is not complete")
    if cfg.forecast_safe and best_state is None:
        raise FloatingPointError("forecast-safe run produced no finite checkpoint")
    # restore_best=False keeps the final epoch's weights. The argmin over the monitor split is
    # itself a selection on that split, so a run that pins its epoch count to spend no such look
    # must not undo the pin by loading it back.
    if best_state is not None and restore_best:
        model.load_state_dict(best_state)
    monitors_val = monitor_split == "validation"
    val_m = evaluate(model, val_dl, A, device) if (eval_validation or monitors_val) else None
    if monitor_split == "test":
        test_m = evaluate(model, test_dl, A, device)
        report = {"best_test_nll": best_nll, "best_epoch": best_epoch, "test": test_m}
        if val_m is not None:
            report["validation"] = val_m
    else:
        test_m = None
        report = {"best_val_nll": best_nll, "validation": val_m}
    (ckpt_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return best_nll, test_m, val_m, model


def load_panel(cfg, verbose=True):
    cfg.validate_forecast_safe()
    if verbose:
        print("building panel...")
    drop = [] if cfg.n_pca is not None else cfg.drop_features
    n_harm = getattr(cfg, "n_harmonics", 1)
    panel = build_panel(cfg.lookback, cfg.horizon, drop, n_harmonics=n_harm,
                        static_bypass=getattr(cfg, "static_bypass", False),
                        forecast_safe=cfg.forecast_safe)
    if cfg.n_pca is not None:
        panel = fit_pca(panel)
        if verbose:
            cumvar = panel.pca_explained_variance_ratio.cumsum()
            k = min(cfg.n_pca, len(cumvar))
            print(f"  PCA: {len(cumvar)} components over predictors "
                  f"({panel.n_harmonic_cols} harmonic + {panel.n_static_cols} static cols "
                  f"held out, passed through raw); "
                  f"top {k} explain {cumvar[k-1]:.1%} of variance")
    elif verbose and cfg.drop_features:
        print(f"  dropped predictors: {sorted(cfg.drop_features)}")
    if verbose:
        print(f"  N={panel.n_nodes} F_cov={panel.cov.shape[-1]} "
              f"harmonics={n_harm} "
              f"lc_classes={panel.n_lc_classes} "
              f"windows: train={len(panel.split_origins['train'])} "
              f"test={len(panel.split_origins['test'])} "
              f"val={len(panel.split_origins['validation'])}")
    return panel


def apply_params(cfg, path):
    """Overlay a JSON hyperparameter dict (e.g. output/model/tune/best_params.json)
    onto cfg. Only keys that resolve to a Config field are applied; tuner metadata
    (trial number, metrics) is ignored. Does not set link — it is fixed per study rather
    than tuned, so it is not in best_params.json and must be passed on the CLI."""
    params = json.loads(Path(path).read_text())
    valid = {f.name for f in fields(cfg)}
    applied, skipped = {}, []
    for k, v in params.items():
        if k in valid:
            setattr(cfg, k, v)
            applied[k] = v
        else:
            skipped.append(k)
    print(f"loaded {len(applied)} params from {path}: "
          + ", ".join(f"{k}={v}" for k, v in applied.items()))
    if skipped:
        print(f"  (ignored non-config keys: {', '.join(skipped)})")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check plumbing")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--link", choices=["logit", "cloglog"], default=None,
                    help="link for the occurrence gate and Beta-mean heads")
    ap.add_argument("--county-embed-dim", type=int, default=None,
                    help="width of the learned per-county embedding fed into the GCN input "
                         "(0 = off)")
    ap.add_argument("--teacher-forcing", action="store_true",
                    help="feed the true previous target into the decoder AR channel during training "
                         "(scheduled sampling: anneal tf_ratio_start -> tf_ratio_end)")
    ap.add_argument("--tf-ratio-start", type=float, default=None,
                    help="teacher-forcing ratio at epoch 1 (default 1.0)")
    ap.add_argument("--tf-ratio-end", type=float, default=None,
                    help="teacher-forcing ratio after tf_anneal_epochs (default 0.0)")
    ap.add_argument("--train-on-traintest", action="store_true",
                    help="merge train+test origins for training; monitor on validation; disables early stopping")
    ap.add_argument("--no-eval-validation", action="store_true",
                    help="never touch the validation split (same guarantee model.tune gives). Use for "
                         "any run whose result feeds a model-selection decision; ignored under "
                         "--train-on-traintest, which monitors validation by design")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-harmonics", type=int, default=None,
                    help="Fourier pairs k=1..n encoding seasonality; PCA studies use 6 and it "
                         "is not stored in best_params.json, so pass it alongside --params")
    ap.add_argument("--static-bypass", action="store_true",
                    help="hold the 12 near-constant predictors (terrain, fuel, land cover, human) "
                         "out of the PCA rotation and feed them raw with tail transforms; "
                         "leaves 39 dynamic predictors to rotate")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help=f"where to write best.pt/metrics.json (default {CKPT_DIR})")
    ap.add_argument("--params", type=str, default=None,
                    help="JSON of hyperparameters to overlay onto the config "
                         "(e.g. output/model/tune/best_params.json); explicit flags below override it")
    args = ap.parse_args()

    cfg = Config()
    if args.params is not None: apply_params(cfg, args.params)
    if args.epochs is not None: cfg.max_epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.device is not None: cfg.device = args.device
    if args.link is not None: cfg.link = args.link
    if args.county_embed_dim is not None: cfg.county_embed_dim = args.county_embed_dim
    if args.teacher_forcing: cfg.teacher_forcing = True
    if args.tf_ratio_start is not None: cfg.tf_ratio_start = args.tf_ratio_start
    if args.tf_ratio_end is not None: cfg.tf_ratio_end = args.tf_ratio_end
    if args.n_harmonics is not None: cfg.n_harmonics = args.n_harmonics
    if args.static_bypass: cfg.static_bypass = True
    if args.seed is not None: cfg.seed = args.seed
    if args.smoke:
        cfg.max_epochs, cfg.patience = 2, 99

    panel = load_panel(cfg)
    if args.smoke:  # keep only a few windows for speed
        for s in panel.split_origins:
            panel.split_origins[s] = panel.split_origins[s][:3]

    monitor_split = "test"
    if args.train_on_traintest:
        n_train = len(panel.split_origins["train"])
        n_test  = len(panel.split_origins["test"])
        combined = sorted(panel.split_origins["train"] + panel.split_origins["test"])
        panel.split_origins["train"] = combined
        panel.split_origins["test"]  = []
        cfg.max_epochs += 100
        cfg.patience = 10 ** 9  # disable early stopping; user will Ctrl+C
        monitor_split = "validation"
        print(f"  train+test combined: {n_train} + {n_test} → {len(combined)} origins")
        print(f"  max_epochs → {cfg.max_epochs}  |  early stopping disabled  |  monitoring val NLL")

    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        ckpt_dir = (CKPT_DIR / "smoke") if args.smoke else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "run.log"
    with open(log_path, "w", buffering=1) as log_file:
        best_nll, test_m, val_m, _ = train_model(cfg, panel, ckpt_dir,
                                                  monitor_split=monitor_split,
                                                  log_file=log_file,
                                                  eval_validation=not args.no_eval_validation)
        final_lines = []
        if test_m is not None:
            final_lines.append("FINAL  test: " + str({k: round(v, 4) for k, v in test_m.items() if isinstance(v, float)}))
        if val_m is not None:
            final_lines.append("FINAL  val : " + str({k: round(v, 4) for k, v in val_m.items() if isinstance(v, float)}))
        final_lines.append(f"saved -> {ckpt_dir}/best.pt, metrics.json, config.json")
        for ln in final_lines:
            print(ln)
            log_file.write(ln + "\n")


if __name__ == "__main__":
    main()
