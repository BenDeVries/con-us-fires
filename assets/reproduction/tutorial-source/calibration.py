"""Calibration checks for the zero-inflated Beta model.

Three diagnostics per split:
  - Conditional Beta coverage: among y_true > 0, empirical coverage of central
    Beta(mu*phi, (1-mu)*phi) intervals at each nominal level (reliability curve).
  - Randomized PIT histogram: the ZIB predictive CDF has an atom pi = 1 - p_occ
    at zero, so for y = 0 draw PIT ~ U(0, pi) and for y > 0 use
    pi + (1-pi) * BetaCDF(y). Uniform under perfect calibration of the FULL
    predictive distribution (gate + Beta jointly).
  - Gate reliability: binned predicted p_occ vs empirical fire frequency with
    Wilson 95% intervals (calibration of the occurrence gate alone).

Run:
  conda run -n fire-nn python calibration.py                    # train + validation
  conda run -n fire-nn python calibration.py --split train      # single split
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta as scipy_beta
from torch.utils.data import DataLoader

from model.config import CKPT_DIR
from model.data import build_panel, WindowDataset, load_pca, apply_saved_pca
from model.model import SpatioTemporalZIB, build_norm_adj
from model.predict import load_cfg
from model.zib import gate_fire_prob


@torch.no_grad()
def collect(split: str, ckpt_dir: Path, device: str):
    cfg = load_cfg(ckpt_dir)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    drop = [] if cfg.n_pca is not None else cfg.drop_features
    n_harm = getattr(cfg, "n_harmonics", 1)
    panel = build_panel(cfg.lookback, cfg.horizon, drop, n_harmonics=n_harm,
                        static_bypass=getattr(cfg, "static_bypass", False))
    if cfg.n_pca is not None:
        pca = load_pca(ckpt_dir / "pca_transform.npz")
        panel = apply_saved_pca(panel, pca, cfg.n_pca)
    A = build_norm_adj(panel.edge_index, panel.n_nodes).to(device)
    model = SpatioTemporalZIB(cfg, panel.cov.shape[-1], panel.n_lc_classes,
                              n_nodes=panel.n_nodes).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    model.eval()

    origins = panel.split_origins[split]
    ds = WindowDataset(panel, origins, cfg.lookback, cfg.horizon)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    P, MU, PHI, Y = [], [], [], []
    for batch in dl:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b, A)
        P.append(gate_fire_prob(out["pi_logit"], cfg.link).cpu().numpy())
        MU.append(out["mu"].cpu().numpy())
        PHI.append(out["phi"].cpu().numpy())
        Y.append(batch["y"].numpy())

    p_occ = np.concatenate(P).ravel()
    mu    = np.concatenate(MU).ravel()
    phi   = np.concatenate(PHI).ravel()
    y     = np.concatenate(Y).ravel()
    return p_occ, mu, phi, y


def beta_coverage(mu: np.ndarray, phi: np.ndarray, y: np.ndarray,
                  level: float = 0.95) -> float:
    """Fraction of y values inside the symmetric Beta interval at `level`."""
    alpha = mu * phi
    beta_ = (1 - mu) * phi
    tail = (1 - level) / 2
    lo = scipy_beta.ppf(tail,   alpha, beta_)
    hi = scipy_beta.ppf(1 - tail, alpha, beta_)
    return float(((y >= lo) & (y <= hi)).mean())


def quantile_coverage_sweep(mu: np.ndarray, phi: np.ndarray, y: np.ndarray,
                             levels: np.ndarray) -> np.ndarray:
    """Empirical coverage at each nominal level — building a reliability curve."""
    alpha = mu * phi
    beta_ = (1 - mu) * phi
    coverage = np.empty(len(levels))
    for i, lev in enumerate(levels):
        tail = (1 - lev) / 2
        lo = scipy_beta.ppf(tail,     alpha, beta_)
        hi = scipy_beta.ppf(1 - tail, alpha, beta_)
        coverage[i] = ((y >= lo) & (y <= hi)).mean()
    return coverage


def randomized_pit(p_occ: np.ndarray, mu: np.ndarray, phi: np.ndarray,
                   y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT for the ZIB predictive CDF (atom pi = 1 - p_occ at zero).

    y = 0  →  PIT ~ U(0, pi);   y > 0  →  PIT = pi + (1-pi) * BetaCDF(y).
    Uniform(0, 1) under perfect calibration.
    """
    pi = 1.0 - p_occ
    return np.where(
        y > 0,
        pi + (1 - pi) * scipy_beta.cdf(y, mu * phi, (1 - mu) * phi),
        rng.uniform(size=y.shape) * pi,
    )


def report_split(split: str, p_occ: np.ndarray, mu: np.ndarray, phi: np.ndarray,
                 y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Print calibration report for one split and return empirical coverage array."""
    n_total = len(y)
    pos_mask = y > 0
    n_pos = pos_mask.sum()

    print(f"{'─'*55}")
    print(f"Split: {split}  |  total obs: {n_total:,}  |  y>0: {n_pos:,} ({n_pos/n_total:.1%})")
    print(f"{'─'*55}")

    cov_95 = beta_coverage(mu[pos_mask], phi[pos_mask], y[pos_mask], level=0.95)
    cov_50 = beta_coverage(mu[pos_mask], phi[pos_mask], y[pos_mask], level=0.50)
    print(f"  Nominal 50% → empirical {cov_50:.1%}  "
          f"({'over' if cov_50 > 0.50 else 'under'}-covered by {abs(cov_50-0.50):.1%})")
    print(f"  Nominal 95% → empirical {cov_95:.1%}  "
          f"({'over' if cov_95 > 0.95 else 'under'}-covered by {abs(cov_95-0.95):.1%})")

    emp_cov = quantile_coverage_sweep(mu[pos_mask], phi[pos_mask], y[pos_mask], levels)

    print(f"\n  {'Nominal':>8}  {'Empirical':>10}  {'Δ':>8}")
    for lev, emp in zip(levels, emp_cov):
        sign = "+" if emp > lev else "-"
        print(f"  {lev:8.0%}  {emp:10.1%}  {sign}{abs(emp-lev):.1%}")

    phi_pos = phi[pos_mask]
    print(f"\n  φ (y>0): mean={phi_pos.mean():.2f}  median={np.median(phi_pos):.2f}  "
          f"p5={np.percentile(phi_pos,5):.2f}  p95={np.percentile(phi_pos,95):.2f}")

    return emp_cov


SPLIT_COLORS = {"train": "#2166ac", "validation": "#d6604d", "test": "#4dac26"}


def plot_calibration(levels: np.ndarray, curves: dict[str, np.ndarray],
                     out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, label="Perfect calibration")

    for split, emp in curves.items():
        ax.plot(levels, emp, "o-", ms=4, lw=1.5,
                color=SPLIT_COLORS.get(split, "grey"), label=split.capitalize())
        ax.fill_between(levels, levels, emp,
                        alpha=0.08, color=SPLIT_COLORS.get(split, "grey"))

    ax.set_xlabel("Nominal coverage (central Beta interval)", fontsize=11)
    ax.set_ylabel("Empirical coverage  (y > 0)", fontsize=11)
    ax.set_title("Beta head calibration curve\n(conditional on fire occurring)", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nCalibration curve → {out_path}")


def plot_pit(pits: dict[str, np.ndarray], out_path: Path, bins: int = 20) -> None:
    fig, axes = plt.subplots(1, len(pits), figsize=(5 * len(pits), 4), squeeze=False)
    for ax, (split, pit) in zip(axes[0], pits.items()):
        ax.hist(pit, bins=bins, range=(0, 1), density=True,
                color=SPLIT_COLORS.get(split, "grey"), alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="black", ls="--", lw=1, label="Uniform (perfect)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Randomized PIT", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{split.capitalize()}  (n={len(pit):,})", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Randomized PIT — full ZIB predictive distribution", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PIT histogram → {out_path}")


def _wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson 95% interval for a binomial proportion (vectorised)."""
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def plot_gate_reliability(gate: dict[str, tuple[np.ndarray, np.ndarray]],
                          out_path: Path, n_bins: int = 15) -> None:
    """Reliability of the occurrence gate: binned p_occ vs empirical P(y > 0)."""
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, label="Perfect calibration")
    for split, (p_occ, occ) in gate.items():
        edges = np.unique(np.quantile(p_occ, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.digitize(p_occ, edges[1:-1]), 0, len(edges) - 2)
        nb = len(edges) - 1
        n_k = np.bincount(idx, minlength=nb).astype(float)
        k_k = np.bincount(idx, weights=occ.astype(float), minlength=nb)
        m = n_k > 0
        n_k, k_k = n_k[m], k_k[m]
        p_k = np.bincount(idx, weights=p_occ, minlength=nb)[m] / n_k
        emp = k_k / n_k
        lo, hi = _wilson(k_k, n_k)
        c = SPLIT_COLORS.get(split, "grey")
        ax.errorbar(p_k, emp, yerr=[emp - lo, hi - emp], fmt="o-", ms=4, lw=1.2,
                    capsize=2, color=c, label=split.capitalize())
        axh.hist(p_occ, bins=40, range=(0, 1), color=c, alpha=0.5,
                 label=split.capitalize())
    ax.set_ylabel("Empirical P(y > 0)", fontsize=11)
    ax.set_title("Gate reliability — equal-count bins, Wilson 95% CI", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=9)
    axh.set_yscale("log")
    axh.set_xlabel("Predicted P(fire occurs)", fontsize=11)
    axh.set_ylabel("Count", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gate reliability → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None,
                    choices=["train", "test", "validation"],
                    help="single split to evaluate; omit for train + validation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    cfg = load_cfg(ckpt_dir)
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Model: {ckpt_dir}  device: {device}")

    levels = np.arange(0.05, 1.0, 0.05)
    splits = [args.split] if args.split else ["train", "validation"]

    rng = np.random.default_rng(0)
    curves: dict[str, np.ndarray] = {}
    pits: dict[str, np.ndarray] = {}
    gate: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in splits:
        print(f"\nLoading {split} split …")
        p_occ, mu, phi, y = collect(split, ckpt_dir, device)
        curves[split] = report_split(split, p_occ, mu, phi, y, levels)
        pits[split] = randomized_pit(p_occ, mu, phi, y, rng)
        gate[split] = (p_occ, y > 0)

    fig_dir = Path("output/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(levels, curves, fig_dir / "beta_calibration.png")
    plot_pit(pits, fig_dir / "pit_hist.png")
    plot_gate_reliability(gate, fig_dir / "gate_reliability.png")


if __name__ == "__main__":
    main()
