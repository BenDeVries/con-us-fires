"""Coherent prediction intervals for the zero-inflated Beta predictive, with and without
parameter uncertainty.

An interval is *coherent* here if it is a quantile pair of the predictive distribution the
model actually emits -- atom of mass pi at y = 0, Beta(mu*phi, (1-mu)*phi) above it. The
Beta-only curves in `calibration.py` are conditional on a fire and so are not that: they
ignore the 92% of the mass sitting on zero.

Parameter uncertainty enters as a **mixture over fits**, never as a spread on (pi, mu, phi):

    F(y) = mean_k F_k(y)

Averaging the parameters would give a single ZIB that is not the mixture of the members'
ZIBs and would produce an interval that is too narrow -- the same error `model.ensemble`
avoids on the density. Members come from `model.ensemble` (GNN, 10 seeds) or
`model.xgb.seed_ensemble` (xgb, 10 seeds); either directory works, since both write
`predictions_<split>.parquet` in the same schema.

Two coverage curves are reported and they answer different questions:

* **Randomized PIT coverage** is the calibration diagnostic. For a distribution with an atom
  it is the only version that is exactly uniform under correct specification, so a deviation
  is evidence of misfit rather than of discreteness.
* **Deterministic coverage** is what the shipped interval delivers. Because a zero row always
  falls inside an interval whose lower endpoint is 0, it is conservative by construction and
  will read high. Quote it when the claim is operational and the PIT version when the claim
  is about calibration.

Both come from a single pass of the Beta CDF: for y > 0 the non-randomized PIT *is* F(y), and
`y <= q(p)` iff `F(y) <= p`, so no root-finding is needed for either curve. Root-finding is
needed only for the interval endpoints themselves, which is why `--width-level` defaults to a
single level.

    conda run -n pytorch python -m model.intervals \\
      --members output/model/v2_trial023/ensemble --label "GNN v2_t23"
    conda run -n pytorch python -m model.intervals \\
      --members output/xgb_spacetime/seed_ensemble --label "xgb space-time"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta

LEVELS = np.arange(0.05, 1.00, 0.05)


def load_members(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Stack every member's (pi, mu, phi) on a shared row order. Returns [K, n] arrays + y + keys."""
    paths = sorted(root.glob(f"seed_*/predictions_{split}.parquet"))
    if not paths:
        raise SystemExit(f"no member predictions under {root}")
    base = pd.read_parquet(paths[0])
    keys = base[["origin_date", "horizon", "county_fips"]]
    y = base.y_true.to_numpy(np.float64)
    PI, MU, PHI = [], [], []
    for p in paths:
        d = pd.read_parquet(p)
        assert d[["origin_date", "horizon", "county_fips"]].equals(keys), f"{p} row set differs"
        PI.append(1.0 - d.p_occ.to_numpy(np.float64))
        MU.append(d.mu.to_numpy(np.float64))
        PHI.append(d.phi.to_numpy(np.float64))
    return (np.stack(PI), np.stack(MU), np.stack(PHI), y, [p.parent.name for p in paths], keys)


def mixture_cdf_pos(y: np.ndarray, PI: np.ndarray, MU: np.ndarray, PHI: np.ndarray) -> np.ndarray:
    """F(y) = mean_k [pi_k + (1-pi_k) F_beta_k(y)], evaluated row-wise over [K, n] parameters.

    Correct for y > 0. At y = 0 this returns mean_k pi_k, the top of the atom, so callers
    handling zero rows must decide between that and the randomized draw below it.
    """
    out = np.zeros_like(y)
    for k in range(PI.shape[0]):
        cdf = scipy_beta.cdf(y, MU[k] * PHI[k], (1 - MU[k]) * PHI[k])
        out += PI[k] + (1 - PI[k]) * cdf
    return out / PI.shape[0]


def mixture_quantile(p: float, PI: np.ndarray, MU: np.ndarray, PHI: np.ndarray,
                     iters: int = 40) -> np.ndarray:
    """q(p) of the mixture, by bisection on (0, 1). Rows with p <= mean_k pi_k return 0."""
    pi_bar = PI.mean(axis=0)
    lo = np.zeros_like(pi_bar)
    hi = np.ones_like(pi_bar)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mixture_cdf_pos(mid, PI, MU, PHI)
        too_low = f < p
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return np.where(p <= pi_bar, 0.0, 0.5 * (lo + hi))


def coverage_curves(F_pos: np.ndarray, pi_bar: np.ndarray, y: np.ndarray,
                    rng: np.random.Generator) -> dict:
    """Randomized-PIT and deterministic coverage at every level, from one CDF pass.

    `F_pos` must be the mixture CDF evaluated at the observed y (only its y > 0 entries are
    read). A zero row's randomized PIT is U(0, pi_bar); its deterministic membership is
    automatic, since the interval's lower endpoint is 0 whenever (1-L)/2 <= pi_bar.
    """
    pos = y > 0
    u = np.where(pos, F_pos, rng.uniform(size=y.shape) * pi_bar)
    pit_cov, det_cov = [], []
    for lev in LEVELS:
        a = (1 - lev) / 2
        pit_cov.append(float(((u >= a) & (u <= 1 - a)).mean()))
        # deterministic: y=0 is inside iff lo=0, i.e. iff a <= pi_bar; y>0 iff F(y) <= 1-a
        inside = np.where(pos, F_pos <= 1 - a, a <= pi_bar)
        det_cov.append(float(inside.mean()))
    return {"levels": [round(float(l), 2) for l in LEVELS],
            "pit_coverage": pit_cov, "det_coverage": det_cov,
            "pit_max_dev": float(np.max(np.abs(np.array(pit_cov) - LEVELS)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True,
                    help="ensemble root holding seed_*/predictions_<split>.parquet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--label", required=True)
    ap.add_argument("--width-level", type=float, default=0.95,
                    help="level at which interval endpoints and widths are computed")
    ap.add_argument("--single", default=None,
                    help="predictions dir supplying the single-fit baseline. Needed for the "
                         "classes whose members are posterior draws rather than refits, where "
                         "member 0 is a draw and the shipped point estimate is a separate file; "
                         "defaults to member 0, which is right for a seed ensemble.")
    ap.add_argument("--seed", type=int, default=0, help="seed for the PIT randomization")
    ap.add_argument("--out", default=None, help="output json (defaults to <members>/intervals.json)")
    args = ap.parse_args()

    root = Path(args.members)
    PI, MU, PHI, y, names, keys = load_members(root, args.split)
    K, n = PI.shape
    print(f"{args.label}: {K} members x {n:,} {args.split} rows")

    rng = np.random.default_rng(args.seed)
    out = {"label": args.label, "split": args.split, "members": names, "n_rows": int(n)}

    # single fit = member 0, the shipped seed; the contrast isolates what the mixture buys
    if args.single:
        s = pd.read_parquet(Path(args.single) / f"predictions_{args.split}.parquet")
        s = keys.merge(s, on=list(keys.columns), how="left", validate="1:1")
        assert np.allclose(s.y_true.to_numpy(), y), "single-fit row set differs from the members'"
        single = {"PI": (1.0 - s.p_occ.to_numpy(np.float64))[None, :],
                  "MU": s.mu.to_numpy(np.float64)[None, :],
                  "PHI": s.phi.to_numpy(np.float64)[None, :]}
        out["single_fit_dir"] = args.single
    else:
        single = {"PI": PI[:1], "MU": MU[:1], "PHI": PHI[:1]}
    F1 = mixture_cdf_pos(y, single["PI"], single["MU"], single["PHI"])
    out["single_fit"] = coverage_curves(F1, single["PI"].mean(axis=0), y,
                                        np.random.default_rng(args.seed))
    FK = mixture_cdf_pos(y, PI, MU, PHI)
    out["mixture"] = coverage_curves(FK, PI.mean(axis=0), y,
                                     np.random.default_rng(args.seed))

    a = (1 - args.width_level) / 2
    for tag, (p_, m_, h_) in (("single_fit", (single["PI"], single["MU"], single["PHI"])),
                              ("mixture", (PI, MU, PHI))):
        hi = mixture_quantile(1 - a, p_, m_, h_)
        lo = mixture_quantile(a, p_, m_, h_)
        w = hi - lo
        out[tag]["width_level"] = args.width_level
        out[tag]["width"] = {"median": float(np.median(w)), "mean": float(w.mean()),
                             "frac_lo_zero": float((lo == 0).mean()),
                             "median_pos": float(np.median(w[y > 0]))}

    dest = Path(args.out) if args.out else root / "intervals.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"  {'level':>6} {'single PIT':>11} {'mixture PIT':>12} {'single det':>11} {'mix det':>9}")
    for i, lev in enumerate(out["single_fit"]["levels"]):
        if round(lev * 100) % 25 == 0 or lev == 0.95:
            print(f"  {lev:6.2f} {out['single_fit']['pit_coverage'][i]:11.4f} "
                  f"{out['mixture']['pit_coverage'][i]:12.4f} "
                  f"{out['single_fit']['det_coverage'][i]:11.4f} "
                  f"{out['mixture']['det_coverage'][i]:9.4f}")
    print(f"  max |PIT dev|  single {out['single_fit']['pit_max_dev']:.4f}  "
          f"mixture {out['mixture']['pit_max_dev']:.4f}")
    print(f"  median {args.width_level:.0%} width (positives)  "
          f"single {out['single_fit']['width']['median_pos']:.3e}  "
          f"mixture {out['mixture']['width']['median_pos']:.3e}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
