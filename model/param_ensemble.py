"""Parameter-uncertainty members for the two classes whose posterior is analytic.

`model.ensemble` and `model.xgb.seed_ensemble` produce members by *refitting*, because a GNN
and a booster have no closed-form posterior. Climatology and INLA do, so refitting them would
be measuring the wrong thing -- a seed ensemble of a closed-form estimator has zero spread.
This module draws their parameter uncertainty directly and writes it in the member schema
`model.intervals` already consumes, so all four classes reach the same mixture CDF.

Only the *marginal* law of each row's (pi, mu, phi) matters here. The mixture CDF at row i is
`mean_k F(y_i | theta_i^(k))`, which never touches the joint law across rows -- so drawing each
cell independently is exact for this purpose, not an approximation.

**Climatology.** The gate is exact: pi_hat is the mean of a Beta(n_zero + a, n_pos + b)
posterior under the fitted empirical-Bayes prior, so the draw is that Beta. mu gets the
asymptotic law of the pooled statistic that supplied it, at the pool the back-off ladder
actually chose, moment-matched onto a Beta. **phi is pinned**, because its estimator turns out
not to be identified: see `climatology_table`. So the floor's mixture carries gate and location
uncertainty and no precision uncertainty, and its width is a lower bound.

**INLA.** No refit and no `config = TRUE` sampling: the stored marginals are enough. Each
reported (lo, hi) pair is a posterior quantile on the response scale, so back-transforming the
pair through the link and halving its span by 1.96 recovers the link-scale SD of a Gaussian
marginal -- the same near-Gaussian-on-the-link assumption `inla_glm_comparison.qmd` already
makes to Gauss-Hermite its way to `e_y`. Draws are centred on the *shipped point estimate*
rather than on the quantile midpoint, so the contrast against the single fit isolates added
spread instead of mixing in a recentring; the two differ by ~0.2 link SD.

Both gates and both means are drawn independently of each other and of phi. INLA's stored
output carries no cross-component covariance, so this is an approximation, and it is the same
one the write-up states for `e_y`.

    conda run -n pytorch python -m model.param_ensemble climatology --split validation
    conda run -n pytorch python -m model.param_ensemble inla --split validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as sbeta

from .climatology import (MIN_POS_MU, MIN_POS_PHI, _beta_binomial_mle, _resolve,
                          _train_cells, build_rows, fit)
from .config import DATA_DIR
from .data import TARGET_COL

Z95 = 1.959963984540054
EPS = 1e-12
COLS = ["origin_date", "horizon", "target_date", "county_fips", "node_id",
        "y_true", "p_occ", "mu", "phi"]


def _pool_size(tr: pd.DataFrame, cells: pd.MultiIndex, min_count: int) -> np.ndarray:
    """Positives in the pool the back-off ladder chose, per cell.

    `_resolve` picks the first level with >= min_count positives; passing the counts as both
    value and count returns that winning count rather than the statistic it supplied.
    """
    pos = tr[tr[TARGET_COL] > 0]
    g_cell, g_cty, g_mon = (pos.groupby(k)[TARGET_COL]
                            for k in (["county_fips", "month"], "county_fips", "month"))
    levels = [("cell", g_cell.size(), g_cell.size()),
              ("county", g_cty.size(), g_cty.size()),
              ("cal_month", g_mon.size(), g_mon.size()),
              ("global", len(pos), len(pos))]
    return _resolve(cells, levels, min_count)[0]


def climatology_table(k_members: int, seed: int, mu_stat: str) -> list[pd.DataFrame]:
    """`k_members` draws of the per-cell (pi, mu, phi), each in the shape `fit()` returns."""
    table, _ = fit(mu_stat)
    tr = pd.read_parquet(DATA_DIR / "train.parquet",
                         columns=["county_fips", "month", TARGET_COL])
    cells = _train_cells(tr)
    idx = pd.MultiIndex.from_frame(table[["county_fips", "month"]])
    assert idx.equals(cells), "fit() and _train_cells disagree on the cell index"

    grp = tr.assign(is_pos=tr[TARGET_COL] > 0).groupby(["county_fips", "month"])["is_pos"]
    n = grp.size().reindex(cells).to_numpy().astype(float)
    n_zero = (grp.size() - grp.sum()).reindex(cells).to_numpy().astype(float)
    a, b = _beta_binomial_mle(n_zero, n)

    m_mu = _pool_size(tr, cells, MIN_POS_MU)
    m_var = _pool_size(tr, cells, MIN_POS_PHI)

    mu, phi = table.mu.to_numpy(), table.phi.to_numpy()
    alpha, betap = mu * phi, (1 - mu) * phi
    sigma2 = mu * (1 - mu) / (phi + 1.0)

    if mu_stat == "median":
        dens = sbeta.pdf(sbeta.ppf(0.5, alpha, betap), alpha, betap)
        se_mu = 1.0 / (2.0 * np.sqrt(m_mu) * np.maximum(dens, 1e-300))
    else:
        se_mu = np.sqrt(sigma2 / m_mu)
    # Var(s^2) = (mu4 - sigma^4)/m, and mu4 = (excess kurtosis + 3) sigma^4
    g2 = sbeta.stats(alpha, betap, moments="k")
    se_var = np.sqrt(np.maximum(g2 + 2.0, 0.0) * sigma2**2 / m_var)

    # mu is a proportion, so draw it as a Beta moment-matched to (mu_hat, se_mu^2) rather
    # than as a clipped normal.
    nu = np.maximum(mu * (1 - mu) / np.maximum(se_mu**2, EPS) - 1.0, EPS)

    # phi is NOT drawn, and the reason is a finding rather than a convenience. Moment-matching
    # a Gamma to (sigma2, se_var^2) reproduces a direct simulation of the estimator almost
    # exactly -- and what it reproduces is a degenerate estimator. alpha = mu*phi is below 1
    # on most cells, so the pool's Beta is a spike at 0 with a rare large draw; a handful of
    # positives then almost always returns s^2 ~ 0 and phi -> inf. The sampling distribution
    # is right, and it says the floor's precision is unidentified. Pinning phi keeps the
    # mixture interpretable and confines the claim to the two parameters that are identified.
    v_shape = sigma2**2 / np.maximum(se_var**2, EPS)
    print(f"  phi left at its point estimate: the variance estimator's sampling distribution "
          f"has shape < 1 on {float((v_shape < 1).mean()):.1%} of cells "
          f"(median {float(np.median(v_shape)):.2f}), i.e. its mode is at zero and phi diverges")

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k_members):
        pi_k = rng.beta(n_zero + a, n - n_zero + b)
        mu_k = np.clip(rng.beta(mu * nu, (1 - mu) * nu), EPS, 1 - EPS)
        out.append(table.assign(pi=pi_k, mu=mu_k))
    return out


def write_climatology(root: Path, split: str, k_members: int, seed: int, mu_stat: str,
                      lookback: int, horizon: int) -> None:
    rows, _, _ = build_rows(split, lookback, horizon)
    rows["month"] = pd.to_datetime(rows["target_date"]).dt.month
    for k, tbl in enumerate(climatology_table(k_members, seed, mu_stat)):
        d = rows.merge(tbl, on=["county_fips", "month"], how="left", validate="m:1")
        d["p_occ"] = 1.0 - d["pi"]
        dest = root / f"seed_{k:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        d[COLS].to_parquet(dest / f"predictions_{split}.parquet", index=False)
    print(f"wrote {k_members} members x {len(rows):,} rows -> {root}")


def _link_sd(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Link-scale SD implied by a response-scale 95% posterior interval."""
    return (logit(np.clip(hi, EPS, 1 - EPS)) - logit(np.clip(lo, EPS, 1 - EPS))) / (2 * Z95)


def write_inla(root: Path, split: str, k_members: int, seed: int, pred_dir: Path) -> None:
    d = pd.read_parquet(pred_dir / f"predictions_{split}.parquet")
    sd_gate = _link_sd(d.p_occ_lo.to_numpy(), d.p_occ_hi.to_numpy())
    sd_mu = _link_sd(d.mu_lo.to_numpy(), d.mu_hi.to_numpy())
    m_gate = logit(np.clip(d.p_occ.to_numpy(), EPS, 1 - EPS))
    m_mu = logit(np.clip(d.mu.to_numpy(), EPS, 1 - EPS))
    phi = d.phi.to_numpy()
    sd_logphi = d.phi_sd.to_numpy() / phi          # delta method; phi is a positive precision

    rng = np.random.default_rng(seed)
    n = len(d)
    for k in range(k_members):
        out = d[["origin_date", "horizon", "target_date", "county_fips", "node_id",
                 "y_true"]].copy()
        out["p_occ"] = expit(m_gate + sd_gate * rng.standard_normal(n))
        out["mu"] = expit(m_mu + sd_mu * rng.standard_normal(n))
        out["phi"] = phi * np.exp(sd_logphi * rng.standard_normal(n))
        dest = root / f"seed_{k:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        out[COLS].to_parquet(dest / f"predictions_{split}.parquet", index=False)
    print(f"wrote {k_members} members x {n:,} rows -> {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=["climatology", "inla"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred-dir", default=None,
                    help="source predictions dir (default output/<which>)")
    ap.add_argument("--out", default=None, help="member root (default <pred-dir>/param_ensemble)")
    ap.add_argument("--mu-stat", default="median", choices=["median", "mean"])
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(f"output/{args.which}")
    root = Path(args.out) if args.out else pred_dir / "param_ensemble"
    if args.which == "climatology":
        write_climatology(root, args.split, args.members, args.seed, args.mu_stat,
                          args.lookback, args.horizon)
    else:
        write_inla(root, args.split, args.members, args.seed, pred_dir)


if __name__ == "__main__":
    main()
