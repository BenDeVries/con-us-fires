"""Model-agnostic regression diagnostics for zero-inflated Beta forecasts.

Reads a predictions parquet and nothing else about the model that produced it, so the
same command runs on the gradient-boosted, neural and latent-Gaussian baselines and the
figures are directly comparable:

    conda run -n pytorch python -m model.diagnostics --pred-dir output/xgb --split test --label "XGB (raw)"
    conda run -n pytorch python -m model.diagnostics --compare output/xgb output/xgb_pca --split test

Required columns: origin_date, horizon, target_date, county_fips, node_id, y_true,
p_occ, mu, phi. Optional: e_y (recomputed as p_occ*mu when absent).

**The unifying object is the Dunn-Smyth randomized quantile residual.** The ZIB
predictive CDF has an atom at zero, so the ordinary residual has no reference
distribution. Draw the randomized PIT u (uniform on (0, 1-p_occ) when y = 0, equal to
the continuous CDF otherwise) and set

    r = Phi^{-1}(u).

Under correct specification r ~ N(0,1) independently across rows. That -- and only that
-- is what licenses applying the classical battery (QQ plots, ACF, Moran's I,
residual-versus-fitted) to a hurdle model. Every check in sections A-E below is a check
on r; section F compares models on proper scores instead.

Two cautions the figures are drawn to respect. First, n is 559,440 on test, so every
omnibus test rejects: read the KS and chi-square *statistics* as effect sizes and the
p-values as formalities. Second, r is randomized, so a single realisation carries
Monte-Carlo noise; --seed controls it and the residual summary is reproducible.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import beta as sbeta, chi2, kstwobign, norm
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from calibration import quantile_coverage_sweep, randomized_pit, _wilson
from .spatial import exclusive_orders, moran_i, morans_permutation, row_norm_W

DATA_DIR = Path("output/data")
OUT_ROOT = Path("output/diagnostics")
EPS = 1e-12
KEYS = ["origin_date", "horizon", "target_date", "county_fips", "node_id"]
NEEDED = KEYS + ["y_true", "p_occ", "mu", "phi"]

REGION = {  # census region by state FIPS, for coarse bias stratification
    **{s: "Northeast" for s in ("09", "23", "25", "33", "34", "36", "42", "44", "50")},
    **{s: "Midwest" for s in ("17", "18", "19", "20", "26", "27", "29", "31", "38",
                              "39", "46", "55")},
    **{s: "South" for s in ("01", "05", "10", "11", "12", "13", "21", "22", "24", "28",
                            "37", "40", "45", "47", "48", "51", "54")},
    **{s: "West" for s in ("04", "06", "08", "16", "30", "32", "35", "41", "49", "53",
                           "56")},
}
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#b2abd2", "#f4a582"]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "model"


# --------------------------------------------------------------------------- loading

def load_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    path = Path(pred_dir) / f"predictions_{split}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_parquet(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} lacks required columns {missing}; it may be an older "
                         f"nowcast file (columns present: {list(df.columns)})")
    cols = NEEDED + (["e_y"] if "e_y" in df.columns else [])
    df = df[cols].copy()
    if "e_y" not in df.columns:
        df["e_y"] = df["p_occ"] * df["mu"]
    for c in ("origin_date", "target_date"):
        df[c] = pd.to_datetime(df[c])
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    # section_comparison's row-set guard is a dtype-sensitive .equals; xgb writes
    # horizon as int16 and the other classes as int64.
    df["horizon"] = df["horizon"].astype("int64")
    df["mu"] = df["mu"].clip(1e-9, 1 - 1e-9)
    df["p_occ"] = df["p_occ"].clip(1e-9, 1 - 1e-9)
    return df


def add_residuals(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = randomized_pit(df["p_occ"].to_numpy(), df["mu"].to_numpy(),
                       df["phi"].to_numpy(), df["y_true"].to_numpy(), rng)
    df["pit"] = np.clip(u, EPS, 1 - EPS)
    df["r"] = norm.ppf(df["pit"].to_numpy())
    return df


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Negative log score and CRPS per row (both lower-is-better, in the same units as
    `model/zib.py:zib_nll`, which is a mean over rows of the same quantity).

    Matches `zib_nll` row for row: both mask the Beta term on zero rows rather than
    clamping y into it, so neither censors the sub-1e-6 positives and the two agree."""
    mu, phi, p = (df[c].to_numpy() for c in ("mu", "phi", "p_occ"))
    y = df["y_true"].to_numpy()
    a, b = mu * phi, (1 - mu) * phi
    pos = y > 0
    ls = np.where(pos,
                  np.log(p) + sbeta.logpdf(np.where(pos, y, 0.5), a, b),
                  np.log1p(-p))
    df["nll"] = -ls
    df["crps"] = zib_crps(p, mu, phi, y)
    return df


def zib_crps(p_occ, mu, phi, y):
    """CRPS of the zero-adjusted Beta in closed form.

    Uses the kernel identity CRPS = E|X-y| - E|X-X'|/2 for X, X' iid from the forecast.
    With X = 0 w.p. pi and Beta(a,b) otherwise,

        E|X-y|  = pi*y + (1-pi)[ y(2F_{a,b}(y) - 1) + mu(1 - 2F_{a+1,b}(y)) ]
        E|X-X'| = 2 pi (1-pi) mu + (1-pi)^2 * 4 B(2a,2b) / ((a+b) B(a,b)^2),

    the second term being the Gini mean difference of a Beta (it evaluates to 1/3 for
    a = b = 1, the uniform case, which is the check used in the unit test).
    """
    a, b = mu * phi, (1 - mu) * phi
    pi = 1.0 - p_occ
    F = sbeta.cdf(y, a, b)
    F1 = sbeta.cdf(y, a + 1, b)
    e_xy = pi * y + (1 - pi) * (y * (2 * F - 1) + mu * (1 - 2 * F1))
    gmd = 4.0 * np.exp(betaln(2 * a, 2 * b) - 2 * betaln(a, b)) / (a + b)
    e_xx = 2 * pi * (1 - pi) * mu + (1 - pi) ** 2 * gmd
    return e_xy - 0.5 * e_xx


# ------------------------------------------------------------------- small estimators

def qbin(x: np.ndarray, n_bins: int):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int), edges
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2), edges


def boot_mean_ci(x: np.ndarray, rng, B: int = 200, alpha: float = 0.05):
    """Bootstrap CI for a mean. Above 1e5 observations the bootstrap and the normal
    approximation agree to plotting precision, so the cheaper form is used there."""
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan, np.nan
    m = float(x.mean())
    if n > 100_000:
        se = float(x.std(ddof=1)) / np.sqrt(n)
        return m, m - 1.96 * se, m + 1.96 * se
    draws = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return m, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def binned_mean(x: np.ndarray, v: np.ndarray, n_bins: int, rng):
    """Mean of v within equal-count bins of x, with bootstrap intervals."""
    idx, _ = qbin(x, n_bins)
    rows = []
    for k in range(idx.max() + 1):
        m = idx == k
        if not m.any():
            continue
        mean, lo, hi = boot_mean_ci(v[m], rng)
        rows.append((float(x[m].mean()), mean, lo, hi, int(m.sum())))
    return np.array(rows).T if rows else np.zeros((5, 0))


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    z = x - x.mean()
    den = float((z * z).sum())
    if den == 0:
        return np.zeros(nlags + 1)
    return np.array([1.0] + [float((z[k:] * z[:-k]).sum()) / den
                             for k in range(1, nlags + 1)])


def pacf_from_acf(r: np.ndarray) -> np.ndarray:
    """Partial autocorrelations by the Durbin-Levinson recursion."""
    K = len(r) - 1
    out = np.zeros(K + 1)
    out[0] = 1.0
    phi = np.zeros((K + 1, K + 1))
    if K >= 1:
        phi[1, 1] = out[1] = r[1]
        for k in range(2, K + 1):
            num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
            phi[k, k] = out[k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return out


def ljung_box(r: np.ndarray, n: int, h: int):
    k = np.arange(1, h + 1)
    q = float(n * (n + 2) * np.sum(r[1:h + 1] ** 2 / (n - k)))
    return q, float(chi2.sf(q, h))


def durbin_watson(x: np.ndarray) -> float:
    z = x - x.mean()
    return float((np.diff(z) ** 2).sum() / (z * z).sum())


def dm_hac(d: np.ndarray, lag: int | None = None):
    """Diebold-Mariano on a per-month mean score difference with Newey-West variance."""
    T = len(d)
    dbar = float(d.mean())
    e = d - dbar
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    s = float((e * e).sum()) / T
    for L in range(1, lag + 1):
        s += 2.0 * (1 - L / (lag + 1.0)) * float((e[L:] * e[:-L]).sum()) / T
    se = np.sqrt(max(s, 1e-30) / T)
    stat = dbar / se
    return {"mean_diff": dbar, "dm_stat": float(stat),
            "p_value": float(2 * norm.sf(abs(stat))), "nw_lag": int(lag), "n_months": T}


def month_block_boot(sums: np.ndarray, counts: np.ndarray, block: int = 3,
                     B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap over whole target months, so all horizons of a month move
    together. `sums`/`counts` hold the per-month score-difference sum and row count, so
    each resample recomputes the exact row-weighted mean."""
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


def brier_decomp(p: np.ndarray, o: np.ndarray, n_bins: int = 20) -> dict:
    """Murphy decomposition BS = reliability - resolution + uncertainty."""
    idx, _ = qbin(p, n_bins)
    nb = idx.max() + 1
    n_k = np.bincount(idx, minlength=nb).astype(float)
    o_k = np.bincount(idx, weights=o, minlength=nb)
    p_k = np.bincount(idx, weights=p, minlength=nb)
    m = n_k > 0
    n_k, obar_k, pbar_k = n_k[m], o_k[m] / n_k[m], p_k[m] / n_k[m]
    obar = float(o.mean())
    n = float(len(p))
    return {"brier": float(((p - o) ** 2).mean()),
            "reliability": float((n_k * (pbar_k - obar_k) ** 2).sum() / n),
            "resolution": float((n_k * (obar_k - obar) ** 2).sum() / n),
            "uncertainty": float(obar * (1 - obar))}


def local_moran(z: np.ndarray, W: np.ndarray, n_perm: int = 499, seed: int = 0):
    """Local Moran's I_i with a conditional-permutation two-sided p-value."""
    n = len(z)
    zs = z - z.mean()
    m2 = float((zs * zs).sum()) / n
    Ii = zs * (W @ zs) / m2
    rng = np.random.default_rng(seed)
    p = np.ones(n)
    for i in range(n):
        nb = np.flatnonzero(W[i])
        if nb.size == 0:
            continue
        pool = np.delete(zs, i)
        samp = rng.choice(pool, size=(n_perm, nb.size), replace=True)
        sim = zs[i] * (samp * W[i, nb]).sum(axis=1) / m2
        p[i] = (1 + int((np.abs(sim) >= abs(Ii[i])).sum())) / (1 + n_perm)
    return Ii, p


# ------------------------------------------------------------------------- geometry

def load_geom():
    path = DATA_DIR / "county_geom.parquet"
    if not path.exists():
        return None
    import geopandas as gpd
    g = gpd.read_parquet(path)
    g["county_fips"] = g["county_fips"].astype(str).str.zfill(5)
    return g.to_crs(5070)


def choropleth(ax, geom, values: pd.Series, title: str, cmap="RdBu_r",
               vlim: float | None = None):
    g = geom.merge(values.rename("v"), left_on="county_fips", right_index=True, how="left")
    if vlim is None:
        vlim = float(np.nanquantile(np.abs(g["v"]), 0.98)) or 1.0
    g.plot(column="v", ax=ax, cmap=cmap, vmin=-vlim, vmax=vlim, linewidth=0,
           legend=True, legend_kwds={"shrink": 0.55},
           missing_kwds={"color": "0.9"})
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


# --------------------------------------------------------------- A. distributional

def section_distributional(df, out_dir, rng) -> dict:
    r = df["r"].to_numpy()
    n = len(r)
    stats = {}

    order = np.sort(r)
    pp = (np.arange(1, n + 1) - 0.5) / n
    theo = norm.ppf(pp)
    cdf = norm.cdf(order)
    i = np.arange(1, n + 1)
    ks = float(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)))
    stats["ks_stat"] = ks
    stats["ks_p"] = float(kstwobign.sf(np.sqrt(n) * ks))
    stats["r_mean"], stats["r_sd"] = float(r.mean()), float(r.std(ddof=1))
    stats["r_skew"] = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    stats["r_kurtosis"] = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)

    step = max(1, n // 4000)
    band = 1.36 / np.sqrt(n)          # KS-based simultaneous 95% envelope
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(theo[::step], order[::step], ".", ms=2.5, color=PALETTE[0])
    lim = [min(theo[0], order[0]), max(theo[-1], order[-1])]
    ax.plot(lim, lim, "--", color="black", lw=1)
    ax.fill_between(theo[::step],
                    norm.ppf(np.clip(pp[::step] - band, 1e-12, 1 - 1e-12)),
                    norm.ppf(np.clip(pp[::step] + band, 1e-12, 1 - 1e-12)),
                    color="grey", alpha=0.25, lw=0, label="95% simultaneous band")
    ax.set_xlabel("N(0,1) quantile")
    ax.set_ylabel("Randomized quantile residual")
    ax.set_title(f"Normal QQ plot of $r$\nKS = {ks:.4f},  n = {n:,}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "A1_qq_residual.png", dpi=150); plt.close(fig)

    nb = 20
    counts, edges = np.histogram(df["pit"], bins=nb, range=(0, 1))
    exp = n / nb
    chi = float(((counts - exp) ** 2 / exp).sum())
    stats["pit_chi2"] = chi
    stats["pit_chi2_p"] = float(chi2.sf(chi, nb - 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(edges[:-1], counts / n * nb, width=1 / nb, align="edge",
                color=PALETTE[0], alpha=0.85, edgecolor="white")
    axes[0].axhline(1.0, color="black", ls="--", lw=1)
    axes[0].set_xlabel("Randomized PIT"); axes[0].set_ylabel("Density")
    axes[0].set_title(f"PIT uniformity: $\\chi^2_{{{nb-1}}}$ = {chi:,.0f}", fontsize=10)

    pos = df["y_true"].to_numpy() > 0
    levels = np.arange(0.05, 1.0, 0.05)
    cov = quantile_coverage_sweep(df["mu"].to_numpy()[pos], df["phi"].to_numpy()[pos],
                                  df["y_true"].to_numpy()[pos], levels)
    stats["beta_coverage"] = {f"{l:.2f}": float(c) for l, c in zip(levels, cov)}
    axes[1].plot([0, 1], [0, 1], "--", color="black", lw=1)
    axes[1].plot(levels, cov, "o-", ms=4, color=PALETTE[1])
    axes[1].set_xlabel("Nominal central Beta interval")
    axes[1].set_ylabel("Empirical coverage ($y>0$)")
    axes[1].set_title(f"Beta head coverage  (n = {pos.sum():,})", fontsize=10)
    axes[1].grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "A2_pit_coverage.png", dpi=150); plt.close(fig)

    p = df["p_occ"].to_numpy()
    o = pos.astype(float)
    idx, _ = qbin(p, 15)
    nb2 = idx.max() + 1
    n_k = np.bincount(idx, minlength=nb2).astype(float)
    k_k = np.bincount(idx, weights=o, minlength=nb2)
    p_k = np.bincount(idx, weights=p, minlength=nb2)
    m = n_k > 0
    emp = k_k[m] / n_k[m]
    lo, hi = _wilson(k_k[m], n_k[m])
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.2, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1)
    # Wilson always brackets p-hat, so a negative half-width is rounding (an all-zero
    # bin puts lo at +1e-22 instead of 0) and matplotlib rejects it.
    ax.errorbar(p_k[m] / n_k[m], emp, yerr=[np.clip(emp - lo, 0, None),
                                            np.clip(hi - emp, 0, None)], fmt="o-", ms=4,
                capsize=2, color=PALETTE[0])
    ax.set_ylabel("Empirical $P(y>0)$")
    ax.set_title("Gate reliability, equal-count bins, Wilson 95%", fontsize=10)
    ax.grid(lw=0.3, alpha=0.5)
    axh.hist(p, bins=40, range=(0, 1), color=PALETTE[0], alpha=0.6)
    axh.set_yscale("log"); axh.set_xlabel("Predicted $P(\\mathrm{fire})$"); axh.set_ylabel("Count")
    fig.savefig(out_dir / "A3_gate_reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats["gate_reliability"] = {"p_pred": (p_k[m] / n_k[m]).tolist(),
                                 "p_emp": emp.tolist(), "n": n_k[m].tolist()}
    return stats


# ---------------------------------------------------------------------- B. spatial

def section_spatial(df, out_dir, geom, n_perm: int):
    """Returns (stats, per-county mean residual indexed by county_fips)."""
    stats = {}
    by_county = df.groupby("county_fips", observed=True)["r"].mean()
    node = df.groupby("county_fips", observed=True)["node_id"].first().astype(int)
    edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
    n_nodes = int(max(edge.max() + 1, node.max() + 1))
    fips_by_node = {int(v): k for k, v in node.items()}
    node_index = pd.Index([fips_by_node.get(i, f"__{i}") for i in range(n_nodes)],
                          name="county_fips")

    rbar = np.full(n_nodes, np.nan)
    rbar[node.to_numpy()] = by_county.to_numpy()
    present = ~np.isnan(rbar)
    rbar = np.where(present, rbar, np.nanmean(rbar))

    W, _ = row_norm_W(edge, n_nodes)
    obs, pval, perms = morans_permutation(rbar, W, n_perm=n_perm, seed=0)
    stats["moran_i"] = float(obs)
    stats["moran_p_perm"] = float(pval)
    stats["moran_expected_null"] = -1.0 / (n_nodes - 1)
    stats["moran_perm_sd"] = float(perms.std(ddof=1))
    stats["n_counties"] = int(present.sum())

    lag = W @ rbar
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(rbar, lag, ".", ms=2, alpha=0.5, color=PALETTE[0])
    b = np.polyfit(rbar, lag, 1)
    xs = np.linspace(rbar.min(), rbar.max(), 10)
    axes[0].plot(xs, np.polyval(b, xs), "-", color=PALETTE[1], lw=1.5)
    axes[0].axhline(lag.mean(), color="grey", lw=0.6)
    axes[0].axvline(rbar.mean(), color="grey", lw=0.6)
    axes[0].set_xlabel(r"$\bar r_i$"); axes[0].set_ylabel(r"$\mathbf{W}\bar r_i$")
    axes[0].set_title(f"Moran scatterplot: I = {obs:.3f} (p = {pval:.3f})", fontsize=10)
    axes[1].hist(perms, bins=40, color="0.7", edgecolor="white")
    axes[1].axvline(obs, color=PALETTE[1], lw=2, label="observed")
    axes[1].axvline(-1 / (n_nodes - 1), color="black", ls="--", lw=1, label="$-1/(n-1)$")
    axes[1].set_xlabel("Moran's I"); axes[1].set_title(f"{n_perm} permutations", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "B1_moran.png", dpi=150); plt.close(fig)

    Ii, p_loc = local_moran(rbar, W, n_perm=499, seed=0)
    stats["local_moran_sig_frac"] = float((p_loc < 0.05).mean())

    if geom is not None:
        idx = node_index
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        choropleth(axes[0], geom, pd.Series(rbar, index=idx),
                   r"Mean randomized quantile residual $\bar r_i$")
        zs = rbar - rbar.mean()
        cat = np.where(p_loc >= 0.05, 0,
                       np.where((zs > 0) & (W @ zs > 0), 1,
                                np.where((zs < 0) & (W @ zs < 0), 2, 3)))
        g = geom.merge(pd.Series(cat, index=idx, name="c"), left_on="county_fips",
                       right_index=True, how="left")
        colors = {0: "0.9", 1: "#b2182b", 2: "#2166ac", 3: "#f7f7a0"}
        labels = {0: "not significant", 1: "high-high", 2: "low-low",
                  3: "high-low / low-high"}
        # GeoDataFrame.plot draws a PatchCollection, which legend() cannot use as a
        # handle, so build proxy patches instead.
        from matplotlib.patches import Patch
        handles = []
        for k, lab in labels.items():
            sub = g[g["c"] == k]
            if len(sub):
                sub.plot(ax=axes[1], color=colors[k], linewidth=0)
                handles.append(Patch(facecolor=colors[k], label=lab))
        axes[1].legend(handles=handles, fontsize=8, loc="lower left")
        axes[1].set_title("Local Moran clusters (conditional permutation, p < 0.05)",
                          fontsize=10)
        axes[1].set_axis_off()
        fig.tight_layout(); fig.savefig(out_dir / "B2_residual_map.png", dpi=150)
        plt.close(fig)

        cen = geom.set_index("county_fips").geometry.centroid
        xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()]) / 1000.0
        pos_map = {f: j for j, f in enumerate(cen.index)}
        keep = np.array([pos_map.get(f, -1) for f in node_index])
        ok = keep >= 0
        D = np.sqrt(((xy[keep[ok]][:, None, :] - xy[keep[ok]][None, :, :]) ** 2).sum(-1))
        zc = rbar[ok]
        bands = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300),
                 (300, 400), (400, 600), (600, 900)]
        Ib, mids = [], []
        for lo_d, hi_d in bands:
            A = ((D > lo_d) & (D <= hi_d)).astype(float)
            np.fill_diagonal(A, 0.0)
            deg = A.sum(1)
            if (deg > 0).sum() < 50:
                continue
            Ib.append(moran_i(zc, A / np.maximum(deg[:, None], 1.0)))
            mids.append((lo_d + hi_d) / 2)
        stats["correlogram"] = {"distance_km": mids, "moran_i": [float(v) for v in Ib]}
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(mids, Ib, "o-", color=PALETTE[0])
        ax.axhline(-1 / (n_nodes - 1), color="black", ls="--", lw=1, label="null $-1/(n-1)$")
        ax.set_xlabel("Distance band midpoint (km)"); ax.set_ylabel("Moran's I")
        ax.set_title("Residual correlogram (EPSG:5070 centroids)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(lw=0.3, alpha=0.5)
        fig.tight_layout(); fig.savefig(out_dir / "B3_correlogram.png", dpi=150); plt.close(fig)

    st = df.assign(state=df["county_fips"].str[:2]).groupby("state")["r"].agg(["mean", "count"])
    st["se"] = df.assign(state=df["county_fips"].str[:2]).groupby("state")["r"].std() / np.sqrt(st["count"])
    st = st.sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(range(len(st)), st["mean"], yerr=1.96 * st["se"], fmt="o", ms=3,
                capsize=2, color=PALETTE[0])
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st.index, rotation=90, fontsize=6)
    ax.set_ylabel(r"mean $r$"); ax.set_xlabel("state FIPS")
    ax.set_title("Residual mean by state (95% CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "B4_by_state.png", dpi=150); plt.close(fig)
    stats["state_bias_max_abs"] = float(st["mean"].abs().max())
    return stats, pd.Series(rbar, index=node_index, name="r_mean")[present]


# --------------------------------------------------------------------- C. temporal

def section_temporal(df, out_dir, geom):
    """Returns (stats, monthly mean residual series, per-county lag-1 autocorrelation)."""
    stats = {}
    monthly = df.groupby("target_date")["r"].agg(["mean", "count"]).sort_index()
    x = monthly["mean"].to_numpy()
    T = len(x)
    nlags = min(24, T // 3)
    a = acf(x, nlags)
    pa = pacf_from_acf(a)
    stats["dw"] = durbin_watson(x)
    # The 26-month test split cannot support lag 12 or 24, so also report the largest
    # feasible lag; without it section C would have no formal independence test there.
    for h in sorted({12, 24, nlags}):
        if nlags >= h >= 1:
            q, pv = ljung_box(a, T, h)
            stats[f"ljung_box_{h}"] = {"Q": q, "p": pv, "df": h}
    stats["ljung_box_max_lag"] = int(nlags)
    stats["acf"] = a.tolist()
    stats["pacf"] = pa.tolist()
    stats["n_months"] = int(T)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    band = 2 / np.sqrt(monthly["count"].to_numpy())
    axes[0].plot(monthly.index, x, "-o", ms=3, color=PALETTE[0])
    axes[0].fill_between(monthly.index, -band, band, color="grey", alpha=0.25, lw=0,
                         label=r"$\pm 2/\sqrt{n_t}$")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylabel(r"mean $r$")
    axes[0].set_title(f"Monthly mean residual   (DW = {stats['dw']:.2f})", fontsize=10)
    axes[0].legend(fontsize=8)
    bart = 1.96 / np.sqrt(T)
    for ax, v, name in ((axes[1], a, "ACF"), (axes[2], pa, "PACF")):
        ax.bar(range(1, nlags + 1), v[1:], width=0.6, color=PALETTE[0])
        ax.axhline(0, color="black", lw=1)
        ax.axhspan(-bart, bart, color="grey", alpha=0.25, lw=0)
        ax.set_ylabel(name); ax.set_xlabel("lag (months)")
    axes[1].set_title(f"Bartlett band $\\pm 1.96/\\sqrt{{T}}$, T = {T}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "C1_temporal.png", dpi=150); plt.close(fig)

    piv = df.pivot_table(index="county_fips", columns="target_date", values="r",
                         aggfunc="mean").sort_index(axis=1)
    M = piv.to_numpy()
    M = M - np.nanmean(M, axis=1, keepdims=True)
    a1 = np.nansum(M[:, 1:] * M[:, :-1], axis=1) / np.nansum(M * M, axis=1)
    stats["county_lag1_acf_mean"] = float(np.nanmean(a1))
    stats["county_lag1_acf_p95"] = float(np.nanquantile(a1, 0.95))

    ncol = 3 if geom is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 4))
    axes[0].hist(a1[np.isfinite(a1)], bins=50, color=PALETTE[0], alpha=0.85,
                 edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_xlabel(r"per-county lag-1 autocorrelation of $r$")
    axes[0].set_ylabel("counties")
    axes[0].set_title(f"mean = {np.nanmean(a1):.3f}", fontsize=10)

    by_m = df.assign(cal=df["target_date"].dt.month).groupby("cal")["r"].agg(["mean", "sem"])
    by_h = df.groupby("horizon")["r"].agg(["mean", "sem"])
    axes[1].errorbar(by_m.index, by_m["mean"], yerr=1.96 * by_m["sem"], fmt="o-", ms=4,
                     capsize=2, color=PALETTE[0], label="calendar month")
    axes[1].errorbar(by_h.index, by_h["mean"], yerr=1.96 * by_h["sem"], fmt="s--", ms=4,
                     capsize=2, color=PALETTE[1], label="horizon $h$")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("month of year  /  horizon (months)")
    axes[1].set_ylabel(r"mean $r$ (95% CI)")
    axes[1].set_title("Leftover seasonality and horizon bias", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(lw=0.3, alpha=0.5)
    stats["by_calendar_month"] = by_m["mean"].round(5).to_dict()
    stats["by_horizon"] = by_h["mean"].round(5).to_dict()

    if geom is not None:
        choropleth(axes[2], geom, pd.Series(a1, index=piv.index),
                   "Per-county lag-1 residual autocorrelation", vlim=0.4)
    fig.tight_layout(); fig.savefig(out_dir / "C2_temporal_county.png", dpi=150); plt.close(fig)

    return stats, monthly, pd.Series(a1, index=piv.index, name="lag1_acf")


# ------------------------------------------------- G. space-time autocorrelation

def residual_cube(df: pd.DataFrame, n_nodes: int):
    """[H, M, N] residual field indexed by (horizon, target month, county node).

    Time is indexed by *target* month, not by origin, so that permuting the time axis
    moves all twelve horizons that share a month together -- the same unit the
    evaluation contract requires bootstrap blocks to respect.
    """
    months = np.sort(df["target_date"].unique())
    horizons = np.sort(df["horizon"].unique())
    cube = np.full((len(horizons), len(months), n_nodes), np.nan)
    cube[pd.Index(horizons).get_indexer(df["horizon"]),
         pd.Index(months).get_indexer(df["target_date"]),
         df["node_id"].to_numpy().astype(int)] = df["r"].to_numpy()
    return cube, months, horizons


def _gram(Z: np.ndarray, Ws: list):
    """G[l,h,a,b] = <W_l z_ha, z_hb>/N and E[l,h,a] = ||W_l z_ha||^2/N.

    Tabulating every month pair up front costs one pass and makes each permutation of
    the time axis a reindex of G rather than a refit.
    """
    H, M, N = Z.shape
    G = np.empty((len(Ws), H, M, M))
    E = np.empty((len(Ws), H, M))
    flat = Z.reshape(H * M, N).T                          # [N, H*M]
    for l, W in enumerate(Ws):
        wz = np.asarray(W @ flat).T.reshape(H, M, N)
        for h in range(H):
            G[l, h] = wz[h] @ Z[h].T / N
            E[l, h] = np.einsum("mn,mn->m", wz[h], wz[h]) / N
    return G, E


def _stacf(G, E, valid, max_lag: int):
    """Pfeifer-Deutsch rho(l,k), pooled over horizons, plus the month-pair count."""
    L, H, M, _ = G.shape
    rho = np.full((L, max_lag + 1), np.nan)
    npair = np.zeros(max_lag + 1, dtype=int)
    e = np.array([E[l][valid].mean() for l in range(L)])
    scale = np.sqrt(e * e[0])
    for k in range(max_lag + 1):
        a = np.arange(M - k)
        m = valid[:, a] & valid[:, a + k]
        npair[k] = n = int(m.sum())
        if n:
            rho[:, k] = np.where(m, G[:, :, a, a + k], 0.0).sum((1, 2)) / n / scale
    return rho, npair


def _perm_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Two-sided permutation p, centred on the permutation mean as in morans_permutation."""
    c = null.mean(axis=0)
    hit = (np.abs(null - c) >= np.abs(obs - c)).sum(axis=0)
    return (1 + hit) / (1 + null.shape[0])


def section_spacetime(df, out_dir, max_lag: int, max_order: int, n_perm: int, seed: int):
    """Section G -- space-time autocorrelation of the residual field.

    Sections B and C each collapse an axis before testing the other: B averages a
    county's residuals over all months, C averages a month's over all counties. Neither
    can see structure that is jointly spatial *and* lagged, which is precisely what a
    space-time autoregression absorbs. rho(l, k) correlates a county's residual with the
    mean residual of its order-l neighbour shell k months earlier, over the same
    exclusive Pfeifer-Deutsch operators `model/starima.py` selects on, so the two tables
    read against each other directly: a lag that screen retained and the model has not
    absorbed survives here as a non-null rho.

    Two nulls, because 559,440 rows are nothing like 559,440 independent units.
    Permuting whole target months -- all horizons moving together -- destroys temporal
    dependence while preserving each month's spatial pattern and each slice's marginal;
    permuting county labels destroys the graph alignment while preserving each county's
    series. The first is the operative null for k >= 1, the second for l >= 1.

    Reported twice: on the raw field, and on the field with each month's cross-sectional
    mean removed. The difference separates a common national factor the model missed
    (a fire year it called wrong everywhere at once) from idiosyncratic county-level
    persistence, which have different remedies.
    """
    edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
    n_nodes = int(max(edge.max() + 1, df["node_id"].max() + 1))
    Z, months, horizons = residual_cube(df, n_nodes)

    obs_mask = np.isfinite(Z)
    valid = obs_mask.any(axis=2)
    if not np.array_equal(valid, obs_mask.all(axis=2)):
        raise SystemExit("section G expects every (horizon, target month) slice to be "
                         "complete over counties; this file is ragged")
    Z = np.where(obs_mask, Z - Z[obs_mask].mean(), 0.0)
    Zd = Z - Z.mean(axis=2, keepdims=True)

    Ws = exclusive_orders(edge, n_nodes, max_order, sparse=True)
    rng = np.random.default_rng(seed)
    stats = {"n_months": int(valid.shape[1]), "n_horizons": int(len(horizons)),
             "n_counties": n_nodes, "max_lag": max_lag, "max_order": max_order,
             "n_perm": n_perm,
             "mean_shell_size": [float(w.getnnz(axis=1).mean()) for w in Ws]}
    out = {}

    for key, field in (("raw", Z), ("county_anomaly", Zd)):
        G, E = _gram(field, Ws)
        rho, npair = _stacf(G, E, valid, max_lag)

        null_t = np.stack([_stacf(G[:, :, s][:, :, :, s], E[:, :, s], valid[:, s],
                                  max_lag)[0]
                           for s in (rng.permutation(valid.shape[1])
                                     for _ in range(n_perm))])
        null_c = np.stack([_stacf(*_gram(field[:, :, p], Ws), valid, max_lag)[0]
                           for p in (rng.permutation(n_nodes) for _ in range(n_perm))])

        out[key] = {"rho": rho, "p_month": _perm_p(rho, null_t),
                    "p_county": _perm_p(rho, null_c),
                    "sd_month": null_t.std(axis=0, ddof=1),
                    "sd_county": null_c.std(axis=0, ddof=1)}
        stats[key] = {"rho": rho.tolist(),
                      "p_month_perm": out[key]["p_month"].tolist(),
                      "p_county_perm": out[key]["p_county"].tolist(),
                      "null_sd_month": out[key]["sd_month"].tolist(),
                      "null_sd_county": out[key]["sd_county"].tolist()}
    stats["n_month_pairs"] = npair.tolist()

    rows = [{"field": key, "order": l, "lag": k,
             "rho": v["rho"][l, k], "p_month_perm": v["p_month"][l, k],
             "p_county_perm": v["p_county"][l, k], "null_sd_month": v["sd_month"][l, k],
             "n_month_pairs": int(npair[k])}
            for key, v in out.items()
            for l in range(max_order + 1) for k in range(max_lag + 1)]
    pd.DataFrame(rows).to_csv(out_dir / "stacf.csv", index=False)

    _plot_stacf(out, npair, max_lag, max_order, out_dir)
    return stats


def _plot_stacf(out, npair, max_lag, max_order, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.6 * 2, 3.6))
    lead = np.zeros((max_order + 1, max_lag + 1), dtype=bool)
    lead[0, 0] = True                      # rho(0,0) = 1 by construction; off the scale
    scale = max(np.nanmax(np.abs(np.where(lead, np.nan, v["rho"])))
                for v in out.values())
    for ax, (key, v) in zip(axes, out.items()):
        m = ax.imshow(v["rho"], cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
        for l in range(max_order + 1):
            for k in range(max_lag + 1):
                if l == 0 and k == 0:
                    continue
                p = max(v["p_month"][l, k], v["p_county"][l, k]) if (l and k) else (
                    v["p_month"][l, k] if k else v["p_county"][l, k])
                ax.text(k, l, "*" if p < 0.05 else "", ha="center", va="center",
                        fontsize=9, color="black")
        ax.set_xticks(range(max_lag + 1)); ax.set_yticks(range(max_order + 1))
        ax.set_xlabel("temporal lag $k$ (months)")
        ax.set_ylabel("spatial order $l$")
        ax.set_title(f"$\\rho(l,k)$ -- {key.replace('_', ' ')}", fontsize=10)
        fig.colorbar(m, ax=ax, fraction=0.03)
    fig.suptitle("Space-time autocorrelation of the randomized quantile residual "
                 "(* = permutation $p<0.05$)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "G1_stacf.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    k = np.arange(max_lag + 1)
    for ax, (key, v) in zip(axes, out.items()):
        rho = v["rho"].copy()
        rho[0, 0] = np.nan               # = 1 by construction; would flatten everything else
        for l in range(max_order + 1):
            ax.plot(k, rho[l], "o-", ms=3, color=PALETTE[l % len(PALETTE)],
                    label=f"$l = {l}$")
        ax.fill_between(k, -1.96 * v["sd_month"][0], 1.96 * v["sd_month"][0],
                        color="0.6", alpha=0.25, lw=0, zorder=0)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("temporal lag $k$ (months)"); ax.set_ylabel(r"$\rho(l,k)$")
        ax.set_title(f"{key.replace('_', ' ')} (band: 95% month-permutation null)",
                     fontsize=10)
        ax.grid(lw=0.3, alpha=0.5); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "G2_stacf_profiles.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------- D. mean and variance adequacy

def section_adequacy(df, out_dir, rng, panel: pd.DataFrame | None) -> dict:
    stats = {}
    r = df["r"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, lab in ((axes[0], "e_y", r"fitted $E[y]$"),
                         (axes[1], "p_occ", r"$P(\mathrm{fire})$"),
                         (axes[2], "mu", r"$\mu$")):
        x = df[col].to_numpy()
        # A panel binned on a quantity that is undefined on the zero rows would be a plot of the
        # positives alone wearing the label of the whole split, which is worse than no plot.
        if np.isnan(x).any():
            ax.set_axis_off()
            ax.set_title(f"{lab}: undefined on zero rows", fontsize=9)
            stats[f"binned_r_vs_{col}"] = None
            continue
        xb, m, lo, hi, nk = binned_mean(x, r, 18, rng)
        ax.errorbar(xb, m, yerr=[m - lo, hi - m], fmt="o-", ms=4, capsize=2,
                    color=PALETTE[0])
        ax.axhline(0, color="black", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel(r"mean $r$ (95% CI)")
        ax.grid(lw=0.3, alpha=0.5)
        stats[f"binned_r_vs_{col}"] = {"x": xb.tolist(), "mean": m.tolist(),
                                       "n": nk.tolist()}
    axes[0].set_title("Residual versus fitted, equal-count bins", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "D1_residual_vs_fitted.png", dpi=150)
    plt.close(fig)

    pos = df["y_true"].to_numpy() > 0
    dpos = df.loc[pos]
    mu_p = dpos["mu"].to_numpy()
    y_p = dpos["y_true"].to_numpy()
    var_model = mu_p * (1 - mu_p) / (1 + dpos["phi"].to_numpy())
    idx, _ = qbin(mu_p, 15)
    rows = []
    for k in range(idx.max() + 1):
        m = idx == k
        if m.sum() < 30:
            continue
        rows.append((float(mu_p[m].mean()), float(y_p[m].var(ddof=1)),
                     float(var_model[m].mean()), int(m.sum())))
    A = np.array(rows).T
    stats["phi_model"] = {"mu": A[0].tolist(), "var_empirical": A[1].tolist(),
                          "var_model": A[2].tolist(), "n": A[3].tolist()}
    stats["phi_model_log_ratio_mean"] = float(np.mean(np.log(A[1] / A[2])))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(A[0], A[1], "o-", color=PALETTE[0], label="empirical $\\mathrm{Var}(y\\mid y>0)$")
    axes[0].plot(A[0], A[2], "s--", color=PALETTE[1], label=r"model $\mu(1-\mu)/(1+\phi)$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\mu$ (equal-count bins)"); axes[0].set_ylabel("variance")
    axes[0].set_title("Beta variance model adequacy", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(lw=0.3, alpha=0.5)
    axes[1].plot(A[0], A[1] / A[2], "o-", color=PALETTE[0])
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\mu$"); axes[1].set_ylabel("empirical / model variance")
    axes[1].set_title(f"mean log ratio = {stats['phi_model_log_ratio_mean']:+.3f}",
                      fontsize=10)
    axes[1].grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "D2_variance_model.png", dpi=150); plt.close(fig)

    strata = {"season": df["target_date"].dt.month.map(SEASON),
              "region": df["county_fips"].str[:2].map(REGION).fillna("other"),
              "horizon": df["horizon"]}
    if panel is not None:
        strata["land cover"] = df.merge(
            panel, on=["county_fips", "target_date"], how="left")["lc_dominant"].to_numpy()
    fig, axes = plt.subplots(1, len(strata), figsize=(4.2 * len(strata), 3.8), squeeze=False)
    for ax, (name, key) in zip(axes[0], strata.items()):
        g = df.assign(_k=np.asarray(key)).groupby("_k")["r"].agg(["mean", "sem", "count"])
        g = g[g["count"] > 200]
        ax.errorbar(range(len(g)), g["mean"], yerr=1.96 * g["sem"], fmt="o", ms=4,
                    capsize=2, color=PALETTE[0])
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([str(i) for i in g.index], rotation=45, fontsize=7)
        ax.set_title(f"mean $r$ by {name}", fontsize=10)
        ax.grid(lw=0.3, alpha=0.5)
        stats[f"by_{slug(name)}"] = {str(k): round(float(v), 5)
                                     for k, v in g["mean"].items()}
    fig.tight_layout(); fig.savefig(out_dir / "D3_strata.png", dpi=150); plt.close(fig)
    return stats


# --------------------------------------------------------------- E. discrimination

def section_discrimination(df, out_dir, rng) -> dict:
    p = df["p_occ"].to_numpy()
    o = (df["y_true"].to_numpy() > 0).astype(int)
    stats = {"gate_auc": float(roc_auc_score(o, p)),
             "gate_ap": float(average_precision_score(o, p)),
             "base_rate": float(o.mean())}
    fpr, tpr, _ = roc_curve(o, p)
    prec, rec, _ = precision_recall_curve(o, p)
    step = max(1, len(fpr) // 3000)

    per_h = {}
    for h, g in df.groupby("horizon"):
        oh = (g["y_true"].to_numpy() > 0).astype(int)
        ph = g["p_occ"].to_numpy()
        auc, ap = float(roc_auc_score(oh, ph)), float(average_precision_score(oh, ph))
        mm = g["target_date"].to_numpy()
        um = np.unique(mm)
        a_b, p_b = [], []
        for _ in range(100):                      # month-block resample, not row resample
            pick = np.isin(mm, rng.choice(um, size=len(um), replace=True))
            if oh[pick].sum() < 5:
                continue
            a_b.append(roc_auc_score(oh[pick], ph[pick]))
            p_b.append(average_precision_score(oh[pick], ph[pick]))
        per_h[int(h)] = {"auc": auc, "ap": ap,
                         "auc_lo": float(np.quantile(a_b, 0.025)) if a_b else np.nan,
                         "auc_hi": float(np.quantile(a_b, 0.975)) if a_b else np.nan,
                         "ap_lo": float(np.quantile(p_b, 0.025)) if p_b else np.nan,
                         "ap_hi": float(np.quantile(p_b, 0.975)) if p_b else np.nan}
    stats["per_horizon"] = per_h

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(fpr[::step], tpr[::step], color=PALETTE[0])
    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=1)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"Gate ROC, AUC = {stats['gate_auc']:.4f}", fontsize=10)
    axes[1].plot(rec[::step], prec[::step], color=PALETTE[1])
    axes[1].axhline(stats["base_rate"], color="black", ls="--", lw=1, label="base rate")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title(f"Gate PR, AP = {stats['gate_ap']:.4f}", fontsize=10)
    axes[1].legend(fontsize=8)
    hs = sorted(per_h)
    for key, c, lab in (("auc", PALETTE[0], "AUC"), ("ap", PALETTE[1], "AP")):
        m = np.array([per_h[h][key] for h in hs])
        lo = np.array([per_h[h][f"{key}_lo"] for h in hs])
        hi = np.array([per_h[h][f"{key}_hi"] for h in hs])
        axes[2].errorbar(hs, m, yerr=[m - lo, hi - m], fmt="o-", ms=4, capsize=2,
                         color=c, label=lab)
    axes[2].set_xlabel("horizon $h$ (months)"); axes[2].set_ylabel("score")
    axes[2].set_title("Discrimination by horizon (month-block 95% CI)", fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(lw=0.3, alpha=0.5)
    for ax in axes[:2]:
        ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "E1_discrimination.png", dpi=150); plt.close(fig)
    return stats


# ---------------------------------------------------------------- F. comparison

def section_comparison(frames: dict[str, pd.DataFrame], out_dir: Path, seed: int) -> dict:
    labels = list(frames)
    ref = labels[0]
    stats = {"reference": ref, "models": {}}
    base = frames[ref]

    for lab, df in frames.items():
        if not df[KEYS[:4]].equals(base[KEYS[:4]]):
            raise SystemExit(f"'{lab}' does not share the row set of '{ref}'")
        o = (df["y_true"].to_numpy() > 0).astype(float)
        stats["models"][lab] = {
            "mean_nll": float(df["nll"].mean()),
            # pandas' mean skips NaN, so a frame with undefined mu on its zero rows would report
            # the CRPS of its positives under the label of the whole split.
            "mean_crps": None if df["mu"].isna().any() else float(df["crps"].mean()),
            **brier_decomp(df["p_occ"].to_numpy(), o),
            "gate_auc": float(roc_auc_score(o, df["p_occ"].to_numpy())),
            "gate_ap": float(average_precision_score(o, df["p_occ"].to_numpy())),
        }

    months = base["target_date"]
    for lab, df in frames.items():
        if lab == ref:
            continue
        entry = {}
        crps_ok = not (df["mu"].isna().any() or base["mu"].isna().any())
        for score in ("nll", "crps"):
            if score == "crps" and not crps_ok:
                entry[score] = None
                continue
            d = (df[score].to_numpy() - base[score].to_numpy())
            g = pd.DataFrame({"m": months, "d": d}).groupby("m")["d"].agg(["sum", "count"])
            g = g.sort_index()
            per_month = (g["sum"] / g["count"]).to_numpy()
            entry[score] = {
                "dm": dm_hac(per_month),
                "block1": dict(zip(("mean", "lo", "hi"),
                                   month_block_boot(g["sum"].to_numpy(),
                                                    g["count"].to_numpy(), 1, 2000, seed))),
                "block3": dict(zip(("mean", "lo", "hi"),
                                   month_block_boot(g["sum"].to_numpy(),
                                                    g["count"].to_numpy(), 3, 2000, seed))),
            }
        stats["models"][lab]["vs_reference"] = entry

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (lab, df) in enumerate(frames.items()):
        c = PALETTE[i % len(PALETTE)]
        for ax, score in ((axes[0], "nll"), (axes[1], "crps")):
            g = df.groupby("horizon")[score].agg(["mean", "sem"])
            ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o-", ms=4,
                        capsize=2, color=c, label=lab)
        g = df.groupby("horizon")["r"].agg(["mean", "sem"])
        axes[2].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o-", ms=4,
                         capsize=2, color=c, label=lab)
    axes[0].set_ylabel("mean negative log score"); axes[0].set_title("Log score by horizon", fontsize=10)
    axes[1].set_ylabel("mean CRPS"); axes[1].set_title("CRPS by horizon", fontsize=10)
    axes[2].set_ylabel(r"mean $r$"); axes[2].axhline(0, color="black", lw=1)
    axes[2].set_title("Residual bias by horizon", fontsize=10)
    for ax in axes:
        ax.set_xlabel("horizon $h$ (months)"); ax.legend(fontsize=8); ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "F1_score_by_horizon.png", dpi=150); plt.close(fig)

    others = [l for l in labels if l != ref]
    if others:
        fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(others), 4))
        for j, lab in enumerate(others):
            for k, (blk, off, col) in enumerate((("block1", -0.12, PALETTE[0]),
                                                 ("block3", 0.12, PALETTE[1]))):
                e = stats["models"][lab]["vs_reference"]["nll"][blk]
                ax.errorbar(j + off, e["mean"],
                            yerr=[[e["mean"] - e["lo"]], [e["hi"] - e["mean"]]],
                            fmt="o", ms=5, capsize=3, color=col,
                            label=f"block {blk[-1]}" if j == 0 else None)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(others))); ax.set_xticklabels(others, rotation=20, fontsize=8)
        ax.set_ylabel(f"mean log score minus {ref}")
        ax.set_title("Paired month-block bootstrap, 95%", fontsize=10)
        ax.legend(fontsize=8); ax.grid(lw=0.3, alpha=0.5)
        fig.tight_layout(); fig.savefig(out_dir / "F2_score_difference.png", dpi=150)
        plt.close(fig)
    return stats


# ------------------------------------------------------------------------- driver

def load_panel_lc(split: str):
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path, columns=["county_fips", "date", "lc_dominant"])
    p["county_fips"] = p["county_fips"].astype(str).str.zfill(5)
    return p.rename(columns={"date": "target_date"})


def run_one(pred_dir: Path, split: str, label: str, seed: int, n_perm: int,
            st_max_lag: int, st_max_order: int, st_perm: int) -> dict:
    out_dir = OUT_ROOT / slug(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = add_scores(add_residuals(load_predictions(pred_dir, split), seed))

    # INLA's conditional design is the training POSITIVES plus the evaluation splits, so on its
    # train file mu is undefined wherever y == 0. The residual still exists there (a zero row's
    # randomized PIT is U(0, pi) and never touches mu), so every residual-based section is
    # unaffected; CRPS and the fitted-value panels of D1 are not.
    mu_nan = df["mu"].isna().to_numpy()
    if mu_nan.any() and (df["y_true"].to_numpy()[mu_nan] > 0).any():
        raise SystemExit(f"{pred_dir} has undefined mu on rows with y > 0; that is a broken "
                         f"prediction file, not a design restriction")
    n_mu_nan = int(mu_nan.sum())
    if n_mu_nan:
        print(f"  mu undefined on {n_mu_nan:,} of {len(df):,} rows (all y == 0) -> "
              f"CRPS and the D1 fitted-value panels are skipped")

    crps_txt = "n/a" if n_mu_nan else f"{df['crps'].mean():.6f}"
    print(f"[{label}] {len(df):,} rows  mean nll {df['nll'].mean():.5f}  mean crps {crps_txt}")

    geom = load_geom()
    if geom is None:
        print("  county_geom.parquet absent -> skipping choropleths "
              "(build it with plot_predictions.py)")

    summary = {"label": label, "pred_dir": str(pred_dir), "split": split,
               "n_rows": int(len(df)), "seed": seed,
               "n_mu_undefined": n_mu_nan,
               "mean_nll": float(df["nll"].mean()),
               "mean_crps": None if n_mu_nan else float(df["crps"].mean())}
    summary["distributional"] = section_distributional(df, out_dir, rng)
    sp, rbar = section_spatial(df, out_dir, geom, n_perm)
    summary["spatial"] = sp
    te, monthly, lag1 = section_temporal(df, out_dir, geom)
    summary["temporal"] = te
    summary["spacetime"] = section_spacetime(df, out_dir, st_max_lag, st_max_order,
                                             st_perm, seed)
    summary["adequacy"] = section_adequacy(df, out_dir, rng, load_panel_lc(split))
    summary["discrimination"] = section_discrimination(df, out_dir, rng)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    county = pd.DataFrame({"r_mean": rbar}).join(lag1.rename("lag1_acf"))
    county.index.name = "county_fips"
    county.reset_index().to_parquet(out_dir / "residual_summary_county.parquet", index=False)
    monthly.reset_index().to_parquet(out_dir / "residual_summary_month.parquet", index=False)
    print(f"[{label}] -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=None, help="directory holding predictions_<split>.parquet")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="two or more prediction directories; the first is the reference")
    ap.add_argument("--labels", nargs="+", default=None, help="labels for --compare, in order")
    ap.add_argument("--split", default="test", choices=["train", "test", "validation"])
    ap.add_argument("--label", default=None,
                    help="names the output dir for both modes; with --compare it overrides the "
                         "default compare_<split>, which is keyed on split alone and so collides "
                         "between different comparisons")
    ap.add_argument("--seed", type=int, default=0, help="seed for the randomized PIT")
    ap.add_argument("--n-perm", type=int, default=999, help="Moran permutations")
    ap.add_argument("--st-max-lag", type=int, default=12,
                    help="deepest temporal lag for the section-G space-time correlogram")
    ap.add_argument("--st-max-order", type=int, default=3,
                    help="deepest exclusive spatial order for section G")
    ap.add_argument("--st-perm", type=int, default=999,
                    help="permutations per section-G null (month-shuffle and "
                         "county-shuffle)")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        labels = args.labels or [Path(d).name for d in args.compare]
        if len(labels) != len(args.compare):
            raise SystemExit("--labels must match --compare in length")
        frames = {}
        for d, lab in zip(args.compare, labels):
            frames[lab] = add_scores(add_residuals(load_predictions(Path(d), args.split),
                                                   args.seed))
            print(f"  loaded {lab}: {len(frames[lab]):,} rows")
        # --compare keyed on split alone silently overwrites an earlier comparison on the same
        # split. --label opts a run into its own directory.
        out_dir = OUT_ROOT / (slug(args.label) if args.label else f"compare_{args.split}")
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = section_comparison(frames, out_dir, args.seed)
        (out_dir / "comparison.json").write_text(json.dumps(stats, indent=2, default=float))
        print(json.dumps(stats["models"], indent=2, default=float)[:2000])
        print(f"-> {out_dir}")
        return

    if not args.pred_dir:
        raise SystemExit("pass --pred-dir or --compare")
    run_one(Path(args.pred_dir), args.split, args.label or Path(args.pred_dir).name,
            args.seed, args.n_perm, args.st_max_lag, args.st_max_order, args.st_perm)


if __name__ == "__main__":
    main()
