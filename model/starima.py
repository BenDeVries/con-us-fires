"""Space-time autoregressive lag selection on the county-month panel (EDA).

Answers one question: **how many temporal lags and how many spatial neighbour orders does
the burned-fraction panel actually carry?** The consumers are the GNN's `cfg.lookback`
(24 vs 12 is still open between the two live candidates) and the depth of its GCN stack,
whose hop count is exactly a spatial lag order.

Structure, and why it is not a package call
-------------------------------------------
No hurdle-Beta STARMA implementation exists (R's `starma` is Gaussian Pfeifer-Deutsch,
`splm`/`spreg` are Gaussian spatial econometrics, `betareg` has no spatial term). None is
needed, because the spatial-econometric machinery exists to handle the *contemporaneous*
term W·y_t, which is endogenous. A purely **auto**regressive space-time specification has
no such term: every lag sits at t-p, strictly in the past, so the lag regressors are
predetermined and are ordinary covariates. That makes the exact hurdle estimable by plain
ML, which is what this module does:

  gate      1{y>0} ~ Bernoulli,  logit P(fire) = logit(1-pi_hat) + LAGS + X
  severity  y|y>0  ~ Beta(mu,phi), logit(mu)   = logit(mu_hat)   + LAGS + X

matching the hurdle every class in the bake-off emits. What is given up is the MA half of
"ARIMA" -- spatially lagged *errors* -- which genuinely would need special machinery and
which neither the GNN encoder nor INLA's arm G uses anyway.

The `logit` link on the severity mean is the link `--link logit` settled for every other
class, so these coefficients are on the same scale as the GNN's and INLA's mu head. Over
the 46,426 train positives `log(y)` and `logit(y)` correlate at 0.99998 (only 116 exceed
0.1, one exceeds 0.5), so the choice is free on fit and taken on comparability -- and
logit cannot back-transform above 1, which for a fraction of a county matters.

The climatology offset
----------------------
Both linear predictors carry a *fixed* offset from `output/climatology/climatology_table
.parquet` -- the shipped train-only, empirical-Bayes-shrunk county x calendar-month ZIB
over all 37,296 cells. It costs zero degrees of freedom, so every retained term measures
lift over the no-skill floor rather than re-discovering it. Without it the temporal lags
would mostly proxy for county identity and season: 3,107 county dummies would be the
alternative, and they are the unshrunk version of the same object with a separation
problem in the 63% of cells that never burn in train.

Covariates
----------
The 39 dynamic predictors are PCA-rotated (`xgb.reduction.fit_global`, sign-anchored on
`erc_max`) and the 12 near-constant ones of `data.STATIC_COLS` bypass the rotation, which
is `inla_glm_comparison.qmd`'s blocks_dyn split and the GNN's `--static-bypass` arm --
so this k is comparable with those, and *not* with xgb's k over all 51. Unlike
`xgb.reduction`, `data.TAIL_TRANSFORMS` **is** applied to the bypassed block: trees are
invariant to monotone per-column maps but a linear predictor is not, and z-scored
`pop_density` peaks at 39 sigma. `lc_dominant` enters as a dummy block scored jointly.

Selection
---------
Backward elimination on BIC, all terms (lags and covariates alike) eligible, intercept and
offset always kept. Each step screens by Wald statistic -- one fit per step rather than one
per candidate, exact for Gaussian and accurate to well past the decision margin at these n
-- then refits exactly to confirm the accepted drop. A term survives when its Wald
statistic exceeds df*log(n), i.e. |z| > 3.62 in the gate (n = 500,388) and |z| > 3.26 in
the severity fit (n ~ 40,400). That is permissive: the informative output is the *pattern*
of what falls out, not the size of the surviving model.

Train split only. Test and validation parquets are never opened -- the leading
`--max-temporal` months of train are spent constructing lags, and every lag of an
estimation row is itself a train month.

    conda run -n pytorch python -m model.starima --max-temporal 24 --max-spatial 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, expit, gammaln, logit, polygamma

from .data import STATIC_COLS, TAIL_TRANSFORMS, CAT_COL
from .spatial import exclusive_orders
from .xgb import reduction as red

DATA_DIR = Path("output/data")
CLIM_DIR = Path("output/climatology")
OUT_DIR = Path("output/starima")

EPS_MU = 1e-12


# ── panel, weights, covariates ───────────────────────────────────────────────

def load_panel(max_spatial: int) -> dict:
    """Train panel as [T, N] matrices plus the covariate design and climatology offsets."""
    meta = json.loads((DATA_DIR / "feature_metadata.json").read_text())
    predictors = sorted(k for k, v in meta.items() if v.get("role") == "predictor")
    numeric = [c for c in predictors if c != CAT_COL]
    dyn = [c for c in numeric if c not in STATIC_COLS]
    static = [c for c in numeric if c in STATIC_COLS]

    cols = ["county_fips", "date", "month", "node_id", "burned_fraction", CAT_COL] + numeric
    df = pd.read_parquet(DATA_DIR / "train.parquet", columns=cols)
    df = df.sort_values(["date", "node_id"]).reset_index(drop=True)

    dates = np.sort(df["date"].unique())
    T, N = len(dates), df["node_id"].nunique()
    if len(df) != T * N:
        raise ValueError(f"train panel is not rectangular: {len(df)} != {T}*{N}")

    y = df["burned_fraction"].to_numpy(np.float64).reshape(T, N)
    occ = (y > 0).astype(np.float64)

    # Tail-correct the bypassed statics on the raw scale, then re-standardise on train.
    scaler = {d["feature"]: (d["mean"], d["std"])
              for d in json.loads((DATA_DIR / "feature_scaler.json").read_text())}
    X_static = df[static].to_numpy(np.float64)
    for j, c in enumerate(static):
        kind = TAIL_TRANSFORMS.get(c)
        if kind is None:
            continue
        m, s = scaler[c]
        raw = np.clip(X_static[:, j] * s + m, 0.0, None)
        v = np.log1p(raw) if kind == "log1p" else np.sqrt(raw)
        X_static[:, j] = (v - v.mean()) / (v.std() or 1.0)

    # Rotation over the 39 dynamic predictors only, fitted on the distinct train rows.
    X_dyn = df[dyn].to_numpy(np.float64)
    rot = red.fit_global(X_dyn, dyn, k=len(dyn))
    pcs = rot.transform(X_dyn)

    lc = df[CAT_COL].to_numpy()
    classes = np.unique(lc)
    lc_dummies = (lc[:, None] == classes[None, 1:]).astype(np.float64)   # first is reference

    cov = np.concatenate([pcs, X_static, lc_dummies], axis=1)
    cov_names = list(rot.out_names) + static + [f"lc_dominant={int(c)}" for c in classes[1:]]
    cov_terms = ([(nm, 1) for nm in rot.out_names] + [(nm, 1) for nm in static]
                 + [("lc_dominant", lc_dummies.shape[1])])

    # Fixed offsets from the shipped climatology: pi is P(y=0), so the gate needs 1-pi.
    tab = pd.read_parquet(CLIM_DIR / "climatology_table.parquet")
    key = df[["county_fips", "month"]].merge(tab, on=["county_fips", "month"], how="left")
    if key[["pi", "mu"]].isna().any().any():
        raise ValueError("climatology table does not cover every (county, calendar month)")
    off_gate = logit(np.clip(1.0 - key["pi"].to_numpy(np.float64), 1e-9, 1 - 1e-9))
    off_sev = logit(np.clip(key["mu"].to_numpy(np.float64), 1e-9, 1 - 1e-9))

    edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
    W = exclusive_orders(edge, N, max_spatial)

    return dict(y=y, occ=occ, dates=dates, T=T, N=N, cov=cov, cov_names=cov_names,
                cov_terms=cov_terms, off_gate=off_gate, off_sev=off_sev, W=W,
                explained=rot.eigen["global"], n_dyn=len(dyn), n_static=len(static),
                static_names=static)


def build_design(P: dict, max_temporal: int, max_spatial: int,
                 channels: tuple[str, ...]) -> tuple[np.ndarray, list[tuple[str, int]], np.ndarray]:
    """Assemble [intercept | lags | covariates] over months `max_temporal`..T-1."""
    T, N = P["T"], P["N"]
    t0 = max_temporal
    n_rows = (T - t0) * N

    # Spatially lag each channel once per order, then read off temporal lags by slicing.
    lagged = {}
    for ch in channels:
        src = P["y"] if ch == "y" else P["occ"]
        for l in range(max_spatial + 1):
            lagged[(ch, l)] = src if l == 0 else (P["W"][l] @ src.T).T

    n_lag = len(channels) * (max_spatial + 1) * max_temporal
    X = np.empty((n_rows, 1 + n_lag + P["cov"].shape[1]), dtype=np.float64)
    X[:, 0] = 1.0
    terms: list[tuple[str, int]] = [("(intercept)", 1)]

    j = 1
    for ch in channels:
        for l in range(max_spatial + 1):
            M = lagged[(ch, l)]
            for p in range(1, max_temporal + 1):
                X[:, j] = M[t0 - p:T - p, :].reshape(-1)
                terms.append((f"{ch}_s{l}_t{p}", 1))
                j += 1

    X[:, j:] = P["cov"].reshape(T, N, -1)[t0:].reshape(n_rows, -1)
    terms += P["cov_terms"]

    keep = np.zeros(T * N, dtype=bool)
    keep.reshape(T, N)[t0:] = True
    return X, terms, keep


# ── estimators ───────────────────────────────────────────────────────────────

def _wgram(X: np.ndarray, w: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    G = np.zeros((X.shape[1], X.shape[1]))
    for s in range(0, X.shape[0], chunk):
        Xc = X[s:s + chunk]
        G += Xc.T @ (Xc * w[s:s + chunk, None])
    return G


def _solve(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    ridge = 1e-10 * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + ridge * np.eye(G.shape[0]), b)


def fit_logistic(X, y, offset, beta0=None, tol=1e-9, max_iter=60) -> dict:
    """IRLS for Bernoulli/logit with a fixed offset. Returns beta, loglik and cov(beta)."""
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    ll_prev = -np.inf
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-10)
        G = _wgram(X, w)
        b = X.T @ (y - mu) + G @ beta
        beta = _solve(G, b)
        eta = offset + X @ beta
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        if abs(ll - ll_prev) < tol * (1.0 + abs(ll)):
            break
        ll_prev = ll
    mu = expit(offset + X @ beta)
    G = _wgram(X, np.maximum(mu * (1.0 - mu), 1e-10))
    return dict(beta=beta, loglik=ll, cov=np.linalg.pinv(G), n_par=X.shape[1])


def _beta_loglik(y_log, y_log1m, mu, phi) -> float:
    a, b = mu * phi, (1.0 - mu) * phi
    return float(np.sum(gammaln(phi) - gammaln(a) - gammaln(b)
                        + (a - 1.0) * y_log + (b - 1.0) * y_log1m))


def fit_beta(X, y, offset, beta0=None, phi0=None, tol=1e-9, max_iter=200) -> dict:
    """Fisher scoring for Beta regression, logit link on mu, constant precision phi.

    Ferrari-Cribari-Neto parametrisation y ~ Beta(mu*phi, (1-mu)*phi), joint information
    over (beta, phi) so the Wald statistics account for the mu-phi correlation.
    """
    y_log, y_log1m, y_star = np.log(y), np.log1p(-y), logit(y)
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    phi = 50.0 if phi0 is None else float(phi0)
    ll_prev = -np.inf

    for _ in range(max_iter):
        mu = np.clip(expit(offset + X @ beta), EPS_MU, 1.0 - EPS_MU)
        a, b = mu * phi, (1.0 - mu) * phi
        mu_star = digamma(a) - digamma(b)
        d = mu * (1.0 - mu)                                    # dmu/deta
        psi_a, psi_b = polygamma(1, a), polygamma(1, b)

        s_beta = X.T @ (phi * (y_star - mu_star) * d)
        s_phi = float(np.sum(mu * (y_star - mu_star) + y_log1m
                             - digamma(b) + digamma(phi)))
        w_bb = phi ** 2 * (psi_a + psi_b) * d ** 2
        c_bp = phi * (psi_a * mu - psi_b * (1.0 - mu)) * d
        i_pp = float(np.sum(mu ** 2 * psi_a + (1.0 - mu) ** 2 * psi_b - polygamma(1, phi)))

        p = X.shape[1]
        I = np.empty((p + 1, p + 1))
        I[:p, :p] = _wgram(X, w_bb)
        I[:p, p] = I[p, :p] = X.T @ c_bp
        I[p, p] = i_pp
        step = _solve(I, np.concatenate([s_beta, [s_phi]]))

        ll = -np.inf
        for _ in range(30):                                    # step-halving on phi > 0
            cand_b, cand_p = beta + step[:p], phi + step[p]
            if cand_p > 1e-6:
                cm = np.clip(expit(offset + X @ cand_b), EPS_MU, 1.0 - EPS_MU)
                ll = _beta_loglik(y_log, y_log1m, cm, cand_p)
                if np.isfinite(ll) and ll >= ll_prev:
                    break
            step = step * 0.5
        if not np.isfinite(ll):
            break
        beta, phi = beta + step[:p], phi + step[p]
        if abs(ll - ll_prev) < tol * (1.0 + abs(ll)):
            ll_prev = ll
            break
        ll_prev = ll

    mu = np.clip(expit(offset + X @ beta), EPS_MU, 1.0 - EPS_MU)
    a, b = mu * phi, (1.0 - mu) * phi
    d = mu * (1.0 - mu)
    psi_a, psi_b = polygamma(1, a), polygamma(1, b)
    p = X.shape[1]
    I = np.empty((p + 1, p + 1))
    I[:p, :p] = _wgram(X, phi ** 2 * (psi_a + psi_b) * d ** 2)
    I[:p, p] = I[p, :p] = X.T @ (phi * (psi_a * mu - psi_b * (1.0 - mu)) * d)
    I[p, p] = float(np.sum(mu ** 2 * psi_a + (1.0 - mu) ** 2 * psi_b - polygamma(1, phi)))
    V = np.linalg.pinv(I)
    return dict(beta=beta, phi=phi, loglik=ll_prev, cov=V[:p, :p], n_par=p + 1)


# ── backward elimination ─────────────────────────────────────────────────────

def _wald(fit: dict, cols: list[int]) -> float:
    """Joint Wald statistic for dropping a term's columns (df = len(cols))."""
    b = fit["beta"][cols]
    V = fit["cov"][np.ix_(cols, cols)]
    return float(b @ np.linalg.solve(V + 1e-14 * np.eye(len(cols)), b))


def backward_bic(X, y, offset, terms, fitter, n_obs, path_csv: Path, label: str,
                 confirm_top: int = 5) -> dict:
    """Drop the term with the smallest Wald statistic while it lowers BIC.

    One fit per step rather than one per candidate: the Wald statistic approximates
    2*(loglik drop), so dBIC ~ df*log(n) - W. Measured against exact refits that
    approximation is worth 0.04 at worst for a single-df term in the gate and 0.66 in the
    severity fit -- but the stop decision itself lands within ~1 of the threshold, which is
    not a margin an approximation can carry. So the *stop* is never taken on the Wald: when
    the screen signals it, the `confirm_top` best-ranked candidates are each refit exactly
    and elimination stops only if all of them really do raise BIC.
    """
    active = list(range(len(terms)))
    col_of = {}
    j = 0
    for i, (_, df) in enumerate(terms):
        col_of[i] = list(range(j, j + df))
        j += df

    cols = [c for i in active for c in col_of[i]]
    t0 = time.time()
    fit = fitter(X[:, cols], y, offset)
    bic = -2 * fit["loglik"] + fit["n_par"] * np.log(n_obs)
    print(f"[{label}] full model: {len(active)} terms, {fit['n_par']} par, "
          f"loglik {fit['loglik']:.2f}, BIC {bic:.2f}  ({time.time() - t0:.1f}s)")

    rows = [dict(step=0, dropped="", df=0, wald=None, loglik=fit["loglik"], bic=bic,
                 n_terms=len(active), n_par=fit["n_par"], confirmed=False)]
    thresh = np.log(n_obs)
    confirmations: list[dict] = []

    def refit_without(i, cur_fit, cur_cols):
        trial = [k for k in active if k != i]
        tcols = [c for k in trial for c in col_of[k]]
        warm = cur_fit["beta"][[cur_cols.index(c) for c in tcols]]
        f = fitter(X[:, tcols], y, offset, beta0=warm)
        return trial, tcols, f, -2 * f["loglik"] + f["n_par"] * np.log(n_obs)

    for step in range(1, len(terms)):
        pos = {i: [cols.index(c) for c in col_of[i]] for i in active}
        ranked = sorted(((i, _wald(fit, pos[i])) for i in active
                         if terms[i][0] != "(intercept)"),
                        key=lambda t: t[1] - terms[t[0]][1] * thresh)
        i_drop, w = ranked[0]
        confirmed = w >= terms[i_drop][1] * thresh

        if confirmed:
            # Wald says stop. Verify the top candidates exactly before believing it.
            best = None
            for i_c, w_c in ranked[:confirm_top]:
                trial, tcols, f_c, bic_c = refit_without(i_c, fit, cols)
                confirmations.append(dict(step=step, term=terms[i_c][0], wald=w_c,
                                          delta_bic=float(bic_c - bic)))
                if bic_c < bic and (best is None or bic_c < best[3]):
                    best = (i_c, trial, tcols, bic_c, f_c, w_c)
            if best is None:
                print(f"[{label}] stop at step {step}: {terms[i_drop][0]} has Wald "
                      f"{w:.2f} >= {terms[i_drop][1] * thresh:.2f}; exact refit of the "
                      f"top {min(confirm_top, len(ranked))} candidates confirms "
                      f"(best dBIC {min(c['delta_bic'] for c in confirmations[-confirm_top:]):+.2f})")
                break
            i_drop, trial, tcols, bic, fit, w = best
            print(f"[{label}] step {step:3d}  Wald said stop but exact refit accepts "
                  f"{terms[i_drop][0]} (dBIC {bic - rows[-1]['bic']:+.2f})")
        else:
            trial, tcols, fit, bic = refit_without(i_drop, fit, cols)

        active, cols = trial, tcols
        rows.append(dict(step=step, dropped=terms[i_drop][0], df=terms[i_drop][1], wald=w,
                         loglik=fit["loglik"], bic=bic, n_terms=len(active),
                         n_par=fit["n_par"], confirmed=confirmed))
        if step % 10 == 0 or step == 1:
            print(f"[{label}] step {step:3d}  drop {terms[i_drop][0]:<16s} "
                  f"Wald {w:7.2f}  BIC {bic:.2f}  ({len(active)} terms)")
        pd.DataFrame(rows).to_csv(path_csv, index=False)

    pd.DataFrame(rows).to_csv(path_csv, index=False)
    se = np.sqrt(np.clip(np.diag(fit["cov"]), 0, None))
    retained = []
    for i in active:
        c = [cols.index(x) for x in col_of[i]]
        retained.append(dict(term=terms[i][0], df=terms[i][1],
                             coef=[float(v) for v in fit["beta"][c]],
                             z=[float(fit["beta"][k] / se[k]) if se[k] > 0 else 0.0
                                for k in c],
                             wald=_wald(fit, c)))
    return dict(fit=fit, active=active, retained=retained, bic=bic, path=rows,
                n_obs=int(n_obs), threshold=float(thresh), confirmations=confirmations)


# ── figures ──────────────────────────────────────────────────────────────────

def make_figures(res: dict, terms, channels, max_temporal, max_spatial, out: Path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = {r["term"]: r for r in res["retained"]}
    fig, axes = plt.subplots(len(channels), 1, figsize=(1 + 0.42 * max_temporal,
                                                        1.6 + 1.5 * len(channels)),
                             squeeze=False)
    for ax, ch in zip(axes[:, 0], channels):
        M = np.full((max_spatial + 1, max_temporal), np.nan)
        for l in range(max_spatial + 1):
            for p in range(1, max_temporal + 1):
                r = kept.get(f"{ch}_s{l}_t{p}")
                if r is not None:
                    M[l, p - 1] = np.sign(r["coef"][0]) * np.sqrt(r["wald"])
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(max_spatial + 1))
        ax.set_yticklabels([f"s{l}" for l in range(max_spatial + 1)])
        ax.set_xticks(range(0, max_temporal, max(1, max_temporal // 12)))
        ax.set_xticklabels(range(1, max_temporal + 1, max(1, max_temporal // 12)))
        ax.set_xlabel("temporal lag (months)")
        ax.set_title(f"{label}: retained {ch} lags (signed sqrt Wald; white = dropped)",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out / f"lag_survival_{label}.png", dpi=130)
    plt.close(fig)

    path = pd.DataFrame(res["path"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path["n_terms"], path["bic"], marker=".", lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("terms remaining")
    ax.set_ylabel("BIC")
    ax.set_title(f"{label}: backward elimination path")
    fig.tight_layout()
    fig.savefig(out / f"bic_path_{label}.png", dpi=130)
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────

def run(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    channels = tuple(args.channels)
    print(f"loading train panel, spatial orders 0..{args.max_spatial}")
    P = load_panel(args.max_spatial)
    for l in range(1, args.max_spatial + 1):
        deg = (P["W"][l] > 0).sum(axis=1)
        print(f"  order {l}: mean {deg.mean():.1f} neighbours, "
              f"{int((deg == 0).sum())} counties with none")

    X, terms, keep = build_design(P, args.max_temporal, args.max_spatial, channels)
    y_all = P["y"].reshape(-1)[keep]
    print(f"design {X.shape[0]:,} x {X.shape[1]} ({len(terms)} terms, "
          f"{X.nbytes / 1e9:.2f} GB)")

    summary = {"config": vars(args) | {"n_rows": int(X.shape[0]), "n_terms": len(terms)}}

    if "gate" in args.runs:
        res = backward_bic(X, (y_all > 0).astype(np.float64), P["off_gate"][keep], terms,
                           fit_logistic, X.shape[0], OUT_DIR / "path_gate.csv", "gate",
                           confirm_top=args.confirm_top)
        (OUT_DIR / "selection_gate.json").write_text(json.dumps(
            {k: v for k, v in res.items() if k != "fit"} | {"config": summary["config"]},
            indent=2, default=float))
        make_figures(res, terms, channels, args.max_temporal, args.max_spatial,
                     fig_dir, "gate")
        summary["gate"] = dict(n_retained=len(res["retained"]), bic=res["bic"])

    if "severity" in args.runs:
        pos = y_all > 0
        res = backward_bic(X[pos], y_all[pos], P["off_sev"][keep][pos], terms,
                           fit_beta, int(pos.sum()), OUT_DIR / "path_severity.csv",
                           "severity", confirm_top=args.confirm_top)
        (OUT_DIR / "selection_severity.json").write_text(json.dumps(
            {k: v for k, v in res.items() if k != "fit"}
            | {"config": summary["config"], "phi": float(res["fit"]["phi"])},
            indent=2, default=float))
        make_figures(res, terms, channels, args.max_temporal, args.max_spatial,
                     fig_dir, "severity")
        summary["severity"] = dict(n_retained=len(res["retained"]), bic=res["bic"],
                                   phi=float(res["fit"]["phi"]), n_pos=int(pos.sum()))

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-temporal", type=int, default=24)
    ap.add_argument("--max-spatial", type=int, default=3)
    ap.add_argument("--channels", nargs="+", default=["y", "occ"], choices=["y", "occ"])
    ap.add_argument("--runs", nargs="+", default=["gate", "severity"],
                    choices=["gate", "severity"])
    ap.add_argument("--confirm-top", type=int, default=5,
                    help="exact refits used to verify the stopping decision")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
