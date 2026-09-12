"""County x calendar-month climatological hurdle-Beta reference forecaster.

The null model the GNN, xgb and INLA hurdles all have to beat: no covariates, no
history, no graph -- just "what has this county done in this calendar month before?"
Estimated on the train split alone and applied unchanged to every forecast origin,
so it is a genuine out-of-sample reference rather than a fitted model.

Each (county, calendar month) cell gets a full zero-inflated Beta, because a point
estimate carries no likelihood:

    pi[c,m]  zero-gate probability, empirical zero rate shrunk toward the pooled
             train rate under an empirical-Bayes Beta prior fit by ML over all cells
    mu[c,m]  Beta mean, the *median* of the positive burned fractions in the cell
    phi[c,m] Beta precision, method of moments -- mu(1-mu)/v - 1 -- with v the sample
             variance of the pool's positives

Shrinkage is not optional. 62.8% of the 37,296 cells contain no positive train month,
so the raw pi-hat is exactly 1 there and the NLL diverges the first time such a county
burns in the test split. mu and phi need the same treatment for a different reason:
they are undefined without positives. Both back off through nested pools --
cell -> county -> calendar month -> global -- taking the first pool with enough
positives (>=1 for mu, >=3 for phi, which needs a variance). mu and phi resolve
independently, and phi is always computed against the mu actually used.

Scored on the identical (origin, horizon, county) row set as the other three models
(559,440 test rows), through the same `zib_nll`, and written in the same predictions
schema -- so `model.diagnostics --pred-dir output/climatology` runs on it unchanged.
Being origin-independent, the climatology's NLL over that set is a multiplicity-
weighted mean over distinct county-months; the unweighted figure is reported too.

    conda run -n pytorch python -m model.climatology
    conda run -n pytorch python -m model.climatology --mu-stat mean   # sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import DATA_DIR
from .data import GATE_COL, TARGET_COL, rebuild_origins
from .zib import compute_metrics, per_horizon_nll, zib_nll

OUT_DIR = Path("output/climatology")
MIN_POS_MU = 1        # positives a pool needs to supply a location
MIN_POS_PHI = 3       # ... and to supply a variance
PHI_FLOOR = 1e-2
POOLS = ("cell", "county", "cal_month", "global")


def _beta_binomial_mle(n_zero: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """ML fit of a Beta(a, b) prior on the per-cell zero rate, over all cells jointly.

    Empirical Bayes rather than a hand-set pseudo-count: the shrinkage strength a+b is
    read off how much the 37,296 cells actually disagree, once binomial sampling noise
    at n~15 is accounted for. Optimised in log-space so both parameters stay positive.
    """
    def nll(theta):
        a, b = np.exp(theta)
        return -float((gammaln(a + b) - gammaln(a) - gammaln(b)
                       + gammaln(a + n_zero) + gammaln(b + n - n_zero)
                       - gammaln(a + b + n)).sum())

    p0 = n_zero.sum() / n.sum()
    res = minimize(nll, np.log([2 * p0, 2 * (1 - p0)]), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 4000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def _resolve(cells: pd.MultiIndex, levels: list[tuple[str, pd.Series, pd.Series]],
             min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """First pool with >= min_count positives wins, cell -> ... -> global.

    `levels` is ordered coarsening: (name, value indexed like the pool, count likewise).
    Returns the resolved values and the index into POOLS that supplied each one.
    """
    out = np.full(len(cells), np.nan)
    src = np.full(len(cells), -1, dtype=np.int8)
    for i, (name, value, count) in enumerate(levels):
        if name == "cell":
            v, c = value.reindex(cells).to_numpy(), count.reindex(cells).to_numpy()
        elif name == "county":
            key = cells.get_level_values("county_fips")
            v, c = value.reindex(key).to_numpy(), count.reindex(key).to_numpy()
        elif name == "cal_month":
            key = cells.get_level_values("month")
            v, c = value.reindex(key).to_numpy(), count.reindex(key).to_numpy()
        else:
            v = np.full(len(cells), float(value))
            c = np.full(len(cells), float(count))
        take = np.isnan(out) & (np.nan_to_num(c) >= min_count) & ~np.isnan(v)
        out[take], src[take] = v[take], POOLS.index(name)
    return out, src


def _estimate(tr: pd.DataFrame, cells: pd.MultiIndex,
              mu_stat: str = "median") -> tuple[pd.DataFrame, dict]:
    """Estimate the per-cell ZIB from whatever train rows `tr` holds.

    Split out from `fit` so the leave-one-year-out folds can reuse it; `cells` is
    passed in rather than derived so every fold's table shares one index.
    """
    tr = tr.copy()
    tr["is_pos"] = tr[TARGET_COL] > 0
    pos = tr[tr["is_pos"]]

    # --- gate: empirical zero rate, shrunk under an EB Beta prior -------------
    grp = tr.groupby(["county_fips", "month"])["is_pos"]
    n = grp.size().reindex(cells).to_numpy().astype(float)
    n_zero = (grp.size() - grp.sum()).reindex(cells).to_numpy().astype(float)
    a, b = _beta_binomial_mle(n_zero, n)
    pi = (n_zero + a) / (n + a + b)

    # --- severity: location and spread, each with its own back-off ladder -----
    stat = (lambda s: s.median()) if mu_stat == "median" else (lambda s: s.mean())
    g_cell, g_cty, g_mon = (pos.groupby(k)[TARGET_COL]
                            for k in (["county_fips", "month"], "county_fips", "month"))
    mu, mu_src = _resolve(cells, [
        ("cell", stat(g_cell), g_cell.size()),
        ("county", stat(g_cty), g_cty.size()),
        ("cal_month", stat(g_mon), g_mon.size()),
        ("global", stat(pos[TARGET_COL]), len(pos)),
    ], MIN_POS_MU)

    # A pool of identical positives has zero variance and no usable precision; NaN it
    # so the ladder walks past rather than dividing by zero.
    g_cell, g_cty, g_mon = (pos.groupby(k)[TARGET_COL]
                            for k in (["county_fips", "month"], "county_fips", "month"))
    nz = lambda s: s.where(s > 0)
    var, var_src = _resolve(cells, [
        ("cell", nz(g_cell.var(ddof=1)), g_cell.size()),
        ("county", nz(g_cty.var(ddof=1)), g_cty.size()),
        ("cal_month", nz(g_mon.var(ddof=1)), g_mon.size()),
        ("global", pos[TARGET_COL].var(ddof=1), len(pos)),
    ], MIN_POS_PHI)

    phi_raw = mu * (1 - mu) / var - 1
    phi = np.maximum(phi_raw, PHI_FLOOR)

    table = pd.DataFrame({"pi": pi, "mu": mu, "phi": phi}, index=cells).reset_index()
    info = {
        "mu_stat": mu_stat,
        "n_cells": int(len(cells)),
        "pooled_train_zero_rate": float(n_zero.sum() / n.sum()),
        "eb_beta_prior": {"a": a, "b": b, "prior_mean": a / (a + b),
                          "prior_strength": a + b},
        "cells_with_no_train_positive": int((n_zero == n).sum()),
        "pi_range": [float(pi.min()), float(pi.max())],
        "mu_source_counts": {POOLS[i]: int((mu_src == i).sum()) for i in range(len(POOLS))},
        "var_source_counts": {POOLS[i]: int((var_src == i).sum()) for i in range(len(POOLS))},
        "phi_floor_bound": int((phi_raw < PHI_FLOOR).sum()),
        "phi_quantiles": {q: float(np.quantile(phi, float(q)))
                          for q in ("0.05", "0.5", "0.95")},
    }
    return table, info


def _train_cells(tr: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [np.sort(tr["county_fips"].unique()), np.sort(tr["month"].unique())],
        names=["county_fips", "month"])


def fit(mu_stat: str = "median") -> tuple[pd.DataFrame, dict]:
    """Estimate the per-(county, calendar month) ZIB from the train split only."""
    tr = pd.read_parquet(DATA_DIR / "train.parquet",
                         columns=["county_fips", "month", TARGET_COL, GATE_COL])
    return _estimate(tr, _train_cells(tr), mu_stat)


def fit_oof(mu_stat: str = "median") -> tuple[pd.DataFrame, dict]:
    """Leave-one-year-out folds of the same estimator, for use as a *feature*.

    Handing the fitted table to a learner as a covariate is target encoding: the cell
    estimate for (county, calendar month) contains the very train label the learner is
    being fit against, so the feature looks better in train than it can ever be at test
    (gate AP 0.58 vs 0.47) and the learner over-trusts it. Refitting per held-out year
    removes that: no train row's feature sees its own target.

    A year holds at most one observation per cell, so leave-one-year-out *is* exact
    leave-one-out here -- the cell loses precisely the row being predicted, and nothing
    else. Test and validation rows keep the full-train table, which is already honest.
    """
    tr = pd.read_parquet(DATA_DIR / "train.parquet",
                         columns=["county_fips", "year", "month", TARGET_COL, GATE_COL])
    cells = _train_cells(tr)
    years = np.sort(tr["year"].unique())
    frames, info = [], {"mu_stat": mu_stat, "n_folds": int(len(years)),
                        "folds": [int(y) for y in years], "per_fold": {}}
    for y in years:
        t, i = _estimate(tr[tr["year"] != y], cells, mu_stat)
        t.insert(0, "fold_year", int(y))
        frames.append(t)
        info["per_fold"][int(y)] = {
            "n_rows_used": int((tr["year"] != y).sum()),
            "cells_with_no_positive": i["cells_with_no_train_positive"],
            "eb_prior_strength": i["eb_beta_prior"]["prior_strength"],
        }
    return pd.concat(frames, ignore_index=True), info


def build_rows(split: str, lookback: int, horizon: int) -> tuple[pd.DataFrame, int, int]:
    """The (origin, horizon, county) row set, ordered so it reshapes to [B, H, N]."""
    frames, ranges = [], {}
    for s in ("train", "test", "validation"):
        d = pd.read_parquet(DATA_DIR / f"{s}.parquet",
                            columns=["county_fips", "date", "node_id", TARGET_COL])
        # Normalise to datetime64[ns] up front. rebuild_origins looks the range endpoints
        # up in a dict keyed by these values, and a Timestamp round-tripped through
        # np.datetime64 lands at microsecond resolution and misses every key -- the same
        # KeyError that makes build_panel unusable outside fire-nn.
        d["date"] = d["date"].to_numpy().astype("datetime64[ns]")
        ranges[s] = (d["date"].min().to_numpy(), d["date"].max().to_numpy())
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)
    dates = np.unique(full["date"].to_numpy())
    origins = rebuild_origins(dates, ranges, lookback, horizon)[split]
    if not origins:
        raise SystemExit(f"split '{split}' has no forecast windows")

    nodes = (full.drop_duplicates("node_id").sort_values("node_id")
             [["node_id", "county_fips"]].reset_index(drop=True))
    N, B, H = len(nodes), len(origins), horizon

    y = np.zeros((len(dates), N), dtype=np.float32)
    t = np.searchsorted(dates, full["date"].to_numpy())
    y[t, full["node_id"].to_numpy()] = full[TARGET_COL].to_numpy()

    o = np.repeat(origins, H * N)
    h = np.tile(np.repeat(np.arange(1, H + 1), N), B)
    tgt = o + h
    rows = pd.DataFrame({
        "origin_date": dates[o],
        "horizon": h,
        "target_date": dates[tgt],
        "county_fips": np.tile(nodes["county_fips"].to_numpy(), B * H),
        "node_id": np.tile(nodes["node_id"].to_numpy(), B * H),
        "y_true": y[tgt, np.tile(nodes["node_id"].to_numpy(), B * H)],
    })
    return rows, B, H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=["train", "test", "validation"])
    ap.add_argument("--mu-stat", default="median", choices=["median", "mean"],
                    help="pool statistic for the Beta mean (default: median)")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-write", action="store_true",
                    help="score only; do not write predictions/metrics")
    ap.add_argument("--oof", action="store_true",
                    help="write only climatology_oof.parquet, the leave-one-year-out "
                         "folds used as an xgb feature; scores nothing")
    args = ap.parse_args()

    if args.oof:
        out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        oof, info = fit_oof(args.mu_stat)
        oof.to_parquet(out_dir / "climatology_oof.parquet", index=False)
        (out_dir / "oof_info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps({k: v for k, v in info.items() if k != "per_fold"}, indent=2))
        print(f"wrote {out_dir}/climatology_oof.parquet ({len(oof):,} rows)")
        return

    table, info = fit(args.mu_stat)
    rows, B, H = build_rows(args.split, args.lookback, args.horizon)
    rows["month"] = pd.to_datetime(rows["target_date"]).dt.month
    rows = rows.merge(table, on=["county_fips", "month"], how="left", validate="m:1")
    assert not rows[["pi", "mu", "phi"]].isna().any().any(), "unresolved climatology cell"

    pi = torch.from_numpy(rows["pi"].to_numpy())
    args_t = (torch.log(pi / (1 - pi)),                       # pi_logit, logit gate link
              torch.from_numpy(rows["mu"].to_numpy()),
              torch.from_numpy(rows["phi"].to_numpy()),
              torch.from_numpy(rows["y_true"].to_numpy().astype(np.float64)))
    metrics = compute_metrics(*args_t, link="logit")
    metrics["per_horizon_nll"] = per_horizon_nll(
        *(t.reshape(B, H, -1) for t in args_t), link="logit")

    # Climatology is origin-independent, so the windowed score is a multiplicity-weighted
    # mean over distinct county-months. Report the unweighted one so the weighting is visible.
    distinct = rows.drop_duplicates(["county_fips", "target_date"])
    dpi = torch.from_numpy(distinct["pi"].to_numpy())
    metrics["nll_distinct_county_months"] = float(zib_nll(
        torch.log(dpi / (1 - dpi)),
        torch.from_numpy(distinct["mu"].to_numpy()),
        torch.from_numpy(distinct["phi"].to_numpy()),
        torch.from_numpy(distinct["y_true"].to_numpy().astype(np.float64)), link="logit"))
    metrics["n_rows"] = int(len(rows))
    metrics["n_distinct_county_months"] = int(len(distinct))

    print(json.dumps({"fit": info, args.split: metrics}, indent=2))

    if args.no_write:
        return
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows["p_occ"] = 1 - rows["pi"]
    rows["e_y"] = rows["p_occ"] * rows["mu"]
    rows.drop(columns=["month", "pi"]).to_parquet(
        out_dir / f"predictions_{args.split}.parquet", index=False)
    table.to_parquet(out_dir / "climatology_table.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {"lookback": args.lookback, "horizon": args.horizon,
         "min_pos_mu": MIN_POS_MU, "min_pos_phi": MIN_POS_PHI,
         "phi_floor": PHI_FLOOR, **info}, indent=2))
    path = out_dir / "metrics.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev[args.split] = metrics
    path.write_text(json.dumps(prev, indent=2))
    print(f"wrote {out_dir}/predictions_{args.split}.parquet ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
