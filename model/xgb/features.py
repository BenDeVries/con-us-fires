"""Feature engineering for the XGBoostLSS h-step wildfire forecast baseline.

We mirror the NN's framing (model/data.py): a global month index over all 264
months, forecast origins assigned to a split iff the whole 12-month horizon lies
inside that split's date range, and history that legitimately reaches across split
boundaries (that is "the past" at forecast time, not leakage).

Each emitted row is one (origin o, horizon h, county n) with target month m = o+h:
  * decoder/target-month covariates COV[m]  -- the NN decoder is fed these too,
  * the horizon index h,
  * origin-anchored history (<= o only): autoregressive lags, rolling means,
    months-since-fire, and a same-calendar-month-last-year seasonal term Y[m-12],
  * queen-neighbor aggregates of that history (a tabular stand-in for the GCN),
  * a train-only per-county baseline (a stand-in for the NN's county embedding).

The historical default reconstructs older conditional designs, including their full-training
county summaries; it is not an admissible origin-safe forecast. New publication runs must
use ``forecast_safe=True``: terrain is frozen at the first training month, land cover is
a constant placeholder, and county summaries contain only observations strictly before
the origin. Every other response-derived input is at or before the origin. Annual,
epoch, weather, climatology and covariate-PCA inputs are excluded in that protocol.

Two optional blocks widen that stand-in into an explicit space-time expansion, both
off by default so every on-disk arm rebuilds its original design:

  * `max_spatial` / `max_temporal` -- fire history on Pfeifer-Deutsch exclusive shells
    s=0..S at origin-anchored lags k=0..P (`st_{y,occ}_s{l}_t{k}`). Bounded by what
    the GNN's encoder consumes: its AR block enters `enc_gcn` at every month of the
    lookback window, so shells <= gcn_layers and lags <= lookback-1 are inside its
    information set.
  * `n_nbr_pcs` -- neighbour means of the leading covariate PCs at the *target* month
    (`nbpc_s{l}_PC{j}`). This is the tabular form of the GNN's `dec_gcn`, which smooths
    target-month covariates over the same shells; without it xgb is the only class with
    no neighbour covariates at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import FORECAST_COLS, STATIC_COLS, _fourier_month
from ..spatial import exclusive_orders
from . import reduction as red_mod

MAX_HARMONICS = 6        # Nyquist for monthly data; see model/data.py:_fourier_month

DATA_DIR = Path("output/data")
CLIM_TABLE = Path("output/climatology/climatology_table.parquet")
CLIM_OOF = Path("output/climatology/climatology_oof.parquet")

# Same conventions as model/data.py.
ID_COLS = {"county_fips", "date", "year", "month", "node_id"}
TARGET_COL = "burned_fraction"
GATE_COL = "fire_occurred"
LEAK_META = {"frac_valid_burn", "frac_valid_obs", "impute_flag", "lc_impute_flag",
             "county_area_km2"}
CAT_COL = "lc_dominant"

MSF_CAP = 240.0          # cap (months) for months-since-last-fire
ROLL_WINDOWS = (3, 6, 12)
AR_LAGS = (1, 3, 6, 12)

ST_CHANNELS = ("y", "occ")
# Shell-0 cells the baseline history block already carries, so the expansion does not
# emit an exact duplicate of an existing column.
ST_SKIP = {("y", 0, 0), ("occ", 0, 0)} | {("y", 0, k) for k in AR_LAGS}


@dataclass
class Dataset:
    splits: dict[str, dict]      # split -> {"X": DataFrame, "y": ndarray, "meta": DataFrame}
    num_features: list[str]
    cat_feature: str
    lc_categories: list[int]
    feature_names: list[str]
    raw_cov_names: list[str] = field(default_factory=list)   # the reducible covariate block
    reduction: str = "none"
    forecast_safe: bool = False


def apply_reduction(ds: Dataset, red) -> Dataset:
    """Replace the target-month covariate block with principal-component scores.

    Only the leading `len(red.cov_names)` columns are touched. Everything downstream
    of them -- the Fourier pair, the origin-anchored history, the neighbour
    aggregates, the county priors, the horizon index and the land-cover category --
    is carried through unrotated, so forecast skill remains attributable to the
    origin anchoring rather than being mixed into the covariate components.
    """
    if red is None:
        return ds
    if ds.forecast_safe:
        raise ValueError("forecast_safe does not permit covariate PCA")
    p = len(red.cov_names)
    if ds.num_features[:p] != list(red.cov_names):
        raise ValueError("reduction covariate block does not match the design's leading columns")
    rest = ds.num_features[p:]
    num_features = list(red.out_names) + rest

    splits = {}
    for split, d in ds.splits.items():
        X = d["X"]
        raw = X[red.cov_names].to_numpy(dtype=np.float32)
        Z = np.empty((len(raw), red.k), dtype=np.float32)
        for lo in range(0, len(raw), 2_000_000):        # bound peak memory on the 5.1M-row train split
            hi = min(lo + 2_000_000, len(raw))
            Z[lo:hi] = red.transform(raw[lo:hi])
        new = pd.concat([pd.DataFrame(Z, columns=red.out_names, index=X.index),
                         X[rest + [CAT_COL]]], axis=1)
        splits[split] = {"X": new, "y": d["y"], "meta": d["meta"]}

    return Dataset(splits=splits, num_features=num_features, cat_feature=CAT_COL,
                   lc_categories=ds.lc_categories,
                   feature_names=num_features + [CAT_COL],
                   raw_cov_names=list(red.cov_names), reduction=red.arm)


_ST_RE = re.compile(r"^st_(?:y|occ)_s(\d+)_t(\d+)$")
_NBPC_RE = re.compile(r"^nbpc_s(\d+)_PC(\d+)$")
_HARM_RE = re.compile(r"^month_(?:sin|cos)(\d+)$")


def select_columns(ds: Dataset, n_spatial: int | None = None,
                   n_temporal: int | None = None, n_nbr_pcs: int | None = None,
                   n_harmonics: int | None = None) -> Dataset:
    """Narrow an assembled design to a smaller (shell, lag, PC, harmonic) budget.

    The tuner assembles the maximal design once and calls this per trial, so searching the
    lag counts costs a column slice rather than a 5M-row re-assembly. One `n_spatial`
    governs both space-time blocks, mirroring the GNN's single `gcn_layers` over its
    encoder and decoder graph convolutions. `None` on any axis leaves that block whole,
    so a study may search the lags, the seasonal order, or both.
    """
    def le(value: int, cap: int | None) -> bool:
        return cap is None or value <= cap

    def keep(name: str) -> bool:
        m = _ST_RE.match(name)
        if m:
            return le(int(m.group(1)), n_spatial) and le(int(m.group(2)), n_temporal)
        m = _NBPC_RE.match(name)
        if m:
            return le(int(m.group(1)), n_spatial) and le(int(m.group(2)), n_nbr_pcs)
        m = _HARM_RE.match(name)
        if m:
            return le(int(m.group(1)), n_harmonics)
        return True

    num_features = [c for c in ds.num_features if keep(c)]
    if num_features == ds.num_features:
        return ds
    cols = num_features + [ds.cat_feature]
    splits = {s: {"X": d["X"][cols], "y": d["y"], "meta": d["meta"]}
              for s, d in ds.splits.items()}
    return Dataset(splits=splits, num_features=num_features, cat_feature=ds.cat_feature,
                   lc_categories=ds.lc_categories, feature_names=cols,
                   raw_cov_names=list(ds.raw_cov_names), reduction=ds.reduction,
                   forecast_safe=ds.forecast_safe)


def _load_full_panel(columns: list[str] | None = None):
    frames, ranges = [], {}
    for split in ("train", "test", "validation"):
        df = pd.read_parquet(DATA_DIR / f"{split}.parquet", columns=columns)
        df["split"] = split
        ranges[split] = (df["date"].min(), df["date"].max())
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["date", "node_id"]).reset_index(drop=True)
    return full, ranges


def _numeric_cov_cols() -> list[str]:
    meta = json.loads((DATA_DIR / "feature_metadata.json").read_text())
    cols = [k for k, v in meta.items()
            if v.get("role") == "predictor" and k != CAT_COL and k not in LEAK_META]
    return sorted(cols)


def _row_normalized_adj(n_nodes: int) -> sp.csr_matrix:
    """Row-normalized queen adjacency: A[i,j] = 1/deg_i for neighbours j of i."""
    edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
    src, dst = edge[0], edge[1]
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def _rolling_mean(grid: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over [t-w+1, t] along axis 0; first w-1 rows left as NaN."""
    csum = np.cumsum(grid, axis=0)
    out = np.full_like(grid, np.nan)
    out[w - 1:] = csum[w - 1:]
    out[w:] -= csum[:-w]
    out[w - 1:] /= w
    return out


def _months_since_fire(occ: np.ndarray) -> np.ndarray:
    """As of month t (inclusive): 0 if fire at t, else months back to the last fire,
    capped at MSF_CAP; capped if no prior fire."""
    T, N = occ.shape
    out = np.empty((T, N), dtype=np.float32)
    last = np.full(N, np.nan)
    idx = np.arange(N)
    for t in range(T):
        fired = occ[t] > 0
        cur = np.where(np.isnan(last), MSF_CAP, np.minimum(t - last, MSF_CAP))
        out[t] = np.where(fired, 0.0, cur)
        last[idx[fired]] = t
    return out


def _nbr(grid: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """Neighbour mean of a [T,N] grid: (A @ grid.T).T."""
    return (A @ grid.T).T


def _st_cells(max_spatial: int, max_temporal: int):
    """(channel, shell, lag) triples of the space-time block, shell-major, lag-ascending."""
    for ch in ST_CHANNELS:
        for l in range(max_spatial + 1):
            for k in range(max_temporal + 1):
                if (ch, l, k) not in ST_SKIP:
                    yield ch, l, k


def _st_grids(Y: np.ndarray, OCC: np.ndarray, W: list, max_spatial: int) -> dict:
    """Spatially lag each channel once per shell; temporal lags are then a row index.

    W[0] is the identity, so shell 0 is the county's own history.
    """
    src = {"y": Y, "occ": OCC}
    return {(ch, l): (src[ch] if l == 0 else (W[l] @ src[ch].T).T.astype(np.float32))
            for ch in ST_CHANNELS for l in range(max_spatial + 1)}


def _nbr_pc_grids(COV: np.ndarray, num_cov: list[str], W: list, max_spatial: int,
                  n_nbr_pcs: int) -> tuple[dict, object]:
    """Neighbour means of the leading covariate PCs, per shell, indexed by global month.

    The rotation is `reduction.fit_from_panel`'s global arm, so it is fitted on the
    *distinct* train-split panel rows (never the assembled design, which repeats each
    target month once per origin) and is the same sign-anchored rotation the reduction
    ladder uses. The columns arrive train-z-scored from step11, so nothing is
    re-standardised.
    """
    rot = red_mod.fit_from_panel(num_cov, "global", n_pca=n_nbr_pcs)
    T, N, _ = COV.shape
    pcs = rot.transform(COV.reshape(T * N, -1).astype(np.float64))
    pcs = pcs.reshape(T, N, rot.k).astype(np.float32)
    grids = {l: np.stack([(W[l] @ pcs[..., j].T).T for j in range(rot.k)], axis=-1)
             .astype(np.float32)
             for l in range(1, max_spatial + 1)}
    return grids, rot


CLIM_COLS = ("clim_p_occ", "clim_mu", "clim_phi")


def _clim_cols(t: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"clim_p_occ": 1.0 - t["pi"].to_numpy(), "clim_mu": t["mu"].to_numpy(),
            "clim_phi": t["phi"].to_numpy()}


def _climatology_grids(node_fips: np.ndarray, n_nodes: int, dates: np.ndarray,
                       train_span: tuple[int, int], oof: bool = False) -> dict[str, np.ndarray]:
    """[T, N] county ZIB parameters indexed by *global month*, not calendar month.

    Read from `model/climatology.py`'s fitted table rather than recomputed, so the arm is
    handed the floor's own estimate: an empirical-Bayes posterior mean over cells, which
    is a far better estimator of a county's seasonal rate than `y_seasonal` = Y[m-12], one
    draw from a 93%-zero process.

    The full-train table is target encoding when used on train rows -- the cell estimate
    contains the label of the row being fit -- which inflates it in-sample (gate AP 0.58
    against 0.47 at test) and makes the booster over-trust it. With `oof`, train months
    instead take the leave-one-year-out fold that excludes their own year, so no train
    row's feature sees its own target. Test and validation months keep the full table,
    which was never contaminated. Indexing by global month is what lets one grid hold
    both regimes.

    `train_span` is required because the split boundary falls mid-year: train ends
    2018-05, so selecting a fold by calendar year alone would also swap the encoding
    under the 2018 test months, which are not contaminated and must keep the full table.
    """
    if not CLIM_TABLE.exists():
        raise SystemExit(f"missing {CLIM_TABLE} -- build it with: "
                         f"conda run -n pytorch python -m model.climatology")
    T = len(dates)
    pos = {f: i for i, f in enumerate(node_fips)}

    def node_idx(t):
        t["county_fips"] = t["county_fips"].astype(str).str.zfill(5)
        idx = t["county_fips"].map(pos)
        if idx.isna().any():
            raise SystemExit("climatology table carries counties absent from node_index.json")
        return idx.to_numpy().astype(int)

    cal = pd.DatetimeIndex(dates).month.to_numpy()
    yr = pd.DatetimeIndex(dates).year.to_numpy()

    full = pd.read_parquet(CLIM_TABLE)
    fn, fm, fv = node_idx(full), full["month"].to_numpy() - 1, _clim_cols(full)
    out = {name: np.full((T, n_nodes), np.nan, dtype=np.float32) for name in CLIM_COLS}
    by_cal = {}
    for name in CLIM_COLS:
        g = np.full((12, n_nodes), np.nan, dtype=np.float32)
        g[fm, fn] = fv[name]
        if np.isnan(g).any():
            raise SystemExit(f"climatology grid {name} is not complete over county x month")
        by_cal[name] = g
        out[name][:] = g[cal - 1]

    if oof:
        if not CLIM_OOF.exists():
            raise SystemExit(f"missing {CLIM_OOF} -- build it with: conda run -n pytorch "
                             f"python -m model.climatology --oof")
        o = pd.read_parquet(CLIM_OOF)
        on, om, ov = node_idx(o), o["month"].to_numpy() - 1, _clim_cols(o)
        oy = o["fold_year"].to_numpy()
        t0, t1 = train_span
        in_train = np.zeros(T, dtype=bool)
        in_train[t0:t1 + 1] = True
        for fold in np.unique(oy):
            rows = np.flatnonzero((yr == fold) & in_train)
            if not len(rows):
                continue
            sel = oy == fold
            for name in CLIM_COLS:
                g = by_cal[name].copy()
                g[om[sel], on[sel]] = ov[name][sel]
                out[name][rows] = g[cal[rows] - 1]
    return out


def _build_origins(dates, ranges, lookback, horizon) -> dict[str, list[int]]:
    date_to_idx = {d: i for i, d in enumerate(dates)}
    origins: dict[str, list[int]] = {}
    for split, (d0, d1) in ranges.items():
        s0, s1 = date_to_idx[np.datetime64(d0)], date_to_idx[np.datetime64(d1)]
        lo = max(lookback - 1, s0 - 1)
        hi = s1 - horizon
        origins[split] = list(range(lo, hi + 1)) if hi >= lo else []
    return origins


def build_dataset(lookback: int = 36, horizon: int = 12,
                  max_origins: int | None = None,
                  climatology: bool = False, clim_oof: bool = False,
                  max_spatial: int = 0, max_temporal: int = 0, n_nbr_pcs: int = 0,
                  static_only: bool = False, n_harmonics: int = 1,
                  want_splits: tuple[str, ...] | None = None,
                  forecast_safe: bool = False) -> Dataset:
    if forecast_safe:
        if static_only or climatology or clim_oof or n_nbr_pcs:
            raise ValueError("forecast_safe forbids static_only, climatology, clim_oof and neighbour PCs")
        if lookback != 48 or horizon != 12:
            raise ValueError("forecast_safe fixes lookback=48 and horizon=12")
        if not 0 <= max_spatial <= 3 or not 0 <= max_temporal <= 47:
            raise ValueError("forecast_safe permits spatial shells 0..3 and temporal lags 0..47")
    if not 1 <= horizon <= 12:
        raise ValueError("horizon must be 1..12: seasonal history otherwise exceeds the origin")
    if not 1 <= n_harmonics <= MAX_HARMONICS:
        raise SystemExit(f"n_harmonics={n_harmonics} outside [1, {MAX_HARMONICS}]; "
                         f"{MAX_HARMONICS} is Nyquist for monthly data")
    # Deepest month any block reads at origin o. The space-time block reaches o-max_temporal,
    # but the fixed history block reaches o-12 whatever max_temporal is (AR_LAGS, and
    # y_seasonal at o-11). Both index Y with a bare `o - k`, which for o < k wraps to the END
    # of the panel rather than raising -- future months, silently, as ordinary feature values.
    reach = max(max(AR_LAGS), max_temporal)
    if reach >= lookback:
        driver = (f"max_temporal={max_temporal}" if max_temporal > max(AR_LAGS)
                  else f"the fixed AR/seasonal block (reaches o-{max(AR_LAGS)})")
        raise SystemExit(
            f"lookback={lookback} is too shallow for {driver}: the earliest train origin is "
            f"month {lookback - 1}, and reading {reach} months back from it would index "
            f"before the panel starts. Needs lookback >= {reach + 1}")
    if n_nbr_pcs > 0 and max_spatial == 0:
        raise SystemExit("n_nbr_pcs needs max_spatial >= 1; every nbpc column is a "
                         "neighbour mean over shell 1 or beyond")
    full, ranges = (_load_full_panel(columns=["date", "node_id", TARGET_COL, GATE_COL,
                                            *FORECAST_COLS])
                    if forecast_safe else _load_full_panel())
    # Historical static_only includes annual/epoch predictors and remains only to rebuild
    # old artifacts. forecast_safe uses the audited terrain allowlist instead.
    num_cov = (sorted(FORECAST_COLS) if forecast_safe else
               [c for c in _numeric_cov_cols() if c in STATIC_COLS] if static_only
               else _numeric_cov_cols())

    dates = np.sort(full["date"].unique())
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    N = int(full["node_id"].max()) + 1
    assert full.groupby("date")["node_id"].nunique().eq(N).all(), "panel not rectangular"

    full["_t"] = full["date"].map(date_to_idx).astype(int)
    full["_n"] = full["node_id"].astype(int)
    ti = full["_t"].to_numpy()
    ni = full["_n"].to_numpy()

    def grid(col, dtype=np.float32):
        g = np.zeros((T, N), dtype=dtype)
        g[ti, ni] = full[col].to_numpy()
        return g

    COV = np.stack([grid(c) for c in num_cov], axis=-1)     # [T, N, F]
    Y = grid(TARGET_COL)                                     # [T, N]
    OCC = grid(GATE_COL)                                     # [T, N]
    LC = (np.zeros((T, N), dtype=np.int64) if forecast_safe
          else grid(CAT_COL, dtype=np.int64))                # [T, N]
    if forecast_safe:
        # Freeze the terrain snapshot at the first training month. Even accidental
        # changes to held-out copies of these constant columns cannot affect features.
        first_train = date_to_idx[np.datetime64(ranges["train"][0])]
        COV = np.broadcast_to(COV[first_train:first_train + 1], COV.shape)
        if not np.isfinite(COV).all() or not np.isfinite(Y).all() or not np.isfinite(OCC).all():
            raise ValueError("forecast_safe requires finite terrain, responses and occurrence")
        if not np.array_equal(OCC, (Y > 0).astype(np.float32)):
            raise ValueError("forecast_safe occurrence must agree with burned_fraction > 0")

    month_of_t = pd.DatetimeIndex(dates).month.to_numpy()    # [T]
    HARM = np.stack([v for _, v in _fourier_month(month_of_t, n_harmonics)],
                    axis=-1).astype(np.float32)              # [T, 2*n_harmonics]

    # ---- origin-anchored history grids (use only data up to the index month) ----
    roll = {w: _rolling_mean(Y, w) for w in ROLL_WINDOWS}
    occ12 = _rolling_mean(OCC, 12)
    msf = _months_since_fire(OCC)

    A = _row_normalized_adj(N)
    nb_roll12 = _nbr(roll[12], A)
    nb_occ12 = _nbr(occ12, A)
    nb_Y = _nbr(Y, A)                                        # neighbour mean of raw Y (for seasonal)

    # ---- optional space-time expansion (exclusive shells; both blocks off at 0) ----
    ST = NBPC = None
    if max_spatial > 0 or max_temporal > 0 or n_nbr_pcs > 0:
        edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
        W = exclusive_orders(edge, N, max_spatial, sparse=True)
        ST = _st_grids(Y, OCC, W, max_spatial)
        if n_nbr_pcs > 0 and max_spatial > 0:
            NBPC, _ = _nbr_pc_grids(COV, num_cov, W, max_spatial, n_nbr_pcs)

    # Legacy fits retain their full-training summaries for reproducible rebuilding.
    # Safe forecasts use expanding county history strictly BEFORE the origin, including
    # observations made available since earlier rolling forecasts were issued.
    t0_tr = date_to_idx[np.datetime64(ranges["train"][0])]
    t1_tr = date_to_idx[np.datetime64(ranges["train"][1])]
    if forecast_safe:
        denom = np.maximum(np.arange(T), 1)[:, None]
        county_mean_y = np.vstack([np.zeros((1, N)), np.cumsum(Y[:-1], axis=0,
                                   dtype=np.float64)]) / denom
        county_occ_rate = np.vstack([np.zeros((1, N)), np.cumsum(OCC[:-1], axis=0,
                                     dtype=np.float64)]) / denom
    else:
        county_mean_y = Y[t0_tr:t1_tr + 1].mean(axis=0)
        county_occ_rate = OCC[t0_tr:t1_tr + 1].mean(axis=0)

    node_index = json.loads((DATA_DIR / "node_index.json").read_text())
    node_fips = np.empty(N, dtype=object)
    for fips, nid in node_index.items():
        node_fips[int(nid)] = str(fips).zfill(5)

    lc_categories = sorted(int(c) for c in np.unique(LC))

    # ---- feature name layout (column order is fixed and reused at predict time) ----
    harm_names = [n for n, _ in _fourier_month(month_of_t[:1], n_harmonics)]
    dec_names = list(num_cov) + harm_names
    hist_names = (["y_o", "occ_o"]
                  + [f"y_lag{k}" for k in AR_LAGS]
                  + [f"y_roll{w}" for w in ROLL_WINDOWS]
                  + ["occ_rate12", "months_since_fire", "y_seasonal"])
    nbr_names = ["nb_y_roll12", "nb_occ_rate12", "nb_y_seasonal"]
    county_names = ["county_mean_y", "county_occ_rate"]
    CLIM = (_climatology_grids(node_fips, N, dates, (t0_tr, t1_tr), oof=clim_oof)
            if climatology else None)
    clim_names = list(CLIM) if CLIM is not None else []
    st_cells = list(_st_cells(max_spatial, max_temporal)) if ST is not None else []
    st_names = [f"st_{ch}_s{l}_t{k}" for ch, l, k in st_cells]
    nbpc_cells = ([(l, j) for l in range(1, max_spatial + 1) for j in range(NBPC[1].shape[-1])]
                  if NBPC is not None else [])
    nbpc_names = [f"nbpc_s{l}_PC{j + 1}" for l, j in nbpc_cells]
    extra_names = ["horizon"]
    num_features = (dec_names + hist_names + nbr_names + county_names
                    + clim_names + st_names + nbpc_names + extra_names)
    feature_names = num_features + [CAT_COL]

    origins = _build_origins(dates, ranges, lookback, horizon)
    if max_origins is not None:
        origins = {s: o[:max_origins] for s, o in origins.items()}
    node_ids = np.arange(N)

    def assemble(split_origins):
        if not split_origins:
            return None
        # Preallocated: the space-time expansion can take the design past 600 columns, and
        # the accumulate-then-concatenate form would hold two copies of a 12 GB train block.
        n_rows = len(split_origins) * horizon * N
        Xnum = np.empty((n_rows, len(num_features)), dtype=np.float32)
        yb, mo, mt, mh = [], [], [], []
        r = 0
        for o in split_origins:
            ms = np.arange(o + 1, o + 1 + horizon)          # target months [H]
            seas = ms - 12                                   # same-month-last-year [H], <= o
            for hi_, m in enumerate(ms):                     # one horizon step at a time
                h = hi_ + 1
                dec = np.concatenate(
                    [COV[m], np.broadcast_to(HARM[m], (N, HARM.shape[1]))], axis=1)
                hist = np.column_stack([
                    Y[o], OCC[o],
                    *[Y[o - k] for k in AR_LAGS],
                    *[roll[w][o] for w in ROLL_WINDOWS],
                    occ12[o], msf[o], Y[seas[hi_]],
                ]).astype(np.float32)
                nbr = np.column_stack([nb_roll12[o], nb_occ12[o], nb_Y[seas[hi_]]]).astype(np.float32)
                county = np.column_stack([county_mean_y[o], county_occ_rate[o]]
                                         if forecast_safe else
                                         [county_mean_y, county_occ_rate]).astype(np.float32)
                extra = np.full((N, 1), h, dtype=np.float32)
                blocks = [dec, hist, nbr, county]
                if CLIM is not None:
                    blocks.append(np.column_stack([CLIM[c][m] for c in clim_names]))
                if st_cells:
                    # `o - k`, never `m - k`: a lag anchored on the target month would sit
                    # after the forecast origin for every horizon h > k.
                    blocks.append(np.column_stack([ST[(ch, l)][o - k] for ch, l, k in st_cells]))
                if nbpc_cells:
                    blocks.append(np.column_stack([NBPC[l][m, :, j] for l, j in nbpc_cells]))
                blocks.append(extra)
                Xnum[r:r + N] = np.concatenate(blocks, axis=1)
                r += N
                yb.append(Y[m].copy())
                mo.append(np.full(N, dates[o]))
                mt.append(np.full(N, dates[m]))
                mh.append(np.full(N, h, dtype=np.int16))
        df = pd.DataFrame(Xnum, columns=num_features)
        # categorical lc, target month, aligned with the row blocks built above
        lc_blocks = []
        for o in split_origins:
            for m in range(o + 1, o + 1 + horizon):
                lc_blocks.append(LC[m])
        df[CAT_COL] = pd.Categorical(np.concatenate(lc_blocks),
                                     categories=lc_categories)
        y = np.concatenate(yb).astype(np.float32)
        meta = pd.DataFrame({
            "origin_date": np.concatenate(mo),
            "horizon": np.concatenate(mh),
            "target_date": np.concatenate(mt),
            "county_fips": np.tile(node_fips, len(yb)),
            "node_id": np.tile(node_ids, len(yb)),
        })
        return {"X": df, "y": y, "meta": meta}

    splits = {}
    for split, ors in origins.items():
        if want_splits is not None and split not in want_splits:
            continue
        built = assemble(ors)
        if built is not None:
            splits[split] = built

    return Dataset(splits=splits, num_features=num_features, cat_feature=CAT_COL,
                   lc_categories=lc_categories, feature_names=feature_names,
                   raw_cov_names=list(num_cov), forecast_safe=forecast_safe)
