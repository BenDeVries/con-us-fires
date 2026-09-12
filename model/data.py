"""Tensorize the county-month panel into spatio-temporal windows.

Panel is a clean rectangle: T months x N=3108 counties. We build a global month
index over all 264 months (train+test+val concatenated) so that an encoder's
36-month history can legitimately reach back across a split boundary -- that is
just "the past" at forecast time, not leakage. A window is assigned to a split by
where its 12-month *horizon* falls; windows whose horizon straddles a boundary are
dropped. Targets from a later split are never visible to an earlier split.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .config import DATA_DIR

# Columns that are never predictors: identifiers, target, the zero-gate label,
# and meta/leakage columns. frac_valid_burn corr 0.9991 with the target.
ID_COLS = {"county_fips", "date", "year", "month", "node_id"}
TARGET_COL = "burned_fraction"
GATE_COL = "fire_occurred"
LEAK_META = {"frac_valid_burn", "frac_valid_obs", "impute_flag", "lc_impute_flag",
             "county_area_km2"}
CAT_COL = "lc_dominant"

# Predictors with no meaningful within-county temporal variation: the five that are exactly
# constant (terrain, fuel) plus the seven annual-cadence step functions (land cover, human),
# which carry <5% of their variance within county and are flat across all 12 months of a year.
# Held out of the PCA rotation so the rotation covers the 39 genuinely dynamic predictors —
# the same split as `inla_glm_comparison.qmd`'s blocks_dyn/blocks_stat.
STATIC_COLS = ("aspect_cos", "aspect_sin", "built_frac", "elev", "evc_mean", "pct_crop",
               "pct_forest", "pct_grass", "pct_shrub", "pct_urban", "pop_density", "slope")

# Strict allowlist: annual/epoch products and fuel composites are not origin-available.
FORECAST_COLS = ("aspect_cos", "aspect_sin", "elev", "slope")

# Bypassing the rotation means these columns reach the GCN input undiluted, so their raw skew
# now matters: z-scored, pop_density reaches 39 sigma and pct_shrub 12. Applied on the
# unscaled values, then re-standardized on train rows only.
TAIL_TRANSFORMS = {"pop_density": "log1p", "built_frac": "sqrt",
                   "pct_urban": "sqrt", "pct_shrub": "sqrt"}


@dataclass
class PanelTensors:
    cov: torch.Tensor          # [T, N, F_cov]  decoder + encoder numeric covariates (scaled)
    cat: torch.Tensor          # [T, N]         lc_dominant class index (long)
    ar: torch.Tensor           # [T, N, 2]      encoder-only autoregressive [burned_fraction, occurrence]
    y: torch.Tensor            # [T, N]         target burned_fraction (unscaled, in [0,1))
    dates: np.ndarray          # [T] datetime64
    edge_index: torch.Tensor   # [2, E]
    n_nodes: int
    cov_names: list[str]
    n_lc_classes: int
    split_origins: dict[str, list[int]]   # split -> list of origin month indices
    node_fips: np.ndarray      # [N] county_fips string indexed by node_id
    # Number of trailing cov columns that are Fourier seasonal encodings. They are always the
    # last block, and they are held out of the PCA rotation (see fit_pca).
    n_harmonic_cols: int = 0
    # Number of cov columns immediately before the harmonic block that are the near-constant
    # predictors of STATIC_COLS, likewise held out of the rotation. Layout is
    # [dynamic | static | harmonic]; 0 means the panel was built without the bypass.
    n_static_cols: int = 0
    static_names: tuple[str, ...] = ()
    split_date_ranges: dict[str, tuple] = field(default_factory=dict)
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    pca_explained_variance_ratio: np.ndarray | None = None
    forecast_safe: bool = False


def _load_full_panel() -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    frames, ranges = [], {}
    for split in ("train", "test", "validation"):
        df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
        df["split"] = split
        ranges[split] = (df["date"].min(), df["date"].max())
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["date", "node_id"]).reset_index(drop=True)
    return full, ranges


def _fourier_month(month: np.ndarray, n_harmonics: int = 1) -> list[tuple[str, np.ndarray]]:
    """Fourier encoding: sin/cos pairs for k=1..n_harmonics (period 12/k months).

    k=6 is Nyquist for monthly data: sin(pi*(m-1)) is analytically zero at every integer
    month (~5e-15 after round-off), so the k=1..6 block has rank 11, not 12 — with an
    intercept it saturates the space of functions of calendar month. The dead column is kept
    rather than special-cased: it contributes nothing to any linear layer, and dropping it
    would change the covariate width of every checkpoint trained at n_harmonics=6."""
    result = []
    for k in range(1, n_harmonics + 1):
        rad = 2.0 * np.pi * k * (month - 1) / 12.0
        result.append((f"month_sin{k}", np.sin(rad)))
        result.append((f"month_cos{k}", np.cos(rad)))
    return result


def _tail_correct(full: pd.DataFrame, cols: list[str]) -> None:
    """Variance-stabilize the heavy-tailed static predictors in place.

    step11 already z-scored every column, so the raw value has to be reconstructed from
    feature_scaler.json before the transform and the result re-standardized afterwards. The
    re-standardization uses train rows only, matching the scaler it replaces."""
    scaler = {d["feature"]: (d["mean"], d["std"])
              for d in json.loads((DATA_DIR / "feature_scaler.json").read_text())}
    train_mask = (full["split"] == "train").to_numpy()
    for c in cols:
        kind = TAIL_TRANSFORMS.get(c)
        if kind is None:
            continue
        mean, std = scaler[c]
        raw = full[c].to_numpy(dtype=np.float64) * std + mean
        v = np.log1p(np.clip(raw, 0.0, None)) if kind == "log1p" \
            else np.sqrt(np.clip(raw, 0.0, None))
        t_mean, t_std = v[train_mask].mean(), v[train_mask].std()
        full[c] = ((v - t_mean) / (t_std if t_std > 0 else 1.0)).astype(np.float32)


def fit_pca(panel: PanelTensors) -> PanelTensors:
    """Fit PCA on the train-split *predictor* columns; pass the harmonics through unrotated.

    The Fourier columns are an exact deterministic function of the month, not measured
    covariates, so rotating them in mixes season into every component and leaves seasonality
    only partially recoverable once the tail is truncated — the model loses the one signal it
    can always compute for free. Holding them out also puts `n_pca` on the same footing as the
    xgb baseline's k: both now count predictor columns only (51), where the GNN's k previously
    ran over 51 + 2*n_harmonics.

    The near-constant predictors of STATIC_COLS bypass the rotation on the same argument: they
    carry county identity rather than dynamics, and rotating them in spreads a per-county
    constant across every component, so truncation leaves it only partially recoverable.

    Output column order is [PC1..PC_F, <statics>, <harmonics>], so the harmonics stay the
    trailing block and `n_harmonic_cols` / `n_static_cols` continue to describe the result."""
    if panel.forecast_safe:
        raise ValueError("forecast_safe forbids PCA")
    h = panel.n_harmonic_cols
    s = panel.n_static_cols
    d0, d1 = panel.split_date_ranges["train"]
    train_mask = (panel.dates >= d0) & (panel.dates <= d1)
    cov_np = panel.cov.numpy()
    T, N, F_all = cov_np.shape
    F = F_all - h - s                                        # rotatable predictor columns
    X = cov_np[..., :F]
    X_train = X[train_mask].reshape(-1, F)
    mean = X_train.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    components = Vt.T
    var = (S ** 2) / (X_train.shape[0] - 1)
    var_ratio = var / var.sum()
    pcs = ((X.reshape(-1, F) - mean) @ components).reshape(T, N, F)
    cov = np.concatenate([pcs, cov_np[..., F:]], axis=-1) if (h or s) else pcs
    return dataclasses.replace(panel,
        cov=torch.from_numpy(cov.astype(np.float32)),
        cov_names=[f"PC{i+1}" for i in range(F)] + panel.cov_names[F:],
        pca_mean=mean, pca_components=components,
        pca_explained_variance_ratio=var_ratio)


def truncate_pca(panel: PanelTensors, n_pca: int) -> PanelTensors:
    """Keep the leading n_pca components plus every unrotated static and harmonic column."""
    h = panel.n_harmonic_cols
    s = panel.n_static_cols
    F = panel.cov.shape[-1] - h - s
    k = min(n_pca, F)
    keep = list(range(k)) + list(range(F, F + s + h))
    return dataclasses.replace(panel,
        cov=panel.cov[:, :, keep],
        cov_names=[panel.cov_names[i] for i in keep])


def truncate_harmonics(panel: PanelTensors, n_harmonics: int) -> PanelTensors:
    """Keep the leading n_harmonics sin/cos pairs of the trailing seasonal block.

    The block is ordered [sin1, cos1, sin2, cos2, ...], so order k survives only if every
    lower order does. That nesting is the point: it makes the axis a smoothness restriction
    on the seasonal cycle rather than free subset selection over frequencies. It also lets
    the tuner slice a panel built once at the maximum order, the same trick that makes n_pca
    searchable without rebuilding.

    Over-requesting raises rather than silently training on fewer orders than config.json
    records — model.predict rebuilds the panel from cfg.n_harmonics, so a mismatch there
    would surface as a checkpoint width error much further downstream."""
    h = panel.n_harmonic_cols
    keep = 2 * n_harmonics
    if keep > h:
        raise ValueError(
            f"panel carries {h} harmonic columns ({h // 2} orders) but n_harmonics="
            f"{n_harmonics} was requested; rebuild the panel at that order.")
    if keep == h:
        return panel
    end = panel.cov.shape[-1] - h + keep
    return dataclasses.replace(panel,
        cov=panel.cov[:, :, :end],
        cov_names=panel.cov_names[:end],
        n_harmonic_cols=keep)


def save_pca(panel: PanelTensors, path: Path):
    np.savez(path, mean=panel.pca_mean, components=panel.pca_components,
             explained_variance_ratio=panel.pca_explained_variance_ratio)


def load_pca(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in ("mean", "components", "explained_variance_ratio")}


def apply_saved_pca(panel: PanelTensors, pca: dict, n_pca: int) -> PanelTensors:
    """Transform raw panel.cov using a saved PCA, truncate to n_pca components.

    The rotation's own row count records how many columns it was fit on, which is what
    distinguishes a checkpoint written before the harmonics were held out (rotation covers
    every column) from one written after (rotation covers predictors only). Reading it off
    the artifact keeps every pre-existing checkpoint reproducing its original panel without
    a config flag to set or forget."""
    if panel.forecast_safe:
        raise ValueError("forecast_safe forbids PCA")
    h = panel.n_harmonic_cols
    s = panel.n_static_cols
    cov_np = panel.cov.numpy()
    T, N, F_all = cov_np.shape
    F = int(pca["components"].shape[0])                      # columns the rotation was fit on
    if F == F_all:
        h, s = 0, 0                  # legacy checkpoint: harmonics were inside the rotation
    elif F == F_all - h - s:
        pass                         # current scheme (s = 0 reproduces the pre-bypass panel)
    elif s and F == F_all - h:
        raise ValueError(
            f"saved PCA was fit on {F} columns, which covers the {s} static predictors, but "
            f"this panel was built with static_bypass=True (and so has them tail-transformed "
            f"and held out). Set static_bypass=False in config.json to reproduce it.")
    else:
        raise ValueError(
            f"saved PCA was fit on {F} columns but this panel has {F_all} ({h} harmonic, "
            f"{s} static). n_harmonics most likely differs from the training run — "
            f"check config.json.")
    k = min(n_pca, F)
    pcs = ((cov_np[..., :F].reshape(-1, F) - pca["mean"])
           @ pca["components"][:, :k]).reshape(T, N, k)
    cov = np.concatenate([pcs, cov_np[..., F:]], axis=-1) if (h or s) else pcs
    return dataclasses.replace(panel,
        cov=torch.from_numpy(cov.astype(np.float32)),
        cov_names=[f"PC{i+1}" for i in range(k)] + (panel.cov_names[F:] if (h or s) else []),
        n_harmonic_cols=h, n_static_cols=s)


def numeric_predictor_names() -> list[str]:
    """Every numeric predictor the panel can carry — the full domain of `drop_features`."""
    meta = json.loads((DATA_DIR / "feature_metadata.json").read_text())
    return sorted(k for k, v in meta.items()
                  if v.get("role") == "predictor" and k != CAT_COL and k not in LEAK_META)


def build_panel(lookback: int, horizon: int,
                drop_features: list[str] | tuple[str, ...] = (),
                n_harmonics: int = 1,
                static_bypass: bool = False,
                forecast_safe: bool = False) -> PanelTensors:
    if forecast_safe and static_bypass:
        raise ValueError("forecast_safe forbids static bypass")
    full, ranges = _load_full_panel()

    dates = np.sort(full["date"].unique())
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    N = int(full["node_id"].max()) + 1
    assert full.groupby("date")["node_id"].nunique().eq(N).all(), "panel not rectangular"

    numeric_cov = list(FORECAST_COLS) if forecast_safe else numeric_predictor_names()
    if drop_features and not forecast_safe:
        unknown = set(drop_features) - set(numeric_cov)
        if unknown:
            raise ValueError(f"drop_features not in panel predictors: {sorted(unknown)}")
        numeric_cov = [c for c in numeric_cov if c not in set(drop_features)]

    # Segregating the statics into a trailing block is what lets fit_pca/apply_saved_pca hold
    # them out by a slice. Left alone when off, so a panel built without the bypass is
    # column-for-column what it was before the option existed.
    static_names: tuple[str, ...] = ()
    if static_bypass:
        static_names = tuple(c for c in numeric_cov if c in STATIC_COLS)
        numeric_cov = [c for c in numeric_cov if c not in STATIC_COLS] + list(static_names)
        _tail_correct(full, list(static_names))

    full["_t"] = full["date"].map(date_to_idx).astype(int)
    full["_n"] = full["node_id"].astype(int)

    def to_grid(col: str, dtype=np.float32) -> np.ndarray:
        g = np.zeros((T, N), dtype=dtype)
        g[full["_t"].to_numpy(), full["_n"].to_numpy()] = full[col].to_numpy()
        return g

    cov_stack = [to_grid(c) for c in numeric_cov]
    if forecast_safe:
        # Terrain is fixed, and using its first training snapshot makes that information
        # boundary explicit even if a later panel file is accidentally modified.
        first_train = date_to_idx[np.datetime64(ranges["train"][0])]
        cov_stack = [np.broadcast_to(g[first_train:first_train + 1], g.shape).copy()
                     for g in cov_stack]
    month = (pd.DatetimeIndex(full["date"]).month.to_numpy() if forecast_safe
             else full["month"].to_numpy())
    harmonics = _fourier_month(month, n_harmonics)
    harm_names = []
    for name, arr in harmonics:
        full[f"_{name}"] = arr
        cov_stack.append(to_grid(f"_{name}"))
        harm_names.append(name)
    cov_names = numeric_cov + harm_names
    cov = np.stack(cov_stack, axis=-1)                       # [T, N, F_cov]

    if forecast_safe:
        classes = np.array([0])
        cat = np.zeros((T, N), dtype=np.int64)
    else:
        classes = np.sort(full[CAT_COL].dropna().unique())
        lc_map = {int(c): i for i, c in enumerate(classes)}
        cat_codes = full[CAT_COL].map(lc_map).fillna(0).astype(int)
        full["_lc"] = cat_codes
        cat = to_grid("_lc", dtype=np.int64).astype(np.int64)   # [T, N]

    y = to_grid(TARGET_COL)                                 # [T, N]
    occ = (y > 0).astype(np.float32) if forecast_safe else to_grid(GATE_COL)
    ar = np.stack([y, occ], axis=-1)                        # [T, N, 2] encoder-only

    edge = np.load(DATA_DIR / "county_graph.npz")["edge_index"]
    edge_index = torch.from_numpy(edge.astype(np.int64))

    split_origins = rebuild_origins(dates, ranges, lookback, horizon)

    node_index = json.loads((DATA_DIR / "node_index.json").read_text())
    node_fips = np.empty(N, dtype=object)
    for fips, nid in node_index.items():
        node_fips[int(nid)] = str(fips).zfill(5)

    return PanelTensors(
        cov=torch.from_numpy(cov),
        cat=torch.from_numpy(cat),
        ar=torch.from_numpy(ar),
        y=torch.from_numpy(y),
        dates=dates,
        edge_index=edge_index,
        n_nodes=N,
        cov_names=cov_names,
        n_lc_classes=len(classes),
        split_origins=split_origins,
        node_fips=node_fips,
        n_harmonic_cols=len(harm_names),
        n_static_cols=len(static_names),
        static_names=static_names,
        split_date_ranges=ranges,
        forecast_safe=forecast_safe,
    )


def rebuild_origins(dates, ranges, lookback, horizon) -> dict[str, list[int]]:
    """Origin o => encoder months [o-lookback+1 .. o], horizon [o+1 .. o+horizon].
    Assign to a split iff the whole horizon lies inside that split's date range.

    Public so a tuner sharing one panel across trials can re-derive the origins for a
    different lookback (`dataclasses.replace(panel, split_origins=...)`) instead of
    rebuilding the whole panel."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must both be positive")
    date_to_idx = {d: i for i, d in enumerate(dates)}
    occupied = set()
    origins: dict[str, list[int]] = {}
    for split, (d0, d1) in ranges.items():
        s0, s1 = date_to_idx[np.datetime64(d0)], date_to_idx[np.datetime64(d1)]
        block = set(range(s0, s1 + 1))
        if s1 < s0 or occupied.intersection(block):
            raise ValueError("split target date ranges must be nonempty and disjoint")
        occupied.update(block)
        lo = max(lookback - 1, s0 - 1)          # need full history and horizon start in split
        hi = s1 - horizon                        # horizon end in split
        origins[split] = list(range(lo, hi + 1)) if hi >= lo else []
    return origins


class WindowDataset(torch.utils.data.Dataset):
    """One item = one forecast origin spanning all N counties."""

    def __init__(self, panel: PanelTensors, origins: list[int], lookback: int, horizon: int):
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must both be positive")
        if any(o - lookback + 1 < 0 or o + horizon >= len(panel.dates) for o in origins):
            raise ValueError("forecast origin has insufficient history or out-of-bounds targets")
        self.p = panel
        self.origins = origins
        self.L = lookback
        self.H = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int):
        o = self.origins[i]
        enc = slice(o - self.L + 1, o + 1)       # [L]
        dec = slice(o + 1, o + 1 + self.H)       # [H]
        return {
            "enc_cov": self.p.cov[enc],          # [L, N, F]
            "enc_cat": self.p.cat[enc],          # [L, N]
            "enc_ar": self.p.ar[enc],            # [L, N, 2]
            "dec_cov": self.p.cov[dec],          # [H, N, F]
            "dec_cat": self.p.cat[dec],          # [H, N]
            "y": self.p.y[dec],                  # [H, N]
            "origin": torch.tensor(o),           # scalar month index of forecast origin
        }
