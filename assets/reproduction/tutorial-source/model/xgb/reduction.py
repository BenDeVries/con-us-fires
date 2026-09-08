"""Principal-component reduction of the target-month covariate block.

Two arms, mirroring `inla_glm_comparison.qmd` §6.2/§6.5 and `model/data.py:fit_pca`:

  * ``global`` -- one rotation over all 51 predictors, top k components. The direct
    analogue of the GNN's ``cfg.n_pca``: both count predictor columns only, because
    both hold the Fourier columns out of the rotation (`model/data.py:fit_pca`), so
    the two k's are on the same scale. Under ``static_bypass`` both instead count the
    39 dynamic predictors, matching `inla_glm_comparison.qmd`'s arm G as well.
  * ``block``  -- a rotation per thematic block (weather, drought, ...), retaining
    enough components for ``var_frac`` of block variance subject to an eigenvalue
    floor. Components stay physically nameable. Single-variable blocks pass through.

Three construction rules that the rest of the pipeline depends on:

  1. The rotation is fitted on the *distinct* train-split panel rows, not on the
     assembled design matrix. The design repeats each target month once per origin
     (up to 12 times), which would weight months by multiplicity; the panel does not.
     `model/data.py:fit_pca` fits on the panel, and matching it is what makes the
     boosted and neural rotations comparable.
  2. The 51 columns arrive already train-z-scored from step11, so a covariance
     rotation equals a correlation rotation. Nothing is re-standardised on input.
  3. Each component's sign is anchored so its loading on a designated reference
     variable is positive, which fixes the orientation of every reported loading.

Scores are divided by their training standard deviation. That is a per-column
monotone map and therefore cannot change a boosted-tree fit; it is done so the
components are on a common scale in the diagnostics and loading tables.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import STATIC_COLS

DATA_DIR = Path("output/data")

# Reference variable per block, copied from inla_glm_comparison.qmd:801 so the two
# documents' loading tables carry the same orientation.
BLOCK_ANCHOR = {
    "weather": "erc_max", "drought": "eddi1y", "waterbalance": "pet",
    "vegetation": "ndvi_lag1", "snow": "snow_frac", "landcover": "pct_forest",
    "human": "pop_density", "terrain": "elev", "fuel": "evc_mean",
}
BLOCK_PREFIX = {
    "weather": "wx", "drought": "dr", "waterbalance": "wb", "vegetation": "vg",
    "snow": "sn", "landcover": "lc", "human": "hm", "terrain": "tr", "fuel": "fu",
}
GLOBAL_ANCHOR = "erc_max"


@dataclass
class Reduction:
    arm: str                     # "global" | "block"
    cov_names: list[str]         # the raw inputs, in the order `rotation` expects
    out_names: list[str]         # PC1..PCk, or wx_PC1, dr_PC1, ...
    mean: np.ndarray             # [p]
    rotation: np.ndarray         # [p, k]
    sd: np.ndarray               # [k] training SD of the un-normalised scores
    eigen: dict                  # per-block or global eigenvalues + cumulative variance
    k: int
    params: dict                 # the arguments that produced this fit

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project raw covariates [n, p] to unit-variance scores [n, k]."""
        if X.shape[1] != len(self.cov_names):
            raise ValueError(f"expected {len(self.cov_names)} columns, got {X.shape[1]}")
        return ((X - self.mean) @ self.rotation) / self.sd


def _svd_block(X: np.ndarray, names: list[str], anchor: str,
               var_frac: float, eig_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (rotation [p,k], score SDs [k], eigenvalues [p], k) for one block."""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (S ** 2) / (X.shape[0] - 1)
    cum = np.cumsum(lam) / lam.sum()
    k = int(np.argmax(cum >= var_frac)) + 1 if (cum >= var_frac).any() else len(lam)
    k = max(1, min(k, int((lam >= eig_floor * lam[0]).sum())))
    V = Vt.T[:, :k]
    # sign anchor: loading on the reference variable is positive
    ai = names.index(anchor) if anchor in names else int(np.argmax(np.abs(V[:, 0])))
    V = V * np.where(V[ai, :] >= 0, 1.0, -1.0)
    sd = (Xc @ V).std(axis=0, ddof=1)
    return V, sd, lam, k


def fit_global(X: np.ndarray, cov_names: list[str], k: int,
               anchor: str = GLOBAL_ANCHOR) -> Reduction:
    """One rotation over every covariate, truncated to the leading `k` components."""
    k = int(min(k, X.shape[1]))
    V, sd, lam, _ = _svd_block(X, cov_names, anchor, var_frac=1.0, eig_floor=0.0)
    V, sd = V[:, :k], sd[:k]
    return Reduction(
        arm="global", cov_names=list(cov_names),
        out_names=[f"PC{i + 1}" for i in range(k)],
        mean=X.mean(axis=0), rotation=V, sd=sd,
        eigen={"global": {"lambda": lam.tolist(),
                          "cum_var": float(lam[:k].sum() / lam.sum())}},
        k=k, params={"arm": "global", "n_pca": k, "anchor": anchor})


def fit_block(X: np.ndarray, cov_names: list[str], groups: dict[str, str],
              var_frac: float = 0.90, eig_floor: float = 0.05) -> Reduction:
    """A rotation per thematic block; single-variable blocks pass through unrotated.

    The returned rotation is block-diagonal over the full covariate vector, so a
    single matmul applies it -- identical call signature to the global arm.
    """
    p = len(cov_names)
    idx = {c: i for i, c in enumerate(cov_names)}
    cols, sds, names, eig = [], [], [], {}

    for block in sorted({groups[c] for c in cov_names}):
        members = sorted(c for c in cov_names if groups[c] == block)
        prefix = BLOCK_PREFIX.get(block, block[:2])
        Xb = X[:, [idx[c] for c in members]]
        if len(members) == 1:                      # pass-through, as in the INLA doc
            V = np.ones((1, 1))
            sd = Xb.std(axis=0, ddof=1)
            lam, kb, out = np.array([1.0]), 1, members
        else:
            V, sd, lam, kb = _svd_block(Xb, members, BLOCK_ANCHOR.get(block, members[0]),
                                        var_frac, eig_floor)
            out = [f"{prefix}_PC{j + 1}" for j in range(kb)]
        full = np.zeros((p, V.shape[1]))
        full[[idx[c] for c in members], :] = V
        cols.append(full)
        sds.append(sd)
        names.extend(out)
        eig[block] = {"p_b": len(members), "k_b": int(kb), "lambda": lam.tolist(),
                      "cum_var": float(lam[:kb].sum() / lam.sum())}

    rotation = np.concatenate(cols, axis=1)
    return Reduction(
        arm="block", cov_names=list(cov_names), out_names=names,
        mean=X.mean(axis=0), rotation=rotation, sd=np.concatenate(sds),
        eigen=eig, k=rotation.shape[1],
        params={"arm": "block", "var_frac": var_frac, "eig_floor": eig_floor})


def _embed_bypass(red: Reduction, X: np.ndarray, cov_names: list[str],
                  dyn: list[str], static: list[str]) -> Reduction:
    """Widen a dynamic-only rotation to the full covariate vector, statics passed through.

    The static block becomes an identity block, so `transform` stays one matmul and
    `apply_reduction` needs no special case. Centring and SD-scaling a passed-through
    column is a per-column monotone map and so cannot change a boosted-tree fit.
    """
    idx = {c: i for i, c in enumerate(cov_names)}
    kc = red.rotation.shape[1]
    rot = np.zeros((len(cov_names), kc + len(static)))
    for j, c in enumerate(dyn):
        rot[idx[c], :kc] = red.rotation[j]
    for j, c in enumerate(static):
        rot[idx[c], kc + j] = 1.0

    si = [idx[c] for c in static]
    mean = X.mean(axis=0)
    sd_static = X[:, si].std(axis=0, ddof=1)
    sd_static[sd_static == 0] = 1.0

    return Reduction(
        arm=red.arm, cov_names=list(cov_names),
        out_names=list(red.out_names) + list(static),
        mean=mean, rotation=rot, sd=np.concatenate([red.sd, sd_static]),
        eigen=red.eigen, k=rot.shape[1],
        params=dict(red.params, static_bypass=True, n_components=kc,
                    static_names=list(static), dynamic_names=list(dyn)))


def save(red: Reduction, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "reduction.npz", mean=red.mean, rotation=red.rotation, sd=red.sd)
    (model_dir / "reduction.json").write_text(json.dumps({
        "arm": red.arm, "cov_names": red.cov_names, "out_names": red.out_names,
        "eigen": red.eigen, "k": red.k, "params": red.params}, indent=2))


def load(model_dir: Path) -> Reduction:
    model_dir = Path(model_dir)
    a = np.load(model_dir / "reduction.npz")
    meta = json.loads((model_dir / "reduction.json").read_text())
    return Reduction(arm=meta["arm"], cov_names=meta["cov_names"],
                     out_names=meta["out_names"], mean=a["mean"], rotation=a["rotation"],
                     sd=a["sd"], eigen=meta["eigen"], k=meta["k"], params=meta["params"])


def predictor_groups() -> dict[str, str]:
    """column -> thematic group, from the panel's feature metadata."""
    meta = json.loads((DATA_DIR / "feature_metadata.json").read_text())
    return {k: v["group"] for k, v in meta.items() if v.get("role") == "predictor"}


def train_panel_matrix(cov_names: list[str]) -> np.ndarray:
    """The distinct train-split panel rows, [n_train_months * n_counties, p].

    This -- not the assembled design matrix -- is what the rotation is fitted on.
    """
    df = pd.read_parquet(DATA_DIR / "train.parquet", columns=list(cov_names))
    return df.to_numpy(dtype=np.float64)


def fit_from_panel(cov_names: list[str], arm: str, n_pca: int | None = None,
                   var_frac: float = 0.90, eig_floor: float = 0.05,
                   static_bypass: bool = False) -> Reduction:
    """Fit the requested arm on the train panel. `cov_names` fixes the column order.

    Under `static_bypass` the rotation covers only the 39 dynamic predictors and the 12
    near-constant ones of `model.data.STATIC_COLS` are appended raw -- the same split as
    the GNN's `--static-bypass` and `inla_glm_comparison.qmd`'s blocks_dyn, which is what
    puts this arm's component count on the same footing as theirs.

    `model.data.TAIL_TRANSFORMS` is deliberately *not* ported. It exists because a bypassed
    39-sigma `pop_density` reaches the GCN's linear input undiluted; trees split on
    thresholds, so any strictly monotone per-column map leaves the fit identical.
    """
    X = train_panel_matrix(cov_names)
    fit_cols, fit_X = cov_names, X
    static: list[str] = []
    if static_bypass:
        static = [c for c in cov_names if c in STATIC_COLS]
        fit_cols = [c for c in cov_names if c not in STATIC_COLS]
        fit_X = X[:, [cov_names.index(c) for c in fit_cols]]

    if arm == "global":
        if n_pca is None:
            raise ValueError("arm 'global' requires n_pca")
        red = fit_global(fit_X, fit_cols, n_pca)
    elif arm == "block":
        red = fit_block(fit_X, fit_cols, predictor_groups(), var_frac, eig_floor)
    else:
        raise ValueError(f"unknown reduction arm: {arm}")

    return _embed_bypass(red, X, cov_names, fit_cols, static) if static_bypass else red
