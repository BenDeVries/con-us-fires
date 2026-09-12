"""SHAP feature-importance report for the trained ZABeta baseline.

Uses XGBoost's native TreeSHAP (pred_contribs) on the multi-output booster --
no shap package required (its tree parser cannot handle categorical splits).
Contributions come back per distribution parameter in param_dict order
(concentration1, concentration0, gate). The gate margin is logit(P(y=0)), so
gate contributions are sign-flipped into an "occurrence" head where positive
SHAP pushes fire probability up. The Beta head is reported per parameter plus
a combined mean(|c1|+|c0|) ranking.

Base EE predictors (role=="predictor" in feature_metadata.json) are the
selection candidates; engineered features (AR lags, rolling means, neighbour
aggregates, county priors, month encoding, horizon) are ranked separately --
they derive from the target and dominate raw importance by construction.

    conda run -n fire-xgb python -m model.xgb.importance --split validation

Artifacts: output/xgb/importance.{csv,json} + output/xgb/figures/importance/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgboostlss.model import XGBoostLSS

from .common import OUT_DIR, make_dmatrix
from .features import CAT_COL, DATA_DIR, _numeric_cov_cols, build_dataset

HEADS = ("occurrence", "concentration1", "concentration0", "beta_combined")
ENGINEERED_GROUPS = ("history", "neighbor", "county_prior",
                     "seasonal_encoding", "horizon", "landcover_cat")


def _feature_groups(feature_names: list[str]) -> pd.DataFrame:
    """Map every model feature to (group, is_base_predictor)."""
    meta = json.loads((DATA_DIR / "feature_metadata.json").read_text())
    base = set(_numeric_cov_cols())
    rows = []
    for f in feature_names:
        if f in base:
            rows.append((f, meta[f].get("group", "unknown"), True))
        elif f.startswith("month_sin") or f.startswith("month_cos"):
            rows.append((f, "seasonal_encoding", False))
        elif f.startswith("nb_"):
            rows.append((f, "neighbor", False))
        elif f.startswith("county_"):
            rows.append((f, "county_prior", False))
        elif f == "horizon":
            rows.append((f, "horizon", False))
        elif f == CAT_COL:
            rows.append((f, "landcover_cat", False))
        else:
            rows.append((f, "history", False))
    return pd.DataFrame(rows, columns=["feature", "group", "is_base_predictor"])


def _stratified_sample(y: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    """Sample n_rows indices preserving the natural positive fraction."""
    n = len(y)
    if n_rows >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(~(y > 0))
    n_pos = min(max(int(round(n_rows * len(pos) / n)), 1), len(pos))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_rows - n_pos, replace=False)])
    idx.sort()
    return idx


def _rank(df: pd.DataFrame, mask=None) -> pd.Series:
    s = df["mean_abs_shap"] if mask is None else df.loc[mask, "mean_abs_shap"]
    return s.rank(ascending=False, method="min").astype("Int64")


def _plot_head(head_df: pd.DataFrame, head: str, top_n: int,
               group_colors: dict, path: Path):
    import matplotlib.pyplot as plt

    base = head_df[head_df.is_base_predictor].nlargest(top_n, "mean_abs_shap")
    eng = head_df[~head_df.is_base_predictor].nlargest(10, "mean_abs_shap")
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(9, 0.3 * (len(base) + len(eng)) + 2.2),
        gridspec_kw={"height_ratios": [max(len(base), 1), max(len(eng), 1)]})

    for ax, sub, title, grey in (
            (ax0, base, f"{head}: top {len(base)} base predictors "
                        "(selection candidates)", False),
            (ax1, eng, "engineered features (context, not candidates)", True)):
        sub = sub.iloc[::-1]
        colors = "0.65" if grey else [group_colors[g] for g in sub.group]
        ax.barh(sub.feature, sub.mean_abs_shap, color=colors)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.margins(y=0.02)
    ax1.set_xlabel("mean |SHAP| (margin space)", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors[g])
               for g in sorted(base.group.unique())]
    ax0.legend(handles, sorted(base.group.unique()), fontsize=7,
               loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_groups(head_df: pd.DataFrame, head: str, group_colors: dict, path: Path):
    import matplotlib.pyplot as plt

    g = (head_df.groupby("group", as_index=False)
         .agg(total=("mean_abs_shap", "sum"),
              is_base=("is_base_predictor", "any"))
         .sort_values("total"))
    colors = ["0.65" if not b else group_colors[gr]
              for gr, b in zip(g.group, g.is_base)]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(g) + 1.5))
    ax.barh(g.group, g.total, color=colors)
    ax.set_title(f"{head}: total mean |SHAP| per feature group "
                 "(grey = engineered)", fontsize=10, loc="left")
    ax.set_xlabel("sum of mean |SHAP|", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    choices=["train", "test", "validation"])
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-dir", default=None,
                    help="directory holding model.pkl + config.json; "
                         "artifacts written here too (defaults to output/xgb)")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else OUT_DIR
    cfg = json.loads((model_dir / "config.json").read_text())
    xgblss = XGBoostLSS.load_model(str(model_dir / "model.pkl"))
    param_order = list(xgblss.dist.param_dict.keys())
    gate_idx = param_order.index("gate")

    ds = build_dataset(cfg["lookback"], cfg["horizon"])
    if args.split not in ds.splits:
        raise SystemExit(f"split '{args.split}' has no forecast windows")
    d = ds.splits[args.split]
    assert list(d["X"].columns) == cfg["feature_names"], \
        "feature columns differ from training"
    feature_names = cfg["feature_names"]
    F = len(feature_names)

    idx = _stratified_sample(d["y"], args.sample_rows, args.seed)
    X = d["X"].iloc[idx]
    print(f"split={args.split} rows={len(d['y']):,} sampled={len(idx):,} "
          f"pos_frac={(d['y'][idx] > 0).mean():.4f} "
          f"(population {(d['y'] > 0).mean():.4f})")

    dmat = make_dmatrix(X)
    print("computing TreeSHAP contributions...")
    contrib = xgblss.booster.predict(dmat, pred_contribs=True)  # (n, P, F+1)
    assert contrib.ndim == 3 and contrib.shape[1] == len(param_order) \
        and contrib.shape[2] == F + 1, f"unexpected contrib shape {contrib.shape}"

    margin = xgblss.booster.predict(dmat, output_margin=True)
    add_err = float(np.abs(contrib.sum(axis=-1) - margin).max())
    print(f"additivity check: max |sum(contrib) - margin| = {add_err:.2e}")
    assert add_err < 1e-4, "contributions do not sum to the margin"

    contrib = contrib[:, :, :F]  # drop bias column
    heads = {
        "occurrence": -contrib[:, gate_idx, :],  # positive = more fire
        "concentration1": contrib[:, param_order.index("concentration1"), :],
        "concentration0": contrib[:, param_order.index("concentration0"), :],
    }

    groups = _feature_groups(feature_names)
    frames, signed_cols = [], []
    for head in HEADS:
        if head == "beta_combined":
            mean_abs = (np.abs(heads["concentration1"])
                        + np.abs(heads["concentration0"])).mean(axis=0)
            signed_cols.append(np.full(F, np.nan))
        else:
            mean_abs = np.abs(heads[head]).mean(axis=0)
            signed_cols.append(heads[head].mean(axis=0))
        hdf = groups.copy()
        hdf["head"] = head
        hdf["mean_abs_shap"] = mean_abs
        hdf["rank_within_head"] = _rank(hdf)
        hdf["rank_base_only"] = _rank(hdf, hdf.is_base_predictor)
        frames.append(hdf)
    table = pd.concat(frames, ignore_index=True)
    table.insert(5, "mean_shap", np.concatenate(signed_cols))

    csv_path = model_dir / f"importance_{args.split}.csv"
    table.to_csv(csv_path, index=False)

    top_base = {
        head: table[table["head"].eq(head) & table.is_base_predictor]
        .nsmallest(args.top_n, "rank_base_only")["feature"].tolist()
        for head in HEADS
    }
    meta_out = {
        "split": args.split, "sample_rows": int(len(idx)), "seed": args.seed,
        "population_rows": int(len(d["y"])),
        "param_order": param_order, "best_iteration": cfg.get("best_iteration"),
        "additivity_max_err": add_err,
        f"top{args.top_n}_base_per_head": top_base,
    }
    json_path = model_dir / f"importance_{args.split}.json"
    json_path.write_text(json.dumps(meta_out, indent=2))

    for head in ("occurrence", "beta_combined"):
        sub = table[table["head"].eq(head) & table.is_base_predictor]
        print(f"\ntop 10 base predictors [{head}]:")
        for _, r in sub.nsmallest(10, "rank_base_only").iterrows():
            print(f"  {r['rank_base_only']:>3}. {r.feature:<18} "
                  f"({r.group})  mean|SHAP|={r.mean_abs_shap:.4f}")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")

        fig_dir = model_dir / "figures" / "importance"
        fig_dir.mkdir(parents=True, exist_ok=True)
        base_groups = sorted(groups.loc[groups.is_base_predictor, "group"].unique())
        cmap = matplotlib.colormaps["tab10"]
        palette = [cmap(i) for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]  # grey = engineered
        group_colors = {g: palette[i % len(palette)]
                        for i, g in enumerate(base_groups)}

        for head in ("occurrence", "beta_combined"):
            hdf = table[table["head"].eq(head)]
            tag = "beta" if head == "beta_combined" else head
            _plot_head(hdf, head, args.top_n, group_colors,
                       fig_dir / f"{args.split}_{tag}_top{args.top_n}.png")
            _plot_groups(hdf, head, group_colors,
                         fig_dir / f"{args.split}_groups_{tag}.png")
        print(f"\nfigures -> {fig_dir}")

    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
