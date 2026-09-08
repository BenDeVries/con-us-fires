"""Per-arm hyperparameter search for the boosted ZABeta baseline.

The reduction ladder (`model/xgb/select_reduction.py`) compares representations under
fixed hyperparameters. That is the right way to measure a representation, but it leaves
the comparison between arms confounded with the fact that the defaults were chosen on
the raw design. This module re-tunes inside each arm so the final comparison is
best-versus-best.

    conda run -n fire-xgb python -m model.xgb.tune --arm raw --trials 25 --study xgb_raw_v1
    conda run -n fire-xgb python -m model.xgb.tune --arm global --n-pca 31 --trials 25 --study xgb_pca_v1
    conda run -n fire-xgb python -m model.xgb.tune --study xgb_pca_v1 --noise-floor --seeds 0 1 2
    conda run -n fire-xgb python -m model.xgb.tune --arm none --climatology --clim-oof \
        --lookback 48 --tune-lags --max-spatial 3 --max-temporal 47 --max-nbr-pcs 8 \
        --trials 40 --study xgb_spacetime_v1

Two protocol rules carried over from `model/tune.py`:

  * **Validation-blind.** The objective is test-split ZIB NLL and the validation split is
    never assembled into a DMatrix here. Open it once, at the end, for the chosen config.
  * **The arm is fixed per study**, exactly as `--link` / `--re-type` are for the neural
    tuner, and resuming a study under a different arm is rejected. Any change to a
    search-space distribution likewise requires a new `--study` name, otherwise the
    parameter drops out of Optuna's intersection search space and multivariate TPE
    silently degrades.

The seed noise floor is not optional here. `subsample` and `colsample_bytree` make each
fit seed-dependent, so an arm gap smaller than the between-seed SD is not evidence of
anything -- `--noise-floor` measures it directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from ..data import STATIC_COLS
from .common import OUT_DIR, make_dmatrix, params_from_predt
from .features import MAX_HARMONICS, apply_reduction, build_dataset, select_columns
from . import reduction as red_mod
from .train import DEFAULT_PARAMS, fit
from .select_reduction import row_nll

TUNE_DIR = OUT_DIR / "tune"

# Keys absent from any arm_spec written before the space-time block existed, plus
# static_only, absent from any spec written before the forecast track existed, plus
# tune_harmonics, absent from any spec written before the seasonal order was searched.
SPEC_DEFAULTS_OFF = {"climatology": False, "clim_oof": False, "tune_lags": False,
                     "max_spatial": 0, "max_temporal": 0, "n_nbr_pcs": 0,
                     "static_only": False, "tune_harmonics": False, "max_harmonics": 1,
                     "forecast_safe": False}


def suggest(trial) -> dict:
    return {
        "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 400.0, log=True),
        "lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.25, 4.0, log=True),
    }


def suggest_lags(trial, spec: dict) -> dict:
    """The design's space-time budget, searched alongside the booster params.

    One `n_spatial` governs both the fire-history shells and the neighbour-covariate
    block, mirroring the GNN's single `gcn_layers` over its encoder and decoder graph
    convolutions. Ranges are capped by the study's `arm_spec`, which is what keeps the
    design inside the GNN's information set.
    """
    out = {}
    if spec["tune_lags"]:
        out["n_spatial"] = trial.suggest_int("n_spatial", 0, spec["max_spatial"])
        out["n_temporal"] = trial.suggest_int("n_temporal", 0, spec["max_temporal"])
        out["n_nbr_pcs"] = trial.suggest_int("n_nbr_pcs", 0, spec["n_nbr_pcs"])
    if spec["tune_harmonics"]:
        out["n_harmonics"] = trial.suggest_int("n_harmonics", 1, spec["max_harmonics"])
    return out


def build_arm(arm: str, n_pca: int | None, var_frac: float, eig_floor: float,
              lookback: int, horizon: int, max_origins: int | None,
              climatology: bool = False, clim_oof: bool = False,
              max_spatial: int = 0, max_temporal: int = 0, n_nbr_pcs: int = 0,
              static_only: bool = False, n_harmonics: int = 1,
              forecast_safe: bool = False):
    if forecast_safe and arm != "none":
        raise ValueError("forecast_safe requires arm='none'")
    if static_only and arm != "none":
        raise SystemExit(f"--reduction arm '{arm}' rotates the target-month covariate block, "
                         f"but --static-only leaves only {len(STATIC_COLS)} near-constant "
                         f"columns there; that rotation is degenerate. Use --arm none.")
    # Validation is never scored here (model.xgb.tune is validation-blind), so it is not
    # assembled either -- at the maximal space-time width it would cost ~4 GB for nothing.
    ds = build_dataset(lookback, horizon, max_origins=max_origins,
                       climatology=climatology, clim_oof=clim_oof,
                       max_spatial=max_spatial, max_temporal=max_temporal,
                       n_nbr_pcs=n_nbr_pcs, static_only=static_only,
                       n_harmonics=n_harmonics, want_splits=("train", "test"),
                       forecast_safe=forecast_safe)
    if arm == "none":
        return ds, None
    red = red_mod.fit_from_panel(ds.raw_cov_names, arm, n_pca=n_pca,
                                 var_frac=var_frac, eig_floor=eig_floor)
    return apply_reduction(ds, red), red


def score(ds, params: dict, rounds: int, esr: int, return_model: bool = False):
    xgblss, best_it, secs = fit(ds, params, num_boost_round=rounds,
                                early_stopping_rounds=esr, verbose_eval=False)
    dtest = make_dmatrix(ds.splits["test"]["X"])
    p = params_from_predt(xgblss.predict(dtest, pred_type="parameters", n_samples=1))
    y = np.asarray(ds.splits["test"]["y"], dtype=np.float64)
    result = (float(row_nll(p, y).mean()), best_it, secs)
    return (*result, xgblss) if return_model else result


def completed_finite_trials(study):
    """Failed/pruned/non-finite trials can never become a published representative."""
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and np.isfinite(t.value)]


def save_progress(study, out_dir: Path):
    done = completed_finite_trials(study)
    if not done:
        return
    board = sorted(({"trial": t.number, "test_nll": t.value,
                     "best_iteration": t.user_attrs.get("best_iteration"),
                     "n_columns": t.user_attrs.get("n_columns"),
                     "seconds": t.user_attrs.get("seconds"), "params": t.params}
                    for t in done), key=lambda d: d["test_nll"])
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2))
    best = dict(board[0]["params"])
    # The design's lag budget is not a booster param -- nest it so `train.py --params`
    # can route it to build_dataset instead of handing xgboost an unknown key.
    design = {k: best.pop(k)
              for k in ("n_spatial", "n_temporal", "n_nbr_pcs", "n_harmonics")
              if k in best}
    params = dict(DEFAULT_PARAMS)
    params.update({k if k != "reg_lambda" else "lambda": v for k, v in best.items()})
    if design:
        params["design"] = design
    if study.user_attrs.get("arm_spec", {}).get("forecast_safe", False):
        params["forecast_safe"] = True
    (out_dir / "best_params.json").write_text(json.dumps(params, indent=2))


def noise_floor(ds, params: dict, seeds: list[int], rounds: int, esr: int) -> dict:
    vals = []
    for s in seeds:
        nll, best_it, secs = score(ds, dict(params, seed=s), rounds, esr)
        print(f"  seed {s}: test NLL {nll:+.5f}  best_it={best_it}  ({secs/60:.1f} min)",
              flush=True)
        vals.append(nll)
    v = np.array(vals)
    return {"seeds": seeds, "test_nll": vals, "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "range": float(v.max() - v.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="study name; also the output subdirectory")
    ap.add_argument("--arm", default=None, choices=["none", "global", "block"],
                    help="reduction arm, fixed for the life of the study")
    ap.add_argument("--n-pca", type=int, default=None)
    ap.add_argument("--var-frac", type=float, default=0.90)
    ap.add_argument("--eig-floor", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=25,
                    help="target TOTAL completed trials in the study, not trials to add, "
                         "so re-running an interrupted study with the same command resumes "
                         "it rather than overshooting")
    ap.add_argument("--startup", type=int, default=8, help="random trials before TPE")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--climatology", action="store_true",
                    help="append the county x calendar-month ZIB triple (see train.py)")
    ap.add_argument("--clim-oof", action="store_true",
                    help="with --climatology, give train rows the leave-one-year-out fold")
    ap.add_argument("--static-only", action="store_true",
                    help="legacy near-static subset in data.STATIC_COLS, including "
                         "annual/epoch fields; not an origin-safe forecast. Use "
                         "--forecast-safe for new forecasts. Requires --arm none and "
                         "its own --study name")
    ap.add_argument("--tune-lags", action="store_true",
                    help="search the design's space-time budget alongside the booster "
                         "params. Fixed for the life of the study, like the GNN's "
                         "--tune-lookback: it changes the search space, so it needs its "
                         "own --study name (see CLAUDE.md)")
    ap.add_argument("--max-spatial", type=int, default=0,
                    help="upper bound on exclusive neighbour shells; cap it at the GNN's "
                         "gcn_layers range so xgb stays inside the GNN's information set")
    ap.add_argument("--max-temporal", type=int, default=0,
                    help="upper bound on origin-anchored lags; cap it at the GNN's "
                         "lookback range. Requires --lookback > this")
    ap.add_argument("--max-nbr-pcs", type=int, default=0,
                    help="upper bound on neighbour-averaged target-month covariate PCs")
    ap.add_argument("--tune-harmonics", action="store_true",
                    help=f"search the seasonal order n_harmonics over [1, --max-harmonics], "
                         f"matching the GNN's --tune-harmonics. The design is built once at "
                         f"the ceiling and each trial slices down, so this costs a column "
                         f"slice. Changes the search space: needs its own --study name")
    ap.add_argument("--max-harmonics", type=int, default=MAX_HARMONICS,
                    help=f"ceiling for --tune-harmonics; {MAX_HARMONICS} is Nyquist for "
                         f"monthly data")
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--forecast-safe", action="store_true")
    ap.add_argument("--save-best-model-dir", default=None,
                    help="persist each finite completed best model, avoiding a redundant refit")
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler seed")
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--noise-floor", action="store_true",
                    help="refit best_params.json at several seeds instead of searching")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    torch.set_num_threads(args.nthread)

    if args.forecast_safe and (args.arm not in (None, "none") or args.static_only
                              or args.climatology or args.clim_oof or args.max_nbr_pcs
                              or args.lookback != 48 or args.horizon != 12):
        raise SystemExit("--forecast-safe requires arm none, lookback48/horizon12, "
                         "no static-only, climatology or neighbour PCs")

    # Checked before create_study: study.db is shared across every xgb study, so a flag
    # combination that will fail must not leave an empty study behind first.
    if args.static_only and args.arm not in (None, "none"):
        raise SystemExit(f"--arm {args.arm} rotates the target-month covariate block, but "
                         f"--static-only leaves only {len(STATIC_COLS)} near-constant columns "
                         f"there; that rotation is degenerate. Use --arm none.")

    out_dir = TUNE_DIR / args.study
    out_dir.mkdir(parents=True, exist_ok=True)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{TUNE_DIR / 'study.db'}"

    study = optuna.create_study(
        direction="minimize", study_name=args.study, storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup,
                                           multivariate=True, group=True))

    # The arm is a study invariant, like --link for the neural tuner: trials scored on
    # different designs are not comparable, and TPE would model them as one surface.
    spec = {"arm": args.arm, "n_pca": args.n_pca, "var_frac": args.var_frac,
            "eig_floor": args.eig_floor, "lookback": args.lookback,
            "horizon": args.horizon, "climatology": args.climatology,
            "clim_oof": args.clim_oof, "tune_lags": args.tune_lags,
            "max_spatial": args.max_spatial, "max_temporal": args.max_temporal,
            "n_nbr_pcs": args.max_nbr_pcs, "static_only": args.static_only,
            "tune_harmonics": args.tune_harmonics,
            "forecast_safe": args.forecast_safe,
            "max_harmonics": args.max_harmonics if args.tune_harmonics else 1}
    stored = study.user_attrs.get("arm_spec")
    if stored is None:
        if args.arm is None:
            raise SystemExit("--arm is required when creating a study")
        study.set_user_attr("arm_spec", spec)
        stored = spec
    else:
        # Studies created before the space-time block carry a shorter spec. The new keys
        # are all off-by-default, so filling them keeps those studies comparing equal.
        stored = {**SPEC_DEFAULTS_OFF, **stored}
        if stored["forecast_safe"] != args.forecast_safe:
            raise SystemExit("study forecast_safe does not match --forecast-safe")
        if args.arm is not None and stored != spec:
            raise SystemExit(f"study '{args.study}' was created with arm_spec {stored}, "
                             f"but this invocation passes {spec}. Use a new --study name.")

    t0 = time.time()
    (out_dir / "arm_spec.json").write_text(json.dumps(stored, indent=2))
    ds, red = build_arm(stored["arm"], stored["n_pca"], stored["var_frac"],
                        stored["eig_floor"], stored["lookback"], stored["horizon"],
                        args.max_origins, climatology=stored["climatology"],
                        clim_oof=stored["clim_oof"], max_spatial=stored["max_spatial"],
                        max_temporal=stored["max_temporal"],
                        n_nbr_pcs=stored["n_nbr_pcs"],
                        static_only=stored["static_only"],
                        n_harmonics=stored["max_harmonics"],
                        forecast_safe=stored["forecast_safe"])
    print(f"arm={stored['arm']} k={red.k if red else 'raw'} "
          f"features={len(ds.num_features)} "
          f"train rows={len(ds.splits['train']['y']):,} "
          f"(assembled in {time.time()-t0:.0f}s)", flush=True)
    if red is not None:
        red_mod.save(red, out_dir)

    if args.noise_floor:
        pfile = out_dir / "best_params.json"
        if not pfile.exists():
            raise SystemExit(f"{pfile} not found; run the search first")
        params = dict(json.loads(pfile.read_text()), device=args.device)
        params.pop("forecast_safe", None)
        design = params.pop("design", None)
        nf_ds = select_columns(ds, **design) if design else ds
        nf = noise_floor(nf_ds, params, args.seeds, args.num_boost_round,
                         args.early_stopping_rounds)
        (out_dir / "noise_floor.json").write_text(json.dumps(nf, indent=2))
        print(f"\nseed noise floor: mean {nf['mean']:+.5f}  SD {nf['sd']:.5f}  "
              f"range {nf['range']:.5f}")
        print("a between-arm gap smaller than this SD is not evidence of a difference")
        return

    candidate = {}

    def objective(trial):
        params = dict(DEFAULT_PARAMS, device=args.device, seed=args.seed, nthread=args.nthread)
        params.update(suggest(trial))
        # The maximal design is assembled once; a trial takes a column-sliced view of it.
        searched = stored["tune_lags"] or stored["tune_harmonics"]
        design = suggest_lags(trial, stored) if searched else {}
        tds = select_columns(ds, **design) if searched else ds
        nll, best_it, secs, fitted = score(tds, params, args.num_boost_round,
                                         args.early_stopping_rounds, return_model=True)
        if not np.isfinite(nll):
            raise optuna.TrialPruned("non-finite test NLL")
        candidate.clear()
        candidate.update(model=fitted, design=design, params=params, ds=tds,
                         best_iteration=best_it, seconds=secs, trial=trial.number)
        trial.set_user_attr("best_iteration", best_it)
        trial.set_user_attr("n_columns", len(tds.num_features))
        trial.set_user_attr("seconds", round(secs, 1))
        print(f"  trial {trial.number}: test NLL {nll:+.5f}  best_it={best_it}  "
              f"cols={len(tds.num_features)}  ({secs/60:.1f} min)", flush=True)
        return nll

    def checkpoint(study, trial):
        save_progress(study, out_dir)
        done = completed_finite_trials(study)
        if not args.save_best_model_dir or not done or not candidate:
            candidate.clear()
            return
        best = min(done, key=lambda t: t.value)
        if best.number == trial.number == candidate["trial"]:
            dest = Path(args.save_best_model_dir)
            dest.mkdir(parents=True, exist_ok=True)
            candidate["model"].save_model(str(dest / "model.pkl"))
            design, tds = candidate["design"], candidate["ds"]
            config = {
                "lookback": stored["lookback"], "horizon": stored["horizon"],
                "forecast_safe": stored["forecast_safe"], "static_only": stored["static_only"],
                "climatology": stored["climatology"], "clim_oof": stored["clim_oof"],
                "reduction": stored["arm"], "raw_cov_names": tds.raw_cov_names,
                "max_spatial": design.get("n_spatial", stored["max_spatial"]),
                "max_temporal": design.get("n_temporal", stored["max_temporal"]),
                "n_nbr_pcs": design.get("n_nbr_pcs", stored["n_nbr_pcs"]),
                "n_harmonics": design.get("n_harmonics", stored["max_harmonics"]),
                "feature_names": tds.feature_names, "num_features": tds.num_features,
                "cat_feature": tds.cat_feature, "lc_categories": tds.lc_categories,
                "params": candidate["params"], "best_iteration": candidate["best_iteration"],
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
                "stabilization": "L2", "train_seconds": candidate["seconds"],
                "selection_split": "test", "study": args.study, "trial": best.number,
                "test_nll": best.value, "arm_spec": stored,
            }
            from .zabeta_hurdle import spec as hurdle_spec
            from .zabeta_hurdle import C0_CAP
            config["distribution"] = hurdle_spec()
            config["c0_cap"] = C0_CAP
            (dest / "config.json").write_text(json.dumps(config, indent=2))
            if red is not None:
                red_mod.save(red, dest)
        candidate.clear()

    n_done = len(completed_finite_trials(study))
    todo = max(0, args.trials - n_done)
    print(f"{n_done} completed trials in study; running {todo} more "
          f"to reach {args.trials}", flush=True)
    if todo:
        study.optimize(objective, n_trials=todo, gc_after_trial=True,
                       callbacks=[checkpoint])

    save_progress(study, out_dir)
    done = completed_finite_trials(study)
    if not done:
        print("no completed trials")
        return
    best = min(done, key=lambda t: t.value)
    print(f"\nstudy '{args.study}': {len(study.trials)} trials, {len(done)} complete")
    print(f"best trial {best.number}: test NLL {best.value:+.5f}")
    print(json.dumps(best.params, indent=2))
    print(f"-> {out_dir}/best_params.json")


if __name__ == "__main__":
    main()
