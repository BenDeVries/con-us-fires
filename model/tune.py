"""Bayesian (TPE) hyperparameter optimization for the GCN-LSTM forecaster.

Minimizes the test-split NLL of the GCN-LSTM zero-inflated-Beta model -- the same metric
`train.py` uses for early stopping and checkpoint selection, so the tuned objective and the
saved checkpoint agree. Balanced accuracy of the occurrence gate is recorded per trial for
reporting only; it does not drive any decision. The panel is built once and reused across
trials; each trial trains to early stopping (with a capped epoch budget) and reports its
running test NLL so Optuna's median pruner can abort clearly-bad configs early.
The study is persisted to a sqlite db, so the run is resumable -- rerun with more `--trials`
to append to the same study, or inspect it later.

The tuner is **validation-blind**: it passes eval_validation=False, so no trial ever evaluates
the held-out validation split and no val_* metric reaches the leaderboard. Open validation once,
at the end, for the single chosen configuration.

Run on the GPU env (fire-nn):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run -n fire-nn python -m model.tune --trials 30
    conda run -n fire-nn python -m model.tune --trials 20         # append more to the study

The gate/Beta-mean link is fixed per study; tune cloglog in a separate study and compare:

    conda run -n fire-nn python -m model.tune --trials 30 --link cloglog --study zib_nll_cloglog

--max-county-embed adds the learned per-county embedding width to the search space — the
model's only carrier of county identity. Because 0 is in range, one such study contains its
own null arm, so the embedding's worth is read off the study directly. Note that comparisons
across d are unpaired: the embedding changes the GCN's input width and so shifts the init RNG
stream, meaning seed cannot be used as a blocking factor here.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math

import optuna
import torch

from .config import Config, CKPT_DIR
from .data import STATIC_COLS, numeric_predictor_names, rebuild_origins
from .train import train_model, load_panel

TUNE_DIR = CKPT_DIR / "tune"

# Nyquist for monthly data: k=6 saturates the space of functions of calendar month, so no
# order above it is identifiable. Used both to build the shared panel and to cap the search,
# so the two cannot drift apart.
MAX_HARMONICS = 6


def static_only_drops():
    """Legacy near-static subset, retained to reproduce historical checkpoints.

    Several retained annual/epoch predictors and the land-cover category are read at the
    target month. This subset is not an origin-available forecast design. Use the explicit
    forecast_safe allowlist for a retrospective forecast using only origin information.
    """
    return [c for c in numeric_predictor_names() if c not in STATIC_COLS]


def suggest_cfg(trial, base_epochs, base_patience, link,
                tune_lookback, max_pcs=None, max_county_embed=0, max_harmonics=0,
                static_bypass=False, no_covariates=False, static_only=False,
                forecast_safe=False, batch_size=None):
    """Sample one hyperparameter configuration.

    link, static_bypass, no_covariates and static_only are fixed per study.
    max_county_embed > 0 adds the county embedding width to the search space (0 is in range,
    so the arm contains its own null). max_harmonics > 0 adds the seasonal order, sliced per
    trial from a panel built at that maximum."""
    cfg = Config()
    cfg.link = link
    if batch_size is not None:
        cfg.batch_size = batch_size
    cfg.forecast_safe = forecast_safe
    if forecast_safe:
        cfg.lookback = 48
        cfg.drop_features = []
        cfg.lc_embed_dim = 0
        cfg.n_samples = 0
    cfg.static_bypass = static_bypass
    # Recorded per trial as well as on the shared panel, so config.json alone is enough for
    # model.predict to rebuild the covariate-free panel.
    if no_covariates:
        cfg.drop_features = numeric_predictor_names()
    if static_only:
        cfg.drop_features = static_only_drops()
    cfg.max_epochs = base_epochs
    cfg.patience = base_patience
    cfg.lr = trial.suggest_float("lr", 3e-4, 6e-3, log=True)
    # 1e-7 under Adam is numerically indistinguishable from 0, so the old floor wasted the
    # bottom decade of this dimension rather than exploring it.
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg.gcn_hidden = trial.suggest_categorical("gcn_hidden", [16, 32, 64] if forecast_safe
                                              else [8, 16, 32, 64, 128])
    cfg.gcn_layers = trial.suggest_int("gcn_layers", 1, 2 if forecast_safe else 3)
    cfg.gcn_dropout = trial.suggest_float("gcn_dropout", 0.0, 0.3)
    # 384 rather than 512: the cuDNN workspace for the wider LSTM has to fit the 12GB card.
    cfg.lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128] if forecast_safe
                                               else [64, 128, 256, 384])
    cfg.lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
    if not forecast_safe:
        cfg.lc_embed_dim = trial.suggest_categorical("lc_embed_dim", [4, 8, 16])
    cfg.head_hidden = trial.suggest_categorical("head_hidden", [0, 32, 64] if forecast_safe
                                               else [0, 32, 64, 128])
    # Searched as an int rather than a categorical so TPE can exploit that the axis is
    # ordered — the cost is linear in d (3108*d params) and so, plausibly, is the benefit.
    if max_county_embed > 0:
        cfg.county_embed_dim = trial.suggest_int("county_embed_dim", 0, max_county_embed)

    if tune_lookback:
        cfg.lookback = trial.suggest_categorical("lookback", [12, 24, 36, 48])
    if max_pcs is not None:
        cfg.n_pca = trial.suggest_int("n_pca", 3, max_pcs)
        cfg.drop_features = []
        cfg.n_harmonics = MAX_HARMONICS
    # Floored at 1, not 0: dropping the seasonal block entirely is a structural question about
    # whether a fixed climatological baseline earns its place, and single-seed TPE cannot
    # resolve a difference that small. A k=0 rung belongs in the multi-seed parsimony ladder
    # instead (model/parsimony.py has no such rung yet).
    if max_harmonics:
        cfg.n_harmonics = trial.suggest_int("n_harmonics", 1, max_harmonics)
    cfg.teacher_forcing = (False if forecast_safe else
                          trial.suggest_categorical("teacher_forcing", [True, False]))
    if cfg.teacher_forcing:
        cfg.tf_ratio_start = trial.suggest_float("tf_ratio_start", 0.3, 1.0)
        cfg.tf_ratio_end = trial.suggest_float("tf_ratio_end", 0.0, 0.3)
    cfg.validate_forecast_safe()
    return cfg


def _fmt(params):
    return " ".join(f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in params.items())


def make_objective(panel, base_epochs, base_patience, link,
                   tune_lookback, study_name, max_pcs=None, max_county_embed=0,
                   max_harmonics=0, static_bypass=False, no_covariates=False,
                   static_only=False, forecast_safe=False, batch_size=None,
                   pruning_policy="median_test_nll"):
    def objective(trial):
        cfg = suggest_cfg(trial, base_epochs, base_patience, link,
                          tune_lookback, max_pcs=max_pcs,
                          max_county_embed=max_county_embed,
                          max_harmonics=max_harmonics,
                          static_bypass=static_bypass,
                          no_covariates=no_covariates,
                          static_only=static_only, forecast_safe=forecast_safe,
                          batch_size=batch_size)
        trial.set_user_attr("config", dataclasses.asdict(cfg))
        trial.set_user_attr("pruning_policy", pruning_policy)
        # WindowDataset slices the encoder with cfg.lookback, so a trial that moves it needs
        # matching origins. Replace on a copy -- the panel is shared across trials.
        panel_t = dataclasses.replace(panel, split_origins=rebuild_origins(
            panel.dates, panel.split_date_ranges, cfg.lookback, cfg.horizon)
        ) if tune_lookback else panel
        # Namespaced per study: trial numbers restart at 0 in every study, so a shared
        # TUNE_DIR/trial_NNN made a later study silently overwrite an earlier one's
        # checkpoints. That destroyed all of logit_embed_harm_v2 on 2026-08-12.
        ckpt = TUNE_DIR / study_name / f"trial_{trial.number:03d}"
        print(f"\n[trial {trial.number}] {_fmt(trial.params)}", flush=True)
        best_nll = [float("inf")]
        best_ba = [0.5]  # max BA seen across epochs; reported only, not optimized

        def report_cb(epoch, metrics):
            nll = metrics["nll"]
            ba = metrics.get("balanced_accuracy", 0.5)
            best_nll[0] = min(best_nll[0], nll)
            best_ba[0] = max(best_ba[0], ba)
            print(f"  [t{trial.number} ep{epoch:3d}] test_nll {nll:+.5f} "
                  f"balanced_acc {ba:.4f} (best_nll {best_nll[0]:+.5f} best_ba {best_ba[0]:.4f})", flush=True)
            trial.report(nll, epoch)
            if trial.should_prune():
                trial.set_user_attr("prune_reason", "median_test_nll")
                trial.set_user_attr("pruned_epoch", epoch)
                print(f"  [t{trial.number}] PRUNED at epoch {epoch}", flush=True)
                raise optuna.TrialPruned()

        try:
            test_nll, test_m, _, _ = train_model(
                cfg, panel_t, ckpt, report_cb=report_cb, verbose=False,
                eval_validation=False)
        except torch.cuda.OutOfMemoryError:
            # A large lstm_hidden x head_hidden draw can exceed the 12GB card; prune the
            # trial instead of killing the study.
            trial.set_user_attr("prune_reason", "cuda_out_of_memory")
            print(f"  [t{trial.number}] OOM — pruned", flush=True)
            raise optuna.TrialPruned()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # test_nll is train_model's best (lowest) test NLL — the checkpoint is selected on
        # it, so test_m reflects that same epoch. Balanced accuracy is recorded for reporting
        # only and does not drive selection.
        trial.set_user_attr("test_balanced_accuracy", test_m.get("balanced_accuracy", 0.5))
        trial.set_user_attr("test_gate_auc", test_m.get("gate_auc"))
        print(f"[trial {trial.number}] DONE: test_nll {test_nll:+.5f} "
              f"test_balanced_acc {test_m.get('balanced_accuracy', 0.5):.4f}", flush=True)
        return test_nll
    return objective


def _attrs(trial):
    """Trial user attrs with validation metrics stripped.

    Studies created before the HPO went validation-blind still carry val_* attrs in
    study.db; filtering here keeps them from being re-emitted into the leaderboard."""
    return {k: v for k, v in trial.user_attrs.items() if not k.startswith("val_")}


def study_dir(study_name):
    """Per-study output directory, created on demand.

    Namespaced for the same reason trial checkpoints are: two studies writing
    TUNE_DIR/leaderboard.json meant the file described whichever ran last, so the summary
    silently stopped matching the checkpoints beside it."""
    d = TUNE_DIR / study_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_progress(study, trial):
    """Optuna callback: rewrite the leaderboard after every trial so it can be watched live."""
    done = completed_trials(study)
    rows = sorted(done, key=lambda t: t.value)  # lower test NLL is better
    board = [{"trial": t.number, "test_nll": t.value, **_attrs(t), **t.params}
             for t in rows]
    sd = study_dir(study.study_name)
    (sd / "leaderboard.json").write_text(json.dumps(board, indent=2))
    if rows:
        b = rows[0]
        (sd / "best_params.json").write_text(json.dumps(
            {"trial": b.number, "test_nll": b.value, **_attrs(b), **b.params},
            indent=2))
        if "config" in b.user_attrs:
            (sd / "best_config.json").write_text(json.dumps(b.user_attrs["config"], indent=2))
        n_pruned = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.PRUNED)
        print(f">>> progress: {len(done)} done / {n_pruned} pruned | "
              f"best test_nll {b.value:+.5f} (trial {b.number})", flush=True)


def completed_trials(study):
    """Pruned trials may carry intermediate values; they are never fitted finalists."""
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and math.isfinite(t.value)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs-per-trial", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--startup", type=int, default=8, help="random TPE startup trials")
    ap.add_argument("--warmup-epochs", type=int, default=8,
                    help="epochs before the pruner may stop a trial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--study", type=str, default=None,
                    help="study name (fixed per config); defaults to 'zib_nll'")
    ap.add_argument("--link", choices=["logit", "cloglog"], default="logit",
                    help="link for the occurrence gate and Beta-mean heads (fixed per study)")
    ap.add_argument("--pca", action="store_true",
                    help="use PCA feature decomposition; n_pca is tuned per trial")
    ap.add_argument("--max-county-embed", type=int, default=0,
                    help="ceiling for the searched per-county embedding width (0 = do not "
                         "search it, leaving existing studies' search space untouched). 16 is "
                         "the suggested ceiling: it matches lc_embed_dim's and costs "
                         "3108*16 = 49,728 params, ~19%% of the tuned model. Changing this "
                         "on an existing study is rejected — pick a new --study name")
    ap.add_argument("--tune-harmonics", action="store_true",
                    help=f"search the seasonal order n_harmonics over [1, {MAX_HARMONICS}]; "
                         f"the panel is built once at {MAX_HARMONICS} and each trial slices "
                         f"the trailing block, so nothing is rebuilt. Off by default so "
                         f"existing studies' search space is untouched; changing it on an "
                         f"existing study is rejected — pick a new --study name")
    ap.add_argument("--static-bypass", action="store_true",
                    help="hold the 12 near-constant predictors out of the PCA rotation and feed "
                         "them raw with tail transforms, leaving 39 dynamic predictors to "
                         "rotate. This changes the n_pca range, so changing it on an existing "
                         "study is rejected — pick a new --study name")
    ap.add_argument("--no-covariates", action="store_true",
                    help="drop all numeric predictors, leaving the Fourier seasonal block as "
                         "the only covariate columns. The arm's inputs are then fire history, "
                         "season, land cover, county identity and the graph — the null-covariate "
                         "counterpart to the climatology floor. Incompatible with --pca and "
                         "--static-bypass, which rotate the block this empties; fixed per study")
    ap.add_argument("--static-only", action="store_true",
                    help="legacy 12-column near-static subset, including annual/epoch and "
                         "land-cover information read at target months; retained only to "
                         "reproduce historical fits. Use --forecast-safe for forecasts. "
                         "Fixed per study")
    ap.add_argument("--tune-lookback", action="store_true",
                    help="search lookback over [12,24,36,48]; test/validation origins are "
                         "unchanged for any lookback <= 185, so the objective stays comparable")
    ap.add_argument("--forecast-safe", action="store_true",
                    help="fixed terrain + calendar + origin fire history only; no land cover, "
                         "PCA or teacher forcing; 48-month history and compact search space")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="fixed origins per minibatch for this run, persisted in each trial's "
                         "full config; batch 1 permits larger recurrent networks on a 12GB GPU")
    ap.add_argument("--min-complete", type=int, default=None,
                    help="stop once this many finite COMPLETE trials exist, subject to --trials "
                         "maximum additional attempts; useful after memory-infeasible attempts")
    ap.add_argument("--retry-trials", type=int, nargs="*", default=[],
                    help="enqueue these prior trial configurations at this run's batch size; "
                         "records retry_of_trial, so GPU infeasibility is not model-quality evidence")
    ap.add_argument("--no-pruning", action="store_true",
                    help="complete a bounded full evaluation with ordinary test early stopping "
                         "but no median pruning; policy is recorded on each trial")
    args = ap.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.min_complete is not None and args.min_complete < 1:
        raise SystemExit("--min-complete must be positive")

    if args.forecast_safe:
        if any((args.pca, args.static_bypass, args.static_only, args.no_covariates,
                args.tune_lookback)):
            raise SystemExit("--forecast-safe cannot be combined with alternate feature/window modes")
        args.tune_harmonics = True
        args.max_county_embed = 16
        if not torch.cuda.is_available():
            raise RuntimeError("forecast_safe GPU tuning requested but CUDA is unavailable")

    if args.no_covariates and (args.pca or args.static_bypass):
        raise SystemExit("--no-covariates empties the predictor block that --pca and "
                         "--static-bypass rotate; they cannot be combined.")
    if args.static_only and args.no_covariates:
        raise SystemExit("--static-only keeps the 12 time-invariant predictors and "
                         "--no-covariates drops every predictor; they are contradictory.")
    if args.static_only and args.static_bypass:
        raise SystemExit("--static-bypass holds STATIC_COLS out of the rotation so the 39 "
                         "dynamic predictors can be rotated, but --static-only drops those 39. "
                         "Nothing would be left to rotate.")
    if args.static_only and args.pca:
        raise SystemExit("--pca clears drop_features to rotate the full predictor block, which "
                         "would undo --static-only. A rotation of 12 near-constant columns is "
                         "degenerate in any case.")

    study_name = args.study or ("zib_nll" + ("_pca" if args.pca else ""))

    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = Config()
    if args.forecast_safe:
        base_cfg.forecast_safe = True
        base_cfg.lookback = 48
        base_cfg.drop_features = []
        base_cfg.lc_embed_dim = 0
        base_cfg.n_samples = 0
    base_cfg.static_bypass = args.static_bypass
    if args.no_covariates:
        base_cfg.drop_features = numeric_predictor_names()
    if args.static_only:
        base_cfg.drop_features = static_only_drops()
    if args.pca:
        base_cfg.drop_features = []
        base_cfg.n_pca = 1
        base_cfg.n_harmonics = MAX_HARMONICS
    if args.tune_harmonics:
        # Build at the ceiling; train_model slices down to each trial's cfg.n_harmonics.
        base_cfg.n_harmonics = MAX_HARMONICS
    panel = load_panel(base_cfg)
    # n_pca ranges over rotatable predictor columns only; the harmonic and static blocks bypass
    # the rotation and are always appended, so they are not part of the budget.
    max_pcs = (panel.cov.shape[-1] - panel.n_harmonic_cols - panel.n_static_cols) \
        if args.pca else None

    # group=True keeps the conditional tf_ratio_* pair inside the multivariate model; under the
    # default group=False they fall outside the intersection search space and go univariate.
    # Optuna's default gamma (ceil(0.1n)) put only 5 trials in the "good" density l(x) at n=50 —
    # far too few to estimate over this search space's ~16 dimensions. Pruned trials count toward
    # n but fill the "above" set first, so they inflate gamma without feeding l(x). gamma is
    # deprecated as of Optuna 4.9 (removal in 6.0) with no replacement, so pinning <6.0 is what
    # keeps this tunable; on removal the split ratio reverts to the 10% that caused the problem.
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup,
                                         multivariate=True, group=True,
                                         gamma=lambda n: max(6, n // 4),
                                         n_ei_candidates=96)
    pruner = (optuna.pruners.NopPruner() if args.no_pruning else
              optuna.pruners.MedianPruner(n_startup_trials=args.startup,
                                          n_warmup_steps=args.warmup_epochs))
    storage = f"sqlite:///{TUNE_DIR / 'study.db'}"
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                study_name=study_name, storage=storage, load_if_exists=True)
    if study.direction != optuna.study.StudyDirection.MINIMIZE:
        raise SystemExit(
            f"study '{study_name}' already exists with direction {study.direction.name}, "
            f"but this objective minimizes test NLL. Pick a new --study name.")
    for attr, val in (("link", args.link), ("pca", args.pca),
                      ("tune_lookback", args.tune_lookback),
                      ("max_county_embed", args.max_county_embed),
                      ("tune_harmonics", args.tune_harmonics),
                      ("static_bypass", args.static_bypass),
                      ("no_covariates", args.no_covariates),
                      ("static_only", args.static_only),
                      ("forecast_safe", args.forecast_safe)):
        existing = study.user_attrs.get(attr)
        if existing is None:
            study.set_user_attr(attr, val)
        elif existing != val:
            raise SystemExit(
                f"study '{study_name}' was created with {attr}={existing!r}, but this run "
                f"has {attr}={val!r}. Mixing {attr} in one study is invalid; pick a new "
                f"--study name.")

    objective = make_objective(panel, args.epochs_per_trial, args.patience, args.link,
                               args.tune_lookback, study_name,
                               max_pcs=max_pcs, max_county_embed=args.max_county_embed,
                               max_harmonics=MAX_HARMONICS if args.tune_harmonics else 0,
                               static_bypass=args.static_bypass,
                               no_covariates=args.no_covariates,
                               static_only=args.static_only, forecast_safe=args.forecast_safe,
                               batch_size=args.batch_size,
                               pruning_policy="disabled_test_early_stopping_only" if args.no_pruning
                                              else "median_test_nll")
    previous = {trial.number: trial for trial in study.trials}
    for number in args.retry_trials:
        if number not in previous:
            raise SystemExit(f"cannot retry missing trial {number}")
        study.enqueue_trial(previous[number].params,
                            user_attrs={"retry_of_trial": number,
                                        "retry_reason": "smaller_batch_after_gpu_memory_failure"})
    def stop_when_complete(study, trial):
        if args.min_complete is not None and len(completed_trials(study)) >= args.min_complete:
            study.stop()

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   show_progress_bar=False, callbacks=[save_progress, stop_when_complete])

    done = completed_trials(study)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if not done:                       # study.best_trial raises if nothing completed
        print(f"\nno completed trials ({len(study.trials)} total, {len(pruned)} pruned) "
              f"— nothing to report")
        return
    best = study.best_trial
    print(f"\n==================== STUDY '{study_name}' ====================")
    print(f"  trials: {len(study.trials)} total | {len(done)} complete | {len(pruned)} pruned")
    print(f"  BEST trial #{best.number}: test_nll={best.value:+.5f} "
          f"| test_auc={best.user_attrs.get('test_gate_auc')}")
    for k, v in best.params.items():
        vs = f"{v:.3g}" if isinstance(v, float) else str(v)
        print(f"    {k:16s} = {vs}")
    print("=============================================================")

    rows = sorted(done, key=lambda t: t.value)
    board = [{"trial": t.number, "test_nll": t.value, **_attrs(t), **t.params}
             for t in rows]
    sd = study_dir(study_name)
    (sd / "leaderboard.json").write_text(json.dumps(board, indent=2))
    (sd / "best_params.json").write_text(json.dumps(
        {"trial": best.number, "test_nll": best.value, **_attrs(best), **best.params},
        indent=2))
    print(f"best ckpt: {sd / f'trial_{best.number:03d}' / 'best.pt'}")
    print(f"wrote {sd/'leaderboard.json'} and {sd/'best_params.json'}")


if __name__ == "__main__":
    main()
