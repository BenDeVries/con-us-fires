# County burned-area models

This is the primary development repository. Work and experiments belong on
`development`; `main` contains the publication and material needed to reproduce it.
The existing forecast tutorial remains available alongside the conditional work.

## Conditional experiment

The task is retrospective prediction of twelve future monthly burned fractions
given their observed environmental covariates and fire history through the origin.
Annual land-cover and interpolated population/built-area products are part of that
conditioning information, including later source epochs. These are not operational
forecasts or causal interventions on weather.

The panel contains 3,108 counties over 2003–2024. Parameters fit on train through
May 2018; test through July 2020 selects configurations and checkpoints. Validation
through December 2024 assesses both frozen representatives. Validation was inspected
in previous work and is not described as untouched.

All trials use identical origins with a 48-month history allowance and complete
12-month target windows. Neural lookback varies from **12 through 48 months in
three-month increments**. PCA compresses 39 dynamic predictors; the other 12
continuous predictors bypass it. The search includes 0–39 PCs, raw predictors, and
0–6 harmonic orders. Zero PCs removes only the dynamic block. Links stay fixed.
Tree county summaries expand using observations strictly before the origin. Other
history features are at or before the origin. No climatology or neighbor-PC blocks
enter the new study.

Each class starts with 28 balanced configurations: every representation rung and
harmonic order occurs four times, and every neural lookback at least twice. A fresh
TPE study then requests 52 adaptive trials. Complete initial coverage is required
before final selection. The top three configurations receive seeds 0, 1, 2; mean
test NLL selects the configuration and its seed-0 fit is the representative.

## Run and resume

Use separate environments for neural/report and XGBoost work. On the development
machine these are `fire-nn` and `fire-xgb`. Dependency files are in `requirements/`;
every worker records and checks its actual environment. CPU inference/reporting can
use the report environment. Exact GPU fitting uses the recorded CUDA PyTorch build.

```bash
conda run -n fire-nn python -m conditional --experiment conditional_v1 prepare
conda run -n fire-nn python -m conditional --experiment conditional_v1 run --model gnn --hours 12
conda run -n fire-xgb python -m conditional --experiment conditional_v1 run --model xgb --hours 12
python -m conditional --experiment conditional_v1 status
```

For the current experiment's user-requested 16-thread tree run, use:

```bash
conda run -n fire-xgb python tools/run_xgb.py --experiment conditional_v1 --threads 16 --hours 12
```

This resource wrapper runs the original source/data checks, resumes the saved trial,
and records the execution override in `xgb/runtime_resources.json` and each trial's
`resource_history.json`. It preserves the frozen experiment identity; the fitted
configuration records the effective thread count. Neural training continues separately.

Repeat either `run` command to continue. Checkpoints contain optimizer/booster state,
random state, early-stopping state, the best checkpoint, and the global epoch/round.
The search journal retains active trials and sampler state. Tree fits reset native
updater state at every round and use global iteration seeds: the native column
sampler keeps RNG state outside model serialization ([XGBoost implementation](https://github.com/dmlc/xgboost/blob/v3.2.0/src/tree/updater_quantile_hist.cc#L552)). One worker per class
holds a filesystem lock. Send SIGINT/SIGTERM to a worker for a cooperative pause;
work since the last completed epoch/round can be replayed after an abrupt shutdown.
Code, data, preprocessing, or environment changes require a new experiment.

To run both workers and automatically advance completed searches through selection
and analysis, use the shared-session supervisor (Python executable paths are explicit):

```bash
python -m conditional.session --experiment conditional_v1 --hours 12 \
  --nn-python /path/to/fire-nn/bin/python \
  --xgb-python /path/to/fire-xgb/bin/python
```

The 12-hour limit is a session boundary, not a claim that the search is complete.
Per-worker logs, progress, SQLite studies, failure records, and session status live
under `output/conditional_v1/`. An initialization structure that fails is replaced
within its representation/harmonic/lookback stratum; repeated failure stops that
worker for investigation. No paused trial is treated as complete.

## Select, assess, and reproduce

After both studies complete:

```bash
conda run -n fire-nn python -m conditional select --model gnn
conda run -n fire-xgb python -m conditional select --model xgb
conda run -n fire-nn python -m conditional predict --model gnn --device cuda
conda run -n fire-xgb python -m conditional predict --model xgb
conda run -n fire-nn python -m conditional report --publish
conda run -n fire-nn python -m conditional package --destination releases/conditional_v1
```

Both representatives are frozen before prediction. Full forecast keys and outcomes
must match; report NLL must reproduce selected checkpoint NLL. Reports reuse forecast
QQ/reliability, ROC/PR, Moran's I, temporal residual and training STACF definitions.
Validation maps show occurrence and mean fraction given a burn by lead.
`--reuse-stacf` reuses hash-matched calculations; a full report repeats 199 shared
month/county permutations. Neither prediction intervals nor class-winner significance
claims are made.

Release creation stages allowlisted source, prepared inputs, fitted transforms,
selected models, predictions, environments, and checksum manifests in a new directory.
It excludes databases, caches, intermediate fits, and logs. It never merges branches
or deploys. The release's `REPRODUCE.md` explains clean-extraction verification,
reporting, prediction, refitting, and rendering. `refit --model ...` is resumable and
writes separate output. `migration-manifest.json` records the imported source bytes.

The prepared panel is local under `output/data/`; it is included in releases rather
than tracked as raw development data. Existing forecast reproduction archives remain
self-contained under `assets/reproduction/`.

## Verification

```bash
conda run -n fire-nn python -m unittest discover -s tests -p 'test_conditional*.py' -v
conda run -n fire-xgb python -m unittest discover -s tests -p 'test_tree*.py' -v
conda run -n fire-nn python -m conditional --experiment conditional_smoke prepare --smoke
```

Run the same search, select, predict, report and package commands against the smoke
experiment, using `--device cpu` for its neural runs. Smoke studies use 16 counties,
two origins per split and short fits, and cannot publish results. `run --pause-after 1`
exercises one-boundary interruption recovery. They never contribute trials to the
full experiment.
