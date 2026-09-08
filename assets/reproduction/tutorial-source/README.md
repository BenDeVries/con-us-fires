# Reproduce Forecasting County Burned Area

Ben DeVries and Codex (OpenAI)

Download `tutorial-source.zip`, `xgb-predictions.zip`, and `gnn-predictions.zip`
from the tutorial's Results and Discussion page. Keep the three archives in one
directory. Extract only `tutorial-source.zip` into a new `tutorial` directory.
The reproduction commands run from that directory and require no repository access,
Earth Engine account, or access to the author's filesystem.

## Recompute the published analysis and render

Use Python 3.10 and Quarto 1.9.38. Create an environment, then run:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python reproduce.py unpack --downloads ..
python reproduce.py report
python reproduce.py render
```

`unpack` verifies the supplied source/input checksums and each compressed prediction
table, reconstructs the common county × origin × horizon keys and observed targets,
and restores both models' three prediction files. It also places the three original
archives in the rendered site's download directory. Keep those downloads available
when rendering, since the tutorial links to them.

`report` recomputes all scores, QQ curves, occurrence reliability, spatial and temporal
summaries, full ROC/PR/AP metrics, and both training STACFs including their 199 shared
permutations. It checks the numerical results against the supplied result manifest,
regenerates the six diagnostic figures, the schematic animation, all twelve leads of
the validation maps, and the result tables. The permutation calculations are the
most expensive part. For a faster run, `python reproduce.py report --reuse-stacf`
reuses the supplied STACF summaries and regenerates their plots while recomputing all
other diagnostics; that option does not independently reproduce the permutations.

`render` builds a fresh six-page Quarto site and checks its resources and links. It
prints the `_site` location. Serve that directory with `python -m http.server 8000`
and visit `http://localhost:8000` to use the maps; browser restrictions on local
`file:` URLs prevent their data requests. Map playback starts only when Play is pressed.

## Inputs and file scope

The analysis starts from the supplied prepared county-month panel. Its three Parquet
files retain only the response, dates and county keys, four standardized terrain
columns, and the occurrence column required by the tree feature loader. The terrain
scaler records the training means and standard deviations. County node IDs and the
adjacency graph define the fitted geography. `counties.json` contains projected,
simplified display boundaries from US Census TIGER/Line 2018; those paths reproduce
the maps without downloading full-resolution geometry.

The prediction archives contain only `p_occ`, `mu`, and `phi` in their original
numeric precision, sorted by origin, horizon, and county FIPS. The response is read
from the common panel. The expected fraction is reconstructed as `p_occ * mu`.
Each archive's `layout.json` records origin ranges, row counts, column types,
parameter checksums, and the hash of its source prediction file.

The source archive includes the selected configurations and checkpoints, the training
and tuning code and its imports, the report/map generators, the six chapter sources,
their stylesheet and open-access bibliography, and the numerical reference summary.
The summary records all trial outcomes, the selected trials, seeds, configurations,
and software versions. Source provenance hashes identify the original experiment;
`files.sha256.json` identifies the compact files distributed here. Prepared-data
packaging does not repeat the remote raster export or certify satellite coverage.

## Predict again or refit the selected configurations

The shipped checkpoint files support a separate prediction check. In the report
environment, regenerate neural predictions into a new directory:

```bash
python reproduce.py predict-gnn --device cpu --split validation
```

For trees, use a separate Python 3.10 environment with `requirements-xgb.txt`, copy
`model.pkl` and `config.json` from `output/xgb_forecast_safe_20260907` to a new
`regenerated/xgb` directory, and run:

```bash
python -m model.xgb.predict --model-dir regenerated/xgb --split validation
```

Repeat with `--split train` and `--split test` for the other periods. To repeat
parameter fitting with the selected settings, use the corresponding environment:

```bash
python reproduce.py refit-xgb --device cpu
python reproduce.py refit-gnn --device cuda
```

These commands write new fits under `refits/`, use test for early stopping, and
leave validation out of fitting. The included `model.tune` and `model.xgb.tune`
modules expose the search procedures through `--help` (install `optuna==4.9.0`
in the neural environment to run its tuner); the reference summary
records the completed searches and their selection rule. Searches and refits can
take hours and depend on available memory.

The original fitting versions are in each checkpoint directory's
`runtime_environment.json`. The neural fit used a CUDA 12.8 PyTorch development
build, `2.12.0.dev20260408+cu128`, under Python 3.11.15. CPU inference and report
reproduction use the separately pinned report environment. A refit with a different
PyTorch build, device, or GPU kernel need not reproduce the selected weights or
scores bit for bit. The saved prediction parameters support reproduction of the
reported results independently of that fitting variability.
