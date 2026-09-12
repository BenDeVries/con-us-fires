"""Apply the published forecast diagnostics to fixed conditional representatives."""
from __future__ import annotations

import shutil
import numpy as np
import pandas as pd
from scipy.stats import norm

from .state import ROOT, read_json, write_json, sha256, digest
from .data import KEYS, SPLITS, Inputs
from .metrics import row_nll


def checked_predictions(experiment, inputs, family, split):
    path = experiment / family / 'selected' / f'predictions_{split}.parquet'
    meta = read_json(path.with_suffix('.json'))
    selection = read_json(experiment / family / 'selection.json')
    if sha256(path) != meta['sha256'] or digest(selection) != meta['selection_sha256']:
        raise ValueError('Prediction provenance mismatch')
    frame = pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)
    canonical = inputs.keys(split).sort_values(KEYS).reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction key')
    pd.testing.assert_frame_equal(frame[KEYS + ['y_true']], canonical, check_dtype=False, check_exact=True)
    if not np.allclose(frame.e_y, frame.p_occ * frame.mu, atol=0, rtol=1e-12):
        raise ValueError('Expected fraction differs from hurdle parameters')
    score = row_nll(frame.y_true, frame.p_occ, frame.mu, frame.phi)
    if not np.isclose(score.mean(), meta['mean_nll'], atol=1e-9, rtol=0):
        raise ValueError('Saved score mismatch')
    return frame


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def build(experiment, publish=False, reuse_stacf=False):
    from .search import verify
    from .__main__ import freeze, selections
    from reporting import prepare_forecast_results as report
    spec = verify(experiment); assessment = freeze(experiment)
    if spec['smoke'] and publish:
        raise ValueError('Smoke results cannot be published')
    inputs = Inputs(experiment, spec)
    out = experiment / 'report'; out.mkdir(exist_ok=True)
    report.MODELS = {'xgb': {'label': 'XGBoostLSS', 'color': '#ad6100'},
                     'gnn': {'label': 'GCN → LSTM', 'color': '#17634e'}}
    weights = report.row_standardized_graph(inputs.edge, inputs.n)
    operators = report.stacf_operators(inputs.edge, inputs.n)
    previous = read_json(out / 'summary.json') if reuse_stacf and (out / 'summary.json').exists() else None
    results, sources = {}, {}
    for family in ('xgb', 'gnn'):
        results[family] = {}
        for split in SPLITS:
            print(f'Scoring {family}/{split}', flush=True)
            frame = checked_predictions(experiment, inputs, family, split)
            path = experiment / family / 'selected' / f'predictions_{split}.parquet'
            sources[f'{family}/{split}'] = sha256(path)
            summary = report.summarize(frame, weights, include_discrimination=split != 'train')
            if split == 'train':
                if previous and previous['predictions'][f'{family}/train'] == sources[f'{family}/train']:
                    summary['stacf'] = previous['results'][family]['train']['stacf']
                else:
                    _, _, f = report.distribution_scores(frame.y_true.to_numpy(), frame.p_occ.to_numpy(),
                                                         frame.mu.to_numpy(), frame.phi.to_numpy())
                    pit = np.random.default_rng(0).random(len(frame)) * (1-frame.p_occ.to_numpy())
                    pos = frame.y_true.to_numpy() > 0
                    pit[pos] = 1-frame.p_occ.to_numpy()[pos] + frame.p_occ.to_numpy()[pos]*f[pos]
                    residual = norm.ppf(np.clip(pit, 1e-12, 1-1e-12))
                    summary['stacf'] = report.observed_stacf(frame, residual, operators,
                        max_lag=min(24, frame.target_date.nunique()-1), n_perm=3 if spec['smoke'] else 199)
            results[family][split] = summary
    summary = clean_json({'experiment': experiment.name, 'protocol_sha256': spec['sha256'],
        'prediction_intervals': False, 'information_set': spec['information_set'],
        'assessment': assessment, 'selections': selections(experiment),
        'predictions': sources, 'results': results,
        'seed_evidence': {f: read_json(experiment / f / 'finalist_evidence.json') for f in ('xgb','gnn')},
        'search': {f: read_json(experiment / f / 'leaderboard.json') for f in ('xgb','gnn')}})
    write_json(out / 'summary.json', summary)
    render_results(summary, out)
    if not spec['smoke']:
        maps(experiment, out / 'maps')
        shutil.copy2(ROOT / 'assets/forecast/validation-maps.js', out / 'validation-maps.js')
    write_page(summary, out / 'conditional.qmd', maps=not spec['smoke'])
    if publish:
        destination = ROOT / 'assets/conditional'; destination.mkdir(exist_ok=True)
        for path in out.iterdir():
            if path.name == 'conditional.qmd':
                shutil.copy2(path, ROOT / path.name)
            elif path.is_dir():
                shutil.copytree(path, destination / path.name, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination / path.name)
    print(f'Report: {out}', flush=True)


def render_results(summary, out):
    from reporting import prepare_forecast_results as report
    report.MODELS = {'xgb': {'label': 'XGBoostLSS', 'color': '#ad6100'},
                     'gnn': {'label': 'GCN → LSTM', 'color': '#17634e'}}
    results = summary['results']
    report.draw_figure(results, ('train','test'), out / 'train-test-diagnostics.png')
    report.draw_figure(results, ('validation',), out / 'validation-diagnostics.png')
    for split in ('test','validation'):
        report.draw_discrimination(results, split, out / f'{split}-discrimination.png')
    limit = report.stacf_color_limit(results)
    for family in ('xgb','gnn'):
        report.draw_train_stacf(results[family], report.MODELS[family]['label'], limit,
                                out / f'{family}-train-G1_stacf.png')


def maps(experiment, out):
    from reporting import prepare_validation_maps as module
    module.ROOT = ROOT
    module.MODEL_DIRS = tuple(str((experiment / f / 'selected').relative_to(ROOT)) for f in ('xgb','gnn'))
    out.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'assets/forecast/maps/counties.json', out / 'counties.json')
    module.build_maps(destination=out, geometry=False)


def write_page(summary, path, maps=True):
    lines = ['---', 'title: "Conditional predictions"', 'subtitle: "Observed future covariates; fixed model assessment"',
             '---', '', 'These retrospective predictions condition on observed target-period environmental covariates, '
             'including annual land-cover and interpolated population/built-area products. They are not forecasts of '
             'those covariates or estimates of causal scenario effects.', '',
             'The searches fit parameters on train and select configurations and checkpoints by test NLL. '
             'The three leading configurations per class are checked across three seeds; the selected configuration’s '
             'seed-0 fit is assessed on validation. This later period was inspected in previous work and is not a pristine holdout.', '',
             '## Selected configurations', '', '| Model | Dynamic representation | Harmonic order | Lookback |',
             '|---|---|---:|---:|']
    for family, selected in summary['selections'].items():
        p = selected['parameters']; rep = 'Raw' if p['representation']=='raw' else f'{p["n_pca"]} PCs'
        lines.append(f'| {family.upper()} | {rep} | {p["n_harmonics"]} | {p.get("lookback",48)} months |')
    for splits, title in [(('train','test'), 'Train and test evidence'), (('validation',), 'Validation assessment')]:
        lines += ['', f'## {title}', '', '| Split | Model | Requests | NLL | CRPS | MAE | Brier |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for split in splits:
            for family in ('xgb','gnn'):
                r = summary['results'][family][split]
                lines.append(f'| {split} | {family.upper()} | {r["rows"]:,} | {r["mean_nll"]:.6f} | '
                    f'{r["mean_crps"]:.7f} | {r["mae_expected_fraction"]:.7f} | {r["brier"]:.6f} |')
        prefix = 'train-test' if 'train' in splits else 'validation'
        lines += ['', f'![{title}: randomized residual QQ, occurrence reliability, and monthly residual means.]'
                  f'(assets/conditional/{prefix}-diagnostics.png)', '']
        for split in splits:
            if split != 'train':
                lines += [f'![{split.title()} occurrence ROC and precision–recall curves.]'
                          f'(assets/conditional/{split}-discrimination.png)', '']
    lines += ['## Spatial and temporal residual patterns', '',
              'The definitions, graph weights, permutation procedure, and dependence limitations are the same as '
              'in the [forecast comparison](comparison.qmd). Shared training STACF plots use 199 month and county '
              'permutations, orders 0–3 and lags 0–24. Repeated requests are dependent; these are exploratory diagnostics.', '']
    for family in ('xgb','gnn'):
        lines += [f'![{family.upper()} training G1 STACF.](assets/conditional/{family}-train-G1_stacf.png)', '']
    lines += ['County-mean Moran’s I, monthly lag-one correlation, seed results, search history, and unrounded '
              'diagnostics are in the [result manifest](assets/conditional/summary.json). Scores describe plug-in '
              'hurdle distributions; no prediction intervals or definitive model-class winner are inferred.', '']
    if maps:
        discussion = (ROOT / 'discussion.qmd').read_text()
        start, end = discussion.index('<div id="validation-maps"'), discussion.index('</script>') + len('</script>')
        lines += ['## Animated validation maps', '',
                  'Top: XGBoostLSS; middle: observations; bottom: GCN–LSTM. Left: occurrence probability or presence. '
                  'Right: mean burned fraction given a burn, or the observed fraction. Leads are displayed separately.', '',
                  discussion[start:end].replace('assets/forecast/', 'assets/conditional/'), '']
    lines += ['## Reproduction', '',
              'The conditional release contains the prepared panel, fitted transformations, selected checkpoints, '
              'prediction tables, training and tuning code, environment versions, and checksums. Follow its README '
              'to recompute diagnostics, regenerate predictions, or refit the selected configurations.', '']
    path.write_text('\n'.join(lines) + '\n')
