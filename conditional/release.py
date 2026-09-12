"""Allowlisted source/data/model release; no Git operations or deployment."""
from __future__ import annotations

import shutil
import zipfile

from .state import ROOT, read_json, write_json, sha256


def package(experiment, destination):
    from .__main__ import freeze
    from .search import verify
    spec = verify(experiment); freeze(experiment)
    summary = read_json(experiment / 'report/summary.json')
    if summary['protocol_sha256'] != spec['sha256']:
        raise ValueError('Report does not belong to selected experiment')
    from .data import Inputs
    from .report import checked_predictions
    inputs = Inputs(experiment, spec)
    for family in ('gnn','xgb'):
        for split in ('train','test','validation'):
            checked_predictions(experiment, inputs, family, split)
            if sha256(experiment / family / 'selected' / f'predictions_{split}.parquet') != summary['predictions'][f'{family}/{split}']:
                raise ValueError('Report is stale relative to its prediction files')
    del inputs
    if destination.exists():
        raise ValueError('Release destination must be new')
    destination.mkdir(parents=True)
    stage = destination / 'source'; stage.mkdir()
    paths = []
    for directory in ('conditional', 'model', 'reporting'):
        paths += list((ROOT / directory).rglob('*.py'))
    paths += list(ROOT.glob('*.qmd'))
    paths += [ROOT / name for name in ('README.md','_quarto.yml','styles.css','references.bib',
                                     'calibration.py','migration-manifest.json','config.py',
                                     'step10_join.py','step11_impute_split_scale.py')]
    paths += list((ROOT / 'requirements').glob('*.txt'))
    paths += [ROOT / relative for relative in spec['inputs']]
    paths += list((ROOT / 'assets/forecast').rglob('*'))
    paths += list((ROOT / 'assets/fig').rglob('*'))
    paths += [p for p in (ROOT / 'assets/reproduction').iterdir() if p.is_file()]
    paths += [experiment / name for name in ('spec.json', 'assessment.json')]
    paths += list((experiment / 'preprocessing').iterdir())
    paths += list((experiment / 'report').rglob('*'))
    for family in ('gnn','xgb'):
        folder = experiment / family
        paths += [folder / name for name in ('selection.json', 'environment.json', 'initialization.json',
                  'leaderboard.json', 'finalist_evidence.json', 'status.json')]
        paths += list((folder / 'selected').iterdir())
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix in ('.db', '.log') or path.name in ('last.pt','last.pkl','journal.pkl'):
            raise ValueError(f'Development artifact entered publication allowlist: {relative}')
        target = stage / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if not spec['smoke']:
        shutil.copytree(experiment / 'report', stage / 'assets/conditional', dirs_exist_ok=True)
        shutil.copy2(experiment / 'report/conditional.qmd', stage / 'conditional.qmd')
        (stage / 'assets/conditional/conditional.qmd').unlink()
    name = experiment.name
    (stage / 'REPRODUCE.md').write_text(f'''# Reproduce {name}

This is a {'SMOKE TEST, not a publication' if spec['smoke'] else 'conditional model release'}.
Run from this extracted directory. No access to the previous repository is needed.

1. Verify all files: `python verify_release.py`.
2. Create Python environments using `requirements/conditional-nn.txt` and
   `requirements/conditional-xgb.txt`; exact fitting versions are recorded under
   `output/{name}/{{gnn,xgb}}/environment.json`.
3. Recompute the analysis in the neural/report environment:
   `python -m conditional --experiment {name} report`.
   `--reuse-stacf` reuses verified STACF calculations while recomputing other scores.
4. Regenerate predictions with each model's environment:
   `python -m conditional --experiment {name} predict --model gnn --device cpu`
   and `python -m conditional --experiment {name} predict --model xgb`.
5. Refit selected settings using `refit --model gnn --device cuda` or `refit --model xgb`.
   Fits use train for parameters and test for early stopping; repeated calls resume them.
6. Render the website: `quarto render`. Serve `_site` with `python -m http.server`
   from that directory to use the interactive maps.

Saved predictions reproduce scores without stochastic refitting. GPU refits can differ
across kernels/devices. Environments must match recorded versions to resume fitting.
To conduct a fresh search, use `prepare` with a new experiment name, followed by `run`.
The complete search histories are included; development SQLite databases are excluded.
''')
    (stage / 'verify_release.py').write_text('''from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / "files.sha256.json").read_text())
for relative, expected in manifest.items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch: {relative}")
print(f"Verified {len(manifest)} files")
''')
    checksums = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    write_json(stage / 'files.sha256.json', checksums)
    archive = destination / f'{name}-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    write_json(destination / 'release-manifest.json', {'experiment': name, 'smoke': spec['smoke'],
        'source_archive': archive.name, 'sha256': sha256(archive), 'files': len(checksums),
        'experiment_sha256': spec['sha256']})
    print(f'Prepared {archive}; no branch was merged or deployed.', flush=True)
