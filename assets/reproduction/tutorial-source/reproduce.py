#!/usr/bin/env python3
"""Reproduce tutorial results from the accompanying, independently downloadable inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "tutorial-writeup/final-product"
MODELS = {"xgb": "output/xgb_forecast_safe_20260907", "gnn": "output/model/forecast_safe_20260907"}
SPLITS = ("train", "test", "validation")
sys.path.insert(0, str(ROOT))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def verify_files():
    for relative, digest in json.loads((ROOT / "files.sha256.json").read_text()).items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or changed input: {relative}")
    print("Source, model and prepared-data checksums verified.", flush=True)


def unpack(downloads):
    import numpy as np
    import pandas as pd

    downloads = downloads.resolve()
    names = ("tutorial-source.zip", "xgb-predictions.zip", "gnn-predictions.zip")
    destination = SITE / "assets/reproduction"
    destination.mkdir(parents=True, exist_ok=True)
    archives = {}
    for name in names:
        path = downloads / name
        if not path.is_file():
            raise FileNotFoundError(f"Download {name} to {downloads} first")
        archives[name] = {"bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        target = destination / name
        if path != target:
            shutil.copy2(path, target)
    (destination / "manifest.json").write_text(json.dumps({"archives": archives}, indent=2) + "\n")
    report = module("forecast_report", "tutorial-writeup/workspace/prepare_forecast_results.py")
    for short, directory in MODELS.items():
        with zipfile.ZipFile(downloads / f"{short}-predictions.zip") as archive:
            layout = json.loads(archive.read("layout.json"))
            for split in SPLITS:
                meta = layout[split]
                raw = archive.read(f"{split}.parquet")
                if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                    raise ValueError(f"Damaged prediction parameters: {short}/{split}")
                import io
                params = pd.read_parquet(io.BytesIO(raw))
                canonical, lookup = report.read_canonical(split, [])
                counties = canonical[["county_fips", "node_id"]].drop_duplicates().sort_values("county_fips")
                dates = pd.date_range(meta["origin_min"], meta["origin_max"], freq="MS")
                n = len(counties)
                frame = pd.DataFrame({
                    "origin_date": np.repeat(dates.to_numpy(), 12 * n),
                    "horizon": np.tile(np.repeat(np.arange(1, 13), n), len(dates)),
                    "county_fips": np.tile(counties.county_fips, len(dates) * 12),
                    "node_id": np.tile(counties.node_id, len(dates) * 12),
                })
                frame["target_date"] = (pd.PeriodIndex(frame.origin_date, freq="M")
                                         + frame.horizon.to_numpy()).to_timestamp()
                index = pd.MultiIndex.from_frame(frame[["target_date", "county_fips", "node_id"]])
                frame["y_true"] = lookup.reindex(index).to_numpy()
                if len(params) != len(frame) or frame.y_true.isna().any():
                    raise ValueError(f"Incomplete forecast grid: {short}/{split}")
                for field in ("p_occ", "mu", "phi"):
                    frame[field] = params[field].to_numpy()
                frame["e_y"] = frame.p_occ * frame.mu
                target = ROOT / directory / f"predictions_{split}.parquet"
                frame.to_parquet(target, index=False, compression="zstd")
                print(f"Restored {short}/{split}: {len(frame):,} forecast requests", flush=True)


def recompute(skip_permutations=False):
    import numpy as np

    report = module("forecast_report", "tutorial-writeup/workspace/prepare_forecast_results.py")
    summary_path = SITE / "assets/forecast/summary.json"
    summary = json.loads(summary_path.read_text())
    graph = np.load(ROOT / "output/data/county_graph.npz")["edge_index"]
    weights = report.row_standardized_graph(graph, 3108)
    operators = report.stacf_operators(graph, 3108)
    for split in SPLITS:
        canonical, lookup = report.read_canonical(split, [])
        for model, spec in report.MODELS.items():
            frame = report.read_predictions(model, spec, split, canonical, lookup,
                                             __import__("pandas").Timestamp("2003-01-01"), [])
            actual = report.summarize(frame, weights,
                                      operators if split == "train" and not skip_permutations else None,
                                      include_discrimination=split in report.DISCRIMINATION_FILES)
            expected = summary["results"][model][split]
            for key in ("mean_nll", "mean_crps", "mae_expected_fraction", "brier", "residual_mean", "residual_sd"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-12,
                                           err_msg=f"{model}/{split}/{key}")
            if skip_permutations and split == "train":
                actual["stacf"] = expected["stacf"]
            elif split == "train":
                compare_nested(actual["stacf"], expected["stacf"], f"{model}/STACF")
            summary["results"][model][split] = actual
            print(f"Recomputed and verified {model}/{split}", flush=True)
    report.write_report(summary, publish=True)
    module("validation_maps", "tutorial-writeup/workspace/prepare_validation_maps.py").build_maps(geometry=False)
    subprocess.run([sys.executable, "tutorial-writeup/workspace/mechanism_anim.py"], check=True)


def compare_nested(actual, expected, label):
    import numpy as np

    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Different fields: {label}")
        for key in expected:
            compare_nested(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different lengths: {label}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_nested(left, right, f"{label}/{i}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, err_msg=label)
    elif actual != expected:
        raise ValueError(f"Different value: {label}")


def neural_date_precision():
    """Keep NumPy date dictionary keys consistent across supported Pandas/Arrow versions."""
    import model.data as data
    original = data._load_full_panel

    def load():
        full, ranges = original()
        full["date"] = full.date.dt.as_unit("us")
        return full, ranges

    data._load_full_panel = load


def refit(short, device):
    directory = ROOT / MODELS[short]
    cfg = json.loads((directory / "config.json").read_text())
    target = ROOT / "refits" / short
    target.mkdir(parents=True, exist_ok=False)
    if short == "gnn":
        neural_date_precision()
        from model.config import Config
        from model.train import load_panel, train_model
        config = Config(**cfg)
        config.device = device
        train_model(config, load_panel(config), target, eval_validation=False)
    else:
        from model.xgb.features import build_dataset
        from model.xgb.train import fit
        dataset = build_dataset(cfg["lookback"], cfg["horizon"], forecast_safe=True,
                                max_spatial=cfg["max_spatial"], max_temporal=cfg["max_temporal"],
                                n_harmonics=cfg["n_harmonics"], want_splits=("train", "test"))
        params = {**cfg["params"], "device": device}
        fitted, best, _ = fit(dataset, params, num_boost_round=cfg["num_boost_round"],
                              early_stopping_rounds=cfg["early_stopping_rounds"],
                              stabilization=cfg["stabilization"], c0_cap=cfg["c0_cap"])
        fitted.save_model(str(target / "model.pkl"))
        (target / "config.json").write_text(json.dumps({**cfg, "params": params, "best_iteration": best}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("unpack", "report", "render", "predict-gnn", "refit-xgb", "refit-gnn"))
    parser.add_argument("--downloads", type=Path, default=ROOT.parent)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--split", default="validation", choices=SPLITS)
    parser.add_argument("--reuse-stacf", action="store_true",
                        help="reuse supplied STACF permutation summaries; all other diagnostics are recomputed")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "unpack":
        verify_files()
        unpack(args.downloads)
    elif args.action == "report":
        recompute(skip_permutations=args.reuse_stacf)
    elif args.action == "render":
        subprocess.run([sys.executable, "tutorial-writeup/build_release.py"], check=True)
    elif args.action == "predict-gnn":
        neural_date_precision()
        sys.argv = ["model.predict", "--ckpt-dir", MODELS["gnn"], "--out-dir", "regenerated/gnn",
                    "--device", args.device, "--n-samples", "0", "--split", args.split]
        runpy.run_module("model.predict", run_name="__main__")
    else:
        refit(args.action.split("-", 1)[1], args.device)


if __name__ == "__main__":
    main()
