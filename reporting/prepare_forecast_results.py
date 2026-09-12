#!/usr/bin/env python3
"""Validate and summarize only the newly selected restricted forecast fits.

Default output is workspace/forecast-results. --publish writes approved diagnostic
assets and replaces marked result blocks in the tutorial chapters. Neither
mode fits models, reselects on validation, or computes prediction intervals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="forecast-mpl-"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="forecast-fonts-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.special import betaln
from scipy.stats import beta, norm
import sklearn
from sklearn.metrics import (average_precision_score, auc as curve_auc, precision_recall_curve,
                             roc_auc_score, roc_curve)
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.diagnostics import residual_cube, _gram, _stacf, _perm_p
from model.spatial import exclusive_orders

SITE = ROOT / "tutorial-writeup/final-product"
EXPERIMENT = "forecast_safe_20260907"
PROTOCOL = ROOT / "output" / EXPERIMENT / "protocol.json"
AUDIT = PROTOCOL.with_name("input_audit.json")
MODELS = {
    "xgb_forecast_safe_20260907": {"label": "XGBoostLSS", "directory": "output/xgb_forecast_safe_20260907",
            "study": "forecast_safe_20260907_xgb", "tune_root": "output/xgb/tune",
            "checkpoint": "model.pkl", "page": "xgboostlss.qmd", "color": "#ad6100"},
    "gnn_forecast_safe_20260907": {"label": "GCN → LSTM", "directory": "output/model/forecast_safe_20260907",
            "study": "forecast_safe_20260907_gnn", "tune_root": "output/model/tune",
            "checkpoint": "best.pt", "page": "pytorch.qmd", "color": "#17634e"},
}
SPLITS = ("train", "test", "validation")
KEYS = ["origin_date", "horizon", "target_date", "county_fips", "node_id"]
PARAMETERS = ["y_true", "p_occ", "mu", "phi", "e_y"]
TERRAIN = ["aspect_cos", "aspect_sin", "elev", "slope"]
PIT_SEED = 0
QQ_PROBS = np.linspace(0.001, 0.999, 999)
STACF_MAX_LAG = 24
STACF_MAX_ORDER = 3
STACF_N_PERM = 199
STACF_PERM_SEED = 271828
STACF_FILES = {"xgb_forecast_safe_20260907": "xgb-train-G1_stacf.png",
               "gnn_forecast_safe_20260907": "gnn-train-G1_stacf.png"}
DISCRIMINATION_FILES = {"test": "test-discrimination.png", "validation": "validation-discrimination.png"}
CURVE_PLOT_BINS = 2048
CODE_INPUTS = (
    "model/config.py", "model/data.py", "model/model.py", "model/train.py",
    "model/tune.py", "model/predict.py", "model/zib.py", "model/spatial.py",
    "model/freeze_forecast.py",
    "model/test_forecast_safe.py", "model/xgb/features.py", "model/xgb/train.py",
    "model/xgb/tune.py", "model/xgb/predict.py", "model/xgb/common.py",
    "model/xgb/zabeta_hurdle.py", "model/xgb/test_forecast_safe.py",
    "model/diagnostics.py", "calibration.py",
    "tutorial-writeup/workspace/test_forecast_report_diagnostics.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def provenance(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked input: {path}")
    require(path.resolve().is_relative_to(ROOT), f"input outside repository: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def load_json(path, sources):
    sources.append(provenance(path))
    return json.loads(path.read_text())


def checked_protocol(sources):
    protocol, audit = (load_json(path, sources) for path in (PROTOCOL, AUDIT))
    require(protocol["experiment"] == audit["experiment"] == EXPERIMENT, "wrong experiment")
    require(protocol["forecast_safe"] is True and protocol["lookback"] == 48
            and protocol["horizon"] == 12, "wrong forecast protocol")
    require(protocol["terrain_columns"] == TERRAIN, "unexpected terrain inputs")
    require(protocol["prediction_intervals"] is False, "intervals outside edition scope")
    require(audit["raw_outcomes_match_splits"] is True and
            audit["terrain_needs_imputation"] is False, "input audit did not pass")
    for relative, expected in audit["sources_sha256"].items():
        source = provenance(ROOT / relative)
        require(source["sha256"] == expected, f"input audit is stale: {relative}")
        sources.append(source)
    return protocol, audit


def study_snapshot(spec, sources, protocol):
    database = ROOT / spec["tune_root"] / "study.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        row = conn.execute("SELECT study_id FROM studies WHERE study_name=?",
                           (spec["study"],)).fetchone()
        require(row is not None, f"missing study {spec['study']}")
        study_id = row[0]
        trials = [dict(trial=number, state=state, test_nll=value)
                  for number, state, value in conn.execute(
                      "SELECT t.number,t.state,v.value FROM trials t LEFT JOIN trial_values v "
                      "ON t.trial_id=v.trial_id AND v.objective=0 "
                      "WHERE t.study_id=? ORDER BY t.number", (study_id,))]
        attributes = {key: json.loads(value) for key, value in conn.execute(
            "SELECT key,value_json FROM study_user_attributes WHERE study_id=?", (study_id,))}
        forbidden = [key for (key,) in conn.execute(
            "SELECT a.key FROM trial_user_attributes a JOIN trials t ON t.trial_id=a.trial_id "
            "WHERE t.study_id=?", (study_id,)) if key.startswith(("val_", "validation"))]
    require(not forbidden, f"validation metrics entered tuning study {spec['study']}: {forbidden}")
    require(len(trials) >= protocol["initial_trial_budget_per_model"], "initial trial budget incomplete")
    require(all(t["state"] not in ("RUNNING", "WAITING") for t in trials), "tuning is still active")
    done = [t for t in trials if t["state"] == "COMPLETE" and
            t["test_nll"] is not None and np.isfinite(t["test_nll"])]
    require(done, "no complete finite trials")
    safe = attributes.get("forecast_safe", attributes.get("arm_spec", {}).get("forecast_safe"))
    require(safe is True, "study was not marked forecast_safe")
    best = min(done, key=lambda t: (t["test_nll"], t["trial"]))
    board = load_json(ROOT / spec["tune_root"] / spec["study"] / "leaderboard.json", sources)
    require(len(board) == len(done), "leaderboard does not cover all completed trials")
    expected = {t["trial"]: t["test_nll"] for t in done}
    require({t["trial"] for t in board} == set(expected), "leaderboard trial IDs do not match study")
    require(all(t["trial"] in expected and np.isclose(t["test_nll"], expected[t["trial"]],
                atol=1e-12, rtol=0) for t in board), "leaderboard differs from study")
    sources.append(provenance(database))
    wal = database.with_name(database.name + "-wal")
    if wal.is_file():
        sources.append(provenance(wal))
    return {"study": spec["study"], "attributes": attributes, "trials": trials,
            "complete_trials": len(done), "selected_trial": best["trial"],
            "selected_test_nll": best["test_nll"], "selection_split": "test",
            "no_validation_trial_attributes": True}


def checked_model(model, spec, sources, protocol):
    directory = ROOT / spec["directory"]
    config = load_json(directory / "config.json", sources)
    require(config.get("forecast_safe") is True, f"{model}: forecast_safe not serialized")
    require(config["lookback"] == 48 and config["horizon"] == 12, f"{model}: wrong windows")
    selection = study_snapshot(spec, sources, protocol)
    checkpoint = provenance(directory / spec["checkpoint"])
    sources.append(checkpoint)
    if spec["checkpoint"] == "model.pkl":
        require(config["raw_cov_names"] == TERRAIN, "unexpected XGBoost covariates")
        require(config["lc_categories"] == [0], "informative land-cover categories")
        require(config["reduction"] == "none" and config["n_nbr_pcs"] == 0 and
                not config["climatology"] and not config["clim_oof"], "unapproved XGBoost features")
        require(config["study"] == spec["study"] and config["selection_split"] == "test" and
                config["trial"] == selection["selected_trial"], "XGBoost selection mismatch")
        require(np.isclose(config["test_nll"], selection["selected_test_nll"], atol=1e-10, rtol=0),
                "XGBoost selection score mismatch")
        require(config["num_boost_round"] <= protocol["xgboost"]["round_cap"], "round budget changed")
        record = load_json(directory / "provenance.json", sources)
        require(record["experiment"] == EXPERIMENT and record["study"] == spec["study"]
                and record["selection_split"] == "test"
                and record["selected_trial"] == selection["selected_trial"], "XGBoost provenance selection mismatch")
        require(record["model_sha256"] == checkpoint["sha256"]
                and record["config_sha256"] == provenance(directory / "config.json")["sha256"]
                and record["prediction_intervals"] is False, "XGBoost provenance artifact mismatch")
        signed_inputs = dict(record["source_sha256"])
        for item in [record["forecast_protocol"], record["input_audit"], record["runtime_environment"],
                     *record["predictions"].values()]:
            signed_inputs[item["path"]] = item["sha256"]
        for relative, expected in signed_inputs.items():
            source = provenance(ROOT / relative)
            require(source["sha256"] == expected, f"XGBoost provenance source changed: {relative}")
            sources.append(source)
        selection["record"] = record
    else:
        require(config["n_pca"] is None and not config["static_bypass"] and
                not config["teacher_forcing"] and config["lc_embed_dim"] == 0,
                "unapproved neural feature or feedback mode")
        require(config["max_epochs"] <= protocol["gnn"]["epoch_cap"], "epoch budget changed")
        study_dir = ROOT / spec["tune_root"] / spec["study"]
        best_config = load_json(study_dir / "best_config.json", sources)
        require(config == best_config, "neural fit config differs from selected trial")
        trial_dir = study_dir / f"trial_{selection['selected_trial']:03d}"
        original = provenance(trial_dir / "best.pt")
        sources.append(original)
        require(checkpoint["sha256"] == original["sha256"], "neural checkpoint differs from selected trial")
        selection["record"] = load_json(directory / "selection.json", sources)
        record = selection["record"]
        require(record["study"] == spec["study"] and record["selection_split"] == "test"
                and record["trial"] == selection["selected_trial"], "neural selection record mismatch")
        require(np.isclose(record["test_nll"], selection["selected_test_nll"], atol=1e-12, rtol=0),
                "neural freeze record score mismatch")
        require(record["checkpoint_sha256"] == checkpoint["sha256"]
                and record["config_sha256"] == provenance(directory / "config.json")["sha256"]
                and record["protocol_sha256"] == provenance(PROTOCOL)["sha256"],
                "neural freeze record artifact hash mismatch")
        require(record["prediction_intervals"] is False, "unexpected neural prediction intervals")
        for relative, expected in record["sources_sha256"].items():
            source = provenance(ROOT / relative)
            require(source["sha256"] == expected, f"neural freeze source changed: {relative}")
            sources.append(source)
        sources.append(provenance(directory / "training_history.json"))
    environment_path = directory / "runtime_environment.json"
    environment = load_json(environment_path, sources) if environment_path.is_file() else None
    selection["record"] = {key: value for key, value in selection["record"].items()
                           if key != "post_launch_changes"}
    return {**spec, "config": config, "selection": selection, "checkpoint": checkpoint,
            "environment": environment}


def read_canonical(split, sources):
    path = ROOT / "output/data" / f"{split}.parquet"
    sources.append(provenance(path))
    df = pd.read_parquet(path, columns=["date", "county_fips", "node_id", "burned_fraction"])
    df["date"] = pd.to_datetime(df["date"])
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    require(not df.duplicated(["date", "county_fips"]).any(), f"duplicate canonical {split} keys")
    require(not df.isna().any().any(), f"missing canonical {split} values")
    require(df.burned_fraction.between(0, 1, inclusive="left").all(), "invalid canonical response")
    require(df.groupby("county_fips").node_id.nunique().eq(1).all(), "county/node mapping changes")
    require(df.groupby("node_id").county_fips.nunique().eq(1).all(), "node IDs are not unique to counties")
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    require(dates.equals(pd.date_range(dates.min(), dates.max(), freq="MS")), "missing panel months")
    require(len(df) == len(dates) * df.county_fips.nunique(), "incomplete canonical county/month grid")
    return df, df.set_index(["date", "county_fips", "node_id"]).burned_fraction.astype(np.float32)


def read_predictions(model, spec, split, canonical, lookup, first_month, sources):
    path = ROOT / spec["directory"] / f"predictions_{split}.parquet"
    sources.append(provenance(path))
    df = pd.read_parquet(path)
    require(set(df.columns) == set(KEYS + PARAMETERS), f"{model}/{split}: unexpected output schema")
    df = df[KEYS + PARAMETERS]
    require(not df.isna().any().any(), f"{model}/{split}: missing values")
    for column in ("origin_date", "target_date"):
        df[column] = pd.to_datetime(df[column])
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    for column in ("horizon", "node_id"):
        require(np.equal(df[column], np.floor(df[column])).all(), f"noninteger {column}")
        df[column] = df[column].astype("int64")
    require(not df.duplicated(KEYS).any(), f"{model}/{split}: duplicate full keys")
    require(not df.duplicated(["origin_date", "horizon", "county_fips"]).any(), "duplicate requests")
    df = df.sort_values(KEYS).reset_index(drop=True)
    month_offset = (df.target_date.dt.year - df.origin_date.dt.year) * 12 + (
        df.target_date.dt.month - df.origin_date.dt.month)
    require(np.array_equal(month_offset, df.horizon), "target/horizon mismatch")
    require(df.origin_date.dt.day.eq(1).all() and df.target_date.dt.day.eq(1).all(), "non-month-start date")
    require(df.horizon.between(1, 12).all(), "invalid horizon")
    lo = max(first_month + pd.DateOffset(months=47), canonical.date.min() - pd.DateOffset(months=1))
    hi = canonical.date.max() - pd.DateOffset(months=12)
    expected_origins = pd.date_range(lo, hi, freq="MS")
    require(pd.DatetimeIndex(sorted(df.origin_date.unique())).equals(expected_origins), "missing or extra origins")
    require(df.groupby("origin_date").size().eq(12 * canonical.county_fips.nunique()).all(),
            "incomplete origin/horizon/county grid")
    index = pd.MultiIndex.from_frame(df[["target_date", "county_fips", "node_id"]])
    truth = lookup.reindex(index).to_numpy()
    require(np.isfinite(truth).all() and np.array_equal(df.y_true.to_numpy(), truth),
            f"{model}/{split}: targets differ from canonical split")
    for column in PARAMETERS:
        df[column] = df[column].astype("float64")
        require(np.isfinite(df[column]).all(), f"invalid {column}")
    require(((df.p_occ > 0) & (df.p_occ < 1)).all(), "invalid occurrence probability")
    require(((df.mu > 0) & (df.mu < 1) & (df.phi > 0)).all(), "invalid Beta parameters")
    require(np.allclose(df.e_y, df.p_occ * df.mu, rtol=2e-6, atol=1e-12), "e_y != p_occ * mu")
    return df


def gate_calibration(p, occurrence):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 21)))
    bins = np.digitize(p, edges[1:-1])
    count = np.bincount(bins)
    valid = count > 0
    predicted = np.bincount(bins, weights=p)[valid] / count[valid]
    observed = np.bincount(bins, weights=occurrence)[valid] / count[valid]
    return {"predicted": predicted.tolist(), "observed": observed.tolist(), "n": count[valid].tolist()}


def distribution_scores(y, p, mu, phi):
    positive = y > 0
    a, b = mu * phi, (1 - mu) * phi
    nll = -np.log1p(-p)
    nll[positive] = -np.log(p[positive]) - beta.logpdf(y[positive], a[positive], b[positive])
    f, f1 = beta.cdf(y, a, b), beta.cdf(y, a + 1, b)
    expected_error = (1 - p) * y + p * (y * (2 * f - 1) + mu * (1 - 2 * f1))
    beta_gini = 4 * np.exp(betaln(2 * a, 2 * b) - 2 * betaln(a, b)) / (a + b)
    crps = expected_error - p * (1 - p) * mu - 0.5 * p * p * beta_gini
    require(np.isfinite(nll).all() and np.isfinite(crps).all(), "nonfinite scores")
    require((crps >= -1e-12).all(), "negative CRPS beyond numerical precision")
    return nll, crps, f


def row_standardized_graph(edge_index, n_nodes):
    """Undirected binary adjacency, no diagonal, unit row sums except islands."""
    edge = np.asarray(edge_index)
    require(edge.ndim == 2 and edge.shape[0] == 2, "invalid graph edge shape")
    require(np.isfinite(edge).all() and np.equal(edge, np.floor(edge)).all(), "noninteger graph node")
    require((edge >= 0).all() and (edge < n_nodes).all(), "graph node outside county index")
    edge = edge.astype(np.int64)
    adjacency = sparse.csr_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])),
                                  shape=(n_nodes, n_nodes))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    return (sparse.diags(inverse) @ adjacency).tocsr()


def moran_i(values, weights):
    """N/S0 * z'Wz / z'z; all counties enter centering, including zero-row islands.

    Return None if variance or total edge weight is zero. This is a descriptive
    coefficient; no independence reference distribution or p-value is attached.
    """
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and weights.shape == (len(values), len(values)), "Moran dimension mismatch")
    require(np.isfinite(values).all(), "nonfinite county residual means")
    total_weight = float(weights.sum())
    if len(values) < 2 or np.ptp(values) == 0 or total_weight == 0:
        return None
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        return None
    return float(len(values) / total_weight * (centered @ (weights @ centered)) / denominator)


def lag1_correlation(values):
    """Pearson correlation of consecutive monthly means; None when undefined."""
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "invalid monthly residual means")
    if len(values) < 3 or np.ptp(values[:-1]) == 0 or np.ptp(values[1:]) == 0:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def stacf_operators(edge_index, n_nodes, max_order=STACF_MAX_ORDER):
    """Reuse the fitted project's exclusive graph-distance shells, without fitting."""
    require(isinstance(max_order, int) and 0 <= max_order < n_nodes, "invalid STACF spatial order")
    first_order = row_standardized_graph(edge_index, n_nodes)  # validates all edge indices
    # The existing constructor uses dense reachability products before returning
    # sparse operators. Bound BLAS threads to avoid oversubscription on the host.
    with threadpool_limits(limits=4, user_api="blas"):
        operators = exclusive_orders(edge_index, n_nodes, max_order, sparse=True)
    used = sparse.csr_matrix((n_nodes, n_nodes))
    for order, operator in enumerate(operators):
        require(operator.shape == (n_nodes, n_nodes) and np.isfinite(operator.data).all(),
                "invalid STACF graph operator")
        sums = np.asarray(operator.sum(axis=1)).ravel()
        require(np.all(np.isclose(sums, 0) | np.isclose(sums, 1)), "STACF rows are not normalized")
        require(used.multiply(operator).nnz == 0, "STACF graph shells overlap")
        if order == 0:
            require((operator - sparse.eye(n_nodes)).nnz == 0, "STACF order zero must be identity")
        used = used + operator
    if max_order >= 1:
        difference = operators[1] - first_order
        require(not difference.nnz or np.max(np.abs(difference.data)) < 1e-14,
                "STACF first-order graph differs from the reported adjacency")
    return operators


def stacf_permutations(n_months, n_nodes, n_perm, seed):
    """Common Monte Carlo draws, reused across models and both centered fields."""
    require(isinstance(n_perm, int) and n_perm >= 2, "STACF permutations must be at least two")
    rng = np.random.default_rng(seed)
    months = np.stack([rng.permutation(n_months) for _ in range(n_perm)])
    counties = np.stack([rng.permutation(n_nodes) for _ in range(n_perm)])
    return months, counties


def matrix_values(matrix):
    return [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]


def stacf_pvalues(rho, null_month, null_county):
    """Legacy centered, two-sided Monte Carlo p-values plus the specified routing.

    Undefined correlations and the deterministic self-correlation have no test.
    The maximum of two null-specific p-values is not a gridwise adjustment.
    """
    values = {}
    for key, null in (("month", null_month), ("county", null_county)):
        require(null.ndim == 3 and null.shape[1:] == rho.shape and len(null) >= 2,
                "invalid STACF permutation array")
        p = _perm_p(rho, null)
        p[~(np.isfinite(rho) & np.isfinite(null).all(axis=0))] = np.nan
        p[0, 0] = np.nan
        values[f"p_{key}"] = p
        values[f"null_mean_{key}"] = null.mean(axis=0)
        values[f"null_sd_{key}"] = null.std(axis=0, ddof=1)
    display = np.maximum(values["p_month"], values["p_county"])
    display[0, 1:] = values["p_month"][0, 1:]
    display[1:, 0] = values["p_county"][1:, 0]
    display[0, 0] = np.nan
    values["p_display"] = display
    return {key: matrix_values(value) for key, value in values.items()}


def stacf_nulls(field, operators, valid, max_lag, month_permutations, county_permutations,
                progress_label=None):
    """Exact original month reindexing and county relabeling; no approximation."""
    gram, energy = _gram(field, operators)
    rho, pairs = _stacf(gram, energy, valid, max_lag)
    month_null = np.stack([_stacf(gram[:, :, order][:, :, :, order], energy[:, :, order],
                                  valid[:, order], max_lag)[0] for order in month_permutations])
    county_null = np.empty((len(county_permutations), len(operators), max_lag + 1))
    for i, order in enumerate(county_permutations):
        county_null[i] = _stacf(*_gram(field[:, :, order], operators), valid, max_lag)[0]
        if progress_label and ((i + 1) % 50 == 0 or i + 1 == len(county_permutations)):
            print(f"STACF {progress_label}: {i + 1}/{len(county_permutations)} county permutations", flush=True)
    return rho, pairs, month_null, county_null


def observed_stacf(df, residual, operators, max_lag=STACF_MAX_LAG, n_perm=0,
                   permutation_seed=STACF_PERM_SEED):
    """Pooled-horizon rho and optional exact month/county permutation references."""
    require(len(operators) > 0, "missing STACF graph operators")
    residual = np.asarray(residual)
    require(residual.ndim == 1 and len(df) == len(residual) and np.isfinite(residual).all(),
            "invalid STACF residuals")
    require(np.equal(df.node_id, np.floor(df.node_id)).all() and
            df.node_id.between(0, operators[0].shape[0] - 1).all(), "STACF county index out of bounds")
    keys = ["horizon", "target_date", "node_id"]
    require(not df.duplicated(keys).any(), "duplicate STACF horizon/month/county")
    frame = df[keys].copy()
    frame["r"] = residual
    cube, months, horizons = residual_cube(frame, operators[0].shape[0])
    require(isinstance(max_lag, int) and 0 <= max_lag < len(months), "invalid STACF temporal lag")
    month_index = pd.DatetimeIndex(months)
    require(month_index.equals(pd.date_range(month_index.min(), month_index.max(), freq="MS")),
            "STACF target months must be consecutive")
    present = np.isfinite(cube)
    valid = present.any(axis=2)
    require(np.array_equal(valid, present.all(axis=2)), "ragged STACF horizon/month county slice")
    require(present.any(), "empty STACF residual cube")
    raw = np.where(present, cube - cube[present].mean(), 0.0)
    # Missing horizon/month slices contain only zeros and remain excluded by valid.
    demeaned = raw - raw.mean(axis=2, keepdims=True)
    permutations = None
    if n_perm:
        permutations = stacf_permutations(len(months), cube.shape[2], n_perm, permutation_seed)
    fields = {}
    for key, field in (("raw", raw), ("county_anomaly", demeaned)):
        with threadpool_limits(limits=4, user_api="blas"), np.errstate(divide="ignore", invalid="ignore"):
            if permutations is None:
                rho, pairs = _stacf(*_gram(field, operators), valid, max_lag)
                pvalues = {}
            else:
                rho, pairs, null_month, null_county = stacf_nulls(
                    field, operators, valid, max_lag, *permutations,
                    progress_label=key if n_perm >= 100 else None)
                pvalues = stacf_pvalues(rho, null_month, null_county)
        require(not np.isinf(rho).any(), "infinite STACF coefficient")
        fields[key] = {"rho": matrix_values(rho), **pvalues}
    return {"split": "train", "rows": len(df), "n_months": len(months),
            "n_horizons": len(horizons), "n_counties": cube.shape[2],
            "max_lag": max_lag, "max_order": len(operators) - 1,
            "horizons": horizons.astype(int).tolist(),
            "target_month_min": str(month_index.min().date()),
            "target_month_max": str(month_index.max().date()),
            "valid_months_per_horizon": valid.sum(axis=1).astype(int).tolist(),
            "n_horizon_month_pairs": pairs.tolist(),
            "n_perm": n_perm, "permutation_seed": permutation_seed if n_perm else None,
            "p_value_resolution": 1 / (n_perm + 1) if n_perm else None,
            "permutation_sha256": ({key: hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
                                    for key, indices in zip(("month", "county"), permutations)}
                                   if permutations is not None else None),
            "mean_shell_size": [float(operator.getnnz(axis=1).mean()) for operator in operators],
            **fields}


def sample_plot_curve(x, y, bins=CURVE_PLOT_BINS):
    """Keep endpoints and each x-bin's first/last/min/max y, in original order.

    This only reduces the points drawn and serialized. AUC/AP are calculated
    from the complete, tie-aware curves before this helper is called.
    """
    x, y = np.asarray(x), np.asarray(y)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0, "invalid plot curve")
    require(np.isfinite(x).all() and np.isfinite(y).all() and ((x >= 0) & (x <= 1)).all(),
            "invalid curve coordinates")
    require(np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0), "curve x must be monotonic")
    require(isinstance(bins, int) and bins > 0, "invalid curve sampling bins")
    groups = np.floor(x * bins).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    ends = np.r_[starts[1:], len(x)]
    selected = [0, len(x) - 1]
    for start, end in zip(starts, ends):
        selected.extend((start, end - 1, start + np.argmin(y[start:end]), start + np.argmax(y[start:end])))
    indices = np.unique(selected)
    return x[indices].tolist(), y[indices].tolist()


def discrimination(occurrence, p, include_curves):
    precision, recall, _ = precision_recall_curve(occurrence, p)
    result = {"gate_auc": float(roc_auc_score(occurrence, p)),
              "gate_average_precision": float(average_precision_score(occurrence, p)),
              "gate_pr_auc": float(curve_auc(recall, precision))}
    if include_curves:
        fpr, tpr, _ = roc_curve(occurrence, p, drop_intermediate=False)
        require(np.isclose(curve_auc(fpr, tpr), result["gate_auc"], atol=1e-12, rtol=0),
                "ROC curve area differs from ROC AUC")
        sampled_fpr, sampled_tpr = sample_plot_curve(fpr, tpr)
        sampled_recall, sampled_precision = sample_plot_curve(recall, precision)
        result["discrimination_curves"] = {
            "roc": {"fpr": sampled_fpr, "tpr": sampled_tpr, "full_points": len(fpr)},
            "precision_recall": {"recall": sampled_recall, "precision": sampled_precision,
                                 "full_points": len(recall)},
            "plot_sampling": {"x_bins": CURVE_PLOT_BINS, "metrics_use_full_curves": True,
                              "method": "Retain endpoints and first/last/min/max y in each x bin, in original order"},
        }
    return result


def summarize(df, weights, stacf_weights=None, include_discrimination=False):
    y, p, mu, phi = (df[column].to_numpy() for column in ("y_true", "p_occ", "mu", "phi"))
    positive = y > 0
    nll, crps, f = distribution_scores(y, p, mu, phi)
    pit = np.random.default_rng(PIT_SEED).random(len(df)) * (1 - p)
    pit[positive] = (1 - p[positive]) + p[positive] * f[positive]
    residual = norm.ppf(np.clip(pit, 1e-12, 1 - 1e-12))
    occurrence = positive.astype(float)
    months = pd.DataFrame({"date": df.target_date, "residual": residual,
                           "nll": nll, "crps": crps}).groupby("date").mean()
    county_means = pd.DataFrame({"node_id": df.node_id, "residual": residual}).groupby(
        "node_id").residual.mean().reindex(np.arange(weights.shape[0])).to_numpy()
    result = {
        "rows": len(df), "positive_rows": int(positive.sum()), "counties": int(df.county_fips.nunique()),
        "origins": int(df.origin_date.nunique()), "target_months": len(months),
        "origin_min": str(df.origin_date.min().date()), "origin_max": str(df.origin_date.max().date()),
        "target_min": str(df.target_date.min().date()), "target_max": str(df.target_date.max().date()),
        "mean_nll": float(nll.mean()), "mean_crps": float(crps.mean()),
        "mae_expected_fraction": float(np.abs(p * mu - y).mean()),
        "mae_positive_mean": float(np.abs(mu[positive] - y[positive]).mean()),
        "mean_expected_fraction": float((p * mu).mean()), "mean_occurrence_probability": float(p.mean()),
        "mean_positive_mu": float(mu[positive].mean()), "mean_observed_fraction": float(y.mean()),
        "occurrence_rate": float(occurrence.mean()), "brier": float(np.mean((p - occurrence) ** 2)),
        **discrimination(occurrence, p, include_discrimination),
        "pit_mean": float(pit.mean()), "pit_sd": float(pit.std(ddof=1)),
        "residual_mean": float(residual.mean()), "residual_sd": float(residual.std(ddof=1)),
        "moran_i_county_mean_residual": moran_i(county_means, weights),
        "lag1_monthly_mean_residual": lag1_correlation(months.residual.to_numpy()),
        "qq_theoretical": norm.ppf(QQ_PROBS).tolist(), "qq_observed": np.quantile(residual, QQ_PROBS).tolist(),
        "gate_calibration": gate_calibration(p, occurrence),
        "monthly": {"date": [str(date.date()) for date in months.index],
                    **{column: months[column].tolist() for column in months}},
        "parameter_ranges": {column: [float(df[column].min()), float(df[column].max())]
                             for column in ("p_occ", "mu", "phi", "e_y")},
        "checks": {"full_keys_unique": True, "forecast_requests_unique": True,
                   "canonical_float32_outcomes_exact": True, "target_dates_match_horizons": True,
                   "complete_forecast_grid": True, "finite_valid_parameters": True,
                   "expected_fraction_matches_parameters": True},
    }
    if stacf_weights is not None:
        result["stacf"] = observed_stacf(df, residual, stacf_weights, n_perm=STACF_N_PERM)
    return result


def draw_figure(results, splits, destination):
    fig, axes = plt.subplots(3 * len(splits), 1, figsize=(8.4, 10.5 * len(splits) + .7), squeeze=False)
    for row, split in enumerate(splits):
        qq, gate, temporal = axes[3 * row:3 * row + 3, 0]
        qq.plot([-3.1, 3.1], [-3.1, 3.1], color="0.4", linestyle="--", linewidth=1)
        gate.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
        temporal.axhline(0, color="0.4", linestyle="--", linewidth=1)
        for model, spec in MODELS.items():
            data = results[model][split]
            style = {"color": spec["color"], "label": spec["label"], "linewidth": 1.5}
            qq.plot(data["qq_theoretical"], data["qq_observed"], **style)
            gate.plot(data["gate_calibration"]["predicted"], data["gate_calibration"]["observed"],
                      marker="o", markersize=3, **style)
            temporal.plot(pd.to_datetime(data["monthly"]["date"]), data["monthly"]["residual"], **style)
        title = {"train": "Train · fitted data", "test": "Test · model development",
                 "validation": "Validation · later assessment"}[split]
        qq.set_title(f"{title}\nRandomized residual QQ", fontsize=10)
        qq.set_xlabel("Standard normal quantile"); qq.set_ylabel("Residual quantile")
        gate.set_title(f"{title}\nOccurrence reliability", fontsize=10)
        gate.set_xlabel("Mean predicted probability"); gate.set_ylabel("Observed positive proportion")
        gate.set_xlim(0, 1); gate.set_ylim(0, 1)
        temporal.set_title(f"{title}\nMonthly residual means", fontsize=10)
        temporal.set_xlabel("Target month"); temporal.set_ylabel("Mean randomized residual")
        temporal.tick_params(axis="x", rotation=25)
        qq.legend(loc="upper left", frameon=False, fontsize=9)
        gate.legend(loc="upper left", frameon=False, fontsize=9)
        temporal.legend(loc="upper left", frameon=False, fontsize=9)
        for ax in (qq, gate, temporal):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.5)
    fig.text(0.5, 0.012, "Single-fit plug-in distributions; PIT randomization seed 0.\n"
             "Matched forecast requests; descriptive curves without confidence bands.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035 if len(splits) > 1 else 0.055, 1, 1))
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def stacf_color_limit(results):
    values = []
    for model in MODELS:
        for field in ("raw", "county_anomaly"):
            rho = np.array(results[model]["train"]["stacf"][field]["rho"], dtype=float)
            rho[0, 0] = np.nan
            values.extend(np.abs(rho[np.isfinite(rho)]).tolist())
    # A zero field still needs a valid, explicitly shared diverging color scale.
    return max(values + [1e-12])


def draw_train_stacf(result, label, color_limit, destination):
    info = result["train"]["stacf"]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.8), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dedede")
    for ax, field, title in zip(axes, ("raw", "county_anomaly"),
                                ("Globally centered residuals", "Monthly cross-county mean removed")):
        rho = np.array(info[field]["rho"], dtype=float)
        shown = np.ma.masked_invalid(rho)
        shown.mask = np.ma.getmaskarray(shown)
        shown.mask[0, 0] = True
        heatmap = ax.imshow(shown, cmap=cmap, vmin=-color_limit, vmax=color_limit,
                            aspect="auto", interpolation="nearest")
        pvalues = np.array(info[field]["p_display"], dtype=float)
        for order in range(info["max_order"] + 1):
            for lag in range(info["max_lag"] + 1):
                value = pvalues[order, lag]
                text = f"{value:.2f}" if np.isfinite(value) else "—"
                color = "white" if np.isfinite(rho[order, lag]) and abs(rho[order, lag]) > .6 * color_limit else "#222222"
                if order == lag == 0:
                    color = "#444444"
                ax.text(lag, order, text, ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks(np.arange(info["max_lag"] + 1))
        ax.set_yticks(np.arange(info["max_order"] + 1))
        ax.set_xticks(np.arange(-.5, info["max_lag"] + 1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, info["max_order"] + 1, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.55, alpha=.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Temporal lag k (months)")
        ax.set_ylabel("Exclusive spatial order l")
        ax.set_title(title, fontsize=10)
    fig.colorbar(heatmap, ax=axes, fraction=0.025, pad=0.02,
                 label="Residual autocorrelation, rho(l,k)")
    fig.suptitle(f"{label} · Train · Space–time residual autocorrelation", fontsize=13)
    fig.supxlabel(f"Cell numbers: two-sided permutation p-values ({info['n_perm']} draws; "
                  f"seed {info['permutation_seed']}; minimum p = {info['p_value_resolution']:.3f}), rounded to two decimals.\n"
                  "Colors: residual autocorrelation on a shared scale.\n"
                  "The masked (0,0) cell is fixed at 1 and has no statistical test.",
                  fontsize=8)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def draw_discrimination(results, split, destination):
    fig, (roc, pr) = plt.subplots(2, 1, figsize=(8.4, 9.4), constrained_layout=True)
    prevalence = None
    for model, spec in MODELS.items():
        data = results[model][split]
        curves = data["discrimination_curves"]
        if prevalence is None:
            prevalence = data["occurrence_rate"]
        require(data["occurrence_rate"] == prevalence, "discrimination plots need matched outcomes")
        roc.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], color=spec["color"], linewidth=1.7,
                 label=f"{spec['label']} · ROC AUC {data['gate_auc']:.4f}")
        pr.plot(curves["precision_recall"]["recall"], curves["precision_recall"]["precision"],
                color=spec["color"], linewidth=1.7,
                label=f"{spec['label']} · PR AUC {data['gate_pr_auc']:.6f}; AP {data['gate_average_precision']:.6f}")
    roc.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1, label="Chance discrimination")
    pr.axhline(prevalence, color="0.45", linestyle="--", linewidth=1,
               label=f"Prevalence {prevalence:.4f}")
    roc.set(xlabel="False positive rate", ylabel="True positive rate", title="Occurrence ROC")
    pr.set(xlabel="Recall", ylabel="Precision", title="Occurrence precision–recall")
    roc.legend(loc="lower right", frameon=True, fontsize=8.5)
    pr.legend(loc="upper right", frameon=True, fontsize=8.5)
    for ax in (roc, pr):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)
    split_label = "Test · model development" if split == "test" else "Validation · later assessment"
    fig.suptitle(f"{split_label} · Occurrence discrimination", fontsize=13)
    fig.supxlabel("Metrics use full tie-aware curves; displayed curves are sampled.\n"
                  "PR AUC is trapezoidal area; AP uses recall-step weighting.", fontsize=8)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def model_block(model, summary):
    spec = summary["model_specs"][model]
    selection, config = spec["selection"], spec["config"]
    lines = [f"The completed search contains {len(selection['trials'])} trials, of which "
             f"{selection['complete_trials']} completed with finite test NLL. "
             f"Trial {selection['selected_trial']} supplied the selected fit."]
    if spec["checkpoint"]["path"].endswith("model.pkl"):
        detail = (f"The fit retains {config['best_iteration'] + 1} boosting rounds, "
                  f"{config['max_spatial']} spatial shell(s), {config['max_temporal']} "
                  f"additional temporal lag(s), and {config['n_harmonics']} calendar harmonic pair(s).")
    else:
        detail = (f"The selected network uses {config['gcn_layers']} graph layer(s) of width "
                  f"{config['gcn_hidden']}, {config['lstm_layers']} LSTM layer(s) of width "
                  f"{config['lstm_hidden']}, and {config['n_harmonics']} calendar harmonic pair(s).")
    return "\n".join(lines + ["", detail, "", "The [result manifest](assets/forecast/summary.json) "
                             "records the complete configuration, selection evidence, and source hashes."])


def score_table(summary, splits):
    lines = ["| Split | Model | Forecast requests | NLL | CRPS | Expected-fraction MAE | Occurrence Brier |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for split in splits:
        for model, spec in MODELS.items():
            data = summary["results"][model][split]
            lines.append(f"| {split.title()} | {spec['label']} | {data['rows']:,} | {data['mean_nll']:.6f} | "
                         f"{data['mean_crps']:.7f} | {data['mae_expected_fraction']:.7f} | {data['brier']:.6f} |")
    return "\n".join(lines)


def residual_table(summary, splits):
    lines = ["Residual summaries use the same randomized residuals as the plots. "
              "Undefined correlations are shown as an em dash.", "",
              "| Split | Model | Residual mean | Residual SD | County-mean Moran I | Monthly lag-1 r |",
              "|---|---|---:|---:|---:|---:|"]
    for split in splits:
        for model, spec in MODELS.items():
            data = summary["results"][model][split]
            values = [data[key] for key in ("residual_mean", "residual_sd",
                      "moran_i_county_mean_residual", "lag1_monthly_mean_residual")]
            formatted = " | ".join("—" if value is None else f"{value:.3f}" for value in values)
            lines.append(f"| {split.title()} | {spec['label']} | {formatted} |")
    return "\n".join(lines)


def comparison_block(summary):
    return "\n\n".join([
        "Train and test comparisons use identical full forecast keys and outcomes for both models.",
        score_table(summary, ("train", "test")), residual_table(summary, ("train", "test")),
        "![Train and test diagnostics for the selected forecast examples. Panels are stacked vertically, "
        "with matching forecast requests in each split.](assets/forecast/train-test-diagnostics.png){#fig-forecast-development}",
    ])


def validation_block(summary):
    return "\n\n".join([
        "Validation assesses the two fixed forecast examples on matching forecast requests.",
        score_table(summary, ("validation",)), residual_table(summary, ("validation",)),
        "![Later-period validation diagnostics for the two fixed forecast examples. The vertically stacked "
        "panels describe the selected fits and do not choose a winner.]"
        "(assets/forecast/validation-diagnostics.png){#fig-forecast-validation}",
    ])


def replace_block(path, kind, body):
    text = path.read_text()
    begin, end = f"<!-- BEGIN FORECAST {kind} RESULTS -->", f"<!-- END FORECAST {kind} RESULTS -->"
    require(text.count(begin) == text.count(end) == 1, f"missing/duplicate result marker: {path}")
    return path, re.sub(re.escape(begin) + r".*?" + re.escape(end),
                        lambda _: f"{begin}\n{body}\n{end}", text, flags=re.DOTALL)


def write_report(summary, publish=False):
    """Render assets and chapter blocks from an already verified numerical summary."""
    results = summary["results"]
    result_blocks = {spec["page"]: model_block(model, summary) for model, spec in MODELS.items()}
    result_blocks["comparison.qmd"] = comparison_block(summary)
    result_blocks["discussion.qmd"] = validation_block(summary)
    summary["result_blocks_sha256"] = {
        name: hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
        for name, body in result_blocks.items()
    }
    kinds = {"comparison.qmd": "COMPARISON", "discussion.qmd": "VALIDATION"}
    pages = [replace_block(SITE / name, kinds.get(name, "MODEL"), body)
             for name, body in result_blocks.items()]
    destination = SITE / "assets/forecast" if publish else Path(__file__).parent / "forecast-results"
    destination.mkdir(parents=True, exist_ok=True)
    draw_figure(results, ("train", "test"), destination / "train-test-diagnostics.png")
    draw_figure(results, ("validation",), destination / "validation-diagnostics.png")
    for model, filename in STACF_FILES.items():
        draw_train_stacf(results[model], MODELS[model]["label"],
                         summary["stacf_settings"]["color_limits"][1], destination / filename)
    for split, filename in DISCRIMINATION_FILES.items():
        draw_discrimination(results, split, destination / filename)
    summary["assets_sha256"] = {
        f"assets/forecast/{name}": hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in ("train-test-diagnostics.png", "validation-diagnostics.png", *STACF_FILES.values(),
                     *DISCRIMINATION_FILES.values())
    }
    summary["plot_layout"] = {"columns": 1, "general_panels_per_split": 3,
                              "stacf_panels_per_model": 2, "discrimination_panels_per_split": 2}
    summary["rendered_at_utc"] = datetime.now(timezone.utc).isoformat()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if publish:
        for path, text in pages:
            path.write_text(text)
    print(f"Prepared checked forecast evidence in {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="write checked public assets and result blocks")
    args = parser.parse_args()
    sources = []
    protocol, audit = checked_protocol(sources)
    specs = {model: checked_model(model, spec, sources, protocol) for model, spec in MODELS.items()}
    results, matched = {model: {} for model in MODELS}, {}
    first_month, previous_end, county_map, weights, stacf_weights = None, None, None, None, None
    for split in SPLITS:
        canonical, lookup = read_canonical(split, sources)
        if first_month is None:
            first_month = canonical.date.min()
            largest = canonical.loc[canonical.burned_fraction.idxmax()]
            training_response = {"maximum": float(largest.burned_fraction),
                                 "county_fips": largest.county_fips,
                                 "target_date": str(largest.date.date())}
        require(previous_end is None or canonical.date.min() == previous_end + pd.DateOffset(months=1),
                "split target periods overlap or are not consecutive")
        previous_end = canonical.date.max()
        current_map = canonical[["county_fips", "node_id"]].drop_duplicates().sort_values("county_fips").reset_index(drop=True)
        if county_map is None:
            county_map = current_map
            require(np.array_equal(np.sort(county_map.node_id), np.arange(len(county_map))),
                    "county node IDs must be contiguous and start at zero")
            graph_path = ROOT / "output/data/county_graph.npz"
            sources.append(provenance(graph_path))
            with np.load(graph_path) as graph:
                weights = row_standardized_graph(graph["edge_index"], len(county_map))
                stacf_weights = stacf_operators(graph["edge_index"], len(county_map))
        require(current_map.equals(county_map), "county/node mapping differs across splits")
        reference = None
        for model, spec in specs.items():
            df = read_predictions(model, spec, split, canonical, lookup, first_month, sources)
            current = df[KEYS + ["y_true"]]
            if reference is None:
                reference = current.copy()
            else:
                require(current.equals(reference), f"{split}: models have different forecast keys or targets")
            results[model][split] = summarize(df, weights, stacf_weights if split == "train" else None,
                                               include_discrimination=split in DISCRIMINATION_FILES)
            if split == "test":
                require(np.isclose(results[model][split]["mean_nll"],
                                   spec["selection"]["selected_test_nll"], atol=2e-6, rtol=0),
                        f"{model}: saved test predictions do not reproduce selected trial NLL")
            print(f"Checked {model}/{split}: {len(df):,} rows", flush=True)
            del df
        matched[split] = {"exact_full_keys_and_outcomes": True, "rows": len(reference)}
    for relative in CODE_INPUTS:
        sources.append(provenance(ROOT / relative))
    sources.append(provenance(Path(__file__).resolve()))
    # Recheck every input before emitting artifacts; a running fit cannot silently replace a source.
    sources = list({source["path"]: source for source in sources}.values())
    for source in sources:
        require(provenance(ROOT / source["path"]) == source, f"source changed during scoring: {source['path']}")
    stacf_limit = stacf_color_limit(results)
    for null in ("month", "county"):
        require(len({results[model]["train"]["stacf"]["permutation_sha256"][null] for model in MODELS}) == 1,
                "models did not use identical STACF permutations")
    summary = {
        "schema_version": 1, "experiment": EXPERIMENT,
        "prediction_intervals": False,
        "training_response": training_response,
        "protocol_sha256": provenance(PROTOCOL)["sha256"],
        "input_audit_sha256": provenance(AUDIT)["sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Descriptive diagnostics for new forecast fits under the recorded restricted information set.",
        "predictive_object": "Single-fit p_occ, mu, phi and e_y; neural deterministic mean-feedback path.",
        "protocol": {key: value for key, value in protocol.items()
                     if key not in {"p_value_interpretation", "validation_interpretation"}},
        "input_audit": audit, "model_specs": specs,
        "stacf_settings": {
            "split": "train", "max_lag": STACF_MAX_LAG, "max_order": STACF_MAX_ORDER,
            "pit_seed": PIT_SEED, "exploratory": True, "permutations": STACF_N_PERM,
            "permutation_seed": STACF_PERM_SEED, "p_value_resolution": 1 / (STACF_N_PERM + 1),
            "shared_permutations_across_models_and_fields": True,
            "null_schemes": {
                "month": "Shuffle whole target-month positions, moving every horizon together; reindex Gram matrices, energies and valid-slice mask",
                "county": "Permute county labels jointly across all horizons and target months, preserving each series and the fixed graph",
            },
            "p_value_formula": "(1 + count(abs(null - mean(null)) >= abs(observed - mean(null)))) / (n_perm + 1)",
            "display_rule": "p_month at order 0/positive lag; p_county at positive order/lag 0; max(p_month,p_county) when both are positive",
            "multiplicity_adjustment": "none; the max-of-two display rule is not a gridwise adjustment",
            "cell_text": "p_display rounded to two decimals; structural (0,0) and undefined tests use an em dash",
            "fields": {"raw": "Globally centered randomized quantile residuals",
                       "county_anomaly": "Cross-county mean removed within each horizon and target month"},
            "pooling": "Products and energies are computed separately within each horizon, then pooled over valid horizon/month pairs",
            "normalization": "Each shell uses its full valid-slice mean squared energy, as in model.diagnostics._stacf",
            "graph": "Row-standardized exclusive graph-distance shells; order zero is identity",
            "color_limits": [-stacf_limit, stacf_limit], "masked_cells": [[0, 0]],
            "undefined_coefficients": "null in data; grey in plots",
            "implementation": ["model.diagnostics.residual_cube", "model.diagnostics._gram",
                               "model.diagnostics._stacf", "model.diagnostics._perm_p",
                               "model.spatial.exclusive_orders"],
        },
        "discrimination_settings": {
            "splits": list(DISCRIMINATION_FILES), "descriptive_only": True,
            "response": "burned_fraction > 0", "score": "p_occ",
            "roc_auc": "Trapezoidal ROC area, equal to the tie-aware probability ranking statistic",
            "pr_auc": "Trapezoidal area under the full precision-recall curve, reported as gate_pr_auc",
            "average_precision": "Recall-step-weighted precision, reported separately as gate_average_precision",
            "plot_sampling": "First/last/min/max y retained per 1/2048 x bin, including endpoints; metrics use full curves",
            "no_threshold_selection": True,
        },
        "spatial_weights": {
            "source": "output/data/county_graph.npz", "symmetrized_binary_adjacency": True,
            "self_loops": False, "normalization": "row standardized; isolated counties have zero rows",
            "nodes": weights.shape[0], "nonzero_weights": weights.nnz,
            "isolated_counties": int(np.count_nonzero(np.asarray(weights.sum(axis=1)).ravel() == 0)),
            "moran_formula": "N / sum(W) * (z.T @ W @ z) / (z.T @ z), z centered over all county means",
            "zero_variance_or_no_edges": "coefficient is undefined and stored as null",
        },
        "notes": [
            "All three splits use matching forecast keys and observed targets across models.",
            "Test selects configurations and checkpoints. Validation assesses the fixed examples.",
            "Later rolling origins may use responses observed since earlier forecasts; no target after the current origin is an input.",
            "Positive float32 targets are promoted to float64 and scored without an epsilon response clamp.",
            "NLL and CRPS are plug-in marginal diagnostics, not estimates of parameter or trajectory uncertainty.",
            "Occurrence AUC/AP use scikit-learn threshold grouping, including ties.",
            "Trapezoidal PR AUC is recorded separately from average precision; curve downsampling affects plotting only.",
            "PIT seed 0 is applied after full-key sorting; only the normal transform clips probabilities to [1e-12,1-1e-12].",
            "Repeated targets and spatial/temporal dependence make diagnostic curves descriptive; no row-IID tests or bands.",
            "Spatial Moran I describes county-mean randomized residuals; temporal lag-1 r correlates consecutive target-month residual means.",
            "Unequal temporal averaging lengths affect county means and their Moran I, so cross-split changes do not establish stronger dependence.",
            "Training STACF preserves horizons and adds centered two-sided month/county permutation p-values; cells use decimal p-values without stars or thresholds.",
            "No prediction quantiles, interval coverage, parameter ensembles, or definitive class winner are reported.",
        ],
        "versions_role": "Diagnostic recomputation and figure rendering only; model_specs.environment records fitting environments.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "pit_seed": PIT_SEED, "matched_evaluation_rows": matched, "sources": sources, "results": results,
    }
    write_report(summary, publish=args.publish)


if __name__ == "__main__":
    main()
