"""Pure, fail-closed checks used before chronological panel preparation."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def validate_shards(files, years):
    found = []
    for path in files:
        match = re.fullmatch(r"combined_(\d{4})\.csv", Path(path).name)
        if match is None:
            raise ValueError(f"Unexpected combined-shard name: {path}")
        found.append(int(match.group(1)))
    expected = set(years)
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError(f"Incomplete/duplicate shard set: missing={sorted(expected - set(found))}, "
                         f"unexpected={sorted(set(found) - expected)}")


def validate_export_keys(raw, county_fips, years):
    keys = ["county_fips", "year", "month"]
    if raw[keys].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError("Missing or duplicate export county-month key")
    expected = pd.MultiIndex.from_product(
        [sorted(county_fips), list(years), range(1, 13)], names=keys)
    actual = pd.MultiIndex.from_frame(raw[keys])
    missing, unexpected = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"Incomplete export keys: {len(missing)} missing, {len(unexpected)} unexpected")


def validate_outcomes(frame):
    """Absent observations are not zero burns; don't impute or silently clip them."""
    y = pd.to_numeric(frame["burned_fraction"], errors="raise").to_numpy(dtype=float)
    occurrence = pd.to_numeric(frame["fire_occurred"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not ((y >= 0) & (y < 1)).all():
        raise ValueError("Missing, nonfinite, or out-of-support burned_fraction; response imputation is forbidden")
    if not np.isfinite(occurrence).all() or not np.array_equal(occurrence, (y > 0).astype(float)):
        raise ValueError("fire_occurred must equal the observed burned_fraction > 0 indicator")


def forward_fill_predictors(frame, columns):
    """Use earlier donors only, leaving no implicit backward-fill fallback.

    This is causal in observation time. Source publication availability is a
    separate requirement; the forecast feature allowlist removes products with
    retrospective annual/epoch reconstruction from the published experiment.
    """
    if {"burned_fraction", "fire_occurred"} & set(columns):
        raise ValueError("Response columns cannot be passed to predictor imputation")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values(["county_fips", "date"])
    if out[["county_fips", "date"]].isna().any().any() or out.duplicated(["county_fips", "date"]).any():
        raise ValueError("Predictor imputation requires unique, observed county/date keys")
    out[columns] = out.groupby("county_fips", sort=False)[columns].ffill()
    missing = out[columns].isna()
    if missing.any().any():
        unresolved = missing.columns[missing.any()].tolist()
        raise ValueError(f"No earlier predictor donor for {unresolved}; backward fill is forbidden")
    return out.reset_index(drop=True)
