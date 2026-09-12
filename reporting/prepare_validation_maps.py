#!/usr/bin/env python3
"""Create synchronized validation choropleth inputs from matched forecast requests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "tutorial-writeup/final-product"
DESTINATION = SITE / "assets/forecast/maps"
MODEL_DIRS = ("output/xgb_forecast_safe_20260907", "output/model/forecast_safe_20260907")


def county_paths(destination, fips):
    """Project and simplify boundaries once; subsequent builds reuse these display paths."""
    import geopandas as gpd

    counties = gpd.read_parquet(ROOT / "output/data/county_geom.parquet")
    counties["county_fips"] = counties.county_fips.astype(str).str.zfill(5)
    counties = counties.set_index("county_fips").loc[fips].to_crs("EPSG:5070")
    geom = counties.geometry.simplify(500, preserve_topology=True)
    xmin, ymin, xmax, ymax = geom.total_bounds
    width, height = 600, 370
    scale = min((width - 12) / (xmax - xmin), (height - 12) / (ymax - ymin))
    xoff = (width - (xmax - xmin) * scale) / 2
    yoff = (height - (ymax - ymin) * scale) / 2
    paths = []
    def polygons(shape):
        if shape.geom_type == "Polygon":
            yield shape
        elif hasattr(shape, "geoms"):
            for part in shape.geoms:
                yield from polygons(part)

    for shape in geom:
        parts = polygons(shape)
        rings = []
        for part in parts:
            for ring in [part.exterior, *part.interiors]:
                points = np.asarray(ring.coords)
                coords = np.column_stack(((points[:, 0] - xmin) * scale + xoff,
                                          (ymax - points[:, 1]) * scale + yoff))
                rings.append("M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in coords) + "Z")
        assert rings, "county has no polygon geometry"
        paths.append("".join(rings))
    data = {"fips": fips, "paths": paths, "width": width, "height": height,
            "projection": "EPSG:5070", "simplification_metres": 500,
            "source": "US Census TIGER/Line 2018 county boundaries",
            "source_url": "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/"}
    (destination / "counties.json").write_text(json.dumps(data, separators=(",", ":")))


def build_maps(destination=DESTINATION, geometry=True):
    destination.mkdir(parents=True, exist_ok=True)
    keys = ["origin_date", "horizon", "target_date", "county_fips", "node_id"]
    frames = []
    for directory in MODEL_DIRS:
        frame = pd.read_parquet(ROOT / directory / "predictions_validation.parquet",
                                columns=keys + ["y_true", "p_occ", "mu"])
        frame["county_fips"] = frame.county_fips.astype(str).str.zfill(5)
        frame = frame.sort_values(keys).reset_index(drop=True)
        assert not frame.duplicated(keys).any()
        assert np.isfinite(frame[["y_true", "p_occ", "mu"]]).all().all()
        assert frame[["y_true", "p_occ", "mu"]].ge(0).all().all()
        assert frame[["y_true", "p_occ", "mu"]].le(1).all().all()
        frames.append(frame)
    xgb, gnn = frames
    pd.testing.assert_frame_equal(xgb[keys + ["y_true"]], gnn[keys + ["y_true"]], check_dtype=False, check_exact=True)
    fips = sorted(xgb.county_fips.unique().tolist())
    assert len(fips) == 3108 and len(xgb) == 1566432
    observed = pd.read_parquet(ROOT / "output/data/validation.parquet",
                               columns=["date", "county_fips", "burned_fraction"])
    lookup = observed.set_index(["date", "county_fips"]).burned_fraction
    actual = lookup.reindex(pd.MultiIndex.from_frame(xgb[["target_date", "county_fips"]]))
    np.testing.assert_array_equal(actual.to_numpy(dtype=np.float32), xgb.y_true.to_numpy(dtype=np.float32))
    if geometry:
        county_paths(destination, fips)
    else:
        assert json.loads((destination / "counties.json").read_text())["fips"] == fips
    horizons = []
    for horizon in range(1, 13):
        left, right = [frame.loc[frame.horizon.eq(horizon)].sort_values(
            ["target_date", "county_fips"]) for frame in frames]
        dates = pd.DatetimeIndex(left.target_date.unique())
        origins = pd.DatetimeIndex(left.origin_date.unique())
        assert len(dates) == len(origins) == 42
        assert dates.equals(pd.date_range(dates.min(), periods=42, freq="MS"))
        assert left.groupby("target_date").county_fips.nunique().eq(len(fips)).all()
        assert np.array_equal(left.county_fips.to_numpy().reshape(42, -1),
                              np.tile(fips, (42, 1)))
        values = np.stack([left.p_occ, left.mu, left.y_true, right.p_occ, right.mu], axis=1)
        values = values.reshape(len(dates), len(fips), 5).transpose(0, 2, 1).astype("<f4")
        filename = f"horizon-{horizon:02d}.bin"
        raw = values.tobytes()
        (destination / filename).write_bytes(raw)
        horizons.append({"horizon": horizon, "file": filename,
                         "dates": dates.strftime("%Y-%m").tolist(),
                         "origins": origins.strftime("%Y-%m").tolist(),
                         "sha256": hashlib.sha256(raw).hexdigest()})
    metadata = {"dtype": "little-endian float32", "shape": [42, 5, len(fips)],
                "fields": ["xgb_p_occ", "xgb_mu", "observed_fraction", "gnn_p_occ", "gnn_mu"],
                "horizons": horizons, "observed_presence": "observed_fraction > 0",
                "geometry": "counties.json",
                "geometry_sha256": hashlib.sha256((destination / "counties.json").read_bytes()).hexdigest(),
                "magnitude_scale": "log1p(value / 0.00001) / log1p(1 / 0.00001)",
                "probability_scale": [0, 1], "magnitude_scale_bounds": [0, 1],
                "display_precision": "float32; scoring uses the original prediction precision"}
    (destination / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Validation maps: 12 horizons × 42 months × {len(fips):,} counties in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-geometry", action="store_true")
    args = parser.parse_args()
    build_maps(geometry=not args.reuse_geometry)
