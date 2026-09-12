"""
Join combined_YYYY.csv shards onto the county×month spine.
All sources are already joined in EE; local work is:
  - derive target columns from burned_m2 / mapped_m2
  - convert pct_* m² sums to fractions
  - validate exact keys before joining; response gaps are never imputed
"""
import os
import pandas as pd
import numpy as np
import json
from glob import glob
from config import *
from model.preprocessing import validate_shards, validate_export_keys, validate_outcomes

RAW  = f"{OUTPUT_DIR}/raw"
DATA = f"{OUTPUT_DIR}/data"

# ── load all combined shards ──────────────────────────────────────────────────

files = sorted(glob(f"{RAW}/combined_*.csv"))
if not files:
    raise FileNotFoundError(f"No combined_*.csv found in {RAW}/. "
                            "Download them from Drive first.")
validate_shards(files, range(START_YEAR, END_YEAR + 1))
os.makedirs(DATA, exist_ok=True)
with open(f"{DATA}/node_index.json") as fh:
    node_index = json.load(fh)
print(f"Loading {len(files)} combined shards...")
raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
raw['county_fips'] = raw['county_fips'].astype(str).str.zfill(5)
for key in ['year', 'month']:
    values = pd.to_numeric(raw[key], errors='raise')
    if not (np.isfinite(values).all() and np.equal(values, np.floor(values)).all()):
        raise ValueError(f'Non-integer or missing export {key}')
    raw[key] = values.astype(int)
validate_export_keys(raw, node_index.keys(), range(START_YEAR, END_YEAR + 1))
print(f"  Validated raw shape: {raw.shape}")

# Drop EE internal columns not needed downstream
raw = raw.drop(columns=[c for c in ['system:index', '.geo'] if c in raw.columns])

# ── derive target columns ─────────────────────────────────────────────────────

raw['burned_m2'] = pd.to_numeric(raw['burned_m2'], errors='raise')
raw['mapped_m2'] = pd.to_numeric(raw['mapped_m2'], errors='raise')
area_m2 = pd.to_numeric(raw['county_area_km2'], errors='raise') * 1e6
if not (np.isfinite(area_m2).all() and (area_m2 > 0).all()
        and np.isfinite(raw['burned_m2']).all()):
    raise ValueError('Missing or invalid burned area/county area; targets cannot be imputed')

raw['burned_fraction'] = raw['burned_m2'] / area_m2
raw['fire_occurred']   = (raw['burned_m2'] / 1e6 > FIRE_FLOOR_KM2).astype(int)
validate_outcomes(raw)
# The legacy mapped-area field equals burned area and cannot establish coverage.
# A future corrected export must decode QA bit 1, retaining shortened-period bit 2;
# Uncertainty is burn-date uncertainty and is not a valid-coverage mask.
raw['frac_valid_burn'] = np.clip(raw['mapped_m2'] / area_m2, 0, 1)
raw = raw.drop(columns=['burned_m2', 'mapped_m2'])

# ── convert lc area sums → fractions ─────────────────────────────────────────

for nm in IGBP_GROUPS:
    col = 'pct_' + nm
    if col in raw.columns:
        raw[col] = np.clip(raw[col] / area_m2, 0, 1)

# forward-fill lc per county (MCD12Q1 lags ~1-2 years)
lc_cols = ['pct_' + nm for nm in IGBP_GROUPS if 'pct_'+nm in raw.columns] + \
          (['lc_dominant'] if 'lc_dominant' in raw.columns else [])
raw = raw.sort_values(['county_fips','year','month'])
lc_impute = raw[lc_cols].isna()
raw[lc_cols] = raw.groupby('county_fips')[lc_cols].ffill()
raw['lc_impute_flag'] = lc_impute.any(axis=1).astype(int)

# ── join onto spine ───────────────────────────────────────────────────────────

with open(f"{DATA}/node_index.json") as fh:
    node_index = json.load(fh)

if not os.path.exists(f"{DATA}/spine.parquet"):
    print("spine.parquet missing — rebuilding from node_index.json + config date range")
    fips_list = sorted(node_index.keys())
    months = pd.date_range(f'{START_YEAR}-01-01', f'{END_YEAR}-12-01', freq='MS')
    spine = (pd.MultiIndex.from_product([fips_list, months], names=['county_fips', 'date'])
             .to_frame(index=False))
    spine['year']  = spine['date'].dt.year
    spine['month'] = spine['date'].dt.month
    spine['county_fips'] = spine['county_fips'].astype(str).str.zfill(5)
    spine.to_parquet(f"{DATA}/spine.parquet", index=False)
    print(f"  Saved spine.parquet: {spine.shape}")
else:
    spine = pd.read_parquet(f"{DATA}/spine.parquet")

spine['county_fips'] = spine['county_fips'].astype(str).str.zfill(5)
validate_export_keys(spine, node_index.keys(), range(START_YEAR, END_YEAR + 1))
df = spine.merge(raw, on=['county_fips','year','month'], how='left', validate='one_to_one')
df['node_id'] = df['county_fips'].map(node_index)
df['date'] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
validate_outcomes(df)

df.to_parquet(f"{DATA}/master.parquet", index=False)
print(f"Master table: {df.shape}")
print(f"  fire_occurred null:    {df['fire_occurred'].isna().sum()}")
print(f"  burned_fraction null:  {df['burned_fraction'].isna().sum()}")
print(f"  ndvi_lag1 null:        {df['ndvi_lag1'].isna().sum() if 'ndvi_lag1' in df.columns else 'col missing'}")
