"""Data-quality flags, imputation, chronological split, train-only scaler, write outputs."""
import pandas as pd
import numpy as np
import json
from config import *
from model.preprocessing import forward_fill_predictors, validate_outcomes

DATA = f"{OUTPUT_DIR}/data"

df = pd.read_parquet(f"{DATA}/master.parquet")

# Missing outcomes must be excluded under an explicit observation rule or fail;
# absence of an export is not evidence of zero burned area.
validate_outcomes(df)
df['fire_occurred'] = df['fire_occurred'].astype(int)
df['lc_impute_flag']  = df['lc_impute_flag'].fillna(0).astype(int)

# ── identify predictor columns ────────────────────────────────────────────────

TARGET_COLS = ['fire_occurred','burned_fraction']
KEY_COLS    = ['county_fips','date','year','month','node_id','county_area_km2']
FLAG_COLS   = ['frac_valid_burn','frac_valid_obs','impute_flag','lc_impute_flag']

exclude = set(TARGET_COLS + KEY_COLS + FLAG_COLS)
pred_cols = [c for c in df.columns if c not in exclude]

# ── quality flags ─────────────────────────────────────────────────────────────

df['frac_valid_obs'] = df[pred_cols].notna().mean(axis=1)
df['impute_flag']    = (df[pred_cols].isna().any(axis=1) | (df['lc_impute_flag'] == 1)).astype(int)

# Each predictor donor must precede its receiving row. This is stronger than
# checking only the dates of missing rows and makes future-to-train filling impossible.
df = forward_fill_predictors(df, pred_cols)

# ── chronological split ───────────────────────────────────────────────────────

mlist = sorted(df['date'].unique())
n = len(mlist)
n_tr = round(SPLIT[0] * n)
n_te = round(SPLIT[1] * n)
n_va = n - n_tr - n_te

train_m = mlist[:n_tr]
test_m  = mlist[n_tr: n_tr + n_te]
val_m   = mlist[n_tr + n_te:]

print(f"Train:  {pd.Timestamp(train_m[0]).date()} → {pd.Timestamp(train_m[-1]).date()}  ({n_tr} months)")
print(f"Test:   {pd.Timestamp(test_m[0]).date()} → {pd.Timestamp(test_m[-1]).date()}  ({n_te} months)")
print(f"Val:    {pd.Timestamp(val_m[0]).date()} → {pd.Timestamp(val_m[-1]).date()}  ({n_va} months)")

train = df[df['date'].isin(train_m)].copy()
test  = df[df['date'].isin(test_m)].copy()
val   = df[df['date'].isin(val_m)].copy()

# ── train-only scaler ─────────────────────────────────────────────────────────

SCALE_COLS = [c for c in pred_cols
              if df[c].dtype.kind in 'fi' and c != 'lc_dominant']

mu = train[SCALE_COLS].mean()
sd = train[SCALE_COLS].std().replace(0, 1)

for part in (train, test, val):
    part[SCALE_COLS] = (part[SCALE_COLS] - mu) / sd

# ── write outputs ─────────────────────────────────────────────────────────────

for name, part in [('train', train), ('test', test), ('validation', val)]:
    out = part.copy()
    out['county_fips'] = out['county_fips'].astype('string')
    out.to_parquet(f"{DATA}/{name}.parquet", index=False)
    print(f"Wrote {name}.parquet  shape={out.shape}  "
          f"fire_rate={out.fire_occurred.mean():.3f}")

pd.DataFrame({'feature': SCALE_COLS, 'mean': mu.values, 'std': sd.values}).to_json(
    f"{DATA}/feature_scaler.json", orient='records', indent=2)

# ── feature metadata ──────────────────────────────────────────────────────────

def meta(col):
    if col in ['ndvi_lag1','evi_lag1']:
        return {'group':'vegetation','source':'MOD13Q1','cadence':'t-1','scaled':True,'role':'predictor'}
    if col.startswith('pct_') or col == 'lc_dominant':
        return {'group':'landcover','source':'MCD12Q1','cadence':'annual','scaled':col!='lc_dominant','role':'predictor'}
    if col in ['elev','slope','aspect_sin','aspect_cos']:
        return {'group':'terrain','source':'3DEP','cadence':'static','scaled':True,'role':'predictor'}
    if col == 'evc_mean':
        return {'group':'fuel','source':'LANDFIRE','cadence':'static','scaled':True,'role':'predictor'}
    if col in ['pop_density','built_frac']:
        return {'group':'human','source':'GPW/GHSL','cadence':'annual','scaled':True,'role':'predictor'}
    if col == 'snow_frac':
        return {'group':'snow','source':'MOD10A1','cadence':'t','scaled':True,'role':'predictor'}
    if col in ['aet','water_deficit','pet','swe']:
        return {'group':'waterbalance','source':'TerraClimate','cadence':'t','scaled':True,'role':'predictor'}
    if col in ['days_erc_p90','days_vpd_p90','days_hdw_p90','days_dry','max_dry_run']:
        return {'group':'weather','source':'GRIDMET','cadence':'t','scaled':True,'role':'predictor'}
    if col in DROUGHT_BANDS or col == 'soil_moisture':
        return {'group':'drought','source':'GRIDMET/TerraClimate','cadence':'t','scaled':True,'role':'predictor'}
    if any(col.startswith(b) for b in GRIDMET_SUM + GRIDMET_MAX + GRIDMET_MEAN) or col.startswith('wind'):
        return {'group':'weather','source':'GRIDMET','cadence':'t','scaled':True,'role':'predictor'}
    return {'group':'unknown','source':'unknown','cadence':'unknown','scaled':True,'role':'predictor'}

feat_meta = {c: meta(c) for c in pred_cols}
with open(f"{DATA}/feature_metadata.json", "w") as fh:
    json.dump(feat_meta, fh, indent=2)

print("\nAll outputs written.")
