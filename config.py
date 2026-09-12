import numpy as np
from pathlib import Path

EE_PROJECT   = "mythic-lead-448717-p4"
OUTPUT_DIR   = str(Path(__file__).resolve().parent / "output")
DRIVE_FOLDER = "fire-dat"

START_YEAR, END_YEAR = 2003, 2024
EQ_AREA = 'EPSG:5070'
NON_CONUS_FIPS = ['02','15','60','66','69','72','78']

FIRE_FLOOR_KM2 = 0.0

SPLIT = (0.70, 0.10, 0.20)

A_COUNTIES  = 'TIGER/2018/Counties'
A_BURN      = 'MODIS/061/MCD64A1'
A_GRIDMET   = 'IDAHO_EPSCOR/GRIDMET'
A_DROUGHT   = 'GRIDMET/DROUGHT'
A_TERRACLIM = 'IDAHO_EPSCOR/TERRACLIMATE'
A_VEG       = 'MODIS/061/MOD13Q1'
A_LANDCOVER = 'MODIS/061/MCD12Q1'
A_TERRAIN   = 'USGS/3DEP/10m_collection'
A_SNOW      = 'MODIS/061/MOD10A1'                 # daily NDSI snow cover
A_GPW       = 'CIESIN/GPWv411/GPW_Population_Density'   # 5-yr epochs 2000–2020
A_GHSL      = 'JRC/GHSL/P2023A/GHS_BUILT_S'       # 5-yr epochs 1975–2030, 100 m
A_EVC       = 'LANDFIRE/Vegetation/EVC/v1_4_0'    # existing vegetation cover

SCALE_BURN    = 463
SCALE_GRIDMET = 4638
SCALE_VEG     = 250
SCALE_LC      = 463
SCALE_TERRAIN = 270
SCALE_SNOW    = 500
SCALE_HUMAN   = 1000    # GPW native ~928 m
SCALE_GHSL    = 100     # GHSL built_surface native
SCALE_FUEL    = 240     # LANDFIRE 30 m coarsened for tractable county means

GRIDMET_SUM  = ['pr']
GRIDMET_MAX  = ['tmmx','vpd','erc','bi']
GRIDMET_MEAN = ['tmmn','rmax','rmin','srad','fm100','fm1000']

# Extreme fire-weather day counts are relative to a per-pixel 90th-percentile
# climatology built by export_climatology.py over the TRAIN years only (no
# test/validation weather informs the threshold → leakage-safe).
CLIMO_ASSET       = f"projects/{EE_PROJECT}/assets/fireweather_p90"
CLIMO_TRAIN_YEARS = (2003, 2017)   # whole train-split years
DRY_DAY_MM        = 1.0            # a day with < 1 mm precip is "dry"

DROUGHT_BANDS = ['pdsi','z','spi90d','spi180d','spei90d','spei180d','eddi90d',
                 'spi1y','spi2y','spei1y','spei2y','eddi1y']

# TerraClimate water-balance bands beyond soil (band → unpacking scale factor).
# 'def' (climatic water deficit) is renamed water_deficit in the export.
TERRACLIM_EXTRA = {'aet': 0.1, 'def': 0.1, 'pet': 0.1, 'swe': 1.0}

IGBP_GROUPS = {
    'forest': [1,2,3,4,5],
    'shrub':  [6,7],
    'grass':  [8,9,10],
    'crop':   [12,14],
    'urban':  [13],
}

# LANDFIRE EVC v1.4.0 class value → canopy-cover midpoint (%). Only vegetation
# cover classes (100–172) are remapped; developed/barren/ag land is set to 0
# cover and open water / snow-ice is masked (see evc_img in export_combined.py).
EVC_REMAP_FROM = [100,
                  101,102,103,104,105,106,107,108,109,
                  111,112,113,114,115,116,117,118,119,
                  121,122,123,124,125,126,127,128,129,
                  150, 151, 152, 153, 161, 162, 163, 171, 172]
EVC_REMAP_TO   = [5,
                  15,25,35,45,55,65,75,85,95,
                  15,25,35,45,55,65,75,85,95,
                  15,25,35,45,55,65,75,85,95,
                  5,  17.5, 42.5, 80, 17.5, 42.5, 80, 35, 80]
