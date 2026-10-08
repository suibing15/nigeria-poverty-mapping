"""
Nigeria Poverty Mapping - Step: Extract satellite + agricultural features for RWI points
Run this from your project ROOT folder (C:\\Nigeria Poverty map) with .venv activated:
    python data\\scripts\\extract_features.py
"""

import ee
import pandas as pd
import time

# ---- 1. Initialize Earth Engine ----
ee.Initialize(project='earth-engine-legacy-project')

# ---- 2. Load RWI data, filter for reliability, sample points ----
SAMPLE_SIZE = 1200
RANDOM_SEED = 42

rwi = pd.read_csv('data/rwi/nga_relative_wealth_index.csv')
print(f"Loaded {len(rwi)} RWI points")

# Keep only the more reliable 75% (lower uncertainty)
rwi = rwi[rwi['error'] < rwi['error'].quantile(0.75)]
print(f"{len(rwi)} points remain after filtering high-uncertainty RWI values")

sample = rwi.sample(n=min(SAMPLE_SIZE, len(rwi)), random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Sampled {len(sample)} points for feature extraction")

# ---- 3. Define Earth Engine feature extraction function ----
def get_features(lat, lon, buffer_m=2400):
    pt = ee.Geometry.Point([lon, lat])
    box = pt.buffer(buffer_m).bounds()

    # Sentinel-2 (median composite, cloud-filtered, most recent full year)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(box)
          .filterDate('2023-01-01', '2023-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median())

    s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # blue, green, red, NIR, SWIR1, SWIR2

    # VIIRS nighttime lights (annual composite)
    viirs = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
             .filterBounds(box)
             .filterDate('2023-01-01', '2023-12-31')
             .select('avg_rad')
             .mean())

    # NDVI (vegetation health - agricultural productivity proxy)
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Land cover class (ESA WorldCover) - identifies cropland vs other
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')

    # Rainfall - annual total (CHIRPS)
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterBounds(box)
              .filterDate('2023-01-01', '2023-12-31')
              .select('precipitation')
              .sum())

    # Elevation and slope (terrain - farming viability, accessibility)
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(srtm)

    # Population density (rural vs dense settlement distinction)
    worldpop = (ee.ImageCollection('WorldPop/GP/100m/pop')
                .filterBounds(box)
                .filterDate('2023-01-01', '2023-12-31')
                .mean())

    try:
        s2_vals = s2.select(s2_bands).reduceRegion(
            ee.Reducer.mean(), box, 10, maxPixels=1e9
        ).getInfo()
        ntl_val = viirs.reduceRegion(
            ee.Reducer.mean(), box, 500, maxPixels=1e9
        ).getInfo()
        ndvi_val = ndvi.reduceRegion(
            ee.Reducer.mean(), box, 10, maxPixels=1e9
        ).getInfo()
        # mode = most common land cover class in the box (10 = cropland in WorldCover)
        lc_val = worldcover.reduceRegion(
            ee.Reducer.mode(), box, 10, maxPixels=1e9
        ).getInfo()
        rain_val = chirps.reduceRegion(
            ee.Reducer.mean(), box, 5000, maxPixels=1e9
        ).getInfo()
        elev_val = srtm.reduceRegion(
            ee.Reducer.mean(), box, 30, maxPixels=1e9
        ).getInfo()
        slope_val = slope.reduceRegion(
            ee.Reducer.mean(), box, 30, maxPixels=1e9
        ).getInfo()
        pop_val = worldpop.reduceRegion(
            ee.Reducer.mean(), box, 100, maxPixels=1e9
        ).getInfo()
        result = {**s2_vals, **ntl_val, **ndvi_val, **lc_val, **rain_val,
                  **elev_val, **slope_val, **pop_val}
    except Exception as e:
        result = {b: None for b in s2_bands}
        result['avg_rad'] = None
        result['NDVI'] = None
        result['Map'] = None
        result['precipitation'] = None
        result['elevation'] = None
        result['slope'] = None
        result['population'] = None
        result['error_msg'] = str(e)

    return result

# ---- 4. Loop through sampled points (with progress + basic rate limiting) ----
records = []
for i, row in sample.iterrows():
    feats = get_features(row['latitude'], row['longitude'])
    feats['latitude'] = row['latitude']
    feats['longitude'] = row['longitude']
    feats['rwi'] = row['rwi']
    records.append(feats)

    if i % 50 == 0:
        print(f"  {i}/{len(sample)} points processed...")
    time.sleep(0.1)  # small pause to avoid hammering the API

# ---- 5. Save merged feature table ----
out = pd.DataFrame(records)
out.to_csv('outputs/features_rwi_sentinel2_ntl.csv', index=False)
print(f"Saved {len(out)} rows to outputs/features_rwi_sentinel2_ntl.csv")
print(out.head())
print("\nColumns saved:", list(out.columns))
