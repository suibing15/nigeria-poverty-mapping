"""
Nigeria Poverty Mapping - Step: Extract satellite + agricultural features for DHS clusters
Run this from your project ROOT folder (C:\\Nigeria Poverty map) with .venv activated:
    python extract_features_dhs.py
"""

import ee
import pandas as pd
import time

# ---- 1. Initialize Earth Engine ----
ee.Initialize(project='earth-engine-legacy-project')

# ---- 2. Load DHS cluster wealth + GPS data (all 1,380 clusters - no sampling needed) ----
dhs = pd.read_csv('data/dhs/dhs_cluster_wealth_gps.csv')
print(f"Loaded {len(dhs)} DHS clusters")

# ---- 3. Define Earth Engine feature extraction function ----
def get_features(lat, lon, buffer_m=2400):
    pt = ee.Geometry.Point([lon, lat])
    box = pt.buffer(buffer_m).bounds()

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(box)
          .filterDate('2023-01-01', '2023-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median())

    s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

    viirs = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
             .filterBounds(box)
             .filterDate('2023-01-01', '2023-12-31')
             .select('avg_rad')
             .mean())

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')

    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterBounds(box)
              .filterDate('2023-01-01', '2023-12-31')
              .select('precipitation')
              .sum())

    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(srtm)

    # FIX: WorldPop's most recent global layer is 2020, not 2023 - this was why
    # population came back empty in the RWI run
    worldpop = (ee.ImageCollection('WorldPop/GP/100m/pop')
                .filterBounds(box)
                .filterDate('2019-01-01', '2020-12-31')
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

# ---- 4. Loop through ALL 1,380 DHS clusters ----
records = []
for i, row in dhs.iterrows():
    feats = get_features(row['latitude'], row['longitude'])
    feats['cluster'] = row['cluster']
    feats['latitude'] = row['latitude']
    feats['longitude'] = row['longitude']
    feats['wealth_index'] = row['wealth_index']
    feats['urban_rural'] = row['urban_rural']
    feats['state'] = row['state']
    records.append(feats)

    if i % 50 == 0:
        print(f"  {i}/{len(dhs)} clusters processed...")
    time.sleep(0.1)

# ---- 5. Save merged feature table ----
out = pd.DataFrame(records)
out.to_csv('outputs/features_dhs_sentinel2_ntl.csv', index=False)
print(f"Saved {len(out)} rows to outputs/features_dhs_sentinel2_ntl.csv")
print(out.head())
print("\nColumns saved:", list(out.columns))
