"""
Nigeria Poverty Mapping - Step: Build DHS cluster-level wealth + GPS table
Merges Household Recode (wealth index) with Geographic Data (GPS coordinates).

Run this from your project ROOT folder with .venv activated.
Requires: pandas, pyreadstat (for .dta), pyshp (for .shp)
    pip install pyreadstat pyshp
"""

import pandas as pd
import shapefile  # pyshp

# ---- 1. Load Household Recode (wealth index) ----
# Update this path to wherever you extracted the HR .dta file
HR_PATH = 'data/dhs/NGHR8BFL.dta'

hr = pd.read_stata(HR_PATH, convert_categoricals=False)
print(f"Loaded HR file: {hr.shape[0]} households, {hr.shape[1]} columns")

hr_small = hr[['hv001', 'hv271']].copy()
hr_small.columns = ['cluster', 'wealth_index_raw']
hr_small['wealth_index'] = hr_small['wealth_index_raw'] / 100000  # DHS scaling convention

# Aggregate to cluster level (mean wealth index per cluster)
cluster_wealth = hr_small.groupby('cluster')['wealth_index'].mean().reset_index()
print(f"Aggregated to {len(cluster_wealth)} clusters")

# ---- 2. Load Geographic Data (GPS coordinates) ----
# Update this path to wherever you extracted the GE .shp file (all .shp/.dbf/.shx/.prj must be together)
GE_PATH = 'data/dhs/NGGE8AFL.shp'

sf = shapefile.Reader(GE_PATH)
fields = [f[0] for f in sf.fields[1:]]  # skip deletion flag field
records = sf.records()

ge = pd.DataFrame(records, columns=fields)
ge_small = ge[['DHSCLUST', 'LATNUM', 'LONGNUM', 'URBAN_RURA', 'ADM1NAME']].copy()
ge_small.columns = ['cluster', 'latitude', 'longitude', 'urban_rural', 'state']
print(f"Loaded GE file: {len(ge_small)} clusters")

# Some clusters have LATNUM/LONGNUM = 0,0 (missing GPS) — drop those
before = len(ge_small)
ge_small = ge_small[(ge_small['latitude'] != 0) & (ge_small['longitude'] != 0)]
print(f"Dropped {before - len(ge_small)} clusters with missing GPS; {len(ge_small)} remain")

# ---- 3. Merge on cluster ID ----
merged = pd.merge(ge_small, cluster_wealth, on='cluster', how='inner')
print(f"\nMerged dataset: {len(merged)} clusters with both GPS and wealth index")
print(merged.head(10))

# ---- 4. Save ----
merged.to_csv('data/dhs/dhs_cluster_wealth_gps.csv', index=False)
print("\nSaved to data/dhs/dhs_cluster_wealth_gps.csv")
print("\nUrban/Rural breakdown:")
print(merged['urban_rural'].value_counts())
