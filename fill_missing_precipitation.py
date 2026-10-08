"""
fill_missing_precipitation.py - Fill missing CHIRPS rainfall values from the nearest cluster that has one.

WHY: CHIRPS rainfall is a 5 km grid that does not cover coastal pixels. Clusters on a coast (for example the
Dakar peninsula in Senegal) can come back with an empty rainfall value even though every other feature is fine.
train_model.py drops any cluster with an empty feature. When the empty clusters are not random (in Senegal all
16 were urban, 14 in Dakar, and they were the wealthiest clusters), dropping them biases the results.
Rainfall changes slowly over short distances, so the value of the nearest cluster that has data is a
reasonable stand-in. This script does that, openly.

USAGE (project root, .venv activated):
    python fill_missing_precipitation.py senegal

WHAT IT DOES
  1. Saves the untouched file as outputs/features_dhs_sentinel2_ntl_<country>_raw.csv (only if not already saved).
  2. Fills each empty 'precipitation' with the value of the nearest cluster that has one (great-circle distance).
  3. Adds two columns so the change is always visible: precip_filled (1 = filled) and precip_fill_km (distance used).
  4. Prints how many were filled, the distances, and a warning if any distance is above 50 km.
  5. Overwrites outputs/features_dhs_sentinel2_ntl_<country>.csv with the filled version.

Only precipitation is filled. Run it only when the empty values are few and clearly caused by coastline or
a similar grid limit. Report it in the methods, and run missing_rainfall_sensitivity.py to show the effect.
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python fill_missing_precipitation.py <country>")
    country = sys.argv[1].lower()
    path = (os.path.join("outputs", "features_dhs_sentinel2_ntl.csv") if country == "nigeria"
            else os.path.join("outputs", f"features_dhs_sentinel2_ntl_{country}.csv"))
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found.")
    raw = path.replace(".csv", "_raw.csv")
    if not os.path.isfile(raw):
        shutil.copy(path, raw)
        print(f"Saved the untouched file as {raw}")

    d = pd.read_csv(raw)
    miss = d["precipitation"].isnull()
    if "error_msg" in d.columns:
        miss &= d["error_msg"].isnull()
    n_miss = int(miss.sum())
    print(f"[{country}] {n_miss} cluster(s) with an empty rainfall value out of {len(d)}")
    if n_miss == 0:
        print("Nothing to fill.")
        return

    ok = d[d["precipitation"].notnull()].reset_index(drop=True)
    tree = BallTree(np.radians(ok[["latitude", "longitude"]].values), metric="haversine")
    dist, idx = tree.query(np.radians(d.loc[miss, ["latitude", "longitude"]].values), k=1)
    km = dist[:, 0] * 6371.0
    d["precip_filled"] = 0
    d["precip_fill_km"] = np.nan
    d.loc[miss, "precipitation"] = ok["precipitation"].values[idx[:, 0]]
    d.loc[miss, "precip_filled"] = 1
    d.loc[miss, "precip_fill_km"] = km

    f = d[d["precip_filled"] == 1]
    print(f"  urban/rural of filled clusters : {f['urban_rural'].value_counts().to_dict()}")
    print(f"  regions of filled clusters     : {f['state'].value_counts().to_dict()}")
    print(f"  mean wealth of filled clusters : {f['wealth_index'].mean():.2f}  (all clusters: {d['wealth_index'].mean():.2f})")
    print(f"  distance to donor cluster (km) : median {np.median(km):.1f}, max {km.max():.1f}")
    if km.max() > 50:
        print("  WARNING: some donors are more than 50 km away. Check these values before using them.")
    d.to_csv(path, index=False)
    print(f"Saved the filled file to {path}")
    print(f"Next: python train_model.py {country}   (and report this step in the methods)")


if __name__ == "__main__":
    main()
