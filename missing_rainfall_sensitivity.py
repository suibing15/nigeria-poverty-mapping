"""
missing_rainfall_sensitivity.py - Does the way we handle missing rainfall values change the results?

When some clusters have no CHIRPS rainfall value (for example coastal clusters in Senegal), there are 3 ways to
proceed. This script runs the same Random Forest and the same validation as train_model.py on each:

    A. Filled      keep every cluster; rainfall of the empty ones filled from the nearest cluster
                   (fill_missing_precipitation.py)                                   <- used in the paper
    B. Dropped     drop the clusters with an empty rainfall value (the default behaviour of train_model.py)
    C. No rainfall keep every cluster; remove rainfall from the features altogether

If A, B and C agree, the handling does not matter. If B differs from A and C, dropping the clusters biased the result.

USAGE (project root, .venv activated; needs outputs/features_dhs_sentinel2_ntl_<country>_raw.csv, created by
fill_missing_precipitation.py):
    python missing_rainfall_sensitivity.py senegal

OUTPUT: results/missing_rainfall_sensitivity_<country>.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from spatial_folds import StableGroupKFold

warnings.filterwarnings("ignore")
FEATS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "avg_rad", "precipitation", "Map", "elevation", "slope", "population"]
RF = dict(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)


def evaluate(d, feats):
    y, X = d["wealth_index"].values, d[feats].values
    kf = KFold(5, shuffle=True, random_state=42)
    out = {"n": len(d)}
    out["random_cv_r2"] = cross_val_score(RandomForestRegressor(**RF), X, y, cv=kf, scoring="r2").mean()
    out["spatial_cv_r2"] = cross_val_score(RandomForestRegressor(**RF), X, y, cv=StableGroupKFold(5),
                                           groups=d["state"].values, scoring="r2").mean()
    for tag, code in (("rural", "R"), ("urban", "U")):
        s = d[d["urban_rural"] == code]
        out[f"{tag}_n"] = len(s)
        out[f"{tag}_cv_r2"] = cross_val_score(RandomForestRegressor(**RF), s[feats].values, s["wealth_index"].values,
                                              cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python missing_rainfall_sensitivity.py <country>")
    c = sys.argv[1].lower()
    raw = os.path.join("outputs", f"features_dhs_sentinel2_ntl_{c}_raw.csv")
    if not os.path.isfile(raw):
        sys.exit(f"ERROR: {raw} not found. Run fill_missing_precipitation.py {c} first.")
    d = pd.read_csv(raw)
    if "error_msg" in d.columns:
        d = d[d["error_msg"].isnull()]
    d = d.dropna(subset=[f for f in FEATS if f != "precipitation"] + ["wealth_index", "state"]).reset_index(drop=True)

    filled = pd.read_csv(os.path.join("outputs", f"features_dhs_sentinel2_ntl_{c}.csv"))
    filled = filled.dropna(subset=FEATS + ["wealth_index", "state"]).reset_index(drop=True)
    dropped = d.dropna(subset=["precipitation"]).reset_index(drop=True)
    no_rain = [f for f in FEATS if f != "precipitation"]

    rows = []
    for name, data, feats in (("A. Filled (used in paper)", filled, FEATS),
                              ("B. Dropped (train_model.py default)", dropped, FEATS),
                              ("C. No rainfall feature", d, no_rain)):
        r = evaluate(data, feats)
        r["variant"] = name
        rows.append(r)
        print(f"{name:<38} n={r['n']:>4}  random {r['random_cv_r2']:.3f}  spatial {r['spatial_cv_r2']:.3f}  "
              f"rural {r['rural_cv_r2']:.3f}  urban {r['urban_cv_r2']:.3f}", flush=True)
    out = pd.DataFrame(rows)[["variant", "n", "random_cv_r2", "spatial_cv_r2", "rural_n", "rural_cv_r2", "urban_n", "urban_cv_r2"]]
    os.makedirs("results", exist_ok=True)
    out.round(4).to_csv(os.path.join("results", f"missing_rainfall_sensitivity_{c}.csv"), index=False)
    print(f"\nSaved results/missing_rainfall_sensitivity_{c}.csv")


if __name__ == "__main__":
    main()
