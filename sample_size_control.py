"""
sample_size_control.py - Is the cross-country ordering of accuracy just a sample-size effect?

Countries differ in number of clusters (e.g. 1,380 vs 560). A Random Forest generally does better with more
training clusters, so differences in R2 between countries could partly reflect sample size rather than
anything about the country. This script re-estimates each country's cross-validated R2 on random
SUBSAMPLES of equal size, repeated many times, using exactly the same model and shuffled 5-fold CV.

    Overall model : every country subsampled to the size of the SMALLEST country
    Rural model   : every country subsampled to the smallest rural count
    Urban model   : every country subsampled to the smallest urban count

USAGE (project root, .venv activated):
    python sample_size_control.py                      all countries with a features file
    python sample_size_control.py --repeats 20 --countries nigeria ethiopia malawi rwanda

OUTPUT: results/sample_size_control.csv
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

FEATS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "avg_rad", "precipitation", "Map",
         "elevation", "slope", "population"]
RF = dict(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
PREFIX = "features_dhs_sentinel2_ntl_"
PREFERRED = ["nigeria", "ethiopia", "malawi", "rwanda", "senegal"]


def features_path(c):
    return (os.path.join("outputs", "features_dhs_sentinel2_ntl.csv") if c == "nigeria"
            else os.path.join("outputs", f"{PREFIX}{c}.csv"))


def discover():
    found = set()
    if os.path.isfile(features_path("nigeria")):
        found.add("nigeria")
    for f in glob.glob(os.path.join("outputs", PREFIX + "*.csv")):
        name = os.path.basename(f)[len(PREFIX):-4]
        if not name.endswith("_raw"):          # *_raw.csv = untouched backup made by fill_missing_precipitation.py
            found.add(name)
    return [c for c in PREFERRED if c in found] + sorted(found - set(PREFERRED))


def cv_r2(X, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    return cross_val_score(RandomForestRegressor(**RF), X, y, cv=kf, scoring="r2").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--groups", nargs="+", choices=["all", "rural", "urban"], default=["all", "rural", "urban"],
                    help="which groups to run (results are merged into the CSV, so you can run them separately)")
    args = ap.parse_args()
    countries = [c.lower() for c in args.countries] if args.countries else discover()
    missing = [c for c in countries if not os.path.isfile(features_path(c))]
    if missing:
        sys.exit(f"ERROR: no features file for {missing}.")

    data = {}
    for c in countries:
        d = pd.read_csv(features_path(c)).dropna(subset=FEATS + ["wealth_index"]).reset_index(drop=True)
        data[c] = d
    groups = {"all": lambda d: d, "rural": lambda d: d[d.urban_rural == "R"], "urban": lambda d: d[d.urban_rural == "U"]}
    target_n = {g: min(len(f(d)) for d in data.values()) for g, f in groups.items()}
    print("Equalised sample sizes:", target_n)

    rows = []
    for g, f in groups.items():
        if g not in args.groups:
            continue
        for c in countries:
            d = f(data[c]).reset_index(drop=True)
            full = cv_r2(d[FEATS].values, d["wealth_index"].values, 42)
            rng = np.random.default_rng(42)
            vals = []
            for r in range(args.repeats):
                idx = rng.choice(len(d), size=target_n[g], replace=False)
                s = d.iloc[idx]
                vals.append(cv_r2(s[FEATS].values, s["wealth_index"].values, r))
            rows.append({"group": g, "country": c, "n_full": len(d), "n_subsample": target_n[g],
                         "R2_full_sample": round(full, 4), "R2_subsample_mean": round(float(np.mean(vals)), 4),
                         "R2_subsample_sd": round(float(np.std(vals)), 4), "repeats": args.repeats})
            print(f"  {g:<6} {c:<9} n {len(d):>5} -> {target_n[g]:>4}   R2 full={full:.3f}   "
                  f"equal-n={np.mean(vals):.3f} (SD {np.std(vals):.3f})", flush=True)

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "sample_size_control.csv")
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        key = set(zip(new["group"], new["country"]))
        old = old[[(g, c) not in key for g, c in zip(old["group"], old["country"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print("\nSaved results/sample_size_control.csv")


if __name__ == "__main__":
    main()
