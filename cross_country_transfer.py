"""
cross_country_transfer.py - Leave-one-country-out (LOCO) transfer test.

QUESTION: if we train the model on the clusters of the OTHER countries only, how well does it predict a
country it has never seen?  (This is the 'out-of-country' test used by Jean et al. 2016 and Yeh et al. 2020;
train_model.py only tests within-country generalisation.)

USAGE (project root, .venv activated; needs the features files in outputs/):
    python cross_country_transfer.py                      uses every country that has a features file
    python cross_country_transfer.py --countries nigeria ethiopia malawi rwanda

METHOD
  * Same 13 features and same Random Forest settings as train_model.py (300 trees, depth 6, min leaf 5, seed 42).
  * The DHS wealth index is constructed WITHIN each country, so its level and spread are not comparable
    across countries. Wealth is therefore standardised (mean 0, SD 1) within each country before pooling.
  * For each held-out country H: train on all clusters of the other countries, predict H.
  * Metrics for H (overall, rural clusters only, urban clusters only):
        R2        raw out-of-country R2 (penalises any bias in level or spread)
        Spearman  rank correlation (scale-free: does the model order clusters correctly?)
        r2_recal  squared Pearson correlation = the R2 that would result after a linear recalibration of
                  the predictions to the held-out country (separates 'wrong level' from 'wrong ranking')
  * Reference: the within-country shuffled 5-fold CV R2 of the same model (unchanged by standardisation).

OUTPUT: results/cross_country_transfer.csv  (one row per held-out country)
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
        found.add(os.path.basename(f)[len(PREFIX):-4])
    return [c for c in PREFERRED if c in found] + sorted(found - set(PREFERRED))


def load(c):
    d = pd.read_csv(features_path(c))
    d = d.dropna(subset=FEATS + ["wealth_index"]).reset_index(drop=True)
    d["country"] = c
    d["wealth_z"] = (d["wealth_index"] - d["wealth_index"].mean()) / d["wealth_index"].std()
    return d


def metrics(y, p):
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    return (r2_score(y, p), spearmanr(y, p)[0], pearsonr(y, p)[0] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+")
    args = ap.parse_args()
    countries = [c.lower() for c in args.countries] if args.countries else discover()
    missing = [c for c in countries if not os.path.isfile(features_path(c))]
    if missing:
        sys.exit(f"ERROR: no features file for {missing}. Run the extraction first.")
    if len(countries) < 2:
        sys.exit("ERROR: need at least two countries for a leave-one-country-out test.")

    data = {c: load(c) for c in countries}
    print("Countries:", ", ".join(f"{c} (n={len(d)})" for c, d in data.items()))

    rows = []
    for held in countries:
        train = pd.concat([data[c] for c in countries if c != held], ignore_index=True)
        test = data[held]
        model = RandomForestRegressor(**RF).fit(train[FEATS].values, train["wealth_z"].values)
        pred = model.predict(test[FEATS].values)
        y = test["wealth_z"].values
        row = {"held_out_country": held, "n_test": len(test), "n_train": len(train),
               "trained_on": "+".join(c for c in countries if c != held)}
        for tag, mask in (("all", np.ones(len(test), bool)),
                          ("rural", (test["urban_rural"] == "R").values),
                          ("urban", (test["urban_rural"] == "U").values)):
            r2, rho, r2c = metrics(y[mask], pred[mask])
            row[f"loco_R2_{tag}"] = round(r2, 4)
            row[f"loco_spearman_{tag}"] = round(rho, 4)
            row[f"loco_r2_recal_{tag}"] = round(r2c, 4)
        row["mean_pred_z"] = round(float(pred.mean()), 3)          # 0 = correct level
        row["sd_pred_z"] = round(float(pred.std()), 3)             # 1 = correct spread
        # reference: within-country shuffled 5-fold CV of the same model
        row["within_country_cv_R2"] = round(float(cross_val_score(
            RandomForestRegressor(**RF), test[FEATS].values, y,
            cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()), 4)
        rows.append(row)
        print(f"  held out {held:<9} LOCO R2={row['loco_R2_all']:.3f}  Spearman={row['loco_spearman_all']:.3f}  "
              f"r2_recal={row['loco_r2_recal_all']:.3f}   (within-country CV R2={row['within_country_cv_R2']:.3f})")

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(os.path.join("results", "cross_country_transfer.csv"), index=False)
    print("\nSaved results/cross_country_transfer.csv")


if __name__ == "__main__":
    main()
