"""
Poverty Mapping - CANONICAL training script (v2), used identically for every country.

WHY THIS SCRIPT EXISTS
The original Nigeria script used cross_val_score(cv=5) with NO explicit shuffle, and the
Nigeria feature CSV happened to be sorted by state, so each CV fold was accidentally a
block of neighbouring states. That produced the earlier 0.625 figure. This script makes
every methodological choice explicit so results do not depend on how a CSV is sorted.

WHAT IT DOES (every country, same settings)
  1. Random 5-fold CV, always shuffle=True, random_state=42.
  2. Spatial CV: whole regions/states held out (GroupKFold).
  3. Held-out test split (for plots).
  4. Feature importance from a model refit on the FULL sample   <-- v2 change (v1 used the 80% split)
  5. Rural / urban stratified models, same settings.
  6. Diagnostics added in v2, because overall R2 alone can mislead:
       - Location-only baseline: how well do latitude/longitude ALONE predict wealth?
         (If wealth is not spatially smooth, no satellite feature can predict it either.)
       - Share of rural wealth variance that lies WITHIN admin units (vs between them).
       - Within-rural Spearman correlation of wealth with population and nightlights.
       - Integrity check: within the same admin unit, urban clusters should be wealthier
         than rural ones. If the wealth file and GPS file were misaligned, this fails.
  7. One summary row per country -> outputs/model_summary_all_countries.csv

TO USE FOR A NEW COUNTRY (v3): just name the country on the command line - nothing to edit:
    python train_model.py senegal
It reads outputs/features_dhs_sentinel2_ntl_<country>.csv  (Nigeria's file has no suffix).
With no argument it uses the CONFIG block below (default: nigeria).
Run from your project ROOT folder with .venv activated.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # files only, never open a window: the Tk GUI backend crashes on Windows
                          # when the Random Forest's worker threads clean up figures
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG - the only section that changes between countries
# =========================================================
COUNTRY = "nigeria"                                          # <-- change per country
INPUT_CSV = "outputs/features_dhs_sentinel2_ntl.csv"          # <-- change per country
REGION_COL = "state"   # admin-1 column; the merge scripts store every country's regions as 'state'

if len(sys.argv) > 1:                       # v3: country given on the command line
    COUNTRY = sys.argv[1].lower()
    INPUT_CSV = ("outputs/features_dhs_sentinel2_ntl.csv" if COUNTRY == "nigeria"
                 else f"outputs/features_dhs_sentinel2_ntl_{COUNTRY}.csv")
    if not os.path.isfile(INPUT_CSV):
        sys.exit(f"ERROR: {INPUT_CSV} not found. Run the extraction first: "
                 f"python extract_features_dhs_country.py {COUNTRY}")

FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'avg_rad',
                'precipitation', 'Map', 'elevation', 'slope', 'population']
TARGET_COL = 'wealth_index'
RF_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
RANDOM_STATE = 42

def rf():
    return RandomForestRegressor(**RF_PARAMS)

def shuffled_cv():
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def f(x, nd=4):
    return round(float(x), nd)

# =========================================================
# 1. Load and clean
# =========================================================
df = pd.read_csv(INPUT_CSV)
print(f"[{COUNTRY}] Loaded {len(df)} clusters")

feature_cols = [c for c in FEATURE_COLS if c in df.columns]
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"WARNING: expected feature(s) not found and will be skipped: {missing_cols}")

if 'error_msg' in df.columns and df['error_msg'].notnull().any():
    print(f"WARNING: {int(df['error_msg'].notnull().sum())} clusters have an extraction error. "
          f"Re-run the extract script until it reports 0 errors.")

df_clean = df.dropna(subset=feature_cols + [TARGET_COL, REGION_COL]).reset_index(drop=True)
print(f"[{COUNTRY}] {len(df_clean)} clean rows after dropping missing values "
      f"(dropped {len(df) - len(df_clean)})")
_dropped = df[~df.index.isin(df.dropna(subset=feature_cols + [TARGET_COL, REGION_COL]).index)]
if len(_dropped) >= 5:
    # Safeguard: dropping is only harmless if the dropped clusters look like the rest.
    _gap = _dropped[TARGET_COL].mean() - df_clean[TARGET_COL].mean()
    _urb = (_dropped["urban_rural"] == "U").mean() if "urban_rural" in _dropped.columns else float("nan")
    print(f"   Dropped clusters: {100 * _urb:.0f}% urban; mean wealth differs from the rest by {_gap:+.2f} "
          f"(SD of wealth in the kept clusters: {df_clean[TARGET_COL].std():.2f}).")
    if abs(_gap) > 0.5 * df_clean[TARGET_COL].std():
        print("   WARNING: the dropped clusters are NOT typical (much richer or poorer than the rest). Dropping them can bias the "
              "results. If the cause is a coastline or similar grid limit, see fill_missing_precipitation.py and "
              "missing_rainfall_sensitivity.py.")

X = df_clean[feature_cols]
y = df_clean[TARGET_COL]
groups = df_clean[REGION_COL]
results = {"country": COUNTRY, "n_clusters": int(len(df_clean)),
           "n_dropped_missing": int(len(df) - len(df_clean))}

# =========================================================
# 2. Random 5-fold CV - explicitly shuffled
# =========================================================
cv_scores = cross_val_score(rf(), X, y, cv=shuffled_cv(), scoring='r2')
print(f"\n=== Random 5-fold CV (shuffled, random_state={RANDOM_STATE}) ===")
print(f"R2 per fold : {np.round(cv_scores, 4)}   Mean: {cv_scores.mean():.4f}")
results["random_cv_r2_mean"] = f(cv_scores.mean())
results["random_cv_r2_folds"] = str([f(v) for v in cv_scores])

# =========================================================
# 3. Spatial CV - whole regions held out
# =========================================================
n_groups = groups.nunique()
if 5 <= n_groups < 10:
    print(f"\nNOTE: only {n_groups} regions - spatial CV holds out roughly one region per fold, "
          f"so it is very coarse. Interpret it with care.")
if n_groups >= 5:
    spatial_scores = cross_val_score(rf(), X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring='r2')
    print(f"\n=== Spatial ({REGION_COL}-grouped) 5-fold CV ===")
    print(f"R2 per fold : {np.round(spatial_scores, 4)}   Mean: {spatial_scores.mean():.4f}")
    results["spatial_cv_r2_mean"] = f(spatial_scores.mean())
    results["spatial_cv_r2_folds"] = str([f(v) for v in spatial_scores])
else:
    print(f"\nWARNING: only {n_groups} distinct '{REGION_COL}' values; need >=5 for spatial CV. Skipping it.")
    results["spatial_cv_r2_mean"] = None
    results["spatial_cv_r2_folds"] = None
results["n_regions"] = int(n_groups)

# =========================================================
# 4. Held-out test split (for plots)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
model_test = rf().fit(X_train, y_train)
y_pred = model_test.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n=== Held-out Test ===  R2: {r2:.4f}   RMSE: {rmse:.4f}")
results["test_r2"] = f(r2)
results["test_rmse"] = f(rmse)

# =========================================================
# 5. Feature importance - FULL-sample refit (v2 change)
# =========================================================
model_full = rf().fit(X, y)
importances = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Feature Importance (overall, full-sample fit, %) ===")
print((importances * 100).round(1))
os.makedirs('outputs', exist_ok=True)
importances.to_csv(f'outputs/feature_importance_{COUNTRY}.csv')
for i in range(3):
    results[f"top{i+1}_feature"] = importances.index[i]
    results[f"top{i+1}_pct"] = f(importances.iloc[i] * 100, 1)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh'); plt.gca().invert_yaxis()
plt.xlabel('Importance'); plt.title(f'Feature Importance - {COUNTRY.title()} DHS Wealth Index')
plt.tight_layout(); plt.savefig(f'outputs/feature_importance_{COUNTRY}.png', dpi=150); plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual DHS Wealth Index'); plt.ylabel('Predicted DHS Wealth Index')
plt.title(f'{COUNTRY.title()}: Predicted vs Actual (test R2={r2:.3f}, CV R2={cv_scores.mean():.3f})')
plt.tight_layout(); plt.savefig(f'outputs/predicted_vs_actual_{COUNTRY}.png', dpi=150); plt.close()
joblib.dump(model_full, f'outputs/rf_model_{COUNTRY}.pkl')

# =========================================================
# 6. Rural vs Urban split - same settings
# =========================================================
print("\n" + "=" * 60)
print(f"URBAN vs RURAL SPLIT - {COUNTRY.upper()}")
print("=" * 60)
for label, code in [('Rural', 'R'), ('Urban', 'U')]:
    sub = df_clean[df_clean['urban_rural'] == code]
    results[f"{code}_n"] = int(len(sub))
    if len(sub) < 30:
        print(f"{label}: too few rows ({len(sub)}) to model reliably")
        results[f"{code}_cv_r2"] = None
        continue
    cv_sub = cross_val_score(rf(), sub[feature_cols], sub[TARGET_COL], cv=shuffled_cv(), scoring='r2')
    imp_sub = pd.Series(rf().fit(sub[feature_cols], sub[TARGET_COL]).feature_importances_,
                        index=feature_cols).sort_values(ascending=False)
    print(f"\n--- {label} (n={len(sub)}) --- Mean CV R2: {cv_sub.mean():.4f}")
    print((imp_sub.head(5) * 100).round(1))
    imp_sub.to_csv(f'outputs/feature_importance_{code}_{COUNTRY}.csv')
    results[f"{code}_cv_r2"] = f(cv_sub.mean())
    results[f"{code}_top_features"] = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in imp_sub.head(3).items())

# =========================================================
# 7. Diagnostics (v2): what limits predictability?
# =========================================================
print("\n" + "=" * 60)
print(f"DIAGNOSTICS - {COUNTRY.upper()}")
print("=" * 60)

def location_only_r2(sub):
    """CV R2 of a model that sees ONLY coordinates (lon scaled by cos(lat))."""
    lat0 = np.deg2rad(sub['latitude'].mean())
    XY = np.c_[sub['latitude'].values, sub['longitude'].values * np.cos(lat0)]
    return cross_val_score(rf(), XY, sub[TARGET_COL].values, cv=shuffled_cv(), scoring='r2').mean()

if {'latitude', 'longitude'}.issubset(df_clean.columns):
    results["location_only_r2_all"] = f(location_only_r2(df_clean))
    for code in ['R', 'U']:
        sub = df_clean[df_clean['urban_rural'] == code]
        results[f"location_only_r2_{code}"] = f(location_only_r2(sub)) if len(sub) >= 30 else None
    print(f"Location-only R2  all={results['location_only_r2_all']}  "
          f"rural={results['location_only_r2_R']}  urban={results['location_only_r2_U']}")

# variance structure
grand = df_clean[TARGET_COL].mean()
sst = ((df_clean[TARGET_COL] - grand) ** 2).sum()
ssb_ru = sum(len(g) * (g.mean() - grand) ** 2 for _, g in df_clean.groupby('urban_rural')[TARGET_COL])
results["share_variance_urban_rural_gap"] = f(ssb_ru / sst)
rural = df_clean[df_clean['urban_rural'] == 'R']
urban = df_clean[df_clean['urban_rural'] == 'U']
results["rural_wealth_variance"] = f(rural[TARGET_COL].var())
results["urban_wealth_variance"] = f(urban[TARGET_COL].var())

if len(rural) > 30 and rural[REGION_COL].nunique() > 1:
    gr = rural[TARGET_COL].mean(); n = len(rural); k = rural[REGION_COL].nunique()
    ssb = sum(len(g) * (g.mean() - gr) ** 2 for _, g in rural.groupby(REGION_COL)[TARGET_COL])
    eta = ssb / ((rural[TARGET_COL] - gr) ** 2).sum()
    adj = 1 - (1 - eta) * (n - 1) / (n - k)
    results["rural_within_unit_share"] = f(1 - adj)
    print(f"Rural wealth variance WITHIN {REGION_COL}s: {100*(1-adj):.0f}%")
    for feat in ['population', 'avg_rad']:
        if feat in rural.columns:
            results[f"rural_spearman_{feat}"] = f(spearmanr(rural[feat], rural[TARGET_COL])[0], 2)
    if 'avg_rad' in rural.columns:
        # share of rural clusters with essentially no detectable night light (VIIRS avg_rad < 0.5)
        results["rural_share_nightlights_below_0_5"] = f((rural['avg_rad'] < 0.5).mean(), 3)

# importance stability: population and nightlights are correlated, so impurity importance
# can swap their order depending on the training sample. Count how often population wins.
if {'population', 'avg_rad'}.issubset(feature_cols):
    rng = np.random.default_rng(RANDOM_STATE)
    B, wins = 40, 0
    for b in range(B):
        idx = rng.integers(0, len(df_clean), len(df_clean))
        m_b = RandomForestRegressor(**{**RF_PARAMS, 'random_state': b}).fit(X.iloc[idx], y.iloc[idx])
        imp_b = pd.Series(m_b.feature_importances_, index=feature_cols)
        wins += int(imp_b['population'] > imp_b['avg_rad'])
    results["bootstrap_pop_gt_nightlights_share"] = f(wins / B, 2)
    results["importance_population_pct"] = f(importances['population'] * 100, 1)
    results["importance_nightlights_pct"] = f(importances['avg_rad'] * 100, 1)
    print(f"IMPORTANCE STABILITY: population > nightlights in {wins}/{B} bootstrap fits "
          f"({100*wins/B:.0f}%). Near 0% or 100% = stable order; near 50% = order is not reliable.")

# integrity check: urban should out-wealth rural inside the same admin unit
gaps = []
for _, g in df_clean.groupby(REGION_COL):
    u = g[g['urban_rural'] == 'U'][TARGET_COL]; r = g[g['urban_rural'] == 'R'][TARGET_COL]
    if len(u) >= 3 and len(r) >= 3:
        gaps.append(u.mean() - r.mean())
if gaps:
    results["integrity_units_tested"] = int(len(gaps))
    results["integrity_units_urban_wealthier"] = int(sum(g > 0 for g in gaps))
    results["integrity_mean_urban_minus_rural"] = f(np.mean(gaps), 2)
    print(f"INTEGRITY: urban wealthier than rural in {sum(g > 0 for g in gaps)}/{len(gaps)} "
          f"{REGION_COL}s (mean gap {np.mean(gaps):.2f}). "
          f"A healthy join shows (nearly) all units positive.")

# =========================================================
# 8. One-row summary per country
# =========================================================
summary_path = 'outputs/model_summary_all_countries.csv'
new_row = pd.DataFrame([results])
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing['country'] != COUNTRY]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row
combined.to_csv(summary_path, index=False)
print(f"\nSummary row for '{COUNTRY}' saved to {summary_path}")
print("Run this script for every country; the CSV becomes your cross-country table.")
