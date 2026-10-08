"""
merge_dhs_country.py - Build the cluster-level wealth + GPS table for ANY DHS/MIS country.

USAGE (from the project ROOT folder, .venv activated):
    python merge_dhs_country.py senegal
    python merge_dhs_country.py rwanda

You do NOT edit any filenames. Put the DHS downloads for the country in  data/<country>/
(unzipped, or still as .zip files - it will unzip them). It searches every sub-folder for:
    Household Recode : a file named like  XXHR##XX.dta   (XX = 2-letter country code)
    Geographic Data  : a file named like  XXGE##XX.shp   (its .dbf/.shx/.prj must sit beside it)
Folder names do not matter. (DHS folder and inner file names often differ, e.g. folder
ETHR8ADT holds ETHR8AFL.dta - that is why this script searches instead of trusting paths.)

If a folder holds MORE than one survey (e.g. Senegal's continuous DHS has one per year), the
script stops and lists the candidates. Either keep only the survey you want in the folder, or
point to it explicitly:
    python merge_dhs_country.py senegal --hr data/senegal/SNHR8BDT/SNHR8BFL.dta --ge data/senegal/SNGE8BFL/SNGE8BFL.shp

OUTPUT: data/<country>/dhs_cluster_wealth_gps.csv with columns
    cluster, latitude, longitude, urban_rural, state, wealth_index, dhs_year
('state' always holds the first-level admin unit: states, regions, provinces or districts.)

It also runs the checks that caught problems before, and prints them for you to read:
cluster counts, HR/GPS match, special strata with no wealth index (like Malawi's refugee camp),
urban/rural counts, number of regions, face-validity ranking, and the urban-vs-rural integrity check.

Requires: pandas, pyreadstat, pyshp   (pip install pandas pyreadstat pyshp)
"""

import argparse
import difflib
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

try:
    import shapefile  # pyshp
except ImportError:
    sys.exit("Missing package 'pyshp'. Install it with:  pip install pyshp pyreadstat")

DATA_DIR = "data"
HR_RE = re.compile(r"^([A-Z]{2})HR[0-9][0-9A-Z][A-Z]{2}\.dta$", re.I)
GE_RE = re.compile(r"^([A-Z]{2})GE[0-9][0-9A-Z][A-Z]{2}\.shp$", re.I)


def die(msg):
    print("\nERROR: " + msg)
    sys.exit(1)


def country_folder(country):
    folder = os.path.join(DATA_DIR, country)
    if os.path.isdir(folder):
        return folder
    avail = ([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
             if os.path.isdir(DATA_DIR) else [])
    msg = f"Folder '{folder}' not found. Folders inside '{DATA_DIR}/': {avail or 'none'}."
    close = difflib.get_close_matches(country, avail, n=3, cutoff=0.6)
    if close:
        msg += (f"\nDid you mean: {', '.join(close)}?  The folder name is also the country name "
                f"printed in results, so rename the folder if it is misspelled.")
    die(msg)


def unzip_all(folder):
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".zip"):
            target = os.path.join(folder, os.path.splitext(name)[0])
            if not os.path.isdir(target):
                print(f"Unzipping {name} ...")
                with zipfile.ZipFile(os.path.join(folder, name)) as z:
                    z.extractall(target)


def find_files(folder, rx):
    hits = []
    for root, _, files in os.walk(folder):
        for f in files:
            if rx.match(f):
                hits.append(os.path.join(root, f))
    return sorted(hits)


def choose(hits, override, label, country, example):
    if override:
        if not os.path.isfile(override):
            die(f"--{label.lower()} file not found: {override}")
        return override
    if len(hits) == 1:
        return hits[0]
    if not hits:
        die(f"No {label} file found anywhere under data/{country}/.\n"
            f"Expected a file named like {example}. Check that you downloaded the "
            f"{'Household Recode (.dta)' if label == 'HR' else 'Geographic Data shapefile'} and, "
            f"if it is still zipped, that the .zip is inside data/{country}/.")
    die(f"Several {label} files found under data/{country}/:\n  " + "\n  ".join(hits) +
        f"\nKeep only the survey you want in that folder, or choose with --{label.lower()} PATH.")


def read_ge(path):
    last = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            try:
                sf = shapefile.Reader(path, encoding=enc)
            except TypeError:           # very old pyshp without the encoding argument
                sf = shapefile.Reader(path)
            fields = [f[0] for f in sf.fields[1:]]
            records = [list(r) for r in sf.records()]
            return pd.DataFrame(records, columns=fields)
        except UnicodeDecodeError as e:
            last = e
    die(f"Could not decode the shapefile attribute table ({last}).")


def main():
    try:                                   # never crash on an unprintable character in a region name
        sys.stdout.reconfigure(errors="replace")
    except Exception:                      # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Merge DHS Household Recode wealth with cluster GPS.")
    ap.add_argument("country", help="folder name under data/, e.g. senegal")
    ap.add_argument("--hr", help="explicit path to the Household Recode .dta")
    ap.add_argument("--ge", help="explicit path to the Geographic Data .shp")
    args = ap.parse_args()
    country = args.country.lower()

    folder = country_folder(country)
    unzip_all(folder)

    hr_path = choose(find_files(folder, HR_RE), args.hr, "HR", country, "XXHR8AFL.dta")
    ge_path = choose(find_files(folder, GE_RE), args.ge, "GE", country, "XXGE8AFL.shp")
    cc_hr = os.path.basename(hr_path)[:2].upper()
    cc_ge = os.path.basename(ge_path)[:2].upper()
    print(f"\nCountry folder : {folder}")
    print(f"Household file : {hr_path}")
    print(f"GPS file       : {ge_path}")
    if cc_hr != cc_ge:
        die(f"The HR file is for country code '{cc_hr}' but the GPS file is for '{cc_ge}'. "
            f"They must come from the same survey.")

    flags = []   # things the user should look at before spending hours on extraction

    # ---------------- Household recode -> cluster wealth ----------------
    try:
        hr = pd.read_stata(hr_path, columns=["hv001", "hv271"], convert_categoricals=False)
    except Exception as e:
        print(f"\nCould not read hv001/hv271 directly ({e}). Looking for wealth variables...")
        full = pd.read_stata(hr_path, convert_categoricals=False)
        die("The household file does not contain the expected cluster number (hv001) and/or wealth "
            f"index factor score (hv271). Columns starting with 'hv27': "
            f"{[c for c in full.columns if str(c).startswith('hv27')]}")
    hr = hr.rename(columns={"hv001": "cluster", "hv271": "wealth_raw"})
    hr["cluster"] = pd.to_numeric(hr["cluster"], errors="coerce")
    hr = hr.dropna(subset=["cluster"])
    hr["cluster"] = hr["cluster"].astype(int)
    hr["wealth_index"] = pd.to_numeric(hr["wealth_raw"], errors="coerce") / 100000  # DHS convention
    cluster_wealth = hr.groupby("cluster")["wealth_index"].mean().reset_index()
    print(f"\nHR file: {len(hr):,} households -> {len(cluster_wealth)} clusters")

    # ---------------- Geographic data ----------------
    ge = read_ge(ge_path)
    print("GPS shapefile columns:", list(ge.columns))
    need = ["DHSCLUST", "LATNUM", "LONGNUM", "URBAN_RURA"]
    missing = [c for c in need if c not in ge.columns]
    if missing:
        die(f"GPS file is missing required column(s) {missing}. Columns found are listed above.")

    region_col = None
    for cand in ("ADM1NAME", "DHSREGNA"):
        if cand in ge.columns:
            filled = ge[cand].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).notna().mean()
            if filled >= 0.95:
                region_col = cand
                break
    if region_col is None:
        die("No usable region-name column (ADM1NAME or DHSREGNA) in the GPS file.")
    print(f"Region column used for 'state': {region_col}")

    ge_small = pd.DataFrame({
        "cluster": pd.to_numeric(ge["DHSCLUST"], errors="coerce"),
        "latitude": pd.to_numeric(ge["LATNUM"], errors="coerce"),
        "longitude": pd.to_numeric(ge["LONGNUM"], errors="coerce"),
        "urban_rural": ge["URBAN_RURA"].astype(str).str.strip().str.upper(),
        "state": ge[region_col].astype(str).str.strip(),
    })
    n_ge_raw = len(ge_small)
    ge_small = ge_small.dropna(subset=["cluster", "latitude", "longitude"])
    ge_small["cluster"] = ge_small["cluster"].astype(int)
    no_gps = (ge_small["latitude"] == 0) & (ge_small["longitude"] == 0)   # DHS codes missing GPS as 0,0
    ge_small = ge_small[~no_gps]
    print(f"GPS file: {n_ge_raw} clusters; dropped {n_ge_raw - len(ge_small)} with missing GPS; "
          f"{len(ge_small)} remain")
    if n_ge_raw - len(ge_small) > 0.05 * n_ge_raw:
        flags.append(f"{n_ge_raw - len(ge_small)} clusters had missing GPS (>5%).")

    # survey year (used later to pick the satellite year)
    dhs_year = None
    if "DHSYEAR" in ge.columns:
        years = pd.to_numeric(ge["DHSYEAR"], errors="coerce").dropna().astype(int)
        if len(years):
            dhs_year = int(years.mode().iloc[0])
            print(f"Survey year in GPS file (DHSYEAR): {sorted(years.unique().tolist())} -> using {dhs_year}")
            if years.nunique() > 1:
                flags.append(f"GPS file lists several survey years {sorted(years.unique().tolist())}; "
                             f"using {dhs_year}. Make sure the folder holds ONE survey, or pass --year later.")
    if dhs_year is None:
        flags.append("No DHSYEAR in GPS file - you must pass --year to the extraction step.")
    ge_small["dhs_year"] = dhs_year if dhs_year is not None else np.nan

    # ---------------- Merge + match check ----------------
    merged = pd.merge(ge_small, cluster_wealth, on="cluster", how="inner")
    match = len(merged) / max(1, min(len(cluster_wealth), len(ge_small)))
    print(f"\nMerged: {len(merged)} clusters have both GPS and a wealth record "
          f"({100 * match:.1f}% of the smaller file)")
    if match < 0.90:
        die("Fewer than 90% of clusters matched between the household file and the GPS file. They are "
            "probably from DIFFERENT surveys (different year or round). Check which files are in the folder.")
    only_hr = len(cluster_wealth) - len(merged)
    if only_hr > 0:
        print(f"  ({only_hr} household-file clusters have no usable GPS)")

    cols = ["cluster", "latitude", "longitude", "urban_rural", "state", "wealth_index", "dhs_year"]
    merged = merged[cols]

    # ---------------- Checks to READ ----------------
    nowealth = merged[merged["wealth_index"].isna()]
    if len(nowealth):
        print(f"\nClusters with NO wealth index: {len(nowealth)}  (regions: "
              f"{nowealth['state'].value_counts().to_dict()})")
        print("  These will be dropped at training. If they form their own odd stratum (e.g. Malawi's "
              "'Dowa (Camps)' refugee-camp domain), check the survey report that this is expected.")
        flags.append(f"{len(nowealth)} clusters have no wealth index (see above).")

    ur = merged["urban_rural"].value_counts()
    print("\nUrban/Rural counts:", ur.to_dict())
    for k, lab in (("U", "urban"), ("R", "rural")):
        if ur.get(k, 0) < 100:
            flags.append(f"Only {ur.get(k, 0)} {lab} clusters - the {lab}-only model will be noisy.")
    if set(ur.index) - {"U", "R"}:
        flags.append(f"Unexpected urban/rural codes {list(set(ur.index) - {'U', 'R'})}.")

    reg = merged["state"].value_counts()
    print(f"\nRegions found ('state'): {len(reg)}")
    print(reg.head(25).to_string())
    if len(reg) < 5:
        flags.append(f"Only {len(reg)} regions: spatial cross-validation will be SKIPPED (needs at least 5).")
    elif len(reg) < 10:
        flags.append(f"Only {len(reg)} regions: spatial cross-validation will be very coarse "
                     f"(each fold holds out about one region).")

    valid = merged.dropna(subset=["wealth_index"])
    g = valid.groupby("state")["wealth_index"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean", ascending=False)
    print("\nFace validity - highest-ranked regions:",
          "; ".join(f"{k} ({v:.2f})" for k, v in g["mean"].head(3).items()))
    print("Face validity - lowest-ranked regions :",
          "; ".join(f"{k} ({v:.2f})" for k, v in g["mean"].tail(3).iloc[::-1].items()))
    print("  -> Do these match what you know of the country? (capital/cities high, remote/arid areas low)")

    gaps = []
    for _, gr in valid.groupby("state"):
        u = gr[gr["urban_rural"] == "U"]["wealth_index"]
        r = gr[gr["urban_rural"] == "R"]["wealth_index"]
        if len(u) >= 3 and len(r) >= 3:
            gaps.append(u.mean() - r.mean())
    if gaps:
        pos = sum(x > 0 for x in gaps)
        print(f"\nIntegrity check: urban clusters wealthier than rural in {pos} of {len(gaps)} regions "
              f"(mean gap {np.mean(gaps):.2f}).")
        if pos < len(gaps):
            flags.append(f"Urban was NOT wealthier than rural in {len(gaps) - pos} region(s): the wealth "
                         f"file and GPS file may be misaligned. Investigate before extracting.")
    else:
        print("\nIntegrity check: no region has >=3 urban and >=3 rural clusters; skipped.")

    # ---------------- Save ----------------
    out = os.path.join(folder, "dhs_cluster_wealth_gps.csv")
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(merged)} clusters to {out}")

    print("\n" + "=" * 70)
    if flags:
        print("REVIEW BEFORE EXTRACTING:")
        for f in flags:
            print("  - " + f)
    else:
        print("ALL CHECKS PASSED. Next:  python extract_features_dhs_country.py " + country)
    print("=" * 70)


if __name__ == "__main__":
    main()
