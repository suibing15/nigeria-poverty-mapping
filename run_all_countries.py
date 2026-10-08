"""
run_all_countries.py - Train and validate EVERY country that is ready, in one command,
and rebuild the cross-country summary table.

USAGE (from the project ROOT folder, .venv activated):
    python run_all_countries.py              train all ready countries, rebuild the summary table
    python run_all_countries.py --status     only show what is ready / unfinished (trains nothing)
    python run_all_countries.py --only rwanda senegal     train just these (other rows are kept)
Options:
    --force          also train countries whose extraction looks unfinished (not recommended)
    --keep-summary   do not reset outputs/model_summary_all_countries.csv before a full run

WHAT IT DOES
  1. Finds countries from outputs/features_dhs_sentinel2_ntl*.csv (Nigeria's file has no suffix)
     and from data/<country>/dhs_cluster_wealth_gps.csv.
  2. Checks each one: extraction complete (same number of rows as the merged file), no failed
     clusters. Unfinished countries are SKIPPED and you are told the exact command to finish them.
  3. Runs  python train_model.py <country>  for each ready country, ONE AT A TIME (they all write to the
     same summary file, so running them in parallel could lose a row). Full logs: outputs/logs/.
  4. Prints the comparison table and warnings, and copies the summary to results/ (ready for git).

A full run takes a few minutes per country. Safe to run again at any time.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

PREFERRED = ["nigeria", "ethiopia", "malawi", "rwanda", "senegal"]
PREFIX = "features_dhs_sentinel2_ntl_"
NIGERIA_FEATURES = os.path.join("outputs", "features_dhs_sentinel2_ntl.csv")
SUMMARY = os.path.join("outputs", "model_summary_all_countries.csv")


def features_path(c):
    return NIGERIA_FEATURES if c == "nigeria" else os.path.join("outputs", f"{PREFIX}{c}.csv")


def merged_path(c):
    p = os.path.join("data", c, "dhs_cluster_wealth_gps.csv")
    if c == "nigeria" and not os.path.isfile(p):
        p = os.path.join("data", "dhs", "dhs_cluster_wealth_gps.csv")   # Nigeria's original location
    return p


def discover():
    found = set()
    if os.path.isfile(NIGERIA_FEATURES):
        found.add("nigeria")
    for f in glob.glob(os.path.join("outputs", PREFIX + "*.csv")):
        name = os.path.basename(f)[len(PREFIX):-4]
        if not name.endswith("_raw"):          # *_raw.csv = untouched backup made by fill_missing_precipitation.py
            found.add(name)
    for f in glob.glob(os.path.join("data", "*", "dhs_cluster_wealth_gps.csv")):
        found.add(os.path.basename(os.path.dirname(f)))
    if os.path.isfile(os.path.join("data", "dhs", "dhs_cluster_wealth_gps.csv")):
        found.add("nigeria")
    found.discard("dhs")
    return [c for c in PREFERRED if c in found] + sorted(found - set(PREFERRED))


def n_rows(path):
    try:
        return len(pd.read_csv(path, usecols=[0]))
    except Exception:                       # noqa: BLE001
        return None


def check(c):
    """Return (state, detail, ready_bool)."""
    f, m = features_path(c), merged_path(c)
    if not os.path.isfile(f):
        if os.path.isfile(m):
            return "NOT EXTRACTED", f"python run_country.py {c}", False
        return "NO DATA", f"put the DHS files in data/{c}/ then: python run_country.py {c}", False
    df = pd.read_csv(f)
    n_f = len(df)
    n_err = int(df["error_msg"].notnull().sum()) if "error_msg" in df.columns else 0
    n_m = n_rows(m) if os.path.isfile(m) else None
    if n_err:
        return "HAS FAILED CLUSTERS", f"{n_err} failed; run: python extract_features_dhs_country.py {c}", False
    if n_m is not None and n_f < n_m:
        return "INCOMPLETE", f"{n_f}/{n_m} clusters; run: python extract_features_dhs_country.py {c}", False
    return "READY", f"{n_f} clusters", True


def print_status(countries):
    print(f"\n{'country':<12} {'state':<22} detail")
    print("-" * 78)
    out = {}
    for c in countries:
        state, detail, ok = check(c)
        out[c] = (state, detail, ok)
        print(f"{c:<12} {state:<22} {detail}")
    return out


def run_training(c):
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    log = os.path.join("outputs", "logs", f"train_{c}.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t0 = time.time()
    r = subprocess.run([sys.executable, "train_model.py", c], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(r.stdout + "\n" + r.stderr)
    return r.returncode, time.time() - t0, log, r


def main():
    ap = argparse.ArgumentParser(description="Train every ready country and rebuild the summary table.")
    ap.add_argument("--only", nargs="+", help="train only these countries")
    ap.add_argument("--status", action="store_true", help="show status only; train nothing")
    ap.add_argument("--force", action="store_true", help="also train countries that look unfinished")
    ap.add_argument("--keep-summary", action="store_true", help="do not reset the summary file")
    args = ap.parse_args()

    if not os.path.isfile("train_model.py"):
        sys.exit("ERROR: run this from the project root folder (train_model.py not found here).")

    countries = [c.lower() for c in args.only] if args.only else discover()
    if not countries:
        sys.exit("No countries found. Expected outputs/features_dhs_sentinel2_ntl*.csv or data/<country>/ folders.")
    states = print_status(countries)
    if args.status:
        return

    todo = [c for c in countries if states[c][2] or (args.force and os.path.isfile(features_path(c)))]
    skipped = [c for c in countries if c not in todo]
    if not todo:
        sys.exit("\nNothing is ready to train. Finish the extraction for the countries above first.")
    if skipped:
        print(f"\nSkipping (not ready): {', '.join(skipped)}  - see the commands above.")

    # fresh summary for a full run, so stale rows from earlier experiments cannot linger
    if not args.only and not args.keep_summary and os.path.isfile(SUMMARY):
        backup = SUMMARY.replace(".csv", f".backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.move(SUMMARY, backup)
        print(f"\nExisting summary moved to {backup}; building a fresh one.")

    failed = []
    print(f"\nTraining {len(todo)} countr{'y' if len(todo) == 1 else 'ies'}, one at a time...")
    for i, c in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {c} ...", flush=True)
        rc, secs, log, r = run_training(c)
        if rc == 0:
            print(f"    done in {secs:.0f}s   (log: {log})")
        else:
            failed.append(c)
            print(f"    FAILED (exit {rc}). Last lines of the log:")
            for line in (r.stdout + r.stderr).strip().splitlines()[-8:]:
                print("      " + line)

    # ---- comparison table ----
    if not os.path.isfile(SUMMARY):
        sys.exit("\nNo summary file was produced - see the errors above.")
    s = pd.read_csv(SUMMARY).set_index("country")
    order = [c for c in PREFERRED if c in s.index] + sorted(set(s.index) - set(PREFERRED))
    s = s.loc[order]
    t = pd.DataFrame({
        "clusters": s["n_clusters"].astype(int),
        "random CV": s["random_cv_r2_mean"].round(3),
        "spatial CV": s["spatial_cv_r2_mean"].round(3),
        "test R2": s["test_r2"].round(3),
        "rural R2": s["R_cv_r2"].round(3),
        "urban R2": s["U_cv_r2"].round(3),
        "regions": s["n_regions"].astype(int),
    })
    print("\n" + "=" * 78)
    print("CROSS-COUNTRY SUMMARY")
    print("=" * 78)
    print(t.to_string())

    # ---- warnings worth reading ----
    warns = []
    for c in s.index:
        r = s.loc[c]
        if pd.notna(r.get("integrity_units_tested")) and r["integrity_units_urban_wealthier"] < r["integrity_units_tested"]:
            warns.append(f"{c}: urban NOT wealthier than rural in "
                         f"{int(r['integrity_units_tested'] - r['integrity_units_urban_wealthier'])} region(s) "
                         f"- check that the wealth and GPS files are aligned.")
        if r["n_regions"] < 10:
            warns.append(f"{c}: only {int(r['n_regions'])} regions - its spatial CV is very coarse.")
        if pd.notna(r.get("bootstrap_pop_gt_nightlights_share")) and 0.2 < r["bootstrap_pop_gt_nightlights_share"] < 0.8:
            warns.append(f"{c}: population vs nightlights importance order is unstable "
                         f"({100 * r['bootstrap_pop_gt_nightlights_share']:.0f}% of resamples) - do not claim which leads.")
    if warns:
        print("\nREAD THESE:")
        for w in warns:
            print("  - " + w)

    os.makedirs("results", exist_ok=True)
    shutil.copy(SUMMARY, os.path.join("results", "model_summary_all_countries.csv"))
    print("\nSummary copied to results/model_summary_all_countries.csv (this is the file to commit to git).")
    if failed:
        print(f"\nFAILED: {', '.join(failed)} - read their logs in outputs/logs/ .")
        sys.exit(1)
    print("All done. Send me results/model_summary_all_countries.csv to rebuild the paper.")


if __name__ == "__main__":
    main()
