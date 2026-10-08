"""
run_country.py - ONE command per country: merge -> (you review) -> extract -> train.

USAGE (from the project ROOT folder, .venv activated):
    python run_country.py senegal
    python run_country.py rwanda

What happens:
  1. merge_dhs_country.py   builds data/<country>/dhs_cluster_wealth_gps.csv and prints the checks.
  2. It then PAUSES and asks whether to continue, because extraction takes hours. Read the checks
     (regions look right? face-validity ranking sensible? 'REVIEW BEFORE EXTRACTING' empty?).
     Skipped automatically when you are resuming an extraction that already started.
  3. extract_features_dhs_country.py   pulls the satellite features (resumable).
  4. train_model.py   trains, validates and appends one row to outputs/model_summary_all_countries.csv.

If it stops (laptop closed, network blip, "clusters still failed"), run the SAME command again.

Options:
    --year 2023     satellite year (default: survey year found in the GPS file)
    --yes           do not pause after the merge checks
    --limit 20      smoke test: extract only 20 clusters, then stop (no training)
    --skip-merge    reuse the existing merged CSV
    --hr PATH --ge PATH   explicit files, only needed if the folder holds several surveys
"""

import argparse
import os
import subprocess
import sys


def run(cmd, what):
    print("\n" + "#" * 72)
    print(f"# {what}")
    print("#   " + " ".join(cmd))
    print("#" * 72)
    rc = subprocess.call(cmd)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--hr")
    ap.add_argument("--ge")
    args = ap.parse_args()
    country = args.country.lower()
    py = sys.executable
    features_csv = os.path.join("outputs", f"features_dhs_sentinel2_ntl_{country}.csv")
    resuming = os.path.exists(features_csv)

    if not args.skip_merge:
        cmd = [py, "merge_dhs_country.py", country]
        if args.hr:
            cmd += ["--hr", args.hr]
        if args.ge:
            cmd += ["--ge", args.ge]
        if run(cmd, f"STEP 1/3  Merge wealth + GPS for {country}") != 0:
            sys.exit("\nMerge failed - fix the problem above and run the same command again.")

    if not (args.yes or resuming):
        ans = input("\nRead the checks above. Continue to the (long) satellite extraction? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit("Stopped before extraction. Run the same command again when you are ready.")

    cmd = [py, "extract_features_dhs_country.py", country]
    if args.year:
        cmd += ["--year", str(args.year)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    rc = run(cmd, f"STEP 2/3  Satellite feature extraction for {country}")
    if rc == 2:
        sys.exit("\nSome clusters failed. Run the SAME command again - only those are retried.")
    if rc != 0:
        sys.exit("\nExtraction stopped with an error - read the message above.")
    if args.limit:
        sys.exit("\nSmoke test done. Run again without --limit for the full extraction and training.")

    if run([py, "train_model.py", country], f"STEP 3/3  Train and validate for {country}") != 0:
        sys.exit("\nTraining failed - read the message above.")
    print(f"\nDONE. {country} has been added to outputs/model_summary_all_countries.csv")


if __name__ == "__main__":
    main()
