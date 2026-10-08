"""
extract_features_dhs_country.py - Extract satellite + agro-environmental features for the
DHS clusters of ANY country, using exactly the same method as Nigeria, Ethiopia and Malawi.

USAGE (from the project ROOT folder, .venv activated, AFTER merge_dhs_country.py):
    python extract_features_dhs_country.py senegal
    python extract_features_dhs_country.py rwanda
Options:
    --year 2023     satellite year. Default = the survey year (DHSYEAR) saved by the merge step.
    --limit 20      only process the first 20 clusters (a quick smoke test of Earth Engine).
                    A later run WITHOUT --limit continues where this one stopped.
    --project NAME  Earth Engine cloud project (default: earth-engine-legacy-project)

METHOD (identical for every country - do not change it between countries):
    window = a square of +/-2.4 km around the cluster point; one calendar year of Sentinel-2
    (cloud <20%, median), VIIRS nightlights, CHIRPS rainfall; WorldPop 2019-2020; SRTM; WorldCover.

SAFE TO INTERRUPT: progress is saved every 25 clusters. Close the laptop, come back tomorrow, run
the same command: it resumes. Clusters that FAILED (e.g. a network blip) are retried on the next
run; each Earth Engine call is also retried 3 times automatically before a cluster is marked failed.
If you change --year after starting, the script refuses to mix years and tells you what to do.

OUTPUT: outputs/features_dhs_sentinel2_ntl_<country>.csv
Exit code 0 = every cluster extracted; 2 = some clusters still failed (just run it again).
"""

import argparse
import os
import sys
import time

import pandas as pd

try:
    import ee
except ImportError:
    sys.exit("Missing package 'earthengine-api'. Install it with:  pip install earthengine-api")

FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "avg_rad", "NDVI", "Map",
            "precipitation", "elevation", "slope", "population"]
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BUFFER_M = 2400
MAX_ATTEMPTS = 3
CHECKPOINT_EVERY = 25


def with_retry(fn, attempts=MAX_ATTEMPTS):
    """Run an Earth Engine call, retrying on transient failures (network blips, rate limits)."""
    last = None
    for a in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:      # noqa: BLE001 - we want to catch every transient failure
            last = e
            if a < attempts:
                time.sleep(3 * a)
    raise last


def get_features(lat, lon, year_start, year_end):
    pt = ee.Geometry.Point([lon, lat])
    box = pt.buffer(BUFFER_M).bounds()

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(box).filterDate(year_start, year_end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).median())
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterBounds(box).filterDate(year_start, year_end).select("avg_rad").mean())
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterBounds(box).filterDate(year_start, year_end).select("precipitation").sum())
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(srtm)
    # WorldPop's consistent global layer stops at 2020 (same for every country)
    worldpop = (ee.ImageCollection("WorldPop/GP/100m/pop")
                .filterBounds(box).filterDate("2019-01-01", "2020-12-31").mean())

    try:
        s2_vals = with_retry(lambda: s2.select(S2_BANDS).reduceRegion(
            ee.Reducer.mean(), box, 10, maxPixels=1e9).getInfo())
        ntl_val = with_retry(lambda: viirs.reduceRegion(
            ee.Reducer.mean(), box, 500, maxPixels=1e9).getInfo())
        ndvi_val = with_retry(lambda: ndvi.reduceRegion(
            ee.Reducer.mean(), box, 10, maxPixels=1e9).getInfo())
        lc_val = with_retry(lambda: worldcover.reduceRegion(
            ee.Reducer.mode(), box, 10, maxPixels=1e9).getInfo())
        rain_val = with_retry(lambda: chirps.reduceRegion(
            ee.Reducer.mean(), box, 5000, maxPixels=1e9).getInfo())
        elev_val = with_retry(lambda: srtm.reduceRegion(
            ee.Reducer.mean(), box, 30, maxPixels=1e9).getInfo())
        slope_val = with_retry(lambda: slope.reduceRegion(
            ee.Reducer.mean(), box, 30, maxPixels=1e9).getInfo())
        pop_val = with_retry(lambda: worldpop.reduceRegion(
            ee.Reducer.mean(), box, 100, maxPixels=1e9).getInfo())
        return {**s2_vals, **ntl_val, **ndvi_val, **lc_val, **rain_val,
                **elev_val, **slope_val, **pop_val}
    except Exception as e:          # noqa: BLE001
        res = {k: None for k in FEATURES}
        res["error_msg"] = str(e)[:300]
        return res


def main():
    ap = argparse.ArgumentParser(description="Extract satellite features for DHS clusters.")
    ap.add_argument("country", help="same name as the folder under data/, e.g. senegal")
    ap.add_argument("--year", type=int, help="satellite year (default: survey year from the merge step)")
    ap.add_argument("--limit", type=int, help="only process the first N clusters (smoke test)")
    ap.add_argument("--project", default="earth-engine-legacy-project")
    args = ap.parse_args()
    country = args.country.lower()

    input_csv = os.path.join("data", country, "dhs_cluster_wealth_gps.csv")
    output_csv = os.path.join("outputs", f"features_dhs_sentinel2_ntl_{country}.csv")
    if not os.path.isfile(input_csv):
        sys.exit(f"\nERROR: {input_csv} not found. Run first:  python merge_dhs_country.py {country}")

    dhs = pd.read_csv(input_csv)
    print(f"[{country}] Loaded {len(dhs)} DHS clusters from {input_csv}")

    # ---- satellite year ----
    if args.year:
        year, source = args.year, "--year option"
    elif "dhs_year" in dhs.columns and dhs["dhs_year"].notna().any():
        year, source = int(dhs["dhs_year"].dropna().mode().iloc[0]), "DHSYEAR in the GPS file"
    else:
        sys.exit("\nERROR: no survey year available. Re-run with  --year YYYY  (the year the DHS fieldwork began).")
    year_start, year_end = f"{year}-01-01", f"{year}-12-31"
    print(f"[{country}] Satellite window: {year_start} to {year_end}  (from {source})")
    print("          Policy: calendar year in which the DHS fieldwork began. Override with --year if needed.")
    if year < 2019:
        print("  WARNING: before 2019 Sentinel-2 surface-reflectance coverage is patchy in many regions; "
              "expect more clusters with missing optical bands.")

    try:
        ee.Initialize(project=args.project)
    except Exception as e:          # noqa: BLE001
        sys.exit(f"\nERROR: could not start Earth Engine ({e}).\n"
                 f"Run  earthengine authenticate  once, then try again.")

    # ---- resume: only SUCCESSFUL clusters of the SAME year count as done ----
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(output_csv):
        done = pd.read_csv(output_csv)
        if "sat_year" in done.columns and done["sat_year"].notna().any():
            old_years = sorted(set(int(y) for y in done["sat_year"].dropna()))
            if old_years != [year]:
                sys.exit(f"\nERROR: {output_csv} was started with satellite year(s) {old_years} but this run "
                         f"uses {year}. Mixing years would corrupt the data.\nDelete {output_csv} to start "
                         f"over with {year}, or re-run with --year {old_years[0]} to continue.")
        good = done[done["error_msg"].isnull()] if "error_msg" in done.columns else done
        failed = len(done) - len(good)
        done_clusters = set(good["cluster"])
        records = good.to_dict("records")
        print(f"[{country}] Found progress: {len(good)} clusters succeeded, {failed} failed "
              f"and will be retried. Resuming...")
    else:
        done_clusters, records = set(), []

    todo = dhs if not args.limit else dhs.head(args.limit)
    pending = sum(1 for c in todo["cluster"] if c not in done_clusters)
    print(f"[{country}] {pending} cluster(s) to extract this run.")

    new_since_save = 0
    t0 = time.time()
    processed = 0
    for _, row in todo.iterrows():
        if row["cluster"] in done_clusters:
            continue
        feats = get_features(row["latitude"], row["longitude"], year_start, year_end)
        feats.update({"cluster": row["cluster"], "latitude": row["latitude"],
                      "longitude": row["longitude"], "wealth_index": row["wealth_index"],
                      "urban_rural": row["urban_rural"], "state": row["state"], "sat_year": year})
        records.append(feats)
        new_since_save += 1
        processed += 1
        if processed % 50 == 0:
            rate = (time.time() - t0) / processed
            print(f"  {processed}/{pending} processed  (~{rate:.1f}s per cluster, "
                  f"about {rate * (pending - processed) / 60:.0f} min left)")
        if new_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(records).to_csv(output_csv, index=False)
            new_since_save = 0
        time.sleep(0.1)

    out = pd.DataFrame(records)
    out.to_csv(output_csv, index=False)
    print(f"\n[{country}] Saved {len(out)} rows to {output_csv}")

    n_failed = int(out["error_msg"].notnull().sum()) if "error_msg" in out.columns else 0
    ok = out[out["error_msg"].isnull()] if "error_msg" in out.columns else out
    incomplete = int(ok[FEATURES].isnull().any(axis=1).sum())
    if incomplete:
        which = ok[FEATURES].isnull().sum()
        print(f"Note: {incomplete} extracted cluster(s) have at least one empty feature "
              f"({which[which > 0].to_dict()}). Usually no cloud-free Sentinel-2 in the year or no CHIRPS pixel. "
              f"train_model.py drops these automatically.")

    if args.limit:
        print(f"\nSmoke test finished ({len(out)} clusters). If it looks right, run again WITHOUT --limit.")
    if n_failed:
        print(f"\n*** {n_failed} cluster(s) FAILED (see the error_msg column). Run the same command again - "
              f"only those clusters are retried. ***")
        sys.exit(2)
    if not args.limit:
        print(f"\nAll clusters extracted - 0 errors. Next:  python train_model.py {country}")


if __name__ == "__main__":
    main()
