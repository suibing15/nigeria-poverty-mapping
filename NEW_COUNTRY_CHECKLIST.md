# Adding a country - checklist

Run everything from the project root (`C:\Nigeria Poverty map`) with `.venv` activated.

## 1. Put the downloads in `data/<country>/`
Only two things are needed, from the DHS download page (take the **.dta** / shapefile versions):
- **Household Recode (HR)** - `.dta`  (NOT "Household Member Recode / PR")
- **Geographic Data (GE)** - shapefile (`.shp`, `.dbf`, `.shx`, `.prj` together)

Zipped or unzipped does not matter, and neither do folder names. One survey per folder.
The folder name is the country name printed in results (`rwanda`, not `ruwanda`).

## 2. Run
    python run_country.py senegal
    python run_country.py rwanda

It merges, **pauses so you can read the checks**, extracts satellite features (hours, resumable),
then trains and adds a row to `outputs/model_summary_all_countries.csv`.

Quick Earth Engine test first (optional):  `python run_country.py senegal --limit 20`

If anything stops - laptop closed, network blip, "clusters still failed" - run the SAME command again.

## 3. What to read at the pause (after the merge)
- `REVIEW BEFORE EXTRACTING` list is empty, or you understand every item.
- Regions look like real places; face-validity ranking is sensible (capital/cities high, remote or arid areas low).
- `Integrity check: urban wealthier than rural in N of N regions` - should be (nearly) all. If not, STOP:
  the wealth file and GPS file may be misaligned.
- Cluster count is close to the survey's published number.
- Satellite year printed by the extraction step = the year the fieldwork began (override with `--year`).

## 4. After training
- `INTEGRITY` line again should be all units positive.
- `IMPORTANCE STABILITY` near 0% or 100% = stable order; near 50% = do not claim which feature leads.
- Send back `outputs/model_summary_all_countries.csv` and the new `features_*.csv` files.

## Step-by-step (if you prefer separate commands)
    python merge_dhs_country.py <country>
    python extract_features_dhs_country.py <country> [--year YYYY] [--limit N]
    python train_model.py <country>

## If something goes wrong
| Message | Meaning / fix |
|---|---|
| `Folder 'data/xxx' not found` | misspelled folder; the message suggests the closest name |
| `Several HR files found` | more than one survey in the folder; keep one, or pass `--hr PATH --ge PATH` |
| `HR is for country code X but GPS is for Y` | files from different countries |
| `Fewer than 90% of clusters matched` | HR and GPS are from different surveys/rounds |
| `does not contain ... hv271` | wealth index not in that household file |
| `could not start Earth Engine` | run `earthengine authenticate` once |
| `was started with satellite year(s) [..]` | you changed `--year` midway; delete the features CSV or use the old year |
| exit code 2 / "clusters FAILED" | transient network failures; run the same command again |
