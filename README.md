# Satellite-Based Wealth Prediction Across African Countries

How well does one satellite-and-machine-learning pipeline, applied unchanged, predict household wealth in different countries?
This repository holds the pipeline and results for DHS-validated models of household wealth (DHS wealth index, cluster level)
built from free satellite and agro-environmental data, in 5 countries and 3,904 clusters.

**Author:** Suleiman Ibrahim Inuwa, MSc Agricultural Economics (First Class), SR University, India
**Contact:** ibrahimsulaimaninuwa2020@gmail.com | [LinkedIn](https://linkedin.com/in/suleiman-ibrahim-inuwa) | ORCID 0009-0000-7857-6817
**Status:** working paper in preparation. More countries are planned.

---

## Results (one script, identical settings, every country)

| Country | Clusters | Random CV R² | Spatial CV R² | Rural-only R² | Urban-only R² | Rural R², location-only baseline | Top feature |
|---|---|---|---|---|---|---|---|
| Nigeria (DHS 2024) | 1,380 | 0.80 | 0.76 | 0.70 | 0.64 | 0.47 | nightlights (66%) |
| Ethiopia (DHS 2024-25) | 796 | 0.77 | 0.63 | 0.62 | 0.60 | 0.54 | population density (49%) |
| Malawi (DHS 2024) | 769 | 0.69 | 0.58 | 0.27 | 0.07 | 0.23 | population density (65%) |
| Rwanda (DHS 2025) | 560 | 0.58 | 0.36 | 0.02 | 0.33 | 0.12 | nightlights (68%) |
| Senegal (Continuous DHS 2023) | 399 | 0.83 | 0.77 | 0.70 | 0.59 | 0.55 | nightlights (87%) |

*CV = cross-validation. "Spatial" holds out whole states or regions with a fixed fold rule (`spatial_folds.py`), so results do not depend on the software version. "Location-only" is the same model given only latitude and longitude.
Wealth-index levels are country-specific and not comparable across countries; R² is.*

**What the comparison shows**
- Overall accuracy ranges from 0.58 (Rwanda) to 0.83 (Senegal). Sample size does not explain the order: Senegal has the smallest sample and the highest accuracy.
- Inside rural and urban areas the countries split in 2. Nigeria, Ethiopia and Senegal do well in both. Malawi and Rwanda do not, and in rural Malawi and Rwanda the satellite features add little or nothing beyond location.
- Across countries, a model trained elsewhere orders an unseen country's clusters reasonably well but gets the level wrong (see `results/cross_country_transfer.csv`).
- The cause of the weak rural accuracy in Malawi and Rwanda is not yet identified. Candidate causes (GPS displacement, noise in cluster means, an old population layer) are untested.

Full results are in `results/`.

## Correction notice

An earlier version of this README and an early Nigeria-only working paper reported a cross-validated R² of 0.625 and a rural advantage of about 2.5x (R² 0.493 rural vs 0.205 urban). Those figures came from cross-validation folds that were **not shuffled**, applied to a feature file sorted by state.
With shuffling enforced, Nigeria's random-CV R² is 0.80 and the rural/urban values are 0.70 / 0.64. All numbers above come from `train_model.py`, which enforces shuffling in code.

## Method (identical for every country)

1. **Target:** DHS wealth index (`hv271`/100000) averaged to the cluster, with GPS and the urban/rural flag.
2. **Features (13), via Google Earth Engine,** for a 4.8 km square around each cluster: Sentinel-2 bands and NDVI, VIIRS nightlights, CHIRPS rainfall, ESA WorldCover, SRTM elevation and slope, WorldPop. One calendar year matched to the year each survey's fieldwork began (WorldPop 2019 to 2020 for all).
3. **Model:** Random Forest (300 trees, max depth 6, min 5 samples per leaf, seed 42). No country-specific tuning.
4. **Validation:** shuffled 5-fold CV; spatial CV; 20% held-out test; separate rural and urban models; leave-one-country-out transfer.
5. **Diagnostics:** location-only baseline; urban-rural variance split; importance stability (40 bootstrap refits); integrity check (urban wealthier than rural inside the same unit); equal-sample-size control.
6. **Missing values:** if a few clusters have an empty rainfall value because of the coastline (16 in Senegal, all urban, 14 in Dakar), `train_model.py` warns when the dropped clusters are not typical. `fill_missing_precipitation.py` fills them from the nearest cluster and `missing_rainfall_sensitivity.py` shows the effect.

## Run a new country

Put the DHS Household Recode (`.dta`) and Geographic Data (shapefile) in `data/<country>/` (zipped or not; folder names do not matter), then:

```
python run_country.py <country>
python run_all_countries.py          # trains every ready country and rebuilds the summary table
```

See `NEW_COUNTRY_CHECKLIST.md` for what to check and how to fix common problems.

## Repository layout

```
run_country.py                    one command per country (merge -> extract -> train)
run_all_countries.py              train all ready countries and rebuild the summary table
merge_dhs_country.py              builds the cluster wealth + GPS table; auto-finds the DHS files
extract_features_dhs_country.py   Earth Engine feature extraction; retries and resumes safely
fill_missing_precipitation.py     fills a few empty coastal rainfall values from the nearest cluster
train_model.py                    the canonical training and validation script (v3)
spatial_folds.py                  fixed spatial-fold rule: identical folds on every computer
cross_country_transfer.py         leave-one-country-out transfer test
sample_size_control.py            equal-sample-size control
missing_rainfall_sensitivity.py   effect of how missing rainfall is handled
NEW_COUNTRY_CHECKLIST.md          step-by-step guide
extract_features_dhs.py, merge_dhs.py                   original Nigeria scripts
*_ethiopia.py, *_malawi.py                              scripts used for those countries
results/                          model_summary_all_countries.csv and the other result tables
archive/                          superseded scripts, kept for transparency
```

## Data

DHS microdata require free registration at [dhsprogram.com](https://dhsprogram.com) and are **not** redistributed here. Satellite and agro-environmental layers are free through Google Earth Engine. Raw data and cluster-level feature tables are git-ignored.

## Limitations

DHS cluster coordinates are randomly displaced for privacy (urban up to 2 km, rural up to 5 km); wealth is aggregated to the cluster; one satellite year per country; 5 countries are too few to explain cross-country differences; within-settlement estimates rest on smaller samples (for example 169 urban clusters in Malawi).

## Related publication

Inuwa, S. I., Sani, M. S., Kumar, B. V., HP, K., Sudhamini, Y., & Hamisu, K. (2025). Evaluating Wheat Cultivation Trends and Yield Performance in Nigeria, India, and Pakistan: An ARIMA-Based Approach. *Journal of Geoscience and Eco-Agricultural Studies*, 2(3), 1-21. DOI: [10.63721/25JGEAS0108](https://doi.org/10.63721/25JGEAS0108)
