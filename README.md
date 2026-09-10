# Nigeria Poverty Mapping — Satellite & Machine Learning

Predicting household wealth across Nigeria from free satellite imagery and agro-environmental data, validated against real household survey ground truth.

**Author:** Suleiman Ibrahim Inuwa · MSc Agricultural Economics (First Class), SR University, India
**Contact:** ibrahimsulaimaninuwa2020@gmail.com · [LinkedIn](https://linkedin.com/in/suleiman-ibrahim-inuwa) · ORCID/DOI: 10.63721/25JGEAS0108

---

## Summary

Household surveys are the gold standard for measuring poverty, but they are expensive, infrequent, and leave large gaps in rural, agriculturally dependent regions. This project tests whether free satellite imagery, processed through machine learning, can approximate survey-grade wealth estimates for Nigeria — and specifically whether it works better in agricultural areas, where the data gap is largest.

Using the **2024 Nigeria Demographic and Health Survey (DHS)** as ground truth (1,380 georeferenced clusters), the pipeline extracts satellite and agro-environmental features via **Google Earth Engine** and trains a **Random Forest** model to predict the DHS household wealth index.

## Headline Results

| Metric | RWI proxy baseline | DHS ground-truth model (this study) |
|---|---|---|
| Mean 5-fold cross-validated R² | 0.064 | **0.625** |
| Spatial (state-grouped) CV R² | — | **≈ 0.74** |
| Held-out test R² | 0.043 | 0.815 |
| Clusters (n) | 1,200 | 1,380 |

**Key finding:** the model explains roughly **2.5× more variance in rural, agricultural clusters (R² = 0.493)** than in urban clusters (R² = 0.205) — evidence that satellite-based welfare monitoring is most useful precisely where household surveys are scarcest.

Nighttime lights are the dominant predictor (~64% of feature importance), but population density and NDVI (vegetation health) contribute meaningfully and specifically in rural clusters, supporting the use of agro-environmental features for agricultural policy targeting.

## Data Sources

| Source | What it provides |
|---|---|
| [Nigeria DHS 2024](https://dhsprogram.com) | Ground-truth household wealth index + cluster GPS coordinates |
| [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) (via Earth Engine) | Multispectral daytime imagery (6 bands) |
| [VIIRS](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG) | Nighttime lights |
| [CHIRPS](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY) | Rainfall |
| [ESA WorldCover](https://esa-worldcover.org/) | Land cover classification |
| [SRTM](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) | Elevation and slope |
| [WorldPop](https://www.worldpop.org/) | Population density |

DHS microdata require free registration at dhsprogram.com and are **not redistributed in this repository** in compliance with data-use terms. All other layers are freely accessible via Google Earth Engine.

## Repository Structure

```
nigeria-poverty-mapping/
├── extract_features_dhs.py   # Pulls satellite/agro-environmental features via Earth Engine for each DHS cluster
├── merge_dhs.py               # Merges extracted features with DHS wealth index and metadata
├── test_ee.py                  # Earth Engine authentication and connectivity check
├── notebooks/                  # Exploratory analysis and modelling notebooks
├── data/                       # (gitignored) raw and processed data — not committed, see Data Sources above
└── README.md
```

## Method

1. **Ground truth:** DHS 2024 household wealth index, averaged to cluster level (n = 1,380), merged with cluster GPS coordinates and urban/rural classification.
2. **Feature extraction:** For each cluster coordinate, 13 features pulled via Google Earth Engine — Sentinel-2 bands (B2, B3, B4, B8, B11, B12), NDVI, VIIRS nighttime lights, CHIRPS precipitation, WorldCover land class, SRTM elevation and slope, WorldPop population density.
3. **Model:** Random Forest regression (300 trees, max depth 6, min 5 samples/leaf), evaluated with:
   - Standard 5-fold cross-validation
   - **Spatial (state-grouped) cross-validation** to remove geographic leakage between neighbouring clusters
   - Separate rural/urban stratified models
4. **Uncertainty:** Per-cluster prediction uncertainty from the spread of predictions across the Random Forest's constituent trees.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/suibing15/nigeria-poverty-mapping.git
cd nigeria-poverty-mapping
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install earthengine-api geemap geopandas rasterio scikit-learn pandas numpy

# Authenticate Earth Engine (one-time)
python test_ee.py

# Extract features for DHS clusters (requires your own DHS data access)
python extract_features_dhs.py

# Merge features with DHS wealth index
python merge_dhs.py
```

## Related Publication

Inuwa, S. I., Sani, M. S., Kumar, B. V., HP, K., Sudhamini, Y., & Hamisu, K. (2025). Evaluating Wheat Cultivation Trends and Yield Performance in Nigeria, India, and Pakistan: An ARIMA-Based Approach. *Journal of Geoscience and Eco-Agricultural Studies*, 2(3), 1–21. DOI: [10.63721/25JGEAS0108](https://doi.org/10.63721/25JGEAS0108)

A working paper describing this satellite poverty-mapping study in full (literature review, methodology, results, limitations) is available on request.

## Limitations

- DHS cluster GPS coordinates are randomly displaced for privacy (0–2 km rural, 0–5 km urban), introducing modest spatial noise.
- Cluster-level aggregation does not capture within-community inequality.
- A single-year satellite composite does not capture inter-annual agricultural variability — a clear direction for future work.

## License

Code released for academic and research use. Please cite the associated publication if you use this pipeline in your own work.

## Contact

Open to collaboration, feedback, and PhD supervision discussions. Reach out via [email](mailto:ibrahimsulaimaninuwa2020@gmail.com) or [LinkedIn](https://linkedin.com/in/suleiman-ibrahim-inuwa).
