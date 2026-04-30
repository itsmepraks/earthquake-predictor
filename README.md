# Earthquake Predictor

Final project by **Anurag Dhungana** and **Prakriti Bista** (May 2025).

ML models that predict building damage from the **2015 Gorkha earthquake** (Mw 7.8) in Nepal. Given a building's structural and location features, the models classify it as Low, Medium, or High damage risk. Training data is the DrivenData "Richter's Predictor" competition dataset (~260k buildings).

Full writeup — methodology, results, limitations, references: [`whitepaper.md`](whitepaper.md).

This is a research / coursework project. The models are tied to one specific event and the dataset's anonymized geographic identifiers — don't use the predictions for real safety decisions.

## What it actually does

The Streamlit app in `app/` lets you:

- Pick a model (Logistic Regression, Random Forest, LightGBM, LightGBM tuned, or SVM)
- Adjust building features (foundation type, roof, age, floors, superstructure materials, etc.) and geo IDs
- Get a risk prediction with the input summary
- View predicted risk distribution across the dataset, model metrics, feature importances, and the data dictionary

A geographic risk map was planned but not implemented — the dataset's `geo_level_1/2/3_id` values are anonymized and we couldn't reliably map them to Nepali admin PCODEs. For the same reason, the SRTM terrain features and admin shapefiles in this repo were processed but never made it into the final models.

Earthquake parameters in the feature set (`main_eq_magnitude`, `main_eq_depth`, `main_eq_epicenter_lat/lon`) are the Gorkha mainshock values applied uniformly to every building. The models don't see spatial variation in ground shaking — that's the biggest limitation and probably why performance plateaus around 72%.

## Models

Trained artifacts live in `models/`. Current numbers from `reports/model_comparison_metrics.csv`:

| Model                | Preprocessing     | Accuracy | ROC AUC | Weighted F1 |
|----------------------|-------------------|----------|---------|-------------|
| Logistic Regression  | OHE + Scale       | 0.697    | 0.872   | 0.69        |
| LightGBM (untuned)   | Ordinal + Scale   | 0.711    | 0.880   | 0.71        |
| Random Forest        | Ordinal + Scale   | 0.717    | 0.845   | 0.71        |
| LinearSVC            | OHE + Scale       | 0.721    | N/A     | 0.72        |
| LightGBM (tuned)     | Ordinal + Scale   | 0.725    | 0.880   | 0.73        |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The geo libraries (`geopandas`, `rasterio`, `rioxarray`) can be a pain on some systems — if pip fails, conda usually works.

## Running the app

```bash
streamlit run app/app.py
```

It expects `data/processed/buildings_features_earthquakes.csv` to exist (it's already in the repo) and the `.joblib` files in `models/`. If a model fails to load it'll log a warning and just disable that option.

## Repo layout

```
app/
  app.py                          Streamlit UI
data/
  data_dictionary.md              Field descriptions
  usgs_himalayan_earthquakes_1900_present_M4.0.csv
  nsc_nepal_earthquakes_1994_present.csv
  nepal_land_data/                Driven Data train/test CSVs
  npl_adm_nd_20240314_ab_shp/     Nepal admin boundary shapefiles
  srtm_raw/                       SRTM elevation tiles
  processed/                      Outputs from src/ scripts
models/                           Trained .joblib models + preprocessors
prototype/
  ml_prototype.py                 Standalone toy LR model on USGS data
src/
  data/                           Download scripts (USGS, NSC, SRTM)
  data_processing.py              Catalog cleaning
  feature_engineering.py
  explore_boundaries.py
  features/                       Terrain feature extraction + merging
  modeling/                       train_baseline, train_lightgbm, train_random_forest, train_svm, tune_lightgbm
  visualization/                  Scripts that generate the figures in reports/images/
reports/
  model_comparison_metrics.csv
  images/                         Figures (target dist, ROC, PR, learning curves, feature importance, etc.)
requirements.txt
```

The various `*.md` files at the root are project notes and writeups:

- [`whitepaper.md`](whitepaper.md) — full final report (methodology, results, ethics, future work)
- [`proposal.md`](proposal.md) — original project proposal
- [`v1.md`](v1.md), [`v2.md`](v2.md) — earlier draft writeups
- [`todo.md`](todo.md), [`graph_plan_todo.md`](graph_plan_todo.md), [`test_data.md`](test_data.md) — working notes

## Reproducing the pipeline

Roughly:

1. Pull catalogs with the scripts in `src/data/` (or use what's already in `data/`).
2. Run `src/data_processing.py` and `src/feature_engineering.py` to build `data/processed/buildings_features_earthquakes.csv`.
3. Train models with the scripts in `src/modeling/`. `tune_lightgbm.py` does the LightGBM hyperparameter search (RandomizedSearchCV, 20 iterations, 3-fold CV).
4. Regenerate figures with the scripts in `src/visualization/`.

The trainers write `.joblib` files into `models/`, which is what the app reads.

`src/features/build_terrain_features.py` and `merge_terrain_features.py` produce `data/processed/terrain_features_by_adm3.csv` from SRTM, but those features were never joined into the training data — same `geo_level_id` mapping problem. The scripts still run; their output just doesn't feed anywhere.

## Loading a model directly

```python
import joblib

model = joblib.load("models/lightgbm_tuned_model.joblib")
preproc = joblib.load("models/lightgbm_preprocessor.joblib")

X = preproc.transform(X_raw)        # X_raw must have the columns in preproc.feature_names_in_
y_pred = model.predict(X)
```

Class labels are 0/1/2 → Low/Medium/High for LightGBM, RF, and SVM. Logistic Regression uses 1/2/3.

## The prototype script

`prototype/ml_prototype.py` is separate from the main app. It pulls earthquakes from the USGS API, labels anything M ≥ 5.0 as "high risk," and trains a logistic regression on `mag`/`depth`/`lat`/`lon`. It was an early sketch — keep or ignore.

## Data sources

- USGS earthquake catalog: https://earthquake.usgs.gov/fdsnws/event/1/
- Nepal NSC catalog
- Driven Data Richter's Predictor: Modeling Earthquake Damage (Nepal building dataset)
- SRTM 30m elevation
- Nepal admin boundaries shapefile

## License

No license file yet. If you plan to share this, add one.
