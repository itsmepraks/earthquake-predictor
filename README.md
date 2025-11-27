# Earthquake Predictor

A research / prototype repository for exploratory earthquake prediction and analysis using seismic catalog data, feature engineering, and classical ML models. This README is tailored to the current repository contents (Python-only code, model artifacts, and datasets) and documents how to reproduce experiments and run the included prototype scripts and application.

Status
- Language: Python (100%)
- Project: Research / prototype (not production / not for operational safety decisions)
- Note: There is currently no LICENSE file in the repository — consider adding one (e.g., MIT) if you plan to publish this publicly.

Quick links
- Data dictionary: data/data_dictionary.md
- Example datasets: data/usgs_himalayan_earthquakes_1900_present_M4.0.csv, data/nsc_nepal_earthquakes_1994_present.csv
- Prototype trainer: prototype/ml_prototype.py
- App entrypoint: app/app.py
- Trained model artifacts: models/*.joblib
- Requirements: requirements.txt

Important disclaimer
This repository is intended for research and experimentation only. Models and outputs here are not validated for operational earthquake forecasting or public-safety decisions. Do not use outputs for evacuation or emergency planning.

Table of contents
- Overview
- Repository structure (actual)
- Requirements & setup
- Data: what’s included and how to prepare it
- Running the prototype experiments
- Running the app
- Models & artifacts
- Development notes: missing/optional files and suggested additions
- Contributing
- Acknowledgements

Overview
--------
This project explores simple ML approaches and feature engineering applied to seismic catalog data. The repository includes:
- data/ with raw and processed CSVs plus a data dictionary
- feature engineering and data processing scripts under src/
- prototype training script(s) under prototype/
- a small app entrypoint at app/app.py
- pre-trained model artifacts (joblib) under models/
- lightweight reports under reports/

Repository structure (reflects current tree)
- app/
  - app.py
- data/
  - data_dictionary.md
  - usgs_himalayan_earthquakes_1900_present_M4.0.csv
  - nsc_nepal_earthquakes_1994_present.csv
  - npl_adm_nd_20240314_ab_shp/ (shapefiles)
  - nepal_land_data/
  - processed/ (placeholder for processed outputs)
  - srtm_raw/
- models/
  - baseline_preprocessor.joblib
  - lightgbm_model.joblib
  - lightgbm_preprocessor.joblib
  - lightgbm_tuned_model.joblib
  - logistic_regression_model.joblib
  - random_forest_model.joblib
  - rf_preprocessor.joblib
  - svm_model.joblib
  - svm_preprocessor.joblib
- prototype/
  - ml_prototype.py
- reports/
  - model_comparison_metrics.csv
  - images/
- src/
  - data_processing.py
  - feature_engineering.py
  - explore_boundaries.py
  - features/ (module)
  - modeling/ (module)
  - visualization/ (module)
- requirements.txt
- graph_plan_todo.md, proposal.md, whitepaper.md, v1.md, v2.md, todo.md, test_data.md

Requirements & setup
--------------------
Recommended: create and activate a virtual environment and install pinned requirements.

Unix / macOS:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (Powershell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to install only a minimal set of common packages:
```
pip install numpy pandas scikit-learn joblib matplotlib geopandas rasterio
```
(Adjust to what your experiments require — check requirements.txt for the full set.)

Data: what’s included & how to prepare it
-----------------------------------------
- data/data_dictionary.md describes available fields and expected formats.
- CSVs included (usgs_himalayan..., nsc_nepal...) are example catalogs already present in data/.
- Expected minimal fields for ingestion (used by scripts in src/):
  - unique event id, time (ISO 8601), latitude, longitude, depth, magnitude
- Use src/data_processing.py and src/feature_engineering.py as starting points to preprocess raw catalogs and produce training feature tables (check those files for function names and expected inputs/outputs).

Running the prototype experiments
-------------------------------
The repository includes a small prototype script:
- prototype/ml_prototype.py — a lightweight entrypoint to run feature extraction + model training/evaluation flows used for experiments.

Example (basic):
```
python prototype/ml_prototype.py
```
Notes:
- The prototype script is intentionally minimal and designed for experimentation. Inspect the script to see supported flags/parameters and to adapt input/output file paths, model hyperparameters, and label definitions.
- Typical workflow: preprocess data with src/data_processing.py -> generate features via src/feature_engineering.py -> train/evaluate models with the prototype script or reuse saved models from models/.

Running the app
---------------
An application entrypoint is present at app/app.py. It may be a lightweight Flask/Streamlit/CLI interface; inspect app/app.py for its requirements and runtime options.

Example:
```
python app/app.py
```
If the app uses environment variables, dependencies, or a web server, consult the top of the file for runtime instructions and adjust your environment accordingly.

Models & artifacts
------------------
Pre-trained artifacts (joblib) are available in the models/ directory. They include simple preprocessors and trained models (RF, LightGBM, logistic regression, SVM). These can be loaded with joblib.load for inference in Python.

Example:
```python
import joblib
model = joblib.load("models/random_forest_model.joblib")
preproc = joblib.load("models/rf_preprocessor.joblib")
X = preproc.transform(X_raw)
y_pred = model.predict_proba(X)[:, 1]  # if probabilistic
```


Acknowledgements & references
-----------------------------
- USGS Earthquake Catalog: https://earthquake.usgs.gov/fdsnws/event/1/
- Data and shapefiles used in this repo (see data/ and data/data_dictionary.md)
- Open-source libraries: pandas, scikit-learn, lightgbm, joblib, geopandas

