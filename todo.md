# To Do

## Phase 1: Project Setup & Environment
- [x] Create directories: `data/`, `src/`, `app/`
- [x] Set up Python virtual environment (venv)
- [x] Create `requirements.txt` with core dependencies (streamlit, scikit-learn, pandas, numpy, geopandas, folium)

*Phase 1 Summary: Successfully set up the project structure, virtual environment, and core dependencies.*

## Phase 2: Data Acquisition & Preprocessing
- [x] Download and load USGS Himalayan earthquake data
- [x] Download and load 2015 Gorkha aftershock dataset (using cleaned NSC data from amitness/earthquakes repo)
- [x] Download and load NASA terrain elevation data (SRTM GL3 tiles)
- [x] Download and load Kaggle supplementary data (DrivenData Building Damage)
- [x] Initialize the streamlit app to view the data and interact with its
- [x] Standardize schemas across datasets (columns, units, datetime formats)
- [x] Handle missing values and outliers, and merge datasets

*Phase 2 Summary: Downloaded USGS, NSC, SRTM, and DrivenData (building) datasets. Standardized schemas for earthquake data (USGS/NSC), combined them, and handled missing depth values.*

## Phase 2.5: Feature Engineering
- [x] Define and apply magnitude thresholds (e.g., Moderate, Strong, Major) to earthquake data.
- [x] Investigate geo_level IDs: Obtain corresponding geographic boundaries (e.g., shapefiles). (Outcome: Found boundaries, but IDs (P-Codes) don't directly match building data geo_level_ids. No coordinates in building data for spatial join. Will proceed using geo_level_ids as primary location features for now.)
- [x] Link earthquake events to building locations/zones. (Approach: Added features from the main 2015 Gorkha event - magnitude, depth, epicenter - uniformly to all buildings due to lack of specific building coordinates or geo_level_id mapping.)
- [-] Extract terrain attributes (elevation, slope, aspect) for building locations/zones using SRTM data. (Skipped/Deferred: Cannot extract terrain features per building/zone due to lack of specific coordinates or geo_level_id mapping/boundaries. SRTM data also not yet downloaded.)
- [x] Combine earthquake features, building features, and terrain features into a unified dataset. (Completed: `data/processed/buildings_features_earthquakes.csv` contains building features, target, and simplified earthquake features. Terrain features omitted.)

*Phase 2.5 Summary: Merged earthquake data, added magnitude categories. Investigated geo IDs, finding no direct link to shapefiles or coordinates; proceeded using geo_level_ids as features. Added main Gorkha event features uniformly to buildings. Skipped terrain feature extraction due to data limitations. Final combined dataset created.*

## Phase 3: Modeling Pipeline
- [ ] Define target variables and thresholds
- [x] Implement data splitting & cross-validation framework
- [x] Train logistic regression baseline model
- [ ] Map probabilities to low/medium/high risk categories
- [x] Evaluate baseline model (ROC AUC, accuracy, confusion matrix)

*Phase 3 Summary: Established the modeling pipeline. Loaded and preprocessed data (scaling numerical, one-hot encoding categorical with sparse output). Split data into train/test sets. Trained a baseline Logistic Regression model and performed initial evaluation (accuracy, AUC, confusion matrix, classification report).*

## Phase 4: Model Evaluation & Tuning
- [-] Analyze performance metrics and feature importances -> (Completed Initial Analysis)
- [x] Address convergence issues
- [x] Refine probability thresholds and risk class bins
- [x] Document insights in `whitepaper.md` and update `proposal.md` if needed
- [-] Analyze feature importance -> (Completed Initial Analysis)
- [x] Implement and evaluate LightGBM model

*Phase 4 Summary: Analyzed the baseline Logistic Regression model (accuracy ~69.7%, AUC ~0.872), identifying location as the key driver. Addressed convergence by switching to 'saga' solver. Implemented and evaluated a LightGBM model (accuracy ~71.1%, AUC ~0.880) using Ordinal Encoding, which showed modest performance improvement and better utilization of building features compared to the baseline. Documented findings in `whitepaper.md`.*

## Phase 5: Streamlit Application Development
- [x] Sketch UI layout: sidebar controls, interactive map, and charts (Initial structure created)
- [x] Integrate preprocessing and inference modules into Streamlit app (Data, Models, Preprocessors loaded; Basic prediction logic added)
- [x] Implement/Review caching with `@st.cache_data`/`@st.cache_resource`
- [x] Refine sidebar input controls for key building features
- [-] Implement interactive map visualization (e.g., Folium) -> Deferred/Blocked: Cannot reliably join geo_level_id with shapefile PCODEs.
- [x] Implement charts for risk distribution and feature importance (e.g., Altair)

*Phase 5 Summary: Initial Streamlit app structure created, including data/model loading, basic prediction logic, sidebar inputs, caching, and results charts. Interactive map visualization is deferred due to challenges mapping geo_level_ids to geographical boundaries.*

## Phase 6: Documentation & Version Control
- [x] Maintain and update `todo.md` with completed and pending tasks
- [x] Create and maintain `data/data_dictionary.md`
- [x] Update `whitepaper.md` with methodology, initial results, and app description -> (Reviewed, up-to-date)

*Phase 6 Summary: Updated todo.md to reflect current status. Created data_dictionary.md with column descriptions for the primary dataset. Reviewed whitepaper.md and confirmed it includes recent modeling and app developments.*

## Phase 7: Advanced Modeling & Hyperparameter Tuning
- [x] Implement Random Forest model (train, eval, feature importance)
- [x] Implement SVM model (train, eval, feature importance if applicable)
- [x] Perform hyperparameter tuning (Grid/RandomizedSearchCV) on top models -> (LGBM Tuned)
- [ ] (Optional/Deferred) Tune Random Forest model
- [x] Compare all models (LogReg, LGBM, RF, SVM, Tuned LGBM) and select final model(s) -> (Tuned LGBM Selected)
- [x] Document model comparison and selection rationale in `whitepaper.md`

*Phase 7 Summary: Implemented Random Forest and LinearSVC models. Tuned LightGBM using RandomizedSearchCV, achieving the best performance (Accuracy: 0.7250, F1: 0.73, AUC: 0.8795). Selected Tuned LightGBM as the final model.*

## Phase 8: Feature & Visualization Enhancements
- [ ] Revisit Feature Engineering (Terrain Data)
  - [ ] Attempt to acquire and process SRTM elevation data
  - [ ] Investigate methods to link terrain features to building data (geo_level_id aggregation?)
  - [ ] Integrate features and retrain/evaluate models if successful
- [ ] Revisit Map Visualization
  - [ ] Investigate geo_level_id to PCODE mapping
  - [ ] Implement choropleth map (ADM level if possible, fallback to geo_level_id)
  - [ ] Integrate map into Streamlit app
- [ ] Enhance Streamlit Application
  - [ ] Add new models (RF, SVM, Tuned) to selector
  - [ ] Update feature importance chart logic
  - [ ] Add model comparison display
  - [ ] Refine UI/UX

## Phase 9: Final Documentation & Wrap-up
- [ ] Update Documentation
  - [ ] Update `whitepaper.md` comprehensively (advanced models, tuning, features, map, final summary)
  - [ ] Update `data/data_dictionary.md` (if new features added)
  - [ ] Mark Phase 7-9 tasks complete in `todo.md`
  - [ ] Create/Update top-level `README.md` (project overview, setup, run instructions)
- [ ] Perform final code review (`src/`, `app/`) for clarity, style, comments
- [ ] Ensure `requirements.txt` is accurate
- [ ] Ensure final code/data/docs are committed to version control