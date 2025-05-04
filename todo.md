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
- [ ] Implement data splitting & cross-validation framework
- [ ] Train logistic regression baseline model
- [ ] Map probabilities to low/medium/high risk categories
- [ ] Evaluate baseline model (ROC AUC, accuracy, confusion matrix)

## Phase 4: Model Evaluation & Tuning
- [ ] Analyze performance metrics and feature importances
- [ ] Refine probability thresholds and risk class bins
- [ ] Document insights in `whitepaper.md` and update `proposal.md` if needed

## Phase 5: Streamlit Application Development
- [ ] Sketch UI layout: sidebar controls, interactive map, and charts
- [ ] Integrate preprocessing and inference modules into Streamlit app
- [ ] Implement caching with `@st.cache_data`

## Phase 6: Documentation & Version Control
- [ ] Maintain and update `todo.md` with completed and pending tasks
- [ ] Update `whitepaper.md` with methodology and initial results