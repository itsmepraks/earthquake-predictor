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
- [ ] Download and load Kaggle supplementary data
- [ ] Initialize the streamlit app to view the data and interact with its
- [ ] Standardize schemas across datasets (columns, units, datetime formats)
- [ ] Handle missing values and outliers, and merge datasets
- [ ] Engineer features: magnitude thresholds, geographical zoning, terrain attributes

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