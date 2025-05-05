# Earthquake Risk Forecasting for Nepal  
**A Preliminary White Paper**

## 1. Introduction  
Nepal's location along the Himalayan tectonic belt makes it one of the world's most earthquake‑prone regions. Over 80% of buildings in Kathmandu are rated structurally vulnerable. The 2015 Gorkha earthquake demonstrated the catastrophic human and economic toll when prediction and preparedness lag behind geophysical reality.

## 2. Objectives  
• Develop a machine‑learning framework to improve short‑term earthquake risk classification in Nepal.  
• Target "High Risk" vs. "Low Risk" zones with better precision than current 40–60% accuracy.  
• Provide a lightweight prototype suitable for low‑resource early‑warning deployments.

## 3. Data Sources  
1. **USGS Seismic Database** – Century‑scale records (magnitude, depth, lat/lon).  
2. **2015 Gorkha Dataset** – Detailed mainshock and aftershock catalog.  
3. **Supplementary** – NASA elevation DEMs, Kaggle soil/terrain layers.

## 4. Methodology  
1. **Data Ingestion & Preprocessing**  
   - Fetch via USGS API  
   - Filter by region (Nepal bounding box)  
   - Feature engineering: magnitude, depth, slope, elevation, distance to known faults  
2. **Baseline Model**  
   - Logistic Regression for binary "High/Low" risk  
   - Train/test split; GridSearchCV for hyperparameter tuning  
3. **Extended Models (next steps)**  
   - Random Forest for feature importance  
   - SVM for non‑linear decision boundaries

## 5. Prototype Implementation  
See `prototype/ml_prototype.py` for a runnable demo:
```python
# snippet here shows data fetch → basic train/evaluate
```

## 6. Preliminary Results & Discussion
*(Initial baseline model results added below.)*

We established a baseline model using Logistic Regression to predict building damage grade (1: Low, 2: Medium, 3: Severe/Collapse) based on the combined dataset (`data/processed/buildings_features_earthquakes.csv`). This dataset includes building characteristics (structural details, usage, age, etc.) and geographic identifiers (`geo_level_1_id` to `geo_level_3_id`). Uniform features from the main 2015 Gorkha event (magnitude, depth, epicenter) were also included for all buildings due to limitations in precise location mapping.

**Preprocessing:**
- Numerical features (e.g., `age`, `area_percentage`) were scaled using `StandardScaler`.
- Categorical features (including building characteristics and geographic IDs) were transformed using `OneHotEncoder`, resulting in a high-dimensional sparse feature set (~12,900 features).

**Baseline Model Performance (Logistic Regression with 'saga' solver, balanced class weights):
**
- **Accuracy:** ~69.7%
- **ROC AUC (Macro Avg, OvR):** ~0.872
- **Key Observations:** The model shows reasonable discriminative ability (good AUC) but achieves modest overall accuracy. Analysis of the confusion matrix and classification report indicates:
    - Good recall for Class 1 (Low Damage) but poor precision (many Class 2 buildings misclassified as Class 1).
    - Moderate recall for Class 2 (Medium Damage), with significant misclassifications into both Class 1 and Class 3.
    - Good recall for Class 3 (Severe Damage), with some misclassification into Class 2.

**Feature Importance (Based on Average Absolute Coefficients):
**
- The most influential features were overwhelmingly specific geographic identifiers, particularly `geo_level_3_id` values, followed by `geo_level_2_id` and `geo_level_1_id`.
- The building feature `plan_configuration` also appeared among the top predictors.
- This suggests that *location* is the dominant factor driving damage prediction in this baseline model, potentially overshadowing specific building characteristics or the simplified earthquake features used.

**Note:** The initial model training faced convergence issues with the default 'lbfgs' solver, likely due to the high dimensionality. Switching to the 'saga' solver resolved this without impacting performance metrics.

These baseline results provide a benchmark. Future work will focus on exploring more complex models (e.g., tree-based ensembles like LightGBM or RandomForest) that might better capture interactions and non-linearities, potentially improving accuracy and leveraging structural features more effectively. Further investigation into different encoding strategies for geographic features might also be beneficial.

**LightGBM Model Results:**

To explore improvements over the baseline, we implemented a LightGBM model. Key differences in approach included:
- Using `OrdinalEncoder` for categorical features, relying on LightGBM's native handling capabilities.
- Training an `LGBMClassifier` with default parameters (except `class_weight='balanced'` and objective/metric settings for multiclass).

**Performance Comparison:**
- **Accuracy:** ~71.1% (vs. ~69.7% for LogReg) - **Improvement**
- **ROC AUC (Macro Avg, OvR):** ~0.880 (vs. ~0.872 for LogReg) - **Slight Improvement**
- **Training Time:** Significantly faster than Logistic Regression (~6 seconds vs. >2 minutes).
- **Confusion Matrix/Classification Report:** Showed similar patterns but slightly better F1-scores, particularly for Grade 2 (Medium Damage).

**Feature Importance (LightGBM):
**
- Geographic identifiers (`geo_level_3_id`, `geo_level_2_id`, `geo_level_1_id`) remained the most dominant features.
- However, LightGBM assigned higher importance to several building characteristics compared to Logistic Regression (e.g., `age`, `foundation_type`, `area_percentage`, various `has_superstructure_*` features appeared in the top 20).

**Conclusion:** The LightGBM model provided a modest improvement in accuracy and AUC compared to the baseline Logistic Regression, while being significantly faster to train. It also appears slightly better at incorporating building feature information alongside the dominant location features. Further gains might be possible through hyperparameter tuning of the LightGBM model.

## 7. Streamlit Application Prototype

To provide an interactive interface for exploring the trained models and their predictions, a prototype web application was developed using Streamlit.

**Purpose:**
*   Allow users to select between the trained classification models (Logistic Regression, LightGBM).
*   Enable users to input specific building characteristics.
*   Display the predicted damage risk category (Low, Medium, High) for the specified building.
*   Visualize overall model performance and behavior through charts.

**Features:**
*   **Model Selection:** Dropdown menu to choose between available models.
*   **Building Feature Input:** Sidebar controls (sliders, dropdowns) for key features like Geographic Level 1 ID, Age, Area Percentage, Height Percentage, and Foundation Type. Other features are currently set to default values (median/mode).
*   **Prediction Display:** Shows the predicted risk category based on the selected model and input features upon clicking a button.
*   **Risk Distribution Chart:** An Altair bar chart showing the predicted distribution of Low, Medium, and High risk categories across the entire dataset for the selected model.
*   **Feature Importance Chart:** An Altair bar chart displaying the top N most important features as determined by the selected model (using coefficient magnitudes for Logistic Regression and internal importance scores for LightGBM).
*   **Data Preview:** Displays the head of the loaded dataset (`buildings_features_earthquakes.csv`).

**Technology Stack:**
*   Frontend/Framework: Streamlit
*   Data Handling: Pandas
*   Machine Learning: Scikit-learn, LightGBM
*   Plotting: Altair
*   Model Persistence: Joblib

**Limitations:**
*   **Map Visualization:** The planned choropleth map visualization showing risk aggregated by geographic region was deferred due to difficulties in reliably joining the building data's `geo_level_id` with the available shapefile PCODEs.
*   **Categorical Inputs:** While `foundation_type` uses descriptive labels, other categorical inputs (`roof_type`, `ground_floor_type`, etc.) display opaque single-letter codes as their original meanings are potentially obfuscated in the source data.
*   **Input Completeness:** Only a subset of building features are currently available as inputs in the sidebar.

This prototype serves as a proof-of-concept for interactively exploring the model predictions.

## 8. Roadmap & Next Steps  
- Integrate elevation and soil features  
- Evaluate Random Forest & SVM benchmarks  
- Build a Streamlit dashboard for risk‑map visualization  
- Optimize for edge/cloud deployment  

## 9. References  
- USGS Earthquake Catalog API  
- 2015 Gorkha seismic analysis papers  
- Python ML ecosystem docs  