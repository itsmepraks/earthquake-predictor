# Nepal Earthquake Building Damage Prediction: A Machine Learning Approach
**Final Project Report**

## Abstract
This report details the development of a machine learning system to predict building damage levels (Low, Medium, High) resulting from the 2015 Gorkha earthquake in Nepal. Utilizing a dataset combining building characteristics from the DrivenData competition with earthquake parameters, various classification models were implemented and evaluated, including Logistic Regression, LightGBM, Random Forest, and Support Vector Machines (SVM). Preprocessing involved scaling numerical features and applying both One-Hot Encoding and Ordinal Encoding for categorical features. Hyperparameter tuning via RandomizedSearchCV identified an optimized LightGBM model as the top performer, achieving approximately 72.5% accuracy and 0.880 ROC AUC. Feature analysis consistently highlighted geographic location identifiers (`geo_level_id`) as the most dominant predictors. An interactive Streamlit application was developed to allow users to explore predictions based on adjustable building features, compare model performance, and view feature importances. Key challenges included difficulties in integrating terrain data and creating geographic map visualizations due to unresolved discrepancies between the dataset's geographic identifiers and available administrative boundary shapefiles.

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Data Sources and Preparation](#2-data-sources-and-preparation)
    1.  [Data Sources](#21-data-sources)
    2.  [Data Merging and Preprocessing](#22-data-merging-and-preprocessing)
3.  [Methodology](#3-methodology)
    1.  [Modeling Pipeline](#31-modeling-pipeline)
4.  [Results](#4-results)
    1.  [Model Performance Comparison](#41-model-performance-comparison)
    2.  [Model Selection](#42-model-selection)
    3.  [Feature Importance Insights](#43-feature-importance-insights)
5.  [Streamlit Application](#5-streamlit-application)
    1.  [Purpose and Features](#51-purpose-and-features)
    2.  [Technology Stack](#52-technology-stack)
    3.  [Limitations and Challenges](#53-limitations-and-challenges)
6.  [Discussion](#6-discussion)
7.  [Conclusion](#7-conclusion)
8.  [Future Work](#8-future-work)
9.  [References](#9-references)

## 1. Introduction
Nepal's geographic position along the convergent boundary between the Indian and Eurasian tectonic plates renders it highly susceptible to seismic activity. The devastating 2015 Gorkha earthquake (Mw 7.8) underscored the vulnerability of the nation's building stock, causing widespread destruction and loss of life. Accurate prediction of building damage based on structural characteristics and earthquake parameters is crucial for risk assessment, disaster preparedness, and resource allocation for future events.

This project aimed to develop and evaluate machine learning models for classifying earthquake-induced building damage in Nepal, using data related to the 2015 Gorkha event. The primary objectives were:
*   To build a robust pipeline for data ingestion, preprocessing, and model training.
*   To evaluate various classification algorithms for predicting damage grades (1: Low, 2: Medium, 3: High/Collapse).
*   To identify key features influencing building damage.
*   To develop an interactive prototype application for exploring model predictions.

## 2. Data Sources and Preparation

### 2.1. Data Sources
The primary datasets used in this project include:
1.  **Building Data (DrivenData):** Sourced from the "Richter's Predictor: Modeling Earthquake Damage" competition hosted by DrivenData.org and downloaded from the competition portal. This includes:
    *   `train_values.csv`: Structural characteristics, usage, ownership, and geographic identifiers (`geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id`) for ~260,000 buildings.
    *   `train_labels.csv`: The corresponding damage grade (1, 2, or 3) for each building in the training set.
    *   (See `data/data_dictionary.md` for detailed feature descriptions).
2.  **Earthquake Data:**
    *   Basic parameters of the main 2015 Gorkha earthquake (Mw 7.8, depth ~15km, epicenter ~28.23°N, 84.73°E) derived from public USGS/NSC catalogs were used.
3.  **Geographic Data:**
    *   Administrative boundary shapefiles for Nepal (`data/npl_adm_nd_20240314_ab_shp/`) were obtained from publicly available sources (likely GADM or similar administrative datasets) for potential map visualization.
    *   SRTM (Shuttle Radar Topography Mission) 3 arc-second Global elevation data was acquired via NASA Earthdata for terrain analysis but could not be reliably integrated (see Section 5.3).

### 2.2. Data Merging and Preprocessing
Data preparation involved several steps to create a unified dataset suitable for modeling:
1.  **Merging:** The building values (`train_values.csv`) and labels (`train_labels.csv`) were merged based on the unique `building_id`.
2.  **Schema Standardization:** Column names were checked for consistency. Data types were appropriate for the initial analysis.
3.  **Feature Engineering (Simplified):** Due to the lack of precise coordinates for individual buildings and difficulties mapping `geo_level_id`s to known administrative boundaries or coordinates, a simplified approach was taken for incorporating earthquake features. The magnitude (7.8), depth (~15km), and epicenter coordinates (~28.23°N, ~84.73°E) of the *main Gorkha shock* were added as uniform features (`main_eq_magnitude`, `main_eq_depth`, `main_eq_epicenter_lat`, `main_eq_epicenter_lon`) to *all* building records.
4.  **Handling Missing Values & Outliers:** The core building dataset had minimal missing values requiring imputation. Outlier analysis did not lead to significant data removal for the features used in the final models.
5.  **Final Dataset:** The resulting dataset, saved as `data/processed/buildings_features_earthquakes.csv`, contains building structural, usage, ownership, location (geo IDs), and simplified earthquake features alongside the target `damage_grade`. This formed the basis for all subsequent modeling.

## 3. Methodology

### 3.1. Modeling Pipeline
A standard machine learning pipeline was implemented using Python libraries Scikit-learn and LightGBM.
1.  **Data Splitting:** The final dataset was split into training (80%) and testing (20%) sets using `train_test_split`, stratifying by the `damage_grade` target variable to ensure representative class distributions in both sets.
2.  **Preprocessing Pipelines:** Scikit-learn `Pipeline` objects were created to chain scaling and encoding steps, preventing data leakage from the test set. Different strategies were employed based on the model type:
    *   **Numerical Features:** Consistently scaled using `StandardScaler`.
    *   **Categorical Features:**
        *   *Strategy 1 (for Linear Models - LogReg, SVM):* `OneHotEncoder` (configured with `handle_unknown='ignore'` and `drop='if_binary'`) was applied within a `ColumnTransformer`. This resulted in a high-dimensional sparse matrix (~12,900 features). The full pipeline (scaling + OHE) was saved as `models/baseline_preprocessor.joblib` (for LogReg) and `models/svm_preprocessor.joblib`.
        *   *Strategy 2 (for Tree-based Models - LGBM, RF):* `OrdinalEncoder` (configured with `handle_unknown='use_encoded_value'`, `unknown_value=-1`) was used within a `ColumnTransformer`. The full pipeline (scaling + Ordinal Encoding) was saved as `models/lightgbm_preprocessor.joblib` and `models/rf_preprocessor.joblib`.
3.  **Model Training:** Four distinct classification algorithms were trained on the preprocessed training data:
    *   Logistic Regression (`LogisticRegression` with `solver='saga'`, `class_weight='balanced'`, `max_iter=1000`)
    *   LightGBM (`LGBMClassifier` with `class_weight='balanced'`, `objective='multiclass'`)
    *   Random Forest (`RandomForestClassifier` with `class_weight='balanced'`, `n_jobs=-1`)
    *   Linear Support Vector Machine (`LinearSVC` with `class_weight='balanced'`, `dual=False`, `max_iter=2000`)
4.  **Hyperparameter Tuning:** `RandomizedSearchCV` (20 iterations, 3-fold cross-validation using `StratifiedKFold`, optimizing for accuracy, `n_jobs=-1`) was used to tune the hyperparameters of the LightGBM model.
5.  **Evaluation:** Models were evaluated on the preprocessed test set using:
    *   Accuracy Score
    *   ROC AUC Score (`roc_auc_score` with `average='macro'`, `multi_class='ovr'`)
    *   Weighted F1-Score (`f1_score` with `average='weighted'`)
    *   Classification Report (`classification_report` providing per-class precision, recall, F1-score)
    *   Confusion Matrix (`confusion_matrix`)
6.  **Feature Importance:** Analyzed using:
    *   Absolute coefficient values (`model.coef_`) for Logistic Regression and LinearSVC.
    *   Internal `feature_importances_` attribute for LightGBM and Random Forest.
    *   For OHE features, importance scores/coefficients corresponding to all categories derived from a single original feature were summed to get an aggregated importance score for that original feature.

## 4. Results

### 4.1. Model Performance Comparison
The performance of the trained models on the held-out test set is summarized below:

| Model                 | Preprocessing   | Accuracy | ROC AUC | Weighted F1 | Grade 1 Recall | Training Time (Approx) | Notes                                        |
| :-------------------- | :-------------- | :------- | :------ | :---------- | :------------- | :--------------------- | :------------------------------------------- |
| Logistic Regression   | OHE + Scale     | 0.697    | 0.872   | 0.69        | ~0.70          | ~2 min                 | 'saga' solver needed for convergence         |
| LightGBM (Untuned)    | Ordinal + Scale | 0.711    | 0.880   | 0.71        | ~0.55          | ~6 sec                 | Faster training                              |
| Random Forest         | Ordinal + Scale | 0.717    | 0.845   | 0.71        | 0.47           | ~7 sec                 | Lower AUC than LGBM                          |
| LinearSVC             | OHE + Scale     | 0.721    | N/A     | 0.72        | 0.76           | ~1 min 48 sec          | Highest untuned Acc/F1, but slow training    |
| **LightGBM (Tuned)**  | Ordinal + Scale | **0.725**| 0.880   | **0.73**    | 0.77           | **Fast Inference**     | Best overall balance, selected final model |

*(Note: Grade 1 Recall values are approximate based on classification reports)*

### 4.2. Model Selection
The **Tuned LightGBM model** demonstrated the best overall performance, achieving the highest accuracy and weighted F1-score while maintaining the excellent AUC of the untuned version. It showed a better balance in classifying the different damage grades compared to other models, particularly improving recall for Grade 1 (Low Damage) significantly over the untuned tree models, and achieving better precision for Grade 1 than LinearSVC. Its fast training and inference times were additional advantages. This model (`models/lightgbm_tuned_model.joblib`) was selected as the primary model for the Streamlit application.

### 4.3. Feature Importance Insights
Across all models, the most influential features were consistently the **geographic location identifiers** (`geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id`). Specific `geo_level_3_id` values often dominated the top importances, suggesting strong spatial autocorrelation in damage patterns – buildings in certain small geographic areas experienced similar damage levels regardless of other characteristics. For models using One-Hot Encoded features (LogReg, SVM), the importance scores for all categories derived from an original feature were summed to provide an aggregated view, which still highlighted the dominance of geographic IDs.

While location dominated, tree-based models (LGBM, RF), especially the tuned LGBM, assigned relatively higher importance to building characteristics compared to the linear models. Features like `age`, `foundation_type`, `count_floors_pre_eq`, `area_percentage`, and various superstructure material flags (`has_superstructure_*`) appeared among the top ~20-30 features, indicating they do contribute to the prediction, albeit less than location.

## 5. Streamlit Application

An interactive web application was developed using Streamlit to facilitate exploration of the model predictions.

### 5.1. Purpose and Features
The application allows users to:
*   **Select a Model:** Choose from the trained models (Logistic Regression, LightGBM, Tuned LightGBM, Random Forest, SVM) via a dropdown menu.
*   **Input Building Features:** Adjust characteristics of a hypothetical building using interactive widgets in the sidebar. These include:
    *   Numerical sliders (e.g., Age, Number of Floors, Area %, Height %).
    *   Categorical dropdowns with descriptive labels (e.g., Foundation Type, Roof Type, Land Surface Condition, Geo Level 1 ID).
    *   Grouped checkboxes for binary features (Superstructure Materials, Secondary Uses).
    *   Input widgets are dynamically generated based on the features required by the selected model. Tooltips provide descriptions for each feature.
*   **View Prediction:** See the predicted damage risk (Low, Medium, High), color-coded for clarity, for the specified building features using the selected model.
*   **Analyze Model Behavior:** View charts generated based on the selected model's performance on the entire dataset:
    *   *Risk Distribution Chart:* Bar chart showing the overall count of buildings predicted in each risk category.
    *   *Feature Importance Chart:* Bar chart displaying the top 20 most important features (aggregated for OHE features where applicable).
*   **Compare Models:** View a table summarizing the key performance metrics (Accuracy, AUC, F1) for all trained models, highlighting the currently selected one.

### 5.2. Technology Stack
*   **Framework:** Streamlit (`streamlit`)
*   **Data Manipulation:** Pandas (`pandas`), NumPy (`numpy`)
*   **Machine Learning:** Scikit-learn (`scikit-learn`), LightGBM (`lightgbm`)
*   **Plotting:** Altair (`altair`)
*   **Model/Preprocessor Persistence:** Joblib (`joblib`)

### 5.3. Limitations and Challenges
*   **Geographic Visualization:** A planned interactive map showing predicted risk aggregated by geographic region could not be implemented. This was due to persistent difficulties in finding a reliable mapping between the `geo_level_id`s present in the building dataset and the PCODEs (Place Codes) used in available administrative boundary shapefiles for Nepal. Without this link, spatially joining predictions to geographic areas was not feasible.
*   **Terrain Feature Integration:** SRTM elevation data was acquired and processed. However, integrating these features (e.g., calculating average elevation or slope per `geo_level_id`) was blocked by the same `geo_level_id`-to-PCODE mapping issue. Therefore, terrain features were not included in the final models.
*   **Input Opacity:** Some categorical dropdowns (e.g., foundation type, roof type) use descriptive labels mapped from single-letter codes found in the original data. While an improvement, the precise meaning of the original codes remains partially ambiguous without definitive metadata from the source competition.

## 6. Discussion
The project successfully developed a pipeline for predicting earthquake-induced building damage in Nepal. The models achieved moderate predictive performance, with the tuned LightGBM reaching ~72.5% accuracy.

The overwhelming importance of `geo_level_id`s suggests that unobserved local factors strongly correlate with damage. These could include variations in ground shaking intensity (not captured by the single main shock parameters used), soil conditions, micro-zonation effects, or local construction practices/material quality variations not fully represented by the available features. The inability to integrate terrain data or create detailed geographic risk maps due to the `geo_level_id` mapping issue significantly limited the potential for spatial analysis and the inclusion of potentially valuable terrain-related features.

The performance plateau around 70-73% accuracy, despite exploring various complex models and tuning, indicates that the predictive power might be inherently limited by the available features. More granular data on ground motion (e.g., intensity measures at building locations or zones), detailed soil properties, and potentially more explicit engineering features might be necessary to substantially improve prediction accuracy.

The Streamlit application provides a valuable tool for interacting with the models and understanding their behavior based on the available data, despite the visualization limitations. The dynamic input generation and model comparison features enhance its utility.

## 7. Conclusion
This project demonstrated the application of machine learning techniques to predict building damage from the 2015 Gorkha earthquake. A tuned LightGBM model provided the best performance among those evaluated, achieving 72.5% accuracy. Geographic location proved to be the most critical predictor, likely acting as a proxy for localized ground shaking intensity and other unmeasured factors. While the models offer some predictive capability, significant improvements would likely require more detailed geospatial data, ground motion information, and potentially resolving the ambiguities surrounding the `geo_level_id` system. The developed Streamlit application serves as an effective interface for exploring the current models' predictions and limitations.

## 8. Future Work
Potential directions for future work include:
*   **Resolving Geo ID Mapping:** Focused effort to find or create a reliable crosswalk between `geo_level_id`s and standard administrative PCODEs or geographic coordinates. This would unlock map visualizations and terrain feature integration.
*   **Alternative Location Features:** Exploring distance to fault lines, distance to the epicenter, or incorporating predicted ground shaking intensity maps (e.g., ShakeMaps) if available and linkable.
*   **Advanced Feature Engineering:** Investigating interaction features or deriving new features from existing ones (e.g., building density ratios).
*   **Model Explainability:** Implementing SHAP (SHapley Additive exPlanations) within the Streamlit app to provide explanations for individual predictions.
*   **Deployment:** Packaging the Streamlit application for deployment (e.g., using Docker, Streamlit Cloud).

## 9. References
*   DrivenData Competition: "Richter's Predictor: Modeling Earthquake Damage" (https://www.drivendata.org/competitions/57/nepal-earthquake/)
*   USGS Earthquake Catalog API (https://earthquake.usgs.gov/fdsnws/event/1/)
*   Scikit-learn Documentation (https://scikit-learn.org/stable/documentation.html)
*   LightGBM Documentation (https://lightgbm.readthedocs.io/en/latest/)
*   Streamlit Documentation (https://docs.streamlit.io/)
*   Relevant research papers on the 2015 Gorkha earthquake and seismic vulnerability in Nepal.