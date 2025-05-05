import os  # Added for path joining
import traceback  # Keep for predict_risk error handling

import altair as alt  # Added
import joblib  # Added
import numpy as np
import pandas as pd
import streamlit as st

# import folium
# from streamlit_folium import st_folium
# import pydeck as pdk # Potentially for 3D maps if needed

# --- Page Config --- # Moved to Top
st.set_page_config(layout="wide", page_title="Nepal Earthquake Risk Predictor")

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # More robust way to get project root
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'buildings_features_earthquakes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Model/Preprocessor Paths --- # Consolidated
MODEL_PATHS = {
    "Logistic Regression": {
        "model": os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'),
        "preprocessor": os.path.join(MODELS_DIR, 'baseline_preprocessor.joblib')
    },
    "LightGBM": {
        "model": os.path.join(MODELS_DIR, 'lightgbm_model.joblib'),
        "preprocessor": os.path.join(MODELS_DIR, 'lightgbm_preprocessor.joblib')
    },
    "LightGBM (Tuned)": {
        "model": os.path.join(MODELS_DIR, 'lightgbm_tuned_model.joblib'),
        "preprocessor": os.path.join(MODELS_DIR, 'lightgbm_preprocessor.joblib') # Assuming tuned uses same preprocessor
    },
    "Random Forest": {
        "model": os.path.join(MODELS_DIR, 'random_forest_model.joblib'),
        "preprocessor": os.path.join(MODELS_DIR, 'rf_preprocessor.joblib') # Assuming RF uses OrdinalEncoder
    },
    "SVM": {
        "model": os.path.join(MODELS_DIR, 'svm_model.joblib'),
        "preprocessor": os.path.join(MODELS_DIR, 'svm_preprocessor.joblib') # Assuming SVM uses OHE/Scaler
    },
}

# --- Risk Category Mapping --- # Updated
RISK_MAP = {
    "Logistic Regression": {1: "Low", 2: "Medium", 3: "High"},
    "LightGBM": {0: "Low", 1: "Medium", 2: "High"}, # Assumes y was y-1 for training
    "LightGBM (Tuned)": {0: "Low", 1: "Medium", 2: "High"}, # Assumes y was y-1 for training
    "Random Forest": {0: "Low", 1: "Medium", 2: "High"}, # Assuming 0,1,2 mapping
    "SVM": {0: "Low", 1: "Medium", 2: "High"} # Assuming 0,1,2 mapping
}

# --- Data Loading ---
@st.cache_data # Cache the data loading
def load_data(path):
    """Loads the processed building features and earthquake data."""
    try:
        df = pd.read_csv(path) # Simplified assuming path is now absolute or correctly relative
        st.info(f"Successfully loaded data from {os.path.basename(path)}")
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Model & Preprocessor Loading --- # Updated
@st.cache_resource # Cache resource loading
def load_all_artifacts(paths_dict):
    """Loads all models and preprocessors defined in the paths dictionary."""
    loaded_artifacts = {}
    for model_name, paths in paths_dict.items():
        model = None
        preprocessor = None
        # Use st.write for debugging logs during loading if needed
        # st.write(f"Loading artifacts for: {model_name}")
        try:
            model_path = paths.get("model")
            preprocessor_path = paths.get("preprocessor")
            if model_path and os.path.exists(model_path):
                 model = joblib.load(model_path)
                 # st.info(f"  Loaded model: {os.path.basename(model_path)}")
            else:
                 st.warning(f"Model file not found or path missing for {model_name}: {model_path}")

            if preprocessor_path and os.path.exists(preprocessor_path):
                 preprocessor = joblib.load(preprocessor_path)
                 # st.info(f"  Loaded preprocessor: {os.path.basename(preprocessor_path)}")
            else:
                 st.warning(f"Preprocessor file not found or path missing for {model_name}: {preprocessor_path}")

            if model is not None and preprocessor is not None:
                 loaded_artifacts[model_name] = {"model": model, "preprocessor": preprocessor}
            else:
                 # Log error but don't necessarily stop the app from loading other models
                 st.error(f"Failed to load complete artifacts for {model_name}. It will be unavailable.")

        except Exception as e:
            st.error(f"Error loading artifacts for {model_name}: {e}")
            # st.text(traceback.format_exc()) # Uncomment for detailed loading errors
    return loaded_artifacts

# --- Load Artifacts --- # Updated
# This is now called once and cached
MODELS = load_all_artifacts(MODEL_PATHS)

# --- Prediction Function ---
def predict_risk(input_df, model_name, models_dict):
    """Preprocesses input data and predicts risk using the selected model."""
    model_info = models_dict.get(model_name)
    if not model_info or model_info["model"] is None or model_info["preprocessor"] is None:
        st.error(f"Model or preprocessor for {model_name} not loaded or incomplete.")
        return None, None # Return None for both prediction and probs/shap

    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    # Get risk map safely
    risk_map = RISK_MAP.get(model_name, {0: "Unknown"}) # Default to handle potential missing keys

    try:
        # TODO: Ensure input_df columns match preprocessor.feature_names_in_ if available
        # This requires saving/loading the feature list used during training
        # st.write(f"Preprocessing with: {type(preprocessor)}") # Debugging
        input_processed = preprocessor.transform(input_df)
        # st.write(f"Shape after preprocessing: {input_processed.shape}") # Debugging

        predictions = model.predict(input_processed)
        # st.write(f"Raw Predictions ({model_name}):", predictions) # Debugging

        # Map numerical predictions to risk categories
        risk_categories = [risk_map.get(pred, "Unknown") for pred in predictions]

        # Attempt to get probabilities or SHAP values (optional, for explanation)
        # ... (Add SHAP logic here if desired, adapting per model) ...
        explanation_values = None # Placeholder for now

        return risk_categories, explanation_values # Return explanations if calculated

    except Exception as e:
        st.error(f"Error during prediction for {model_name}: {e}")
        # st.text(traceback.format_exc()) # Uncomment for detailed prediction error
        return None, None

# --- Visualization Functions ---

@st.cache_data # Cache plots based on data and model name
def plot_risk_distribution(df, model_name, _models_dict, feature_columns):
    """Generates predictions for the dataset and plots risk distribution."""
    model_info = _models_dict.get(model_name)
    if not model_info or model_info["model"] is None or model_info["preprocessor"] is None:
        st.warning(f"Cannot generate distribution: Model/Preprocessor for {model_name} not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    risk_map = RISK_MAP.get(model_name, {0: "Unknown"}) # Use get for safety

    # Prepare features carefully
    # Check if feature_columns exist in df
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns needed for prediction in the loaded data: {missing_cols}")
        return None
    X = df[feature_columns].copy() # Use copy to avoid SettingWithCopyWarning

    try:
        X_processed = preprocessor.transform(X)
        predictions = model.predict(X_processed)
        risk_categories = pd.Series([risk_map.get(pred, "Unknown") for pred in predictions])
        risk_counts = risk_categories.value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']

        chart = alt.Chart(risk_counts).mark_bar().encode(
            x=alt.X('Risk Category', sort=['Low', 'Medium', 'High', 'Unknown']), # Include Unknown
            y='Count',
            tooltip=['Risk Category', 'Count']
        ).properties(
            title=f'Predicted Risk Distribution ({model_name})'
        ).interactive()
        return chart
    except Exception as e:
        st.error(f"Error generating risk distribution plot for {model_name}: {e}")
        # st.text(traceback.format_exc())
        return None

# Function to safely get feature names (Helper - kept internal to this script)
def _get_feature_names(preprocessor, original_feature_list):
    """Gets feature names from preprocessor or falls back to original list."""
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            # Use get_feature_names_out for transformers like OneHotEncoder
            return preprocessor.get_feature_names_out()
        except Exception as e:
            st.warning(f"Could not get feature names from preprocessor using get_feature_names_out: {e}. Falling back.")
            # Fallback if get_feature_names_out fails or needs input_features
            if hasattr(preprocessor, 'feature_names_in_'):
                return preprocessor.feature_names_in_
            else:
                 st.warning("Preprocessor has no get_feature_names_out or feature_names_in_. Using original list.")
                 return original_feature_list # Fallback to original list
    elif hasattr(preprocessor, 'feature_names_in_'):
         return preprocessor.feature_names_in_ # For pipelines where this might be set
    else:
        # Fallback for simple transformers or if names aren't stored
        st.warning("Preprocessor type doesn't store feature names. Using original list.")
        return original_feature_list

@st.cache_data # Cache plots based on model name and feature columns list tuple
def plot_feature_importance(model_name, _models_dict, _feature_columns_tuple, top_n=20):
    """Plots the top N feature importances for the selected model."""
    _feature_columns = list(_feature_columns_tuple) # Convert tuple back to list
    model_info = _models_dict.get(model_name)
    if not model_info or model_info["model"] is None:
        st.warning(f"Cannot plot importance: Model {model_name} not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info.get("preprocessor") # Get preprocessor safely
    importance_df = None
    plot_title = f'Top {top_n} Feature Importances ({model_name})'

    try:
        # --- Strategy: Get names based on model type / preprocessor --- #
        feature_names = None
        if model_name in ["LightGBM", "LightGBM (Tuned)", "Random Forest"]:
            # These models use feature_importances_ and typically work with original features
            # if OrdinalEncoder was used. Use the passed _feature_columns.
            feature_names = list(_feature_columns) # Ensure it's a list
            if hasattr(model, 'feature_importances_'):
                 importances = model.feature_importances_
                 if len(feature_names) == len(importances):
                      importance_df = pd.DataFrame({
                          'feature': feature_names,
                          'importance': importances
                      }).sort_values(by='importance', ascending=False).head(top_n)
                 else:
                      st.error(f"Feature name/importance mismatch for {model_name}: {len(feature_names)} vs {len(importances)}")
                      return None
            else:
                 st.warning(f"{model_name} model missing feature_importances_ attribute.")
                 return None

        elif model_name in ["Logistic Regression", "SVM"]:
            # These models use coef_ and need names AFTER potential OHE.
            if hasattr(model, 'coef_'):
                 if preprocessor:
                      try:
                           feature_names = _get_feature_names(preprocessor, _feature_columns)
                           # Average absolute coefficients across classes for OvR/Multinomial
                           if model.coef_.ndim > 1:
                               avg_abs_coef = np.mean(np.abs(model.coef_), axis=0)
                           else: # Handle case of binary classification / single set of coefs
                               avg_abs_coef = np.abs(model.coef_.flatten()) # Flatten ensure 1D

                           if len(feature_names) == len(avg_abs_coef):
                                importance_df_raw = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance': avg_abs_coef
                                })
                                # Attempt to group OHE features (heuristic)
                                def get_original_feature(name):
                                     # Simpler logic: Remove prefix before first '__' if present
                                     parts = name.split('__')
                                     if len(parts) > 1:
                                          base_name = parts[-1]
                                     else:
                                          base_name = name
                                     # Attempt to remove trailing _category if OHE added it
                                     if '_' in base_name:
                                         original = base_name.rsplit('_', 1)[0]
                                         # Check if rsplit actually split anything sensible
                                         if original:
                                              return original
                                         else: # If only underscore was present?
                                              return base_name
                                     else:
                                         return base_name

                                importance_df_raw['original_feature'] = importance_df_raw['feature'].apply(get_original_feature)
                                grouped_importance = importance_df_raw.groupby('original_feature')['importance'].sum().reset_index()
                                grouped_importance.columns = ['feature', 'importance'] # Rename columns for consistency
                                importance_df = grouped_importance.sort_values(by='importance', ascending=False).head(top_n)
                                plot_title = f'Top {top_n} Aggregated Feature Importances ({model_name} - Abs Coef)'

                           else:
                                st.error(f"Feature name/coefficient mismatch for {model_name}: {len(feature_names)} vs {len(avg_abs_coef)}")
                                return None

                      except Exception as e:
                           st.error(f"Error processing {model_name} coefficients/feature names: {e}")
                           # st.text(traceback.format_exc())
                           return None
                 else:
                      st.warning(f"Preprocessor not available for {model_name}. Cannot map coefficients to feature names.")
                      return None
            else:
                 st.warning(f"{model_name} model missing coefficients (coef_) attribute.")
                 return None
        else:
            st.warning(f"Feature importance plotting not implemented for {model_name}")
            return None

        # --- Generate Plot --- #
        if importance_df is not None and not importance_df.empty:
            chart = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X('importance', title='Importance Score'),
                y=alt.Y('feature', title='Feature', sort='-x'), # Sort by importance
                tooltip=['feature', 'importance']
            ).properties(
                title=plot_title
            ).interactive() # Make chart interactive
            return chart
        else:
            # Removed redundant warning here, handled in specific cases above
            # st.warning(f"Could not generate feature importance data for {model_name}.")
            return None

    except Exception as e:
        st.error(f"Error generating feature importance plot for {model_name}: {e}")
        # st.text(traceback.format_exc())
        return None

# --- App Setup ---
st.title("🇳🇵 Nepal Earthquake Risk Predictor")
st.markdown("""
This application predicts the risk level (Low, Medium, High) for buildings in Nepal based on the 2015 Gorkha earthquake data.
Select options in the sidebar to provide building details and see the prediction. Models available include Logistic Regression, LightGBM, Random Forest, and SVM.
""")

# --- Load data --- # Moved lower, after artifact loading attempt
df = load_data(DATA_PATH)

# --- Sidebar Inputs ---
st.sidebar.header("Controls")

# Check if MODELS dictionary is populated
if not MODELS:
    st.sidebar.error("No models were loaded successfully. Prediction is unavailable.")
    st.stop() # Stop execution if no models are loaded

available_models = list(MODELS.keys())
default_model_index = available_models.index("LightGBM (Tuned)") if "LightGBM (Tuned)" in available_models else 0

selected_model_name = st.sidebar.selectbox(
    "Select Model", available_models, index=default_model_index
)
st.sidebar.markdown("---")

# Check if data loaded before proceeding
if df is None:
    st.error("Data loading failed. Cannot proceed with sidebar setup or predictions.")
    st.stop()

# Define Feature Columns (Globally for now - POTENTIAL ISSUE if preprocessing differs)
# TODO: Ideally, get this list based on the selected preprocessor/model
FEATURE_COLUMNS = []
try:
    # Exclude target and any known ID columns that shouldn't be features
    exclude_cols = ['damage_grade', 'building_id']
    FEATURE_COLUMNS = [col for col in df.columns if col not in exclude_cols]
    # Convert to tuple for caching in plot functions
    FEATURE_COLUMNS_TUPLE = tuple(FEATURE_COLUMNS)
    # st.write("Feature Columns Used:", FEATURE_COLUMNS) # Debugging
except Exception as e:
    st.error(f"Failed to define FEATURE_COLUMNS from loaded data: {e}")
    FEATURE_COLUMNS_TUPLE = tuple() # Set empty tuple

# Utility to get unique sorted values for selectbox options
@st.cache_data
def get_unique_sorted(series):
    """Safely gets unique sorted values from a pandas Series."""
    try:
        return sorted(series.unique())
    except Exception as e:
        st.warning(f"Could not get unique values for sidebar option: {e}")
        return [] # Return empty list on error

# --- Sidebar: Building Features ---
st.sidebar.subheader("Building Features")

# Create inputs dynamically based on FEATURE_COLUMNS
sidebar_inputs = {}

# Define engineered EQ features (should not be interactive inputs)
engineered_eq_cols = ['main_eq_magnitude', 'main_eq_depth', 'main_eq_epicenter_lat', 'main_eq_epicenter_lon']

st.sidebar.markdown("**Reference Earthquake Features**")
for eq_col in engineered_eq_cols:
    if eq_col in df.columns:
        # Display the constant value (assuming it's constant)
        val = df[eq_col].iloc[0]
        st.sidebar.text(f"{eq_col.replace('_', ' ').title()}: {val:.4f}") # Format float

st.sidebar.markdown("**Interactive Building Features**")

# Numerical Inputs - identified by dtype
num_cols = df[FEATURE_COLUMNS].select_dtypes(include=np.number).columns
for col in num_cols:
    # Skip constant EQ features and geo IDs / binary flags
    if col in engineered_eq_cols or col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'] or col.startswith('has_'):
        continue

    min_val = float(df[col].min())
    max_val = float(df[col].max())
    median_val = float(df[col].median())
    # Handle potential NaN/Inf in min/max/median
    if pd.isna(min_val) or pd.isna(max_val) or pd.isna(median_val):
         st.sidebar.warning(f"Could not determine range/median for {col}. Using default input.")
         sidebar_inputs[col] = st.sidebar.number_input(f"{col.replace('_', ' ').title()}", value=0.0)
         continue # Skip slider creation if range is invalid

    # Check if min and max are the same
    if min_val == max_val:
        st.sidebar.text(f"{col.replace('_', ' ').title()}: {min_val}") # Display constant value
        sidebar_inputs[col] = min_val # Store the constant value
    # Use number_input for flexibility or slider if range is reasonable
    elif max_val - min_val > 500 or max_val > 10000: # Heuristic for large ranges
         sidebar_inputs[col] = st.sidebar.number_input(f"{col.replace('_', ' ').title()}", value=median_val)
    else:
         sidebar_inputs[col] = st.sidebar.slider(f"{col.replace('_', ' ').title()}", min_value=min_val, max_value=max_val, value=median_val)

# Categorical Inputs - identified by dtype object or few unique numbers
cat_cols = df[FEATURE_COLUMNS].select_dtypes(include=['object', 'category']).columns.tolist()
# Add potential numerical categoricals like geo_ids
num_cat_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
cat_cols.extend([col for col in num_cat_cols if col in df.columns and col not in engineered_eq_cols]) # Exclude EQ cols here too

for col in cat_cols:
    unique_vals = get_unique_sorted(df[col])
    if unique_vals:
        # Find default index robustly
        default_value = df[col].mode()[0] if not df[col].mode().empty else unique_vals[0]
        try:
            default_ix = unique_vals.index(default_value)
        except ValueError:
            default_ix = 0 # Fallback to first option

        sidebar_inputs[col] = st.sidebar.selectbox(f"{col.replace('_', ' ').title()}", unique_vals, index=default_ix)
    else:
        st.sidebar.warning(f"No unique values found for {col}.")
        sidebar_inputs[col] = None # Or provide a default text input?

# Binary Inputs (Checkboxes) - identified by 'has_' prefix
st.sidebar.markdown("---")
st.sidebar.subheader("Binary Features (Superstructure/Use)")
has_cols = [col for col in FEATURE_COLUMNS if col.startswith('has_')]
has_secondary_use_toggle = None

for col in has_cols:
    label = col.replace('has_', '').replace('_', ' ').title()
    # Special handling for secondary use toggle
    if col == 'has_secondary_use':
        has_secondary_use_toggle = st.sidebar.checkbox(label, value=False)
        sidebar_inputs[col] = int(has_secondary_use_toggle)
    elif col.startswith('has_secondary_use_'):
        # Only show these if the main toggle is True
        if has_secondary_use_toggle:
            sidebar_inputs[col] = st.sidebar.checkbox(label, value=False)
        else:
            sidebar_inputs[col] = 0 # Default to 0 if main toggle is off
    else: # Superstructure flags
        sidebar_inputs[col] = st.sidebar.checkbox(label, value=False)

# Convert boolean checkbox values to int (0 or 1)
for key, value in sidebar_inputs.items():
    if isinstance(value, bool):
        sidebar_inputs[key] = int(value)

# --- Main Panel ---
st.header("Prediction Results")

# Create Input DataFrame from sidebar selections
# Filter sidebar_inputs to ensure only expected features are included
filtered_input_data = {k: v for k, v in sidebar_inputs.items() if k in FEATURE_COLUMNS and v is not None}

# Add engineered earthquake features (these were constant in the merged data)
if not df.empty:
     # Add the constant values directly from the source df
     for eq_col in engineered_eq_cols:
         if eq_col in FEATURE_COLUMNS: # Check if model expects it
             if eq_col in df.columns:
                 filtered_input_data[eq_col] = df[eq_col].iloc[0]
             else:
                 st.warning(f"Engineered feature {eq_col} expected by model but not found in data. Using 0.")
                 filtered_input_data[eq_col] = 0.0
else:
     st.warning("Cannot add earthquake features as data loading failed.")
     # Add defaults if required by FEATURE_COLUMNS
     for eq_col in engineered_eq_cols:
         if eq_col in FEATURE_COLUMNS:
              filtered_input_data[eq_col] = 0.0

# Create DataFrame with the correct column order specified by FEATURE_COLUMNS
input_df = pd.DataFrame([filtered_input_data])
# Reindex to ensure all FEATURE_COLUMNS are present and in order, fill missing with NaN initially
input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=np.nan)

# Check for missing values that weren't handled (e.g., categorical failed)
if input_df.isnull().values.any():
     st.error("Input data has missing values after processing sidebar inputs. Prediction cannot proceed.")
     st.dataframe(input_df.isnull().sum().reset_index(name='missing_count').query('missing_count > 0'))
     st.stop()


# Predict risk
if not input_df.empty and selected_model_name in MODELS:
    # No need to check for nulls again here if we stopped above
    st.write(f"Predicting using: **{selected_model_name}**")
    predicted_risk_categories, explanation_output = predict_risk(input_df, selected_model_name, MODELS)

    if predicted_risk_categories:
        risk_category = predicted_risk_categories[0]
        st.subheader("Predicted Damage Risk:")
        if risk_category == "Low":
            st.success(f"**{risk_category}**")
        elif risk_category == "Medium":
            st.warning(f"**{risk_category}**")
        elif risk_category == "High":
            st.error(f"**{risk_category}**")
        else:
            st.info(f"**{risk_category}**") # Handle 'Unknown' case

        # Display SHAP/Explanation if available (Placeholder)
        # if explanation_output is not None:
        #    st.subheader("Prediction Explanation (SHAP)")
        #    st_shap(explanation_output) # Requires streamlit_shap
    else:
        st.error("Prediction failed. Check logs or error messages above.")

# Removed redundant checks for empty/null input_df here as they are handled above
# elif input_df.isnull().values.any():
#      st.error("Prediction cannot be performed because some input values are missing.")
else:
    st.warning("Cannot perform prediction. Input data might be missing or model not selected/loaded.")


# --- Analysis Section ---
st.markdown("---")
st.header("Model Analysis")

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution (Overall)")
    # Plot distribution using FEATURE_COLUMNS_TUPLE for caching
    if FEATURE_COLUMNS_TUPLE:
        risk_dist_chart = plot_risk_distribution(df.copy(), selected_model_name, MODELS, list(FEATURE_COLUMNS_TUPLE))
        if risk_dist_chart:
            st.altair_chart(risk_dist_chart, use_container_width=True)
        else:
            st.info("Could not generate risk distribution plot.")
    else:
        st.warning("Feature columns not defined. Cannot generate risk distribution plot.")

with col2:
    st.subheader("Feature Importance")
    # Plot importance using FEATURE_COLUMNS_TUPLE for caching
    if FEATURE_COLUMNS_TUPLE:
        importance_chart = plot_feature_importance(selected_model_name, MODELS, FEATURE_COLUMNS_TUPLE)
        if importance_chart:
            st.altair_chart(importance_chart, use_container_width=True)
        else:
            st.info(f"Feature importance calculation not available or failed for {selected_model_name}.")
    else:
        st.warning("Feature columns not defined. Cannot generate feature importance plot.")


# --- Model Comparison Section --- # Added Section
st.markdown("---")
st.header("Model Performance Comparison")
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'model_comparison_metrics.csv')

@st.cache_data
def load_metrics(path):
    """Loads model comparison metrics."""
    try:
        metrics_df = pd.read_csv(path)
        return metrics_df
    except FileNotFoundError:
        st.error(f"Metrics file not found at {path}. Cannot display comparison.")
        return None
    except Exception as e:
        st.error(f"Error loading metrics file: {e}")
        return None

metrics = load_metrics(METRICS_PATH)
if metrics is not None:
    st.dataframe(metrics.set_index('Model')) # Set Model as index for better display
else:
    st.info("Model performance metrics are not available.")


st.markdown("---")
st.subheader("Raw Data Sample")
st.dataframe(df.head()) 