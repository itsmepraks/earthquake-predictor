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

# --- Categorical Feature Mappings (Code to Label) ---
# To be used for displaying user-friendly labels in selectboxes
FEATURE_VALUE_MAPS = {
    "land_surface_condition": {
        "n": "Flat",
        "o": "Moderate slope",
        "t": "Steep slope"
    },
    "foundation_type": {
        "h": "Adobe/Mud",
        "i": "Bamboo/Timber",
        "r": "RC (Reinforced Concrete)",
        "u": "Brick/Cement Mortar",
        "w": "Stone/Cement Mortar"
    },
    "roof_type": {
        "n": "RCC/RB/RBC",
        "q": "Bamboo/Timber-Light roof",
        "x": "Bamboo/Timber-Heavy roof"
    },
    "ground_floor_type": {
        "f": "Mud/Adobe",
        "m": "Mud Mortar-Stone/Brick",
        "v": "Cement-Stone/Brick",
        "x": "Timber",
        "z": "Other"
    },
    "other_floor_type": {
        "j": "Timber",
        "q": "RCC/RB/RBC",
        "s": "Tiled/Stone/Slate",
        "x": "Mud/Adobe"
    },
    "position": {
        "j": "Attached-1 side",
        "o": "Attached-2 sides",
        "s": "Not attached",
        "t": "Attached-3 sides"
    },
    "plan_configuration": {
        "a": "A-shape", "c": "C-shape", "d": "Rectangular", "f": "F-shape",
        "h": "H-shape", "l": "L-shape", "m": "Multi-projected", "n": "N-shape",
        "o": "Others", "q": "Square", "s": "S-shape", "t": "T-shape",
        "u": "U-shape", "z": "Z-shape"
    },
    "legal_ownership_status": {
        "a": "Attached",
        "r": "Rented",
        "v": "Private",
        "w": "Other/Unknown"
    }
    # geo_level_ids are numerous, so direct mapping isn't practical here.
    # They will remain as numerical inputs or direct code selectboxes.
}

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

# --- Helper function to get display value for selectbox ---
def get_display_value(options_dict, key):
    return f"{key} ({options_dict[key]})"

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

# --- Helper Functions (moved up, including load_metrics) ---
def _get_feature_names(preprocessor, original_feature_list):
    """
    Attempts to get feature names after transformation by a preprocessor.
    Handles ColumnTransformer and other common scikit-learn transformers.
    """
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception: # Catch broad exception as sklearn versions/implementations vary
            pass # Fall through to other methods

    if hasattr(preprocessor, 'transformers_'): # ColumnTransformer
        output_features = []
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == 'drop':
                continue
            if transformer == 'passthrough':
                # For passthrough, column names are typically indices or original names
                if all(isinstance(c, int) for c in columns):
                     output_features.extend([original_feature_list[i] for i in columns])
                else: # Assume columns are names
                     output_features.extend(columns)
            elif hasattr(transformer, 'get_feature_names_out'):
                try:
                    fitted_columns = [original_feature_list[i] if isinstance(i, int) else i for i in columns]
                    transformer_feature_names = list(transformer.get_feature_names_out(fitted_columns))
                    output_features.extend(transformer_feature_names)
                except TypeError: 
                     try:
                        transformer_feature_names = list(transformer.get_feature_names_out())
                        output_features.extend(transformer_feature_names)
                     except Exception:
                        output_features.extend(columns) 
                except Exception:
                    output_features.extend(columns)
            elif hasattr(transformer, 'categories_'): 
                for i, col_idx_or_name in enumerate(columns):
                    col_name = original_feature_list[col_idx_or_name] if isinstance(col_idx_or_name, int) else col_idx_or_name
                    if hasattr(transformer, 'categories_') and i < len(transformer.categories_): # Check against current transformer's categories
                        categories = transformer.categories_[i]
                        for category in categories:
                            output_features.append(f"{col_name}_{category}")
                    else: 
                        output_features.append(col_name)
            else: 
                output_features.extend(columns)
        return output_features

    if hasattr(preprocessor, 'steps'): # It's a Pipeline
        return _get_feature_names(preprocessor.steps[-1][1], original_feature_list)

    return original_feature_list

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
        input_processed_np = preprocessor.transform(input_df)
        processed_feature_names = _get_feature_names(preprocessor, list(input_df.columns))

        # Convert to dense if sparse, then create DataFrame
        if hasattr(input_processed_np, "toarray"): # Check for sparse matrix
            input_processed_dense = input_processed_np.toarray()
        else: # Assumes it's already a NumPy array or compatible type for DataFrame
            input_processed_dense = input_processed_np

        if len(processed_feature_names) == input_processed_dense.shape[1]:
            input_processed_df = pd.DataFrame(input_processed_dense, columns=processed_feature_names)
        else:
            st.error(f"Mismatch in predict_risk: processed feature names ({len(processed_feature_names)}) vs data columns ({input_processed_dense.shape[1]}) for {model_name}. Using generic column names.")
            input_processed_df = pd.DataFrame(input_processed_dense) # Fallback to generic column names

        predictions = model.predict(input_processed_df)
        risk_categories = [risk_map.get(pred, "Unknown") for pred in predictions]
        explanation_values = None # Placeholder for now
        return risk_categories, explanation_values

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
        X_processed_np = preprocessor.transform(X)
        processed_feature_names = _get_feature_names(preprocessor, feature_columns)

        # Convert to dense if sparse, then create DataFrame
        if hasattr(X_processed_np, "toarray"): # Check for sparse matrix
            X_processed_dense = X_processed_np.toarray()
        else: # Assumes it's already a NumPy array or compatible type for DataFrame
            X_processed_dense = X_processed_np
        
        if len(processed_feature_names) == X_processed_dense.shape[1]:
            X_processed_df = pd.DataFrame(X_processed_dense, columns=processed_feature_names)
        else:
            st.error(f"Mismatch in plot_risk_distribution: processed names ({len(processed_feature_names)}) vs data columns ({X_processed_dense.shape[1]}) for {model_name}. Using generic names for plot.")
            X_processed_df = pd.DataFrame(X_processed_dense) # Fallback to generic column names

        predictions = model.predict(X_processed_df)
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

@st.cache_data # Cache plots based on model name and feature columns list tuple
def plot_feature_importance(model_name, _models_dict, _feature_columns_tuple, top_n=20):
    """Plots feature importance for the selected model if available."""
    model_info = _models_dict.get(model_name)
    if not model_info or model_info["model"] is None:
        st.warning(f"Feature importance not available for {model_name} or model not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info.get("preprocessor") # Get preprocessor if available

    importances = None
    feature_names = list(_feature_columns_tuple) # Convert tuple back to list for manipulation

    # st.write(f"Model type for feature importance: {type(model)}") # Debugging model type

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, coef_ might be multi-dimensional for multi-class
        if model.coef_.ndim > 1:
            # Averaging coefficients across classes for a single importance score
            # Or, one might choose to plot for a specific class if relevant
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_[0]) # For binary or single output linear models
    else:
        st.warning(f"Cannot extract feature importance for model type: {type(model).__name__}")
        return None

    if importances is None:
        st.warning(f"Could not compute importances for {model_name}")
        return None

    # Get processed feature names
    # This needs to be robust based on the preprocessor used for the model
    if preprocessor:
        try:
            processed_feature_names = _get_feature_names(preprocessor, feature_names)
            # st.write(f"Original features: {feature_names}")
            # st.write(f"Processed features from preprocessor: {processed_feature_names}")

            if len(processed_feature_names) == len(importances):
                feature_names = processed_feature_names
            else:
                st.warning(f"Mismatch in length between processed feature names ({len(processed_feature_names)}) and importances ({len(importances)}). Falling back to original feature names. This might be incorrect.")
                # If there's a mismatch, it might indicate an issue with _get_feature_names or
                # how features are handled post-preprocessing by the model.
                # Fallback to original feature names, but this is a point of potential error.
                if len(feature_names) != len(importances):
                     st.error(f"Critical mismatch: Original feature count ({len(feature_names)}) also differs from importance count ({len(importances)}). Cannot plot.")
                     return None

        except Exception as e:
            st.warning(f"Error getting processed feature names for {model_name}: {e}. Using original feature names.")
            if len(feature_names) != len(importances):
                st.error(f"Critical mismatch with original features: Original feature count ({len(feature_names)}) differs from importance count ({len(importances)}). Cannot plot.")
                return None
    else: # No preprocessor found, assume model was trained on original features (unlikely for complex models)
        st.warning(f"No preprocessor found for {model_name}. Assuming model was trained directly on input features (this might be incorrect).")
        if len(feature_names) != len(importances):
            st.error(f"Critical mismatch (no preprocessor): Original feature count ({len(feature_names)}) differs from importance count ({len(importances)}). Cannot plot.")
            return None


    # st.write(f"Final feature names for importance plot: {feature_names}")
    # st.write(f"Importances values: {importances}")


    # Create DataFrame for plotting
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n)

    chart = alt.Chart(importance_df).mark_bar().encode(
        x='importance:Q',
        y=alt.Y('feature:N', sort='-x'),
        tooltip=['feature', 'importance']
    ).properties(
        title=f"Top {top_n} Feature Importances ({model_name})"
    )
    return chart

# Helper to get original feature names if one-hot encoded
# This version tries to be more robust for different transformers in ColumnTransformer
def get_original_feature(name):
    # Simpler logic: Remove prefix before first '__' if present
    # This is a common pattern for OneHotEncoder, but might need adjustment
    # if other transformers (like Scaler) are named and come first.
    parts = name.split('__')
    if len(parts) > 1:
        # Check if the first part is a known transformer name prefix
        # This list might need to be expanded based on your preprocessor
        known_transformer_prefixes = ['onehotencoder', 'ordinalencoder', 'passthrough', 'remainder', 'pipeline', 'standardscaler', 'minmaxscaler', 'robustscaler']
        # A more specific check for common transformer prefixes from ColumnTransformer
        # This is still heuristic. The ideal solution is to get feature_names_out_
        # from each transformer if possible.
        potential_transformer_name = parts[0].lower()
        is_transformer_prefix = any(potential_transformer_name.startswith(prefix) for prefix in known_transformer_prefixes)

        if is_transformer_prefix:
            return parts[-1] # Assume the last part is the original feature name or category
        else:
            return name # If the prefix isn't a known transformer, it might be part of the feature name itself
    return name


@st.cache_data
def get_unique_sorted(series):
    """Gets unique sorted values from a Pandas Series."""
    return sorted(series.unique())

# --- Streamlit App Layout ---

# --- Main App ---
def main():
    st.title("🇳🇵 Nepal Earthquake Damage Risk Predictor")
    st.markdown("Predict the risk of building damage from earthquakes in Nepal based on building characteristics.")

    # --- Data Loading ---
    data = load_data(DATA_PATH)
    if data is None:
        st.stop() # Stop execution if data loading fails

    # Identify feature columns (excluding target and any ID columns not used as features)
    # This needs to be robust. Assuming 'damage_grade' is target.
    # And 'building_id' is an identifier.
    potential_feature_cols = [col for col in data.columns if col not in ['damage_grade', 'building_id']]


    # --- Sidebar for Inputs ---
    st.sidebar.header("Configure Prediction Inputs")
    selected_model_name = st.sidebar.selectbox(
        "Choose Prediction Model",
        list(MODELS.keys()),
        help="Select the machine learning model to use for prediction."
    )

    # Dynamically get the feature columns expected by the selected model's preprocessor
    # This assumes the preprocessor stores feature_names_in_
    current_preprocessor = MODELS.get(selected_model_name, {}).get('preprocessor')
    if current_preprocessor and hasattr(current_preprocessor, 'feature_names_in_'):
        APP_FEATURE_COLUMNS = list(current_preprocessor.feature_names_in_)
    else:
        # Fallback: Use a predefined list or all columns if feature_names_in_ is not available
        # This fallback might not be accurate if preprocessors differ significantly
        st.sidebar.warning(f"Could not determine exact feature set for {selected_model_name} from preprocessor. Using a general set. Predictions might be affected if this is incorrect.")
        APP_FEATURE_COLUMNS = potential_feature_cols # Or a manually curated default list


    input_data = {}

    # Use a form for all inputs
    with st.sidebar.form(key='prediction_input_form'):
        st.subheader("Building & Location Characteristics")
        st.caption("Adjust the features below to get a risk prediction.")

        # Grouping inputs using expanders
        with st.expander("🌍 Geographical Location IDs", expanded=True):
            # Numerical Inputs for Geo IDs (as they are high cardinality)
            # Using number_input for geo_level_ids as they are numerical but treated as categories by some models
            # It's crucial these match the training data values.
            # For geo_level_ids, providing selectboxes with all unique values would be too long.
            # Defaulting to common/median values from the dataset if possible, or min.
            default_geo1 = data['geo_level_1_id'].median() if 'geo_level_1_id' in data else 0
            default_geo2 = data['geo_level_2_id'].median() if 'geo_level_2_id' in data else 0
            default_geo3 = data['geo_level_3_id'].median() if 'geo_level_3_id' in data else 0

            if 'geo_level_1_id' in APP_FEATURE_COLUMNS:
                input_data['geo_level_1_id'] = st.number_input(
                    'Geo Level 1 ID',
                    min_value=int(data['geo_level_1_id'].min()),
                    max_value=int(data['geo_level_1_id'].max()),
                    value=int(default_geo1),
                    step=1,
                    help="Identifier for the first geographic region (0-30)."
                )
            if 'geo_level_2_id' in APP_FEATURE_COLUMNS:
                input_data['geo_level_2_id'] = st.number_input(
                    'Geo Level 2 ID',
                    min_value=int(data['geo_level_2_id'].min()),
                    max_value=int(data['geo_level_2_id'].max()),
                    value=int(default_geo2),
                    step=1,
                    help="Identifier for the second geographic region (0-1427)."
                )
            if 'geo_level_3_id' in APP_FEATURE_COLUMNS:
                input_data['geo_level_3_id'] = st.number_input(
                    'Geo Level 3 ID',
                    min_value=int(data['geo_level_3_id'].min()),
                    max_value=int(data['geo_level_3_id'].max()),
                    value=int(default_geo3),
                    step=1,
                    help="Identifier for the third geographic region (0-12567)."
                )

        with st.expander("🏗️ Structural Properties", expanded=False):
            if 'count_floors_pre_eq' in APP_FEATURE_COLUMNS:
                input_data['count_floors_pre_eq'] = st.slider(
                    'Number of Floors',
                    min_value=int(data['count_floors_pre_eq'].min()),
                    max_value=int(data['count_floors_pre_eq'].max()), # Max floors from data
                    value=int(data['count_floors_pre_eq'].median()), # Default to median
                    step=1,
                    help="Number of floors in the building before the earthquake."
                )
            if 'age_building' in APP_FEATURE_COLUMNS:
                input_data['age_building'] = st.slider(
                    'Age of Building (Years)',
                    min_value=int(data['age_building'].min()),
                    max_value=int(data['age_building'].max()), # Max age from data
                    value=int(data['age_building'].median()),
                    step=5,
                    help="Age of the building in years."
                )
            if 'height_ft_pre_eq' in APP_FEATURE_COLUMNS: # Renamed from area_percentage
                input_data['height_ft_pre_eq'] = st.slider( # Changed from area_percentage
                    'Building Height (ft)', # Renamed from area_percentage
                    min_value=int(data['height_ft_pre_eq'].min()), # Renamed from area_percentage
                    max_value=int(data['height_ft_pre_eq'].max()), # Max height, Renamed from area_percentage
                    value=int(data['height_ft_pre_eq'].median()), # Renamed from area_percentage
                    step=1,
                    help="Height of the building in feet before the earthquake." # Renamed from area_percentage
                )

        with st.expander("🧱 Material Properties", expanded=False):
            for feature, mapping in FEATURE_VALUE_MAPS.items():
                if feature in APP_FEATURE_COLUMNS:
                    # Create a list of (display_value, code) for the selectbox
                    # Sort options by key (code) to maintain consistency if needed
                    sorted_options = sorted(mapping.items())
                    display_options = [get_display_value(mapping, code) for code, _ in sorted_options]
                    actual_codes = [code for code, _ in sorted_options]

                    # Find default or first option's code
                    default_code = actual_codes[0] if actual_codes else None
                    if feature in data:
                        # Try to set default to the most frequent value if it exists in mapping
                        most_frequent = data[feature].mode()[0]
                        if most_frequent in actual_codes:
                            default_code = most_frequent

                    selected_display_value = st.selectbox(
                        label=feature.replace("_", " ").title(),
                        options=display_options,
                        index=actual_codes.index(default_code) if default_code and default_code in actual_codes else 0,
                        help=f"Type of {feature.replace('_', ' ')}."
                    )
                    # Map selected display value back to code
                    selected_code = actual_codes[display_options.index(selected_display_value)]
                    input_data[feature] = selected_code


        with st.expander("📐 Design and Plan", expanded=False):
            if 'plan_configuration' in APP_FEATURE_COLUMNS and 'plan_configuration' in FEATURE_VALUE_MAPS:
                mapping = FEATURE_VALUE_MAPS['plan_configuration']
                sorted_options = sorted(mapping.items())
                display_options = [get_display_value(mapping, code) for code, _ in sorted_options]
                actual_codes = [code for code, _ in sorted_options]
                default_code = data['plan_configuration'].mode()[0] if 'plan_configuration' in data and data['plan_configuration'].mode()[0] in actual_codes else actual_codes[0]
                selected_display_value = st.selectbox(
                        "Plan Configuration",
                        options=display_options,
                        index=actual_codes.index(default_code),
                        help="Building plan configuration."
                )
                input_data['plan_configuration'] = actual_codes[display_options.index(selected_display_value)]

            if 'position' in APP_FEATURE_COLUMNS and 'position' in FEATURE_VALUE_MAPS:
                mapping = FEATURE_VALUE_MAPS['position']
                sorted_options = sorted(mapping.items())
                display_options = [get_display_value(mapping, code) for code, _ in sorted_options]
                actual_codes = [code for code, _ in sorted_options]
                default_code = data['position'].mode()[0] if 'position' in data and data['position'].mode()[0] in actual_codes else actual_codes[0]
                selected_display_value = st.selectbox(
                        "Building Position",
                        options=display_options,
                        index=actual_codes.index(default_code),
                        help="Position of the building relative to other buildings."
                )
                input_data['position'] = actual_codes[display_options.index(selected_display_value)]


        with st.expander("🏠 Usage Metrics", expanded=False):
            if 'count_families' in APP_FEATURE_COLUMNS:
                input_data['count_families'] = st.slider(
                    'Number of Families',
                    min_value=int(data['count_families'].min()),
                    max_value=int(data['count_families'].max()), # Max families from data
                    value=int(data['count_families'].median()),
                    step=1,
                    help="Number of families residing in the building."
                )
            # Boolean features related to usage (superstructure)
            bool_features_display = {
                "has_secondary_use": "Has Secondary Use?",
                "has_secondary_use_agriculture": "Secondary Use: Agriculture?",
                "has_secondary_use_hotel": "Secondary Use: Hotel?",
                "has_secondary_use_rental": "Secondary Use: Rental?",
                "has_secondary_use_institution": "Secondary Use: Institution?",
                "has_secondary_use_school": "Secondary Use: School?",
                "has_secondary_use_industry": "Secondary Use: Industry?",
                "has_secondary_use_health_post": "Secondary Use: Health Post?",
                "has_secondary_use_gov_office": "Secondary Use: Gov Office?",
                "has_secondary_use_use_police": "Secondary Use: Police?", # Typo in original? use_police
                "has_secondary_use_other": "Secondary Use: Other?"
            }
            for feature, display_name in bool_features_display.items():
                if feature in APP_FEATURE_COLUMNS:
                    default_val = bool(data[feature].mode()[0]) if feature in data else False
                    input_data[feature] = st.checkbox(display_name, value=default_val)


        with st.expander("⚖️ Ownership", expanded=False):
            if 'legal_ownership_status' in APP_FEATURE_COLUMNS and 'legal_ownership_status' in FEATURE_VALUE_MAPS:
                mapping = FEATURE_VALUE_MAPS['legal_ownership_status']
                sorted_options = sorted(mapping.items())
                display_options = [get_display_value(mapping, code) for code, _ in sorted_options]
                actual_codes = [code for code, _ in sorted_options]
                default_code = data['legal_ownership_status'].mode()[0] if 'legal_ownership_status' in data and data['legal_ownership_status'].mode()[0] in actual_codes else actual_codes[0]

                selected_display_value = st.selectbox(
                        "Legal Ownership Status",
                        options=display_options,
                        index=actual_codes.index(default_code),
                        help="Legal ownership status of the land."
                )
                input_data['legal_ownership_status'] = actual_codes[display_options.index(selected_display_value)]


        with st.expander("🏛️ Superstructure Features (Materials)", expanded=False):
            # Boolean flags for superstructure types
            superstructure_bool_features_display = {
                "has_superstructure_adobe_mud": "Superstructure: Adobe/Mud",
                "has_superstructure_mud_mortar_stone": "Superstructure: Mud Mortar Stone",
                "has_superstructure_stone_flag": "Superstructure: Stone Flag",
                "has_superstructure_cement_mortar_stone": "Superstructure: Cement Mortar Stone",
                "has_superstructure_mud_mortar_brick": "Superstructure: Mud Mortar Brick",
                "has_superstructure_cement_mortar_brick": "Superstructure: Cement Mortar Brick",
                "has_superstructure_timber": "Superstructure: Timber",
                "has_superstructure_bamboo": "Superstructure: Bamboo",
                "has_superstructure_rc_non_engineered": "Superstructure: RC Non-Engineered",
                "has_superstructure_rc_engineered": "Superstructure: RC Engineered",
                "has_superstructure_other": "Superstructure: Other"
            }
            for feature, display_name in superstructure_bool_features_display.items():
                if feature in APP_FEATURE_COLUMNS:
                    default_val = bool(data[feature].mode()[0]) if feature in data else False
                    input_data[feature] = st.checkbox(display_name, value=default_val)
        
        # Form submit button
        submitted = st.form_submit_button("Predict Risk")


    # --- Main Panel for Results ---
    # Only run prediction if the form has been submitted
    if submitted:
        # Filter input_data to only include features expected by the model (APP_FEATURE_COLUMNS)
        # This is crucial because input_data might temporarily hold keys for all possible features
        # while APP_FEATURE_COLUMNS is specific to the selected model.
        filtered_input_data = {k: v for k, v in input_data.items() if k in APP_FEATURE_COLUMNS}

        # Create a DataFrame from the filtered inputs
        # Ensure the order of columns matches APP_FEATURE_COLUMNS
        try:
            input_df = pd.DataFrame([filtered_input_data], columns=APP_FEATURE_COLUMNS)
            # Convert boolean features explicitly to int (0 or 1) if preprocessor expects numerical
            # This depends on how your preprocessor handles booleans.
            # If OneHotEncoder handles bools, this might not be needed.
            # If models expect numerical 0/1, this is important.
            for col in input_df.select_dtypes(include='bool').columns:
                input_df[col] = input_df[col].astype(int)

            # st.write("Input DataFrame for Prediction:", input_df) # Debugging
            # st.write("Columns in input_df:", input_df.columns.tolist()) # Debugging
            # st.write("APP_FEATURE_COLUMNS:", APP_FEATURE_COLUMNS) # Debugging

        except Exception as e:
            st.error(f"Error creating input DataFrame: {e}. Check feature consistency.")
            st.error(f"Filtered input data: {filtered_input_data}")
            st.error(f"Expected columns by preprocessor: {APP_FEATURE_COLUMNS}")
            input_df = None # Ensure it's None if creation fails

        if input_df is not None:
            prediction, _ = predict_risk(input_df, selected_model_name, MODELS) # Ignoring SHAP for now

            if prediction:
                st.subheader(f"Predicted Risk ({selected_model_name}):")
                # Displaying the first prediction as we are predicting for a single instance
                st.metric(label="Damage Risk Level", value=str(prediction[0]))
                if prediction[0] == "High":
                    st.error("🔴 High Risk Predicted")
                elif prediction[0] == "Medium":
                    st.warning("🟡 Medium Risk Predicted")
                else:
                    st.success("🟢 Low Risk Predicted")

                st.markdown("---")
                st.subheader("Input Summary:")
                # Display the input data used for prediction
                # Create a DataFrame for better display, transpose for readability
                input_summary_df = pd.DataFrame.from_dict(filtered_input_data, orient='index', columns=['Value'])
                input_summary_df.index.name = "Feature"
                st.dataframe(input_summary_df, use_container_width=True)

            else:
                st.error("Could not retrieve a prediction. Please check logs or inputs.")
        else:
            st.warning("Prediction could not be made due to input DataFrame creation issues.")


    st.markdown("---")

    # --- Tabs for Additional Information ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Distribution",
        "⚖️ Model Performance Comparison",
        "💡 Feature Importances",
        "ℹ️ Data Dictionary"
    ])

    with tab1:
        st.header("Overall Risk Distribution in Dataset")
        st.markdown(f"Distribution of predicted damage risk categories using **{selected_model_name}** on the full dataset.")
        # Note: plot_risk_distribution expects feature_columns that exist in `data`
        # Ensure APP_FEATURE_COLUMNS used here are valid for the `data` DataFrame
        valid_cols_for_dist_plot = [col for col in APP_FEATURE_COLUMNS if col in data.columns]
        if not valid_cols_for_dist_plot:
             st.warning(f"No valid features found to generate risk distribution for {selected_model_name} based on current APP_FEATURE_COLUMNS.")
        else:
            risk_dist_chart = plot_risk_distribution(data, selected_model_name, MODELS, valid_cols_for_dist_plot)
            if risk_dist_chart:
                st.altair_chart(risk_dist_chart, use_container_width=True)
            else:
                st.info("Risk distribution chart could not be generated for the selected model or data.")

    with tab2:
        st.header("Model Performance Comparison")
        st.markdown("Comparison of performance metrics across different models.")
        # Load metrics from CSV - this should be generated by a separate script
        metrics_df = load_metrics(os.path.join(BASE_DIR, 'reports', 'model_performance_summary.csv'))
        if metrics_df is not None:
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=pd.IndexSlice[:, ['Accuracy', 'F1 (Weighted)', 'AUC (OvR Macro)']], color='lightgreen'), use_container_width=True)
        else:
            st.warning("Model performance summary not available.")

    with tab3:
        st.header("Feature Importances")
        st.markdown(f"Importance of features for the **{selected_model_name}** model.")
        # Convert APP_FEATURE_COLUMNS to tuple for caching
        feature_importance_chart = plot_feature_importance(selected_model_name, MODELS, tuple(APP_FEATURE_COLUMNS))
        if feature_importance_chart:
            st.altair_chart(feature_importance_chart, use_container_width=True)
        else:
            st.info(f"Feature importance plot not available or not applicable for {selected_model_name}.")

    with tab4:
        st.header("Data Dictionary")
        st.markdown("Description of features used in the dataset.")
        # Consider loading this from data_dictionary.md or a CSV for better maintenance
        # For now, displaying a placeholder or a sample of FEATURE_VALUE_MAPS
        st.write(FEATURE_VALUE_MAPS) # Simple display, can be improved
        st.markdown("For a full data dictionary, please refer to `data/data_dictionary.md`.")


if __name__ == "__main__":
    main() 