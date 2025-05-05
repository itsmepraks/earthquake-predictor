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
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # Get project root
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'buildings_features_earthquakes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model/Preprocessor Paths
LOG_REG_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
LOG_REG_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'baseline_preprocessor.joblib')
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_model.joblib')
LGBM_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'lightgbm_preprocessor.joblib')

# Risk Category Mapping (Adjust based on model training)
RISK_MAP = {
    "Logistic Regression": {1: "Low", 2: "Medium", 3: "High"},
    "LightGBM": {0: "Low", 1: "Medium", 2: "High"} # Assumes y was y-1 for training
}

# --- Data Loading ---
@st.cache_data # Cache the data loading
def load_data(path):
    """Loads the processed building features and earthquake data."""
    try:
        # Use BASE_DIR to construct absolute path if needed, assuming path is relative from root
        abs_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        df = pd.read_csv(abs_path)
        st.success(f"Successfully loaded data from {os.path.basename(abs_path)}")
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {abs_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Model & Preprocessor Loading --- # Added Section
@st.cache_resource # Cache resource loading (models, preprocessors)
def load_joblib(path):
    """Loads a joblib file from the specified path."""
    try:
        artifact = joblib.load(path)
        st.info(f"Successfully loaded artifact: {os.path.basename(path)}")
        return artifact
    except FileNotFoundError:
        st.error(f"Error: Artifact file not found at {path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the artifact from {path}: {e}")
        return None

# --- Load Artifacts ---
log_reg_model = load_joblib(LOG_REG_MODEL_PATH)
log_reg_preprocessor = load_joblib(LOG_REG_PREPROCESSOR_PATH)
lgbm_model = load_joblib(LGBM_MODEL_PATH)
lgbm_preprocessor = load_joblib(LGBM_PREPROCESSOR_PATH)

# Dictionary to hold loaded models and preprocessors
MODELS = {
    "Logistic Regression": {"model": log_reg_model, "preprocessor": log_reg_preprocessor},
    "LightGBM": {"model": lgbm_model, "preprocessor": lgbm_preprocessor}
}

# --- Prediction Function --- # Added Section
def predict_risk(input_df, model_name, models_dict):
    """Preprocesses input data and predicts risk using the selected model."""
    model_info = models_dict.get(model_name)
    if not model_info or model_info["model"] is None or model_info["preprocessor"] is None:
        st.error(f"Model or preprocessor for {model_name} not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    risk_map = RISK_MAP[model_name]

    try:
        # Ensure input_df columns match the order expected by the preprocessor
        # This requires knowing the original feature order used during preprocessor fitting.
        # For simplicity, assume input_df is already ordered correctly. A more robust
        # solution would explicitly reorder based on preprocessor.feature_names_in_
        st.write("Input DataFrame for Preprocessing:", input_df)
        st.write("Preprocessor type:", type(preprocessor))

        input_processed = preprocessor.transform(input_df)
        st.write("Shape after preprocessing:", input_processed.shape)

        predictions = model.predict(input_processed)
        st.write("Raw Predictions:", predictions)

        # Map numerical predictions to risk categories
        risk_categories = [risk_map.get(pred, "Unknown") for pred in predictions]
        return risk_categories

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Print traceback for debugging
        st.text(traceback.format_exc())
        return None
# --- End Prediction Function ---

# --- Visualization Functions --- # Added Section

@st.cache_data # Cache plots based on data and model name
def plot_risk_distribution(df, model_name, _models_dict):
    """Generates predictions for the dataset and plots risk distribution."""
    model_info = _models_dict.get(model_name)
    if not model_info or model_info["model"] is None or model_info["preprocessor"] is None:
        st.warning(f"Cannot generate distribution: Model/Preprocessor for {model_name} not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    risk_map = RISK_MAP[model_name]

    # Prepare features (assuming FEATURE_COLUMNS is defined globally)
    X = df[FEATURE_COLUMNS]

    try:
        X_processed = preprocessor.transform(X)
        predictions = model.predict(X_processed)
        risk_categories = pd.Series([risk_map.get(pred, "Unknown") for pred in predictions])
        risk_counts = risk_categories.value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']

        chart = alt.Chart(risk_counts).mark_bar().encode(
            x=alt.X('Risk Category', sort=['Low', 'Medium', 'High']),
            y='Count',
            tooltip=['Risk Category', 'Count']
        ).properties(
            title=f'Predicted Risk Distribution ({model_name})'
        )
        return chart
    except Exception as e:
        st.error(f"Error generating risk distribution plot: {e}")
        st.text(traceback.format_exc())
        return None

@st.cache_data # Cache plots based on model name
def plot_feature_importance(model_name, _models_dict, feature_columns, top_n=20):
    """Plots the top N feature importances for the selected model."""
    model_info = _models_dict.get(model_name)
    if not model_info or model_info["model"] is None:
        st.warning(f"Cannot plot importance: Model {model_name} not loaded.")
        return None

    model = model_info["model"]
    preprocessor = model_info["preprocessor"] # Needed for LogReg names
    importance_df = None

    try:
        if model_name == "LightGBM":
            if hasattr(model, 'feature_importances_') and feature_columns:
                importance_df = pd.DataFrame({
                    'feature': feature_columns, # Assumes feature_columns order matches training
                    'importance': model.feature_importances_
                }).sort_values(by='importance', ascending=False).head(top_n)
            else:
                st.warning("LightGBM model missing feature importances or feature columns list.")
                return None

        elif model_name == "Logistic Regression":
            if hasattr(model, 'coef_') and preprocessor is not None:
                try:
                    feature_names_out = preprocessor.get_feature_names_out()
                    # Average absolute coefficients across classes for OvR/Multinomial
                    avg_abs_coef = np.mean(np.abs(model.coef_), axis=0)
                    importance_df_raw = pd.DataFrame({
                        'feature': feature_names_out,
                        'importance': avg_abs_coef
                    })
                    # Attempt to group OHE features (heuristic)
                    importance_df_raw['original_feature'] = importance_df_raw['feature'].apply(
                        lambda x: x.split('__')[1].rsplit('_', 1)[0] if '__' in x and 'cat__' in x and x.split('__')[1].rsplit('_', 1)[0] != x.split('__')[1] else x.split('__')[1]
                    )
                    grouped_importance = importance_df_raw.groupby('original_feature')['importance'].sum().reset_index()
                    grouped_importance.columns = ['feature', 'importance'] # Rename columns for consistency
                    importance_df = grouped_importance.sort_values(by='importance', ascending=False).head(top_n)

                except Exception as e:
                    st.error(f"Error processing Logistic Regression coefficients: {e}")
                    st.text(traceback.format_exc())
                    return None
            else:
                st.warning("Logistic Regression model missing coefficients or preprocessor info.")
                return None
        else:
            st.warning(f"Feature importance plotting not implemented for {model_name}")
            return None

        if importance_df is not None and not importance_df.empty:
            chart = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X('importance', title='Importance'),
                y=alt.Y('feature', title='Feature', sort='-x'), # Sort by importance
                tooltip=['feature', 'importance']
            ).properties(
                title=f'Top {top_n} Feature Importances ({model_name})'
            )
            return chart
        else:
            st.warning("Could not generate feature importance data.")
            return None

    except Exception as e:
        st.error(f"Error generating feature importance plot: {e}")
        st.text(traceback.format_exc())
        return None

# --- End Visualization Functions ---

# --- App Setup ---
# Title
st.title("🇳🇵 Nepal Earthquake Risk Predictor")
st.markdown("""
This application predicts the risk level (Low, Medium, High) for buildings in Nepal based on the 2015 Gorkha earthquake data.
Select options in the sidebar to provide building details and see the prediction.
""")

# Load data
df = load_data(DATA_PATH)

# Check if data loaded
if df is None:
    st.error("Data loading failed. Cannot proceed.")
    st.stop()

# Get original feature columns (excluding target and ID) from loaded data
# These are needed to create the sample input dataframe
FEATURE_COLUMNS = df.drop(columns=['damage_grade', 'building_id'], errors='ignore').columns.tolist()

# --- Sidebar Inputs --- # Updated Section
st.sidebar.header("Controls")
selected_model_name = st.sidebar.selectbox(
    "Select Model", list(MODELS.keys()), index=1
)
st.sidebar.markdown("---")
st.sidebar.subheader("Building Features")

# Dictionary to hold user inputs
user_inputs = {}

# --- Define Input Widgets ---
# Numerical Inputs
user_inputs['geo_level_1_id'] = st.sidebar.slider(
    "Geo Level 1 ID",
    min_value=int(df['geo_level_1_id'].min()),
    max_value=int(df['geo_level_1_id'].max()),
    value=int(df['geo_level_1_id'].median())
)
user_inputs['age'] = st.sidebar.number_input(
    "Building Age (Years)",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=int(df['age'].median()),
    step=5
)
user_inputs['area_percentage'] = st.sidebar.slider(
    "Area Percentage",
    min_value=int(df['area_percentage'].min()),
    max_value=int(df['area_percentage'].max()),
    value=int(df['area_percentage'].median())
)
user_inputs['height_percentage'] = st.sidebar.slider(
    "Height Percentage",
    min_value=int(df['height_percentage'].min()),
    max_value=int(df['height_percentage'].max()),
    value=int(df['height_percentage'].median())
)

# Categorical Inputs
def get_unique_sorted(col_name):
    return sorted(df[col_name].unique().tolist())

# Foundation Type with descriptive labels
foundation_type_options = get_unique_sorted('foundation_type')
foundation_type_map = {
    'h': 'H: Pile',
    'i': 'I: Isolated',
    'r': 'R: Raft/Mat',
    'u': 'U: Under-reamed',
    'w': 'W: Well'
    # Add other codes if they exist, mapping to themselves or a description
}
# Ensure all options from data are in the map, defaulting to the code itself if not found
for code in foundation_type_options:
    if code not in foundation_type_map:
        foundation_type_map[code] = f"{code}: Unknown/Other"

user_inputs['foundation_type'] = st.sidebar.selectbox(
    "Foundation Type",
    options=foundation_type_options,
    format_func=lambda x: foundation_type_map.get(x, x), # Display descriptive name
    index=foundation_type_options.index(df['foundation_type'].mode()[0]) # Default to mode
)

# Other categorical inputs (keep showing codes for now, as meanings are unknown)
user_inputs['roof_type'] = st.sidebar.selectbox(
    "Roof Type (Code)", # Indicate it's a code
    options=get_unique_sorted('roof_type'),
    index=get_unique_sorted('roof_type').index(df['roof_type'].mode()[0])
)
user_inputs['ground_floor_type'] = st.sidebar.selectbox(
    "Ground Floor Type (Code)", # Indicate it's a code
    options=get_unique_sorted('ground_floor_type'),
    index=get_unique_sorted('ground_floor_type').index(df['ground_floor_type'].mode()[0])
)
# --- End Input Widgets ---

# Check if selected model and preprocessor loaded successfully
if MODELS[selected_model_name]["model"] is None or MODELS[selected_model_name]["preprocessor"] is None:
    st.error(f"Failed to load the selected model ({selected_model_name}) or its preprocessor.")
    st.stop()

# --- Main Area ---

# --- Prediction Demonstration --- # Updated Section
st.header("Prediction Example")

# Create input DataFrame from sidebar selections
# Ensure all FEATURE_COLUMNS are present, using defaults for those not in sidebar yet
sample_input_dict = {}
for col in FEATURE_COLUMNS:
    if col in user_inputs:
        sample_input_dict[col] = [user_inputs[col]] # Use sidebar value
    # Use defaults for features not yet added to sidebar
    elif pd.api.types.is_numeric_dtype(df[col]):
        sample_input_dict[col] = [df[col].median()]
    else:
        sample_input_dict[col] = [df[col].mode()[0]]

sample_input_df = pd.DataFrame(sample_input_dict)
# Ensure column order matches FEATURE_COLUMNS for the preprocessor
sample_input_df = sample_input_df[FEATURE_COLUMNS]

st.write(f"Predicting risk for building with selected features (using {selected_model_name}):")
# Optionally hide the full dataframe display if it gets too large
# st.dataframe(sample_input_df)
st.write("Selected Input Features:")
st.json({k: v[0] for k, v in sample_input_dict.items() if k in user_inputs}) # Show only sidebar inputs

if st.button("Predict Risk for Selected Features"):
    # Make sure predict_risk is defined
    if 'predict_risk' in globals():
        predicted_risk = predict_risk(sample_input_df, selected_model_name, MODELS)
        if predicted_risk:
            st.success(f"Predicted Risk Category: **{predicted_risk[0]}**")
        else:
            st.warning("Prediction failed. Check error messages above.")
    else:
        st.error("Prediction function not defined correctly.")
# --- End Prediction Demonstration ---

st.header("Model Insights & Data Exploration") # Updated Section

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution")
    # Generate and display the risk distribution chart
    risk_dist_chart = plot_risk_distribution(df, selected_model_name, MODELS)
    if risk_dist_chart:
        st.altair_chart(risk_dist_chart, use_container_width=True)
    else:
        st.markdown("*(Could not generate risk distribution chart)*")

with col2:
    st.subheader("Feature Importance")
    # Generate and display the feature importance chart
    # Pass FEATURE_COLUMNS for LightGBM name mapping
    feature_imp_chart = plot_feature_importance(selected_model_name, MODELS, FEATURE_COLUMNS)
    if feature_imp_chart:
        st.altair_chart(feature_imp_chart, use_container_width=True)
    else:
        st.markdown("*(Could not generate feature importance chart)*")

st.markdown("---")
st.header("Data Preview")
# Display loaded data if available
st.dataframe(df.head())

# --- Footer ---
st.markdown("---")
st.markdown("Developed for the Earthquake Risk Prediction Project.") 