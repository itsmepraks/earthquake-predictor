import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
DATA_PATH = 'data/processed/buildings_features_earthquakes.csv'
TARGET_COLUMN = 'damage_grade'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Constants ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'baseline_preprocessor.joblib')

# --- Data Loading ---
logging.info(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit()

# Clean potential leading/trailing whitespace in column names
df.columns = df.columns.str.strip()

# --- Feature and Target Separation ---
logging.info("Separating features and target variable...")
X = df.drop(columns=[TARGET_COLUMN, 'building_id']) # Drop target and identifier
y = df[TARGET_COLUMN]
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")


# --- Feature Preprocessing ---
logging.info("Setting up preprocessing pipelines...")

# Identify feature types based on column names and potential dtype (adjust as needed)
# Assuming geo_level IDs > 10 implies categorical treatment, adjust if needed.
categorical_features = [
    col for col in X.columns if
    'geo_level' in col or
    X[col].dtype == 'object' or
    col in ['land_surface_condition', 'foundation_type', 'roof_type',
             'ground_floor_type', 'other_floor_type', 'position',
             'plan_configuration', 'legal_ownership_status'] or
    'has_superstructure' in col or
    'has_secondary_use' in col and col != 'has_secondary_use' # Exclude the summary column if it exists
]

numerical_features = [
    col for col in X.columns if
    col not in categorical_features and
    pd.api.types.is_numeric_dtype(X[col])
]

logging.info(f"Identified {len(categorical_features)} categorical features.")
logging.info(f"Identified {len(numerical_features)} numerical features.")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Use handle_unknown='ignore' in case test set has categories not seen in train
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) - should be none if lists are correct
)

# --- Data Splitting ---
logging.info(f"Splitting data into train and test sets (test_size={TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Important for imbalanced classes
)
logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
logging.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# --- Applying Preprocessing ---
# Note: Fit the preprocessor ONLY on the training data
logging.info("Applying preprocessing to training data...")
X_train_processed = preprocessor.fit_transform(X_train)
logging.info("Applying preprocessing to testing data...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
try:
    # Get feature names from the ColumnTransformer
    feature_names_out = preprocessor.get_feature_names_out()
    logging.info(f"Processed training data shape: {X_train_processed.shape}")
    logging.info(f"Number of features after preprocessing: {len(feature_names_out)}")
except Exception as e:
    logging.warning(f"Could not get feature names from preprocessor: {e}")
    feature_names_out = None # Fallback

logging.info("Preprocessing and data splitting complete.")

# --- Baseline Model Training (Logistic Regression) ---
logging.info("Training baseline Logistic Regression model...")

# Instantiate the model
# Using 'balanced' class weight due to potential imbalance in damage grades
# Increased max_iter for convergence on potentially large dataset
log_reg = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000, # Increased iterations
    class_weight='balanced', # Handle imbalance
    solver='saga', # Changed solver
    multi_class='auto' # Automatically handles multi-class (likely OvR)
)

# Train the model on the processed training data
try:
    log_reg.fit(X_train_processed, y_train)
    logging.info("Logistic Regression model training complete.")

    # --- Save Model and Preprocessor ---
    logging.info("Saving model and preprocessor...")
    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logging.info(f"Ensured directory exists: {MODEL_DIR}")

    # Save the trained model
    joblib.dump(log_reg, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    # Save the fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logging.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    # --- End Save Section ---

except Exception as e:
    logging.error(f"Error during model training or saving: {e}")
    exit()

# --- Feature Importance Analysis ---
logging.info("Calculating feature importances...")
try:
    if feature_names_out is not None and hasattr(log_reg, 'coef_'):
        # Average absolute coefficients across classes for OvR
        avg_abs_coef = np.mean(np.abs(log_reg.coef_), axis=0)

        # Create a DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names_out,
            'importance': avg_abs_coef
        })

        # Group by original feature name (extract from processed name)
        # Example processed name: 'cat__geo_level_1_id_1', 'num__age'
        importance_df['original_feature'] = importance_df['feature'].apply(
            lambda x: x.split('__')[1].rsplit('_', 1)[0] if '__' in x and '_id_' not in x else x.split('__')[1] # Heuristic to group OHE features
            # A more robust way might be needed depending on exact naming
        )

        # Sum importances for OHE features belonging to the same original feature
        grouped_importance = importance_df.groupby('original_feature')['importance'].sum()

        # Sort and get top N features
        top_n = 20
        top_features = grouped_importance.sort_values(ascending=False).head(top_n)

        logging.info(f"--- Top {top_n} Feature Importances (Summed Abs Coef) ---")
        for feature, importance in top_features.items():
            logging.info(f"{feature}: {importance:.4f}")
        logging.info("--- End Feature Importances ---")

    else:
        logging.warning("Could not calculate feature importances (missing names or coefficients).")

except Exception as e:
    logging.error(f"Error calculating feature importances: {e}")

# --- Model Evaluation ---
logging.info("Evaluating model on the test set...")

# Make predictions
y_pred = log_reg.predict(X_test_processed)

# Get probability scores for ROC AUC
# Needed for multi-class ROC AUC calculation
y_pred_proba = log_reg.predict_proba(X_test_processed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# ROC AUC score for multi-class
# Using One-vs-Rest (OvR) strategy, average='macro' gives unweighted mean
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

# Print evaluation results
logging.info("--- Evaluation Results ---")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"ROC AUC (Macro OvR): {roc_auc:.4f}")
logging.info("\nConfusion Matrix:")
# Use print for better formatting of matrix
print(conf_matrix)
logging.info("\nClassification Report:")
# Use print for better formatting of report
print(class_report)
logging.info("--- End Evaluation ---")

# Next steps: Map probabilities to risk categories (if needed) and Phase 4 analysis.

# Example of how to access processed data (as numpy arrays)
# print("Sample processed training features:")
# print(X_train_processed[:2])
# print("Sample training target:")
# print(y_train[:5]) 