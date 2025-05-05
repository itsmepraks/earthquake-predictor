import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC  # Import LinearSVC

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib') # Updated model path
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'svm_preprocessor.joblib') # Updated preprocessor path

# Define constants
DATA_PATH = 'data/processed/buildings_features_earthquakes.csv'
TARGET_COLUMN = 'damage_grade'
TEST_SIZE = 0.2
RANDOM_STATE = 42

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

# Convert target variable to 0-based index
y = y - 1
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
logging.info(f"Target values unique (0-based): {np.unique(y)}")

# --- Feature Preprocessing ---
logging.info("Setting up preprocessing pipelines (OHE for categorical)...")

# Identify feature types
categorical_features = [
    col for col in X.columns if
    'geo_level' in col or
    X[col].dtype == 'object' or
    col in ['land_surface_condition', 'foundation_type', 'roof_type',
             'ground_floor_type', 'other_floor_type', 'position',
             'plan_configuration', 'legal_ownership_status'] or
    'has_superstructure' in col or
    ('has_secondary_use' in col and col != 'has_secondary_use')
]

numerical_features = [
    col for col in X.columns if
    col not in categorical_features and
    pd.api.types.is_numeric_dtype(X[col])
]

logging.info(f"Identified {len(categorical_features)} categorical features.")
logging.info(f"Identified {len(numerical_features)} numerical features.")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
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
logging.info("Applying preprocessing to training data...")
X_train_processed = preprocessor.fit_transform(X_train)
logging.info("Applying preprocessing to testing data...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
try:
    feature_names_out = preprocessor.get_feature_names_out()
    logging.info(f"Processed training data shape: {X_train_processed.shape}")
    logging.info(f"Number of features after preprocessing: {len(feature_names_out)}")
except Exception as e:
    logging.warning(f"Could not get feature names from preprocessor: {e}")
    feature_names_out = None # Fallback

logging.info("Preprocessing and data splitting complete.")

# --- SVM Model Training (LinearSVC) --- # Updated Section Title
logging.info("Training Linear SVM (LinearSVC) model...")

# Instantiate the model
svm_clf = LinearSVC( # Changed class and variable name
    random_state=RANDOM_STATE,
    max_iter=2000, # Increased iterations, might need more
    class_weight='balanced', # Handle imbalance
    dual=False, # Recommended when n_samples > n_features
    C=1.0 # Default regularization
)

# Train the model on the processed training data
try:
    svm_clf.fit(X_train_processed, y_train)
    logging.info("LinearSVC model training complete.")

    # --- Save Model and Preprocessor ---
    logging.info("Saving SVM model and preprocessor...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    logging.info(f"Ensured directory exists: {MODEL_DIR}")

    joblib.dump(svm_clf, MODEL_PATH) # Use updated path
    logging.info(f"Model saved to {MODEL_PATH}")

    joblib.dump(preprocessor, PREPROCESSOR_PATH) # Use updated path
    logging.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    # --- End Save Section ---

except Exception as e:
    logging.error(f"Error during SVM model training or saving: {e}")
    exit()

# --- Feature Importance Analysis (Coefficients) ---
logging.info("Calculating feature importances (coefficients) for LinearSVC...")
try:
    if feature_names_out is not None and hasattr(svm_clf, 'coef_'):
        # Average absolute coefficients across classes
        avg_abs_coef = np.mean(np.abs(svm_clf.coef_), axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names_out,
            'importance': avg_abs_coef
        })

        # Group by original feature name (heuristic for OHE)
        importance_df['original_feature'] = importance_df['feature'].apply(
            lambda x: x.split('__')[1].rsplit('_', 1)[0] if '__' in x and 'cat__' in x and x.split('__')[1].rsplit('_', 1)[0] != x.split('__')[1] else x.split('__')[1]
        )
        grouped_importance = importance_df.groupby('original_feature')['importance'].sum()

        top_n = 20
        top_features = grouped_importance.sort_values(ascending=False).head(top_n)

        logging.info(f"--- Top {top_n} Feature Importances (Summed Abs Coef - LinearSVC) ---")
        for feature, importance in top_features.items():
            logging.info(f"{feature}: {importance:.4f}")
        logging.info("--- End Feature Importances ---")

    else:
        logging.warning("Could not calculate feature importances (missing names or coefficients).")

except Exception as e:
    logging.error(f"Error calculating SVM feature importances: {e}")

# --- Model Evaluation ---
logging.info("Evaluating LinearSVC model on the test set...")

# Make predictions
y_pred = svm_clf.predict(X_test_processed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Grade 1', 'Grade 2', 'Grade 3'])

# Print evaluation results
logging.info("--- Evaluation Results (LinearSVC) ---") # Updated label
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info("\nConfusion Matrix (Labels: 0, 1, 2):")
print(conf_matrix)
logging.info("\nClassification Report:")
print(class_report)
logging.info("--- End Evaluation ---") 