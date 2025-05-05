import logging
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'lightgbm_preprocessor.joblib')

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

# Convert target variable to 0-based index for LightGBM (0, 1, 2)
# LightGBM expects class labels from 0 to n_classes-1
y = y - 1
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
logging.info(f"Target values unique (0-based): {np.unique(y)}")


# --- Feature Preprocessing --- 
# LightGBM can handle categorical features directly if integer-encoded
logging.info("Setting up preprocessing pipelines (Ordinal for categorical)...")

# Identify feature types
categorical_features = [
    col for col in X.columns if
    'geo_level' in col or
    X[col].dtype == 'object' or
    col in ['land_surface_condition', 'foundation_type', 'roof_type',
             'ground_floor_type', 'other_floor_type', 'position',
             'plan_configuration', 'legal_ownership_status'] or
    'has_superstructure' in col or
    'has_secondary_use' in col and col != 'has_secondary_use'
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

# Use OrdinalEncoder for categorical features
categorical_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) # Handle potential new categories in test
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
    X, y, # Using 0-based target
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
logging.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# --- Applying Preprocessing ---
logging.info("Applying preprocessing to training data...")
X_train_processed = preprocessor.fit_transform(X_train)
logging.info("Applying preprocessing to testing data...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names after transformation
feature_names_out = numerical_features + categorical_features # Order based on ColumnTransformer

# Identify indices of categorical features AFTER transformation
categorical_feature_indices = [feature_names_out.index(col) for col in categorical_features]

logging.info(f"Processed training data shape: {X_train_processed.shape}")
logging.info(f"Number of features after preprocessing: {X_train_processed.shape[1]}")
logging.info("Preprocessing and data splitting complete.")


# --- LightGBM Model Training ---
logging.info("Training LightGBM model...")

# Instantiate the model
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3, # We have 3 damage grades (0, 1, 2)
    metric='multi_logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1, # Use all available cores
    class_weight='balanced' # Handle class imbalance
)

# Train the model
try:
    lgbm.fit(
        X_train_processed,
        y_train,
        categorical_feature=categorical_feature_indices, # Tell LightGBM which features are categorical
        eval_set=[(X_test_processed, y_test)], # Use test set for early stopping if desired
        # callbacks=[lgb.early_stopping(10, verbose=True)] # Optional: uncomment for early stopping
    )
    logging.info("LightGBM model training complete.")

    # --- Save Model and Preprocessor ---
    logging.info("Saving LightGBM model and preprocessor...")
    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logging.info(f"Ensured directory exists: {MODEL_DIR}")

    # Save the trained model
    joblib.dump(lgbm, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    # Save the fitted preprocessor (includes scaler and ordinal encoder)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logging.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    # --- End Save Section ---

except Exception as e:
    logging.error(f"Error during LightGBM training or saving: {e}")
    exit()

# --- Feature Importance Analysis ---
logging.info("Calculating LightGBM feature importances...")
try:
    importance_df = pd.DataFrame({
        'feature': feature_names_out,
        'importance': lgbm.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top_n = 20
    logging.info(f"--- Top {top_n} Feature Importances (LightGBM) ---")
    for index, row in importance_df.head(top_n).iterrows():
        logging.info(f"{row['feature']}: {row['importance']}")
    logging.info("--- End Feature Importances ---")
except Exception as e:
    logging.error(f"Error calculating feature importances: {e}")


# --- Model Evaluation ---
logging.info("Evaluating LightGBM model on the test set...")

# Make predictions
y_pred = lgbm.predict(X_test_processed)
y_pred_proba = lgbm.predict_proba(X_test_processed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Grade 1', 'Grade 2', 'Grade 3']) # Use 0-based labels

# ROC AUC score for multi-class
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

# Print evaluation results
logging.info("--- Evaluation Results (LightGBM) ---")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"ROC AUC (Macro OvR): {roc_auc:.4f}")
logging.info("\nConfusion Matrix (Labels: 0, 1, 2):")
print(conf_matrix)
logging.info("\nClassification Report:")
print(class_report)
logging.info("--- End Evaluation ---") 