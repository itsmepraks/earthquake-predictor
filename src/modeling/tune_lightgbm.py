import logging
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_tuned_model.joblib') # Tuned model path
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'lightgbm_preprocessor.joblib') # Re-use preprocessor path if consistent

# Define constants
DATA_PATH = 'data/processed/buildings_features_earthquakes.csv'
TARGET_COLUMN = 'damage_grade'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER_SEARCH = 20 # Number of parameter settings that are sampled
CV_FOLDS = 3 # Number of cross-validation folds

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
X = df.drop(columns=[TARGET_COLUMN, 'building_id'])
y = df[TARGET_COLUMN] - 1 # 0-based target
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

# --- Feature Preprocessing (Same as train_lightgbm.py) ---
logging.info("Setting up preprocessing pipelines (Ordinal for categorical)...")
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

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
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
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
logging.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# --- Applying Preprocessing ---
logging.info("Applying preprocessing to training data...")
X_train_processed = preprocessor.fit_transform(X_train)
logging.info("Applying preprocessing to testing data...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names and indices after transformation
feature_names_out = numerical_features + categorical_features
categorical_feature_indices = [feature_names_out.index(col) for col in categorical_features]

logging.info(f"Processed training data shape: {X_train_processed.shape}")
logging.info("Preprocessing complete.")

# --- LightGBM Hyperparameter Tuning with RandomizedSearchCV ---
logging.info(f"Starting RandomizedSearchCV for LightGBM (n_iter={N_ITER_SEARCH}, cv={CV_FOLDS})...")
start_time = time.time()

# Define the parameter space
param_distributions = {
    'n_estimators': sp_randint(100, 1000),
    'learning_rate': sp_uniform(0.01, 0.2), # exploration range
    'num_leaves': sp_randint(20, 60),
    'max_depth': sp_randint(5, 15),
    'reg_alpha': sp_uniform(0, 1),
    'reg_lambda': sp_uniform(0, 1),
    'colsample_bytree': sp_uniform(0.6, 0.4), # Range from 0.6 to 1.0
    'subsample': sp_uniform(0.6, 0.4) # Range from 0.6 to 1.0
}

# Instantiate the base model
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss', # Can also use 'multi_error'
    random_state=RANDOM_STATE,
    n_jobs=1, # RandomizedSearchCV handles parallelization over folds/iters
    class_weight='balanced'
)

# Set up RandomizedSearchCV
# Scoring: Use accuracy or roc_auc_ovr_weighted. Accuracy is simpler.
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_distributions,
    n_iter=N_ITER_SEARCH,
    cv=CV_FOLDS,
    scoring='accuracy', # Or 'roc_auc_ovr_weighted'
    n_jobs=-1, # Use all available cores for search
    random_state=RANDOM_STATE,
    verbose=1 # Set verbosity level
)

# Define fit parameters to pass categorical features to LGBM inside the search
fit_params = {
    "categorical_feature": categorical_feature_indices
}

try:
    # Run the search
    random_search.fit(X_train_processed, y_train, **fit_params)

    tuning_duration = time.time() - start_time
    logging.info(f"RandomizedSearchCV finished in {tuning_duration:.2f} seconds.")

    # Get the best estimator
    best_lgbm = random_search.best_estimator_
    logging.info("Best parameters found by RandomizedSearchCV:")
    logging.info(random_search.best_params_)

    # --- Save Best Model and Preprocessor ---
    logging.info("Saving the best LightGBM model and the preprocessor...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the best model found by the search
    joblib.dump(best_lgbm, MODEL_PATH)
    logging.info(f"Best tuned model saved to {MODEL_PATH}")

    # Save the preprocessor (fitted on the original training data)
    # Check if PREPROCESSOR_PATH already exists from train_lightgbm.py run
    if not os.path.exists(PREPROCESSOR_PATH):
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logging.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    else:
         logging.info(f"Preprocessor already exists at {PREPROCESSOR_PATH}, not overwriting.")
    # --- End Save Section ---

except Exception as e:
    logging.error(f"Error during LightGBM tuning or saving: {e}")
    exit()

# --- Feature Importance Analysis (Best Model) ---
logging.info("Calculating feature importances for the best tuned model...")
try:
    importance_df = pd.DataFrame({
        'feature': feature_names_out,
        'importance': best_lgbm.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top_n = 20
    logging.info(f"--- Top {top_n} Feature Importances (Tuned LightGBM) ---")
    for index, row in importance_df.head(top_n).iterrows():
        logging.info(f"{row['feature']}: {row['importance']}")
    logging.info("--- End Feature Importances ---")
except Exception as e:
    logging.error(f"Error calculating feature importances: {e}")

# --- Model Evaluation (Best Model) ---
logging.info("Evaluating the best tuned LightGBM model on the test set...")

# Make predictions
y_pred = best_lgbm.predict(X_test_processed)
y_pred_proba = best_lgbm.predict_proba(X_test_processed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Grade 1', 'Grade 2', 'Grade 3'])
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

# Print evaluation results
logging.info("--- Evaluation Results (Tuned LightGBM) ---")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"ROC AUC (Macro OvR): {roc_auc:.4f}")
logging.info("\nConfusion Matrix (Labels: 0, 1, 2):")
print(conf_matrix)
logging.info("\nClassification Report:")
print(class_report)
logging.info("--- End Evaluation ---") 