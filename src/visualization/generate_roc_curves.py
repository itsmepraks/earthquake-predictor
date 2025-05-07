import os
from itertools import cycle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig7_roc_curves.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLASSES = 3 # Damage grades 1, 2, 3

# Model configurations (only those that support predict_proba)
MODEL_CONFIGS = {
    "Tuned LightGBM": {
        "model_file": os.path.join(MODELS_DIR, "lightgbm_tuned_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "lightgbm_preprocessor.joblib")
    },
    "Logistic Regression": {
        "model_file": os.path.join(MODELS_DIR, "logistic_regression_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "baseline_preprocessor.joblib")
    }
}
DAMAGE_GRADE_LABELS = ['Grade 1 (Low)', 'Grade 2 (Medium)', 'Grade 3 (High)']

def load_data(file_path):
    """Loads data and splits it into features and target."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None
    if 'damage_grade' not in df.columns:
        print(f"Error: Target column 'damage_grade' not found in {file_path}")
        return None, None
    X = df.drop('damage_grade', axis=1)
    y = df['damage_grade']
    return X, y

def generate_plots():
    """Generates and saves ROC curves for specified models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_full, y_full = load_data(PROCESSED_DATA_PATH)
    if X_full is None:
        return

    # Binarize the output for OvR ROC curve calculation. Classes are 1, 2, 3.
    # We need to map them to 0, 1, 2 for label_binarize if they are not already.
    # Assuming y_full contains original labels (e.g., 1, 2, 3). If they are 0-indexed, this is fine.
    # Let's check unique values and adjust if necessary.
    # The models are trained on 1,2,3, so label_binarize will handle it if classes=[1,2,3] is passed.
    y_binarized = label_binarize(y_full, classes=sorted(y_full.unique()))
    if y_binarized.shape[1] == 1: # Handles binary case, ensure it's multi-class
        # This shouldn't happen for N_CLASSES=3, but as a safeguard:
        y_binarized = label_binarize(y_full, classes=sorted(y_full.unique()) if len(y_full.unique()) > 2 else [1,2,3])

    _, X_test, _, y_test_binarized = train_test_split(
        X_full, y_binarized, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )

    num_models = len(MODEL_CONFIGS)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6))
    if num_models == 1: 
        axes = [axes]

    lw = 2 # line width

    for i, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[i]
        print(f"Processing ROC for model: {model_name}")

        try:
            model = joblib.load(config["model_file"])
            preprocessor = joblib.load(config["preprocessor_file"])
        except FileNotFoundError as e:
            print(f"Error loading model or preprocessor for {model_name}: {e}")
            ax.text(0.5, 0.5, f'Error loading files for\n{model_name}', ha='center', va='center')
            ax.set_title(model_name, fontsize=14)
            continue
        
        X_test_processed = preprocessor.transform(X_test)
        y_score = model.predict_proba(X_test_processed)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        for j, color in zip(range(N_CLASSES), colors):
            fpr[j], tpr[j], _ = roc_curve(y_test_binarized[:, j], y_score[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])
            ax.plot(fpr[j], tpr[j], color=color, lw=lw,
                    label=f'ROC {DAMAGE_GRADE_LABELS[j]} (AUC = {roc_auc[j]:.2f})')

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[k] for k in range(N_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(N_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
        mean_tpr /= N_CLASSES
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        ax.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name}\nROC Curves (OvR)', fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.5)

    plt.suptitle('ROC Curves for Top Models (One-vs-Rest)', fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plots() 