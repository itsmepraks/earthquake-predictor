import os
from itertools import cycle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig8_pr_curves.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLASSES = 3 # Damage grades 1, 2, 3

# Model configurations
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
    """Generates and saves Precision-Recall curves for specified models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_full, y_full = load_data(PROCESSED_DATA_PATH)
    if X_full is None:
        return

    y_binarized = label_binarize(y_full, classes=sorted(y_full.unique()))
    if y_binarized.shape[1] == 1: 
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
        print(f"Processing PR for model: {model_name}")

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
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        for j, color in zip(range(N_CLASSES), colors):
            precision[j], recall[j], _ = precision_recall_curve(y_test_binarized[:, j], y_score[:, j])
            average_precision[j] = average_precision_score(y_test_binarized[:, j], y_score[:, j])
            ax.plot(recall[j], precision[j], color=color, lw=lw,
                    label=f'PR {DAMAGE_GRADE_LABELS[j]} (AP = {average_precision[j]:.2f})')

        # Compute micro-average Precision-Recall curve and AP
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_binarized.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(y_test_binarized, y_score, average="micro")
        ax.plot(recall["micro"], precision["micro"],
                label=f'Micro-average PR (AP = {average_precision["micro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{model_name}\nPrecision-Recall Curves (OvR)', fontsize=14)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.5)

    plt.suptitle('Precision-Recall Curves for Top Models (One-vs-Rest)', fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plots() 