import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig6_confusion_matrices.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model configurations
MODEL_CONFIGS = {
    "Tuned LightGBM": {
        "model_file": os.path.join(MODELS_DIR, "lightgbm_tuned_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "lightgbm_preprocessor.joblib")
    },
    "Logistic Regression": {
        "model_file": os.path.join(MODELS_DIR, "logistic_regression_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "baseline_preprocessor.joblib")
    },
    "LinearSVC": {
        "model_file": os.path.join(MODELS_DIR, "svm_model.joblib"), # Assuming this is LinearSVC
        "preprocessor_file": os.path.join(MODELS_DIR, "svm_preprocessor.joblib")
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
    
    # Assuming 'damage_grade' is the target and other columns are features from the processed file
    # The actual feature list used for training is embedded in the preprocessor
    if 'damage_grade' not in df.columns:
        print(f"Error: Target column 'damage_grade' not found in {file_path}")
        return None, None
        
    X = df.drop('damage_grade', axis=1)
    y = df['damage_grade']
    return X, y

def generate_plots():
    """Generates and saves confusion matrices for specified models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_full, y_full = load_data(PROCESSED_DATA_PATH)
    if X_full is None:
        return

    # Split data to get the same test set used for evaluation
    # Stratify to ensure class distribution is similar in train and test sets
    _, X_test, _, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )

    num_models = len(MODEL_CONFIGS)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1: # Ensure axes is always an array
        axes = [axes]

    for i, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[i]
        print(f"Processing model: {model_name}")

        try:
            model = joblib.load(config["model_file"])
            preprocessor = joblib.load(config["preprocessor_file"])
        except FileNotFoundError as e:
            print(f"Error loading model or preprocessor for {model_name}: {e}")
            ax.text(0.5, 0.5, f'Error loading files for\n{model_name}', ha='center', va='center')
            ax.set_title(model_name, fontsize=14)
            continue
        
        # Preprocess the test data
        # Ensure X_test has the features expected by the preprocessor
        # The preprocessor itself knows which columns to transform
        X_test_processed = preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_) # Use model.classes_ for correct label order
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=DAMAGE_GRADE_LABELS, yticklabels=DAMAGE_GRADE_LABELS,
                    annot_kws={"size": 10})
        ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=14)
        ax.set_xlabel('Predicted Damage Grade', fontsize=12)
        if i == 0:
            ax.set_ylabel('Actual Damage Grade', fontsize=12)
        else:
            ax.set_ylabel('') # Avoid repetition
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Confusion Matrices for Top Performing Models', fontsize=18, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plots() 