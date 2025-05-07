import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define file paths
MODEL_PATH = "models/lightgbm_tuned_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig9_feature_importance_lgbm.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Number of top features to display
TOP_N_FEATURES = 20

def generate_plot():
    """Generates and saves a feature importance plot for the Tuned LightGBM model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Model and preprocessor loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model or preprocessor: {e}")
        # Create a placeholder plot indicating error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error loading model/preprocessor:\n{e}', 
                horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
        try:
            plt.savefig(OUTPUT_FILE_PATH)
            print(f"Error placeholder plot saved to {OUTPUT_FILE_PATH}")
        except Exception as save_e:
            print(f"Error saving placeholder plot: {save_e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Error: Model does not have feature_importances_ attribute.")
        return

    # Get feature names from the preprocessor
    # The preprocessor is likely a ColumnTransformer
    try:
        # If the preprocessor is a Pipeline, get the ColumnTransformer step
        if hasattr(preprocessor, 'named_steps'):
            # Assuming the ColumnTransformer is named 'preprocessor' or is the first step if unnamed
            ct_step_name = next((name for name, step in preprocessor.named_steps.items() if hasattr(step, 'get_feature_names_out')), None)
            if not ct_step_name and preprocessor.steps:
                 # Fallback: try to find it by checking type (might be risky if multiple CTs)
                 ct_step_name = next((name for name, step in preprocessor.steps if hasattr(step, 'get_feature_names_out')), None)
            
            if ct_step_name:
                 column_transformer = preprocessor.named_steps[ct_step_name]
            elif hasattr(preprocessor, 'get_feature_names_out'): # Preprocessor IS the CT
                 column_transformer = preprocessor
            else:
                print("Could not find ColumnTransformer in the preprocessor pipeline.")
                # Attempt to get feature names directly if preprocessor itself has the method (e.g. it IS a ColumnTransformer)
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                     print("Preprocessor does not have get_feature_names_out and ColumnTransformer not found.")
                     # Fallback: assume number of features matches importance length
                     feature_names = [f"feature_{i}" for i in range(len(importances))]
        elif hasattr(preprocessor, 'get_feature_names_out'): # Preprocessor is the ColumnTransformer itself
            column_transformer = preprocessor
            feature_names = column_transformer.get_feature_names_out()
        else:
            print("Preprocessor structure not recognized or does not support get_feature_names_out.")
            # Fallback: assume number of features matches importance length
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        if 'column_transformer' in locals() and hasattr(column_transformer, 'get_feature_names_out'):
            feature_names = column_transformer.get_feature_names_out()
        elif not 'feature_names' in locals(): # if still not set
            print("Critical: Could not retrieve feature names. Using generic names.")
            feature_names = [f"feature_{i}" for i in range(len(importances))]

    except Exception as e:
        print(f"Error getting feature names from preprocessor: {e}")
        # Fallback: assume number of features matches importance length
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    if len(importances) != len(feature_names):
        print(f"Warning: Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}). Using generic names.")
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Select top N features
    top_features_df = feature_importance_df.head(TOP_N_FEATURES)

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
    plt.title(f'Top {TOP_N_FEATURES} Feature Importances (Tuned LightGBM)', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plot() 