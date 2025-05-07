import os
import re  # For OHE feature name parsing

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define file paths
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig10_comparative_feature_importance.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
MODELS_DIR = "models"

TOP_N_FEATURES = 15 # Show top 15 for comparison plots

MODEL_CONFIGS = {
    "Tuned LightGBM": {
        "model_file": os.path.join(MODELS_DIR, "lightgbm_tuned_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "lightgbm_preprocessor.joblib"),
        "type": "tree"
    },
    "Logistic Regression": {
        "model_file": os.path.join(MODELS_DIR, "logistic_regression_model.joblib"),
        "preprocessor_file": os.path.join(MODELS_DIR, "baseline_preprocessor.joblib"),
        "type": "linear"
    }
}

def get_feature_names_from_preprocessor(preprocessor, importances_len):
    """Helper to get feature names from a preprocessor, robustly."""
    try:
        # If the preprocessor is a Pipeline, get the ColumnTransformer step
        if hasattr(preprocessor, 'named_steps'):
            ct_step = None
            # Try to find by common names or by type if it has get_feature_names_out
            for name, step in preprocessor.named_steps.items():
                if hasattr(step, 'get_feature_names_out'):
                    ct_step = step
                    break
            if not ct_step and preprocessor.steps: # Last resort for unnamed step
                 ct_step = next((step for _, step in preprocessor.steps if hasattr(step, 'get_feature_names_out')), None)
            
            if ct_step and hasattr(ct_step, 'get_feature_names_out'):
                 return ct_step.get_feature_names_out()
            elif hasattr(preprocessor, 'get_feature_names_out'): # Preprocessor IS the CT or has the method
                 return preprocessor.get_feature_names_out()
            else:
                print("Could not find ColumnTransformer with get_feature_names_out in pipeline.")
        elif hasattr(preprocessor, 'get_feature_names_out'): # Preprocessor is the ColumnTransformer itself
            return preprocessor.get_feature_names_out()
        else:
            print("Preprocessor structure not recognized or does not support get_feature_names_out.")
    except Exception as e:
        print(f"Error getting feature names: {e}")
    print(f"Fallback: Using generic feature names for preprocessor: {type(preprocessor)}")
    return [f"feature_{i}" for i in range(importances_len)]

def aggregate_ohe_importances(feature_names, importances):
    """Aggregates importances for OHE features."""
    aggregated = {}
    for name, imp in zip(feature_names, importances):
        # OHE features from ColumnTransformer often look like 'transformername__featurename_category'
        # e.g., 'onehotencoder__foundation_type_r' or 'remainder__age'
        # We want to group by 'featurename' part if it's OHE, or take full name if not (e.g. scaled numerical)
        match = re.match(r"^[a-zA-Z0-9]+__([a-zA-Z0-9_]+?)(?:_[a-zA-Z0-9_]+)?$", name)
        if match:
            original_feature_name = match.group(1)
            # Check if the part after __ is one of the original feature names (more robust)
            # This part requires knowing original feature names, which is tricky here.
            # Simpler approach: if it contains '_', assume it could be OHE and group by base before last '_'.
            # This is a heuristic.
            parts = name.split('__')
            if len(parts) > 1:
                base_name_candidate = parts[1]
                if '_' in base_name_candidate and not any(num_feat_part in base_name_candidate for num_feat_part in ['age', 'count', 'area', 'height']): # Avoid splitting numericals
                    original_feature_name = base_name_candidate.rsplit('_', 1)[0]
                else:
                    original_feature_name = base_name_candidate # Likely numerical or already simple cat
            else:
                 original_feature_name = name # No prefix, take as is
        else:
            original_feature_name = name # No prefix, take as is

        aggregated[original_feature_name] = aggregated.get(original_feature_name, 0) + imp
    return pd.Series(aggregated)

def generate_plot():
    """Generates comparative feature importance plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_models = len(MODEL_CONFIGS)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8))
    if num_models == 1: axes = [axes]

    for i, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[i]
        print(f"Processing feature importances for: {model_name}")
        try:
            model = joblib.load(config["model_file"])
            preprocessor = joblib.load(config["preprocessor_file"])
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            ax.text(0.5,0.5, f'Error loading {model_name}', ha='center', va='center')
            ax.set_title(model_name, fontsize=14)
            continue

        importances = None
        if config["type"] == "tree":
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
        elif config["type"] == "linear":
            if hasattr(model, 'coef_'):
                # For multi-class, average absolute coefficients across classes
                if model.coef_.ndim > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_)
        
        if importances is None:
            print(f"Could not get importances for {model_name}")
            ax.text(0.5,0.5, f'No importances for {model_name}', ha='center', va='center')
            ax.set_title(model_name, fontsize=14)
            continue

        raw_feature_names = get_feature_names_from_preprocessor(preprocessor, len(importances))

        if config["type"] == "linear" and any("__" in name for name in raw_feature_names):
            print(f"Aggregating OHE features for {model_name}")
            # The baseline_preprocessor for LogReg has OneHotEncoder
            # Need to ensure the raw_feature_names are the *output* of the CT
            # The get_feature_names_from_preprocessor should ideally return this.
            aggregated_importances_series = aggregate_ohe_importances(raw_feature_names, importances)
            feature_importance_df = aggregated_importances_series.reset_index()
            feature_importance_df.columns = ['feature', 'importance']
        else:
            feature_importance_df = pd.DataFrame({'feature': raw_feature_names, 'importance': importances})
        
        top_features_df = feature_importance_df.sort_values(by='importance', ascending=False).head(TOP_N_FEATURES)

        sns.barplot(x='importance', y='feature', data=top_features_df, ax=ax, palette='viridis')
        ax.set_title(f'{model_name}\nTop {TOP_N_FEATURES} Features', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle('Comparative Feature Importances', fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plot() 