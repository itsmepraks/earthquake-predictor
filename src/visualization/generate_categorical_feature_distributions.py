import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig3_categorical_feature_distributions.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Features to plot
CATEGORICAL_FEATURES = [
    'foundation_type',
    'roof_type',
    'ground_floor_type',
    'land_surface_condition',
    'geo_level_1_id' # Will handle top N for this separately
]

DESCRIPTIVE_TITLES = {
    'foundation_type': 'Distribution of Foundation Types',
    'roof_type': 'Distribution of Roof Types',
    'ground_floor_type': 'Distribution of Ground Floor Types',
    'land_surface_condition': 'Distribution of Land Surface Conditions',
    'geo_level_1_id': 'Distribution of Top 10 Geo Level 1 IDs'
}

TOP_N_GEO = 10

def generate_plots():
    """
    Generates and saves bar charts for key categorical features.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    # Check if all features are in the dataframe (except geo_level_1_id handled separately)
    base_features = [f for f in CATEGORICAL_FEATURES if f != 'geo_level_1_id']
    missing_features = [col for col in base_features if col not in df.columns]
    if 'geo_level_1_id' not in df.columns:
        missing_features.append('geo_level_1_id')
    
    if missing_features:
        print(f"Error: The following features are not in the dataframe: {missing_features}")
        return

    # Create the plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18)) # Adjusted for 5 plots, last one empty
    axes = axes.flatten()

    for i, feature in enumerate(CATEGORICAL_FEATURES):
        ax = axes[i]
        if feature == 'geo_level_1_id':
            top_n_values = df[feature].value_counts().nlargest(TOP_N_GEO)
            sns.barplot(x=top_n_values.index, y=top_n_values.values, ax=ax, order=top_n_values.index, palette="viridis")
            ax.set_xlabel(f'Top {TOP_N_GEO} Categories', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.countplot(y=df[feature], ax=ax, order = df[feature].value_counts().index, palette="viridis", hue=df[feature], legend=False)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_ylabel('Category', fontsize=12)
        
        ax.set_title(DESCRIPTIVE_TITLES.get(feature, feature), fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Hide the last empty subplot if the number of features is odd
    if len(CATEGORICAL_FEATURES) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Distributions of Key Categorical Features', fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plots() 