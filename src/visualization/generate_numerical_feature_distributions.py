import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig2_numerical_feature_distributions.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Features to plot
NUMERICAL_FEATURES = [
    'age',
    'count_floors_pre_eq',
    'area_percentage',
    'height_percentage'
]

DESCRIPTIVE_TITLES = {
    'age': 'Age of Building (Years)',
    'count_floors_pre_eq': 'Number of Floors (Pre-Earthquake)',
    'area_percentage': 'Normalized Area of Building Footprint',
    'height_percentage': 'Normalized Height of Building'
}

def generate_plots():
    """
    Generates and saves histograms for key numerical features.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    # Check if all features are in the dataframe
    missing_features = [col for col in NUMERICAL_FEATURES if col not in df.columns]
    if missing_features:
        print(f"Error: The following features are not in the dataframe: {missing_features}")
        return

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten() # Flatten the 2x2 array for easy iteration

    for i, feature in enumerate(NUMERICAL_FEATURES):
        sns.histplot(df[feature], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(DESCRIPTIVE_TITLES.get(feature, feature), fontsize=14)
        axes[i].set_xlabel('Value', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
    
    plt.suptitle('Distributions of Key Numerical Features', fontsize=18, y=1.02)
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plots() 