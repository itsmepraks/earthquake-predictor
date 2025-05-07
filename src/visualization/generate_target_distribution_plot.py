import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig1_target_distribution.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def generate_plot():
    """
    Generates and saves a bar chart of the target variable (damage_grade) distribution.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the processed data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    # Calculate the distribution of damage_grade
    target_distribution = df['damage_grade'].value_counts().sort_index()

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_distribution.index, y=target_distribution.values, palette="viridis")
    plt.title('Distribution of Damage Grade (Target Variable)', fontsize=16)
    plt.xlabel('Damage Grade', fontsize=14)
    plt.ylabel('Number of Buildings', fontsize=14)
    plt.xticks(ticks=[0, 1, 2], labels=['Grade 1 (Low)', 'Grade 2 (Medium)', 'Grade 3 (High)'])
    
    # Add text annotations for counts
    for i, count in enumerate(target_distribution.values):
        plt.text(i, count + (0.01 * target_distribution.values.max()), str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plot() 