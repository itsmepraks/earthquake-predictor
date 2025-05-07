import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define file paths
RAW_DATA_PATH = "data/nepal_land_data/train_values.csv" # Using raw data to show initial missing values
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig4_missing_value_distribution.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def generate_plot():
    """
    Generates and saves a bar chart of missing value percentages for features in the raw dataset.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    # Calculate percentage of missing values for each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Filter out columns with no missing values
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

    if missing_percentage.empty:
        print("No missing values found in the dataset.")
        # Optionally, create a blank plot or a plot with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No missing values found in the dataset.', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=15, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Missing Value Analysis', fontsize=16)
    else:
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=missing_percentage.index, y=missing_percentage.values, palette="viridis")
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values per Feature (Raw Data)', fontsize=16)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Percentage Missing (%)', fontsize=14)
        plt.ylim(0, 100) # Ensure y-axis goes to 100 if there are high percentages

        # Add text annotations for percentages
        for i, percentage in enumerate(missing_percentage.values):
            plt.text(i, percentage + 1, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plot() 