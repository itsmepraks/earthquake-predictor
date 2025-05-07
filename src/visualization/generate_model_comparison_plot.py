import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define file paths
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig5_model_comparison_metrics.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Hardcoded model performance data (from whitepaper.md, Section 4.1)
# Note: LinearSVC ROC AUC is N/A, represented as 0 for plotting and will be annotated.
data = {
    'Model': [
        'Logistic Regression', 'LightGBM (Untuned)', 'Random Forest', 
        'LinearSVC', 'LightGBM (Tuned)'
    ],
    'Accuracy': [0.697, 0.711, 0.717, 0.721, 0.725],
    'ROC AUC (Macro OvR)': [0.872, 0.880, 0.845, 0.0, 0.880], # LinearSVC N/A -> 0
    'Weighted F1-score': [0.69, 0.71, 0.71, 0.72, 0.73]
}

def generate_plot():
    """
    Generates and saves a grouped bar chart for model performance comparison.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(data)
    
    # Melt the DataFrame to long format for seaborn's barplot
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    
    plt.title('Comprehensive Model Performance Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.0) # Scores are typically between 0 and 1
    plt.legend(title='Metric', fontsize=10, title_fontsize=12)
    
    # Add text annotations for each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points', fontsize=8)

    # Add a note for LinearSVC ROC AUC
    plt.text(0.99, 0.01, '*LinearSVC ROC AUC is N/A and represented as 0.',
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=9, style='italic')

    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_plot() 