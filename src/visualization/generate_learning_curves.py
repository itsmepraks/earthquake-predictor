import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

# Define file paths
PROCESSED_DATA_PATH = "data/processed/buildings_features_earthquakes.csv"
MODEL_PATH = "models/lightgbm_tuned_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"
OUTPUT_DIR = "reports/images"
OUTPUT_FILENAME = "fig11_learning_curves_lgbm.png"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

RANDOM_STATE = 42
N_SPLITS_CV = 3 # Number of folds for cross-validation in learning curve

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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features),
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : Axes object, default=None
        Axes to use for plotting the curves.
    ylim : tuple, shape (ymin, ymax), default=None
        Defines minimum and maximum y-values plotted.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title, fontsize=16)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples", fontsize=14)
    axes.set_ylabel("Score (Accuracy)", fontsize=14)

    train_sizes_abs, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True, random_state=RANDOM_STATE, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid(alpha=0.5)
    axes.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes_abs, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best", fontsize=12)
    return plt

def generate_main_plot():
    """Generates and saves learning curves for the Tuned LightGBM model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data(PROCESSED_DATA_PATH)
    if X is None or y is None:
        print("Failed to load data. Aborting learning curve generation.")
        return

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Tuned LightGBM model and preprocessor loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model or preprocessor: {e}")
        # Optionally create an error placeholder plot here if needed
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # Create a pipeline with the preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"Generating learning curves using {N_SPLITS_CV}-fold stratified CV. This may take a while...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_learning_curve(pipeline, "Learning Curves (Tuned LightGBM)", X, y, axes=ax, ylim=(0.6, 1.01),
                        cv=cv_strategy, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_FILE_PATH)
        print(f"Plot saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    generate_main_plot() 