# Graph Generation Plan for Whitepaper

This document outlines the plan for generating graphs and visualizations to be included in `whitepaper.md`.

## Overall Goal
To enhance the whitepaper with visual representations of data distributions, model performance comparisons, feature importances, and application features, making the report more engaging and findings clearer.

## Proposed Graphs & Visualizations

### Section 2: Data Sources and Preparation
- [x] **Target Variable Distribution:**
    - Type: Bar chart
    - Data: `damage_grade` counts or percentages (Grade 1, 2, 3)
    - Purpose: Illustrate class imbalance.
- [x] **Key Numerical Feature Distributions (Examples):**
    - Type: Histograms or density plots
    - Data: `age`, `count_floors_pre_eq`, `area_percentage`, `height_percentage`
    - Purpose: Show underlying distributions of important continuous features.
- [x] **Key Categorical Feature Distributions (Examples):**
    - Type: Bar charts
    - Data: `foundation_type`, `roof_type`, `ground_floor_type`, `land_surface_condition`, top N `geo_level_1_id`s
    - Purpose: Show frequency of important discrete features.
- [x] **(Optional) Missing Value Visualization:**
    - Type: Bar chart or heatmap
    - Data: Percentage of missing values per feature (from raw building data, if notable)
    - Purpose: Visually represent data completeness.
    - Path: `reports/images/fig4_missing_value_distribution.png`

### Section 4: Results

#### Section 4.1: Model Performance Comparison
- [x] **Comprehensive Model Metrics Comparison:**
    - Type: Grouped bar chart
    - Data: Accuracy, ROC AUC (Macro OvR), Weighted F1-score across all models (LogReg, LGBM-Untuned, RF, LinearSVC, LGBM-Tuned).
    - Purpose: Strong visual for overall model performance ranking.
    - Path: `reports/images/fig5_model_comparison_metrics.png`
- [x] **Confusion Matrices (Top Models):**
    - Type: Heatmaps
    - Data: Confusion matrices for Tuned LightGBM, and potentially Logistic Regression and/or LinearSVC.
    - Purpose: Illustrate error patterns and per-class performance.
    - Path: `reports/images/fig6_confusion_matrices.png`
- [x] **(Optional) ROC Curves (Top Models):**
    - Type: Line plots (OvR ROC curves for each class)
    - Data: For top 2-3 models on a single plot.
    - Purpose: Detailed comparison of class discrimination ability.
    - Path: `reports/images/fig7_roc_curves.png`
- [x] **(Optional) Precision-Recall Curves (Top Models):**
    - Type: Line plots
    - Data: For top 2-3 models, especially useful for imbalanced classes.
    - Purpose: Show trade-off between precision and recall.
    - Path: `reports/images/fig8_pr_curves.png`

#### Section 4.3: Feature Importance Insights
- [x] **Feature Importances for Final Model (Tuned LightGBM):**
    - Type: Horizontal bar chart
    - Data: Top N (e.g., 15-20) feature importances.
    - Purpose: Clearly show key drivers for the best model.
    - Path: `reports/images/fig9_feature_importance_lgbm.png`
- [x] **(Optional) Comparative Feature Importances (2-3 Models):**
    - Type: Small multiples of horizontal bar charts
    - Data: E.g., Tuned LGBM vs. Logistic Regression.
    - Purpose: Visually contrast feature prioritization by different model types.
    - Path: `reports/images/fig10_comparative_feature_importance.png`

### Section 5: Streamlit Application
- [ ] **Screenshot: Main Input Sidebar:**
    - Type: Image (Screenshot) - Placeholder text added in whitepaper.md
    - Purpose: Showcase application's user input interface.
- [ ] **Screenshot: Example Prediction Output:**
    - Type: Image (Screenshot) - Placeholder text added in whitepaper.md
    - Purpose: Illustrate how predictions are presented to the user.
- [ ] **Screenshot: Risk Distribution Chart (from App):**
    - Type: Image (Screenshot) - Placeholder text added in whitepaper.md
    - Purpose: Show an example of in-app analytics.
- [ ] **Screenshot: Feature Importance Chart (from App):**
    - Type: Image (Screenshot) - Placeholder text added in whitepaper.md
    - Purpose: Show an example of in-app model explainability.

### Appendix / Advanced Discussion (Optional)
- [x] **(Optional) Learning Curves (Final Model):**
    - Type: Line plot
    - Data: Training and validation scores vs. number of training samples for Tuned LightGBM.
    - Purpose: Diagnose bias/variance and potential for improvement with more data.
- [ ] **(Optional) Partial Dependence Plots (PDPs) / Individual Conditional Expectation (ICE) Plots:**
    - Type: Line plots
    - Data: For top few features of the Tuned LightGBM model.
    - Purpose: Illustrate marginal effect of specific features on predictions.

## Image Storage
- All generated images will be stored in a new directory: `reports/images/`
- Images should be named descriptively (e.g., `fig1_target_distribution.png`, `fig2_model_comparison_metrics.png`).

## Tools for Generation
- Python libraries: `Matplotlib`, `Seaborn`, `Altair`.
- Screenshots: Standard OS screenshot tools. 