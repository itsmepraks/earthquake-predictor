# Earthquake Risk Forecasting for Nepal  
**A Preliminary White Paper**

## 1. Introduction  
Nepal’s location along the Himalayan tectonic belt makes it one of the world’s most earthquake‑prone regions. Over 80% of buildings in Kathmandu are rated structurally vulnerable. The 2015 Gorkha earthquake demonstrated the catastrophic human and economic toll when prediction and preparedness lag behind geophysical reality.

## 2. Objectives  
• Develop a machine‑learning framework to improve short‑term earthquake risk classification in Nepal.  
• Target “High Risk” vs. “Low Risk” zones with better precision than current 40–60% accuracy.  
• Provide a lightweight prototype suitable for low‑resource early‑warning deployments.

## 3. Data Sources  
1. **USGS Seismic Database** – Century‑scale records (magnitude, depth, lat/lon).  
2. **2015 Gorkha Dataset** – Detailed mainshock and aftershock catalog.  
3. **Supplementary** – NASA elevation DEMs, Kaggle soil/terrain layers.

## 4. Methodology  
1. **Data Ingestion & Preprocessing**  
   - Fetch via USGS API  
   - Filter by region (Nepal bounding box)  
   - Feature engineering: magnitude, depth, slope, elevation, distance to known faults  
2. **Baseline Model**  
   - Logistic Regression for binary “High/Low” risk  
   - Train/test split; GridSearchCV for hyperparameter tuning  
3. **Extended Models (next steps)**  
   - Random Forest for feature importance  
   - SVM for non‑linear decision boundaries

## 5. Prototype Implementation  
See `prototype/ml_prototype.py` for a runnable demo:
```python
# snippet here shows data fetch → basic train/evaluate
```

## 6. Preliminary Results & Discussion  
*(To be populated once prototype runs on real data.)*

## 7. Roadmap & Next Steps  
- Integrate elevation and soil features  
- Evaluate Random Forest & SVM benchmarks  
- Build a Streamlit dashboard for risk‐map visualization  
- Optimize for edge/cloud deployment  

## 8. References  
- USGS Earthquake Catalog API  
- 2015 Gorkha seismic analysis papers  
- Python ML ecosystem docs  