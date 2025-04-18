## Proposal:
Nepal faces extreme seismic risks due to its position along the Himalayan tectonic belt, with over 80% of buildings in Kathmandu vulnerable to collapse. The devastating 2015 Gorkha earthquake highlighted the urgent need for improved seismic risk prediction to save lives and reduce damage. Current earthquake prediction systems achieve only 40-60% accuracy in Nepal’s complex geology, leaving communities unprepared. This project will develop a machine learning (ML) framework to improve short-term earthquake risk forecasting while optimizing low-resource early warning systems for Nepali infrastructure. 

## Data Sources Intended to Use

USGS Seismic Database: Contains historical himalayan earthquake records, including magnitude, depth, latitude, and longitude, spanning over a century. 
2015 Gorkha Earthquake Dataset: Provides detailed information on the mainshock and aftershocks recorded across Nepal.
Supplementary Data: Terrain elevation data from NASA Earthdata and data from Kaggle.

## Machine Learning Algorithms and Techniques Intended to Use

Logistic Regression: To classify areas as "High Risk" or "Low Risk" based on seismic patterns.
Random Forest: To identify key factors influencing earthquake risk, such as depth and soil type.
Support Vector Machine (SVM): To improve classification accuracy for spatial seismic data.
Parameter Tuning: Techniques like GridSearchCV will be used to optimize hyperparameters (e.g., regularization strength for logistic regression or tree depth for random forests) to enhance model performance.

### Intended Implementation Tools
Python Stack: PyTorch, Scikit-learn, TensorFlow and Streamlit / Preswald 

### Expected Outcomes
A machine learning model capable of predicting earthquake risks with improved accuracy over traditional methods.
Visualizations of high-risk zones in Nepal to assist policymakers and disaster management teams in preparedness efforts.
Feature importance analysis (e.g., sediment depth vs. magnitude)
