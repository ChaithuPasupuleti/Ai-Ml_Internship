# AI-ML Internship Project - Week 1: Foundation

## 1. Project Overview
An end-to-end machine learning pipeline workspace built for model tracking, data manipulation, and experimentation using Python, Pandas, Scikit-Learn, and MLflow.

## 2. Literature & Repository Reviews (3 References)
* **Reference 1: Scikit-Learn Documentation & Best Practices**
  * *Summary:* Focuses on standardized data preprocessing pipelines, feature scaling, and consistent model evaluation interfaces.
* **Reference 2: MLflow Tracking Server Architecture**
  * *Summary:* Explores centralized experiment logging, metric tracking, and artifact storage for reproducible machine learning workflows.
* **Reference 3: NumPy Vectorized Operations in Data Pipelines**
  * *Summary:* Highlights high-performance array manipulation and memory efficiency essential for large-scale feature engineering.

## 3. Model Architecture & Evaluation Metrics
* **Model Architecture:** Baseline classification/regression pipeline using Scikit-Learn estimators, structured with modular preprocessing and inference transformers.
* **Evaluation Metrics:** 
  * Classification: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  * Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.

## 4. Data Pipeline Skeleton
The project directory structure is organized as follows:
```text
ai-ml-internship/
│
├── data/               # Raw and processed data storage
├── notebooks/          # Exploratory data analysis notebooks
├── src/                # Source code for data ingestion, preprocessing, and training
│   ├── __init__.py
│   ├── preprocess.py
│   └── train.py
├── numpy_basics.py     # NumPy foundational notes and exercises
└── README.md           # Project documentationgit add README.md