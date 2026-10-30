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

# Week 1 Day 2: Pandas Data Manipulation & Export Analysis

## 📁 Project Folder Structure
* `venv/` - Virtual environment folder for managing Python dependencies and libraries.
* `pandas_manipulation.py` - Core script implementing data loading, filtering, groupby aggregations, merging, and pivot tables.
* `cleaned_dataset.csv` - Exported processed dataset in CSV format.
* `cleaned_dataset.parquet` - Space-efficient compressed output dataset in Parquet format.

## 🛠️ Key Python Modules & Files
* **`pandas_manipulation.py`**: 
  - Loads a real-world dataset into a Pandas DataFrame and inspects its shape, data types, and first 10 rows.
  - Performs advanced data filtering (e.g., survival and class conditions).
  - Computes group-based metrics using `groupby()`, data combinations using `merge()`, and summary tables using `pivot_table()`.
  - Exports data into both CSV and Parquet formats, validating that Parquet is significantly more space-efficient (approximately 1.6x smaller).

## 📊 Data Flow Diagram