## 🚀 Day 3: Data Loading, Cleaning & Inspection

### **Overview**
Day 3 focuses on robust data ingestion, schema validation, missing value sanitization, and automated unit testing adhering to MLOps best practices (PEP-8 formatting, linting, and safe path guards).

---

### **📁 Core Modules & Files**
* `data_loader.py` - Core script responsible for robust data ingestion, fallback paths, and schema validation.
* `data_cleaning.py` - Handles missing value treatment, null dropping, and dataset sanitization with empty-DataFrame guards.
* `tests/test_data_loader.py` - Unit test suite utilizing `pytest` and `tmp_path` fixtures to validate data loader and cleaner functions.
* `LICENSE` - Open-source MIT License for project distribution.

---

### **⚡ Quick Start & Execution Guide**

1. **Install Dependencies:**
   Ensure your virtual environment is active and required packages are installed:
   ```bash
   pip install -r requirements.txt

   ## 📊 Day 4: Exploratory Data Analysis (EDA)

### Files
- `eda.py` - runs describe(), info(), null checks and saves all plots
- `observations.md` - 5 observations from the summary stats
- `eda_narrative.md` - 200-word EDA narrative
- `outputs/` - distributions, correlation heatmap, top-10 category charts

### Run

### Day 5: Data Visualization (Matplotlib & Seaborn)
* **Objective:** Complete hands-on tasks for exploratory data visualization using Matplotlib and Seaborn.
* **Key Tasks Implemented:**
  * Generated distribution plots to analyze data spread and skewness (e.g., Age, Fare).
  * Created categorical comparison and count plots.
  * Developed correlation heatmaps to identify feature relationships.
  * Authored exploratory data analysis (EDA) narratives and observations.
* **Outputs:** Visualizations and plots saved to the project directory.
*


# Week 2 Day 1: Feature Engineering Pipeline (Scaling, Encoding, & Selection)
* **Objective:** Build an end-to-end, leak-free preprocessing and feature selection pipeline using `scikit-learn`.
* **Key Implementations:**
  * **Missing Value Imputation:** Handled numeric columns using median imputation and categorical columns using most-frequent imputation.
  * **Feature Scaling:** Standardized numeric attributes using `StandardScaler`.
  * **Categorical Encoding:** Applied `OneHotEncoder(drop='first')` to prevent the dummy-variable trap.
  * **Feature Selection:** Integrated `SelectKBest(k=5)` directly into a unified `Pipeline` to ensure all transformations fit strictly on the training split, avoiding data leakage.
* **Visualizations & Inspection:**
  * Generated side-by-side histograms comparing raw vs. standard-scaled numeric distributions.
  * Extracted and displayed feature importance scores for the top-selected features.
* **Pull Request:** Merged via PR #5 (`feat: feature engineering – encoding, scaling & top-5 selection`).
*

### Week 2 Day 2: Feature Scaling, SelectKBest, and RFE Tasks

## Overview
Day 2 focuses on advanced feature engineering techniques, examining how different scalers handle skewed distributions and outliers, applying statistical feature selection (`SelectKBest`), and implementing model-driven feature elimination (`RFE`).

## Core Modules & Tasks

* **Task 1: Scalers & Skewed Data**
  * Evaluated and compared `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `Normalizer`.
  * Handled simulated skewed distributions and outlier-heavy columns to understand scaling trade-offs.

* **Task 2: SelectKBest Feature Selection**
  * Implemented `SelectKBest` using ANOVA F-value (`f_classif`) for standard classification features.
  * Implemented Chi-Square (`chi2`) feature selection on non-negative scaled data using `MinMaxScaler`.

* **Task 3: Recursive Feature Elimination (RFE)**
  * Configured `RFE` utilizing a `LogisticRegression` estimator to recursively rank and eliminate low-importance features.
  * Logged feature support, rankings, and automated elimination summaries.

## Quick Start & Execution Guide

1. **Install Dependencies:** Ensure your environment has the required libraries installed:
   ```bash
   pip install -r requirements.txt 

   ## Week 2 - Day 3: Imbalanced Data Pipeline & MLflow Tracking

### 🚀 What Was Accomplished
* **Imbalanced Dataset Handling:** Implemented a robust `imbalanced-learn` pipeline (`ImbPipeline`) integrating `SMOTENC` to handle mixed numerical and categorical features for class imbalance.
* **Feature Processing Setup:** Configured a `ColumnTransformer` with `StandardScaler` for numerical attributes and an `OrdinalEncoder` for categorical columns.
* **Model Training:** Trained a `RandomForestClassifier` (with 200 estimators and balanced class weights) on the resampled training split.
* **MLflow Tracking Integration:** 
  * Set up MLflow tracking experiments (`SMOTE_Imbalanced_Classification`)[span_0](start_span)[span_0](end_span).
  * Logged key parameters (`model`, `n_estimators`, `class_weight`, `smote_k_neighbors`, `random_state`) and performance metrics (`precision macro`, `recall macro`, `f1 macro`)[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span).
  * Secured model logging using `skops_trusted_types` to whitelist custom third-party components like `SMOTENC`.
* **Model Serialization:** Successfully exported and saved the end-to-end trained pipeline locally as `smote_rf_pipeline.pkl` for downstream inference[span_3](start_span)[span_3](end_span).

### 📊 Evaluation Results
* Evaluated model performance on the test set using classification reports, tracking macro averages and class-specific metrics to ensure reliable handling of minority classes.
*  