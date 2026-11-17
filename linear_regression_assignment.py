import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# --- 0. MLflow setup (local, fresh database + artifact folder) ---
os.environ.pop("MLFLOW_TRACKING_URI", None)  # ignore any old setting

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "mlartifacts_v2"
ARTIFACT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow_v2.db').as_posix()}")

EXPERIMENT_NAME = "california_housing_regression_v2"
if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_DIR.as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 1. Load Data from Scikit-Learn ---
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 2. Train Models with Hyperparameter Tuning ---
models = {}

# Linear Regression (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
models["Linear_Regression"] = lr

# Ridge Grid Search
ridge_params = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(
    Ridge(), ridge_params, scoring="neg_root_mean_squared_error", cv=5
)
ridge_grid.fit(X_train, y_train)
models["Ridge_L2"] = ridge_grid.best_estimator_
print("Best Ridge Alpha:", ridge_grid.best_params_)

# Lasso Grid Search
lasso_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}
lasso_grid = GridSearchCV(
    Lasso(max_iter=10000),
    lasso_params,
    scoring="neg_root_mean_squared_error",
    cv=5,
)
lasso_grid.fit(X_train, y_train)
models["Lasso_L1"] = lasso_grid.best_estimator_
print("Best Lasso Alpha:", lasso_grid.best_params_)

# --- 3. Evaluate and Compare Models ---
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation RMSE
    cv_scores = cross_val_score(
        model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5
    )
    cv_rmse = -np.mean(cv_scores)

    results.append(
        {
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "CV_RMSE": cv_rmse,
        }
    )

results_df = pd.DataFrame(results)
print("\n--- Model Performance Comparison Table ---")
print(results_df.to_string(index=False))

# --- 4. Identify Best Model & Log with MLflow ---
best_row = results_df.loc[results_df["RMSE"].idxmin()]
best_model_name = best_row["Model"]
best_model = models[best_model_name]

print(f"\nBest model based on RMSE: {best_model_name}")

with mlflow.start_run():
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("rmse", float(best_row["RMSE"]))
    mlflow.log_metric("r2", float(best_row["R2"]))
    mlflow.sklearn.log_model(best_model, name="model")

print("MLflow logging completed successfully.")

# --- 5. Plotting Results ---
best_pred = best_model.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, best_pred, alpha=0.5, color="navy")
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
)
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"{best_model_name} - Actual vs Predicted")

# Residuals Plot
residuals = y_test - best_pred
axes[1].scatter(best_pred, residuals, alpha=0.5, color="crimson")
axes[1].axhline(y=0, color="k", linestyle="--", lw=2)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residuals")
axes[1].set_title(f"{best_model_name} - Residuals")

plt.tight_layout()
plt.show()