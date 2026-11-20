# tuning.py - Grid Search vs Randomised Search, logged to MLflow
import time

import mlflow
from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import make_data


def main():
    X_tr, X_val, y_tr, y_val = make_data()
    pipe = make_pipeline(PolynomialFeatures(), StandardScaler(), Ridge(max_iter=20000))

    grid_space = {
        "polynomialfeatures__degree": [2, 4, 6, 8, 10, 12],
        "ridge__alpha": [0.001, 0.01, 0.1, 1, 10],
    }
    random_space = {
        "polynomialfeatures__degree": list(range(2, 13)),
        "ridge__alpha": loguniform(1e-4, 1e2),
    }

    searches = [
        ("GridSearch", GridSearchCV(
            pipe, grid_space, cv=5, scoring="neg_mean_squared_error")),
        ("RandomSearch", RandomizedSearchCV(
            pipe, random_space, n_iter=15, cv=5,
            scoring="neg_mean_squared_error", random_state=42)),
    ]

    mlflow.set_experiment("w4d2-bias-variance")
    for name, search in searches:
        with mlflow.start_run(run_name=name):
            start = time.time()
            search.fit(X_tr, y_tr)
            seconds = time.time() - start

            val_mse = mean_squared_error(y_val, search.predict(X_val))
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_mse", -search.best_score_)
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("fit_seconds", seconds)
            print(f"{name}: {search.best_params_} | val MSE={val_mse:.4f} | {seconds:.2f}s")


if __name__ == "__main__":
    main()