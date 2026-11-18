"""W3D5: Hyperparameter tuning with GridSearchCV and RandomizedSearchCV.

Tunes SVM and KNN classifiers on the Iris dataset and logs parameters,
metrics and the best model for each search to MLflow.
"""

from typing import Union

import mlflow
import mlflow.sklearn
from scipy.stats import loguniform, randint
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 20
EXPERIMENT_NAME = "w3d5-hyperparameter-tuning"

SearchCV = Union[GridSearchCV, RandomizedSearchCV]


def load_data():
    """Load Iris and return a stratified train/test split."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


def build_pipelines():
    """Scaling matters for both SVM and KNN, so wrap each in a Pipeline."""
    svm_pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(random_state=RANDOM_STATE))]
    )
    knn_pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
    )
    return svm_pipe, knn_pipe


def build_search_spaces():
    """Return (grid, random) parameter spaces for SVM and KNN."""
    # GridSearch tries every combination
    svm_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", 0.01, 0.1],
    }
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }

    # RandomSearch samples N random combinations from distributions
    svm_random = {
        "clf__C": loguniform(1e-2, 1e3),
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__kernel": ["linear", "rbf"],
    }
    knn_random = {
        "clf__n_neighbors": randint(1, 30),
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }
    return svm_grid, knn_grid, svm_random, knn_random


def run_search(name: str, search: SearchCV, X_train, X_test, y_train, y_test) -> None:
    """Fit a search object, evaluate on the test set, and log to MLflow."""
    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, search.predict(X_test))

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(search.best_estimator_, "model", serialization_format="cloudpickle")

        print(
            f"{name}: CV={search.best_score_:.3f} | "
            f"Test={test_acc:.3f} | Params={search.best_params_}"
        )


def main() -> None:
    # Runs are stored in a local ./mlruns folder (view with `mlflow ui`)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    svm_pipe, knn_pipe = build_pipelines()
    svm_grid, knn_grid, svm_random, knn_random = build_search_spaces()

    common = {"cv": CV_FOLDS, "n_jobs": -1, "scoring": "accuracy"}
    random_common = {**common, "n_iter": N_ITER, "random_state": RANDOM_STATE}
    data = (X_train, X_test, y_train, y_test)

    searches = [
        ("SVM-GridSearch", GridSearchCV(svm_pipe, svm_grid, **common)),
        ("SVM-RandomSearch", RandomizedSearchCV(svm_pipe, svm_random, **random_common)),
        ("KNN-GridSearch", GridSearchCV(knn_pipe, knn_grid, **common)),
        ("KNN-RandomSearch", RandomizedSearchCV(knn_pipe, knn_random, **random_common)),
    ]

    for name, search in searches:
        run_search(name, search, *data)


if __name__ == "__main__":
    main()
