"""
W3D4 - SVM vs KNN comparison
* Tune hyperparameters with GridSearchCV
* Show the effect of scaling on KNN
* Capture fit / predict time
* Log params, metrics and the best model to MLflow
"""
import time
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
CV_FOLDS = 5
mlflow.set_experiment("svm-vs-knn")

# 1. Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 2. Experiments: name -> (pipeline, hyperparameter grid)
knn_grid = {
    "model__n_neighbors": [3, 5, 7, 11],
    "model__weights": ["uniform", "distance"],
}
experiments = {
    "SVM_scaled": (
        Pipeline([("scale", StandardScaler()), ("model", SVC())]),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    ),
    "KNN_scaled": (
        Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
        knn_grid,
    ),
    "KNN_unscaled": (  # shows why scaling matters for distance-based models
        Pipeline([("model", KNeighborsClassifier())]),
        knn_grid,
    ),
}

# 3. Train, evaluate, log
for name, (pipe, grid) in experiments.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"cv_folds": CV_FOLDS, "random_state": SEED})

        search = GridSearchCV(pipe, grid, cv=CV_FOLDS, scoring="f1_weighted")

        t0 = time.time()
        search.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        preds = search.predict(X_test)
        predict_time = time.time() - t0

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1,
            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
        })
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

        print(f"{name:13} acc={acc:.3f} f1={f1:.3f} "
              f"fit={fit_time:.2f}s predict={predict_time:.4f}s {search.best_params_}")