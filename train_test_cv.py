# --------------------------------------------------------------
# 0. Limit threads (fixes the OpenBLAS memory error on Windows)
#    These lines MUST stay at the very top, before numpy is imported
# --------------------------------------------------------------
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------
import numpy as np
from datetime import datetime

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

import mlflow
import mlflow.sklearn

# Use the SAME database that `mlflow ui` reads from
# (run this script from the same folder where you run `mlflow ui`)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())

# --------------------------------------------------------------
# 2. Settings
# --------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CONFIG = {
    "test_size": 0.2,
    "n_splits_cv": 5,
    "scoring": "balanced_accuracy",
    "model_name": "RandomForestClassifier",
    "scaler_name": "RobustScaler",
    "n_estimators": 200,
    "run_name": f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
}

# --------------------------------------------------------------
# 3. Data
# --------------------------------------------------------------
X, y = make_classification(
    n_samples=1_000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=RANDOM_SEED,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    stratify=y,
    random_state=RANDOM_SEED,
)

print(f"TRAIN shape: {X_train.shape}, TEST shape: {X_test.shape}")

# --------------------------------------------------------------
# 4. Model pipeline and cross-validation setup
# --------------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("scaler", RobustScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=CONFIG["n_estimators"],
            random_state=RANDOM_SEED,
            n_jobs=2,
        )),
    ]
)

cv = StratifiedKFold(
    n_splits=CONFIG["n_splits_cv"],
    shuffle=True,
    random_state=RANDOM_SEED,
)

# --------------------------------------------------------------
# 5. Train, evaluate and log everything to MLflow
# --------------------------------------------------------------
with mlflow.start_run(run_name=CONFIG["run_name"]):

    mlflow.log_params({
        "model": CONFIG["model_name"],
        "scaler": CONFIG["scaler_name"],
        "n_estimators": CONFIG["n_estimators"],
        "n_splits_cv": CONFIG["n_splits_cv"],
        "scoring": CONFIG["scoring"],
        "test_size": CONFIG["test_size"],
    })

    # Cross-validation on training data
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=CONFIG["scoring"],
    )
    mlflow.log_metric("cv_mean_score", float(cv_scores.mean()))
    mlflow.log_metric("cv_std_score", float(cv_scores.std()))

    # Train on full training set, test on unseen test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_acc = pipeline.score(X_test, y_test)
    mlflow.log_metric("test_balanced_accuracy", float(test_bal_acc))
    mlflow.log_metric("test_accuracy", float(test_acc))

    # Save the model
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("CV scores:", cv_scores)
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Test balanced accuracy: {test_bal_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

print("Done. Open http://127.0.0.1:5000 and refresh Training runs.")