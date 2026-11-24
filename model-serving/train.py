"""Train a classifier on imbalanced data and save it for serving."""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"
# Local folder inside the project (works on Windows, Mac and Linux)
# SQLite database inside the project (MLflow's recommended local backend)
MLFLOW_URI = "sqlite:///mlflow.db"


def main() -> float:
    np.random.seed(42)

    # 1. Imbalanced dataset (95% class 0, 5% class 1)
    X, y = make_classification(
        n_samples=5000, n_features=10, weights=[0.95, 0.05], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print("Class counts before SMOTE:", np.bincount(y_train))

    # 2. SMOTE on training data only
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print("Class counts after SMOTE: ", np.bincount(y_res))

    # 3. Class weighting
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_res, y_res)

    # 4. Threshold optimisation
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    # thresholds has one fewer element than precision/recall,
    # so drop the last F1 value before taking the argmax
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_score(y_test, probs >= best_threshold))
    print(f"Best threshold = {best_threshold:.3f}, F1 = {best_f1:.3f}")

    # 5. MLflow logging (versioned model + threshold artifact)
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="imbalanced_logreg"):
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("smote", True)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1", best_f1)
        mlflow.sklearn.log_model(model, artifact_path="model")
        Path("threshold.txt").write_text(str(best_threshold))
        mlflow.log_artifact("threshold.txt")

    # 6. Save for the API, then reload as a sanity check
    joblib.dump({"model": model, "threshold": best_threshold}, MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    assert np.isclose(loaded["threshold"], best_threshold)
    print(f"Saved {MODEL_PATH}")
    return best_f1


if __name__ == "__main__":
    main()