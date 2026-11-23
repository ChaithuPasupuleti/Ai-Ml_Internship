"""W4D3: Model serialisation with pickle and joblib."""
import os
import time
import pickle
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score)

MODEL_DIR = "models"
PKL_PATH = os.path.join(MODEL_DIR, "model.pkl")
JOBLIB_PATH = os.path.join(MODEL_DIR, "model.joblib")


def save_pickle(model, path):
    """Save with pickle; return (seconds, size in KB)."""
    start = time.time()
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return time.time() - start, os.path.getsize(path) / 1024


def save_joblib(model, path, compress=3):
    """Save with joblib; return (seconds, size in KB)."""
    start = time.time()
    joblib.dump(model, path, compress=compress)
    return time.time() - start, os.path.getsize(path) / 1024


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data; stratify keeps the class ratio the same in train and test
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Train
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluate (accuracy alone lies, so use more metrics)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))

    # 4. Serialise with both libraries
    pk_time, pk_kb = save_pickle(model, PKL_PATH)
    jl_time, jl_kb = save_joblib(model, JOBLIB_PATH)
    print(f"pickle: {pk_time:.3f}s, {pk_kb:.1f} KB")
    print(f"joblib: {jl_time:.3f}s, {jl_kb:.1f} KB")

    # 5. Reload and verify
    with open(PKL_PATH, "rb") as f:
        pk_model = pickle.load(f)
    jl_model = joblib.load(JOBLIB_PATH)

    assert np.array_equal(pk_model.predict(X_test), y_pred), "Pickle model mismatch!"
    assert np.array_equal(jl_model.predict(X_test), y_pred), "Joblib model mismatch!"
    print("Loaded models match the original predictions")


if __name__ == "__main__":
    main()