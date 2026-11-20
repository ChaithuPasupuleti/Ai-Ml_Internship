"""
W4D1 - Model Evaluation Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC.

Trains a Logistic Regression on an imbalanced dataset (90% / 10%) and
reports the metrics that accuracy alone would hide.
"""
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create a 90/10 imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42,
    )
    return X, y


def train_model(balanced: bool = False):
    """Stratified split, train, and return (model, X_train, X_test, y_train, y_test).

    balanced=True sets class_weight="balanced" so the model penalises mistakes
    on the minority class more (usually raises recall, lowers precision).
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced" if balanced else None
    ).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def main() -> None:
    model, X_train, X_test, y_train, y_test = train_model(balanced=True)
    # Before (balanced=False): recall for class 1 was 0.05, precision 0.50, accuracy 0.90.
# After (balanced=True): recall is 0.74, precision is 0.23, accuracy 0.72.
# Balancing catches far more real positives but makes more false alarms (FP 2 -> 102).
# Prioritise recall when missing a positive is costly (e.g. disease, fraud detection).

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    # Confusion matrix layout: [[TN, FP], [FN, TP]]
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = harmonic mean
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Both use probabilities, not hard 0/1 predictions.
    # PR-AUC (average precision) is often more informative on imbalanced data.
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("PR-AUC :", round(average_precision_score(y_test, y_prob), 4))

    # Plots: close each figure so curves don't overlap
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.savefig("roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve")
    plt.savefig("pr.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved confusion.png, roc.png and pr.png")


if __name__ == "__main__":
    main()