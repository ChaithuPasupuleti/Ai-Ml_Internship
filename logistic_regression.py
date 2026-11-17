"""
Week 3 Day 2: Logistic Regression & Classification Tasks
Author: Chaithanya Pasupuleti
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")


def plot_decision_boundary(model, X, y, title: str) -> None:
    """Plot a 2-D decision boundary. X must have exactly 2 columns,
    and `model` must have been trained on those same 2 columns."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette="deep",
        edgecolor="k",
        legend=False,
    )
    plt.title(title)
    plt.xlabel("Feature 0 (scaled)")
    plt.ylabel("Feature 1 (scaled)")
    plt.tight_layout()
    mlflow.log_figure(fig, "decision_boundary.png")
    plt.show()
    plt.close()


def run_binary_classification_tasks() -> None:
    """Load breast-cancer data, train LogisticRegression, log with MLflow."""
    print("--- Binary Classification (Breast Cancer) ---")

    # ---- Data -------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Scaling (fit on train only to avoid data leakage) -----
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ---- MLflow ------------------------------------------------
    mlflow.set_experiment("Logistic_Regression_Binary")
    with mlflow.start_run(run_name="logreg_binary"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 10000,
                "class_weight": "balanced",
                "scaled": True,
            }
        )

        model = LogisticRegression(
            solver="lbfgs", max_iter=10000, class_weight="balanced"
        )
        model.fit(X_train_sc, y_train)

        # Log the trained model
        mlflow.sklearn.log_model(model, name="logreg_binary")

        # ---- Predictions & Metrics --------------------------------
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        # ---- 5-fold cross-validation (more reliable than one split) --
        cv_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=10000, class_weight="balanced"),
        )
        cv_scores = cross_val_score(cv_pipe, X, y, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

        mlflow.log_metrics(metrics)

        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k.capitalize():<18}: {v:.4f}")

        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # ---- Confusion Matrix Heatmap ----------------------------
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)
        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        fig.savefig("confusion_matrix.png", dpi=150)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.show()
        plt.close()

        # ---- ROC Curve -------------------------------------------
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}", color="darkorange")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        mlflow.log_figure(fig, "roc_curve.png")
        plt.show()
        plt.close()

        # ---- Decision Boundary -----------------------------------
        # The main model uses all 30 features, so it can't be plotted in 2-D.
        # Train a separate 2-feature model purely for visualisation.
        model_2d = LogisticRegression(
            solver="lbfgs", max_iter=10000, class_weight="balanced"
        )
        model_2d.fit(X_train_sc[:, :2], y_train)

        plot_decision_boundary(
            model_2d,
            X_test_sc[:, :2],
            y_test,
            "Decision Boundary (Breast Cancer - First 2 Features)",
        )


def run_multiclass_classification_tasks() -> None:
    """Load Iris data, compare One-vs-Rest vs Softmax strategies, log with MLflow."""
    print("\n--- Multi-Class Classification (Iris) ---")

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_i)
    X_test_sc = scaler.transform(X_test_i)

    mlflow.set_experiment("Logistic_Regression_Multiclass")
    with mlflow.start_run(run_name="logreg_multiclass"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 10000,
                "scaled": True,
            }
        )

        # One-vs-Rest (OvR): one binary classifier per class
        ovr_model = OneVsRestClassifier(
            LogisticRegression(solver="lbfgs", max_iter=10000)
        )
        ovr_model.fit(X_train_sc, y_train_i)
        ovr_pred = ovr_model.predict(X_test_sc)
        ovr_acc = accuracy_score(y_test_i, ovr_pred)

        # Softmax (Multinomial): the default for lbfgs with >2 classes
        softmax_model = LogisticRegression(solver="lbfgs", max_iter=10000)
        softmax_model.fit(X_train_sc, y_train_i)
        softmax_pred = softmax_model.predict(X_test_sc)
        softmax_acc = accuracy_score(y_test_i, softmax_pred)

        mlflow.log_metrics({"ovr_accuracy": ovr_acc, "softmax_accuracy": softmax_acc})

        comparison_df = pd.DataFrame(
            {
                "Strategy": ["One-vs-Rest (OvR)", "Softmax (Multinomial)"],
                "Accuracy": [ovr_acc, softmax_acc],
            }
        )

        print("\nMulti-Class Strategy Comparison:")
        print(comparison_df.to_string(index=False))

        print("\nClassification Report (Softmax / Multinomial):")
        print(classification_report(y_test_i, softmax_pred))

        mlflow.sklearn.log_model(softmax_model, name="logreg_multiclass")


if __name__ == "__main__":
    run_binary_classification_tasks()
    run_multiclass_classification_tasks()