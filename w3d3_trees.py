"""W3D3: Decision Trees & Random Forests with MLflow tracking."""
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def init_mlflow() -> None:
    """Select the MLflow experiment (runs are stored locally in ./mlruns)."""
    mlflow.set_experiment("w3d3-decision-trees")


def load_data():
    """Load the Breast Cancer dataset and split it into train/test sets."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return data, X_train, X_test, y_train, y_test


def train_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
) -> DecisionTreeClassifier:
    """Train a decision tree with the given impurity criterion and depth."""
    model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=42
    )
    return model.fit(X_train, y_train)


def evaluate(model, X_train, X_test, y_train, y_test):
    """Return (train accuracy, test accuracy) for a fitted model."""
    return (
        accuracy_score(y_train, model.predict(X_train)),
        accuracy_score(y_test, model.predict(X_test)),
    )


def main() -> None:
    """Compare tree settings, visualise a tuned tree, and train a forest."""
    data, X_train, X_test, y_train, y_test = load_data()
    init_mlflow()

    # 1. Compare Gini vs Entropy (information gain) and depths (overfitting)
    for criterion in ["gini", "entropy"]:
        for depth in [None, 2, 3, 5, 8]:
            with mlflow.start_run(run_name=f"tree-{criterion}-{depth}"):
                model = train_tree(X_train, y_train, criterion, depth)
                tr, te = evaluate(model, X_train, X_test, y_train, y_test)
                mlflow.log_params(
                    {"model": "tree", "criterion": criterion, "max_depth": str(depth)}
                )
                mlflow.log_metrics({"train_acc": tr, "test_acc": te, "gap": tr - te})
                print(f"{criterion:8} depth={depth}: train={tr:.3f} test={te:.3f}")

    # 2. Visualise a tuned tree
    best = train_tree(X_train, y_train, "gini", max_depth=3)
    plt.figure(figsize=(16, 8))
    plot_tree(
        best,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
    )
    plt.savefig("tree.png")
    plt.close()

    # 3. Random Forest
    with mlflow.start_run(run_name="random-forest"):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        tr, te = evaluate(rf, X_train, X_test, y_train, y_test)
        mlflow.log_params({"model": "rf", "n_estimators": 100, "max_depth": 5})
        mlflow.log_metrics({"train_acc": tr, "test_acc": te})
        mlflow.log_artifact("tree.png")
        mlflow.sklearn.log_model(
            rf,
            name="rf-model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"Random Forest: train={tr:.3f} test={te:.3f}")


if __name__ == "__main__":
    main()