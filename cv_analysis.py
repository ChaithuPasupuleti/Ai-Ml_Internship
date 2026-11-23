"""
W4D1 - Cross-validation and learning curves.

1. KFold vs StratifiedKFold on an imbalanced dataset.
2. Learning curves to diagnose overfitting vs underfitting, including a
   depth-limited tree to show how regularisation reduces overfitting.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.tree import DecisionTreeClassifier

from eval_metrics import load_data


def compare_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
    """Compare KFold vs StratifiedKFold and print mean +/- std per metric."""
    model = LogisticRegression(max_iter=1000)
    scoring = ["precision", "recall", "roc_auc"]
    splitters = {
        "KFold": KFold(n_splits=n_splits, shuffle=True, random_state=42),
        "StratifiedKFold": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
    }

    for name, cv in splitters.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"\n{name}")
        for metric in scoring:
            scores = res[f"test_{metric}"]
            print(f"  {metric:10s}: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Observed: mean scores are almost the same for KFold and StratifiedKFold.
        # Stratified gave a lower std for ROC-AUC (0.022 vs 0.034) because every fold
        # keeps the 90/10 class ratio. Precision std was not lower (0.128 vs 0.070)
        # because the model predicts very few positives (recall ~0.04), so precision
        # is computed from only a few predictions per fold.


def plot_learning_curves(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
    """Plot ROC-AUC learning curves for three models side by side."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree (no depth limit)": DecisionTreeClassifier(random_state=42),
        "DecisionTree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)

    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=StratifiedKFold(n_splits, shuffle=True, random_state=42),
            scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 8),
        )
        train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)

        ax.plot(sizes, train_mean, marker="o", label="Train")
        ax.plot(sizes, val_mean, marker="o", label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("ROC-AUC")
        ax.legend()
        ax.grid(alpha=0.3)

        print(f"\n{name}: final train={train_mean[-1]:.3f}, validation={val_mean[-1]:.3f}")

    # Reading the curves:
    # - DecisionTree (no depth limit): train ROC-AUC is ~1.0 but validation is much
    #   lower. The big gap means OVERFITTING (it memorises the training data).
    # - DecisionTree (max_depth=5): limiting depth is regularisation. The gap
    #   between train and validation shrinks, so overfitting is reduced.
    # - LogisticRegression: train and validation are close together. If both were
    #   low, that would be UNDERFITTING (model too simple / high bias).
    fig.tight_layout()
    fig.savefig("learning_curve.png", dpi=150)
    plt.close(fig)
    print("\nSaved learning_curve.png")


def main() -> None:
    X, y = load_data()
    compare_kfold(X, y)
    plot_learning_curves(X, y)


if __name__ == "__main__":
    main()