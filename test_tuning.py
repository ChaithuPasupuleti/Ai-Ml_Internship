"""test_tuning.py: sanity check for GridSearchCV on the Iris dataset."""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def test_grid_search_finds_good_model() -> None:
    """Check that a small KNN grid search reaches over 90% CV accuracy on Iris."""
    X, y = load_iris(return_X_y=True)

    # Stratified, shuffled folds with a fixed seed give a repeatable split
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid={"n_neighbors": [3, 5]},
        cv=cv,
        scoring="accuracy",
    )
    gs.fit(X, y)

    assert gs.best_score_ > 0.9, f"Best CV score {gs.best_score_:.3f} is not above 0.9"
