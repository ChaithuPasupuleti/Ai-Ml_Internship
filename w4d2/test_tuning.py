# test_tuning.py
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import make_data


def test_grid_search_gives_reasonable_error():
    X_tr, X_val, y_tr, y_val = make_data()
    pipe = make_pipeline(PolynomialFeatures(), StandardScaler(), Ridge(max_iter=20000))
    grid = {"polynomialfeatures__degree": [2, 4, 6], "ridge__alpha": [0.01, 0.1, 1]}
    search = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_squared_error")
    search.fit(X_tr, y_tr)
    assert mean_squared_error(y_val, search.predict(X_val)) < 0.5


def test_best_params_are_from_grid():
    X_tr, X_val, y_tr, y_val = make_data()
    pipe = make_pipeline(PolynomialFeatures(), Ridge())
    search = GridSearchCV(pipe, {"polynomialfeatures__degree": [2, 3]}, cv=3)
    search.fit(X_tr, y_tr)
    assert search.best_params_["polynomialfeatures__degree"] in [2, 3]