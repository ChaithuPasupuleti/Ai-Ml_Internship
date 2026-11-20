# regularisation.py - Ridge (L2) vs Lasso (L1) on an overfit-prone degree-12 model
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import make_data

DEGREE = 12
ALPHAS = np.logspace(-5, 1, 13)  # 0.00001 ... 10


def build(reg):
    return make_pipeline(PolynomialFeatures(DEGREE), StandardScaler(), reg)


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X_tr, X_val, y_tr, y_val = make_data()

    # Baseline: no regularisation (expect overfitting)
    base = build(LinearRegression()).fit(X_tr, y_tr)
    print(f"No regularisation: val MSE={mean_squared_error(y_val, base.predict(X_val)):.4f}")

    for name, Reg in [("Ridge", Ridge), ("Lasso", Lasso)]:
        print(f"\n{name}")
        best_alpha, best_cv = None, np.inf
        for alpha in ALPHAS:
            model = build(Reg(alpha=alpha, max_iter=200000))
            # 5-fold CV on the TRAIN set only
            cv_mse = -cross_val_score(
                model, X_tr, y_tr, cv=5, scoring="neg_mean_squared_error"
            ).mean()
            model.fit(X_tr, y_tr)
            val_mse = mean_squared_error(y_val, model.predict(X_val))
            extra = ""
            if name == "Lasso":
                zeros = int(np.sum(model[-1].coef_ == 0))
                extra = f" | zero coefs={zeros}/{DEGREE + 1}"
            print(f"  alpha={alpha:<10.5f} cv MSE={cv_mse:.4f} val MSE={val_mse:.4f}{extra}")
            if cv_mse < best_cv:
                best_alpha, best_cv = alpha, cv_mse
        print(f"  -> best alpha by cross-validation: {best_alpha:.5f}")


if __name__ == "__main__":
    main()