# bias_variance.py - shows underfitting vs overfitting as model complexity grows
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from data import make_data


def main():
    X_tr, X_val, y_tr, y_val = make_data()
    degrees = list(range(1, 16))
    train_err, val_err = [], []

    for d in degrees:
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(X_tr, y_tr)
        train_err.append(mean_squared_error(y_tr, model.predict(X_tr)))
        val_err.append(mean_squared_error(y_val, model.predict(X_val)))

    plt.plot(degrees, train_err, marker="o", label="Train MSE")
    plt.plot(degrees, val_err, marker="o", label="Validation MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()
    plt.savefig("bias_variance.png", dpi=300, bbox_inches="tight")
    print("Saved bias_variance.png")


if __name__ == "__main__":
    main()