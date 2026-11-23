# data.py - shared synthetic dataset used by every script
import numpy as np
from sklearn.model_selection import train_test_split


def make_data(seed=42):
    """Return X_train, X_val, y_train, y_val for y = sin(x) + noise."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.rand(100, 1) * 6, axis=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.3, 100)
    return train_test_split(X, y, test_size=0.3, random_state=seed)