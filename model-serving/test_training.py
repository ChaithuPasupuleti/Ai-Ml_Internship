"""Test that training produces a valid artifact."""
from pathlib import Path

import joblib

from train import MODEL_PATH, main


def test_training_creates_model():
    f1 = main()
    assert Path(MODEL_PATH).exists()
    assert f1 > 0.5  # reasonable baseline on this synthetic data
    artifact = joblib.load(MODEL_PATH)
    assert 0.0 < artifact["threshold"] < 1.0