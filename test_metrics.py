"""Simple tests for W4D1. Run with:  pytest -v"""
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from eval_metrics import train_model


def test_auc_better_than_random():
    # Hand-made example: positives are ranked mostly above negatives
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) > 0.5


def test_precision_recall_known_values():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]  # TP=2, FN=1, FP=1, TN=2
    assert precision_score(y_true, y_pred) == 2 / 3
    assert recall_score(y_true, y_pred) == 2 / 3


def test_model_beats_random_on_imbalanced_data():
    model, X_train, X_test, y_train, y_test = train_model()
    y_prob = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, y_prob) > 0.7