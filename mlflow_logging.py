"""
W4D1 - Log the experiment to MLflow.

Run eval_metrics.py and cv_analysis.py first so the PNG files exist.
Then view results with:  mlflow ui   (open http://127.0.0.1:5000)
"""
import os

import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from eval_metrics import train_model


def main():
    model, X_train, X_test, y_train, y_test = train_model()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mlflow.set_experiment("w4d1-model-evaluation")

    with mlflow.start_run(run_name="w4d1-logreg"):
        # Parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("class_ratio", "90/10")

        # Metrics
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        # Plots (only if they have been generated)
        for path in ["roc.png", "pr.png", "learning_curve.png"]:
            if os.path.exists(path):
                mlflow.log_artifact(path)
            else:
                print(f"Warning: {path} not found, skipping")

    print("Run logged. Start the UI with: mlflow ui")


if __name__ == "__main__":
    main()