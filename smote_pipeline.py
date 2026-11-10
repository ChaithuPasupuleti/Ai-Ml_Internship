# --------------------------------------------------------------
#  week-2-project : End-to-End Preprocessing + SMOTENC + RF
#  Author: Chaithanya Pasupuleti (AI/ML Intern)
#  Stack: CrewAI • LangGraph • MLflow • Ragas • MLOps
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
import mlflow
import mlflow.sklearn


def build_preprocessor(numerical_cols, categorical_cols):
    """Impute numerical and categorical columns without OHE yet."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


def build_postprocessor(categorical_features_indices):
    """One-Hot encode categoricals using integer indices after SMOTENC outputs a numpy array."""
    return ColumnTransformer(
        transformers=[
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features_indices)
        ],
        remainder="passthrough",   # Keep the numerical columns untouched
    )


def create_imbalanced_pipeline(
    numerical_cols,
    categorical_cols,
    categorical_features_indices,
    rf_params=None,
    k_neighbors: int = 5,
):
    """
    Returns an ImbPipeline:
        1️⃣ Impute (numeric + categorical)  
        2️⃣ SMOTENC (on raw categorical columns)  
        3️⃣ One-Hot encode categoricals + keep numericals scaled  
        4️⃣ RandomForest classifier
    """
    if rf_params is None:
        rf_params = {"n_estimators": 200, "random_state": 42, "class_weight": "balanced"}

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    smote = SMOTENC(
        categorical_features=categorical_features_indices,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    
    # Since preprocessor places numerical columns first, 
    # categorical columns in the transformed array start at index `len(numerical_cols)`
    num_len = len(numerical_cols)
    transformed_cat_indices = list(range(num_len, num_len + len(categorical_cols)))
    
    postprocessor = build_postprocessor(transformed_cat_indices)
    classifier = RandomForestClassifier(**rf_params)

    pipeline = ImbPipeline(
        steps=[
            ("impute", preprocessor),
            ("smote", smote),
            ("encode", postprocessor),
            ("clf", classifier),
        ]
    )
    return pipeline


def train_evaluate_log(
    X,
    y,
    numerical_cols,
    categorical_cols,
    cat_idx_raw,
    test_size=0.2,
    random_state=42,
):
    """
    Splits data, fits pipeline, evaluates, and logs metrics/artifacts to MLflow.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = create_imbalanced_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        categorical_features_indices=cat_idx_raw,
    )

    mlflow.set_experiment("SMOTE_Imbalanced_Classification")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model": "RandomForest",
                "n_estimators": 200,
                "class_weight": "balanced",
                "smote_k_neighbors": 5,
                "random_state": random_state,
                "test_size": test_size,
            }
        )

        pipe.fit(X_train, y_train)

        # Evaluation
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, output_dict=True)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", macro_f1)
        for label, metrics in report.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                mlflow.log_metric(f"f1_class_{label}", metrics["f1-score"])

        # Log model artifact with trusted types
        mlflow.sklearn.log_model(
            pipe,
            "model",
            skops_trusted_types=[
                "imblearn.over_sampling._smote.base.SMOTENC",
                "imblearn.pipeline.Pipeline",
                "sklearn.tree._tree.Tree",
                "numpy.dtype",
                "scipy.sparse._csr.csr_matrix",
            ],
        )
        print("Training complete, metrics and model successfully logged to MLflow.")