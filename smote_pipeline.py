import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import joblib

# 1. Load and clean your dataset
df = pd.read_csv('cleaned_dataset.csv')  # Replace with your actual filename if needed
X = df.drop(columns=['Survived'])         # Replace 'Survived' with your actual label column
y = df['Survived']

# 2. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Identify column types & calculate safe k for SMOTENC
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

cat_idx = [X_train.columns.get_loc(col) for col in categorical_cols]

# Dynamically set k_neighbors based on minority class count
minority_cnt = Counter(y_train).most_common()[-1][1]
k = min(5, minority_cnt - 1)

print('Before SMOTE:', Counter(y_train))

# 4. Preprocessing setup
cat_encoder = Pipeline(steps=[
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocess_before_smote = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', cat_encoder, categorical_cols)
    ]
)

# 5. Full pipeline with SMOTENC and Random Forest Classifier
smote = SMOTENC(
    categorical_features=cat_idx,
    random_state=42,
    k_neighbors=k
)

pipeline = ImbPipeline(steps=[
    ('preprocess', preprocess_before_smote),
    ('smote', smote),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced'
    ))
])

# 6. MLflow Tracking & Training
mlflow.set_experiment('SMOTE_Imbalanced_Classification')

with mlflow.start_run():
    mlflow.log_params({
        'model': 'RandomForest',
        'n_estimators': 200,
        'class_weight': 'balanced',
        'smote_k_neighbors': k,
        'random_state': 42
    })
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Predict & evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
   # Log model to MLflow with all required trusted types
mlflow.sklearn.log_model(
    pipeline, 
    'model',
    skops_trusted_types=[
        'imblearn.over_sampling._smote.base.SMOTENC',
        'imblearn.pipeline.Pipeline',
        'sklearn.tree._tree.Tree',
        'numpy.dtype',
        'scipy.sparse._csr.csr_matrix'
    ]
)