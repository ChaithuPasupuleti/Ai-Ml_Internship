import pandas as pd
import numpy as np
from smote_pipeline import create_imbalanced_pipeline

def test_create_imbalanced_pipeline():
    # Create a tiny mock dataset with mixed numerical and categorical features
    df = pd.DataFrame({
        'num_feature_1': [1.5, 2.0, 3.1, 4.0, 5.2, 0.5],
        'num_feature_2': [10, 20, 15, 40, 50, 5],
        'cat_feature_1': ['A', 'B', 'A', 'B', 'A', 'B']
    })
    y = np.array([0, 0, 0, 1, 1, 1])
    
    numerical_cols = ['num_feature_1', 'num_feature_2']
    categorical_cols = ['cat_feature_1']
    # Index of categorical column in original dataframe
    cat_idx_raw = [2] 
    
    pipeline = create_imbalanced_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        categorical_features_indices=cat_idx_raw,
        k_neighbors=1
    )
    
    # Verify fitting works without errors
    pipeline.fit(df, y)
    preds = pipeline.predict(df)
    
    assert len(preds) == len(y), "Prediction length must match target length"
    print("Pipeline unit test passed successfully!")

if __name__ == "__main__":
    test_create_imbalanced_pipeline()