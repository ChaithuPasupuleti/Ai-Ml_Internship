from pathlib import Path
import pandas as pd
from data_loader import load_dataset, clean_dataset


def test_load_and_clean(tmp_path):
    # create a tiny CSV
    csv = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2, None]}).to_csv(csv, index=False)

    df = load_dataset(csv)
    assert len(df) == 3

    df_clean = clean_dataset(df)
    assert len(df_clean) == 2  # row with None is dropped
