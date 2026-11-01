import pandas as pd


def load_dataset(file_path):
  """Loads the dataset from a CSV file."""
  return pd.read_csv(file_path)


def clean_dataset(df):
  """Cleans the dataset by handling missing values and data types."""
  if df.empty:
    return df
  # Add your cleaning steps here
  return df