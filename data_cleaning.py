import pandas as pd

# 1. Load dataset (replace with your actual file path)
df = pd.read_csv("cleaned_dataset.csv")

# 2. Inspect data
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Clean data (dropping rows with missing values)
df_cleaned = df.dropna()
print(f"\nRows after cleaning: {len(df_cleaned)}")
