#day 2 pandas_manipulation
import os
import pandas as pd

# --- 1. LOAD A REAL DATASET ---
# Using a public CSV dataset (Titanic dataset)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print('--- Dataset Info ---')
print('Shape:', df.shape)
print('\nData Types:\n', df.dtypes)
print('\nFirst 10 Rows:\n', df.head(10))


# --- 2. PERFORM OPERATIONS (Filter, Gr
# oupby, Merge, Pivot Table) ---

# A. Filtering: Select passengers who survived and are in Pclass 1
filtered_df = df[(df['Survived'] == 1) & (df['Pclass'] == 1)]
print('\nFiltered DataFrame Shape (Survived & Pclass 1):', filtered_df.shape)

# B. Groupby: Calculate average fare and age grouped by Pclass and Sex
grouped_df = (
    df.groupby(['Pclass', 'Sex'])
    .agg({'Fare': 'mean', 'Age': 'mean'})
    .reset_index()
)
print('\nGroupby Results (Mean Fare & Age):\n', grouped_df)

# C. Merge: Create a dummy department/class lookup table and merge it
class_mapping = pd.DataFrame(
    {'Pclass': [1, 2, 3], 'Class_Name': ['First', 'Second', 'Third']}
)
merged_df = pd.merge(df, class_mapping, on='Pclass', how='left')
print('\nMerged DataFrame Preview (with Class_Name):\n', merged_df.head())

# D. Pivot Table: Average fare by Pclass and Survived status
pivot_tb = df.pivot_table(
    values='Fare', index='Pclass', columns='Survived', aggfunc='mean'
)
print('\nPivot Table (Average Fare by Pclass & Survived):\n', pivot_tb)


# --- 3. EXPORT TO CSV & PARQUET AND COMPARE SIZES ---
csv_filename = 'cleaned_dataset.csv'
parquet_filename = 'cleaned_dataset.parquet'

# Export files
merged_df.to_csv(csv_filename, index=False)

# Note: Parquet requires pyarrow or fastparquet installed (pip install pyarrow)
try:
  merged_df.to_parquet(parquet_filename, index=False)

  # Compare file sizes
  csv_size = os.path.getsize(csv_filename)
  parquet_size = os.path.getsize(parquet_filename)

  print('\n--- File Size Comparison ---')
  print(f'CSV File Size: {csv_size / 1024:.2f} KB')
  print(f'Parquet File Size: {parquet_size / 1024:.2f} KB')
  print(
      f'Parquet is approximately {(csv_size / parquet_size):.1f}x smaller than'
      ' CSV.'
  )
except Exception as e:
  print('\nParquet export note (install pyarrow if needed):', e)