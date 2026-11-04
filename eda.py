from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

csv_path = Path("cleaned_dataset.csv")
if not csv_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {csv_path}")
df = pd.read_csv(csv_path)
print(df.describe())
df.info()
print(df.isnull().sum())
# 1. Distribution of each numeric column
num_cols = df.select_dtypes(include="number").columns
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distributions.png")
plt.close()
# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()
# 3. Top-10 category counts
cat_cols = ["Sex", "Embarked", "Class_Name"]
for col in cat_cols:
    plt.figure(figsize=(8, 5))
    df[col].value_counts(dropna=False).head(10).plot(kind="bar", title=f"Top 10: {col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"top10_{col}.png")
    plt.close()