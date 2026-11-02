import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("cleaned_dataset.csv")

print(df.describe())
df.info()
print(df.isnull().sum())

# 1. Distribution of each numeric column
num_cols = df.select_dtypes(include="number").columns
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig("outputs/distributions.png")
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# 3. Top-10 category counts
cat_cols = ["Sex", "Embarked", "Class_Name"]
for col in cat_cols:
    plt.figure(figsize=(8, 5))
    df[col].value_counts().head(10).plot(kind="bar", title=f"Top 10: {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/top10_{col}.png")
    plt.close()