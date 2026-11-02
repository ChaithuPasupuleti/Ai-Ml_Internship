import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_dataset.csv")   

print(df.describe())
df.info()
print(df.isnull().sum()) 

# 1. Distribution of each numeric column
num_cols = df.select_dtypes(include="number").columns
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig("outputs/distributions.png")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# 3. Top-10 category counts (repeat per categorical column)
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col].value_counts().head(10).plot(kind="bar", title=f"Top 10: {col}")
    plt.tight_layout()
    plt.savefig(f"top10_{col}.png")
    plt.show() 