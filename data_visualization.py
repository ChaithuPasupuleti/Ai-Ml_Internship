"""
Day 5: Data Visualisation - Matplotlib & Seaborn
Author: Chaithanya
Description: Generates key exploratory data analysis plots for the dataset.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style for clean, professional plots
sns.set_theme(style="whitegrid")

# Ensure the output directory exists
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(file_path):
  """Loads the dataset from a Parquet or CSV file."""
  try:
    if file_path.endswith(".parquet"):
      df = pd.read_parquet(file_path)
    else:
      df = pd.read_csv(file_path)
    print(f"Successfully loaded dataset from {file_path}")
    return df
  except FileNotFoundError:
    print(
        f"Error: The file {file_path} was not found. Please check the path."
    )
    return None


def plot_distributions(df):
  """Generates and saves EDA distribution plots and correlation heatmap."""
  if df is None:
    print("No dataframe provided to plot.")
    return

  print("Generating data visualizations...")

  # 1. Numerical distribution plot (e.g., Age distribution)
  if "Age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"].dropna(), kde=True, color="blue")
    plt.title("Age Distribution and Density", fontsize=14, fontweight="bold")
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"))
    plt.close()
    print("Saved: age_distribution.png")

  # 2. Categorical comparison plot (e.g., Survival by Class)
  if "Pclass" in df.columns and "Survived" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set2")
    plt.title(
        "Survival Count by Passenger Class", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Passenger Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "survival_by_class.png"))
    plt.close()
    print("Saved: survival_by_class.png")

  # 3. Correlation Heatmap
  plt.figure(figsize=(10, 8))
  numeric_df = df.select_dtypes(include=[np.number])
  if not numeric_df.empty:
    sns.heatmap(
        numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    print("Saved: correlation_heatmap.png")

  print(f"All plots successfully saved inside the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
  # Point to your actual cleaned dataset file
  dataset_path = "cleaned_dataset.parquet"

  df = load_data(dataset_path)
  if df is not None:
    plot_distributions(df)