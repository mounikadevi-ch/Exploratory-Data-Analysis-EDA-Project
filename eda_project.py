# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("students.csv")  # Change filename as needed

# Display basic information
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Visualization
# -----------------------------

# Histogram
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.savefig("histograms.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Pair Plot
sns.pairplot(df.select_dtypes(include=np.number))
plt.savefig("pairplot.png")
plt.show()

# Boxplots
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# -----------------------------
# Correlation Analysis
# -----------------------------
correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Strong Correlations
print("\nStrong Correlations (>0.7):")

for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        corr = correlation_matrix.loc[row, col]

        if row != col and abs(corr) > 0.7:
            print(f"{row} and {col}: {corr:.2f}")

# -----------------------------
# Generate Report
# -----------------------------
with open("EDA_Report.txt", "w") as report:

    report.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("="*40 + "\n\n")

    report.write(f"Dataset Shape: {df.shape}\n\n")

    report.write("Missing Values:\n")
    report.write(str(df.isnull().sum()))
    report.write("\n\n")

    report.write("Statistical Summary:\n")
    report.write(str(df.describe()))
    report.write("\n\n")

    report.write("Correlation Matrix:\n")
    report.write(str(correlation_matrix))

print("\nEDA Completed Successfully!")
print("Report saved as EDA_Report.txt")