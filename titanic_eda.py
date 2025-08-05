# Titanic EDA Analysis (Python Script)
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ========================================
# 1. Load Dataset
# ========================================
df = pd.read_csv("Titanic-Dataset.csv")  # Replace with actual path
print("Shape of dataset:", df.shape)

# ========================================
# 2. Basic Info
# ========================================
print("\n--- Basic Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# ========================================
# 3. Summary Statistics
# ========================================
print("\n--- Summary Statistics ---")
print(df.describe(include="all"))

print("\nUnique values per column:\n", df.nunique())

# ========================================
# 4. Histograms for Numeric Features
# ========================================
numeric_features = ["Age", "Fare"]

for col in numeric_features:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"screenshots/{col}_histogram.png")
    plt.close()

# ========================================
# 5. Boxplots for Outliers
# ========================================
for col in numeric_features:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"screenshots/{col}_boxplot.png")
    plt.close()

# ========================================
# 6. Countplots for Categorical Features
# ========================================
categorical_features = ["Sex", "Pclass", "Embarked"]

for col in categorical_features:
    plt.figure()
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.savefig(f"screenshots/{col}_countplot.png")
    plt.close()

# ========================================
# 7. Survival Analysis
# ========================================
plt.figure()
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("screenshots/survival_count.png")
plt.close()

plt.figure()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.savefig("screenshots/survival_by_gender.png")
plt.close()

plt.figure()
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.savefig("screenshots/survival_by_class.png")
plt.close()

# ========================================
# 8. Correlation Matrix
# ========================================
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

# ========================================
# 9. Insights & Observations
# ========================================
print("\n--- Key Insights ---")
print("1. Missing values in Age, Cabin, Embarked.")
print("2. Fare is highly skewed and has outliers.")
print("3. Females have higher survival rate than males.")
print("4. Higher class passengers (Pclass=1) survived more.")
print("5. Age distribution roughly normal but with some skewness.")

print("\nEDA Completed. All plots saved in 'screenshots/' folder.")
