# OIBSIP-DataAnalytics-L1-Task-3-Cleaning-Data-
The objective of this task is to demonstrate professional-level data cleaning skills by taking a messy and inconsistent dataset and systematically transforming it into a clean, consistent, and analysis-ready dataset.
CODE:
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

df = pd.read_csv("Titanic-Dataset.csv")

print("Original Dataset Shape:", df.shape)
print("\nOriginal Data Types:")
print(df.dtypes)

print("\nOriginal Missing Values:")
print(df.isnull().sum())

print("\nOriginal Duplicate Rows:", df.duplicated().sum())

print("\nOriginal Statistical Summary:")
display(df.describe(include="all").T)

# Save original information for before-vs-after comparison
rows_before = len(df)
duplicates_before = df.duplicated().sum()
nulls_before = df.isnull().sum().sum()
dtypes_before = df.dtypes.copy()

# Check value range anomalies
print("\nValue Range Anomalies:")
print("Negative Age:", (df["Age"] < 0).sum())
print("Negative Fare:", (df["Fare"] < 0).sum())
print("Invalid Survived:", (~df["Survived"].isin([0, 1])).sum())
print("Invalid Pclass:", (~df["Pclass"].isin([1, 2, 3])).sum())
print("Negative SibSp:", (df["SibSp"] < 0).sum())
print("Negative Parch:", (df["Parch"] < 0).sum())

# Create a copy for cleaning
df_clean = df.copy()

# Missing data handling
# Age: median is used because it is numerical and median is less affected by outliers.
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Cabin: "Unknown" preserves the records without falsely creating a cabin value.
df_clean["Cabin"] = df_clean["Cabin"].fillna("Unknown")

# Embarked: mode is appropriate because it is categorical with very few missing values.
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# Standardise text columns
text_columns = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]

for col in text_columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Standardise Sex values
df_clean["Sex"] = df_clean["Sex"].str.lower().map({
    "male": "Male",
    "m": "Male",
    "female": "Female",
    "f": "Female"
})

# Standardise Embarked values
df_clean["Embarked"] = df_clean["Embarked"].str.upper()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Outlier detection using IQR
numeric_columns = ["Age", "SibSp", "Parch", "Fare"]

outlier_report = []

for col in numeric_columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[col] < lower_bound) |
        (df_clean[col] > upper_bound)
    ]

    outlier_report.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": len(outliers)
    })

outlier_report = pd.DataFrame(outlier_report)

print("\nOutlier Report:")
display(outlier_report)

# Outliers are retained because they can represent legitimate passengers.
# For example, a high Fare or large family size can be a valid observation.

# Correct data types
df_clean["PassengerId"] = df_clean["PassengerId"].astype("string")
df_clean["Survived"] = pd.to_numeric(df_clean["Survived"], errors="coerce").astype("Int64")
df_clean["Pclass"] = pd.to_numeric(df_clean["Pclass"], errors="coerce").astype("Int64")
df_clean["Name"] = df_clean["Name"].astype("string")
df_clean["Sex"] = df_clean["Sex"].astype("string")
df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce").astype(float)
df_clean["SibSp"] = pd.to_numeric(df_clean["SibSp"], errors="coerce").astype("Int64")
df_clean["Parch"] = pd.to_numeric(df_clean["Parch"], errors="coerce").astype("Int64")
df_clean["Ticket"] = df_clean["Ticket"].astype("string")
df_clean["Fare"] = pd.to_numeric(df_clean["Fare"], errors="coerce").astype(float)
df_clean["Cabin"] = df_clean["Cabin"].astype("string")
df_clean["Embarked"] = df_clean["Embarked"].astype("string")

# Check final missing values
nulls_after = df_clean.isnull().sum().sum()

# Check final duplicates
duplicates_after = df_clean.duplicated().sum()

# Check expected data types
expected_dtypes = {
    "PassengerId": "string",
    "Survived": "Int64",
    "Pclass": "Int64",
    "Name": "string",
    "Sex": "string",
    "Age": "float64",
    "SibSp": "Int64",
    "Parch": "Int64",
    "Ticket": "string",
    "Fare": "float64",
    "Cabin": "string",
    "Embarked": "string"
}

dtype_errors_before = sum(
    str(df[col].dtype) != expected_dtypes[col]
    for col in expected_dtypes
)

dtype_errors_after = sum(
    str(df_clean[col].dtype) != expected_dtypes[col]
    for col in expected_dtypes
)

rows_after = len(df_clean)

# Before vs After summary
summary = pd.DataFrame({
    "Metric": [
        "Total Null Values",
        "Duplicate Rows",
        "Row Count",
        "Data Type Errors"
    ],
    "Before Cleaning": [
        nulls_before,
        duplicates_before,
        rows_before,
        dtype_errors_before
    ],
    "After Cleaning": [
        nulls_after,
        duplicates_after,
        rows_after,
        dtype_errors_after
    ]
})

print("\nBefore vs After Cleaning Summary:")
display(summary)

# Column-level comparison
comparison = pd.DataFrame({
    "Column": df.columns,
    "Null_Before": df.isnull().sum().values,
    "Null_After": df_clean.isnull().sum().values,
    "Dtype_Before": df.dtypes.astype(str).values,
    "Dtype_After": df_clean.dtypes.astype(str).values
})

print("\nColumn-Level Comparison:")
display(comparison)

# Final dataset preview
print("\nCleaned Dataset:")
display(df_clean.head())

print("\nFinal Data Types:")
print(df_clean.dtypes)

print("\nFinal Missing Values:")
print(df_clean.isnull().sum())

print("\nFinal Duplicate Rows:", df_clean.duplicated().sum())

# Save cleaned dataset
output_file = "Titanic-Dataset-Cleaned.csv"
df_clean.to_csv(output_file, index=False)

print("\nCleaned dataset saved as:", output_file)
print("Final Dataset Shape:", df_clean.shape)
