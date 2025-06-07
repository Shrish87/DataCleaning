import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import math

# Load dataset
try:
    df = pd.read_csv("Titanic-Dataset.xls")
    print("✅ Dataset loaded successfully.\n")
except FileNotFoundError:
    print("❌ File not found. Ensure 'Titanic-Dataset.xls' is in the same folder.")
    exit()

# 1. Basic info and null values
print("🔹 BASIC INFO:")
print(df.info())
print("\n🔹 MISSING VALUES:")
print(df.isnull().sum())

# 2. Handle missing values
# Fill numeric columns with mean or median
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Fare' in df.columns:
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

# Fill categorical columns with mode
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\n✅ Missing values handled.")

# 3. Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\n✅ Categorical variables encoded.")

# 4. Normalize/Standardize numerical features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Numerical features standardized.")

# 5. Visualize outliers using boxplots
print("\n🔹 OUTLIER VISUALIZATION:")
num_cols = len(numeric_cols)
rows = math.ceil(num_cols / 3)

plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(numeric_cols):
    plt.subplot(rows, 3, i + 1)
    sns.boxplot(y=df[col], color='orange')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 5. Remove outliers using IQR method
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df_cleaned = remove_outliers_iqr(df.copy(), numeric_cols)

print(f"\n✅ Outliers removed. Remaining rows: {df_cleaned.shape[0]}")

# Final preview
print("\n🔹 Cleaned Dataset Preview:")
print(df_cleaned.head())
