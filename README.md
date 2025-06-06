# 🧹 Titanic Dataset - Data Cleaning & Preprocessing

This repository contains a Python script that performs end-to-end data cleaning and preprocessing on the **Titanic Dataset**. It demonstrates essential data preparation steps commonly used in data science and machine learning pipelines, including missing value handling, encoding, normalization, and outlier detection.

---

## 📁 Dataset

The dataset used is the classic **Titanic survival dataset**. It contains passenger details such as age, gender, class, fare, and survival status.

> 🔸 **File used**: `Titanic-Dataset.csv` (make sure this is in the same directory as the script)

---

## 🔧 Features & Workflow

The script follows these key steps:

1. **Import and explore the dataset**
   - View data types and check for null values

2. **Handle missing values**
   - Use mean, median, or mode imputation based on the feature type

3. **Convert categorical features to numeric**
   - Apply `LabelEncoder` from `scikit-learn` for encoding

4. **Normalize numerical features**
   - Use `StandardScaler` to standardize features (mean = 0, std = 1)

5. **Visualize and remove outliers**
   - Plot boxplots using `seaborn` and remove outliers using the IQR method

6. **Preview the cleaned dataset**
   - Output the transformed dataset after all cleaning steps

---

## 📊 Sample Output

- Console output showing:
  - Missing values before and after cleaning
  - Dataset summary
  - Encoded and scaled feature values

- Plots:
  - Boxplots of numerical features before and after outlier removal

---

## 📦 Dependencies

Make sure the following Python libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
