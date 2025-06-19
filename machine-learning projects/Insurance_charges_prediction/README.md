# Insurance Charges Prediction 📊

This project builds a simple **Linear Regression model** to predict insurance charges based on a person's demographic and health-related information, such as age, gender, region, BMI, and smoking habits.

The project demonstrates **data preprocessing**, **feature engineering**, **model training**, and **evaluation** in a clean and reproducible workflow.

---

## 🔍 Project Overview

- **Dataset**: `insurance.csv` (contains age, sex, BMI, children, smoker status, region, and charges)
- **Goal**: Predict the `charges` (insurance cost) using a regression model.
- **Approach**:
  - Clean and preprocess the data.
  - Encode categorical variables.
  - Handle anomalies like missing values and text formats.
  - Train a Linear Regression model.
  - Evaluate the model performance.
  - Predict on new validation data (`validation_dataset.csv`).

---

## 📁 Files in the Project

| File Name                  | Description |
|---------------------------|-------------|
| `insurance.csv`           | Original training dataset. |
| `validation_dataset.csv`  | New data for prediction after training. |
| `insurance_charges_prediction.py` | Python script for preprocessing, model training, and prediction. |
| `README.md`               | Project documentation. |

---

## 🧪 Technologies & Libraries

- Python
- Pandas
- NumPy
- scikit-learn

---

## 🧹 Preprocessing Steps

1. Standardized column values (e.g., gender and smoker).
2. Removed unwanted characters (e.g., `$` from charges).
3. Converted boolean and categorical columns using `get_dummies`.
4. Dropped irrelevant or redundant columns.
5. Ensured no missing values remain.

---

## ⚙️ Model Training

- Used `LinearRegression` from `sklearn`.
- Split the dataset into **train** and **test** sets (80/20).
- Evaluated using:
  - **R² Score**
  - **RMSE (Root Mean Squared Error)**

---


