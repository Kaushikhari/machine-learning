# CO₂ Emission Prediction using Linear Regression

This project predicts vehicle CO₂ emissions based on engine specifications and categorical features such as fuel type, vehicle class, and transmission type.

## 📂 Dataset
- Source: [Kaggle - CO2 Emissions Canada](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- Cleaned and preprocessed before applying the model

## 📌 Steps Covered

1. **Data Cleaning**
   - Dropped unnecessary columns
   - Handled categorical variables (transmission, fuel type, vehicle class)
2. **One-Hot Encoding**
   - Applied on categorical features
3. **Train-Test Split**
   - 80% training, 20% testing
4. **Model Building**
   - Linear Regression using `scikit-learn`
5. **Model Evaluation**
   - R² Score: ~0.99
   - RMSE, MAE reported
6. **Visualization**
   - Coefficient graph, residual plot using `matplotlib`
7. **Real-World Prediction**
   - Predict CO₂ emission for a custom vehicle input

## 🧠 Key Learnings
- Feature selection
- Importance of encoding
- Residual analysis and coefficient interpretation
- Using `.predict()` for real-world testing

## 🔧 Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📊 Example Output
R² Score: 0.9907
RMSE: 5.63
Predicted CO₂ Emission for given vehicle: 200.0 g/km