# Energy Consumption Prediction Using XGBoost  

This project focuses on forecasting hourly power consumption (MW) for the PJME region using machine learning. The dataset contains hourly energy usage data from **2002 to 2018**.

## 📊 Project Goal  
To predict future electricity demand based on past consumption trends using time-based features. This helps utility companies manage power generation efficiently.

## ✅ Methods Used  

### 📌 Data Preprocessing  
- Converted `'Datetime'` column to datetime format  
- Set `Datetime` as index  
- Handled **duplicate timestamps** by averaging using `groupby().mean()`

### 📌 Feature Engineering  
Extracted useful time-based features from the Datetime index:
- `hour`, `dayofweek`, `month`, `quarter`, `year`, `dayofyear`

### 📌 Train/Test Split  
- **Training Set:** Data from 2002–2015  
- **Testing Set:** Data from 2015–2018  

### 📌 Model Used  
- **XGBoost Regressor**  
  - `n_estimators = 1000`  
  - `learning_rate = 0.01`  
  - `max_depth = 3`  
  - `early_stopping_rounds = 50`  

### 📌 Evaluation Metrics  
- Root Mean Squared Error (RMSE)  
- R² Score  
