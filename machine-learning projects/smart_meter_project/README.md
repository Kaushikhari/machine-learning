# Smart Meter Power Consumption Prediction using Linear Regression and Artificial Neural Networks 🔌⚡

This project uses machine learning to predict household electricity consumption (Global Active Power in kW) based on voltage, sub-metering, and time features. The goal is to help utilities and smart grid companies understand consumption patterns in real-time.

---

## 📊 Models Used
- **Linear Regression** – Fast baseline with good performance on linear data
- **Artificial Neural Network (ANN)** – Captures nonlinear usage patterns more effectively

---

## 🧠 Input Features
- Voltage
- Global Reactive Power
- Global Intensity
- Sub-metering 1, 2, 3
- Hour, Day of Week, Month, Weekend Indicator

---

## 📈 Evaluation Metrics (on test data)
| Metric | Linear Regression | ANN |
|--------|-------------------|-----|
| R²     | 0.998             | 0.9988 |
| RMSE   | 0.0359            | 0.0355 |
| MAE    | 0.0217            | 0.0215 |
| MAPE   | -                 | 4.61% |

---

## ✅ Real-World Validation
Tested on actual household data for **June 1, 2010**  
📊 Visual shows high agreement between actual and predicted power usage

---

## 🔧 Future Work
- Introduce **LSTM** for time-series forecasting
- Add **anomaly detection** using Autoencoders or Isolation Forest
- Experiment with **real-time Edge AI deployment**

---

## 🗂️ Dataset Info
- Over 2 million records (Dec 2006–Nov 2010)
- 1-minute sampling rate
- Time-series format
- This project uses the [Household Electric Power Consumption dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) from Kaggle.
- Please download it from the link above and place the file `household_power_consumption.txt` inside the `data/` folder.