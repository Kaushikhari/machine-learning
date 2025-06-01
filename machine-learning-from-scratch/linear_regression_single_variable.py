import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('students_score.txt')
print(df.head())
w = 9
b = 2
y = df['Score']
total_error = 0
m = len(df)
for i in range(len(df)):
    x = df['Hours'][i]
    y_pred = (w*x)+b
    y_actual = y[i]
    error = y_pred - y_actual
    error_squared = error ** 2
    total_error += error_squared
    print(f"For x = {x}, predicted y = {y_pred},actual y = {y_actual}, error = {error}, error_squared = {error_squared}")


mean_squared_error = total_error/m
print("Cost Function: ", mean_squared_error)

plt.xlabel('Hours')
plt.ylabel('Score')
plt.title('Actual Value')
plt.scatter(df.Hours, df.Score, color='red', marker='+',label='Actual')

predicted_scores = [(w * x) + b for x in df['Hours']]
plt.plot(df['Hours'], predicted_scores, color='blue', label='Predicted Line')

plt.legend()
plt.grid(True)
plt.show()