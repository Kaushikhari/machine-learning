import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Hours': [2, 4, 6, 8, 10],
    'Attendance': [60, 70, 80, 90, 100],
    'Score': [50, 60, 65, 75, 85]
}

df = pd.DataFrame(data)
print(df)

w1 = 2
w2 = 0.5
b = 10

df['Predicted_Score'] = (w1 * df['Hours']) + (w2 * df['Attendance']) + b
df['Error'] = df['Predicted_Score'] - df['Score']
df['Error_square'] = df['Error'] ** 2
mse = df['Error_square'].mean()
print("Cost Function:", mse)
print(df)

plt.scatter(df['Hours'], df['Score'], color='red', label='Actual Scores', marker='o')
plt.scatter(df['Hours'], df['Predicted_Score'], color='blue', label='Predicted Scores', marker='x')

plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Actual vs Predicted Scores')
plt.legend()
plt.grid(True)
plt.show()