import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([0, 0, 0, 1, 1])
m = len(Y)


def sigmoid(z):
    return 1/(1+np.exp(-z))


w = 0
b = 0
alpha = 0.0001
epochs = 10000

for epoch in range(epochs):
    Z = (w * X) + b
    y_pred = sigmoid(Z)   # sigmoid activation
    # Compute Gradients
    dw = (1 / m) * np.dot((y_pred - Y), X)
    db = (1 / m) * np.sum(y_pred - Y)
    # update weights
    w -= alpha * dw
    b -= alpha * db
    # Cost Function
    cost = -(1 / m) * np.sum(Y * np.log(y_pred + 1e-8) + (1 - Y) * np.log(1 - y_pred + 1e-8))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")


# Prediction
def predict(xi):
    prob = sigmoid(w * xi + b)
    return 1 if prob >= 0.5 else 0


print("\nPredictions:")
for xi in X:
    print(f"x = {xi}, Predicted = {predict(xi)}")
