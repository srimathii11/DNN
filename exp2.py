import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Historical temperature data
data = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

# Prepare input and output
X = data[:-1].reshape(-1, 1)
y = data[1:]

# Create and train ANN
model = MLPRegressor(hidden_layer_sizes=(10, 5),
                     max_iter=5000, random_state=1)
model.fit(X, y)

# Predict temperature
predictions = model.predict(X)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, predictions))

print("Actual Temperature:", y)
print("Predicted Temperature:", np.round(predictions, 2))
print("RMSE:", round(rmse, 2))
print("Next Temperature:", round(model.predict([[34]])[0], 2))