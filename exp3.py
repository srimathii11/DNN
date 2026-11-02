import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Sample sensor dataset
X = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5],
    [5, 6], [6, 7], [7, 8], [8, 9],
    [9, 10], [10, 11], [11, 12], [12, 13]
])

# Activity labels: 0-Walking, 1-Running, 2-Sitting
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create ANN model
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict activity
pred = model.predict(X_test)

# Display results
activities = ['Walking', 'Running', 'Sitting']

print("Actual Activities:", [activities[i] for i in y_test])
print("Predicted Activities:", [activities[i] for i in pred])
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")