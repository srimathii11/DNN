from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()
X = digits.data / 16.0
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train DNN model
model = MLPClassifier(hidden_layer_sizes=(64, 32),
                      max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Actual Digit:", y_test[0])
print("Recognized Digit:", predictions[0])