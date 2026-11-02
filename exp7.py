import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample transaction data
X = np.array([
    [100, 2], [120, 3], [110, 2], [5000, 20],
    [130, 3], [140, 2], [6000, 25], [150, 3],
    [160, 2], [7000, 30]
])

# 0 = Normal, 1 = Fraud
y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape for LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create LSTM model
model = Sequential([
    LSTM(16, input_shape=(1, 2)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=20, verbose=0)

# Predict transactions
pred = (model.predict(X_test) > 0.5).astype(int)

print("Actual:", y_test)
print("Predicted:", pred.flatten())
print("Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1] * 100)