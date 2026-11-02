import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample dataset
texts = [
    "I love this movie",
    "This is a wonderful film",
    "Amazing experience",
    "I enjoyed the movie",
    "I hate this movie",
    "Very bad experience",
    "This film is boring",
    "Worst movie ever"
]

# 1 = Positive, 0 = Negative
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=6)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# Build LSTM model
model = Sequential([
    Embedding(1000, 16),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=30, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Predict sentiment
review = ["This movie is amazing"]
seq = tokenizer.texts_to_sequences(review)
seq = pad_sequences(seq, maxlen=6)

prediction = model.predict(seq, verbose=0)

print("Review:", review[0])
print("Sentiment:", "Positive" if prediction[0][0] > 0.5 else "Negative")