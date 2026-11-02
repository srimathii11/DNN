import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample email dataset
emails = [
    "Win free money now",
    "Claim your lottery prize",
    "Get free offers today",
    "Congratulations you won",
    "Hello how are you",
    "Meeting is scheduled today",
    "Please check your email",
    "See you tomorrow"
]

# 1 = Spam, 0 = Not Spam
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Tokenize emails
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(emails)

X = tokenizer.texts_to_sequences(emails)
X = pad_sequences(X, maxlen=6)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# Build GRU model
model = Sequential([
    Embedding(1000, 16),
    GRU(16),
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

# Test model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Predict new email
email = ["Win a free prize now"]
seq = tokenizer.texts_to_sequences(email)
seq = pad_sequences(seq, maxlen=6)

prediction = model.predict(seq, verbose=0)

print("Email:", email[0])
print("Prediction:", "Spam" if prediction[0][0] > 0.5 else "Not Spam")