import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
train = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(64, 64),
    batch_size=32
)

test = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(64, 64),
    batch_size=32
)

# Create CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train, epochs=5)

# Evaluate model
loss, accuracy = model.evaluate(test)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# Predict sample image
image = tf.keras.utils.load_img(
    "sample.jpg", target_size=(64, 64)
)
image = tf.keras.utils.img_to_array(image) / 255.0
image = tf.expand_dims(image, axis=0)

prediction = model.predict(image)
classes = ["Diseased", "Healthy"]

print("Predicted Disease Status:", classes[prediction.argmax()])