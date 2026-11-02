import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# Load cartoon dataset
data = tf.keras.utils.image_dataset_from_directory(
    "cartoon_dataset",
    image_size=(64, 64),
    batch_size=32,
    label_mode=None
)

data = data.map(lambda x: (x / 127.5) - 1)

# Generator
generator = Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(8*8*128),
    layers.Reshape((8, 8, 128)),
    layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                           activation='relu'),
    layers.Conv2DTranspose(32, 4, strides=2, padding='same',
                           activation='relu'),
    layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                           activation='tanh')
])

# Discriminator
discriminator = Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, 4, strides=2, padding='same'),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Compile models
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train GAN
for epoch in range(5):
    for real_images in data:
        size = tf.shape(real_images)[0]

        noise = tf.random.normal((size, 100))
        fake_images = generator(noise)

        discriminator.trainable = True
        discriminator.train_on_batch(real_images, tf.ones((size, 1)))
        discriminator.train_on_batch(fake_images, tf.zeros((size, 1)))

        discriminator.trainable = False
        noise = tf.random.normal((size, 100))
        gan.train_on_batch(noise, tf.ones((size, 1)))

    print("Epoch:", epoch + 1, "Completed")

# Generate cartoon image
noise = tf.random.normal((1, 100))
image = generator.predict(noise, verbose=0)[0]

plt.imshow((image + 1) / 2)
plt.axis('off')
plt.title("Generated Cartoon Character")
plt.show()