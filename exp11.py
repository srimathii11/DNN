import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras import layers, models

# Load clean audio
audio, sr = librosa.load("clean.wav", duration=5)

# Add noise
noisy = audio + 0.05 * np.random.randn(len(audio))

# Prepare training data
X = noisy[:len(noisy)//512*512].reshape(-1, 512)
Y = audio[:len(audio)//512*512].reshape(-1, 512)

# Build Autoencoder
model = models.Sequential([
    layers.Input(shape=(512,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='linear')
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, Y, epochs=20, verbose=1)

# Denoise audio
denoised = model.predict(X)
denoised = denoised.flatten()

# Save output
sf.write("denoised.wav", denoised, sr)

print("Audio denoising completed!")
print("Output saved as denoised.wav")