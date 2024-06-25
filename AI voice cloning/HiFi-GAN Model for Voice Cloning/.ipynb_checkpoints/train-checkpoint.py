#training and Fine-Tuning
# train.py
# train.py

import tensorflow as tf
from data_loader import load_wav_files, preprocess_audio, pad_audio_sequences
from models import hifigan_generator, hifigan_discriminator

# Load and preprocess data
audio_files = load_wav_files('data/audio/')
mel_spectrograms = preprocess_audio(audio_files)

# Pad sequences to uniform length
max_length = max(len(mel) for mel in mel_spectrograms)
mel_spectrograms = pad_audio_sequences(mel_spectrograms, max_length)
audio_files = pad_audio_sequences(audio_files, max_length)

# Convert data to tensors
mel_spectrograms = tf.convert_to_tensor(mel_spectrograms, dtype=tf.float32)
audio_files = tf.convert_to_tensor(audio_files, dtype=tf.float32)

# Define the HiFi-GAN generator and discriminator models
generator = hifigan_generator((max_length, 80))
discriminator = hifigan_discriminator((max_length, 1))

# Compile the models
generator.compile(optimizer='adam', loss='mse')
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop for GAN
epochs = 10
batch_size = 8

for epoch in range(epochs):
    for i in range(0, len(mel_spectrograms), batch_size):
        real_audio = audio_files[i:i+batch_size]
        mel_spec_batch = mel_spectrograms[i:i+batch_size]

        # Generate fake audio
        fake_audio = generator.predict(mel_spec_batch)

        # Train discriminator
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_audio, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_audio, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Train generator
        misleading_labels = tf.ones((batch_size, 1))
        g_loss = discriminator.train_on_batch(fake_audio, misleading_labels)

    print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")

# Save the trained models
generator.save('hifigan_generator_model.h5')
discriminator.save('hifigan_discriminator_model.h5')
