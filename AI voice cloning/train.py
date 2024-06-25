#training and Fine-Tuning
# train.py

import tensorflow as tf
from data_loader import load_wav_files, load_text_files, preprocess_texts, preprocess_audio
from models import tacotron2_model, wavenet_model, fastspeech2_model, hifigan_generator, hifigan_discriminator

# Load and preprocess data
audio_files = load_wav_files('data/audio/')
text_files = load_text_files('data/text/')
text_sequences, word_index = preprocess_texts(text_files, max_length=50)
mel_spectrograms = preprocess_audio(audio_files)

# Convert to tensor
text_sequences = tf.convert_to_tensor(text_sequences, dtype=tf.float32)
mel_spectrograms = tf.convert_to_tensor(mel_spectrograms, dtype=tf.float32)

# Tacotron 2 training
tacotron2 = tacotron2_model((50,), len(word_index) + 1)
tacotron2.compile(optimizer='adam', loss='mse')
tacotron2.fit(text_sequences, mel_spectrograms, epochs=10)

# WaveNet training (assuming mel spectrograms to audio mapping)
wavenet = wavenet_model((None, 80))
wavenet.compile(optimizer='adam', loss='mse')
wavenet.fit(mel_spectrograms, audio_files, epochs=10)

# FastSpeech 2 training
fastspeech2 = fastspeech2_model((50,), len(word_index) + 1)
fastspeech2.compile(optimizer='adam', loss='mse')
fastspeech2.fit(text_sequences, mel_spectrograms, epochs=10)

# HiFi-GAN training
hifi_g = hifigan_generator((None, 80))
hifi_d = hifigan_discriminator((None, 1))
hifi_g.compile(optimizer='adam', loss='mse')
hifi_d.compile(optimizer='adam', loss='binary_crossentropy')

# Dummy training loop for GAN
for epoch in range(10):
    with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
        fake_audio = hifi_g(mel_spectrograms)
        real_output = hifi_d(audio_files)
        fake_output = hifi_d(fake_audio)
        
        g_loss = tf.keras.losses.mean_squared_error(fake_audio, audio_files)
        d_loss = tf.keras.losses.binary_crossentropy(real_output, tf.ones_like(real_output)) + \
                 tf.keras.losses.binary_crossentropy(fake_output, tf.zeros_like(fake_output))
    
    grads_g = tape_g.gradient(g_loss, hifi_g.trainable_variables)
    grads_d = tape_d.gradient(d_loss, hifi_d.trainable_variables)
    
    hifi_g.optimizer.apply_gradients(zip(grads_g, hifi_g.trainable_variables))
    hifi_d.optimizer.apply_gradients(zip(grads_d, hifi_d.trainable_variables))

    print(f"Epoch {epoch+1}, Generator Loss: {g_loss.numpy().mean()}, Discriminator Loss: {d_loss.numpy().mean()}")
