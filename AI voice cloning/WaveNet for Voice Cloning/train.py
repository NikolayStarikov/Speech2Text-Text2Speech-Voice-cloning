#Training and Fine-Tuning 
# train.py

# train.py

import tensorflow as tf
from data_loader import load_wav_files, preprocess_audio, pad_audio_sequences
from models import wavenet_model

# Load and preprocess data
audio_files = load_wav_files('data/audio/')
mel_spectrograms = preprocess_audio(audio_files)

# Pad sequences to uniform length
max_length = max(len(mel) for mel in mel_spectrograms)
mel_spectrograms = pad_audio_sequences(mel_spectrograms, max_length)

# Convert data to tensors
mel_spectrograms = tf.convert_to_tensor(mel_spectrograms, dtype=tf.float32)

# Define the WaveNet model
wavenet = wavenet_model((max_length, 80))

# Train the WaveNet model
wavenet.fit(mel_spectrograms, mel_spectrograms, epochs=10, batch_size=8)

# Save the trained model
wavenet.save('wavenet_model.h5')

