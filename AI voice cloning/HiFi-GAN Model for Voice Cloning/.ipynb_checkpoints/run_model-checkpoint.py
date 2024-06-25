#Running the Model
# run_model.py
# run_model.py

import numpy as np
import tensorflow as tf
import librosa
from models import hifigan_generator
from data_loader import preprocess_audio, pad_audio_sequences

# Load the trained HiFi-GAN generator model
generator = tf.keras.models.load_model('hifigan_generator_model.h5')

# Define the input audio (e.g., load a new audio file)
input_audio_path = 'data/new_audio.wav'
audio, sr = librosa.load(input_audio_path, sr=22050)

# Preprocess the input audio
mel_spectrogram = preprocess_audio([audio])[0]

# Pad the sequence
max_length = generator.input_shape[1]
mel_spectrogram = pad_audio_sequences([mel_spectrogram], max_length)

# Convert audio to tensor
mel_spectrogram = tf.convert_to_tensor(mel_spectrogram, dtype=tf.float32)

# Generate audio using the trained generator
generated_audio = generator.predict(mel_spectrogram)

# Print the generated audio (or save it to a file, etc.)
print(generated_audio)

# Optionally save the generated audio to a file
output_audio_path = 'data/generated_audio.wav'
librosa.output.write_wav(output_audio_path, generated_audio[0], sr=22050)

