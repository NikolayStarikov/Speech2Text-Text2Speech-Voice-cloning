#Running the Model
# run_model.py
# run_model.py

import numpy as np
import tensorflow as tf
from models import wavenet_model
from data_loader import preprocess_audio, pad_audio_sequences

# Load the trained WaveNet model
model = tf.keras.models.load_model('wavenet_model.h5')

# Define the input audio (e.g., load a new audio file)
input_audio_path = 'data/new_audio.wav'
audio, sr = librosa.load(input_audio_path, sr=22050)

# Preprocess the input audio
mel_spectrogram = preprocess_audio([audio])[0]

# Pad the sequence
max_length = model.input_shape[1]
mel_spectrogram = pad_audio_sequences([mel_spectrogram], max_length)

# Convert audio to tensor
mel_spectrogram = tf.convert_to_tensor(mel_spectrogram, dtype=tf.float32)

# Generate audio using the trained model
generated_audio = model.predict(mel_spectrogram)

# Print the generated audio (or save it to a file, etc.)
print(generated_audio)
