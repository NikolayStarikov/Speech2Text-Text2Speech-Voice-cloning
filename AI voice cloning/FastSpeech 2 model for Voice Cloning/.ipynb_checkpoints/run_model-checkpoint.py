#Running the Model
# run_model.py
# run_model.py

import numpy as np
import tensorflow as tf
from models import fastspeech_model
from data_loader import preprocess_texts

# Load the trained FastSpeech model
model = tf.keras.models.load_model('fastspeech_model.h5')

# Define the input text
input_text = "Hello, this is a test."

# Preprocess the input text
text_sequences, _ = preprocess_texts([input_text], max_length=50)

# Convert text to tensor
text_sequences = tf.convert_to_tensor(text_sequences, dtype=tf.float32)

# Generate mel spectrogram using the trained model
mel_spectrogram = model.predict(text_sequences)

# Print the mel spectrogram
print(mel_spectrogram)
