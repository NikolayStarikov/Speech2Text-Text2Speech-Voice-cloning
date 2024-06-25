#Training and Fine-Tuning 
# train.py

import tensorflow as tf
from data_loader import load_wav_files, load_text_files, preprocess_texts, preprocess_audio
from models import tacotron2_model

# Load and preprocess data
audio_files = load_wav_files('data/audio/')
text_files = load_text_files('data/text/')
text_sequences, word_index = preprocess_texts(text_files, max_length=50)
mel_spectrograms = preprocess_audio(audio_files)

# Convert data to tensors
text_sequences = tf.convert_to_tensor(text_sequences, dtype=tf.float32)
mel_spectrograms = tf.convert_to_tensor(mel_spectrograms, dtype=tf.float32)

# Define the Tacotron 2 model
tacotron2 = tacotron2_model((50,), len(word_index) + 1)

# Train the Tacotron 2 model
tacotron2.fit(text_sequences, mel_spectrograms, epochs=10, batch_size=8)

# Save the trained model
tacotron2.save('tacotron2_model.h5')
