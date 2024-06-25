#Dataset Preparation
# data_loader.py

import os
import numpy as np
import librosa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_wav_files(data_path, sample_rate=22050):
    wav_files = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_path, file_name)
            wav, sr = librosa.load(file_path, sr=sample_rate)
            wav_files.append(wav)
    return wav_files

def load_text_files(data_path):
    text_files = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as file:
                text_files.append(file.read().strip())
    return text_files

def preprocess_texts(texts, max_length):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer.word_index

def preprocess_audio(audio_files, sample_rate=22050):
    mel_specs = []
    for audio in audio_files:
        mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=80)
        mel_specs.append(librosa.power_to_db(mel_spec, ref=np.max))
    return mel_specs
