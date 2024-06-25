# Dataset Preparation data_loader.py
# data_loader.py

import os
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_wav_files(data_path, sample_rate=22050):
    """
    Load all WAV files from the specified directory.

    Args:
        data_path (str): Path to the directory containing WAV files.
        sample_rate (int): Sample rate for loading audio files.

    Returns:
        list: List of loaded audio files as numpy arrays.
    """
    wav_files = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_path, file_name)
            wav, sr = librosa.load(file_path, sr=sample_rate)
            wav_files.append(wav)
    return wav_files

def preprocess_audio(audio_files, sample_rate=22050):
    """
    Convert audio files to mel spectrograms.

    Args:
        audio_files (list): List of audio files as numpy arrays.
        sample_rate (int): Sample rate for loading audio files.

    Returns:
        list: List of mel spectrograms.
    """
    mel_specs = []
    for audio in audio_files:
        mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=80)
        mel_specs.append(librosa.power_to_db(mel_spec, ref=np.max))
    return mel_specs

def pad_audio_sequences(audio_sequences, max_length):
    """
    Pad audio sequences to a uniform length.

    Args:
        audio_sequences (list): List of audio sequences.
        max_length (int): Maximum length to pad the sequences.

    Returns:
        np.ndarray: Padded audio sequences.
    """
    return pad_sequences(audio_sequences, maxlen=max_length, padding='post', dtype='float32')
