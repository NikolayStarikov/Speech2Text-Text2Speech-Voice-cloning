#models Definitions
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, GRU, LSTM, Conv1D, BatchNormalization, Bidirectional, Add
from tensorflow.keras.models import Model

def tacotron2_model(input_shape, vocab_size):
    """
    Define the Tacotron 2 model architecture.

    Args:
        input_shape (tuple): Shape of the input sequences (excluding batch size).
        vocab_size (int): Size of the vocabulary.

    Returns:
        Model: Compiled Tacotron 2 model.
    """
    # Input layer for text sequences
    inputs = Input(shape=input_shape)

    # Embedding layer to convert text into dense vector representations
    embeddings = Embedding(vocab_size, 256)(inputs)

    # Prenet: two fully connected layers with ReLU activation
    prenet = tf.keras.layers.Dense(256, activation='relu')(embeddings)
    prenet = tf.keras.layers.Dense(256, activation='relu')(prenet)

    # Encoder: three convolutional layers followed by a bidirectional LSTM
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(prenet)
    encoder = BatchNormalization()(encoder)
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Bidirectional(LSTM(256, return_sequences=True))(encoder)

    # Attention mechanism
    attention = tf.keras.layers.Attention()([encoder, encoder])

    # Decoder: two LSTM layers
    decoder_input = Add()([attention, encoder])
    decoder = LSTM(512, return_sequences=True)(decoder_input)
    decoder = LSTM(512, return_sequences=True)(decoder)

    # Output layer for mel spectrograms
    outputs = Dense(80, activation='linear')(decoder)

    # Create and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
