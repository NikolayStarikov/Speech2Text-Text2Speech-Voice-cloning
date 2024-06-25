#Model Definitions
# models.py
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.models import Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformer encoder block.

    Args:
        inputs (tensor): Input tensor.
        head_size (int): Size of the attention heads.
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden layer size in feed forward network.
        dropout (float): Dropout rate.

    Returns:
        tensor: Output tensor.
    """
    # Self-attention
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed forward network
    x_ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x_ff = tf.keras.layers.Dropout(dropout)(x_ff)
    x_ff = tf.keras.layers.Dense(inputs.shape[-1])(x_ff)
    x = tf.keras.layers.Add()([x_ff, x])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def fastspeech_model(input_shape, vocab_size, num_layers=4, head_size=256, num_heads=4, ff_dim=1024, dropout=0.1):
    """
    Define the FastSpeech model architecture.

    Args:
        input_shape (tuple): Shape of the input sequences (excluding batch size).
        vocab_size (int): Size of the vocabulary.
        num_layers (int): Number of transformer layers.
        head_size (int): Size of the attention heads.
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden layer size in feed forward network.
        dropout (float): Dropout rate.

    Returns:
        Model: Compiled FastSpeech model.
    """
    inputs = Input(shape=input_shape)
    x = Embedding(vocab_size, head_size)(inputs)
    x = Dropout(dropout)(x)

    # Stack of transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Length regulator would be implemented here for prosody control

    # Output layer for mel spectrogram prediction
    outputs = Dense(80, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
