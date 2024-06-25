#Model Definitions
# models.py
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, LeakyReLU, BatchNormalization, Add, Activation
from tensorflow.keras.models import Model

def hifigan_generator(input_shape):
    """
    Define the HiFi-GAN generator model architecture.

    Args:
        input_shape (tuple): Shape of the input sequences (excluding batch size).

    Returns:
        Model: Compiled HiFi-GAN generator model.
    """
    inputs = Input(shape=input_shape)

    x = Conv1DTranspose(512, 16, 8, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv1DTranspose(256, 16, 8, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv1DTranspose(128, 16, 8, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv1DTranspose(1, 16, 8, padding='same')(x)
    outputs = Activation('tanh')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def hifigan_discriminator(input_shape):
    """
    Define the HiFi-GAN discriminator model architecture.

    Args:
        input_shape (tuple): Shape of the input sequences (excluding batch size).

    Returns:
        Model: Compiled HiFi-GAN discriminator model.
    """
    inputs = Input(shape=input_shape)

    x = Conv1D(128, 4, 2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(256, 4, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(512, 4, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(1, 4, 2, padding='same')(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
