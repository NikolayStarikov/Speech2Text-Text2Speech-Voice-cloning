#models Definitions
# models.py
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Add, Activation
from tensorflow.keras.models import Model

def residual_block(x, filters, dilation_rate):
    """
    Define a residual block with dilated convolutions.

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of filters for the convolution layers.
        dilation_rate (int): Dilation rate for the dilated convolution.

    Returns:
        Tensor: Output tensor after applying the residual block.
    """
    conv_filter = Conv1D(filters, kernel_size=2, dilation_rate=dilation_rate, padding='causal', activation='tanh')(x)
    conv_gate = Conv1D(filters, kernel_size=2, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([conv_filter, conv_gate])
    x = Conv1D(filters, kernel_size=1, activation='tanh')(x)
    return x

def wavenet_model(input_shape):
    """
    Define the WaveNet model architecture.

    Args:
        input_shape (tuple): Shape of the input sequences (excluding batch size).

    Returns:
        Model: Compiled WaveNet model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    for rate in [1, 2, 4, 8, 16, 32]:
        x = residual_block(x, filters=32, dilation_rate=rate)
    
    outputs = Conv1D(1, kernel_size=1, activation='tanh')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
