#Model Definitions
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, GRU, LSTM, Conv1D, BatchNormalization, Bidirectional, Add
from tensorflow.keras.models import Model

def tacotron2_model(input_shape, vocab_size):
    inputs = Input(shape=input_shape)
    embeddings = Embedding(vocab_size, 256)(inputs)
    prenet = tf.keras.layers.Dense(256, activation='relu')(embeddings)
    prenet = tf.keras.layers.Dense(256, activation='relu')(prenet)
    
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(prenet)
    encoder = BatchNormalization()(encoder)
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv1D(512, kernel_size=5, padding='same', activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Bidirectional(LSTM(256, return_sequences=True))(encoder)
    
    attention = tf.keras.layers.Attention()([encoder, encoder])
    
    decoder_input = Add()([attention, encoder])
    decoder = LSTM(512, return_sequences=True)(decoder_input)
    decoder = LSTM(512, return_sequences=True)(decoder)
    
    outputs = Dense(80, activation='linear')(decoder)
    
    model = Model(inputs, outputs)
    return model

def wavenet_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    for rate in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(32, kernel_size=2, dilation_rate=rate, padding='causal', activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, activation='tanh')(x)
    model = Model(inputs, outputs)
    return model

def fastspeech2_model(input_shape, vocab_size):
    inputs = Input(shape=input_shape)
    embeddings = Embedding(vocab_size, 256)(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(embeddings)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    # Length regulator, pitch and energy predictors can be added here
    
    decoder = Bidirectional(LSTM(256, return_sequences=True))(x)
    outputs = Dense(80, activation='linear')(decoder)
    
    model = Model(inputs, outputs)
    return model

def hifigan_generator(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv1DTranspose(256, 16, 8, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1DTranspose(128, 16, 8, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1DTranspose(1, 16, 8, padding='same')(x)
    outputs = tf.keras.layers.Activation('tanh')(x)
    model = Model(inputs, outputs)
    return model

def hifigan_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(128, 4, 2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv1D(256, 4, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv1D(1, 4, 2, padding='same')(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x)
    model = Model(inputs, outputs)
    return model
