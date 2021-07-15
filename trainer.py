import os
import pickle
import numpy as np
from types import SimpleNamespace

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MAX_LEN = 12
IN_DIM = 3

global_selected = [
    '000725',
    '002714',
    '300059',
    '510300',
    '510500',
    '600036',
    '600867',
    '600999',
]

data_dir = 'data'


def trainer(code, time_range='2021-04'):
    with open(os.path.join(data_dir, ''.join([code, '_', time_range, '.pkl'])), 'rb') as f:
        d = pickle.load(f)

    # keys: 'enc_in', 'dec_in', 'output'
    data = SimpleNamespace(**d)
    data.dec_in = data.dec_in.reshape((-1, 12, 1))
    data.output = data.output.reshape((-1, 12, 1))
    print(data.enc_in.shape, data.dec_in.shape, data.output.shape)
    print(data.enc_in[0, 0], data.dec_in[0, 0], data.output[0, 0])

    model = seq2seq_model_builder()

    model.compile(loss='mse', optimizer='adam')
    # print(model.summary())

    history = model.fit([data.enc_in, data.dec_in], data.output, epochs=10, batch_size=32)

    return model


def seq2seq_model_builder(HIDDEN_DIM=100):

    encoder_inputs = keras.layers.Input(shape=(MAX_LEN, IN_DIM), dtype='float',)
    encoder_LSTM = keras.layers.LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)

    decoder_inputs = keras.layers.Input(shape=(MAX_LEN, 1), dtype='float',)
    decoder_LSTM = keras.layers.LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs, initial_state=[state_h, state_c])

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def inference(code, , time_range='2021-04', model=None):
    if model is None:
        model = trainer(code, time_range)

    


if __name__ == '__main__':
    # model = trainer('510300')

    inference('510300')

    os.system('say "Mission complete."')
