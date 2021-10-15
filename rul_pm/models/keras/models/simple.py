from typing import List, Tuple

import tensorflow as tf
from typing import Optional
from tensorflow.keras import Input, Sequential, regularizers
from tensorflow.keras.layers import (GRU, BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPool1D)


def build_FCN(input_shape: Tuple[int, int],
              layers_sizes: List[int],
              dropout: float,
              l2: Optional[float] = None,
              batch_normalization: bool = True):
    """The model contains a series
    a fully connected model with a RELU activation.


    Parameters
    -----------
    layers_sizes: List[int]
                  List of layer sizes

    dropout: float
             Dropout rate used in each layer

    l2: Optional[float], default None
        Strength of the l2 regularization for each dense layer

    batch_normalization: bool
                         Whether to use batch normalization
    """
    s = Sequential()
    s.add(Flatten(input_shape=input_shape))
    for layer_size in layers_sizes:
        s.add(
            Dense(layer_size,
                  activation='relu',
                  kernel_regularizer=(regularizers.l2(l2)
                                      if l2 is not None else None)))
        s.add(Dropout(dropout))
        if batch_normalization:
            s.add(BatchNormalization())
    s.add(Dense(1, activation='relu'))
    return s


def build_convolutional(
        input_shape: Tuple[int, int],
        layers_sizes,
        dropout,
        l2,
        padding='same',
        activation='relu',
        learning_rate=0.001):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    Parameters
    ----------
    self: list of tuples (filters: int, kernel_size: int)
          Each element of the list is a layer of the network. The first element of the tuple contaings
          the number of filters, the second one, the kernel size.
    """

    s = Sequential()
    s.add(Input(input_shape))
    for filters, kernel_size in layers_sizes:
        s.add(
            Conv1D(filters=filters,
                   strides=1,
                   kernel_size=kernel_size,
                   padding=padding,
                   activation='relu'))
        s.add(MaxPool1D(pool_size=2, strides=2))
    s.add(Flatten())
    s.add(Dense(50, activation='relu'))
    s.add(Dropout(dropout))
    s.add(BatchNormalization())
    s.add(Dense(1, activation=activation))
    return s


def build_recurrent(input_shape, layers,
                    recurrent_type: str,
                    dropout: float):

    def layer_type(self):
        if recurrent_type == 'LSTM':
            return tf.compat.v1.keras.layers.CuDNNLSTM
        elif recurrent_type == 'GRU':
            return GRU
        raise ValueError('Invalid recurrent layer type')

    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, n_filters in enumerate(layers):
        if i == len(layers) - 1:
            model.add(layer_type()(n_filters))
        else:
            model.add(layer_type()(n_filters, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='relu'))
    return model
