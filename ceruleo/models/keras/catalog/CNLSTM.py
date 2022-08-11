from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import (LSTM, Conv1D, Dense, Dropout, Flatten,
                                     MaxPool1D, Reshape)


def CNLSTM(
    input_shape: Tuple[int, int],
    *,
    n_conv_layers: int,
    initial_convolutional_size: int,
    layers_recurrent: List[int],
    hidden_size: Tuple[int, int],
    dropout: float,
):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    * Temporal Convolutional Memory Networks forRemaining Useful Life Estimation of Industrial Machinery
    * Lahiru Jayasinghe, Tharaka Samarasinghe, Chau Yuen, Jenny Chen Ni Low, Shuzhi Sam Ge

    [Reference](https://ieeexplore.ieee.org/abstract/document/8754956?casa_token=B_YvavFGulsAAAAA:f2k2I8pH1lM3sOcSGlXEF29seYPK1GPa9Od2-TwnhNeFyWvRRUAqkUOdWUNIyy9FPJHhsGM)

    Parameters:

        input_shape: Input shape of the iterator
        n_conv_layers: Number of convolutional layers. Each convolutional layers is composed by:
            a 1D-convolution:  kernelsize=2, strides=1,padding=same, activation=ReLu
            and a 1D-max-pooling   poolsize=2, strides=2, padding=same
        initial_convolutional_size: The number of filters of the first convolutional layers.
              Next ones will have the power of 2 of the previous one
        layers_recurrent: Number of current layers. Each recurrent layer is composed by an LSTM layer
        hidden_size: After the convolutional layers the signal is projected via a RELU layer and then
                     reshaped again as a time series of size (hidden_size[0], hidden_size[1])
        dropout: Droput factor
    """

    model = Sequential()
    model.add(Input(shape=input_shape))

    for n_filters in range(n_conv_layers):
        model.add(
            Conv1D(
                filters=initial_convolutional_size,
                strides=1,
                kernel_size=2,
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPool1D(pool_size=2, strides=2))
        initial_convolutional_size = initial_convolutional_size * 2

    model.add(Flatten())
    model.add(Dense(hidden_size[0] * hidden_size[1], activation="relu"))
    model.add(Dropout(dropout))
    model.add(Reshape(hidden_size))

    for i, n_filters in enumerate(layers_recurrent):

        model.add(
            LSTM(
                n_filters, return_sequences=i < len(layers_recurrent) - 1
            )
        )

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="relu"))
    return model
