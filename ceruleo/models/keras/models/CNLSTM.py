from typing import Tuple
import tensorflow as tf
from ceruleo.models.keras.keras import KerasTrainableModel
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPool1D, Reshape


def CNLSTM(
    input_shape: Tuple[int, int],
    layers_convolutionals,
    layers_recurrent,
    hidden_size,
    dropout: float,
):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    * Temporal Convolutional Memory Networks forRemaining Useful Life Estimation of Industrial Machinery
    * Lahiru Jayasinghe∗, Tharaka Samarasinghe†, Chau Yuen∗, Jenny Chen Ni Low§, Shuzhi Sam Ge‡


    Parameters
    ----------
    layers_convolutionals: list of int
        Number of convolutional layers. Each convolutional layers is composed by:
        * 1D-convolution:  kernelsize=2, strides=1,padding=same, activation=ReLu
        * 1D-max-pooling   poolsize=2, strides=2, padding=same
    layers_recurrent: list of int
        Number of current layers. Each recurrent layer is composed by:
        * LSTM
    dropout:float
    window: int
    batch_size: int
    step: int
    transformer, shuffle, models_path,
    patience:int. Default:4
    cache_size:int. Default 30
    """

    model = Sequential()
    model.add(Input(shape=input_shape))

    for n_filters in range(layers_convolutionals):
        model.add(
            Conv1D(
                filters=f, strides=1, kernel_size=2, padding="same", activation="relu"
            )
        )
        model.add(MaxPool1D(pool_size=2, strides=2))
        f = f * 2

    model.add(Flatten())
    model.add(Dense(hidden_size[0] * hidden_size[1], activation="relu"))
    model.add(Dropout(dropout))
    model.add(Reshape(hidden_size))

    for n_filters in layers_recurrent:
        model.add(tf.compat.v1.keras.layers.CuDNNLSTM(n_filters))

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="relu"))
    return model
