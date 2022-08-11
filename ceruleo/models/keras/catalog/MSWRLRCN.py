from typing import Tuple
import tensorflow as tf
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, MaxPool1D
from ceruleo.models.keras.layers import ExpandDimension, ResidualShrinkageBlock
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (

    BatchNormalization,

    Dense,

    Conv1D,
    ReLU,
    LSTM,
    Bidirectional,
    Concatenate,

)
from typing import Tuple


def MSWRLRCN(input_shape: Tuple[int, int]):
    """MSWR-LRCN: A new deep learning approach to remaining useful life estimation of bearings
        Yongyi Chen, Dan Zhang, Wen-an Zhang
    
    [Reference](https://doi.org/10.1016/j.conengprac.2021.104969)
    """

    def ConvBlock(n_filters: int, kernel_size: int):
        return Sequential(
            [
                Conv1D(n_filters, kernel_size, padding='same'),
                BatchNormalization(),
                ReLU(),
                MaxPool1D(2),
            ]
        )

    input = Input(input_shape)
    x = input
    x1 = ConvBlock(input_shape[-1], 16)(x)
    x = ConvBlock(input_shape[-1], 32)(x1)
    x = ResidualShrinkageBlock()(x)
    x = GlobalAveragePooling1D()(x)
    
    x1 = Bidirectional(LSTM(20))(x1)
    x = Concatenate()([x1, x])
    x = ExpandDimension(2)(x)
    
    x = Bidirectional(LSTM(20, return_sequences=True))(x)
    x = tf.keras.activations.softsign(x)
    x = Bidirectional(LSTM(20, return_sequences=True))(x)
    x = tf.keras.activations.softsign(x)
    x = Bidirectional(LSTM(20))(x)
    x = tf.keras.activations.softsign(x)
    x = Dense(64, activation='relu' )(x)
    x = Dense(1, activation='relu' )(x)
    return Model(inputs=[input], outputs=[x])

