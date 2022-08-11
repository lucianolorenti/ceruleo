from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv1D, Dense,
                                     GlobalAveragePooling1D, MaxPool1D)


def _inception_module(
    input_tensor,
    use_bottleneck: bool,
    bottleneck_size: int,
    kernel_size: int,
    inception_number: int,
    nb_filters: int,
    stride=1,
    activation="linear",
):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = Conv1D(
            filters=bottleneck_size,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [kernel_size // (2**i) for i in range(inception_number)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(
            Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size_s[i],
                strides=stride,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_inception)
        )

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)

    conv_6 = Conv1D(
        filters=nb_filters,
        kernel_size=1,
        padding="same",
        activation=activation,
        use_bias=False,
    )(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    return x


def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(
        filters=int(out_tensor.shape[-1]),
        kernel_size=1,
        padding="same",
        use_bias=False,
    )(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation("relu")(x)
    return x


def InceptionTime(
    input_shape: Tuple[int, int],
    *,
    nb_filters: int = 32,
    use_residual: bool = True,
    use_bottleneck: bool = True,
    depth: int = 6,
    kernel_size: int = 41,
    bottleneck_size: int = 32,
    inception_number: int = 3,
) -> tf.keras.Model:
    """InceptionTime


    Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2020).
    Inceptiontime: Finding alexnet for time series classification.
    Data Mining and Knowledge Discovery, 34(6), 1936-1962.

    [Reference](https://link.springer.com/article/10.1007/s10618-020-00710-y)


    Parameters:

        input_shape: Input shape
        nb_filters: number of fiters
        use_residual: Wether to use residual connections
        use_bottleneck : bool, optional
            _description_, by default True
        depth: max_depth
        kernel_size: kernel size
        bottleneck_size: bottleneck size
        inception_number: iception number

    Returns:

        model: Model
    """

    input = Input(shape=input_shape)

    x = input
    input_res = input

    for d in range(depth):

        x = _inception_module(
            x,
            use_bottleneck,
            bottleneck_size,
            kernel_size,
            inception_number,
            nb_filters,
        )

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    output_layer = Dense(1, activation="relu")(gap_layer)

    model = Model(inputs=input, outputs=output_layer)

    return model
