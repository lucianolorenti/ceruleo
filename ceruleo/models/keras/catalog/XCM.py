from typing import Tuple
import numpy as np
import tensorflow as tf

from ceruleo.models.keras.layers import ExpandDimension, RemoveDimension
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    GlobalAveragePooling1D,
)


def XCM(input_shape: Tuple[int, int], *, n_filters: int = 128, filter_window: int = 7):
    """
    A modified version of the model presented in  XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification

    The classification head was replaced by an regression head

    Fauvel, K., Lin, T., Masson, V., Fromont, É., & Termier, A. (2021). 
    Xcm: An explainable convolutional neural network for multivariate time series classification. Mathematics, 
    9(23), 3137.


    """

    input = Input(input_shape)
    x = input

    x2d = ExpandDimension()(x)
    x2d = Conv2D(n_filters, (input_shape[0], 1), padding="same", name="first_conv2d")(
        x2d
    )

    x2d = BatchNormalization()(x2d)

    x2d = Activation("relu")(x2d)
    model_fisrt_conv2d = Model(
        inputs=[input],
        outputs=[x2d],
    )
    x2d_input = Input(shape=x2d.shape[1:])

    x2d = Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="relu",
    )(x2d_input)

    x2d = RemoveDimension(3)(x2d)

    input = Input(shape=(input_shape[0], input_shape[1]))
    x = input
    x1d = Conv1D(
        n_filters,
        filter_window,
        padding="same",
    )(x)

    x1d = BatchNormalization()(x1d)

    x1d = Activation("relu")(x1d)
    model_fisrt_conv1d = Model(
        inputs=[input],
        outputs=[x1d],
    )
    x1d_input = Input(shape=x1d.shape[1:])

    x1d = Conv1D(
        1,
        1,
        padding="same",
        activation="relu",
    )(x1d_input)

    x = Concatenate()([x2d, x1d])

    x = Conv1D(n_filters, filter_window, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(n_filters, activation="relu")(x)
    output = Dense(1, activation="relu", name="RUL_regressor")(x)

    model_regression = Model(inputs=[x2d_input, x1d_input], outputs=[output])
    model_input = Input(shape=(input_shape[0], input_shape[1]))
    conv2d = model_fisrt_conv2d(model_input)
    conv1d = model_fisrt_conv1d(model_input)
    output = model_regression([conv2d, conv1d])

    model = Model(
        inputs=[model_input],
        outputs=[output],
    )
    return model,  (model_fisrt_conv1d, model_fisrt_conv2d, model_regression)


def explain( model_extras, input):
    model_fisrt_conv1d, model_fisrt_conv2d, model_regression = model_extras
    data_input = np.expand_dims(input, axis=0)
    with tf.GradientTape() as tape:
        first_conv1d_layer_output = model_fisrt_conv1d(data_input)
        first_conv2d_layer_output = model_fisrt_conv2d(data_input)
        tape.watch(first_conv2d_layer_output)

        output = model_regression(
            [first_conv2d_layer_output, first_conv1d_layer_output]
        )  #
        grads = tape.gradient(output, first_conv2d_layer_output)
        filter_weight = np.mean(grads, axis=(0, 1, 2))

    mmap = np.sum(np.squeeze(first_conv2d_layer_output) * filter_weight, axis=(2))

    with tf.GradientTape() as tape:
        first_conv2d_layer_output = model_fisrt_conv2d(data_input)
        first_conv1d_layer_output = model_fisrt_conv1d(data_input)
        tape.watch(first_conv1d_layer_output)

        output = model_regression(
            [first_conv2d_layer_output, first_conv1d_layer_output]
        )

    grads1d = tape.gradient(output, first_conv1d_layer_output)
    filter1d_weight = np.mean(grads1d, axis=(0, 1))
    v = np.sum(np.squeeze(first_conv1d_layer_output) * filter1d_weight, axis=1)
    return (mmap, v)
