import typing
import warnings
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Activation,
    GlobalAveragePooling2D,
    Permute,
)
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, MaxPool1D


def ExpandDimension(dim: int = -1):
    return Lambda(lambda x: K.expand_dims(x, dim))


def RemoveDimension(axis=0):
    return Lambda(lambda x: K.squeeze(x, axis=axis))


class ConcreteDropout(tf.keras.layers.Layer):
    """Concrete Dropout layer class from https://arxiv.org/abs/1705.07832.
    Dropout Feature Ranking for Deep Learning Models
    Chun-Hao Chang
    Ladislav Rampasek
    Anna Goldenberg

    Parameters:

        dropout_regularizer: Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min: Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min: Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name: String, name of the layer.

    """

    def __init__(
        self,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.9,
        name=None,
        training=True,
        **kwargs
    ):

        super(ConcreteDropout, self).__init__(name=name, **kwargs)
        assert init_min <= init_max, "init_min must be lower or equal to init_max."

        self.dropout_regularizer = dropout_regularizer

        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.training = training

    def build(self, input_shape):
        self.window = input_shape[-2]
        self.number_of_features = input_shape[-1]
        input_shape = tensor_shape.TensorShape(input_shape)

        self.p_logit = self.add_weight(
            name="p_logit",
            shape=[self.number_of_features],
            initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
            dtype=tf.float32,
            trainable=True,
        )

    def concrete_dropout(self, p, x):
        eps = K.cast_to_floatx(K.epsilon())
        temp = 1.0 / 10.0
        unif_noise = K.random_uniform(shape=[self.number_of_features])
        drop_prob = (
            K.log(p + eps)
            - K.log(1.0 - p + eps)
            + K.log(unif_noise + eps)
            - K.log(1.0 - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=True):

        p = K.sigmoid(self.p_logit)

        dropout_regularizer = p * K.log(p)
        dropout_regularizer += (1.0 - p) * K.log(1.0 - p)
        dropout_regularizer *= self.dropout_regularizer * self.number_of_features
        regularizer = K.sum(dropout_regularizer)
        self.add_loss(regularizer)

        x = self.concrete_dropout(p, inputs)

        return x


class ResidualShrinkageBlock(tf.keras.layers.Layer):
    """ResidualShrinkageBlock

    """
    def build(self, input_shape):
        self.blocks = []
        for i in range(2):
            self.blocks.append(
                Sequential(
                    [
                        BatchNormalization(),
                        Activation("relu"),
                        Conv2D(1, (1, 1), padding="same"),
                    ]
                )
            )

        self.abs = Lambda(lambda x: tf.abs(x))
        self.abs_mean = Sequential(
            [
                RemoveDimension(3),
                Permute((1, 2)),
                GlobalAveragePooling1D(),
                ExpandDimension(2),
                ExpandDimension(3),
            ]
        )
        self.shrinkage = Sequential(
            [
                Flatten(),
                Dense(input_shape[-1]),
                BatchNormalization(),
                Activation("relu"),
                Dense(
                    input_shape[-1], kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                ),
                Activation("sigmoid"),
                ExpandDimension(2),
                ExpandDimension(3),
            ]
        )

    def call(self, inputs):
        x = ExpandDimension()(inputs)

        x = self.blocks[0](x)

        residual = self.blocks[1](x)

        residual_abs = self.abs(residual)

        abs_mean = self.abs_mean(residual_abs)

        scales = self.shrinkage(abs_mean)
        thres = abs_mean * scales
        thres = Permute((2, 1, 3))(thres)
        sub = (residual_abs) - thres

        zeros = sub - sub

        n_sub = tf.keras.layers.maximum([sub, zeros])

        residual = tf.keras.backend.sign(residual) * n_sub
        residual = RemoveDimension(3)(residual)
        return residual + inputs
