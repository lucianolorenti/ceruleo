"""
Attention meachanism
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Add, Conv1D, Dense, Dropout, Lambda,
                                     Permute)


class Attention(tf.keras.Model):
    """
    Temporal pattern attention for multivariate time series forecasting
    Shun-Yao Shih, Fan-Keng Sun & Hung-yi Lee

    They propose using a set of filters to extract time-invariant temporal patterns,
    similar to transforming time series data into its “frequency domain”.
    Then they propose a novel attention mechanism to select relevant time series,
    and use its frequency domain information for multivariate forecasting.
    """

    def __init__(self, number_of_filters, attention_size, dropout=0.1):
        super(Attention, self).__init__()
        self.filter_size = 1
        self.number_of_filters = number_of_filters
        self.attention_size = attention_size
        self.dropout = dropout

        self.permute = Permute((2, 1))
        self.expand = Lambda(lambda x: K.expand_dims(x))
        self.dropout = Dropout(self.dropout)
        self.squeeze = Lambda(lambda x: K.squeeze(x, 1))
        self.prev_states = Lambda(lambda x: x[:, :-1, :])
        self.last_state = Lambda(lambda x: x[:, -1, :])
        self.permute1 = Permute((2, 1))

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(self.number_of_filters,
                                        input_shape[2]),
                                 trainable=True)

        self.conv_layer = Conv1D(self.number_of_filters,
                                 self.attention_size,
                                 padding='same')
        self.denseHt = Dense(self.number_of_filters, use_bias=False)
        self.denseVt = Dense(self.number_of_filters, use_bias=False)

        super(Attention, self).build(input_shape)

    def call(self, x):

        prev_states = self.prev_states(x)
        prev_states = self.permute(prev_states)
        prev_states = self.conv_layer(prev_states)
        prev_states = self.dropout(prev_states)
        last_state = self.last_state(x)
        score = K.sigmoid(K.batch_dot(
            prev_states, K.dot(last_state, K.transpose(self.W))))

        score = K.expand_dims(score, axis=2)
        score = K.repeat_elements(score, rep=prev_states.shape[2], axis=2)
        vt = K.sum(tf.math.multiply(score, prev_states), axis=2)

        vt = Add()([self.denseHt(vt), self.denseVt(last_state)])

        return vt
