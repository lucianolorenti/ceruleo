

import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension, MultiHeadAttention
from rul_pm.models.keras.losses import time_to_failure_rul
from rul_pm.models.model import TrainableModel
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, RNN, Activation, Add,
                                     AveragePooling1D, BatchNormalization,
                                     Bidirectional, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GaussianNoise, Lambda, Layer,
                                     LayerNormalization, LSTMCell, Masking,
                                     MaxPool1D, MaxPool2D, MaxPooling2D,
                                     Permute, Reshape, Softmax,
                                     SpatialDropout1D, StackedRNNCells,
                                     UpSampling1D, ZeroPadding2D)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class Single_Global_SelfAttn_Module(tf.keras.layers.Layer):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1,
            return_attns=False):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = Conv2D(n_kernels, (window, w_kernel),
                            activation='relu')
        self.in_linear = Dense(d_model)
        self.out_linear = Dense(n_kernels)
        self.return_attns = return_attns

    def call(self, x):

        x = ExpandDimension()(x)
        x2 = self.conv2(x)
        x2 = Dropout(self.drop_prob)(x2)
        x = tf.squeeze(x2, axis=1)
        x = Permute((1, 2))(x)
        src_seq = self.in_linear(x)

        enc_output = self.layer_stack(src_seq)
        enc_output = self.out_linear(enc_output)
        return enc_output


class Single_Local_SelfAttn_Module(tf.keras.Model):

    def __init__(
            self,
            window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = Conv2D(n_kernels, (local, w_kernel),
                            padding='same', activation='relu')
        self.pooling1 = MaxPooling2D((1, n_multiv))  # (1, n_multiv)
        self.in_linear = Dense(d_model)
        self.out_linear = Dense(n_kernels)

    def call(self, x):

        x = ExpandDimension()(x)
        x1 = self.conv1(x)

        x1 = self.pooling1(x1)

        x1 = Dropout(self.drop_prob)(x1)

        x = tf.squeeze(x1, axis=2)

        src_seq = self.in_linear(x)

        enc_output = src_seq
        enc_output = self.layer_stack(enc_output)

        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(tf.keras.Model):

    def __init__(self, window):

        super(AR, self).__init__()
        self.linear = Dense(1)

    def call(self, x):
        x = Permute((2, 1))(x)
        x = self.linear(x)
        x = Permute((2, 1))(x)
        return x


class DSANet(KerasTrainableModel):

    def __init__(self, window, batch_size, step, transformer, shuffle, models_path, patience, cache_size,
                 local,  n_kernels, w_kernel, d_model, d_inner,
                 n_layers, n_head, drop_prob):
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(DSANet, self).__init__(window, batch_size, step,
                                     transformer, shuffle, models_path, patience, cache_size)

        # parameters from dataset
        self.window = window
        self.local = local
        self.n_kernels = n_kernels
        self.w_kernel = w_kernel

        # hyperparameters of model
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.drop_prob = drop_prob

    def __build_model(self):
        """
        Layout model
        """
        n_features = 80
        self.n_multiv = n_features
        self.ar = AR(window=self.window)
        self.W_output1 = Dense(1)
        self.dropout = Dropout(self.drop_prob)
        self.active_func = Activation('tanh')

    def build_model(self):

        # n_features = self.transformer.n_features
        self.__build_model()
        n_features = 80
        x = Input(shape=(self.window, n_features))

        # Global Self Attention
        sgsf = ExpandDimension()(x)
        sgsf = Conv2D(self.n_kernels, (self.window, self.w_kernel),
                      activation='relu')(sgsf)
        sgsf = Dropout(self.drop_prob)(sgsf)
        sgsf = tf.squeeze(sgsf, axis=1)
        sgsf = Dense(self.d_model)(sgsf)
        sgsf = MultiHeadAttention(
            head_size=self.n_head, num_heads=self.d_model)([sgsf, sgsf, sgsf])
        sgsf = PositionwiseFeedForward(
            self.d_model, self.d_inner, dropout=self.drop_prob)(sgsf)
        sgsf = Dense(self.n_kernels)(sgsf)

        # Global Self Attention
        slsf = ExpandDimension()(x)
        slsf = Conv2D(self.n_kernels, (self.local, self.w_kernel),
                      activation='relu')(slsf)
        slsf = MaxPool2D((self.window, 1), padding='same')(slsf)
        slsf = Dropout(self.drop_prob)(slsf)
        slsf = tf.squeeze(slsf, axis=1)
        slsf = Dense(self.d_model)(slsf)
        slsf = MultiHeadAttention(
            head_size=self.n_head, num_heads=self.d_model)([slsf, slsf, slsf])
        slsf = PositionwiseFeedForward(
            self.d_model, self.d_inner, dropout=self.drop_prob)(slsf)

        slsf = Dense(self.n_kernels)(slsf)

        sf_output = Concatenate(axis=2)([sgsf, slsf])
        sf_output = Dropout(self.drop_prob)(sf_output)
        sf_output = self.W_output1(sf_output)

        ar_output = self.ar(x)

        output = Add()([sf_output, ar_output])
        output = Flatten()(output)
        output = Dense(1)(output)
        model = Model(
            inputs=[x],
            outputs=[output],
        )
        return model

    @property
    def name(self):
        return "DSANet"
