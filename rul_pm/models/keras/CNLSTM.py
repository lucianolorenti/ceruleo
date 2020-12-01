

import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension
from rul_pm.models.keras.losses import time_to_failure_rul
from rul_pm.models.model import TrainableModel
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    GRU, LSTM, RNN, Activation, Add, AveragePooling1D, BatchNormalization,
    Bidirectional, Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten,
    GaussianNoise, Lambda, Layer, LayerNormalization, LSTMCell, Masking,
    MaxPool1D, MaxPooling2D, Permute, Reshape, Softmax, SpatialDropout1D,
    StackedRNNCells, UpSampling1D, ZeroPadding2D)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError


class CNLSTM(KerasTrainableModel):
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

    def __init__(self,
                 layers_convolutionals,
                 layers_recurrent,
                 hidden_size,
                 dropout: float, window: int,
                 batch_size: int, step: int, transformer, shuffle, models_path,
                 patience: int = 4, cache_size: int = 30):
        super(CNLSTM, self).__init__(window,
                                     batch_size,
                                     step,
                                     transformer,
                                     shuffle,
                                     models_path,
                                     patience=4,
                                     cache_size=30)
        self.layers_convolutionals = layers_convolutionals
        self.layers_recurrent = layers_recurrent
        self.dropout = dropout
        self.hidden_size = hidden_size

    def build_model(self):
        n_features = self.transformer.n_features
        model = Sequential()
        model.add(Input(shape=(self.window, n_features)))
        f = n_features
        for n_filters in range(self.layers_convolutionals):
            model.add(Conv1D(filters=f, strides=1,
                             kernel_size=2, padding='same', activation='relu'))
            model.add(MaxPool1D(pool_size=2, strides=2))
            f = f * 2

        model.add(Flatten())
        model.add(Dense(self.hidden_size[0] *
                        self.hidden_size[1], activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Reshape(self.hidden_size))

        for n_filters in self.layers_recurrent:
            model.add(tf.compat.v1.keras.layers.CuDNNLSTM(n_filters))

        # model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='relu'))
        return model

    @property
    def name(self):
        return 'CNLSTM'

    def get_params(self, deep=False):
        params = super().get_params(deep)
        params.update({
            'dropout': self.dropout,
            'layers_convolutionals': self.layers_convolutionals,
            'layers_recurrent': self.layers_recurrent
        })
        return params
