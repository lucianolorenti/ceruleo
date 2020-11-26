import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.keras.layers import ExpandDimension
from rul_pm.models.keras.losses import time_to_failure_rul
from rul_pm.models.model import TrainableModel
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import (GRU, LSTM, RNN, Activation, Add,
                                     AveragePooling1D, BatchNormalization,
                                     Bidirectional, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GaussianNoise, Lambda, Layer,
                                     LayerNormalization, LSTMCell, Masking,
                                     MaxPool1D, Permute, Reshape,
                                     SpatialDropout1D, StackedRNNCells,
                                     UpSampling1D, ZeroPadding2D)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

class FCN(KerasTrainableModel):
    def __init__(self,
                 layers_sizes,
                 dropout,
                 l2,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=7):
        super(FCN, self).__init__(window,
                                  batch_size,
                                  step,
                                  transformer,
                                  shuffle,
                                  models_path,
                                  patience=patience)
        self.layers_ = []
        self.layers_sizes = layers
        self.dropout = dropout
        self.l2 = l2

    def build_model(self):
        s = Sequential()
        s.add(Flatten())
        for l in self.layers_sizes:
            s.add(
                Dense(l,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(self.l2)))
            s.add(Dropout(self.dropout))
            s.add(BatchNormalization())
        s.add(Dense(1, activation='relu'))
        return s

    @property
    def name(self):
        return 'FCN'

    def get_params(self):
        params = super().get_params()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers_sizes': self.layers_sizes
        })
        return params


class ConvolutionalSimple(KerasTrainableModel):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    Parameters
    ----------
    self: list of tuples (filters: int, kernel_size: int)
          Each element of the list is a layer of the network. The first element of the tuple contaings
          the number of filters, the second one, the kernel size.
    """

    def __init__(self, layers_sizes, dropout, l2,  window,
                 batch_size, step, transformer, shuffle, models_path,
                 patience=4, cache_size=30, padding='same', activation='relu',
                 learning_rate=0.001):
        super(ConvolutionalSimple, self).__init__(window,
                                                  batch_size,
                                                  step,
                                                  transformer,
                                                  shuffle,
                                                  models_path,
                                                  patience=4,
                                                  cache_size=30,
                                                  learning_rate=learning_rate)
        self.layers_ = []
        self.layers_sizes = layers_sizes
        self.dropout = dropout
        self.l2 = l2
        self.padding = padding
        self.activation = activation

    def build_model(self):
        s = Sequential()
        n_features = self.transformer.n_features
        s.add(Input((self.window, n_features)))
        for filters, kernel_size in self.layers_sizes:
            s.add(
                Conv1D(filters=filters,
                       strides=1,
                       kernel_size=kernel_size,
                       padding=self.padding,
                       activation='relu'))
            s.add(MaxPool1D(pool_size=2, strides=2))
        s.add(Flatten())
        s.add(Dense(50, activation='relu'))
        s.add(Dropout(self.dropout))
        s.add(BatchNormalization())
        s.add(Dense(1, activation=self.activation))
        return s

    @property
    def name(self):
        return 'ConvolutionalSimple'

    def get_params(self, deep=False):
        params = super().get_params()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers_sizes': self.layers_sizes,
            'padding': self.padding
        })
        return params


class SimpleRecurrent(KerasTrainableModel):
    def __init__(self,
                 layers,
                 recurrent_type: str,
                 dropout: float, window: int,
                 batch_size: int, step: int, transformer, shuffle, models_path,
                 patience: int = 4, cache_size: int = 30):
        super(SimpleRecurrent, self).__init__(window,
                                              batch_size,
                                              step,
                                              transformer,
                                              shuffle,
                                              models_path,
                                              patience=4,
                                              cache_size=30)
        self.layers = layers
        self.recurrent_type = recurrent_type
        self.dropout = dropout

    def layer_type(self):
        if self.recurrent_type == 'LSTM':
            return tf.compat.v1.keras.layers.CuDNNLSTM
        elif self.recurrent_type == 'GRU':
            return GRU
        raise ValueError('Invalid recurrent layer type')

    def build_model(self):
        n_features = self.transformer.n_features
        model = Sequential()
        model.add(Input(shape=(self.window, n_features)))
        for i, n_filters in enumerate(self.layers):
            if i == len(self.layers) - 1:
                model.add(self.layer_type()(
                    n_filters))
            else:
                model.add(self.layer_type()(
                    n_filters, return_sequences=True))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='relu'))
        return model

    @property
    def name(self):
        return 'Recurrent'

    def get_params(self, deep=False):
        params = super().get_params(deep)
        params.update({
            'dropout': self.dropout,
            'layers': self.layers,
            'recurrent_type': self.recurrent_type
        })
        return params

