


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
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import (GRU, LSTM, RNN, Activation, Add,
                                     AveragePooling1D, BatchNormalization,
                                     Bidirectional, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GaussianNoise, Lambda, Layer,
                                     LayerNormalization, LSTMCell, Masking,
                                     MaxPool1D, MaxPooling2D, Permute, Reshape,
                                     Softmax, SpatialDropout1D,
                                     StackedRNNCells, UpSampling1D,
                                     ZeroPadding2D)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

class MVCNN(KerasTrainableModel):

    """
        Model presented in Remaining useful life estimation in prognostics using deep convolution neural networks

        Deafult parameters reported in the article
        Number of filters:	10
        Window size:	30/20/30/15
        Filter length: 10

        Neurons in fully-connected layer	100
        Dropout rate	0.5
        batch_size = 512


        Parameters
        -----------
        n_filters : int

        filter_size : int

        window: int

        batch_size: int
        step: int
        transformer
        shuffle
        models_path
        patience: int = 4
        cache_size: int = 30



    """

    def __init__(self,
                 shape: tuple,                 
                 dropout: float,
                 window: int,
                 batch_size: int,
                 step: int, transformer,
                 shuffle, models_path,
                 patience: int = 4,
                 cache_size: int = 30,
                 **kwargs):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         cache_size=cache_size,
                         **kwargs)
        self.shape = shape        
        self.dropout = dropout
      

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate,
                                      beta_1=0.85,
                                      beta_2=0.9,
                                      epsilon=0.001,
                                      amsgrad=True),
            metrics=self.metrics)

    def build_model(self):
        n_features = self.transformer.n_features

        input = Input(shape=(self.window, n_features))
        x = input
        x = Permute((2, 1))(x)
        x = Reshape((self.shape[0], self.shape[1], self.window))(x)  


     
        x = Conv2D(self.window, (1, 1), activation='relu', padding='same')(x)

        x1 = Conv2D(self.window, (2, 2), activation='relu', padding='same')(x)
        x1 = Conv2D(self.window, (2, 2), activation='relu', padding='same')(x1)

        x2 = Conv2D(self.window, (3, 3), activation='relu', padding='same')(x)
        x2 = Conv2D(self.window, (3, 3), activation='relu', padding='same')(x2)

        x3 = Conv2D(self.window, (5, 5), activation='relu', padding='same')(x)
        x3 = Conv2D(self.window, (5, 5), activation='relu', padding='same')(x3)

        x = Concatenate(axis=1)([x, x1, x2, x3])
        x = Conv2D(self.window, self.shape)(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        output = Dense(1)(x)
        model = Model(
            inputs=[input],
            outputs=[output],
        )
        return model

    @property
    def name(self):
        return "MVCNN"