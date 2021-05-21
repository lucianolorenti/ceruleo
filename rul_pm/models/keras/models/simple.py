from typing import List

import tensorflow as tf
from typing import Optional
from rul_pm.models.keras.keras import KerasTrainableModel
from tensorflow.keras import Input, Sequential, regularizers
from tensorflow.keras.layers import (GRU, BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPool1D)


class FCN(KerasTrainableModel):
    """The model contains a series
    a fully connected model with a RELU activation.


    Parameters
    -----------
    layers_sizes: List[int]
                  List of layer sizes

    dropout: float
             Dropout rate used in each layer

    l2: Optional[float], default None
        Strength of the l2 regularization for each dense layer

    batch_normalization: bool
                         Whether to use batch normalization
    """
    def __init__(self,
                 layers_sizes: List[int],
                 dropout: float,
                 l2: Optional[float] = None,
                 batch_normalization: bool = True,
                 **kwargs):

        super(FCN, self).__init__(**kwargs)
        self.layers_ = []
        self.layers_sizes = layers_sizes
        self.dropout = dropout
        self.l2 = l2
        self.batch_normalization = batch_normalization

    def build_model(self, input_shape):
        s = Sequential()
        s.add(Flatten(input_shape=input_shape))
        for layer_size in self.layers_sizes:
            s.add(
                Dense(layer_size,
                      activation='relu',
                      kernel_regularizer=(regularizers.l2(self.l2)
                                          if self.l2 is not None else None)))
            s.add(Dropout(self.dropout))
            if self.batch_normalization:
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
    def __init__(self,
                 layers_sizes,
                 dropout,
                 l2,
                 padding='same',
                 activation='relu',
                 learning_rate=0.001):
        super(ConvolutionalSimple, self).__init__(learning_rate=learning_rate)
        self.layers_ = []
        self.layers_sizes = layers_sizes
        self.dropout = dropout
        self.l2 = l2
        self.padding = padding
        self.activation = activation

    def build_model(self, input_shape):
        s = Sequential()
        s.add(Input(input_shape))
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
                 dropout: float,
                 **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
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
                model.add(self.layer_type()(n_filters))
            else:
                model.add(self.layer_type()(n_filters, return_sequences=True))

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
