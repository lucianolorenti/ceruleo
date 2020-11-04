import logging
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.keras.losses import time_to_failure_rul
from rul_pm.models.model import TrainableModel
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, RNN, Activation, Add,
                                     AveragePooling1D, BatchNormalization,
                                     Bidirectional, Concatenate, Conv1D, Dense,
                                     Dropout, Flatten, GaussianNoise, Lambda,
                                     Layer, LayerNormalization, LSTMCell,
                                     Masking, MaxPool1D, Reshape,
                                     SpatialDropout1D, StackedRNNCells,
                                     UpSampling1D)
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

logger = logging.getLogger(__name__)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered.
    """

    def __init__(self, batcher):
        super().__init__()
        self.batcher = batcher

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                logger.info('Batch %d: Invalid loss, terminating training' %
                            (batch))
                self.model.stop_training = True
                self.batcher.stop = True


class KerasTrainableModel(TrainableModel):
    def __init__(self,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=4,
                 output_size:int=1,
                 cache_size=30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         output_size=output_size,
                         cache_size=cache_size)
        self.compiled = False

    def load_best_model(self):
        self.model.load_weights(self.model_filepath)

    def _checkpoint_callback(self):
        return ModelCheckpoint(filepath=self.model_filepath,
                               save_weights_only=True,
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True)

    def _results(self):
        params = super()._results()
        params.update({
            'best_val_loss': np.min(self.history.history['val_loss']),
            'val_loss': self.history.history['val_loss'],
            'train_loss': self.history.history['loss'],
        })
        return params

    def predict(self, dataset, step=None, batch_size=512):
        step = self.computed_step if step is None else step
        n_features = self.transformer.n_features
        batcher = get_batcher(dataset,
                              self.window,
                              batch_size,
                              self.transformer,
                              step,
                              shuffle=False,
                              cache_size=self.cache_size)
        batcher.restart_at_end = False

        def gen_dataset():
            for X, _ in batcher:
                yield X

        b = tf.data.Dataset.from_generator(
            gen_dataset, (tf.float32),
            (tf.TensorShape([None, self.window, n_features])))
        return self.model.predict(b)

    def input_shape(self):
        n_features = self.transformer.n_features
        return (self.window, n_features)

    def compile(self):
        self.compiled=  True
        self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001))

    def _generate_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features
        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None])))
        return a,b

    def reset(self):
        tf.keras.backend.clear_session()
        self._model = None
        self.compiled = False
        

    def fit(self, train_dataset, validation_dataset, verbose=1, epochs=50, overwrite=True, reset=True, refit_transformer=True):
        if not overwrite and Path(self.results_filename).resolve().is_file():
            logger.info(f'Results already present {self.results_filename}')
            return
        if refit_transformer:
            self.transformer.fit(train_dataset)
        if reset:
            self.reset()
        if not self.compiled:
            self.compile()

        logger.info('Creating batchers')
        train_batcher = get_batcher(train_dataset,
                                    self.window,
                                    self.batch_size,
                                    self.transformer,
                                    self.computed_step,
                                    shuffle=self.shuffle,
                                    cache_size=self.cache_size)
        val_batcher = get_batcher(validation_dataset,
                                  self.window,
                                  self.batch_size,
                                  self.transformer,
                                  20,
                                  shuffle=False,
                                  cache_size=self.cache_size)
        val_batcher.restart_at_end = False

        early_stopping = EarlyStopping(patience=self.patience)
        model_checkpoint_callback = self._checkpoint_callback()

        a,b = self._generate_batcher(train_batcher, val_batcher)


        logger.info('Start fitting')
        logger.info(self.model_filepath)
        self.history = self.model.fit(
            a,
            verbose=verbose,
            steps_per_epoch=len(train_batcher),
            epochs=epochs,
            validation_data=b,
            validation_steps=len(val_batcher),
            callbacks=[
                early_stopping,
                # lr_callback,
                # TerminateOnNaN(train_batcher),
                model_checkpoint_callback
            ])

        self.save_results()
        return self.load_results()


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
                 patience=4, cache_size=30, padding='same', activation='relu'):
        super(ConvolutionalSimple, self).__init__(window,
                                                  batch_size,
                                                  step,
                                                  transformer,
                                                  shuffle,
                                                  models_path,
                                                  patience=4,
                                                  cache_size=30)
        self.layers_ = []
        self.layers_sizes = layers_sizes
        self.dropout = dropout
        self.l2 = l2
        self.padding = padding
        self.activation = activation

    def build_model(self):
        s = Sequential()
        
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

    def build_model(self):
        n_features = self.transformer.n_features
        model = Sequential()
        model.add(Input(shape=(self.window, n_features)))
        for n_filters in self.layers_convolutionals:
            model.add(Conv1D(filters=n_filters, strides=1,
                             kernel_size=2, padding='same', activation='relu'))
            model.add(MaxPool1D(pool_size=2, strides=2))

        # model.add(Flatten())
        #w = 10
        #n = 1
        #model.add(Dense(w*n, activation='relu'))
        # model.add(Dropout(self.dropout))
        #model.add(Reshape((w, n)))

        recurrents = []
        for n_filters in self.layers_recurrent:
            recurrents.append(
                LSTMCell(n_filters, recurrent_dropout=0.5, dropout=0.5))

        stacked_lstm = tf.keras.layers.StackedRNNCells(recurrents)
        model.add(RNN(stacked_lstm))

        model.add(Dropout(self.dropout))

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


class MultiTaskRUL(KerasTrainableModel):
    """
        A Multi task network that learns to regress the RUL and the Time to failure

        Two Birds with One Network: Unifying Failure Event Prediction and Time-to-failure Modeling
        Karan Aggarwal, Onur Atan, Ahmed K. Farahat, Chi Zhang, Kosta Ristovski, Chetan Gupta

        The target

        Parameters
        -----------
        layers_lstm : List[int]
                      Number of LSTM layers
        layers_dense : List[int]
                       Number of dense layers
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
                 layers_lstm : List[int],
                 layers_dense : List[int],
                 window: int,
                 batch_size: int,
                 step: int, transformer, shuffle, models_path,
                 patience: int = 4, cache_size: int = 30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=4,
                         cache_size=30)
        self.layers_dense = layers_dense
        self.layers_lstm = layers_lstm

    def compile(self):
        super().compile()
        self.model.compile(
            optimizer=optimizers.Adam(lr=0.001),
            loss=time_to_failure_rul(weights={
                0: 1.,
                1: 2.
            }),
            # {
            #    'rul': MeanSquaredError(),
            #    'ttf': BinaryCrossentropy(from_logits=True),
            # },
            loss_weights=[1.0, 1.0],
        )

    @property
    def name(self):
        return "MultiTaskRULTTF"

    def build_model(self):
        n_features = self.transformer.n_features

        input = Input(shape=(self.window, n_features))
        x = input

        if len(self.layers_lstm) > 1:
            for n_filters in self.layers_lstm:
                x = LSTM(n_filters, recurrent_dropout=0.2,
                         return_sequences=True)(x)

        x = LSTM(n_filters, recurrent_dropout=0.2, return_sequences=False)(x)

        for n_filters in self.layers_dense:
            x = Dense(n_filters, activation='elu')(x)

        RUL_output = Dense(1, activation='elu', name='rul')(x)

        FP_output = Dense(1, activation='sigmoid', name='ttf')(x)

        output = tf.keras.layers.Concatenate(axis=1)([RUL_output, FP_output])
        model = Model(
            inputs=[input],
            outputs=[output],
        )
        return model

    def _generate_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features
        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, 2])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, 2])))
        return a,b
