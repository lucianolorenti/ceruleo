from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    GRU, LSTM, Activation, Add, AveragePooling1D, BatchNormalization,
    Bidirectional, Concatenate, Conv1D, Dense, Dropout, Flatten, GaussianNoise,
    Lambda, Layer, LayerNormalization, Masking, MaxPool1D, Reshape,
    SpatialDropout1D, UpSampling1D)

from rul_pm.models.model import TrainableModel
import numpy as np
import logging
import tensorflow as tf
from rul_pm.iterators.batcher import get_batcher

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
                 cache_size=30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         cache_size=cache_size)

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

    def predict(self, dataset, step=None):
        step = self.step if step is None else step
        n_features = self.transformer.n_features
        batcher = get_batcher(dataset,
                              self.window,
                              512,
                              self.transformer,
                              step,
                              shuffle=False,
                              cache_size=self.cache_size)
        batcher.restart_at_end = False

        def gen_dataset():
            for X, y in batcher:
                yield X

        b = tf.data.Dataset.from_generator(
            gen_dataset, (tf.float32),
            (tf.TensorShape([None, self.window, n_features])))
        return self.model.predict(b)

    def input_shape(self):
        n_features = self.transformer.n_features
        return (self.window, n_features)

    def fit(self, train_dataset, validation_dataset, verbose=1, epochs=50, overwrite=True):
        if not overwrite and Path(self.results_filename).resolve().is_file():
            logger.info(f'Results already present {self.results_filename}')
            return
        if not self.transformer.fitted_:
            self.transformer.fit(train_dataset)
        logger.info('Creating batchers')
        train_batcher = get_batcher(train_dataset,
                                    self.window,
                                    self.batch_size,
                                    self.transformer,
                                    self.step,
                                    shuffle=self.shuffle,
                                    cache_size=self.cache_size)
        val_batcher = get_batcher(validation_dataset,
                                  self.window,
                                  self.batch_size,
                                  self.transformer,
                                  self.step,
                                  shuffle=False,
                                  cache_size=self.cache_size)
        val_batcher.restart_at_end = False

        early_stopping = EarlyStopping(patience=self.patience)
        model_checkpoint_callback = self._checkpoint_callback()

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
                TerminateOnNaN(train_batcher),
                model_checkpoint_callback
            ])

        self.save_results()
        return self.load_results()


class FCN(KerasTrainableModel):
    def __init__(self,
                 layers,
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
        self.layers_sizes_ = layers
        self.dropout = dropout
        self.l2 = l2

    def build_model(self):
        s = Sequential()
        s.add(Flatten())
        for l in self.layers_sizes_:
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

    def parameters(self):
        params = super().parameters()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers': self.layers_sizes_
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

    def __init__(self, layers, dropout, l2, window,
                 batch_size, step, transformer, shuffle, models_path,
                 patience=4, cache_size=30):
        super(ConvolutionalSimple, self).__init__(window,
                                                  batch_size,
                                                  step,
                                                  transformer,
                                                  shuffle,
                                                  models_path,
                                                  patience=4,
                                                  cache_size=30)
        self.layers_ = []
        self.layers_sizes_ = layers
        self.dropout = dropout
        self.l2 = l2

    def build_model(self):
        s = Sequential()
        for filters, kernel_size in self.layers_sizes_:
            s.add(
                Conv1D(filters=filters,
                       strides=1,
                       kernel_size=kernel_size,
                       padding='same',
                       activation='relu'))
            s.add(MaxPool1D(pool_size=2, strides=2))
        s.add(Flatten())
        s.add(Dense(50, activation='relu'))
        s.add(Dropout(self.dropout))
        s.add(BatchNormalization())
        s.add(Dense(1, activation='relu'))
        return s

    @property
    def name(self):
        return 'ConvolutionalSimple'

    def parameters(self):
        params = super().parameters()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers': self.layers_sizes_
        })
        return params


class CNLSTM(KerasTrainableModel):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    Parameters
    ----------
    self: list of tuples (filters: int, kernel_size: int)
          Each element of the list is a layer of the network. The first element of the tuple contaings
          the number of filters, the second one, the kernel size.
    """

    def __init__(self, layers_convolutionals,
                 layers_recurrent,
                 dropout, window,
                 batch_size, step, transformer, shuffle, models_path,
                 patience=4, cache_size=30):
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
        model = Sequential()
        model.add(Input(shape=(window, n_features)))
        for n_filters in self.layers_convolutionals:
            model.add(Conv1D(filters=n_filters, strides=1,
                             kernel_size=2, padding='same', activation='relu'))
            model.add(MaxPool1D(pool_size=2, strides=2))

        model.add(Flatten())
        model.add(Dense(window * n_features, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Reshape((window, n_features)))

        for n_filters in self.layers_recurrent:
            model.add(LSTM(n_filters*3, dropout=0.2))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='relu'))
        return model

    @property
    def name(self):
        return 'CNLSTM'

    def parameters(self):
        params = super().parameters()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers': self.layers_sizes_
        })
        return params
