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

logger = logging.getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


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
                 output_size: int = 1,
                 cache_size=30,
                 callbacks: list = [],
                 learning_rate=0.001,
                 metrics=[root_mean_squared_error],
                 loss='mse'):
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
        self.callbacks = callbacks
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss

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

    def _predict(self, model, dataset, step=None, batch_size=512):
        step = self.computed_step if step is None else step
        n_features = self.transformer.n_features
        batcher = get_batcher(dataset,
                              self.window,
                              batch_size,
                              self.transformer,
                              step,
                              shuffle=False,
                              output_size=self.output_size,
                              cache_size=self.cache_size)
        batcher.restart_at_end = False

        def gen_dataset():
            for X, _ in batcher:
                yield X

        b = tf.data.Dataset.from_generator(
            gen_dataset, (tf.float32),
            (tf.TensorShape([None, self.window, n_features])))
        return model.predict(b)

    def predict(self, dataset, step=None, batch_size=512):
        return self._predict(self.model, dataset, step=step, batch_size=batch_size)

    def input_shape(self):
        n_features = self.transformer.n_features
        return (self.window, n_features)

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate),
            metrics=self.metrics)

    def _generate_keras_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features

        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, self.output_size])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, self.output_size])))
        return a, b

    def reset(self):
        tf.keras.backend.clear_session()
        self._model = None
        self.compiled = False

    def fit(self, train_dataset, validation_dataset, verbose=1,
            epochs=50, overwrite=True, reset=True, refit_transformer=True, class_weight=None):
        if not overwrite and Path(self.results_filename).resolve().is_file():
            logger.info(f'Results already present {self.results_filename}')
            return
        if refit_transformer:
            self.transformer.fit(train_dataset)
        if reset:
            self.reset()
        if not self.compiled:
            self.compile()

        self.print_summary = True

        if self.print_summary:
            self.model.summary()

        train_batcher, val_batcher = self._create_batchers(
            train_dataset, validation_dataset)
        early_stopping = EarlyStopping(patience=self.patience)
        model_checkpoint_callback = self._checkpoint_callback()

        a, b = self._generate_keras_batcher(train_batcher, val_batcher)

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
            ] +
            self.callbacks,
            class_weight=class_weight)

        self.save_results()
        return self.load_results()

    def build_model(self):
        raise NotImplementedError
