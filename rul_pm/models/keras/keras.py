import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from pandas.io.formats.format import buffer_put_lines
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.model import BatchTrainableModel
from rul_pm.store.store import store
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

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


def keras_batcher(model, train_batcher, val_batcher, output_shape):
    n_features = model.transformer.n_features

    def gen_train():
        for X, y, w in train_batcher:
            yield X, y, w

    def gen_val():
        for X, y, w in val_batcher:
            yield X, y, w

    a = (tf.data.Dataset.from_generator(
        gen_train,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape(output_shape),
            tf.TensorShape([None, 1]))))
    b = (tf.data.Dataset.from_generator(
        gen_val,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape(output_shape),
            tf.TensorShape([None, 1])))
         )
    if model.prefetch_size is not None:
        a = a.prefetch(model.batch_size*2)
        b = b.prefetch(model.batch_size*2)
    return a, b


def keras_regression_batcher(model, train_batcher, val_batcher):
    return keras_batcher(model,
                         train_batcher,
                         val_batcher,
                         [None, model.output_size, 1])


def keras_autoencoder_batcher(model, train_batcher, val_batcher):
    n_features = model.transformer.n_features

    def gen_train():
        for X, _, w in train_batcher:
            yield X, X, w

    def gen_val():
        for X, _, w in val_batcher:
            yield X, X, w

    a = (tf.data.Dataset.from_generator(
        gen_train,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape([None, 1]))))
    b = (tf.data.Dataset.from_generator(
        gen_val,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape([None, model.window, n_features]),
            tf.TensorShape([None, 1])))
         )
    if model.prefetch_size is not None:
        a = a.prefetch(model.batch_size*2)
        b = b.prefetch(model.batch_size*2)
    return a, b

   

class KerasTrainableModel(BatchTrainableModel):
    """
        Base class for keras models

    Parameters
    ----------
    patience: int
              Patience for early stopping
    """

    def __init__(self,
                 patience: int = 4,
                 callbacks: list = [],
                 learning_rate=0.001,
                 metrics=[root_mean_squared_error],
                 loss='mse',
                 prefetch_size: Optional[int] = None,
                 batcher_generator=keras_regression_batcher,
                 **kwargs):
        super().__init__(**kwargs)
        self.compiled = False
        self.callbacks = callbacks
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss
        self.patience = patience
        self.prefetch_size = prefetch_size
        self.batcher_generator = batcher_generator


    def get_params(self, deep=False):
        d = super().get_params(deep)
        d['patience'] = self.patience
        d['loss'] = self.loss
        d['learning_rate'] = self.learning_rate
        return d

    def load_best_model(self):
        print(self.model_filepath)
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

    def _predict(self, model, dataset, step=None, batch_size=512, evenly_spaced_points: Optional[int] = None):
        step = self.computed_step if step is None else step

        batcher = get_batcher(dataset,
                              self.window,
                              batch_size,
                              self.transformer,
                              step,
                              shuffle=False,
                              output_size=self.output_size,
                              cache_size=self.cache_size,
                              evenly_spaced_points=evenly_spaced_points,
                              restart_at_end=False,
                              add_last=self.add_last)

        output = []
        for X, _, _ in batcher:
            input_tensor = tf.convert_to_tensor(X)
            output_tensor = self.model(input_tensor)
            output.append(output_tensor.numpy())

        return np.concatenate(output)

    def predict(self, dataset, step=None, batch_size=512, evenly_spaced_points: Optional[int] = None):
        return self._predict(self.model, dataset, step=step, batch_size=batch_size, evenly_spaced_points=evenly_spaced_points)

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate),
            metrics=self.metrics)

    def _generate_keras_batcher(self, train_batcher, val_batcher):
        return self.batcher_generator(self, train_batcher, val_batcher)

    def reset(self):
        tf.keras.backend.clear_session()
        self._model = None
        self.compiled = False

    def fit(self, train_dataset, validation_dataset=None, verbose=1,
            epochs=50, overwrite=True, reset=True, refit_transformer=True, class_weight=None,
            print_summary=True):
        if not overwrite and Path(self.results_filename).resolve().is_file():
            logger.info(f'Results already present {self.results_filename}')
            return

        if refit_transformer:
            self.transformer.fit(train_dataset)
        if reset:
            self.reset()
        if not self.compiled:
            self.compile()

        if print_summary:
            self.model.summary()

        train_batcher, val_batcher = self._create_batchers(
            train_dataset, validation_dataset)
        early_stopping = EarlyStopping(patience=self.patience)
        model_checkpoint_callback = self._checkpoint_callback()

        a, b = self._generate_keras_batcher(train_batcher, val_batcher)

        logger.debug('Start fitting')
        logger.debug(self.model_filepath)
        self.history = self.model.fit(
            a,
            verbose=verbose,
            steps_per_epoch=len(train_batcher),
            epochs=epochs,
            validation_data=b,
            validation_steps=len(val_batcher),
            callbacks=[
                early_stopping,
                model_checkpoint_callback
            ] +
            self.callbacks,
            class_weight=class_weight)

        return self.history

    def build_model(self):
        raise NotImplementedError


