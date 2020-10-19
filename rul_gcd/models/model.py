
import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from rul_gcd.iterators.batcher import Batcher, get_batcher
from rul_gcd.iterators.iterators import WindowedDatasetIterator
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

tf.keras.backend.set_floatx('float32')


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
                logger.info(
                    'Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                self.batcher.stop = True




def json_to_str(elem):
    if callable(elem):
        return elem.__name__
    elif isinstance(elem, np.ndarray):
        #return elem.tolist()
        return []
    else:
        return str(elem)
    

class TrainableModel(Model):
    def __init__(self, window, batch_size, step, transformer, shuffle, models_path, patience=4, ):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.step = step
        self.transformer = transformer
        self.shuffle = shuffle
        self.patience = patience
        self.models_path = models_path
        self.model_filename_ = None
        self.checkpoint_filepath_ = None

    @property
    def model_filename(self):
        if self.model_filename_ is None:
            hash_object = self._hash_parameters()
            self.model_filename_ = self.name + '_' + hash_object
        return self.model_filename_

    @property
    def checkpoint_filepath(self):
        if self.checkpoint_filepath_ is None:
            self.checkpoint_filepath_ = str(self.models_path / self.model_filename)
        return self.checkpoint_filepath_

    @property
    def name(self):
        raise NotImplementedError

    def _hash_parameters(self):
        return hashlib.md5(self._parameter_to_json()).hexdigest()
    
    def _parameter_to_json(self):
        return json.dumps(
                    self.parameters(),
                    sort_keys=True,
                    default=json_to_str
                ).encode('utf-8')

    def parameters(self):
        return {
            'window': self.window,
            'batch_size': self.batch_size,
            'step': self.step,
            'shuffle': self.shuffle,
            'patience': self.patience,
            'transformer': self.transformer.description()
        }

    def load_best_model(self):
        self.load_weights(self.checkpoint_filepath)

    def _checkpoint_callback(self):
        return ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

    @property
    def results_filename(self):
        return self.checkpoint_filepath + 'results_.pkl'

    def load_results(self):
        with open(self.results_filename, 'rb') as infile:
            return pickle.load(infile)

    def save_results(self):
        logger.info(f'Writing results {self.results_filename}')
        data = {
                'parameters': self.parameters(),
                'best_val_loss': np.min(self.history.history['val_loss']),
                'val_loss': self.history.history['val_loss'],
                'train_loss': self.history.history['loss'],
                'model_file': self.checkpoint_filepath
            }
        with open(self.results_filename, 'wb') as outfile:
            pickle.dump(data, outfile)

    def true_values(self, dataset, step=None):
        step = self.step if step is None else step
        batcher = get_batcher(dataset, self.window, 512,
                                  self.transformer, step, shuffle=False)
        batcher.restart_at_end = False
        trues = []
        for _, y in batcher:
            trues.extend(y)
        return trues
        
    @property
    def n_features(self):
        return self.transformer.n_features

    def model(self):
        # workaround https://github.com/tensorflow/tensorflow/issues/25036
        # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model/55236388#55236388
        x = Input(shape=(self.window, self.n_features))
        return Model(inputs=[x], outputs=self.call(x))

    def predict(self, dataset, step=None):
        step = self.step if step is None else step
        n_features = self.transformer.n_features
        batcher = get_batcher(dataset, self.window, 512,
                                  self.transformer, step, shuffle=False)
        batcher.restart_at_end = False
        def gen_dataset():
            for X, y in batcher:
                yield X

        b = tf.data.Dataset.from_generator(gen_dataset,
                            (tf.float32),
                            (tf.TensorShape([None, self.window, n_features])))
        return super().predict_generator(b)

    def input_shape(self):
        n_features = self.transformer.n_features
        return (self.window, n_features)

    def fit(self, train_dataset, validation_dataset, epochs=50, verbose=1):
        #if Path(self.results_filename).resolve().is_file():
        #    logger.info(f'Results already present {self.results_filename}')
        #    return None
        if not self.transformer.fitted_:
            self.transformer.fit(train_dataset)
        logger.info('Creating batchers')
        train_batcher = get_batcher(train_dataset,  self.window, self.batch_size,
                                    self.transformer, self.step, shuffle=self.shuffle)
        val_batcher = get_batcher(validation_dataset, self.window, self.batch_size,
                                  self.transformer, self.step, shuffle=False)
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
        a = tf.data.Dataset.from_generator(gen_train,
                                   (tf.float32, tf.float32),
                                   (tf.TensorShape([None, self.window, n_features]), tf.TensorShape([None])))
        b = tf.data.Dataset.from_generator(gen_val,
                                   (tf.float32, tf.float32),
                                   (tf.TensorShape([None, self.window, n_features]), tf.TensorShape([None])))

        logger.info('Start fitting')
        logger.info(self.checkpoint_filepath)
        self.history = super().fit(a,
                                   verbose=verbose,
                                   steps_per_epoch=len(train_batcher),
                                   epochs=epochs,
                                   validation_data=b,
                                   validation_steps=len(val_batcher),
                                   callbacks=[early_stopping,
                                              # lr_callback,
                                              TerminateOnNaN(train_batcher),
                                              model_checkpoint_callback])
        self.save_results()
        return self.load_results()
