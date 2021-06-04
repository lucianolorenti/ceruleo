import logging
import dill
from copy import copy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from rul_pm.iterators.batcher import Batcher
from rul_pm.models.model import TrainableModel
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




def keras_regression_batcher(batcher):
    n_features = batcher.n_features

    def generator_function():
        for X, y, w in batcher:
            yield X, y, w

    a = tf.data.Dataset.from_generator(
        generator_function,
        output_signature=(
            tf.TensorSpec(
                shape=(None, batcher.window_size, n_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None, batcher.output_shape, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ),
    )

    if batcher.prefetch_size is not None:
        a = a.prefetch(batcher.batch_size * 2)
    return a


def keras_autoencoder_batcher(batcher):
    n_features = batcher.n_features

    def gen_train():
        for X, w in batcher:
            yield X, X, w


    a = tf.data.Dataset.from_generator(
        gen_train,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, batcher.window_size, n_features]),
            tf.TensorShape([None, batcher.window_size, n_features]),
            tf.TensorShape([None, 1]),
        ),
    )

    if batcher.prefetch_size is not None:
        a = a.prefetch(batcher.batch_size * 2)

    return a


class KerasTrainableModel(TrainableModel):
    """Base class of keras models

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001
    metrics : list, optional
        Additional metrics to compute, by default [root_mean_squared_error]
    loss : str, optional
        Loss to use at moment of training, by default 'mse'
    prefetch_size : Optional[int], optional
        Prefetch size of the tensorflow iterator, by default None
    batcher_generator : [type], optional
        Function that generate the keras iterators, by default keras_regression_batcher.
        Possible values are keras_autoencoder_batcher or keras_regression_batcher
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        metrics=[root_mean_squared_error],
        loss="mse",
        prefetch_size: Optional[int] = None,
        batcher_generator=keras_regression_batcher,
    ):

        super().__init__()
        self.compiled = False
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss
        self.prefetch_size = prefetch_size
        self.batcher_generator = batcher_generator

    def get_params(self, deep=False) -> dict:
        """Get model parameters

        Parameters
        ----------
        deep : bool, optional
            By default False

        Returns
        -------
        dict
            Add patience, loss and learning_rate to the params dict
        """
        d = super().get_params(deep)
        d["patience"] = self.patience
        d["loss"] = self.loss
        d["learning_rate"] = self.learning_rate
        return d

    def _predict(self, batcher: Batcher):
        #output = []
        #for X, _, _ in self.batcher_generator(batcher):
        #    input_tensor = tf.convert_to_tensor(X)
        #    output_tensor = self.model(input_tensor)
        #    output.append(output_tensor.numpy())
        output = self.model.predict(self.batcher_generator(batcher))

        return np.concatenate(output)

    def predict(self, batcher: Batcher):
        batcher = copy(batcher)
        batcher.restart_at_end = False
        return self._predict(batcher)

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate),
            metrics=self.metrics,
        )

    def reset(self):
        tf.keras.backend.clear_session()
        self._model = None
        self.compiled = False

    def fit(
        self,
        train_batcher: Batcher,
        val_batcher: Optional[Batcher] = None,
        verbose=1,
        epochs=50,
        reset=True,
        class_weight=None,
        print_summary=True,
        callbacks: List[tf.keras.callbacks.Callback] = [],
    ):
        """Fit the model with the given dataset batcher

        Parameters
        ----------
        train_batcher : Batcher
            Batcher of the training set
        val_batcher : Batcher, optional
            Batcher of the validation set, by default None
        verbose : int, optional
            Verbosity level, by default 1
        epochs : int, optional
            Number of epochs to train, by default 50
        reset : bool, optional
            whether to reset the model weights, by default True
        class_weight : str or function, optional
            [description], by default None
        print_summary : bool, optional
            whether to print the model summary, by default True

        Returns
        -------

            Training history
        """
        if reset:
            self.reset()
        self._input_shape = train_batcher.input_shape
        self._model = self.build_model(self._input_shape)
        if not self.compiled:
            self.compile()

        if print_summary:
            self.model.summary()

        a = self.batcher_generator(train_batcher)
        b = self.batcher_generator(val_batcher)

        logger.debug("Start fitting")
        history = self.model.fit(
            a,
            verbose=verbose,
            steps_per_epoch=len(train_batcher),
            epochs=epochs,
            validation_data=b,
            validation_steps=len(val_batcher),
            callbacks=callbacks,
            class_weight=class_weight,
        )
        self.history = history.history
        return self.history

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Abstract method for creating the model.
        It should return the tensorflow model

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Input shape of the model
        """
        raise NotImplementedError

    def save(self, model_path: Union[str, Path]):
        if isinstance(model_path, str):
            model_path = Path(model_path).resolve()
        self.model.save(model_path)
        with open(model_path / "object_data.pickle", "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(model_path: Union[str, Path], custom_object: dict = {}):
        if isinstance(model_path, str):
            model_path = Path(model_path).resolve()
        with open(model_path / "object_data.pickle", "rb") as file:
            d = dill.load(file)
        custom_object["root_mean_squared_error"] = root_mean_squared_error
        d._model = tf.keras.models.load_model(model_path, custom_objects=custom_object)
        return d

    def __getstate__(self):
        d = copy(self.__dict__)
        d["_model"] = None
        return d
