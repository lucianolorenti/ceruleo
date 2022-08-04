
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.iterators.batcher import Batcher
from ceruleo.iterators.iterators import (NotWeighted, SampleWeight,
                                         WindowedDatasetIterator)
from ceruleo.iterators.shufflers import AbstractShuffler, NotShuffled
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


class EstimatorWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, Xy : Tuple[np.ndarray, np.ndarray], y=None, **fit_params):
        X, y = Xy
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, Xy, **transform_params):
        X, y = Xy
        return self.estimator.predict(X, **transform_params)


class SKLearnTimeSeriesWindowTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer_builder, 
                 window_size: int,
                 step: int = 1,
                 output_size: int = 1,
                 shuffler: AbstractShuffler = NotShuffled(),
                 sample_weight:  SampleWeight= NotWeighted(),
                 right_closed: bool = True):
        self.transformer = transformer_builder()
        self.window_size=window_size
        self.output_size = output_size
        self.step = step
        self.shuffler = shuffler 
        self.sample_weight = sample_weight
        self.right_closed = right_closed

    def fit(self, dataset: AbstractTimeSeriesDataset):
        self.transformer.fit(dataset)
        return self
       
    def _iterator(self, dataset: AbstractTimeSeriesDataset):
        return WindowedDatasetIterator(
            dataset.map(self.transformer),
            self.window_size,
            self.step,
            self.output_size,
            shuffler=self.shuffler,
            sample_weight=self.sample_weight,
            right_closed=self.right_closed
        )

    def transform(self, dataset: AbstractTimeSeriesDataset):
        X, y, sw = self._iterator(dataset).get_data()
        return X, y

    def true_values(self, dataset: AbstractTimeSeriesDataset):
        X, y, sw = self._iterator(dataset).get_data()
        return y




def train_model(model, train_iterator: WindowedDatasetIterator, val_windowed_iterator: Optional[WindowedDatasetIterator] = None, **fit_kwargs):
    """Fit the model with the given dataset iterator

    Parameters
    ----------
    train_iterator : WindowedDatasetIterator
        Dataset iterator from which obtain data to fit

    Keyword arguments
    -----------------
    fit_kwargs:
        Arguments for the fit method

    Returns
    -------
    SKLearn model
        self
    """
    X, y, sample_weight = train_iterator.get_data()

    params = {}
    try:
        from xgboost import XGBRegressor
        if val_windowed_iterator is not None and isinstance(model, XGBRegressor):
            X_val, y_val, _ = val_windowed_iterator.get_data()
            fit_kwargs.update({'eval_set': [(X_val, y_val)]})
    except Exception as e:
        logger.error(e)

    return model.fit(X, y.ravel(), **fit_kwargs, sample_weight=sample_weight)


def predict(model, dataset_iterator: WindowedDatasetIterator):
    """Get the predictions for the given iterator

    Parameters
    ----------
    dataset_iterator : WindowedDatasetIterator
        Dataset iterator from which obtain data to predict

    Returns
    -------
    np.array
        Array with the predictiosn
    """
    X, _, _ = dataset_iterator.get_data()
    return model.predict(X)


def fit_batch(model, train_batcher: Batcher, val_batcher: Batcher, n_epochs=15):
    """Fit the model using the given batcher

    Parameters
    ----------
    model: SKLearn Model
    train_batcher : Batcher
        Train dataset batcher
    val_batcher : Batcher
        Validation dataset batcher
    n_epochs : int, optional
        Number of epochs, by default 15

    Returns
    -------
    SKLearn model
        self
    """

    history = []
    for r in range(n_epochs):
        for X, y in train_batcher:
            model.partial_fit(
                np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])), y
            )
        y_true, y_pred = predict(val_batcher)
        history.append(mean_squared_error(y_true, y_pred))
    return model, history


def predict_batch(model, dataset_batcher: Batcher):
    """Predict the values using the given batcher

    Parameters
    ----------
    model: SKLearn model
    dataset_batcher : Batcher


    Returns
    -------
    np.array
        Predictions array
    """
    y_pred = []
    for X, y in dataset_batcher:
        y_pred.extend(
            np.squeeze(
                model.predict(
                    np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
                )
            )
        )
    return np.squeeze(np.array(y_pred))
