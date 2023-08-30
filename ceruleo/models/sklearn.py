"""This module provides interoperability 

Scikit learn models can be used wit the ceruleo Transformers


The `TimeSeriesWindowTransformer`  is a scikit-learn transformers that takes
a transformer and iterator paramaters, to build a `WindowedDatasetIterator`
and generate the X and y.

Since the X and Y are generated by one scikit-learn transformer step, `EstimatorWrapper`
takes the (X,y) output of a `TimeSeriesWindowTransformer` and calls the rest of the scikit-learn
pipeline spreading the (X,y)  to positional parameters.

Finally `CeruleoRegressor`, is a class similar to the `sklearn.compose.TransformedTargetRegressor`.
Takes the transformer, and a regressor scikit-learn pipeline or model, and automatically builds
a scikit-learn pipeline using `WindowedDatasetIterator` and `EstimatorWrapper`.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractRunToFailureCyclesDataset
from ceruleo.iterators.batcher import Batcher
from ceruleo.iterators.iterators import (
    NotWeighted,
    SampleWeight,
    WindowedDatasetIterator,
)
from ceruleo.iterators.shufflers import AbstractShuffler, NotShuffled
from ceruleo.transformation.functional.transformers import Transformer

import sklearn.pipeline as sk_pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics._scorer import get_scorer

logger = logging.getLogger(__name__)


class EstimatorWrapper(TransformerMixin, BaseEstimator):
    """Wrapper around sklearn estimators to allow calling the fit and predict

    The transformer keeps the X and y together. This wrapper
    divide the X,y and call the fit(X,y) and predict(X,y) of the estimator

    Parameters:

        estimator: A scikit-learn estimator
    """

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, Xy: Tuple[np.ndarray, np.ndarray], y=None, **fit_params):
        X, y = Xy
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, Xy, **transform_params):
        X, y = Xy
        return self.estimator.predict(X, **transform_params)


class TimeSeriesWindowTransformer(TransformerMixin, BaseEstimator):
    """A scikit-learn transformer for obtaining a windowed time-series from the run-to-cycle failures

    Parameters:

        transformer:
        window_size: Window size of the iterator
        step: Stride of the iterators
        horizon: Horizon of the predictions
        shuffler: Shuffler of the data
        sample_weight: Sample weights callable
        right_closed: Wether to include the last point in each sliding window
        padding: Wether to pad when the lookback window is not complete
    """

    def __init__(
        self,
        transformer: Transformer,
        window_size: int,
        step: int = 1,
        horizon: int = 1,
        shuffler: AbstractShuffler = NotShuffled(),
        sample_weight: SampleWeight = NotWeighted(),
        right_closed: bool = True,
        padding: bool = True,
    ):
        self.transformer = transformer
        self.window_size = window_size
        self.horizon = horizon
        self.step = step
        self.shuffler = shuffler
        self.sample_weight = sample_weight
        self.right_closed = right_closed
        self.padding = padding

    def fit(self, dataset: AbstractRunToFailureCyclesDataset):
        self.transformer.fit(dataset)
        return self

    def _iterator(self, dataset: AbstractRunToFailureCyclesDataset):
        return WindowedDatasetIterator(
            dataset.map(self.transformer),
            self.window_size,
            self.step,
            self.horizon,
            shuffler=self.shuffler,
            sample_weight=self.sample_weight,
            right_closed=self.right_closed,
            padding=self.padding,
        )

    def transform(self, dataset: AbstractRunToFailureCyclesDataset):
        X, y, sw = self._iterator(dataset).get_data()
        return X, y.ravel()

    def true_values(self, dataset: AbstractRunToFailureCyclesDataset):
        X, y, sw = self._iterator(dataset).get_data()
        return y.ravel()
    
    def get_params(self, deep=None):
        params = super().get_params(deep)
        if deep:
            params['ts_window_transformer__transformer'] = self.transformer.get_params(deep)
        return params


class CeruleoRegressor(RegressorMixin, BaseEstimator):
    """A regressor wrapper similar to sklearn.compose.TransformedTargetRegressor


    Parameters:

        features_transformer: The transformer
        regressor: A scikit-learn regressor
    """

    def __init__(
        self, ts_window_transformer: TimeSeriesWindowTransformer, regressor, **kwargs
    ):
        self.ts_window_transformer = ts_window_transformer
        self.regressor = regressor
        self._update()

    def _update(self):
        self.wrapped_regressor = EstimatorWrapper(self.regressor)

        self.pipe = sk_pipeline.make_pipeline(
            self.ts_window_transformer, self.wrapped_regressor
        )

    def fit(self, dataset: AbstractRunToFailureCyclesDataset, y=None):
        self.pipe.fit(dataset)
        return self

    def predict(self, data):
        if isinstance(data, np.ndarray):
            return self.regressor.predict(data)
        else:
            return self.pipe.predict(data)

    def score(self, dataset, y=None):
        X, y = self.ts_window_transformer.transform(dataset)
        return super().score(dataset, y)

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self._update()
        return self


class CeruleoMetricWrapper:
    """A wrapper around sklearn metrics

        Example:

            grid_search = GridSearchCV(
                estimator=regressor_gs,
                param_grid={
                    'regressor': [RandomForestRegressor(max_depth=5)]
                },
                scoring=CeruleoMetricWrapper('neg_mean_absolute_error')
    )
    """

    def __init__(self, scoring):
        if callable(scoring):
            self.scorers = scoring
        elif isinstance(scoring, str):
            self.scorers = get_scorer(scoring)

    def __call__(self, estimator: CeruleoRegressor, dataset, y=None):
        X, y = estimator.ts_window_transformer.transform(dataset)
        return self.scorers(estimator, X, y)


def train_model(
    model,
    train_iterator: WindowedDatasetIterator,
    val_windowed_iterator: Optional[WindowedDatasetIterator] = None,
    **fit_kwargs
):
    """Fit the model with the given dataset iterator

    Parameters:

        train_iterator:


    Keyword arguments:

        fit_kwargs: Arguments for the fit method

    Returns:

        model: SKLearn model
    """
    X, y, sample_weight = train_iterator.get_data()

    params = {}
    try:
        from xgboost import XGBRegressor

        if val_windowed_iterator is not None and isinstance(model, XGBRegressor):
            X_val, y_val, _ = val_windowed_iterator.get_data()
            fit_kwargs.update({"eval_set": [(X_val, y_val)]})
    except Exception as e:
        logger.error(e)

    return model.fit(X, y.ravel(), **fit_kwargs, sample_weight=sample_weight)


def predict(model, dataset_iterator: WindowedDatasetIterator):
    """Get the predictions for the given iterator

    Parameters:

        dataset_iterator: Dataset iterator from which obtain data to predict

    Returns:

        array: Array with the predictiosn
    """
    X, _, _ = dataset_iterator.get_data()
    return model.predict(X)


def fit_batch(model, train_batcher: Batcher, val_batcher: Batcher, n_epochs=15):
    """Fit the model using the given batcher

    Parameters:

        model: SKLearn Model
        train_batcher: Train dataset batcher
        val_batcher: Validation dataset batcher
        n_epochs: Number of epochs, by default 15

    Returns:

        model: the model
        history: history of errors
    """

    history = []
    for r in range(n_epochs):
        for X, y in train_batcher:
            model.partial_fit(np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])), y)
        y_true, y_pred = predict(val_batcher)
        history.append(mean_squared_error(y_true, y_pred))
    return model, history


def predict_batch(model, dataset_batcher: Batcher):
    """Predict the values using the given batcher

    Parameters:

        model: SKLearn model
        dataset_batcher: The batcher


    Returns:

        RUL_predicted: Predictions array
    """
    y_pred = []
    for X, y in dataset_batcher:
        y_pred.extend(
            np.squeeze(
                model.predict(np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))
            )
        )
    return np.squeeze(np.array(y_pred))
