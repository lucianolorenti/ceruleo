from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from temporis.iterators.batcher import Batcher
from temporis.iterators.iterators import WindowedDatasetIterator
from tqdm.auto import tqdm
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import pickle


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
    if val_windowed_iterator is not None and isinstance(model, XGBRegressor):
        X_val, y_val, _ = val_windowed_iterator.get_data()
        fit_kwargs.update({'eval_set': [(X_val, y_val)]})

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
