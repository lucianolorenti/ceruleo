import numpy as np
import pandas as pd
from rul_pm.iterators.batcher import Batcher
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.models.model import TrainableModel
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error


class SKLearnModel(TrainableModel):
    """Wrapper around scikit-learn models

    Parameters
    ----------
    model: BaseEstimator
        A scikit-learn model
    """

    def __init__(self, model: BaseEstimator, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        if not hasattr(self.model, "fit"):
            raise ValueError("Model must allow to fit")

    def build_model(self):
        return self._model

    def fit(self, train_iterator: WindowedDatasetIterator, **kwargs):
        """Fit the model with the given dataset iterator

        Parameters
        ----------
        train_iterator : WindowedDatasetIterator
            Dataset iterator from which obtain data to fit

        Keyword arguments
        -----------------
        kwargs:
            Arguments for the fit method

        Returns
        -------
        SKLearnModel
            self
        """
        X, y, sample_weight = train_iterator.get_data()
        self.model.fit(X, y.ravel(), **kwargs, sample_weight=sample_weight)
        return self

    def predict(self, dataset_iterator: WindowedDatasetIterator):
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
        return self.model.predict(X)

    def get_params(self, deep: bool = False) -> dict:
        """Obtain the model parameters

        Parameters
        ----------
        deep : bool, optional
            Wether to obtain the parameters for each element, by default False

        Returns
        -------
        dict
            Model parameters
        """
        out = super().get_params(deep=deep)
        out["model"] = self.model
        if deep and hasattr(self.model, "get_params"):
            for key, value in self.model.get_params(deep=True).items():
                out["model__%s" % key] = value
        return out

    def set_params(self, **params):
        ## TODO Check invalid parameters
        model_params = {}
        for name, value in params.items():
            if "__" in name:
                model_params[name.split("__")[1]] = value
        for name in model_params.keys():
            params.pop(f"model__{name}")

        super().set_params(**params)
        self.model.set_params(**model_params)
        return self


class BatchSKLearnModel(TrainableModel):
    """A wrapper around scikit-learn models that allows partial_fit

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The keras model
    """

    def __init__(self, model: BaseEstimator):
        super().__init__()
        self._model = model
        if not hasattr(self.model, "partial_fit"):
            raise ValueError("Model must allow to partial_fit")

    def build_model(self):
        return self._model

    def fit(self, train_batcher: Batcher, val_batcher: Batcher, n_epochs=15):
        """Fit the model using the given batcher

        Parameters
        ----------
        train_batcher : Batcher
            Train dataset batcher
        val_batcher : Batcher
            Validation dataset batcher
        n_epochs : int, optional
            Number of epochs, by default 15

        Returns
        -------
        BatchSKLearnModel
            self
        """

        self.history = []
        for r in range(n_epochs):
            for X, y in train_batcher:
                self.model.partial_fit(
                    np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])), y
                )
            y_true, y_pred = self.predict(val_batcher)
            self.histroy.append(mean_squared_error(y_true, y_pred))
        return self

    def predict(self, dataset_batcher: Batcher):
        """Predict the values using the given batcher

        Parameters
        ----------
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
                    self.model.predict(
                        np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
                    )
                )
            )
        return y_pred
