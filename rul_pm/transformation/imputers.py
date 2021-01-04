import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class PandasRemoveInf(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.replace([np.inf, -np.inf], np.nan)

    def partial_fit(self, X, y=None):
        return self


class PandasMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median = X.median(axis=0).to_dict()
        return self

    def transform(self, X, y=None):
        return X.fillna(value=self.median)

    def partial_fit(self, X, y=None):
        return self


class PandasMeanImputer(BaseEstimator, TransformerMixin):
    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        self.mean = X.mean(axis=0).to_dict()
        return self

    def transform(self, X, y=None):
        return X.fillna(value=self.mean)


class RollingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, func):
        self.window_size = window_size
        self.function = func

    def fit(self, X, y=None):
        self.default_value = np.mean(X,  axis=0)
        self.default_value[~np.isfinite(self.default_value)] = 0
        return self

    def transform(self, X):
        X = X.copy()
        row, features = np.where(~np.isfinite(X))
        min_limit = np.maximum(row - self.window_size, 0)
        max_limit = np.minimum(row + self.window_size, X.shape[0])
        for r, min_r, max_r, f in zip(row, min_limit, max_limit, features):
            X[r, f] = self.function(X[min_r:max_r, f])
            if ~np.isfinite(X[r, f]):
                X[r, f] = self.default_value[f]
        return X

    def partial_fit(self, X, y=None):
        return self


class RollingMedianImputer(RollingImputer):
    def __init__(self, window_size):
        super().__init__(window_size, np.median)


class RollingMeanImputer(RollingImputer):
    def __init__(self, window_size):
        super().__init__(window_size, np.mean)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return X.ffill()

    def partial_fit(self, X, y=None):
        return self
