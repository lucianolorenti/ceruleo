import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class NullProportionSelector(BaseEstimator, TransformerMixin):
    def __init__(self, min_null_proportion=0.5):
        self.min_null_proportion = min_null_proportion

    def fit(self, X, y=None):
        logger.info(f'Features before NullProportionSelector {X.shape[1]}')
        self.not_null_proportion = np.mean(np.isfinite(X), axis=0)
        self.mask = self.not_null_proportion > self.min_null_proportion

        logger.info(
            f'Features before NullProportionSelector {np.sum(self.mask)}')
        return self

    def transform(self, X):

        return X[:, self.mask]


class ByNameFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=[]):
        self.features = features
        self.features_indices = None

    def fit(self, df, y=None):
       
        if len(self.features) > 0:
            features = [f for f in self.features if f in set(df.columns)]
        else:
            features = list(set(df.columns))
        self.features_computed_ = sorted(features)
        return self

    def transform(self, X):
        return X.loc[:, self.features_computed_]

    @property
    def n_features(self):
        return len(self.features_computed_)


class LocateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = list(X.columns)
        for name, pos in self.features.items():
            a, b = cols.index(name), pos
            cols[b], cols[a] = cols[a], cols[b]
            X = X[cols]
        return X


class DiscardByNameFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=[]):
        self.features = features
        self.features_indices = None

    def fit(self, df, y=None):
        self.feature_columns = [f for f in df.columns if f not in self.features]
        return self

    def transform(self, X):
        return X.loc[:, self.feature_columns]

    @property
    def n_features(self):
        return len(self.features)


class PandasVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        self.variances_ = X.var(skipna=False)
        self.selected_columns_ = X.columns[self.variances_ > self.t]

        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return X[self.selected_columns_].copy()


class PandasNullProportionSelector(BaseEstimator, TransformerMixin):
    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        self.null_proportion = X.isnull().mean()
        self.selected_columns_ = X.columns[self.null_proportion < self.t]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return X[self.selected_columns_].copy()
