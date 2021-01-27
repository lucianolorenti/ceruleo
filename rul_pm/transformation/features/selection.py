import logging
from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep

logger = logging.getLogger(__name__)


class NullProportionSelector(TransformerStep):
    def __init__(self, min_null_proportion=0.5, name: Optional[str] = None):
        super().__init__(name)
        self.min_null_proportion = min_null_proportion

    def fit(self, X, y=None):
        logger.info(f'Features before NullProportionSelector {X.shape[1]}')
        self.not_null_proportion = np.mean(np.isfinite(X), axis=0)
        self.mask = self.not_null_proportion > self.min_null_proportion

        logger.info(
            f'Features before NullProportionSelector {np.sum(self.mask)}')
        return self

    def transform(self, X):

        return X.loc[:, self.mask].copy()


class ByNameFeatureSelector(TransformerStep):

    def __init__(self, features=[], name: Optional[str] = None):
        super().__init__(name)
        self.features = features
        self.features_indices = None
        self.features_computed_ = []

    def partial_fit(self, df, y=None):
        if len(self.features) > 0:
            features = [f for f in self.features if f in set(df.columns)]
        else:
            features = list(set(df.columns))

        if len(self.features_computed_) == 0:
            self.features_computed_ = features
        else:
            self.features_computed_ = [
                f for f in self.features_computed_ if f in features]
        return self

    def fit(self, df, y=None):
        if len(self.features) > 0:
            features = [f for f in self.features if f in set(df.columns)]
        else:
            features = list(set(df.columns))
        self.features_computed_ = sorted(features)
        return self

    def transform(self, X):
        return X.loc[:, self.features_computed_].copy()

    @property
    def n_features(self):
        return len(self.features_computed_)


class LocateFeatures(TransformerStep):
    def __init__(self, features, name: Optional[str] = None):
        super().__init__(name)
        self.features = features

    def transform(self, X):
        cols = list(X.columns)
        for name, pos in self.features.items():
            a, b = cols.index(name), pos
            cols[b], cols[a] = cols[a], cols[b]
            X = X[cols]
        return X


class DiscardByNameFeatureSelector(TransformerStep):
    def __init__(self, features=[], name: Optional[str] = None):
        super().__init__(name)
        self.features = features
        self.features_indices = None

    def fit(self, df, y=None):
        self.feature_columns = [
            f for f in df.columns if f not in self.features]
        return self

    def transform(self, X):
        return X.loc[:, self.feature_columns]

    @property
    def n_features(self):
        return len(self.features)


class PandasVarianceThreshold(TransformerStep):
    def __init__(self, min_variance: float, name: Optional[str] = None):
        super().__init__(name)
        self.min_variance = min_variance
        self.selected_columns_ = None

    def partial_fit(self, X, y=None):
        variances_ = X.var(skipna=True)
        partial_selected_columns_ = X.columns[variances_ > self.min_variance]
        if self.selected_columns_ is not None and len(partial_selected_columns_) < len(self.selected_columns_)*0.5:
            logger.warning(type(self).__name__)
            logger.warning(
                f'Life removed more than a half of the columns. Shape {X.shape}')
            logger.warning(
                f'Current: {len(self.selected_columns_)}. New ones: {len(partial_selected_columns_)}')
        if self.selected_columns_ is None:
            self.selected_columns_ = partial_selected_columns_
        else:
            self.selected_columns_ = [
                f for f in self.selected_columns_ if f in partial_selected_columns_]
        if len(self.selected_columns_) == 0:
            logger.warning(type(self).__name__)
            logger.warning('All features were removed')
        return self

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        self.variances_ = X.var(skipna=True)
        self.selected_columns_ = X.columns[self.variances_ > self.min_variance]
        logger.debug(
            f'Dropped columns {[c for c in X.columns if c not in self.selected_columns_]}')
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return X[self.selected_columns_].copy()


class PandasNullProportionSelector(TransformerStep):
    def __init__(self, max_null_proportion: float, name: Optional[str] = None):
        super().__init__(name)
        self.max_null_proportion = max_null_proportion
        self.selected_columns_ = None

    def partial_fit(self, X, y=None):
        null_proportion = X.isnull().mean()

        partial_selected_columns_ = X.columns[null_proportion <
                                              self.max_null_proportion]
        if self.selected_columns_ is not None and len(partial_selected_columns_) < len(self.selected_columns_)*0.5:
            logger.warning(type(self).__name__)
            logger.warning(
                f'Life removed more than a half of the columns. Shape {X.shape}')
            logger.warning(
                f'Current: {len(self.selected_columns_)}. New ones: {len(partial_selected_columns_)}')
        if self.selected_columns_ is None:
            self.selected_columns_ = partial_selected_columns_
        else:
            self.selected_columns_ = [
                f for f in self.selected_columns_ if f in partial_selected_columns_]
        if len(self.selected_columns_) == 0:
            logger.warning(type(self).__name__)
            logger.warning('All features were removed')
        return self

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        self.null_proportion = X.isnull().mean()
        self.selected_columns_ = X.columns[self.null_proportion <
                                           self.max_null_proportion]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return X[self.selected_columns_].copy()
