from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasToNumpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class TargetIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) > 1:
            return X.iloc[:, -1].values
        else:
            return X.values


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array*1


class PandasTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns)


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
