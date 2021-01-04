import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class PandasToNumpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
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
        self.transformer.fit(X.values)
        return self

    def partial_fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        if hasattr(self.transformer, 'partial_fit'):
            self.transformer.partial_fit(X.values)
        else:
            self.transformer.fit(X.values)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input array must be a data frame')
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns)


def _partial_fit_transform_one(transformer,
                               X,
                               y,
                               weight,
                               message_clsname='',
                               message=None,
                               **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'partial_fit'):
            res = transformer.partial_fit(X, y, **fit_params).transform(X)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)

    if weight is None:
        return res, transformer
    return res * weight, transformer


def fit_transform(self, func, X, y=None, **fit_params):
    self._validate_transformers()
    result = [
        _partial_fit_transform_one(
            transformer=trans,
            X=X,
            y=y,
            weight=weight,
            **fit_params)
        for name, trans, weight in self._iter()]

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


class PandasFeatureUnion(FeatureUnion):

    def partial_fit(self,  X, y=None):
        self._validate_transformers()
        for name, trans, weight in self._iter():
            trans.partial_fit(X, y)
        return self

    def merge_dataframes_by_column(self, Xs):
        names = [name for name, _, _ in self._iter()]
        X = Xs[0]
        X.columns = [f'{names[0]}_{c}' for c in X.columns]
        for name, otherX in zip(names[1:], Xs[1:]):
            for c in otherX.columns:
                X[f'{name}_{c}'] = otherX[c]
        return X

    def transform(self, X):
        Xs = []
        for name, trans, weight in self._iter():
            Xs.append(_transform_one(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
