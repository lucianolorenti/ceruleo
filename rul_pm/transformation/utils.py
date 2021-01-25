import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep
from scipy import sparse
from sklearn.pipeline import FeatureUnion, _transform_one


class PandasToNumpy(TransformerStep):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class IdentityTransformer(TransformerStep):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array*1


class PandasTransformerWrapper(TransformerStep):
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
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns, index=X.index)


class PandasFeatureUnion(FeatureUnion):

    def partial_fit(self,  X, y=None):
        self._validate_transformers()
        for name, trans, weight in self._iter():
            trans.partial_fit(X, y)

        return self

    def merge_dataframes_by_column(self, Xs):
        # indices = [X.index for X in Xs]
        # TODO: Check equal indices
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


class PandasConcatenate(TransformerStep):
    def __call__(self, steps):
        return PandasFeatureUnion(
            transformer_list=[(f'{step.__class__.__name__}_{i}', step)
                              for i, step in enumerate(steps)]
        )


def column_names_window(columns: list, window: int) -> list:
    """

    Parameters
    ----------
    columns: list
             List of column names

    window: int
            Window size

    Return
    ------
    Column names with the format: w_{step}_{feature_name}
    """
    new_columns = []
    for w in range(1, window+1):
        for c in columns:
            new_columns.append(f'w_{w}_{c}')
    return new_columns
