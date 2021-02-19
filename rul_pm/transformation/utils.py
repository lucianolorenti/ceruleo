from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.pipeline import FeatureUnion, _transform_one


class PandasToNumpy(TransformerStep):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class IdentityTransformer(TransformerStep):

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array*1


class PandasTransformerWrapper(TransformerStep):
    def __init__(self, transformer, name: Optional[str] = None):
        super().__init__(name)
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
