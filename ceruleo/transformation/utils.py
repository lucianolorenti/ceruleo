from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ceruleo.transformation import TransformerStep
from ceruleo.transformation.features.tdigest import TDigest
from sklearn.pipeline import FeatureUnion, _transform_one


class PandasToNumpy(TransformerStep):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class TransformerLambda(TransformerStep):
    def __init__(self, f, name: Optional[str] = None):
        super().__init__(name)
        self.f = f

    def transform(self, X, y=None):
        return self.f(X)


class IdentityTransformerStep(TransformerStep):
    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array * 1


class SKLearnTransformerWrapper(TransformerStep):
    def __init__(self, transformer, name: Optional[str] = None):
        super().__init__(name)
        self.transformer = transformer

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        self.transformer.fit(X.values)
        return self

    def partial_fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        if hasattr(self.transformer, "partial_fit"):
            self.transformer.partial_fit(X.values)
        else:
            self.transformer.fit(X.values)
        return self

    def _column_names(self, X) -> List[str]:
        return X.columns

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return pd.DataFrame(
            self.transformer.transform(X), columns=self._column_names(X), index=X.index
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
    for w in range(1, window + 1):
        for c in columns:
            new_columns.append(f"w_{w}_{c}")
    return new_columns


def build_tdigest(tdigest, values, column):
    return column, tdigest.merge_sorted(values)


class QuantileEstimator:
    """Approximate the quantile of each feature in the dataframe
       using t-digest
    """
    def __init__(self, tdigest_size:int = 200, max_workers:int = 1, subsample:Optional[Union[int, float]] = None):
        self.tdigest_dict = None
        self.tdigest_size = tdigest_size
        self.max_workers = max_workers
        self.subsample = subsample

    def update(self, X: pd.DataFrame):
        if X.shape[0] < 2:
            return self

        columns = X.columns
        
        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest(self.tdigest_size) for c in columns}

        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for i, c in enumerate(columns):

                x = X.iloc[:, i].dropna()
                if self.subsample is not None:

                    if isinstance( self.subsample, int):
                        points_to_sample = self.subsample
                    else:
                        points_to_sample = self.subsample * X.shape[0]
                    step = max(int(X.shape[0] / float(points_to_sample)), 1)
                    x = x.iloc[::step]
                x = x.sort_values().values
                t_digest = self.tdigest_dict[c]
                results.append(executor.submit(build_tdigest, t_digest, x, c))

        for r in results:
            c, tdigest = r.result()
            self.tdigest_dict[c] = tdigest
        return self

    def estimate_quantile(self, *args, **kwargs):
        return self.quantile(*args, **kwargs)
        
    def quantile(
        self, q: float, feature: Optional[str] = None
    ) -> Union[pd.Series, float]:
        """Estimate the quantile for a set of features
        
        Parameters
        ----------
        q:float
          The quantile to estimate
        feature:Optional[Str] """
        if feature is not None:
            return self.tdigest_dict[feature].estimate_quantile(q)
        else:
            return pd.Series(
                {
                    c: self.tdigest_dict[c].estimate_quantile(q)
                    for c in self.tdigest_dict.keys()
                }
            )

class QuantileComputer:
    """Approximate the quantile of each feature in the dataframe
       using t-digest
    """
    def __init__(self, subsample_rate:float = 1):
        self.values = None
        self.tdigest_size = subsample_rate

    def update(self, X: pd.DataFrame):
        if X.shape[0] < 2:
            return self

        columns = X.columns
        
        if self.values_dict is None:
            self.values = X.copy()
        else:
            self.values = pd.concat(self.values, X)

        
        return self

    def quantile(
        self, q: float, feature: Optional[str] = None
    ) -> Union[pd.Series, float]:
        if feature is not None:
            return self.values.quantile(q)
        else:
            return self.values[feature].quantile(q)


class Literal(TransformerStep):
    def __init__(self, literal, *args):
        super().__init__(*args)
        self.literal = literal 
        
    def transform(self, X):
        return self.literal


def ensure_step(step):
    if isinstance(step, TransformerStep):
        return step    
    return Literal(step)
