from typing import Optional

import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep
import numpy as np


class PandasMinMaxScaler(TransformerStep):
    def __init__(self,
                 range: tuple,
                 name: Optional[str] = None,
                 clip: bool = True,
                 robust:bool = False):
        super().__init__(name)
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_min = None
        self.data_max = None
        self.clip = clip
        self.robust = robust

    def partial_fit(self, df, y=None):
        if self.robust:
            partial_data_min = df.quantile(0.45)
            partial_data_max = df.quantile(0.55)
        else:
            partial_data_min = df.min()
            partial_data_max = df.max()
        if self.data_min is None:
            self.data_min = partial_data_min
            self.data_max = partial_data_max
        else:
            self.data_min = (pd.concat([self.data_min, partial_data_min],
                                       axis=1).min(axis=1))
            self.data_max = (pd.concat([self.data_max, partial_data_max],
                                       axis=1).max(axis=1))
        return self

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        X = ((X - self.data_min) / (self.data_max - self.data_min) *
             (self.max - self.min)) + self.min
        if self.clip:
            X.clip(lower=self.min, upper=self.max, inplace=True)
        return X


class PandasStandardScaler(TransformerStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.std = None
        self.mean = None

    def partial_fit(self, df, y=None):
        partial_data_mean = df.mean()
        partial_data_std = df.std()
        if self.mean is None:
            self.mean = partial_data_mean
            self.std = partial_data_std
        else:
            self.mean = (pd.concat([self.mean, partial_data_mean],
                                   axis=1).mean(axis=1))
            self.std = (pd.concat([self.std, partial_data_std],
                                  axis=1).mean(axis=1))
        return self

    def fit(self, df, y=None):
        self.mean = df.mean()
        self.std = df.std()
        return self

    def transform(self, X):
        return (X - self.mean) / (self.std)


class PandasRobustScaler(TransformerStep):
    def __init__(self,
                 range: tuple,
                 name: Optional[str] = None,
                 clip: bool = True):
        super().__init__(name)
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_median = None
        self.data_max = None
        self.clip = clip

    def partial_fit(self, df, y=None):
        partial_data_median = df.median()
        partial_data_max = df.max()
        if self.data_min is None:
            self.data_min = partial_data_median
            self.data_max = partial_data_max
        else:
            self.data_median = (pd.concat([self.data_median, partial_data_median],
                                       axis=1).median(axis=1))
            self.data_max = (pd.concat([self.data_max, partial_data_max],
                                       axis=1).max(axis=1))
        return self

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        X = ((X - self.data_min) / (self.data_max - self.data_min) *
             (self.max - self.min)) + self.min
        if self.clip:
            X.clip(lower=self.min, upper=self.max, inplace=True)
        return X
        

class ScaleInvRUL(TransformerStep):
    """
    Scale binary columns according the inverse of the RUL.
    Usually this will be used before a CumSum transformation

    Parameters
    ----------
    rul_column: str
                Column with the RUL
    """
    def __init__(self, rul_column: str, name: Optional[str] = None):
        super().__init__(name)
        self.RUL_list_per_column = {}
        self.penalty = {}
        self.rul_column_in = rul_column
        self.rul_column = None

    def partial_fit(self, X: pd.DataFrame):
        if self.rul_column is None:
            self.rul_column = self.column_name(X, self.rul_column_in)
        columns = [c for c in X.columns if c != self.rul_column]
        for c in columns:
            mask = X[X[c] > 0].index
            if len(mask) > 0:
                RUL_list = self.RUL_list_per_column.setdefault(c, [])
                RUL_list.extend((1+(X[self.rul_column].loc[mask].values / X[self.rul_column].max())).tolist())

        for k in self.RUL_list_per_column.keys():

            self.penalty[k] = (1 / np.median(self.RUL_list_per_column[k]))

    def transform(self, X: pd.DataFrame):
        columns = [c for c in X.columns if c != self.rul_column]
        X_new = pd.DataFrame(index=X.index)
        for c in columns:
            if (c in self.penalty):
                X_new[c] = X[c] * self.penalty[c]
        return X_new
