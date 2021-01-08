
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, range: tuple):
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_min = None
        self.data_max = None

    def partial_fit(self, df, y=None):
        partial_data_min = df.min()
        partial_data_max = df.max()
        if self.data_min is None:
            self.data_min = partial_data_min
            self.data_max = partial_data_max
        else:
            self.data_min = (pd
                             .concat([self.data_min, partial_data_min], axis=1)
                             .min(axis=1))
            self.data_max = (pd
                             .concat([self.data_max, partial_data_max], axis=1)
                             .max(axis=1))

        return self

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        return ((X-self.data_min)/(self.data_max-self.data_min) * (self.max - self.min)) + self.min
