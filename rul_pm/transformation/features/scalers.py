import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, r: tuple):
        self.range = range
        self.min = r[0]
        self.max = r[1]

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        return ((X-self.data_min)/(self.data_max-self.data_min) * (self.max - self.min)) + self.min
