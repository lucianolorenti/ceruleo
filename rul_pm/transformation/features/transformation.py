
import logging
from typing import Optional

import emd
import numpy as np
import pandas as pd
from rul_pm.transformation.features.extraction import (compute, roll_matrix,
                                                       stats_order)
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


class MeanCentering(TransformerStep):
    def __init__(self):
        super().__init__()
        self.N = 0
        self.sum = None

    def fit(self, X, y=None):
        self.mean = X.mean()

    def partial_fit(self, X, y=None):
        if self.sum is None:
            self.sum = X.sum()
        else:
            self.sum += X.sum()

        self.N += X.shape[0]
        self.mean = self.sum / self.N

    def transform(self, X):
        return X - self.mean


class MeanFilter(TransformerStep):
    def __init__(self,
                 window: int,
                 min_periods: int = 15,
                 name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def transform(self, X, y=None):
        return X.rolling(self.window,
                         min_periods=self.min_periods).mean(skip_na=True)