
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
