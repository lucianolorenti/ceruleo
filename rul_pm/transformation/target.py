from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class PicewiseRUL(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class PicewiseRULQuantile(PicewiseRUL):
    def __init__(self, quantile):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.max_life_ = np.quantile(X, self.quantile)
        return self


class PicewiseRULThreshold(PicewiseRUL):
    def __init__(self, max_life):
        self.max_life_ = max_life

    def fit(self, X, y=None):
        self.max_life_ = np.maximum(X, self.max_life_)
        return self