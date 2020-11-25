import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LimitExceededCumSum(BaseEstimator, TransformerMixin):
    def __init__(self, lambda_=1.5, drop_original=False):
        self.lambda_ = lambda_
        self.UCL = None
        self.LCL = None
        self.drop_original = drop_original

    def fit(self, X, y=None):
        mean = np.nanmean(X, axis=0)
        s = np.sqrt(self.lambda_ / (2-self.lambda_))*np.nanstd(X, axis=0)
        self.UCL = mean + 3*s
        self.LCL = mean - 3*s
        return self

    def transform(self, X):
        mask = (
            (X < (self.LCL)) |
            (X > (self.UCL))
        )
        if self.drop_original:
            return np.cumsum(mask, axis=0)
        else:
            return np.concatenate((X, np.cumsum(mask, axis=0)), axis=1)
