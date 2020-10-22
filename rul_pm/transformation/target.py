from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class PicewiseRUL(BaseEstimator, TransformerMixin):
    def __init__(self, method='median'):
        self.method = method
        

    def fit(self, X, y=None):
        self.max_life = np.quantile(X, 0.75)
        return self

    def transform(self, X):
        return np.clip(x, 0, self.max_life)
