import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RandomAugmentantor(BaseEstimator, TransformerMixin):
    def __init__(self, scale, augment_proability=0.2):
        self.scale = scale
        self.augment_proability = augment_proability

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        t = pd.DataFrame(np.random.normal(0, scale=self.scale, size=X.shape),
                         index=X.index,
                         columns=X.columns)
        return X + t
