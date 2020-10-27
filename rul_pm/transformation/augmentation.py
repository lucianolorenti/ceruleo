from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RandomAugmentantor(BaseEstimator, TransformerMixin):
    def __init__(self, scale, augment_proability=0.2):
        self.scale = scale 
        self.augment_proability = augment_proability

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if np.random.rand() < self.augment_proability:
            return X + np.random.normal(0, scale=self.scale)
        else:
            return X
        


