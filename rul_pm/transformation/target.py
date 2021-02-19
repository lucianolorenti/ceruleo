from typing import Optional

import numpy as np
from rul_pm.transformation.transformerstep import TransformerStep
from rul_pm.transformation.utils import IdentityTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline


class HealthPercentage(TransformerStep):
    def transform(self, X):
        return (X / X.iloc[0, 0])*100


class PicewiseRUL(TransformerStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.max_life = np.inf

    def transform(self, X):
        return np.clip(X, 0, self.max_life)


class PicewiseRULQuantile(PicewiseRUL):
    def __init__(self, quantile, name: Optional[str] = None):
        super().__init__(name)
        self.quantile = quantile

    def fit(self, X, y=None):
        self.max_life = np.quantile(X, self.quantile)
        return self


class PicewiseRULThreshold(PicewiseRUL):
    """
    Clip the RUL by a predefined threshold

    target = np.min(target, max_life)

    Parameters
    ----------
    max_life:float


    """

    def __init__(self, max_life: float, name: Optional[str] = None):
        super().__init__(name)
        self.max_life = max_life


class TTEBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X < 0.2


def combined_target_pipeline(preprocess):
    return Pipeline(
        [
            ('preprocess', preprocess if preprocess is not None else 'passthrough'),
            ('union', FeatureUnion(transformer_list=[
                ("RUL", Pipeline([
                    ('selector', IdentityTransformer()),
                ])),
                ("TTF", Pipeline([
                    ('binarizer', TTEBinarizer())
                ]))
            ]))
        ])


class TargetIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) > 1:
            return X.iloc[:, -1]
        else:
            return X
