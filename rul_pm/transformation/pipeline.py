from typing import Optional

from numpy.lib.arraysetops import isin
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from sklearn.pipeline import Pipeline


def transform_given_list(X, steps):
    for t in steps:
        if t == 'passthrough':
            continue
        X = t.transform(X)
    return X


class LivesPipeline(Pipeline):
    def fit(self, X, y=None, apply_before=None):

        if not isinstance(X, AbstractLivesDataset):
            super().fit(X, y)

        from rul_pm.transformation.featureunion import PandasFeatureUnion

        estimators = apply_before.copy() if apply_before is not None else []
        for _, est in self.steps:

            if est == 'passthrough':
                continue
            if isinstance(est, LivesPipeline):
                est.fit(X, apply_before=estimators)
            elif isinstance(est, PandasFeatureUnion):
                est.fit(X, apply_before=estimators)
            else:
                for life in X:
                    est.partial_fit(transform_given_list(life, estimators))
            estimators.append(est)
