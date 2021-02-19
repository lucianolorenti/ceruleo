from numpy.lib.arraysetops import isin
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from sklearn.pipeline import Pipeline


class LivesPipeline(Pipeline):
    def fit(self, X, y=None, apply_before=None):
        def _transform_given_list(X, steps):
            for t in steps:
                if t == 'passthrough':
                    continue
                X = t.transform(X)
            return X

        if not isinstance(X, AbstractLivesDataset):
            super().fit(X, y)

        estimators = apply_before.copy() if apply_before is not None else []
        for name, est in self.steps:

            if est == 'passthrough':
                continue
            if isinstance(est, LivesPipeline):
                est.fit(X, apply_before=estimators)
            else:
                for life in X:
                    est.partial_fit(_transform_given_list(life, estimators))
            estimators.append(est)
