from rul_pm.transformation.pipeline import LivesPipeline
from sklearn.base import BaseEstimator, TransformerMixin


class TransformerStep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def __call__(self, step):
        from rul_pm.transformation.utils import PandasFeatureUnion

        if isinstance(step, TransformerStep) or isinstance(step, PandasFeatureUnion):
            return LivesPipeline(steps=[
                (f'{step.__class__.__name__}_1', step),
                (f'{self.__class__.__name__}_2', self),
            ])

        elif isinstance(step, LivesPipeline):
            pipe = step
            i = len(pipe.steps) + 1
            pipe.steps.append(
                (f'{self.__class__.__name__}_{i}', self)
            )
            return pipe
