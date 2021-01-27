from copy import copy
from typing import Optional

from numpy.lib.arraysetops import isin
from rul_pm.transformation.pipeline import LivesPipeline
from sklearn.base import BaseEstimator, TransformerMixin


class Concatenate:
    def __init__(self, name=None):
        self.name_ = name
        self.prevs = []
        self.step_names = None

    @property
    def name(self):
        if self.name_ is not None:
            return self.name_
        else:
            return self.__class__.__name__

    def __call__(self, prevs: list, names: Optional[list] = None):
        self.step_names = copy(names)
        for prev in prevs:
            self.prevs.append(prev)
        return self

    def build(self, parent_pipe: LivesPipeline = None):
        from rul_pm.transformation.utils import PandasFeatureUnion

        prev_steps = None
        if self.step_names is not None:
            prev_steps = zip(self.step_names, self.prevs)
        else:
            prev_steps = zip(
                [f'step_{i}' for i, n in enumerate(self.prevs)],
                self.prevs)
        union = []
        for name, prev in prev_steps:
            union.append((name, prev.build()))

        union = PandasFeatureUnion(transformer_list=union)
        if parent_pipe is None:
            return union
        else:
            i = len(parent_pipe.steps) + 1
            name = self.name
            parent_pipe.steps.insert(
                0,
                (f'{name}_{i}', union))
            return parent_pipe


class TransformerStep(BaseEstimator, TransformerMixin):
    def __init__(self, name=None):
        self.name_ = name
        self.prevs = []

    @ property
    def name(self):
        if self.name_ is not None:
            return self.name_
        else:

            return self.__class__.__name__

    def add_prev(self, child):
        self.prevs.append(child)

    def __call__(self, prev):
        step = self
        step.add_prev(prev)
        return step

    def build(self, parent_pipe: LivesPipeline = None) -> LivesPipeline:
        if parent_pipe is None:
            parent_pipe = LivesPipeline(steps=[('initial', 'passthrough')])
        i = len(parent_pipe.steps) + 1
        name = self.name
        parent_pipe.steps.insert(0, (f'{name}_{i}', self))
        if len(self.prevs) > 0:
            self.prevs[0].build(parent_pipe=parent_pipe)
        return parent_pipe

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self
