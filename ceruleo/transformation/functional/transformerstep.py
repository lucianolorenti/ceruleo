from copy import copy
from typing import List, Optional

import pandas as pd
from ceruleo.transformation.functional.mixin import TransformerStepMixin
from sklearn.base import TransformerMixin


class TransformerStep(TransformerStepMixin, TransformerMixin):
    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def column_name(self, df: pd.DataFrame, cname: str):
        columns = [c for c in df.columns if cname in c]
        if len(columns) == 0:
            raise ValueError("{cname} is not present in the dataset")
        return columns[0]

    def find_feature(self, X: pd.DataFrame, name: str) -> Optional[str]:
        matches = [c for c in X.columns if name in c]
        if len(matches) > 0:
            return matches[0]
        else:
            return None

    def description(self):
        return f"{self.name}"




    def __add__(self, other):
        from ceruleo.transformation.features.operations import Sum
        from ceruleo.transformation.utils import ensure_step
        return Sum()([self, ensure_step(other)])

    def __truediv__(self, other):
        from ceruleo.transformation.features.operations import Divide
        from ceruleo.transformation.utils import ensure_step
        return Divide()([self, ensure_step(other)])
