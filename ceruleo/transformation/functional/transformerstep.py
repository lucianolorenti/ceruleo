"""Transformer step is the base class of all transformers

The pipeline will use the steps to fit and transform the run-to-failure
cycles
"""
from copy import copy
from typing import List, Optional

import pandas as pd
from ceruleo.transformation.functional.mixin import TransformerStepMixin
from sklearn.base import TransformerMixin


class TransformerStep(TransformerStepMixin, TransformerMixin):
    """Base class of all transformation step

    """
    def partial_fit(self, X:pd.DataFrame, y=None) -> "TransformerStep":
        """Fit a single run-to-failure cycle

        Parameters:

            X: Features of the run-to-failure cycle


        Returns:
            TransformerStep: The same step
        """
        return self

    def fit(self, X, y=None)  -> "TransformerStep":
        """Fit the complete set of run-to-failure cycles

        Parameters:

            X: Features of the all the run-to-failure cycles


        Returns:
            TransformerStep: The same step
        """
        return self

    def find_feature(self, X: pd.DataFrame, name: str) -> Optional[str]:
        """Find the feature that best maches the columns in X

        Parameters:
            X: A run-to-failure cycle
            name: The name of the feature to find

        Returns:
            The name of the columns if it was found, else None
        
        """
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
