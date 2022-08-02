from typing import Optional, Union


import pandas as pd
from temporis.transformation import TransformerStep
from temporis.transformation.features.tdigest import TDigest
import numpy as np
from scipy.signal import find_peaks
from temporis.iterators.iterators import (
    RelativePosition,
    RelativeToEnd,
    RelativeToStart,
)


class SliceRows(TransformerStep):
    """Center the data with respect to the mean"""

    def __init__(
        self,
        initial: Union[int, RelativePosition] = RelativeToStart(0),
        final: Union[int, RelativePosition] = RelativeToEnd(0),
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.initial = initial
        self.final = final

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Center the input life

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the same index as the input with the
            data centered with respect to the mean of the fiited dataset
        """
        if isinstance(self.initial, RelativePosition):
            initial = self.initial.get(X.shape[0])
        else:
            initial = self.initial
        if isinstance(self.final, RelativePosition):
            final = self.final.get(X.shape[0])
        else:
            final = self.final

        return X.iloc[initial:final, :].copy()
