from typing import Optional, Union


import pandas as pd
from ceruleo.transformation import TransformerStep
from ceruleo.transformation.features.tdigest import TDigest
import numpy as np
from scipy.signal import find_peaks
from ceruleo.iterators.iterators import (
    RelativePosition,
    RelativeToEnd,
    RelativeToStart,
)


class SliceRows(TransformerStep):
    """
    Slice portion of the run-to-failure cycle
    
    Parameters:
        initial: Initial position of the slice, default RelativeToStart(0)
        final: Final position of the slice, default RelativeToEnd(0)
    """

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
        """ 
        Slice the run-to-failure cycle

        Parameters:
            X: Input dataframe
        
        Returns:
            A dataframe with the sliced run-to-failure cycle
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
