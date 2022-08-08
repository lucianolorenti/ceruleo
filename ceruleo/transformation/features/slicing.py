from typing import Union

import pandas as pd
from ceruleo.iterators.iterators import (RelativePosition, RelativeToEnd,
                                         RelativeToStart)
from ceruleo.transformation import TransformerStep


class SliceRows(TransformerStep):
    """Slice the run-to-failure cycle 
    
    Parameters:

        initial: The initial position
        final: The final position
    
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

        if isinstance(self.initial, RelativePosition):
            initial = self.initial.get(X.shape[0])
        else:
            initial = self.initial
        if isinstance(self.final, RelativePosition):
            final = self.final.get(X.shape[0])
        else:
            final = self.final

        return X.iloc[initial:final, :].copy()
