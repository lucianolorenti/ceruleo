from typing import List, Optional
from ceruleo.transformation.functional.transformerstep import TransformerStep
from copy import copy
from pyinform import active_info, block_entropy, entropy_rate
import pandas as pd
import numpy as np


ENTROPY_MEASURES_VALID_STATS = [
    "local_active_information",
    "local_block_entropy",
    "local_entropy_rate",
]


class LocalEntropyMeasures(TransformerStep):
    """Compute diverse entropy measures

    For each feature present in the life a number of feature will be computed for each time stamp

    The possible features are:

    - Local Active Information
    - Local Block Entropy
    - Local Entropy Rate


    Parameters
    ----------
    min_points : int, optional
        The minimun number of points of the expanding window, by default 2
    to_compute : List[str], optional
        List of the features to compute, by default None
        Valid values are:
            'local_active_information'          
            'local_block_entropy'
            'local_entropy_rate'
    name : Optional[str], optional
        Name of the step, by default None

    """

    def __init__(
        self, window: int = 2, to_compute: List[str] = None, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.window = window

        if to_compute is None:
            self.to_compute = copy(ENTROPY_MEASURES_VALID_STATS)
        else:
            for f in to_compute:
                if f not in ENTROPY_MEASURES_VALID_STATS:
                    raise ValueError(
                        f"Invalid feature to compute {f}. Valids are {ENTROPY_MEASURES_VALID_STATS}"
                    )
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self


    def _local_active_information(self, s: pd.Series):
        return active_info(s.values, self.window, local=True)

    def _local_block_entropy(self, s: pd.Series):
        return block_entropy(s.values, self.window, local=True)

    def _local_entropy_rate(self, s: pd.Series):
        return entropy_rate(s.values, self.window, local=True)

    def transform(self, X):

        X_new_n_columns = len(X.columns) * len(self.to_compute)
        i = 0

        columns = np.empty((X_new_n_columns,), dtype=object)
        for c in X.columns:
            for stats in self.to_compute:
                columns[i] = f"{c}_{stats}"
                i += 1
        
        


        data = np.empty((len(X.index), len(columns),))
        data[:] = np.nan
        X_new = pd.DataFrame(data, index=X.index, columns=columns)

        for c in X.columns:
            for stats in self.to_compute:
                data = np.squeeze(getattr(self, f"_{stats}")(X[c]))
                X_new.loc[:, f"{c}_{stats}"].iloc[-len(data):] = data
        X_new[np.isinf(X_new) | np.isnan(X_new)] = np.nan
        return X_new
