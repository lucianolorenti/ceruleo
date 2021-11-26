from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from ceruleo.transformation.functional.mixin import TransformerStepMixin



class Concatenate(TransformerStepMixin):
    """Join two transformers

    Parameters
    ----------
    add_prefix : bool, optional
        Whether to add prefix , by default True
    """
    def __init__(self, *, add_prefix:bool= True):

        super().__init__()
        self.add_prefix = add_prefix 


    def merge_dataframes_by_column(self, Xs):
        # indices = [X.index for X in Xs]
        # TODO: Check equal indices
        names = [n.name for n in self.previous]
        def new_feature_name(otherX, name, i):
            if self.add_prefix:
                return otherX.add_prefix(f'{name}_{i}_')
            else:
                return otherX
        X = pd.concat(
            [new_feature_name(otherX, name, i) for i, (name, otherX) in enumerate(zip(names, Xs))], axis=1
        )
        return X

    def transform(self, Xs: List[pd.DataFrame]):
        if not Xs:
            # All transformers are None
            return np.zeros((Xs[0].shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        return params
