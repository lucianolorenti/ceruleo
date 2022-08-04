"""
In some cases It's useful to cast the datatype of subset of columns

"""
import logging
from typing import Optional

import pandas as pd
from ceruleo.transformation import TransformerStep

logger = logging.getLogger(__name__)


class CastTo(TransformerStep):
    """Cast to a given datatype

    Example:

        step = CastTo(type='float32')

    Parameters:

        type: Type name to convert to
        name: Name of the step

    """
    def __init__(self, *, type:str, name:Optional[str]=None):
        super().__init__(name=name)
        self.type = type


    def transform(self, X: pd.DataFrame):
        return X.astype(self.type)
