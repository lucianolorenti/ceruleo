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
        type: Data Type to cast to
        name: Name of the step, by default None

    """

    def __init__(self, *, type: str, name: Optional[str] = None):
        super().__init__(name=name)
        self.type = type

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cast to a given datatype

        Parameters:
            X: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        return X.astype(self.type)
