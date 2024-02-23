"""
In some cases It's useful to cast the datatype of subset of columns

"""
import logging
from typing import List, Optional

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


class ToDateTime(TransformerStep):
    """Cast to a datetime

    Example:
        step = ToDateTime()

    Parameters:
        name: Name of the step, by default None

    """

    def __init__(
        self,
        *,
        units: str = "s",
        name: Optional[str] = None,
        columns: Optional[List[any]] = None,
        index: bool = False,
        
    ):
        """Cast to a datetime

        Parameters
        ----------
        units : str, optional
            The units to transform, by default "s"
        name : Optional[str], optional
            Name of the step, by default None
        columns : Optional[List[any]], optional
            Columns to transform, by default None
        index : bool, optional
            Wether to transform the index or not, by default False

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        super().__init__(name=name)
        if index is False and columns is None:
            raise ValueError("You must specify the columns to transform")
        if columns is not None and index is True:
            raise ValueError("You can't specify columns and index at the same time")
        self.index = index
        self.columns = columns
        self.units = units

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cast to a datetime

        Parameters:
            X: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        X = X.copy()
        if self.index:
            X.index = pd.to_datetime(X.index, unit=self.units)
        else:
            X[self.columns] = X[self.columns].apply(pd.to_datetime)
        return X