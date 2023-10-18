
from typing import List
from ceruleo.transformation.functional.transformerstep import TransformerStep
import pandas as pd
from functools import reduce 


class Sum(TransformerStep):
    """ 
    Concatenate multiple run-to-failure cycles vertically
    """
    def transform(self, X: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Apply the concatenation 

        Parameters:
            X: List of run-to-failure cycles to concatenate

        Returns:
            A dataframe with the concatenated run-to-failure cycles
        """
        return reduce(lambda x, y: x.add(y, fill_value=0), X)


class Divide(TransformerStep):
    """
    Divide multiple run-to-failure cycles vertically
    """
    def transform(self, X: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Apply the division

        Parameters:
            X: List of run-to-failure cycles to divide

        Returns:
            A dataframe with the divided run-to-failure cycles
        """
        return reduce(lambda x, y: x.divide(y, fill_value=0), X)