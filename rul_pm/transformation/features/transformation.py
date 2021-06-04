
import logging
from typing import Optional

import emd
import numpy as np
import pandas as pd
from rul_pm.transformation.features.extraction import (compute, roll_matrix,
                                                       stats_order)
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


class MeanCentering(TransformerStep):
    """Center the data with respect to the mean
    """
    def __init__(self):
        super().__init__()
        self.N = 0
        self.sum = None

    def fit(self, X:pd.DataFrame, y=None):
        """Compute the mean of the dataset

        Parameters
        ----------
        X : pd.DataFrame
            the input dataset


        Returns
        -------
        MeanCentering
            self
        """
        self.mean = X.mean()

    def partial_fit(self, X:pd.DataFrame, y=None):
        """Compute incrementally the mean of the dataset

        Parameters
        ----------
        X : pd.DataFrame
            the input life

        Returns
        -------
        MeanCentering
            self
        """
        if self.sum is None:
            self.sum = X.sum()
        else:
            self.sum += X.sum()

        self.N += X.shape[0]
        self.mean = self.sum / self.N
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
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
        return X - self.mean


class Square(TransformerStep):
    """Compute the square of the values of each feature
    """
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        """Transform the input life with the square of the values

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input with
            the square of the values
        """
        return (X.pow(2))

class Sqrt(TransformerStep):
    """Compute the sqrt of the values of each feature
    """
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        """Transform the input life with the sqrt of the values

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input with
            the sqrt of the values
        """
        return (X.pow(1./2))


class Scale(TransformerStep):
    """Scale each feature by a given vaulue

    Parameters
    ----------
    scale_factor : float
        Scale factor
    name : Optional[str], optional
        Name of the step, by default None
    """
    def __init__(self, scale_factor: float, name: Optional[str] = None):
        super().__init__(name)
        self.scale_factor = scale_factor

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the scaled life

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input with the scaled features
        """
        return X * self.scale_factor




class ExpandingCentering(TransformerStep):
    """Center the life using an expanding window

    .. highlight:: python
    .. code-block:: python

        X - X.expanding().mean()

    """
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """Transform the live centering it using an expanding window

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input with the 
            data centered
        """
        return X - X.expanding().mean()


class ExpandingNormalization(TransformerStep):
    """Normalize the life features using an expanding window
    
    .. highlight:: python
    .. code-block:: python

        (X - X.expanding().mean()) / (X.expanding().std())
   
    """
    def transform(self, X):
        """Transform the live normalized it using an expanding window

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input with the 
            data normalized
        """
        return (X - X.expanding().mean()) / (X.expanding().std())



class Accumulate(TransformerStep):
    """Compute the accumulated sum of each feature.

    This is useful for binary features to compute count
    """
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """Transform the input life computing the cumulated sum

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input
            with the cumulated sum of the features
        """
        return X.cumsum()


class Diff(TransformerStep):
    """Compute the 1 step difference of each feature.
    """
    def transform(self, X):
        """Transform the input life computing the 1 step difference

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input
            with the difference of the features
        """
        return X.diff()




class StringConcatenate(TransformerStep):
    """Compute the 1 step difference of each feature.
    """
    def transform(self, X):
        """Transform the input life computing the 1 step difference

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            Return a new DataFrame with the same index as the input
            with the difference of the features
        """
        new_X = pd.DataFrame(index=X.index)
        new_X['concatenation'] = X.agg('-'.join, axis=1)
        return new_X