import logging
import multiprocessing
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from rul_pm.transformation.features.extraction import (compute, roll_matrix,
                                                       stats_order)
from rul_pm.transformation.transformerstep import TransformerStep
from rul_pm.transformation.utils.utils import sliding_window_view
from scipy.signal import detrend, hilbert


class MeanCentering(TransformerStep):
    """Center the data with respect to the mean"""

    def __init__(self):
        super().__init__()
        self.N = 0
        self.sum = None

    def fit(self, X: pd.DataFrame, y=None):
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

    def partial_fit(self, X: pd.DataFrame, y=None):
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
        return X - self.mean


class Square(TransformerStep):
    """Compute the square of the values of each feature"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return X.pow(2)


class Sqrt(TransformerStep):
    """Compute the sqrt of the values of each feature"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return X.pow(1.0 / 2)


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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
    """Compute the 1 step difference of each feature."""

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


class SplitIntoWindows(TransformerStep):
    """Split the features of the lives in windows[summary]

    Parameters
    ----------
    window_size : int
        Size of each window
    stride : int
        Strides of the windows
    """

    def __init__(self, window_size: int, stride: int, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride

    def transform(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """Transform the given life in a list of DataFrame, each element is a windowed vision of the signal

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        List[pd.DataFrame]
            The life data splitted in windows
        """

        indices = sliding_window_view(
            X.index.values, (self.window_size,), (self.stride,)
        )
        data = []
        for w in range(indices.shape[0]):
            data.append(X.loc[indices[w, :], :].copy())
        return data


class Envelope(TransformerStep):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for c in X.columns:
            x = np.abs(hilbert(X[c]))
            x = detrend(x, type='constant')
            X_new[c] = np.abs(np.fft.fft(x))
        return X_new


class Windowed(TransformerStep):
    """Apply a transformation on each window

    Parameters
    ----------
    step: TransformerStep
    """

    def __init__(self, step: TransformerStep, **kwargs):
        super().__init__(**kwargs)
        self.step = step

    def transform(self, X: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Transform the windows obtained with SplitOnWindows

        Parameters
        ----------
        X : List[pd.DataFrame]
            The windowed data of a life computed with SplitOnWindows

        Returns
        -------
        List[pd.DataFrame]
            The life data splitted in windows after the transformation
        """
        pool = multiprocessing.Pool(8)
        transformed = list(pool.map(self.step.transform, X))
        return transformed


class ConcatenateWindows(TransformerStep):
    def transform(self, X: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(X).reset_index(drop=True)
